"""合成JSONL偏移索引夹具：只写自动清理的自造小JSONL，不读G或媒体。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
from futsalmot.adapters import SyntheticBBoxReader,read_synthetic_bbox_sample
from test_adapters import synthetic_fixture,write_json


def fixture(root,frames=3,cameras=("C5","C1")):
    m,row=synthetic_fixture(root)
    m.update(frame_count=frames,camera_ids=list(cameras))
    write_json(root/"seed_manifest.json",m)
    for cam in cameras:
        camera=dict(schema="futsalmot_camera_metadata",schema_version="1.0.0",camera_id=cam,width=100,height=50)
        write_json(root/cam/"camera.json",camera)
        rows=[]
        for frame in range(1,frames+1):
            r=deepcopy(row);r.update(camera_id=cam,frame_index=frame,source_step=10**frame)
            rows.append(r)
        (root/cam/"annotations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    return m


def rewrite_rows(path,transform):
    rows=[json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")


class SyntheticIndexTests(unittest.TestCase):
    def test_original_function_parity_v3_and_exact_one_parse_on_warm_seek(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            reader=SyntheticBBoxReader(root)
            for cam in ("C5","C1"):
                for frame in (3,1,2):
                    actual=reader.read(cam,frame)
                    self.assertEqual(actual,read_synthetic_bbox_sample(root,cam,frame))
                    self.assertEqual(actual["dataset_version"],"v3_source")
                    self.assertFalse(actual["release_validation_performed"])
            with mock.patch("futsalmot.adapters.json.loads",wraps=json.loads) as counted:
                reader.read("C1",3)
                self.assertEqual(counted.call_count,1)
            with mock.patch("futsalmot.adapters.json.loads",wraps=json.loads) as counted:
                read_synthetic_bbox_sample(root,"C1",3)
                self.assertEqual(counted.call_count,5) # 原入口metadata2+从头3行。
            info=reader.cache_info()
            self.assertEqual(info["indexed_frames"],6)
            self.assertEqual(info["offset_bytes"],6*16)
            self.assertEqual(info["index_builds"],2)
            changed=reader.read("C5",1);changed["targets"][0]["source_bbox"][0]=-999
            self.assertNotEqual(reader.read("C5",1)["targets"][0]["source_bbox"][0],-999)

    def test_LRU_and_frame_budget_keep_old_source_binding(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            r=SyntheticBBoxReader(root,max_indexed_cameras=2,max_indexed_frames=3)
            r.read("C5",1);r.read("C1",1)
            self.assertEqual(r.cache_info()["indexed_cameras"],("C1",))
            self.assertEqual(r.cache_info()["indexed_frames"],3)
            r.read("C5",2)
            self.assertEqual(r.cache_info()["index_builds"],3)
            # C1虽被LRU淘汰，旧文件绑定仍保留；变更后不能静默重新接纳。
            path=root/"C1/annotations.jsonl";path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"来源"):r.read("C1",1)
            limited=SyntheticBBoxReader(root,max_indexed_frames=2)
            with self.assertRaisesRegex(ValueError,"预算"):limited.read("C5",1)
            self.assertEqual(limited.cache_info()["indexed_frames"],0)
            one=SyntheticBBoxReader(root,max_indexed_cameras=1,max_indexed_frames=6)
            one.read("C5",1)
            # 只检查逻辑容量，不读取已变更C1。
            self.assertEqual(one.cache_info()["indexed_cameras"],("C5",))

    def test_metadata_JSONL_and_file_identity_changes_fail_without_refresh(self):
        for name in ("seed_manifest.json","C5/camera.json","C5/annotations.jsonl"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as t:
                root=Path(t);fixture(root)
                r=SyntheticBBoxReader(root);r.read("C5",1)
                path=root/name;path.write_bytes(path.read_bytes()+b" ")
                with self.assertRaisesRegex(ValueError,"来源"):r.read("C5",2)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);r=SyntheticBBoxReader(root);r.read("C5",1)
            path=root/"C5/annotations.jsonl";info=path.stat();raw=path.read_bytes()
            path.unlink();path.write_bytes(raw);os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
            # NTFS可能立即复用已删除fileID，保留一个旧文件句柄才能可靠检验替换身份。
            # 若身份恰复用，此用例用mtime变更触发，不宣称抗stat伪装。
            if path.stat().st_ino == info.st_ino:
                os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns+1_000_000_000))
            with self.assertRaisesRegex(ValueError,"来源"):r.read("C5",1)

    def test_missing_extra_blank_or_wrong_frame_and_identity_are_rejected(self):
        cases=[lambda rows:rows.pop(),lambda rows:rows.append(deepcopy(rows[-1])),
               lambda rows:rows[1].update(frame_index=3),
               lambda rows:rows[1].update(dataset_version="v4_fabricated"),
               lambda rows:rows[1]["objects"][0].update(track_id=99)]
        for change in cases:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);fixture(root);rewrite_rows(root/"C5/annotations.jsonl",change)
                with self.assertRaises(ValueError):SyntheticBBoxReader(root).read("C5",1)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);p=root/"C5/annotations.jsonl"
            p.write_bytes(p.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"多行|空行"):SyntheticBBoxReader(root).read("C5",1)

    def test_source_mutation_during_build_detected_and_no_index_published(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);r=SyntheticBBoxReader(root)
            import futsalmot.adapters as adapters
            original=adapters._synthetic_bbox_row
            changed=False
            def mutate(*args,**kwargs):
                nonlocal changed
                result=original(*args,**kwargs)
                if not changed:
                    changed=True
                    path=root/"C5/annotations.jsonl";info=path.stat()
                    os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns+1_000_000_000))
                return result
            with mock.patch("futsalmot.adapters._synthetic_bbox_row",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"来源变化"):r.read("C5",1)
            self.assertEqual(r.cache_info()["indexed_frames"],0)

    def test_pickle_has_cold_offsets_and_retains_prior_bindings(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);r=SyntheticBBoxReader(root);expected=r.read("C5",3)
            state=r.__getstate__()
            self.assertNotIn("_cache",state)
            self.assertNotIn("_manifest",state)
            restored=pickle.loads(pickle.dumps(r)) # 仅本测试刚生成的可信自造对象。
            self.assertEqual(restored.cache_info()["indexed_frames"],0)
            self.assertEqual(restored.read("C5",3),expected)
            encoded=pickle.dumps(r)
            path=root/"C5/annotations.jsonl";path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"来源"):pickle.loads(encoded)

    def test_request_split_bounds_and_budget_types(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);r=SyntheticBBoxReader(root)
            for cam,frame in [("P01",1),("C5",0),("C5",4),("C5",True)]:
                with self.assertRaises(ValueError):r.read(cam,frame)
            with self.assertRaisesRegex(ValueError,"split"):r.read("C5",1,split="dev")
            with self.assertRaisesRegex(ValueError,"split"):SyntheticBBoxReader(root,split="dev")
            for bad in (0,True,1.5):
                with self.assertRaises(ValueError):SyntheticBBoxReader(root,max_indexed_cameras=bad)
                with self.assertRaises(ValueError):SyntheticBBoxReader(root,max_indexed_frames=bad)

if __name__=="__main__":unittest.main()
