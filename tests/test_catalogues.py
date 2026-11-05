"""canonical小metadata夹具；不生成或读取媒体/GT/JSONL/COCO。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from futsalmot.catalogues import synthetic_bbox_catalogue, MAX_RELOCATION_BYTES
from futsalmot.layout import VERSION, _canonical_sha, _expected_seed_files
from futsalmot.adapters import _source_stat
from futsalmot.data import CycleSampler
from futsalmot.dataset import RequestBBoxDataset


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(base):
    published=base/"synthetic_research_v1"/VERSION
    target=base/"synthetic"
    rows,canonical_rows,entries=[],[],[]
    total_files=total_bytes=total_rgb=0
    for i in range(20):
        number=30001+i;sid=f"seed_{number:08x}";split="train" if i<16 else "dev"
        directory=split+"/"+sid
        cameras=["C5","C1","C2","C3","C4","P02"]
        seed=dict(schema="futsalmot_seed_manifest",schema_version="1.0.0",dataset_id="fixture_synth",
                  dataset_version="v3_original" if i==0 else VERSION,seed_id=sid,root_seed=f"{number:08x}",
                  split=split,frame_count=2,camera_ids=cameras,trajectory_uid="a"*64)
        source=published/directory/"seed_manifest.json"
        sha=put(source,seed)
        destination=target/directory/"seed_manifest.json"
        destination.parent.mkdir(parents=True)
        os.link(source,destination)
        identity=_source_stat(destination)
        counts=dict(rgb_frames=12,instance_masks=12,number_masks=12,files=51,bytes=identity[2])
        row=dict(root_seed=number,seed_id=sid,split=split,camera_ids=cameras,
                 frames_per_camera=2,seed_manifest_sha256=sha,counts=counts)
        rows.append(row)
        canonical_rows.append({**deepcopy(row),"directory":directory,"trajectory_uid":"a"*64,
                               "original_seed_dataset_version":seed["dataset_version"]})
        for name in sorted(_expected_seed_files(row)):
            is_seed=name=="seed_manifest.json"
            entries.append({"source_relative":directory+"/"+name,"target_relative":directory+"/"+name,
                            "file_id":{"dev":identity[0] if is_seed else 1,"ino":identity[1] if is_seed else 999},
                            "bytes":identity[2] if is_seed else 0,"mtime_ns":identity[3] if is_seed else 0})
        total_files+=51;total_bytes+=identity[2];total_rgb+=12
    totals=dict(files=total_files,bytes=total_bytes,rgb_frames=total_rgb)
    manifest=dict(schema="futsalmot_dataset_manifest_v1",dataset_id="fixture_synth",dataset_version=VERSION,
                  diagnostic_only=False,seeds=rows,totals=totals)
    manifest["fingerprint"]=_canonical_sha(manifest)
    mpath=published/"dataset_manifest.json"
    mh=put(mpath,manifest)
    vpath=published/"validation_receipt.json"
    vh=put(vpath,dict(status="PASS",fingerprint=manifest["fingerprint"],
                      dataset_manifest_sha256=mh,all_owned_sessions_normal_exit=True))
    gpath=base/"batch/G5_complete.json"
    gh=put(gpath,dict(status="published",fingerprint=manifest["fingerprint"],release_path=str(published),
                      manifest_sha256=mh,validation_sha256=vh))
    source=dict(release_root=str(published),dataset_manifest=str(mpath),manifest_sha256=mh,
                validation_receipt=str(vpath),validation_sha256=vh,g5_receipt=str(gpath),g5_sha256=gh,
                fingerprint=manifest["fingerprint"],dataset_version=VERSION)
    receipt=dict(schema_version="synthetic_relocation_v1",status="PASS",phase="ALL_LINKS_VERIFIED",
                 pilot={"status":"PASS"},source_root=str(published),target_root=str(target),
                 source_binding=deepcopy(source),source_metadata={
                     str(p):{"sha256":sha,"stat":list(_source_stat(p))}
                     for p,sha in ((mpath,mh),(vpath,vh),(gpath,gh))},
                 files=entries,expected_linked_files=total_files,created_links=total_files,
                 verified_links=total_files,logical_linked_media_and_annotation_bytes=total_bytes,
                 destination_logical_bytes=total_bytes,media_copy_bytes=0,
                 original_release_retained=True,original_receipts_rewritten=False,
                 completed_seed_directories=[row["directory"] for row in canonical_rows])
    index=dict(schema_version="canonical_synthetic_dataset_v1",status="READY",readiness="READY",
               relocation_status="PASS",relocation="RELOCATION.json",source_release=source,
               seeds=canonical_rows,totals=totals)
    ipath=target/"dataset.json"
    put(ipath,index);put(target/"RELOCATION.json",receipt)
    return ipath,index,receipt


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        guards=[]
        for name in ("_lazy_init","init","is_available"):
            guard=patch("torch.cuda."+name,side_effect=AssertionError("不得探测/初始化CUDA"))
            guard.start();guards.append(guard)
        for guard in guards:self.addCleanup(guard.stop)

    def test_order_original_v3_and_existing_Dataset_constructor_without_samples(self):
        with tempfile.TemporaryDirectory() as d:
            path,index,_=fixture(Path(d))
            original=Path.open
            def guarded(p,*args,**kwargs):
                if p.suffix.lower() in (".png",".jpg",".jsonl") or p.name=="person_keypoints_coco.json":
                    raise AssertionError("不能打开媒体/GT/COCO")
                return original(p,*args,**kwargs)
            with patch.object(Path,"open",guarded):
                result=synthetic_bbox_catalogue(path)
                dev=synthetic_bbox_catalogue(path,split="dev")
                dataset=RequestBBoxDataset(result["catalogue"],view="train")
                val=RequestBBoxDataset(dev["catalogue"],view="val")
            self.assertEqual(result["counts"]["images"],192)
            self.assertEqual(result["counts"]["seeds"],16)
            self.assertEqual(result["counts"]["cameras"],96)
            self.assertEqual(dev["counts"]["images"],48)
            self.assertEqual(len(dataset),192);self.assertEqual(len(val),48)
            self.assertEqual(dataset._readers,{})
            expected=[f"detect:synthetic:fixture_synth:v3_original:seed_00007531:C5:{f:06d}" for f in (1,2)]
            self.assertEqual(result["ordered_uids"][:2],expected)
            self.assertIn(":C1:000001",result["ordered_uids"][2])
            first=result["catalogue"][expected[0]]
            self.assertEqual(set(first),{"domain","seed_dir","camera_id","frame_index","split"})
            self.assertEqual(first["seed_dir"],str(path.parent/index["seeds"][0]["directory"]))
            sampler=CycleSampler(result["ordered_uids"],task="detect",domain="S",seed=17)
            self.assertEqual(result["binding"]["ordered_pool_sha256"],sampler.pool_sha)
            self.assertEqual(result["counts"]["unique_uids"],192)

    def test_READY_and_PASS_required_before_any_seed_sample_metadata(self):
        changes=[lambda x:x.update(status="BUILDING"),lambda x:x.update(readiness="UNUSABLE_UNTIL_READY"),
                 lambda x:x.update(relocation_status="BUILDING")]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                path,index,_=fixture(Path(d));change(index);put(path,index)
                (path.parent/"RELOCATION.json").unlink()
                with self.assertRaisesRegex(ValueError,"READY"):
                    synthetic_bbox_catalogue(path)
        with tempfile.TemporaryDirectory() as d:
            path,_,receipt=fixture(Path(d))
            receipt["status"]="FAIL_STOP";put(path.parent/"RELOCATION.json",receipt)
            with self.assertRaisesRegex(ValueError,"PASS"):
                synthetic_bbox_catalogue(path)
            with self.assertRaisesRegex(ValueError,"test"):
                synthetic_bbox_catalogue(path,split="test")

    def test_bindings_paths_original_versions_duplicates_and_totals_refused(self):
        changes=[
            lambda x:x["source_release"].update(fingerprint="b"*64),
            lambda x:x["seeds"][0].update(directory="../escape"),
            lambda x:x["seeds"][0].update(directory="/absolute"),
            lambda x:x["seeds"][0].update(original_seed_dataset_version=VERSION),
            lambda x:x["seeds"][0].update(trajectory_uid="b"*64),
            lambda x:x["seeds"][1].update(seed_id=x["seeds"][0]["seed_id"]),
            lambda x:x["totals"].update(rgb_frames=999),
        ]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                path,index,_=fixture(Path(d));change(index);put(path,index)
                with self.assertRaises(ValueError):synthetic_bbox_catalogue(path)
        with tempfile.TemporaryDirectory() as d:
            path,_,receipt=fixture(Path(d))
            receipt["files"].pop();put(path.parent/"RELOCATION.json",receipt)
            with self.assertRaisesRegex(ValueError,"计数"):
                synthetic_bbox_catalogue(path)

    def test_source_metadata_or_seed_bytes_change_is_not_refreshed(self):
        with tempfile.TemporaryDirectory() as d:
            path,index,_=fixture(Path(d))
            source=Path(index["source_release"]["dataset_manifest"])
            source.write_bytes(source.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA"):
                synthetic_bbox_catalogue(path)
        with tempfile.TemporaryDirectory() as d:
            path,index,_=fixture(Path(d))
            seed=path.parent/index["seeds"][0]["directory"]/"seed_manifest.json"
            seed.write_bytes(seed.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA"):
                synthetic_bbox_catalogue(path)

    def test_RELOCATION_larger_than_4MiB_read_once_and_512MiB_limit_before_open(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,receipt=fixture(Path(d))
            receipt["fixture_padding"]="x"*(4*1024*1024+10)
            rp=path.parent/"RELOCATION.json";put(rp,receipt)
            original=Path.open;opens=[]
            def counted(p,*args,**kwargs):
                if p==rp:opens.append(str(p))
                return original(p,*args,**kwargs)
            with patch.object(Path,"open",counted):
                result=synthetic_bbox_catalogue(path)
            self.assertEqual(len(opens),1)
            self.assertGreater(result["binding"]["relocation"]["stat"][2],4*1024*1024)
            self.assertEqual(result["binding"]["relocation"]["sha256"],hashlib.sha256(rp.read_bytes()).hexdigest())
            original_stat=_source_stat
            def oversized(p):
                value=original_stat(p)
                return (value[0],value[1],MAX_RELOCATION_BYTES+1,value[3]) if Path(p)==rp else value
            with patch("futsalmot.catalogues._source_stat",oversized):
                with self.assertRaisesRegex(ValueError,"上限"):
                    synthetic_bbox_catalogue(path)

    def test_reparse_ancestor_and_duplicate_JSON_keys_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path,index,_=fixture(Path(d))
            bad_parent=path.parent/"train"
            original=Path.lstat
            def reparse(p,*args,**kwargs):
                if p==bad_parent:return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
                return original(p,*args,**kwargs)
            with patch.object(Path,"lstat",reparse):
                with self.assertRaisesRegex(ValueError,"重解析"):
                    synthetic_bbox_catalogue(path)
            raw=path.read_text(encoding="utf-8")
            path.write_text(raw[:-1]+',"status":"READY"}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"重复JSON"):
                synthetic_bbox_catalogue(path)


if __name__=="__main__":unittest.main()
