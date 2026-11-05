"""仅TemporaryDirectory自造索引/标注；不读取真实数据或媒体。"""
import copy
import csv
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from futsalmot.adapters import RealBBoxReader, read_real_bbox_sample
from test_adapters import real_fixture, write_json


def second_clip(index_path, index, gt):
    seq = copy.deepcopy(index["sequences"][0])
    seq["clip_uid"] = "source:clip2"
    seq["sequence_path"] = "sequences/clip2"
    seq["frame_uid_pattern"] = "source:clip2:{frame_number:06d}"
    seq["track_namespace"] = "clip2:original_ID"
    index["sequences"].append(seq)
    target = index_path.parent / seq["sequence_path"]
    (target / "gt").mkdir(parents=True)
    (target / "gt/gt.txt").write_bytes(gt.read_bytes())
    info = gt.parent.parent / "gameinfo.ini"
    if info.exists():
        (target / "gameinfo.ini").write_bytes(info.read_bytes())
    write_json(index_path, index)
    return target / "gt/gt.txt"


class AnnotationCacheTests(unittest.TestCase):
    def test_cached_reader_has_original_two_source_parity(self):
        for source, split in (("soccernet_tracking", "train"), ("sportsmot_football", "val")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as d:
                path, _, _ = real_fixture(Path(d), source, split)
                reader = RealBBoxReader(path)
                for frame in (1, 3):
                    uid = f"detect:source:clip:{frame:06d}"
                    self.assertEqual(reader.read(uid, split=split),
                                     read_real_bbox_sample(path, uid, split=split))
                result = reader.read("detect:source:clip:000001", split=split)
                if source == "sportsmot_football":
                    self.assertEqual(result["targets"][0]["track_id"], 0)
                else:
                    self.assertEqual([r["ignore_reason"] for r in result["ignore_targets"]],
                                     ["non_target_role", "non_target_role", "unknown_role", "invalid_gt_flag"])

    def test_same_clip_GT_and_roles_are_opened_once(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, gt = real_fixture(Path(d))
            reader = RealBBoxReader(path)
            opens = []
            original = Path.open
            def counted(p, *args, **kwargs):
                if p.name in ("gt.txt", "gameinfo.ini"):
                    opens.append(p.name)
                return original(p, *args, **kwargs)
            with patch.object(Path, "open", counted):
                reader.read("detect:source:clip:000001")
                reader.read("detect:source:clip:000003")
                reader.read("detect:source:clip:000001")
            self.assertEqual(opens.count("gt.txt"), 1)
            self.assertEqual(opens.count("gameinfo.ini"), 1)
            self.assertEqual(reader.cache_info()["hits"], 2)
            self.assertEqual(reader.cache_info()["clip_loads"], 1)
            self.assertEqual(reader.cache_info()["cached_rows"], 7)
            self.assertFalse((gt.parent.parent / "img1").exists())

    def test_LRU_clip_and_row_limits_evict_without_losing_source_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            path, index, gt = real_fixture(Path(d))
            second_clip(path, index, gt)
            reader = RealBBoxReader(path, max_cached_clips=1, max_cached_rows=20)
            for clip in ("clip", "clip2", "clip"):
                reader.read(f"detect:source:{clip}:000001")
            self.assertEqual(reader.cache_info()["clip_loads"], 3)
            self.assertEqual(reader.cache_info()["cached_clips"], ("source:clip",))
            self.assertEqual(reader.cache_info()["cached_rows"], 7)
            by_rows = RealBBoxReader(path, max_cached_clips=2, max_cached_rows=10)
            by_rows.read("detect:source:clip:000001")
            by_rows.read("detect:source:clip2:000001")
            self.assertEqual(by_rows.cache_info()["cached_clips"], ("source:clip2",))
            self.assertEqual(by_rows.cache_info()["cached_rows"], 7)
            # 已逐出的clip仍绑定旧来源，不会静默刷新。
            gt.write_text(gt.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "变化"):
                by_rows.read("detect:source:clip:000001")

    def test_single_clip_exceeding_row_budget_and_invalid_limits_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, _ = real_fixture(Path(d))
            reader = RealBBoxReader(path, max_cached_rows=6)
            with self.assertRaisesRegex(ValueError, "max_cached_rows"):
                reader.read("detect:source:clip:000001")
            self.assertEqual(reader.cache_info()["cached_clips"], ())
            self.assertEqual(reader.cache_info()["cached_rows"], 0)
            for kwargs in ({"max_cached_rows": 0}, {"max_cached_clips": 0},
                           {"max_cached_clips": 1.5}, {"max_cached_rows": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    RealBBoxReader(path, **kwargs)

    def test_UID_and_split_must_pass_before_any_GT_or_gameinfo_access(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, gt = real_fixture(Path(d), split="test")
            gt.unlink()
            (gt.parent.parent / "gameinfo.ini").unlink()
            original = Path.open
            def guarded(p, *args, **kwargs):
                if p.name in ("gt.txt", "gameinfo.ini"):
                    raise AssertionError("不应读取test标注")
                return original(p, *args, **kwargs)
            with patch.object(Path, "open", guarded):
                reader = RealBBoxReader(path)
                with self.assertRaisesRegex(ValueError, "train/val"):
                    reader.read("detect:source:clip:000001", split="test")
                with self.assertRaisesRegex(ValueError, "UID"):
                    reader.read("detect:source:clip:000001")
                with self.assertRaisesRegex(ValueError, "UID"):
                    reader.read("unknown", split="val")

    def test_returned_mutation_does_not_pollute_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, _ = real_fixture(Path(d))
            reader = RealBBoxReader(path)
            original = reader.read("detect:source:clip:000001")
            changed = reader.read("detect:source:clip:000001")
            changed["targets"][0]["source_bbox"][0] = 999
            changed["targets"][0]["original_gt_row"][2] = "corrupt"
            changed["ignore_targets"].clear()
            changed["source_limitations"].append("bad")
            changed["yolo_labels"][0][1] = 99
            self.assertEqual(reader.read("detect:source:clip:000001"), original)

    def test_index_GT_gameinfo_and_same_size_replacement_are_refused(self):
        for kind in ("index", "gt", "gameinfo", "replace_gt"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                path, _, gt = real_fixture(Path(d))
                reader = RealBBoxReader(path)
                reader.read("detect:source:clip:000001")
                source = {"index": path, "gt": gt, "gameinfo": gt.parent.parent/"gameinfo.ini",
                          "replace_gt": gt}[kind]
                if kind == "replace_gt":
                    info = source.stat()
                    replacement = source.with_name("replacement.txt")
                    replacement.write_bytes(source.read_bytes())
                    os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
                    os.replace(replacement, source)
                else:
                    source.write_bytes(source.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "变化"):
                    reader.read("detect:source:clip:000001")

    def test_source_change_during_GT_or_role_read_is_refused(self):
        import futsalmot.adapters as adapters
        with tempfile.TemporaryDirectory() as d:
            path, _, gt = real_fixture(Path(d))
            reader = RealBBoxReader(path)
            original_csv = csv.reader
            def changing_csv(stream):
                first = True
                for row in original_csv(stream):
                    yield row
                    if first:
                        first = False
                        gt.write_bytes(gt.read_bytes() + b"\n")
            with patch("futsalmot.adapters.csv.reader", changing_csv):
                with self.assertRaisesRegex(ValueError, "变化"):
                    reader.read("detect:source:clip:000001")
            self.assertEqual(reader.cache_info()["cached_rows"], 0)
        with tempfile.TemporaryDirectory() as d:
            path, _, gt = real_fixture(Path(d))
            reader = RealBBoxReader(path)
            original_roles = adapters._roles
            def changing_roles(info):
                roles = original_roles(info)
                info.write_bytes(info.read_bytes() + b"\n")
                return roles
            with patch("futsalmot.adapters._roles", changing_roles):
                with self.assertRaisesRegex(ValueError, "变化"):
                    reader.read("detect:source:clip:000001")

    def test_pickle_reconstructs_cold_cache_with_same_identity_and_no_handles(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, gt = real_fixture(Path(d))
            reader = RealBBoxReader(path, max_cached_clips=1, max_cached_rows=10)
            expected = reader.read("detect:source:clip:000001")
            state = reader.__getstate__()
            self.assertNotIn("_cache", state)
            self.assertNotIn("frames", state)
            payload = pickle.dumps(reader)
            clone = pickle.loads(payload)
            self.assertEqual(clone.cache_info()["cached_clips"], ())
            self.assertEqual(clone.cache_info()["cached_rows"], 0)
            self.assertEqual(clone.read("detect:source:clip:000001"), expected)
            gt.write_bytes(gt.read_bytes() + b"\n")
            cold_clone = pickle.loads(payload)
            with self.assertRaisesRegex(ValueError, "变化"):
                cold_clone.read("detect:source:clip:000001")
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "spawn"):
                pickle.loads(payload)


if __name__ == "__main__":
    unittest.main()
