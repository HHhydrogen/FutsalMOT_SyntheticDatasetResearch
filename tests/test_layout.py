"""只用TemporaryDirectory的小型CPU夹具；不接触真实媒体或GT。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from futsalmot.layout import (
    VERSION, SCHEMAS, COUNT_KEYS, plan_synthetic_layout,
    _canonical_sha, _seed_file_plan, _stat_files,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(base):
    root = base / "synthetic_research_v1" / VERSION
    root.mkdir(parents=True)
    rows, totals = [], dict.fromkeys(COUNT_KEYS, 0)
    for i in range(20):
        number = 30001 + i
        sid = f"seed_{number:08x}"
        split = "train" if i < 16 else "dev"
        cams = ["C5", "C1", "C2", "C3", "C4", "P02"]
        selection = dict(mode="active_contiguous_v1", full_frame_count=300, frame_count=100,
                         start_step=20, end_step_exclusive=120, full_frames_sha256="c"*64,
                         full_meta_sha256="d"*64, selection_sha256="e"*64)
        seed = {"schema": "futsalmot_seed_manifest", "schema_version": "1.0.0",
                "dataset_id": "futsalmot_synth_multitask",
                "dataset_version": "v3_original_capture" if i == 0 else VERSION,
                "seed_id": sid, "root_seed": f"{number:08x}", "split": split,
                "frame_count": 100, "fps": 10, "camera_ids": cams,
                "trajectory_uid": "b"*64, "source_selection": selection,
                "generation": {"protocol_sha256": "a"*64, "code_bundle_sha256": "a"*64}}
        path = root / split / sid / "seed_manifest.json"
        digest = put(path, seed)
        # 只造小型元数据，明确没有造假媒体；全量枚举必须另行拒绝。
        counts = dict(rgb_frames=600, instance_masks=600, number_masks=600,
                      person_records=6000, pose_points=102000, files=1815,
                      bytes=path.stat().st_size, visible_number_records=0,
                      truncated_records=0, out_of_frame_records=0)
        row = dict(root_seed=number, seed_id=sid, split=split, frames_per_camera=100,
                   camera_ids=cams, seed_manifest_sha256=digest,
                   file_stat_fingerprint="a" * 64, counts=counts)
        row.update(source_selection=selection, source_frames_sha256="b"*64,
                   schemas={n: hashlib.sha256(b"{}").hexdigest() for n in SCHEMAS[:6]},
                   release_schemas={n: hashlib.sha256(b"{}").hexdigest() for n in SCHEMAS[6:]})
        for name in ("recipe_sha256", "protocol_sha256", "code_bundle_sha256", "module_dll_sha256",
                     "pose_calibration_sha256", "verification_sha256", "session_exits_sha256",
                     "coco_export_sha256"):
            row[name] = "a"*64
        rows.append(row)
        for key in COUNT_KEYS:
            totals[key] += counts[key]
    hashes = {name: put(root / "schemas" / name, {}) for name in SCHEMAS}
    manifest = dict(schema="futsalmot_dataset_manifest_v1",
                    dataset_id="futsalmot_synth_multitask", dataset_version=VERSION,
                    diagnostic_only=False, fps=10, resolution=[3840, 2160],
                    seeds=rows, schema_sha256=hashes, totals=totals,
                    frozen_sources_sha256="a"*64, plan_sha256="b"*64, release_attempt="fixture",
                    fingerprint_scope="canonical_manifest_without_fingerprint_including_source_validation_stat_bindings_no_rgb_byte_hash")
    g5_path = base / "batch" / "G5_complete.json"

    def rebind():
        manifest.pop("fingerprint", None)
        manifest["fingerprint"] = _canonical_sha(manifest)
        msha = put(root / "dataset_manifest.json", manifest)
        receipt = dict(schema="futsalmot_release_validation_v1", status="PASS",
                       dataset_manifest_sha256=msha, fingerprint=manifest["fingerprint"],
                       seed_count=20, camera_count=120, expected_files=36310,
                       all_owned_sessions_normal_exit=True,
                       rgb_frames=totals["rgb_frames"], person_records=totals["person_records"],
                       pose_points=totals["pose_points"], limitations=["fixture"])
        vsha = put(root / "validation_receipt.json", receipt)
        logical = totals["bytes"] + sum((root / "schemas" / n).stat().st_size for n in SCHEMAS)
        logical += (root / "dataset_manifest.json").stat().st_size
        logical += (root / "validation_receipt.json").stat().st_size
        put(g5_path, dict(status="published", release_path=str(root),
                         fingerprint=manifest["fingerprint"], manifest_sha256=msha,
                         validation_sha256=vsha, files=36310, logical_bytes=logical))
    rebind()
    return root, g5_path, manifest, rebind


class LayoutGateTests(unittest.TestCase):
    def test_unpublished_and_unclosed_never_enumerate(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / "synthetic_research_v1" / VERSION
            with patch("futsalmot.layout._stat_files", side_effect=AssertionError("禁止枚举")):
                result = plan_synthetic_layout(root, base / "G5_complete.json", enumerate_files=True)
                self.assertEqual(result["status"], "AWAITING")
                root, g5, _, _ = fixture(base)
                result = plan_synthetic_layout(root, g5, enumerate_files=True)
                self.assertEqual(result["status"], "AWAITING")
            self.assertFalse((base / "synthetic").exists())

    def test_metadata_split_UID_and_original_capture_version_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root, g5, _, _ = fixture(Path(d))
            before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                      for p in root.rglob("*") if p.is_file()}
            with patch("futsalmot.layout._stat_files", side_effect=AssertionError("默认不可枚举")):
                result = plan_synthetic_layout(root, g5, writers_closed=True)
            self.assertEqual(result["status"], "READY_FOR_ENUMERATION")
            seeds = result["candidate_dataset"]["seeds"]
            self.assertEqual(sum(s["split"] == "train" for s in seeds), 16)
            self.assertEqual(sum(s["split"] == "dev" for s in seeds), 4)
            self.assertEqual(seeds[0]["trajectory_uid"], "b"*64)
            self.assertEqual(seeds[0]["original_seed_dataset_version"], "v3_original_capture")
            self.assertEqual(before, {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in root.rglob("*") if p.is_file()})
            self.assertFalse(result["writes_performed"])
            self.assertFalse((Path(d) / "synthetic").exists())

    def test_missing_or_mismatched_bindings_block(self):
        with tempfile.TemporaryDirectory() as d:
            root, g5, manifest, rebind = fixture(Path(d))
            manifest["seeds"][0].pop("seed_manifest_sha256")
            rebind()
            self.assertEqual(plan_synthetic_layout(root, g5, writers_closed=True)["status"], "BLOCKED")
        with tempfile.TemporaryDirectory() as d:
            root, g5, _, _ = fixture(Path(d))
            receipt = json.loads((root / "validation_receipt.json").read_text())
            receipt["status"] = "FAIL"
            put(root / "validation_receipt.json", receipt)
            result = plan_synthetic_layout(root, g5, writers_closed=True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("绑定", result["reason"])

    def test_manifest_membership_not_directory_count(self):
        with tempfile.TemporaryDirectory() as d:
            root, g5, manifest, rebind = fixture(Path(d))
            manifest["seeds"].pop()
            rebind()
            result = plan_synthetic_layout(root, g5, writers_closed=True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("20", result["reason"])

    def test_destination_conflict_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root, g5, _, _ = fixture(base)
            target = base / "synthetic"
            target.mkdir()
            (target / "other.txt").write_text("不得覆盖")
            result = plan_synthetic_layout(root, g5, writers_closed=True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual((target / "other.txt").read_text(), "不得覆盖")
            result = plan_synthetic_layout(root, g5, target_root=base / "outside", writers_closed=True)
            self.assertEqual(result["status"], "BLOCKED")
            result = plan_synthetic_layout(root / ".." / VERSION, g5)
            self.assertEqual(result["status"], "BLOCKED")
            result = plan_synthetic_layout(base / "_staging" / "payload", g5)
            self.assertEqual(result["status"], "BLOCKED")

    def test_missing_source_protocol_binding_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root, g5, manifest, rebind = fixture(Path(d))
            manifest["seeds"][0].pop("protocol_sha256")
            rebind()
            result = plan_synthetic_layout(root, g5, writers_closed=True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("protocol_sha256", result["reason"])

    def test_reparse_ancestor_refused_without_following(self):
        from types import SimpleNamespace
        import stat
        from futsalmot.layout import _safe_path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            original = Path.lstat
            def fake_lstat(p, *args, **kwargs):
                if p == path:
                    return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
                return original(p, *args, **kwargs)
            with patch.object(Path, "lstat", fake_lstat):
                with self.assertRaisesRegex(ValueError, "重解析点"):
                    _safe_path(path / "child")

    def test_metadata_only_is_not_full_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root, g5, _, _ = fixture(Path(d))
            result = plan_synthetic_layout(root, g5, writers_closed=True, enumerate_files=True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["files"], [])
            self.assertFalse((Path(d) / "synthetic").exists())

    def test_exact_leaf_plan_preserves_split_and_rejects_extra(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root, target = base / "release", base / "synthetic"
            row = dict(split="dev", seed_id="seed_00007531", camera_ids=["C5"], frames_per_camera=1)
            seed = root / row["split"] / row["seed_id"]
            names = ["seed_manifest.json", "trajectory.jsonl", "person_keypoints_coco.json",
                     "C5/camera.json", "C5/annotations.jsonl", "C5/img1/000001.png",
                     "C5/mask/000001.png", "C5/number_mask/000001.png"]
            for name in names:
                p = seed / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"tiny fixture; not an image")
            records, _ = _stat_files(seed)
            row.update(file_stat_fingerprint=_canonical_sha(records),
                       counts={"files": len(records), "bytes": sum(r[1] for r in records)})
            plan = _seed_file_plan(root, target, row)
            self.assertEqual(len(plan), 8)
            self.assertTrue(all(p["relative"].startswith("dev/seed_00007531/") for p in plan))
            self.assertTrue(all(p["operation"] == "hardlink" for p in plan))
            self.assertFalse(target.exists())
            (seed / "extra").mkdir()
            with self.assertRaisesRegex(ValueError, "目录白名单"):
                _seed_file_plan(root, target, row)


if __name__ == "__main__":
    unittest.main()
