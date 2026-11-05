"""直接G输出的小控制快照夹具；临时目录自动清理，无GPU/真实数据。"""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from futsalmot import output_storage as storage


class OutputStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="futsalmot_direct_output_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "research"
        self.control = self.root / "runs" / "example_run"
        self.control.mkdir(parents=True)
        self.base = Path(self.temp.name) / "training" / "runs"
        self.base.mkdir(parents=True)
        self.target = self.base / self.control.name
        self.kwargs = {"research_root": self.root, "output_base": self.base}

    def bind(self):
        return storage.create_output_binding(self.control, **self.kwargs)

    def sample(self):
        (self.control / "code" / "futsalmot").mkdir(parents=True)
        (self.control / "code" / "futsalmot" / "runner.py").write_bytes(b"# frozen source\n")
        (self.control / "code" / "requirements.lock").write_bytes(b"public==1\n")
        (self.control / "admission.json").write_bytes(b'{"status":"PASS"}\n')
        (self.control / "job_table.json").write_bytes(b'{"jobs":[100]}\n')
        (self.control / "launch.py").write_bytes(b"print('controller')\n")
        (self.control / "status.json").write_bytes(b'{"state":"RUNNING"}\n')
        return self.bind()

    def capture(self, final=False):
        return storage.capture_control(self.control, final=final, **self.kwargs)

    def test_binding_identical_bytes_and_exact_resolution(self):
        result = self.bind()
        left = (self.control / storage.MARKER_NAME).read_bytes()
        self.assertEqual(left, (self.target / storage.MARKER_NAME).read_bytes())
        self.assertEqual(result["schema"], "futsalmot_direct_output_v1")
        self.assertEqual(result["storage_mode"], "direct_g")
        self.assertNotIn("state", result)
        self.assertEqual(storage.resolve_output_group(self.control, **self.kwargs), self.target)

    def test_existing_output_is_never_overwritten(self):
        self.target.mkdir()
        sentinel = self.target / "old_status.json"
        sentinel.write_bytes(b"failure evidence")
        with self.assertRaises(FileExistsError):
            self.bind()
        self.assertEqual(sentinel.read_bytes(), b"failure evidence")
        self.assertFalse((self.control / storage.MARKER_NAME).exists())

    def test_existing_source_marker_is_never_overwritten(self):
        marker = self.control / storage.MARKER_NAME
        marker.write_bytes(b"historical pointer")
        with self.assertRaises(FileExistsError):
            self.bind()
        self.assertEqual(marker.read_bytes(), b"historical pointer")
        self.assertFalse(self.target.exists())

    def test_path_boundaries_reject_relative_nested_and_dotdot(self):
        for bad in (Path("runs/example_run"), self.control / "nested", self.control / ".." / self.control.name):
            with self.subTest(path=str(bad)), self.assertRaises(ValueError):
                storage.create_output_binding(bad, **self.kwargs)
        inside = self.root / "outputs" / "runs"
        inside.mkdir(parents=True)
        with self.assertRaises(ValueError):
            storage.create_output_binding(self.control, research_root=self.root, output_base=inside)
        self.assertFalse(self.target.exists())

    def test_missing_source_rejected_without_creating_target(self):
        with self.assertRaises(ValueError):
            storage.create_output_binding(self.control.parent / "missing", **self.kwargs)
        self.assertFalse((self.base / "missing").exists())

    def test_reparse_ancestor_rejected_before_mutation(self):
        original = Path.lstat
        bad = self.base.parent
        def fake(path):
            if path == bad:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path)
        with patch.object(Path, "lstat", fake), self.assertRaisesRegex(ValueError, "reparse"):
            self.bind()
        self.assertFalse(self.target.exists())

    def test_symlink_source_rejected_before_mutation(self):
        original = Path.lstat
        def fake(path):
            if path == self.control:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return original(path)
        with patch.object(Path, "lstat", fake), self.assertRaisesRegex(ValueError, "symlink"):
            self.bind()
        self.assertFalse(self.target.exists())

    def test_marker_unknown_target_or_one_sided_tamper_rejected(self):
        self.bind()
        marker = self.control / storage.MARKER_NAME
        original = marker.read_bytes()
        value = json.loads(original)
        value["output_group"] = str(self.base / "arbitrary")
        marker.write_bytes(storage._encode(value))
        with self.assertRaises(ValueError):
            storage.resolve_output_group(self.control, **self.kwargs)
        marker.write_bytes(original)
        (self.target / storage.MARKER_NAME).write_bytes(b"{}")
        with self.assertRaises(ValueError):
            storage.resolve_output_group(self.control, **self.kwargs)

    def test_duplicate_json_key_rejected(self):
        self.bind()
        marker = self.control / storage.MARKER_NAME
        marker.write_bytes(b'{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(ValueError, "重复"):
            storage.resolve_output_group(self.control, **self.kwargs)

    def test_initial_copies_only_immutable_files_and_preserves_source(self):
        self.sample()
        expected = {"RUN_LOCATION.json", "admission.json", "job_table.json", "launch.py",
                    "code/requirements.lock", "code/futsalmot/runner.py"}
        before = {path.relative_to(self.control).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.control.rglob("*") if path.is_file()}
        result = self.capture()
        self.assertEqual({row["path"] for row in result["files"]}, expected)
        destination = self.target / "control"
        self.assertFalse((destination / "status.json").exists())
        self.assertFalse((destination / storage.MANIFEST_NAME).exists())
        self.assertFalse(result["final"])
        self.assertIsNone(result["manifest_sha256"])
        for relative in expected:
            self.assertEqual((destination / relative).read_bytes(), before[relative][0])
        for relative, original in before.items():
            path = self.control / relative
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
        self.assertFalse((self.target / ".control_capture.lock").exists())

    def test_initial_same_bytes_repeat_is_idempotent(self):
        self.sample()
        first = self.capture()
        second = self.capture()
        self.assertEqual(first, second)

    def test_final_seals_exact_hashes_and_repeated_call_unchanged(self):
        self.sample()
        (self.control / "STOP.json").write_bytes(b'{"reason":"user controlled stop"}\n')
        self.capture()
        self.assertFalse((self.target / "control" / "STOP.json").exists())
        (self.control / "status.json").write_bytes(b'{"state":"FAIL_STOP","error":"original"}\n')
        (self.control / "dispatch.json").write_bytes(b'{"pid":1234}\n')
        (self.control / "R100_protocol.json").write_bytes(b'{"epochs":100}\n')
        result = self.capture(final=True)
        manifest_path = Path(result["manifest_path"])
        before = (manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns)
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(before[0]).hexdigest())
        manifest = json.loads(before[0])
        self.assertIn("STOP.json", {row["path"] for row in manifest["files"]})
        self.assertEqual(manifest["scope"], "control_snapshot_only")
        self.assertEqual(manifest["state"], "SEALED")
        self.assertEqual(manifest["media_files_copied"], 0)
        self.assertNotIn("archive", manifest["schema"])
        for row in manifest["files"]:
            data = (self.target / "control" / row["path"]).read_bytes()
            self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (row["bytes"], row["sha256"]))
        self.assertEqual(self.capture(final=True), result)
        self.assertEqual((manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns), before)
        with self.assertRaises(ValueError):
            self.capture()

    def test_sealed_source_change_rejected_and_failure_evidence_unchanged(self):
        self.sample()
        self.capture(final=True)
        destination = self.target / "control"
        old = (destination / "status.json").read_bytes()
        receipt = (destination / storage.MANIFEST_NAME).read_bytes()
        (self.control / "status.json").write_bytes(b'{"state":"COMPLETE"}')
        with self.assertRaisesRegex(ValueError, "已封闭"):
            self.capture(final=True)
        self.assertEqual((destination / "status.json").read_bytes(), old)
        self.assertEqual((destination / storage.MANIFEST_NAME).read_bytes(), receipt)

    def test_sealed_target_missing_file_rejected_without_repair(self):
        self.sample()
        self.capture(final=True)
        deleted = self.target / "control" / "launch.py"
        deleted.unlink()
        with self.assertRaisesRegex(ValueError, "已封闭目标"):
            self.capture(final=True)
        self.assertFalse(deleted.exists())

    def test_initial_conflict_rejected_without_replacing_either_side(self):
        self.sample()
        self.capture()
        source = self.control / "launch.py"
        target = self.target / "control" / "launch.py"
        source.write_bytes(b"new version")
        old = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.capture(final=True)
        self.assertEqual(target.read_bytes(), old)
        self.assertEqual(source.read_bytes(), b"new version")
        self.assertFalse((self.target / "control" / "status.json").exists())

    def test_unexpected_weights_predictions_and_directories_rejected(self):
        self.sample()
        for relative in ("weights.pt", "predictions.npz", "steps.jsonl", "unknown.json", "code/futsalmot/model.pt"):
            with self.subTest(relative=relative):
                path = self.control / relative
                path.write_bytes(b"must not copy")
                with self.assertRaisesRegex(ValueError, "白名单"):
                    self.capture()
                path.unlink()
        path = self.control / "R100_seed17"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "白名单目录"):
            self.capture()
        self.assertFalse((self.target / "control").exists())

    def test_unexpected_destination_file_preserved_and_rejected(self):
        self.sample()
        self.capture()
        extra = self.target / "control" / "alien.pt"
        extra.write_bytes(b"do not overwrite")
        with self.assertRaises(ValueError):
            self.capture(final=True)
        self.assertEqual(extra.read_bytes(), b"do not overwrite")

    def test_per_file_and_total_limits_fail_before_copy(self):
        self.sample()
        (self.control / "admission.json").write_bytes(b"x" * 2048)
        with patch.object(storage, "MAX_FILE_BYTES", 1024), self.assertRaisesRegex(ValueError, "16MiB"):
            self.capture()
        self.assertFalse((self.target / "control").exists())
        (self.control / "admission.json").write_bytes(b"{}")
        with patch.object(storage, "MAX_TOTAL_BYTES", 1), self.assertRaisesRegex(ValueError, "32MiB"):
            self.capture()
        self.assertFalse((self.target / "control").exists())

    def test_g_training_tree_is_neither_traversed_nor_copied(self):
        self.sample()
        media = self.target / "R100_seed17" / "weights"
        media.mkdir(parents=True)
        sentinel = media / "checkpoint.pt"
        sentinel.write_bytes(b"fake-heavy-evidence")
        original = Path.iterdir
        def guard(path):
            if path == self.target or path == media or path == media.parent:
                raise AssertionError("must not traverse G training tree")
            return original(path)
        with patch.object(Path, "iterdir", guard):
            self.capture(final=True)
        self.assertEqual(sentinel.read_bytes(), b"fake-heavy-evidence")
        self.assertFalse((self.target / "control" / "R100_seed17").exists())

    def test_concurrent_capture_lock_is_never_removed_or_replaced(self):
        self.sample()
        lock = self.target / ".control_capture.lock"
        lock.write_bytes(b"other process identity")
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual(lock.read_bytes(), b"other process identity")

    def test_reparse_control_tree_rejected_without_following(self):
        self.sample()
        original = Path.lstat
        bad = self.control / "code"
        def fake(path):
            if path == bad:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path)
        with patch.object(Path, "lstat", fake), self.assertRaisesRegex(ValueError, "reparse"):
            self.capture()
        self.assertFalse((self.target / "control").exists())

    def test_source_mutation_during_capture_is_not_sealed(self):
        self.sample()
        collect = storage._collect
        count = 0
        def mutate(source, *, final):
            nonlocal count
            count += 1
            if count == 2:
                (self.control / "status.json").write_bytes(b'{"state":"mutated"}')
            return collect(source, final=final)
        with patch.object(storage, "_collect", mutate), self.assertRaisesRegex(RuntimeError, "变化"):
            self.capture(final=True)
        self.assertFalse((self.target / "control" / storage.MANIFEST_NAME).exists())
        self.assertFalse((self.target / ".control_capture.lock").exists())


if __name__ == "__main__":
    unittest.main()
