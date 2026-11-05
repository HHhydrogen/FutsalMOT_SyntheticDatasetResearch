"""准入代码的纯CPU/临时metadata检查，不导入Torch、不读G/权重/真实数据。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import futsalmot.preflight as module


class PreflightTests(unittest.TestCase):
    def test_exact_R50_storage_and_checkpoint_count(self):
        result = module._storage_budget(123456789, 6650344, n_train=2100, n_val=750)
        self.assertEqual(result["updates"], 6563)
        self.assertEqual(result["checkpoints"], 133)
        self.assertEqual(result["validation_passes"], 8)
        self.assertEqual(result["prediction_images"], 6000)
        self.assertEqual(result["required_bytes"],
                         __import__("math").ceil(123456789 * 133 * 1.1) + 6650344 * 6000 + 9 * 1024**3)
        self.assertEqual(result["minimum_free_after_budget_bytes"], 100 * 1024**3)
        # 50轮恰好1000更新：final与1000评价不重复；checkpoint末尾亦不重复。
        exact = module._storage_budget(1, 1, n_train=320, n_val=1)
        self.assertEqual(exact["updates"], 1000)
        self.assertEqual(exact["checkpoints"], 21)
        self.assertEqual(exact["validation_passes"], 2)

    def test_missing_GPU_release_writes_failure_without_any_backend_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict("sys.modules", {"torch": None, "futsalmot.runner_data": None}):
                with self.assertRaisesRegex(RuntimeError, "gpu_release_confirmed"):
                    module.preflight_calibration(temporary, "not_opened_index", "not_opened_weights")
            result = json.loads((Path(temporary) / "admission.json").read_text(encoding="utf8"))
            self.assertEqual(result["status"], "FAIL_STOP")
            self.assertEqual(result["diagnostic_updates"], 0)
            self.assertIsNone(result["checkpoint_bytes"])
            self.assertFalse(result["test_read"])
            self.assertEqual(list(Path(temporary).iterdir()), [Path(temporary) / "admission.json"])
            with self.assertRaises(FileExistsError):
                module.preflight_calibration(temporary, "not_opened_index", "not_opened_weights")

    def test_explicit_offline_and_bounded_budget_before_backend_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict("os.environ", {"YOLO_OFFLINE": "false"}):
                with mock.patch.dict("sys.modules", {"torch": None}):
                    with self.assertRaisesRegex(RuntimeError, "YOLO_OFFLINE"):
                        module.preflight_calibration(temporary, "index", "weights", gpu_release_confirmed=True)
            self.assertEqual(json.loads((Path(temporary) / "admission.json").read_text(
                encoding="utf8"))["status"], "FAIL_STOP")
        for duration in (0, 601, True, 1.5):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                module.preflight_calibration("not_created", "index", "weights", max_seconds=duration)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module._storage_budget(value, 1, n_train=2100, n_val=750)

    def test_write_once_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            module._write_once(target, {"status": "fixture"})
            with self.assertRaises(FileExistsError):
                module._write_once(target, {"status": "changed"})
            self.assertEqual(json.loads(target.read_text()), {"status": "fixture"})
        with self.assertRaises(ValueError):
            module._safe_path("outside/../escape")


if __name__ == "__main__":
    unittest.main()
