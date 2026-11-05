"""小Tensor数值primitive检查；无模型、优化器step、数据文件或GPU。"""
import builtins
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from futsalmot.batches import collate_bbox_samples
from futsalmot.training import (
    prepare_detection_batch, criterion_backward_loss, initialize_ema_stage,
    update_ema_state, train, MODEL_BACKEND_IMPLEMENTED,
)


def batch(empty=False):
    image = torch.zeros((3, 960, 960), dtype=torch.uint8)
    image[0].fill_(255)
    image[1].fill_(128)
    n = 0 if empty else 1
    sample = {
        "img": image, "cls": torch.zeros((n, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[.5, .5, .2, .3]], dtype=torch.float32) if n else torch.empty((0, 4)),
        "ignore_bboxes": torch.tensor([[0., 0., 20., 20.]]) if n else torch.empty((0, 4)),
        "uid": "fixture:one", "uid_field": "sample_uid", "domain": "R",
        "targets": [{"source": "fixture"}] if n else [], "ignore_targets": [], "absent_targets": [],
        "geometry": {"input_shape": (960, 960), "channel_order": "RGB", "original_shape": (960, 960)},
    }
    result = collate_bbox_samples([sample])
    result["request_metadata"] = [{"exposure_index": 7, "nested": {"augmentation_seed": 123}}]
    return result


class TrainingPrimitiveTests(unittest.TestCase):
    def setUp(self):
        for name in ("_lazy_init", "init", "is_available"):
            context = patch.object(torch.cuda, name, side_effect=AssertionError("禁止CUDA初始化/探测"))
            context.start()
            self.addCleanup(context.stop)

    def test_batch_single_normalization_device_dtypes_and_metadata_copy(self):
        original = batch()
        old_image = original["img"].clone()
        old_labels = original["bboxes"].clone()
        old_metadata = copy.deepcopy(original["request_metadata"])
        result = prepare_detection_batch(original, device="cpu", non_blocking=True)
        self.assertEqual(result["img"].dtype, torch.float32)
        self.assertEqual(result["img"].device.type, "cpu")
        self.assertEqual(result["img"][0, 0, 0, 0].item(), 1)
        self.assertAlmostEqual(result["img"][0, 1, 0, 0].item(), 128 / 255, places=7)
        self.assertEqual(result["img"][0, 2, 0, 0].item(), 0)
        for key in ("cls", "bboxes", "ignore_bboxes"):
            self.assertEqual(result[key].dtype, torch.float32)
            self.assertEqual(result[key].device.type, "cpu")
            self.assertNotEqual(result[key].data_ptr(), original[key].data_ptr())
        for key in ("batch_idx", "ignore_batch_idx"):
            self.assertEqual(result[key].dtype, torch.int64)
        self.assertEqual(result["uids"], original["uids"])
        self.assertEqual(result["geometry"], original["geometry"])
        self.assertEqual(result["request_metadata"], old_metadata)
        result["request_metadata"][0]["nested"]["augmentation_seed"] = -1
        result["geometry"][0]["original_shape"] = (-1, -1)
        result["bboxes"].fill_(0)
        self.assertTrue(torch.equal(original["img"], old_image))
        self.assertTrue(torch.equal(original["bboxes"], old_labels))
        self.assertEqual(original["request_metadata"], old_metadata)
        with self.assertRaisesRegex(ValueError, "重复|float"):
            prepare_detection_batch(result, device="cpu")
        empty = prepare_detection_batch(batch(empty=True), device="cpu")
        self.assertEqual(tuple(empty["bboxes"].shape), (0, 4))
        self.assertEqual(tuple(empty["ignore_batch_idx"].shape), (0,))

    def test_batch_rejects_uninitialized_cuda_and_invalid_inputs(self):
        original = batch()
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA尚未初始化"):
                prepare_detection_batch(original, device="cuda:0")
        bad = dict(original, img=original["img"].float())
        with self.assertRaises(ValueError):
            prepare_detection_batch(bad, device="cpu")
        bad = dict(original, ignore_batch_idx=torch.tensor([1], dtype=torch.int64))
        with self.assertRaises(ValueError):
            prepare_detection_batch(bad, device="cpu")
        bad = dict(original, batch_idx=torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            prepare_detection_batch(bad, device="cpu")
        with self.assertRaises(ValueError):
            prepare_detection_batch(original, device="meta")

    def test_native_three_vector_sum_over_16_keeps_expected_gradient(self):
        native = torch.tensor([1., 2., 3.], requires_grad=True)
        # 模拟microbatch=2的原生返回；不能在helper里再次乘2或除8。
        result = criterion_backward_loss((native * 2, native.detach().clone()))
        self.assertEqual(result["loss"].item(), 12 / 16)
        self.assertTrue(result["loss"].requires_grad)
        self.assertEqual(result["components"]["box"].item(), 2 / 16)
        self.assertFalse(result["components"]["cls"].requires_grad)
        self.assertTrue(torch.equal(result["native_loss_items"], native.detach()))
        result["loss"].backward()
        self.assertTrue(torch.equal(native.grad, torch.full((3,), 2 / 16)))

    def test_loss_shape_finiteness_and_detached_items_are_required(self):
        first = torch.ones(3, requires_grad=True)
        second = first.detach()
        bad = [
            (torch.ones(4), second), (first.reshape(3, 1), second),
            (first, torch.ones(2)), (torch.tensor([1, 2, 3]), second),
            (first, torch.ones(3, requires_grad=True)),
            (torch.tensor([1., float("nan"), 2.]), second),
            (first, torch.tensor([1., float("inf"), 2.])),
            (first, torch.ones(3, dtype=torch.float64)), (first,),
            (torch.full((3,), 3e38), second),
        ]
        for output in bad:
            with self.subTest(output=output), self.assertRaises(ValueError):
                criterion_backward_loss(output)

    def test_ema_hand_calculation_buffers_immutability_and_phase_reset(self):
        raw = {"weight": torch.tensor([1., 2.], requires_grad=True),
               "running_mean": torch.tensor([4.]), "num_batches_tracked": torch.tensor(2),
               "flag": torch.tensor(True)}
        state = initialize_ema_stage(raw)
        self.assertEqual(state["updates"], 0)
        before = {k: v.clone() for k, v in state["state_dict"].items()}
        self.assertNotEqual(state["state_dict"]["weight"].data_ptr(), raw["weight"].data_ptr())
        with torch.no_grad():
            raw["weight"].add_(2)
            raw["running_mean"].fill_(8)
            raw["num_batches_tracked"].fill_(7)
            raw["flag"].fill_(False)
        raw_before = {k: v.detach().clone() for k, v in raw.items()}
        updated = update_ema_state(state, raw, update=1)
        decay = .9999 * (1 - math.exp(-1 / 2000))
        torch.testing.assert_close(updated["state_dict"]["weight"],
                                   before["weight"] * decay + raw["weight"] * (1-decay),
                                   rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(updated["state_dict"]["running_mean"],
                                   before["running_mean"] * decay + raw["running_mean"] * (1-decay))
        self.assertEqual(updated["state_dict"]["num_batches_tracked"].item(), 7)
        self.assertFalse(updated["state_dict"]["flag"].item())
        for key in raw:
            self.assertTrue(torch.equal(state["state_dict"][key], before[key]))
            self.assertTrue(torch.equal(raw[key], raw_before[key]))
            self.assertFalse(updated["state_dict"][key].requires_grad)
        second = update_ema_state(updated, raw, update=2)
        self.assertEqual(second["updates"], 2)
        reset = initialize_ema_stage(raw)
        self.assertEqual(reset["updates"], 0)
        for key in raw:
            self.assertTrue(torch.equal(reset["state_dict"][key], raw[key]))
            self.assertNotEqual(reset["state_dict"][key].data_ptr(), raw[key].data_ptr())

    def test_ema_rejects_key_shape_dtype_nonfinite_or_skipped_update(self):
        raw = {"w": torch.ones(2), "count": torch.tensor(1, dtype=torch.int64)}
        state = initialize_ema_stage(raw)
        for bad_raw in (
            {"w": torch.ones(2, dtype=torch.float16)},
            {"w": torch.ones(2, dtype=torch.float64)},
            {"w": torch.ones(2, dtype=torch.complex64)},
            {"w": torch.tensor([float("nan")])}, {"w": "not_tensor"}, {},
        ):
            with self.subTest(raw=bad_raw), self.assertRaises(ValueError):
                initialize_ema_stage(bad_raw)
        for bad_raw in ({"w": torch.ones(2)}, {"w": torch.ones(3), "count": raw["count"]},
                        {"w": raw["w"], "count": raw["count"].to(torch.int32)}):
            with self.subTest(raw=bad_raw), self.assertRaises(ValueError):
                update_ema_state(state, bad_raw, update=1)
        for step in (0, 2, -1, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                update_ema_state(state, raw, update=step)
        with self.assertRaises(ValueError):
            update_ema_state({**state, "updates": True}, raw, update=1)

    def test_import_and_budget_stay_lazy_and_main_entry_remains_BLOCKED(self):
        import futsalmot.training as module
        path = Path(module.__file__)
        spec = importlib.util.spec_from_file_location("fixture_training_lazy", path)
        loaded = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("import/dry-run不可导入Torch")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded_import):
            spec.loader.exec_module(loaded)
            self.assertEqual(loaded.plan(11, 7)["N_R"], 7)
            self.assertEqual(loaded.train({})["status"], "BLOCKED")
        cfg = {"gate": {key: True for key in (
            "protocol_frozen", "data_frozen", "synthetic_complete", "gpu_released",
            "resource_precheck_passed", "numeric_precheck_passed", "training_authorized")},
            "numeric": {"mode_frozen": True}}
        self.assertFalse(MODEL_BACKEND_IMPLEMENTED)
        self.assertEqual(train(cfg)["updates"], 0)
        self.assertEqual(train(cfg)["status"], "BLOCKED")
        from futsalmot.evaluation import evaluate
        self.assertEqual(evaluate(cfg, split="test")["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
