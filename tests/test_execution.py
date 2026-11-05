"""显式reservation的纯CPU夹具；真实组件链仅使用自造临时小图/GT。"""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from futsalmot.data import CycleSampler, PrefetchSampler
from futsalmot.execution import iter_reserved_microbatches
import futsalmot.execution as execution

_OWN_DATASET_MODULE = "test_dataset" not in sys.modules
dataset_fixture_module = importlib.import_module("test_dataset")
from futsalmot.dataset import RequestBBoxDataset
import futsalmot.dataset as dataset_module

_THREAD = None
_CV_THREAD = None
_BLOCKS = []


def setUpModule():
    global _THREAD, _CV_THREAD
    _THREAD = torch.get_num_threads()
    torch.set_num_threads(2)
    _CV_THREAD = dataset_fixture_module.cv2.getNumThreads()
    dataset_fixture_module.cv2.setNumThreads(2)
    for name in ("_lazy_init", "init", "is_available"):
        guard = patch.object(torch.cuda, name, side_effect=AssertionError("CPU夹具禁止CUDA初始化/探测"))
        guard.start()
        _BLOCKS.append(guard)


def tearDownModule():
    for guard in reversed(_BLOCKS):
        guard.stop()
    if _THREAD is not None:
        torch.set_num_threads(_THREAD)
    if _CV_THREAD is not None:
        dataset_fixture_module.cv2.setNumThreads(_CV_THREAD)
    if _OWN_DATASET_MODULE:
        dataset_fixture_module.tearDownModule()


def sampler(stream="R", uids=None):
    return PrefetchSampler(CycleSampler(uids or ["u0", "u1", "u2", "u3"],
                                       task="detect", domain=stream, seed=17))


class TinyExplicitDataset:
    def __init__(self, corrupt=None):
        self.calls = []
        self.image_refs = []
        self.max_live_samples = 0
        self.corrupt = corrupt

    def __getitem__(self, request):
        self.calls.append(deepcopy(request))
        image = torch.zeros((3, 960, 960), dtype=torch.uint8)
        self.image_refs.append(weakref.ref(image))
        self.max_live_samples = max(self.max_live_samples, sum(ref() is not None for ref in self.image_refs))
        stream = request["stream"]
        domain = request["data_domain"]
        root = Path(__file__).absolute().parent
        reference = ({"domain": "R", "index_path": str(root / "never_opened.json"), "split": "train"}
                     if domain == "R" else
                     {"domain": "S", "seed_dir": str(root / "never_opened_seed"),
                      "camera_id": "C5", "frame_index": 1, "split": "train"})
        result = {
            "img": image, "cls": torch.empty((0, 1), dtype=torch.float32),
            "bboxes": torch.empty((0, 4), dtype=torch.float32),
            "ignore_bboxes": torch.empty((0, 4), dtype=torch.float32),
            "uid": request["uid"], "uid_field": "sample_uid", "domain": domain,
            "geometry": {"input_shape": (960, 960), "channel_order": "RGB"},
            "targets": [], "ignore_targets": [], "absent_targets": [],
            "request": deepcopy(request), "source_reference": reference,
            "image_source_binding": {"path": str(root / "never_opened.png"), "stat": (1, 2, 100, 3)},
            "sampling_stream": stream, "data_domain": domain,
        }
        if self.corrupt:
            self.corrupt(result)
        return result


class ExecutionTests(unittest.TestCase):
    def test_real_fixture_full_component_chain_requires_explicit_commit_and_replays(self):
        with tempfile.TemporaryDirectory() as d:
            catalogue, _, _ = dataset_fixture_module.fixture(Path(d))
            dataset = RequestBBoxDataset(catalogue)
            q = sampler("R", [dataset_fixture_module.R_UID])
            before = q.state_dict()
            reservation = q.reserve(16)
            snapshot = deepcopy(reservation)
            original_decode = dataset_module.decode_rgb_image
            with patch("futsalmot.dataset.decode_rgb_image", wraps=original_decode) as decode:
                offsets, requests = [], []
                for output in iter_reserved_microbatches(reservation, dataset, microbatch_size=4):
                    offsets.append(output["microbatch_offset"])
                    self.assertEqual(output["effective_batch_size"], 16)
                    self.assertEqual(output["reservation_token"], reservation["token"])
                    self.assertFalse(output["commit_performed"])
                    for item in output["sample_lineage"]:
                        self.assertEqual(item["source_reference"]["index_path"],
                                         catalogue[dataset_fixture_module.R_UID]["index_path"])
                        self.assertEqual(item["sampling_stream"], "R")
                        requests.append(item["request"])
                    del output
                self.assertEqual(offsets, [0, 4, 8, 12])
                self.assertEqual(decode.call_count, 16)
                self.assertEqual(q.state_dict(), before)
                self.assertEqual(q.committed_draws, 0)
                self.assertEqual(q.pending_batches, 1)
                self.assertEqual(reservation, snapshot)
                self.assertEqual(requests, [{**r, "stream": "R", "data_domain": "R"}
                                            for r in snapshot["samples"]])
                committed = q.commit(reservation["token"])  # 仅模拟caller显式提交；没有优化器。
                self.assertEqual(committed, snapshot)
                self.assertEqual(q.committed_draws, 16)
                next_reservation = q.reserve(16)
                partial = iter_reserved_microbatches(next_reservation, dataset, microbatch_size=4)
                output = next(partial)
                del output
                partial.close()
                self.assertEqual(decode.call_count, 20)
                self.assertEqual(q.committed_draws, 16)
                self.assertEqual(q.cancel_pending(), 1)
                replay = q.reserve(16)
                self.assertNotEqual(replay["token"], next_reservation["token"])
                self.assertEqual(replay["samples"], next_reservation["samples"])
                for output in iter_reserved_microbatches(replay, dataset, microbatch_size=4):
                    del output
                self.assertEqual(decode.call_count, 36)
                self.assertEqual(q.committed_draws, 16)  # 全部读取仍不会自动提交。
                q.cancel_pending()

    def test_loading_is_lazy_bounded_and_does_not_hold_previous_sample_images(self):
        q = sampler()
        reservation = q.reserve(16)
        dataset = TinyExplicitDataset()
        iterator = iter_reserved_microbatches(reservation, dataset, microbatch_size=4)
        self.assertEqual(dataset.calls, [])
        first = next(iterator)
        self.assertEqual(len(dataset.calls), 4)
        self.assertLessEqual(dataset.max_live_samples, 4)
        self.assertEqual(sum(ref() is not None for ref in dataset.image_refs), 0)
        del first
        second = next(iterator)
        self.assertEqual(len(dataset.calls), 8)
        self.assertEqual(second["microbatch_offset"], 4)
        self.assertLessEqual(dataset.max_live_samples, 4)
        del second
        iterator.close()
        self.assertEqual(len(dataset.calls), 8)
        self.assertEqual(q.committed_draws, 0)

    def test_all_allowed_microbatch_sizes_read_only_current_group(self):
        for size in (1, 2, 4, 8, 16):
            with self.subTest(size=size):
                q, dataset = sampler(), TinyExplicitDataset()
                iterator = iter_reserved_microbatches(q.reserve(16), dataset, microbatch_size=size)
                output = next(iterator)
                self.assertEqual(len(dataset.calls), size)
                self.assertEqual(output["img"].shape[0], size)
                self.assertEqual(q.committed_draws, 0)
                del output
                iterator.close()
                q.cancel_pending()

    def test_R_extra_mapping_and_reservation_snapshot_preserve_original_requests(self):
        q = sampler("R_extra")
        reservation = q.reserve(16)
        original = deepcopy(reservation)
        dataset = TinyExplicitDataset()
        iterator = iter_reserved_microbatches(reservation, dataset, microbatch_size=2)
        self.assertEqual(reservation, original)
        reservation["samples"][0]["uid"] = "changed_after_iterator_created"
        output = next(iterator)
        first = original["samples"][0]
        self.assertEqual(dataset.calls[0], {**first, "stream": "R_extra", "data_domain": "R"})
        self.assertEqual(output["uids"][0], first["uid"])
        self.assertEqual(output["domains"], ["R", "R"])
        self.assertTrue(all(row["sampling_stream"] == "R_extra" for row in output["sample_lineage"]))
        del output
        iterator.close()
        self.assertEqual(q.committed_draws, 0)

    def test_invalid_envelopes_and_request_order_fail_before_any_read(self):
        good = sampler().reserve(16)
        changes = [
            lambda r: r.update(count=15), lambda r: r.update(count=True),
            lambda r: r.update(samples=r["samples"][:-1]), lambda r: r.update(token="fake"),
            lambda r: r.update(stream="new"), lambda r: r.update(data_domain="S"),
            lambda r: r["samples"][0].pop("epoch"),
            lambda r: r["samples"][0].update(augmentation_seed=True),
            lambda r: r["samples"][0].update(stream="S"),
            lambda r: r["samples"][0].update(domain="S"),
            lambda r: r["samples"][0].update(split="test"),
            lambda r: r["samples"].reverse(),
        ]
        for change in changes:
            with self.subTest(change=change):
                bad, dataset = deepcopy(good), TinyExplicitDataset()
                change(bad)
                with self.assertRaises(ValueError):
                    iter_reserved_microbatches(bad, dataset, microbatch_size=4)
                self.assertEqual(dataset.calls, [])
        for size in (0, 3, 32, True, 4.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                iter_reserved_microbatches(good, TinyExplicitDataset(), microbatch_size=size)

    def test_missing_or_wrong_Dataset_lineage_is_rejected_in_first_group(self):
        changes = [
            lambda s: s.pop("request"),
            lambda s: s.update(source_reference={}),
            lambda s: s.update(uid="wrong"),
            lambda s: s["request"].update(exposure_index=999),
            lambda s: s.update(sampling_stream="S"),
        ]
        for change in changes:
            with self.subTest(change=change):
                dataset = TinyExplicitDataset(change)
                iterator = iter_reserved_microbatches(sampler().reserve(16), dataset, microbatch_size=2)
                with self.assertRaises(ValueError):
                    next(iterator)
                self.assertLessEqual(len(dataset.calls), 2)

    def test_collate_None_corrupt_or_reordered_lineage_is_rejected(self):
        real_collate = execution._collate
        for mode in ("none", "request", "order", "source"):
            with self.subTest(mode=mode):
                def corrupt(samples):
                    output = real_collate(samples)
                    if mode == "none":
                        output["sample_lineage"][0] = None
                    elif mode == "request":
                        output["sample_lineage"][0]["request"]["augmentation_seed"] = -1
                    elif mode == "order":
                        output["sample_lineage"].reverse()
                    else:
                        output["sample_lineage"][0]["image_source_binding"]["stat"] = (9, 9, 9, 9)
                    return output
                dataset = TinyExplicitDataset()
                with patch("futsalmot.execution._collate", corrupt):
                    iterator = iter_reserved_microbatches(sampler().reserve(16), dataset, microbatch_size=2)
                    with self.assertRaises(ValueError):
                        next(iterator)
                self.assertEqual(len(dataset.calls), 2)


if __name__ == "__main__":
    unittest.main()
