"""纯metadata与小型CPU collate桥接夹具；无模型/参数更新/真实数据。"""
from copy import deepcopy
import builtins
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

from futsalmot.step_records import trace_microbatch, build_update_record


def collated(offset=0, size=4, stream="R", start=0):
    domain = "R" if stream == "R_extra" else stream
    base = Path(__file__).absolute().parent
    lineage, uids = [], []
    for i in range(offset, offset + size):
        uid = "fixture:" + str(i % 3)
        uids.append(uid)
        reference = ({"domain": "R", "split": "train", "index_path": str(base/"not_read.json")}
                     if domain == "R" else {"domain": "S", "split": "train", "seed_dir": str(base/"not_read"),
                                           "camera_id": "C5", "frame_index": 1})
        lineage.append({
            "request": {"uid": uid, "domain": stream, "stream": stream, "data_domain": domain,
                        "exposure_index": start + i, "epoch": (start + i)//3,
                        "augmentation_seed": 100 + start + i},
            "source_reference": reference,
            "image_source_binding": {"path": str(base/"not_read.png"), "stat": (1, 2, 3, 4)},
            "sampling_stream": stream, "data_domain": domain})
    return {"reservation_token": "a"*32+":0:0", "effective_batch_size": 16,
            "microbatch_offset": offset, "uids": uids, "domains": [domain]*size,
            "sample_lineage": lineage}


def traces(stream="R"):
    return [trace_microbatch(collated(offset, stream=stream)) for offset in (0, 4, 8, 12)]


def measurements():
    return dict(update=1, phase="final", phase_update=1, loss=35.,
                loss_components={"box": 1., "cls": 2., "dfl": 3.},
                loss_definition="fixture independent caller-supplied aggregate/components",
                backward_loss=.375, backward_components={"box": .1, "cls": .2, "dfl": .3},
                lr=.001, momentum=.937,
                parameter_groups=[{"name": "bias", "lr": .001, "momentum": .937},
                                  {"name": "weight", "lr": .001, "momentum": .937}],
                gradient_norm_before_clip=12., gradient_norm_after_clip=10.,
                previous_exposures={"R": 0, "S": 0}, cumulative_exposures={"R": 16, "S": 0},
                elapsed_seconds=.12)


class StepRecordTests(unittest.TestCase):
    def test_trace_copies_only_metadata_without_retaining_RGB_or_mutable_sources(self):
        class Blob:
            pass
        payload = Blob()
        pointer = weakref.ref(payload)
        value = collated()
        value["img"] = payload
        result = trace_microbatch(value)
        del value["img"], payload
        self.assertIsNone(pointer())
        self.assertNotIn("img", result)
        result["sample_lineage"][0]["request"]["augmentation_seed"] = -1
        result["sample_lineage"][0]["source_reference"]["split"] = "test"
        self.assertEqual(value["sample_lineage"][0]["request"]["augmentation_seed"], 100)
        self.assertEqual(value["sample_lineage"][0]["source_reference"]["split"], "train")

    def test_builder_preserves_measurements_lineage_and_explicit_cumulative_counts(self):
        rows, actual = traces("R_extra"), measurements()
        old = deepcopy(rows)
        result = build_update_record(rows, **actual)
        self.assertEqual((result["domain"], result["data_domain"]), ("R_extra", "R"))
        self.assertEqual((result["R_exposures"], result["S_exposures"]), (16, 0))
        self.assertEqual(result["cumulative_exposures"], {"R": 16, "S": 0})
        self.assertEqual(result["phase_update"], 1)
        self.assertEqual(result["loss"], 35.)
        self.assertEqual(result["backward_loss"], .375)
        self.assertEqual(result["loss_components"], actual["loss_components"])
        self.assertEqual(result["parameter_groups"], actual["parameter_groups"])
        self.assertEqual(result["sample_lineage"], [item for row in rows for item in row["sample_lineage"]])
        json.dumps(result, allow_nan=False)
        result["sample_lineage"][0]["source_reference"]["split"] = "bad"
        self.assertEqual(rows, old)
        actual = measurements()
        actual.update(update=2, previous_exposures={"R": 16, "S": 0},
                      cumulative_exposures={"R": 16, "S": 16})
        result = build_update_record(traces("S"), **actual)
        self.assertEqual((result["R_exposures"], result["S_exposures"]), (0, 16))

    def test_trace_rejects_missing_or_wrong_lineage_offsets_and_request_fields(self):
        changes = [
            lambda v: v.pop("reservation_token"),
            lambda v: v.update(effective_batch_size=True),
            lambda v: v.update(microbatch_offset=2),
            lambda v: v["sample_lineage"].__setitem__(0, None),
            lambda v: v["sample_lineage"][0].pop("source_reference"),
            lambda v: v["sample_lineage"][0]["request"].pop("stream"),
            lambda v: v["sample_lineage"][0]["request"].update(augmentation_seed=True),
            lambda v: v["sample_lineage"][0]["image_source_binding"].update(path="relative.png"),
            lambda v: v["sample_lineage"][0]["source_reference"].update(split="test"),
            lambda v: v["uids"].reverse(),
        ]
        for change in changes:
            with self.subTest(change=change):
                value = collated()
                change(value)
                with self.assertRaises(ValueError):
                    trace_microbatch(value)

    def test_builder_rejects_missing_groups_mixed_token_stream_and_duplicate_exposure(self):
        cases = []
        rows = traces()
        cases.append(rows[:-1])
        bad = deepcopy(rows)
        bad[0], bad[1] = bad[1], bad[0]
        cases.append(bad)
        bad = deepcopy(rows)
        bad[1]["reservation_token"] = "b"*32+":0:0"
        cases.append(bad)
        bad = deepcopy(rows)
        bad[-1] = traces("S")[-1]
        cases.append(bad)
        bad = deepcopy(rows)
        for item in bad[1]["sample_lineage"]:
            item["request"]["exposure_index"] -= 1
        cases.append(bad)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_update_record(value, **measurements())

    def test_actual_measurements_and_cumulative_counts_cannot_be_missing_or_fabricated(self):
        for key in ("loss", "backward_loss", "parameter_groups", "previous_exposures",
                    "gradient_norm_after_clip", "elapsed_seconds"):
            actual = measurements()
            actual.pop(key)
            with self.subTest(key=key), self.assertRaises(TypeError):
                build_update_record(traces(), **actual)
        changes = [
            {"loss": float("nan")}, {"lr": float("inf")}, {"momentum": 1},
            {"gradient_norm_before_clip": -1}, {"phase_update": 2}, {"update": True},
            {"loss_definition": ""}, {"loss_components": {"box": 1}},
            {"previous_exposures": {"R": 16, "S": 0}},
            {"cumulative_exposures": {"R": 15, "S": 0}},
            {"parameter_groups": []},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_update_record(traces(), **{**measurements(), **change})

    def test_module_and_metadata_functions_do_not_import_torch(self):
        import futsalmot.step_records as module
        spec = importlib.util.spec_from_file_location("metadata_only_fixture", module.__file__)
        loaded = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("纯metadata禁止导入Torch")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            spec.loader.exec_module(loaded)
            copied = [loaded.trace_microbatch(collated(i)) for i in (0, 4, 8, 12)]
            self.assertEqual(loaded.build_update_record(copied, **measurements())["R_exposures"], 16)
        value = collated()
        value["sample_lineage"][0]["request"]["not_metadata"] = object()
        with self.assertRaisesRegex(ValueError, "Tensor|自定义"):
            trace_microbatch(value)

    def test_real_execution_collate_trace_builder_and_temporary_RunRecorder_compatibility(self):
        owned = "test_execution" not in sys.modules
        fixture_module = importlib.import_module("test_execution")
        import torch
        from futsalmot.data import CycleSampler, PrefetchSampler
        from futsalmot.execution import iter_reserved_microbatches
        from futsalmot.records import RunRecorder
        prior_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with patch("torch.cuda._lazy_init", side_effect=AssertionError("禁止CUDA初始化")):
                q = PrefetchSampler(CycleSampler(["u0", "u1"], task="detect", domain="R_extra", seed=17))
                reservation = q.reserve(16)
                dataset = fixture_module.TinyExplicitDataset()
                rows = []
                for output in iter_reserved_microbatches(reservation, dataset, microbatch_size=4):
                    rows.append(trace_microbatch(output))
                    del output
                self.assertEqual(len(dataset.calls), 16)
                self.assertEqual(q.committed_draws, 0)
                record = build_update_record(rows, **measurements())
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d)/"fixture"
                    recorder = RunRecorder(path, {"kind": "metadata_fixture_no_optimizer_update"})
                    recorder.step(record)
                    written = json.loads((path/"steps.jsonl").read_text(encoding="utf-8"))
                    self.assertEqual(written["update"], 1)
                    self.assertEqual(written["domain"], "R_extra")
                    self.assertEqual(len(written["sample_lineage"]), 16)
                self.assertEqual(q.committed_draws, 0)
                q.cancel_pending()
        finally:
            torch.set_num_threads(prior_threads)
            if owned:
                fixture_module.tearDownModule()


if __name__ == "__main__":
    unittest.main()
