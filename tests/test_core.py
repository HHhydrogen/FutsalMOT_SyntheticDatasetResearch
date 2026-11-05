"""有界CPU契约检查；不依赖真实GT、图片、Torch或GPU。"""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from futsalmot.data import CycleSampler, detection_train_uids, instance_mask_contract, pose_supervision_masks
from futsalmot.training import STRATEGIES, budget_updates, domain_at, plan, schedule_at, train
from futsalmot.evaluation import evaluate
from futsalmot.records import RunRecorder, temporary_diagnostic, validate_checkpoint_contract

class BudgetTests(unittest.TestCase):
    def test_real_examples_and_padding(self):
        for n, expected in [(2100, 6563), (2933, 9166)]:
            p = plan(16000, n)
            self.assertEqual(p["U_S"], 20000)
            self.assertEqual(p["U_R"], expected)
            self.assertTrue(0 <= p["padding_R"] < 16)
            self.assertEqual(p["strategies"]["S->R"]["R_exposures"], p["strategies"]["R+S"]["R_exposures"])

    def test_exact_domain_counts_and_ordinals(self):
        for us in range(2, 12):
            for ur in range(2, 12):
                steps = [domain_at("R+S", j, us, ur) for j in range(1, us+ur+1)]
                self.assertEqual([i for d, i in steps if d == "R"], list(range(1, ur+1)))
                self.assertEqual([i for d, i in steps if d == "S"], list(range(1, us+1)))
                self.assertEqual(domain_at("R->R", us, us, ur), ("R_extra", us))
                self.assertEqual(domain_at("R->R", us+1, us, ur), ("R", 1))

    def test_schedule_endpoints_and_phase_reset(self):
        for final in [False, True]:
            first = schedule_at(0, 1000, final_stage=final)
            k = first["warmup_updates"]
            peak = 0.001 if final else 0.01
            self.assertEqual(first["lr"], 0)
            self.assertAlmostEqual(schedule_at(k, 1000, final_stage=final)["lr"], peak)
            last = schedule_at(1000, 1000, final_stage=final)
            self.assertAlmostEqual(last["lr"], peak*0.01)
            self.assertAlmostEqual(last["ema_decay"], 0.9999*(1-math.exp(-0.5)))
        from futsalmot.training import update_spec
        self.assertEqual(update_spec("S->R", 11, 10, 20)["step"], 1)

class DataContractTests(unittest.TestCase):
    def test_sampler_json_roundtrip_and_domain_independence(self):
        a = CycleSampler(["a", "b", "c"], task="detect", domain="R", seed=17)
        first = a.take(3)
        self.assertEqual({r["uid"] for r in first}, {"a", "b", "c"})
        a.take(2)
        state = json.loads(json.dumps(a.state_dict()))
        b = CycleSampler(["a", "b", "c"], task="detect", domain="R", seed=17)
        b.load_state_dict(state)
        self.assertEqual(a.take(20), b.take(20))
        with self.assertRaises(ValueError):
            CycleSampler(["other"], task="detect", domain="R", seed=17).load_state_dict(state)

    def test_train_UID_expansion_never_includes_val_test(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"dataset.json"
            seqs = [{"split": split, "detection_frames": [1,3], "frame_uid_pattern": split+":{frame_number:06d}", "detection_sample_uid_pattern": "detect:{frame_uid}"} for split in ["train", "val", "test"]]
            p.write_text(json.dumps({"sequences": seqs}), encoding="utf-8")
            self.assertEqual(detection_train_uids(p), ["detect:train:000001", "detect:train:000003"])

    def test_pose_unknown_instance_not_negative_target(self):
        points = [[0,0,0] for _ in range(17)]
        points[1], points[2] = [10,20,1], [30,40,2]
        valid = pose_supervision_masks(points, pose_supervision_valid=True)
        self.assertEqual(sum(valid["coordinate_valid"]), 2)
        self.assertTrue(all(valid["kobj_loss_valid"]))
        unknown = pose_supervision_masks(points, pose_supervision_valid=False)
        self.assertFalse(any(unknown["coordinate_valid"]))
        self.assertFalse(any(unknown["kobj_loss_valid"]))
        self.assertTrue(unknown["bbox_loss_valid"])

    def test_mask_fragments_and_hole_preserved(self):
        mask = [[1,1,1,0,2],[1,0,1,0,0],[1,1,1,0,2]]
        before = json.dumps(mask)
        c = instance_mask_contract(mask, [1,2])
        self.assertEqual(c["pixel_counts"], {1:8,2:2})
        self.assertEqual(before, json.dumps(mask))
        self.assertFalse(c["model_adapter_implemented"])
        with self.assertRaises(ValueError): instance_mask_contract(mask, [1,1,2])

class GateAndRecordsTests(unittest.TestCase):
    def test_gates_stay_blocked_even_with_true_flags(self):
        cfg = {"gate": {name: True for name in ["protocol_frozen", "data_frozen", "synthetic_complete", "gpu_released", "resource_precheck_passed", "numeric_precheck_passed", "training_authorized", "test_open_authorized"]}, "numeric": {"mode_frozen": True}}
        self.assertEqual(train(cfg)["status"], "BLOCKED")
        self.assertEqual(evaluate(cfg, split="test")["status"], "BLOCKED")
        self.assertFalse(evaluate(cfg)["sample_content_read"])

    def test_temporary_diagnostic_cleanup_including_exception(self):
        with temporary_diagnostic({"kind": "fixture"}) as record:
            location = record.directory
            record.event({"event": "fixture_not_training"})
            self.assertTrue((location / "events.jsonl").exists())
        self.assertFalse(location.exists())
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with temporary_diagnostic() as record:
                failed_location = record.directory
                raise RuntimeError("fixture failure")
        self.assertFalse(failed_location.exists())

    def test_cli_dataset_override_defaults_and_cross_source_refusal(self):
        from futsalmot.__main__ import main
        def invoke(args):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(args)
            return code, json.loads(out.getvalue() if code != 1 else err.getvalue())
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/experiment.yaml").read_text(encoding="utf-8"))
            for panel, count in [("soccernet", 2), ("sportsmot", 3)]:
                matches, sequences = [], []
                for split in ["train", "val", "test"]:
                    uid = panel + ":" + split
                    matches.append({"match_uid": uid, "split": split})
                    sequences.append({"match_uid": uid, "split": split, "clip_uid": uid, "complete": True, "selected": True, "official_frame_count": 20, "detection_frames": list(range(1, count+1)), "frame_uid_pattern": uid+":{frame_number:06d}", "detection_sample_uid_pattern": "detect:{frame_uid}"})
                p = base / (panel + ".json")
                p.write_text(json.dumps({"schema_version": "canonical_real_dataset_v1", "source_panel_id": panel, "status": "READY_SOURCE_LOCAL", "joint_use_allowed": False, "cross_source_disjoint_certified": False, "matches": matches, "sequences": sequences}), encoding="utf-8")
                cfg["real_datasets"][panel] = str(p)
            cfg["candidate_budget"]["real_panel"] = "soccernet"
            cfg["candidate_budget"]["N_R"] = 2
            config_path = base / "experiment.yaml"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            common = ["--config", str(config_path)]
            code, result = invoke(["inspect", *common])
            self.assertEqual(code, 0)
            self.assertEqual(set(result["datasets"]), {"soccernet", "sportsmot"})
            code, result = invoke(["dry-run", "--synthetic-images", "16000", *common])
            self.assertEqual((code, result["real_panel"], result["N_R"]), (0, "soccernet", 2))
            code, result = invoke(["dry-run", "--dataset", "sportsmot", "--synthetic-images", "16000", *common])
            self.assertEqual((code, result["real_panel"], result["N_R"]), (0, "sportsmot", 3))
            self.assertEqual(result["UID_binding"]["R"]["unique_train_samples"], 3)
            self.assertTrue(all("sportsmot:" in row["uid"] for row in result["UID_binding"]["R"]["first_effective_batch"]))
            code, result = invoke(["dry-run", "--dataset", "all", "--synthetic-images", "16000", *common])
            self.assertEqual((code, result["status"]), (1, "ERROR"))
            self.assertIn("不允许跨库混合", result["reason"])
            cfg["candidate_budget"]["real_panel"] = "sportsmot"
            cfg["candidate_budget"]["N_R"] = None
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            code, result = invoke(["dry-run", "--synthetic-images", "16000", *common])
            self.assertEqual((code, result["real_panel"], result["N_R"]), (0, "sportsmot", 3))

    def test_checkpoint_and_append_only_records(self):
        cp = dict(schema_version="test", raw_model={}, ema={}, optimizer={}, rng={}, sampler={}, phase="final", phase_updates=0, committed_updates=0, source_bindings={"fixture":"test"}, numeric_mode="fp32", scaler=None)
        validate_checkpoint_contract(cp)
        with self.assertRaises(ValueError): validate_checkpoint_contract({**cp, "scaler": {}})
        validate_checkpoint_contract({**cp, "numeric_mode": "amp_bf16"})
        with self.assertRaises(ValueError): validate_checkpoint_contract({**cp, "numeric_mode": "amp_bf16", "scaler": {}})
        validate_checkpoint_contract({**cp, "numeric_mode": "amp_fp16", "scaler": {}})
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"fixture_run"
            r = RunRecorder(p, {"kind": "fixture_not_training"})
            row = dict(update=1, phase="final", domain="R", uids=["fixture"], loss=1, lr=.001, gradient_norm=1, R_exposures=1, S_exposures=0)
            r.step(row)
            with self.assertRaises(ValueError): r.step(row)
            with self.assertRaises(FileExistsError): RunRecorder(p, {})
            self.assertEqual(json.loads((p/"steps.jsonl").read_text())["update"], 1)

if __name__ == "__main__": unittest.main()
