"""纯内存资源算例，不读数据、测硬件、加载权重或创建运行目录。"""
import json
import unittest

from futsalmot.resources import candidate_resource_plan, checkpoint_count, MODEL_NAMES
from futsalmot.training import domain_at, plan


class ResourceMathTests(unittest.TestCase):
    def test_known_39_job_and_independent_calibration_example(self):
        result = candidate_resource_plan(16000, 2100)
        formal, calibration = result["formal_totals"], result["calibration_totals"]
        self.assertEqual((formal["jobs"], formal["total_updates"],
                          formal["complete_checkpoint_count"]), (39, 716268, 14412))
        self.assertEqual((calibration["jobs"], calibration["total_updates"],
                          calibration["complete_checkpoint_count"]), (2, 19688, 397))
        self.assertEqual(result["combined_totals"]["complete_checkpoint_count"], 14809)
        self.assertEqual(formal["R_exposures"] + formal["S_exposures"], 716268 * 16)
        self.assertEqual(calibration["S_exposures"], 0)
        self.assertEqual([j["epochs_r"] for j in result["calibration_jobs"]], [50, 100])
        self.assertTrue(all(j["model"] == "yolo11s" and j["training_seed"] == 17
                            for j in result["calibration_jobs"]))
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["resource_pass"])
        self.assertFalse(result["files_created"])
        self.assertIsNone(result["storage_path"])
        self.assertIsNone(result["complete_run_bytes"])
        json.dumps(result, allow_nan=False)

    def test_checkpoint_terminal_deduplication_and_stage_zero_is_separate(self):
        for updates, expected in ((1, 2), (49, 2), (50, 2), (51, 3), (100, 3), (101, 4)):
            with self.subTest(updates=updates):
                self.assertEqual(checkpoint_count(updates), expected)
        # S20*40/16=50，R50*16/16=50；长组两阶段各2份，边界不能合并。
        result = candidate_resource_plan(40, 16)
        long_job = next(j for j in result["formal_jobs"] if j["strategy"] == "S->R")
        self.assertEqual(long_job["complete_checkpoint_count"], 4)
        self.assertEqual([s["checkpoint_count"] for s in long_job["stages"]], [2, 2])
        self.assertEqual(long_job["stages"][0]["global_final_update"],
                         long_job["stages"][1]["global_start_update"])
        self.assertTrue(long_job["stages"][1]["initial_separate_from_previous_final"])
        self.assertEqual(result["formal_totals"]["complete_checkpoint_count"], 120)

    def test_jobs_and_stage_exposures_match_existing_plan_and_interleave(self):
        n_s, n_r = 11, 7
        expected = plan(n_s, n_r)
        result = candidate_resource_plan(n_s, n_r)
        jobs = result["formal_jobs"]
        self.assertEqual(len({j["candidate_job_id"] for j in jobs}), 39)
        for model in MODEL_NAMES:
            group = [j for j in jobs if j["model"] == model]
            self.assertEqual(len(group), 15 if model == "yolo11s" else 6)
            self.assertEqual({j["training_seed"] for j in group}, {17, 29, 43})
        for job in jobs:
            with self.subTest(job=job["candidate_job_id"]):
                base = expected["strategies"][job["strategy"]]
                self.assertEqual((job["total_updates"], job["R_exposures"], job["S_exposures"]),
                                 (base["updates"], base["R_exposures"], base["S_exposures"]))
                for stage in job["stages"]:
                    domains = [domain_at(job["strategy"], step, expected["U_S"], expected["U_R"])[0]
                               for step in range(stage["global_start_update"] + 1,
                                                 stage["global_final_update"] + 1)]
                    self.assertEqual(stage["S_exposures"], 16 * domains.count("S"))
                    self.assertEqual(stage["R_exposures"], 16 * sum(d != "S" for d in domains))

    def test_unmeasured_and_partial_measurements_are_not_zero_or_PASS(self):
        result = candidate_resource_plan(16000, 2100)
        self.assertEqual(result["status"], "CANDIDATE_UNMEASURED")
        self.assertIsNone(result["formal_totals"]["checkpoint_only_bytes"])
        self.assertTrue(all(j["checkpoint_only_bytes"] is None for j in result["formal_jobs"]))
        partial = candidate_resource_plan(16000, 2100,
                                           complete_state_peak_bytes={"yolo11s": 100})
        self.assertEqual(partial["status"], "CANDIDATE_UNMEASURED")
        self.assertIsNone(partial["formal_totals"]["checkpoint_only_bytes"])
        self.assertIsNone(partial["combined_totals"]["checkpoint_only_bytes"])
        self.assertEqual(partial["calibration_totals"]["checkpoint_only_bytes"], 39700)
        self.assertEqual(set(partial["formal_totals"]["unmeasured_models"]),
                         {"yolo11n", "yolo11m", "yolo11l", "yolo11x"})
        explicit_none = candidate_resource_plan(11, 7,
                                                complete_state_peak_bytes={"yolo11s": None})
        self.assertIsNone(explicit_none["calibration_totals"]["checkpoint_only_bytes"])

    def test_caller_full_state_measurements_are_weighted_checkpoint_only_estimates(self):
        sizes = {model: 100 * (i + 1) for i, model in enumerate(MODEL_NAMES)}
        result = candidate_resource_plan(16000, 2100, complete_state_peak_bytes=sizes)
        expected = sum(j["complete_checkpoint_count"] * sizes[j["model"]]
                       for j in result["formal_jobs"])
        self.assertEqual(result["formal_totals"]["checkpoint_only_bytes"], expected)
        self.assertEqual(result["calibration_totals"]["checkpoint_only_bytes"], 397 * sizes["yolo11s"])
        self.assertEqual(result["combined_totals"]["checkpoint_only_bytes"],
                         expected + 397 * sizes["yolo11s"])
        self.assertEqual(result["status"], "CANDIDATE_CONDITIONAL_CHECKPOINT_ESTIMATE")
        self.assertFalse(result["resource_pass"])
        self.assertFalse(result["measurement_provenance_verified"])
        self.assertFalse(result["hardware_measured"])
        self.assertIsNone(result["wall_time_seconds"])
        self.assertTrue(result["excluded_resource_items"])

    def test_R100_formal_candidate_does_not_merge_or_change_calibration(self):
        result = candidate_resource_plan(16000, 2100, epochs_r=100)
        self.assertEqual(result["formal_totals"]["total_updates"], 952500)
        self.assertEqual(result["formal_totals"]["complete_checkpoint_count"], 19128)
        self.assertEqual(result["calibration_totals"]["complete_checkpoint_count"], 397)
        self.assertEqual(result["budget"]["nominal_epochs_R"], 100)
        self.assertEqual(result["budget"]["nominal_epochs_S"], 20)
        self.assertEqual(result["budget"]["effective_batch"], 16)

    def test_invalid_quantities_measurements_or_budget_expansion_rejected(self):
        for args in ((0, 2100), (16000, -1), (True, 1), (1.5, 1), (1, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                candidate_resource_plan(*args)
        for value in (0, -1, 1.5, True, "model.pt", {"weights_only": 100}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate_resource_plan(11, 7, complete_state_peak_bytes={"yolo11s": value})
        for measurements in ([], {"yolo11s.pt": 100}, {"new_model": 100}):
            with self.subTest(measurements=measurements), self.assertRaises(ValueError):
                candidate_resource_plan(11, 7, complete_state_peak_bytes=measurements)
        for epochs in (0, 25, 50.0, True, 200):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                candidate_resource_plan(11, 7, epochs_r=epochs)
        for value in (0, -1, True, 50.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_count(value)


if __name__ == "__main__":
    unittest.main()
