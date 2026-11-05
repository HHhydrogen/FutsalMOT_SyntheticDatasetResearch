"""真实pycocotools CPU给定框夹具；不读取真实数据/图像，不运行模型。"""
import copy
from importlib import metadata
import json
import math
import unittest
from unittest import mock

from futsalmot.coco_metrics import coco_player_ap, PYCOCOTOOLS_VERSION

BOX = [0, 0, 10, 10]
FAR = [30, 30, 40, 40]


def pred(box=BOX, score=0.9, class_id=0):
    return {"bbox": list(box), "score": score, "class_id": class_id}


def image(gt=(), predictions=(), **extra):
    return {"positive_gt": list(gt), "predictions": list(predictions), "ignore": [], **extra}


def run(images, order=None):
    return coco_player_ap(images, image_uid_order=list(images) if order is None else order)


class CocoMetricsTests(unittest.TestCase):
    def test_actual_backend_perfect_default_params_and_full_raw_arrays(self):
        self.assertEqual(metadata.version("pycocotools"), PYCOCOTOOLS_VERSION)
        inputs = {"a": image([BOX], [pred()], width=100, height=100)}
        original = copy.deepcopy(inputs)
        result = run(inputs)
        self.assertEqual(inputs, original)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["metric_name"], "coco_player_bbox_AP@100")
        self.assertTrue(result["official_evaluator"])
        self.assertFalse(result["official_benchmark"])
        for key in ("AP50_95", "AP50", "AP75", "AP_small", "AR_max100"):
            self.assertAlmostEqual(result[key], 1)
        self.assertIsNone(result["AP_medium"])
        self.assertIsNone(result["AP_large"])
        self.assertEqual(result["params"]["maxDets"], [1, 10, 100])
        self.assertEqual(result["params"]["useCats"], 1)
        self.assertEqual(result["params"]["catIds"], [1])
        self.assertEqual(len(result["params"]["iouThrs"]), 10)
        self.assertEqual(len(result["params"]["recThrs"]), 101)
        self.assertEqual(result["params"]["areaRng"],
                         [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]])
        raw = result["evaluation"]
        self.assertEqual(raw["counts"], [10, 101, 1, 4, 3])
        self.assertEqual(len(raw["precision"]), 10)
        self.assertEqual(len(raw["precision"][0]), 101)
        self.assertEqual(len(raw["precision"][0][0][0]), 4)
        self.assertEqual(len(raw["precision"][0][0][0][0]), 3)
        self.assertEqual(len(raw["recall"][0][0]), 4)
        self.assertEqual(len(raw["scores"][0]), 101)
        self.assertEqual(len(result["raw_stats"]), 12)
        self.assertEqual(result["raw_stats"][4:6], [-1, -1])
        json.dumps(result, allow_nan=False)

    def test_duplicate_is_FP_and_background_ranking_affects_official_AP(self):
        data = {"a": image([BOX, FAR], [
            pred(score=0.9), pred(score=0.8), pred(FAR, 0.7)])}
        result = run(data)
        self.assertAlmostEqual(result["AP50"], (51 + 50 * 2 / 3) / 101)
        all_area = result["evaluation"]["eval_images"][0]
        self.assertEqual(all_area["dtMatches"][0], [1, 0, 2])
        self.assertEqual(all_area["dtIgnore"][0], [False, False, False])
        fp_first = run({"a": image([BOX], [pred(FAR, .95), pred(score=.8)])})
        self.assertAlmostEqual(fp_first["AP50_95"], 0.5)

    def test_empty_predictions_and_no_GT_have_official_distinct_results(self):
        positive = run({"a": image([BOX])})
        self.assertEqual(positive["AP50_95"], 0)
        self.assertEqual(positive["AR_max100"], 0)
        self.assertEqual(positive["status"], "OK")
        for data in ({}, {"a": image()}, {"a": image([], [pred()])}):
            with self.subTest(data=data):
                result = run(data)
                self.assertEqual(result["status"], "NO_POSITIVE_GT")
                self.assertIsNone(result["AP50_95"])
                self.assertEqual(result["raw_stats"], [-1] * 12)
                self.assertEqual(result["evaluation"]["precision"][0][0][0][0][2], -1)
                json.dumps(result, allow_nan=False)

    def test_original_out_of_image_GT_coordinates_and_area_preserved(self):
        outside = [-10, -20, 10, 30]
        result = run({"a": image([outside], [pred(outside)], width=5, height=5)})
        gt = result["input"]["coco_gt"]["annotations"][0]
        self.assertEqual(gt["bbox"], [-10, -20, 20, 50])
        self.assertEqual(gt["area"], 1000)
        self.assertEqual(gt["iscrowd"], 0)
        self.assertEqual(gt["category_id"], 1)
        self.assertAlmostEqual(result["AP50_95"], 1)

    def test_stable_frozen_UID_order_and_prediction_ties(self):
        a = image([BOX], [pred(score=.5)])
        b = image([], [pred(score=.5)])
        one = run({"b": b, "a": a}, ["a", "b"])
        two = run({"a": a, "b": b}, ["a", "b"])
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["AP50_95"], 1)
        self.assertAlmostEqual(run({"a": a, "b": b}, ["b", "a"])["AP50_95"], .5)
        self.assertEqual(one["uid_mapping"], [
            {"image_uid": "a", "image_id": 1}, {"image_uid": "b", "image_id": 2}])
        within = run({"a": image([BOX], [pred(FAR, .5), pred(score=.5)])})
        self.assertAlmostEqual(within["AP50_95"], .5)

    def test_maxdet100_before_matching_and_no_additional_score_filter(self):
        detections = [pred(FAR, .9)] * 100 + [pred(score=.1)]
        result = run({"a": image([BOX], detections)})
        self.assertEqual(result["AP50_95"], 0)
        self.assertEqual(result["counts"]["predictions_input"], 101)
        self.assertEqual(len(result["input"]["coco_detections"]), 101)
        self.assertEqual(len(result["evaluation"]["eval_images"][0]["dtIds"]), 100)
        self.assertAlmostEqual(run({"a": image([BOX], [pred(score=0)])})["AP50_95"], 1)

    def test_size_metrics_use_original_continuous_box_area(self):
        boxes = [[0, 0, 10, 10], [20, 0, 60, 40], [100, 0, 200, 100]]
        result = run({"a": image(boxes, [pred(box) for box in boxes])})
        for name in ("AP_small", "AP_medium", "AP_large",
                     "AR_small", "AR_medium", "AR_large"):
            self.assertAlmostEqual(result[name], 1)
        self.assertEqual([row["area"] for row in result["input"]["coco_gt"]["annotations"]],
                         [100, 1600, 10000])

    def test_ignore_invalid_boxes_scores_categories_dimensions_and_UIDs_rejected(self):
        invalid = [
            {"a": image([BOX], [pred()], ignore=[BOX])},
            {"a": {"positive_gt": [], "predictions": []}},
            {"a": image([[0, 0, 0, 1]])},
            {"a": image([[0, 0, math.inf, 1]])},
            {"a": image([], [pred(score=math.nan)])},
            {"a": image([], [pred(score=1.1)])},
            {"a": image([], [pred(class_id=1)])},
            {"a": image([], [pred(class_id=True)])},
            {"a": image(width=0)},
            {"a": image(height=True)},
            {"": image()},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                run(data)
        for order in ([], ["a", "a"], ["b"], None):
            with self.subTest(order=order), self.assertRaises(ValueError):
                coco_player_ap({"a": image()}, image_uid_order=order)

    def test_backend_is_lazy_and_missing_or_wrong_version_fails_explicitly(self):
        import futsalmot.coco_metrics as module
        with mock.patch.object(module.metadata, "version", return_value="2.0.9"):
            with self.assertRaisesRegex(RuntimeError, "BLOCKED"):
                run({"a": image()})
        with mock.patch.object(module.metadata, "version",
                               side_effect=metadata.PackageNotFoundError("pycocotools")):
            with self.assertRaisesRegex(RuntimeError, "不会自动安装"):
                run({"a": image()})


if __name__ == "__main__":
    unittest.main()
