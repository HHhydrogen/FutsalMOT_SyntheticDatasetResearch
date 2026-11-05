"""给定手算框的CPU测试；不读取图像、GT文件，不加载模型或创建报告。"""
import copy
import json
import math
import unittest

from futsalmot.metrics import player_ignore_ap

BOX = [0, 0, 2, 2]


def prediction(box=BOX, score=0.9, class_id=0):
    return {"bbox": list(box), "score": score, "class_id": class_id}


def image(gt=(), predictions=(), ignore=()):
    return {"positive_gt": list(gt), "predictions": list(predictions), "ignore": list(ignore)}


def at(result, threshold):
    return next(row for row in result["per_iou"] if row["iou"] == threshold)


class MetricsTests(unittest.TestCase):
    def test_perfect_and_continuous_area_without_plus_one(self):
        inputs = {"a": image([BOX], [prediction()])}
        before = copy.deepcopy(inputs)
        result = player_ignore_ap(inputs)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["metric_name"], "custom_player_ignore_AP@300")
        self.assertFalse(result["official"])
        self.assertEqual(result["AP50_95"], 1)
        self.assertEqual(result["AP50"], 1)
        self.assertEqual(result["AP75"], 1)
        self.assertEqual(inputs, before)
        touching = player_ignore_ap({"a": image([[0, 0, 1, 1]], [prediction([1, 0, 2, 1])])})
        self.assertEqual(touching["AP50_95"], 0)
        # 两个大框面积相加会溢出，单框面积仍合法；IoU仍须为1。
        large = [0, 0, 1e154, 1e154]
        self.assertEqual(player_ignore_ap({"a": image([large], [prediction(large)])})["AP50_95"], 1)

    def test_high_rank_false_positive_lowers_AP_and_101_point_recall_grid(self):
        fp = prediction([10, 10, 11, 11], 0.95)
        result = player_ignore_ap({"a": image([BOX], [fp, prediction(score=0.8)])})
        self.assertEqual(result["AP50_95"], 0.5)
        self.assertEqual(at(result, 0.5)["TP"], 1)
        self.assertEqual(at(result, 0.5)["FP"], 1)
        # 2 GT只召回1个，0..0.50共51个召回采样点有precision=1。
        half_recall = player_ignore_ap({"a": image([BOX, [10, 10, 12, 12]], [prediction()])})
        self.assertAlmostEqual(half_recall["AP50"], 51 / 101)

    def test_duplicate_predictions_and_image_matching_are_independent(self):
        result = player_ignore_ap({
            "a": image([BOX], [prediction(score=0.9), prediction(score=0.8)]),
            "b": image([BOX], [prediction(score=0.7)]),
            "c": image([], [prediction(score=0.6)]),
        })
        self.assertEqual(at(result, 0.5)["GT"], 2)
        self.assertEqual(at(result, 0.5)["TP"], 2)
        self.assertEqual(at(result, 0.5)["FP"], 2)
        self.assertEqual(at(result, 0.5)["ignored"], 0)

    def test_highest_IoU_and_each_threshold_get_independent_matching(self):
        gt_b, gt_a = [1, 0, 11, 10], [0, 0, 10, 10]
        result = player_ignore_ap({
            "a": image([gt_b, gt_a], [
                prediction(gt_a, 0.9),
                prediction([0, 0, 9, 10], 0.8),
            ])
        })
        self.assertEqual(at(result, 0.5)["TP"], 2)
        self.assertEqual(at(result, 0.8)["TP"], 1)
        self.assertEqual(at(result, 0.8)["FP"], 1)
        fractional = player_ignore_ap({
            "a": image([[0, 0, 5, 2]], [prediction([0, 0, 3, 2])])
        })
        self.assertEqual(fractional["AP50"], 1)
        self.assertEqual(fractional["AP75"], 0)
        self.assertAlmostEqual(fractional["AP50_95"], 0.3)

    def test_positive_priority_and_duplicate_can_be_absorbed_by_ignore(self):
        result = player_ignore_ap({"a": image(
            [BOX], [prediction(score=0.9), prediction(score=0.8)], [BOX])})
        self.assertEqual(result["AP50_95"], 1)
        for row in result["per_iou"]:
            self.assertEqual((row["TP"], row["FP"], row["ignored"]), (1, 0, 1))
            self.assertEqual(row["GT"], 1)

    def test_ignore_IoA_half_boundary_and_any_region_not_union(self):
        far_gt = [10, 10, 11, 11]
        result = player_ignore_ap({"a": image(
            [far_gt], [prediction()], [[1, 0, 2, 2]])})
        self.assertEqual((at(result, 0.5)["FP"], at(result, 0.5)["ignored"]), (0, 1))
        self.assertEqual(result["AP50_95"], 0)
        result = player_ignore_ap({"a": image(
            [far_gt], [prediction([0, 0, 4, 1])],
            [[0, 0, 1, 1], [3, 0, 4, 1]])})
        # 两个区域各自IoA=.25，不把二者的并集当成一个.5区域。
        self.assertEqual((at(result, 0.5)["FP"], at(result, 0.5)["ignored"]), (1, 0))
        result = player_ignore_ap({"a": image(
            [far_gt], [prediction()], [[100, 100, 101, 101], BOX])})
        self.assertEqual(at(result, 0.5)["ignored"], 1)

    def test_empty_GT_is_undefined_but_positive_GT_without_predictions_is_zero(self):
        for inputs in ({}, {"a": image()}, {"a": image([], [prediction()])}):
            with self.subTest(inputs=inputs):
                result = player_ignore_ap(inputs)
                self.assertEqual(result["status"], "NO_POSITIVE_GT")
                self.assertIsNone(result["AP50_95"])
                self.assertIsNone(result["AP50"])
                self.assertTrue(all(row["AP"] is None for row in result["per_iou"]))
                json.dumps(result, allow_nan=False)
        result = player_ignore_ap({"a": image([BOX])})
        self.assertEqual(result["AP50_95"], 0)
        self.assertEqual(at(result, 0.5)["FN"], 1)

    def test_strict_score_threshold_and_maxdet_happens_before_ignore(self):
        result = player_ignore_ap({"a": image([BOX], [
            prediction(score=0.001), prediction(score=0.0010001), prediction(score=0.0)])})
        self.assertEqual(result["counts"]["predictions_evaluated"], 1)
        self.assertEqual(result["counts"]["predictions_below_or_equal_conf"], 2)
        self.assertEqual(result["AP50"], 1)
        far_gt = [1000, 0, 1001, 1]
        predictions = [prediction(score=0.9) for _ in range(300)]
        predictions.append(prediction(far_gt, score=0.8))
        result = player_ignore_ap({"a": image([far_gt], predictions, [BOX])})
        self.assertEqual(result["counts"]["predictions_truncated_by_maxdet"], 1)
        self.assertEqual(result["counts"]["predictions_evaluated"], 300)
        self.assertEqual(at(result, 0.5)["ignored"], 300)
        self.assertEqual(result["AP50_95"], 0)  # 不用ignore腾出的名额补回低分TP。

    def test_ties_use_UID_order_then_original_prediction_index(self):
        a = image([], [prediction(score=0.5)])
        b = image([BOX], [prediction(score=0.5)])
        left = player_ignore_ap({"a": a, "b": b})
        right = player_ignore_ap({"b": b, "a": a})
        self.assertEqual(left, right)
        self.assertEqual(left["AP50"], 0.5)
        explicit = player_ignore_ap({"a": a, "b": b}, image_uid_order=["b", "a"])
        self.assertEqual(explicit["AP50"], 1)
        same_image = player_ignore_ap({"a": image([BOX], [
            prediction([10, 10, 11, 11], 0.5), prediction(score=0.5)])})
        self.assertEqual(same_image["AP50"], 0.5)

    def test_invalid_inputs_and_frozen_order_are_rejected(self):
        invalid_predictions = [
            prediction([0, 0, 0, 1]), prediction([1, 0, 0, 1]),
            prediction([0, 0, math.inf, 1]), prediction([0, 0, math.nan, 1]),
            prediction([0, 0, 1, 1], math.nan), prediction(score=-0.1),
            prediction(score=1.1), prediction(score=True), prediction(class_id=1),
            prediction(class_id=False), {"bbox": [0, 0, 1], "score": 0.9},
            # 无效框即使低于阈值也不能被静默绕过。
            prediction([0, 0, 0, 1], 0.0),
        ]
        for bad in invalid_predictions:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                player_ignore_ap({"a": image([BOX], [bad])})
        with self.assertRaises(ValueError):
            player_ignore_ap({"a": {"positive_gt": [], "predictions": []}})
        with self.assertRaises(ValueError):
            player_ignore_ap({"a": image([[0, 0, -1, 1]])})
        with self.assertRaises(ValueError):
            player_ignore_ap({"a": image(ignore=[[0, 0, math.inf, 1]])})
        for order in ([], ["a", "a"], ["other"]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                player_ignore_ap({"a": image()}, image_uid_order=order)
        with self.assertRaises(ValueError):
            player_ignore_ap({1: image()})
        with self.assertRaises(ValueError):
            player_ignore_ap([])


if __name__ == "__main__":
    unittest.main()
