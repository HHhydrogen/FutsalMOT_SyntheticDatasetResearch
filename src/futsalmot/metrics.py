"""给定框的自定义player/ignore AP@300；不是推理或完整评价入口。

调用方须保证：同一冻结UID视图、原图连续xyxy、预测已完成且没有NMS超时/
部分结果、仅person类、模型与推理协议已冻结。本模块不读取数据、不裁框、不做
NMS、不加载网络。原GT框保持不裁剪；面积是连续乘积，没有像素坐标+1。

positive先一对一匹配；剩余预测对任一显式ignore框的交集/预测面积>=0.5则
忽略。ignore可吸收多个预测，且不进入召回分母。若positive与ignore重叠，
重复预测也可能被ignore吸收；这是本自定义规则的局限，不是COCO crowd语义。
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real

METRIC_NAME = "custom_player_ignore_AP@300"
IOU_THRESHOLDS = tuple((50 + 5 * i) / 100 for i in range(10))
CONFIDENCE_THRESHOLD = 0.001
IGNORE_IOA_THRESHOLD = 0.5
MAX_DETECTIONS = 300


def _sequence(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(label + "必须是有序序列")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label + "必须是有限实数")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + "必须是有限实数")
    return number


def _box(value, label):
    value = _sequence(value, label)
    if len(value) != 4:
        raise ValueError(label + "必须是四个xyxy坐标")
    x1, y1, x2, y2 = (_number(v, label) for v in value)
    width, height = x2 - x1, y2 - y1
    area = width * height
    if width <= 0 or height <= 0 or not math.isfinite(area) or area <= 0:
        raise ValueError(label + "必须有有限正面积")
    return (x1, y1, x2, y2, area)


def _intersection(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def _iou(a, b):
    inter = _intersection(a, b)
    # 先按较大面积缩放，避免两个合法大框面积相加时溢出。
    scale = max(a[4], b[4])
    fraction = inter / scale
    return fraction / (a[4] / scale + b[4] / scale - fraction)


def _ap101(outcomes, positive_count):
    if not positive_count:
        return None
    if not outcomes:
        return 0.0
    tp, fp, recalls, precision = 0, 0, [], []
    for is_tp in outcomes:
        tp += int(is_tp)
        fp += int(not is_tp)
        recalls.append(tp / positive_count)
        precision.append(tp / (tp + fp))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    sampled = []
    for i in range(101):
        position = bisect_left(recalls, i / 100)
        sampled.append(precision[position] if position < len(precision) else 0.0)
    return math.fsum(sampled) / 101


def player_ignore_ap(images, *, image_uid_order=None):
    """计算固定规则的AP，返回0..1尺度指标及每IoU计数；不修改输入。

    images是映射：
      image_uid -> {
        "positive_gt": [[x1,y1,x2,y2], ...],
        "ignore": [[x1,y1,x2,y2], ...],  # 必须显式提供，允许空
        "predictions": [
          {"bbox": [x1,y1,x2,y2], "score": 0.9, "class_id": 0}, ...
        ]
      }

    class_id省略时按调用约定为0；显式非0拒绝。score须有限且在[0,1]；
    所有框须有限正面积，即使预测会被score阈值过滤也校验其输入。
    image_uid_order须是完整、唯一的冻结UID顺序；省略则采用UID字典序，
    不受dict插入顺序影响。同分依次按该UID顺序和原预测序号稳定排序；
    GT同IoU时优先原GT序号。每图score>0.001后最多保留300条，先截断再
    做ignore匹配，不因ignored预测腾出名额。每IoU独立贪心一对一匹配。
    """
    if not isinstance(images, Mapping):
        raise ValueError("images必须是image UID到记录的映射")
    if any(not isinstance(uid, str) or not uid for uid in images):
        raise ValueError("image UID必须是非空字符串")
    if image_uid_order is None:
        order = sorted(images)
        order_policy = "lexicographic_image_uid"
    else:
        order = list(_sequence(image_uid_order, "image_uid_order"))
        if any(not isinstance(uid, str) or not uid for uid in order):
            raise ValueError("image_uid_order必须包含非空字符串")
        if len(order) != len(set(order)) or set(order) != set(images):
            raise ValueError("image_uid_order必须完整且无重复")
        order_policy = "explicit_frozen_image_uid_order"

    counts = {"images": len(order), "positive_gt": 0, "ignore_regions": 0,
              "predictions_input": 0, "predictions_below_or_equal_conf": 0,
              "predictions_after_conf": 0, "predictions_truncated_by_maxdet": 0,
              "predictions_evaluated": 0}
    candidates, ground_truth = [], {}
    for image_rank, uid in enumerate(order):
        image = images[uid]
        if not isinstance(image, Mapping) or not {"positive_gt", "ignore", "predictions"} <= image.keys():
            raise ValueError("每图必须显式提供positive_gt、ignore、predictions")
        positives = [_box(v, uid + ":positive_gt")
                     for v in _sequence(image["positive_gt"], "positive_gt")]
        ignores = [_box(v, uid + ":ignore")
                   for v in _sequence(image["ignore"], "ignore")]
        predictions = _sequence(image["predictions"], "predictions")
        ground_truth[uid] = positives
        counts["positive_gt"] += len(positives)
        counts["ignore_regions"] += len(ignores)
        counts["predictions_input"] += len(predictions)
        selected = []
        for original_index, prediction in enumerate(predictions):
            if not isinstance(prediction, Mapping) or not {"bbox", "score"} <= prediction.keys():
                raise ValueError("prediction必须提供bbox和score")
            category = prediction.get("class_id", 0)
            if isinstance(category, bool) or not isinstance(category, Integral) or category != 0:
                raise ValueError("只接受class_id=0的person预测")
            box = _box(prediction["bbox"], uid + ":prediction")
            score = _number(prediction["score"], "score")
            if not 0 <= score <= 1:
                raise ValueError("score必须在[0,1]")
            if score > CONFIDENCE_THRESHOLD:
                selected.append((score, original_index, box))
            else:
                counts["predictions_below_or_equal_conf"] += 1
        selected.sort(key=lambda item: (-item[0], item[1]))
        counts["predictions_after_conf"] += len(selected)
        counts["predictions_truncated_by_maxdet"] += max(0, len(selected) - MAX_DETECTIONS)
        selected = selected[:MAX_DETECTIONS]
        counts["predictions_evaluated"] += len(selected)
        for score, index, box in selected:
            overlaps = [_iou(box, gt) for gt in positives]
            ignored = any(_intersection(box, region) / box[4] >= IGNORE_IOA_THRESHOLD
                          for region in ignores)
            candidates.append((score, image_rank, index, uid, overlaps, ignored))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    per_iou = []
    for threshold in IOU_THRESHOLDS:
        matched = {uid: set() for uid in order}
        outcomes, tp, fp, ignored_count = [], 0, 0, 0
        for _, _, _, uid, overlaps, ignore_hit in candidates:
            best, best_iou = None, -1.0
            for gt_index, overlap in enumerate(overlaps):
                if gt_index not in matched[uid] and overlap >= threshold and overlap > best_iou:
                    best, best_iou = gt_index, overlap
            if best is not None:
                matched[uid].add(best)
                outcomes.append(True)
                tp += 1
            elif ignore_hit:
                ignored_count += 1
            else:
                outcomes.append(False)
                fp += 1
        per_iou.append({"iou": threshold, "AP": _ap101(outcomes, counts["positive_gt"]),
                        "GT": counts["positive_gt"], "TP": tp, "FP": fp,
                        "FN": counts["positive_gt"] - tp, "ignored": ignored_count,
                        "recall": tp / counts["positive_gt"] if counts["positive_gt"] else None})
    defined = bool(counts["positive_gt"])
    return {
        "metric_name": METRIC_NAME, "official": False,
        "status": "OK" if defined else "NO_POSITIVE_GT",
        "AP50_95": math.fsum(row["AP"] for row in per_iou) / len(per_iou) if defined else None,
        "AP50": per_iou[0]["AP"], "AP75": per_iou[5]["AP"],
        "per_iou": per_iou, "counts": counts,
        "protocol": {"class_id": 0, "area": "all", "max_detections": MAX_DETECTIONS,
                     "confidence_strictly_greater_than": CONFIDENCE_THRESHOLD,
                     "ignore_ioa_greater_or_equal": IGNORE_IOA_THRESHOLD,
                     "iou_thresholds": list(IOU_THRESHOLDS), "recall_points": 101,
                     "image_uid_order": order, "tie_order_policy": order_policy,
                     "nms_performed": False, "caller_must_guarantee_complete_nms": True},
    }
