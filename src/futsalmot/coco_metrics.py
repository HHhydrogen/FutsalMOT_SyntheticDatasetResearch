"""标准COCO BBox给定框评价；只接收已完成NMS/逆几何的person预测。

本模块调用固定pycocotools 2.0.10的COCO/COCOeval，不实现另一套AP公式，
不读取图像、标注文件或网络，不做NMS、置信度筛选、裁框或推理。结果采用
COCO默认IoU/召回/面积/maxDets设置；数据集和person类别子集仍由调用者冻结，
使用官方评价器不表示获得COCO或SoccerNet官方排行榜成绩。

positive_gt保持原图连续xyxy和未裁面积，全部映射为category_id=1、iscrowd=0。
ignore必须显式为空；非球员预测作为背景FP，不能把ignore框冒充COCO crowd。
已知困难实例不筛除，官方按面积分组的处理原样保留。相同分数由COCO稳定
排序，顺序取冻结image_uid_order和每图原预测序号。输入不被原地修改。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
import copy
from importlib import metadata
import io
import math
from numbers import Integral, Real

PYCOCOTOOLS_VERSION = "2.0.10"
METRIC_NAME = "coco_player_bbox_AP@100"
STAT_NAMES = (
    "AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
)


def _sequence(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(label + "必须是有序序列")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label + "必须是有限实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(label + "必须是有限实数")
    return value


def _xywh(value, label):
    value = _sequence(value, label)
    if len(value) != 4:
        raise ValueError(label + "必须是四个连续xyxy坐标")
    x1, y1, x2, y2 = [_number(v, label) for v in value]
    w, h = x2 - x1, y2 - y1
    area = w * h
    if w <= 0 or h <= 0 or not math.isfinite(area) or area <= 0:
        raise ValueError(label + "必须有有限正面积")
    return [x1, y1, w, h], area


def _backend():
    try:
        version = metadata.version("pycocotools")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("BLOCKED: 需要已安装且锁定的pycocotools 2.0.10；不会自动安装") from exc
    if version != PYCOCOTOOLS_VERSION:
        raise RuntimeError("BLOCKED: pycocotools版本须为2.0.10，实际为" + version)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


def _json_native(value):
    """仅序列化官方返回的标量/数组；不改变数值或计算评价。"""
    if isinstance(value, Mapping):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def coco_player_ap(images, *, image_uid_order):
    """以官方默认COCO BBox协议评价给定框，返回0..1尺度指标和完整原始数组。

    images[uid]必须提供positive_gt、predictions、ignore=[]。预测为
    {"bbox": [x1,y1,x2,y2], "score": 0..1, "class_id": 0}；class_id省略
    时按person0约定。可提供正整数width/height，仅作来源元数据，不裁框。
    image_uid_order必须完整、唯一且显式，映射为递增整数image_id。

    官方无GT的stats/数组项保留-1；便于阅读的metrics对应项为None。
    有GT且无预测时由官方评价器给出AP=0。该函数不保存文件；调用者须
    将返回的input/params/evaluation精确绑定实际模型及预测协议后保存。
    """
    if not isinstance(images, Mapping):
        raise ValueError("images必须是UID到记录的映射")
    if any(not isinstance(uid, str) or not uid for uid in images):
        raise ValueError("image UID必须是非空字符串")
    order = list(_sequence(image_uid_order, "image_uid_order"))
    if (any(not isinstance(uid, str) or not uid for uid in order)
            or len(order) != len(set(order)) or set(order) != set(images)):
        raise ValueError("image_uid_order必须完整且无重复")
    gt_dataset = {"info": {}, "images": [], "categories": [
        {"id": 1, "name": "person", "supercategory": "person"}], "annotations": []}
    detections = []
    uid_mapping = []
    for image_id, uid in enumerate(order, start=1):
        image = images[uid]
        if (not isinstance(image, Mapping)
                or not {"positive_gt", "predictions", "ignore"} <= image.keys()):
            raise ValueError("每图必须显式提供positive_gt、predictions、ignore")
        ignores = _sequence(image["ignore"], uid + ":ignore")
        if len(ignores):
            raise ValueError("BLOCKED: 标准COCO主评价要求ignore为空，不能映射为iscrowd")
        image_record = {"id": image_id}
        for dim in ("width", "height"):
            if dim in image:
                number = image[dim]
                if isinstance(number, bool) or not isinstance(number, Integral) or number <= 0:
                    raise ValueError(dim + "必须是正整数")
                image_record[dim] = int(number)
        gt_dataset["images"].append(image_record)
        uid_mapping.append({"image_uid": uid, "image_id": image_id})
        for box in _sequence(image["positive_gt"], uid + ":positive_gt"):
            bbox, area = _xywh(box, uid + ":positive_gt")
            gt_dataset["annotations"].append({
                "id": len(gt_dataset["annotations"]) + 1, "image_id": image_id,
                "category_id": 1, "bbox": bbox, "area": area, "iscrowd": 0,
            })
        for prediction in _sequence(image["predictions"], uid + ":predictions"):
            if not isinstance(prediction, Mapping) or not {"bbox", "score"} <= prediction.keys():
                raise ValueError("prediction必须提供bbox和score")
            cls = prediction.get("class_id", 0)
            if isinstance(cls, bool) or not isinstance(cls, Integral) or cls != 0:
                raise ValueError("只接受class_id=0的person预测")
            bbox, _ = _xywh(prediction["bbox"], uid + ":prediction")
            score = _number(prediction["score"], "score")
            if not 0 <= score <= 1:
                raise ValueError("score必须在[0,1]")
            detections.append({"image_id": image_id, "category_id": 1,
                               "bbox": bbox, "score": score})

    COCO, COCOeval = _backend()
    # COCO API会原地加入annotation字段，因此传独立副本以保留精确输入收据。
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(gt_dataset)
        gt.createIndex()
        if detections:
            dt = gt.loadRes(copy.deepcopy(detections))
        else:
            # 官方loadRes访问anns[0]，空结果使用同一个官方COCO容器建空索引。
            dt = COCO()
            dt.dataset = {"info": {}, "images": copy.deepcopy(gt_dataset["images"]),
                          "categories": copy.deepcopy(gt_dataset["categories"]),
                          "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType="bbox")
        evaluator.params.imgIds = [row["image_id"] for row in uid_mapping]
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats.tolist()
    p = evaluator.params
    metrics = {name: (None if value < 0 else float(value))
               for name, value in zip(STAT_NAMES, stats)}
    return {
        "metric_name": METRIC_NAME,
        "official_evaluator": True,
        "official_benchmark": False,
        "implementation": "pycocotools.COCOeval",
        "pycocotools_version": PYCOCOTOOLS_VERSION,
        "status": "OK" if gt_dataset["annotations"] else "NO_POSITIVE_GT",
        **metrics,
        "metrics": metrics,
        "raw_stats": stats,
        "raw_stats_names": list(STAT_NAMES),
        "counts": {"images": len(order), "positive_gt": len(gt_dataset["annotations"]),
                   "predictions_input": len(detections), "ignore_regions": 0},
        "uid_mapping": uid_mapping,
        "params": {
            "iouType": p.iouType, "iouThrs": p.iouThrs.tolist(),
            "recThrs": p.recThrs.tolist(), "maxDets": list(p.maxDets),
            "areaRng": copy.deepcopy(p.areaRng), "areaRngLbl": list(p.areaRngLbl),
            "useCats": int(p.useCats), "catIds": [int(v) for v in p.catIds],
            "imgIds": [int(v) for v in p.imgIds],
        },
        "evaluation": {
            "counts": list(evaluator.eval["counts"]),
            "precision": evaluator.eval["precision"].tolist(),
            "recall": evaluator.eval["recall"].tolist(),
            "scores": evaluator.eval["scores"].tolist(),
            "eval_images": _json_native(evaluator.evalImgs),
            "missing_gt_sentinel": -1,
            "precision_axes": ["IoU", "recall", "category", "area", "maxDets"],
            "recall_axes": ["IoU", "category", "area", "maxDets"],
        },
        "input": {"coco_gt": gt_dataset, "coco_detections": detections},
        "protocol": {
            "image_order": "explicit_frozen_image_uid_order",
            "equal_score_order": "COCO stable mergesort; image_id then input prediction index",
            "prediction_precondition": "completed person0 NMS and original-image inverse geometry",
            "score_filter": "none in evaluator adapter",
            "gt_boxes": "original unclipped continuous xyxy converted to xywh; no difficulty filter",
            "ignore": "must be explicitly empty; all positive iscrowd=0",
            "non_player_predictions": "background FP under ordinary COCO matching",
            "benchmark_scope": "caller-defined dataset/person subset; not an official leaderboard claim",
        },
    }
