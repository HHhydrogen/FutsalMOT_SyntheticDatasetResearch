"""给定已decode框的候选CPU后处理与小型评价落盘；不是head decoder/模型推理。

只接960输入坐标XYXY/score/class逐行列表或CPU浮点Nx6 Tensor。先筛person，
再同类NMS；其它79类不占person max_det，不能冒称跨80类完整原生验证口径。
固定FP32 CPU torchvision NMS，记录同分及原输入序号，不扰动score造新tie规则。
正式ignore/AP/预处理定义仍待直接讨论，本模块始终candidate_only、非官方。
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import stat

VERSIONS = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "ultralytics": "8.3.237"}
DEFINITION = {
    "candidate_only": True, "official_benchmark": False, "input": "decoded_960_xyxy_score_class",
    "class_id": 0, "confidence_strictly_greater_than": .001, "nms_iou": .7, "max_det": 300,
    "nms_device": "cpu", "nms_dtype": "float32", "tta": False,
    "filter_order": ["person", "confidence", "nms", "max_det", "restore", "clip_drop_degenerate"],
    "tie_policy": "native_CPU_NMS_input_order_recorded_no_cross_backend_tie_guarantee",
    "degenerate_policy": "discard_after_restore_clip_no_maxdet_refill",
    "metric": "custom_player_ignore_AP@300", "metric_formally_adopted": False,
}


@lru_cache(maxsize=1)
def _backend():
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise RuntimeError("仅核对过锁定Torch/Torchvision/Ultralytics版本")
    import torch
    from torchvision.ops import nms
    return torch, nms


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("需要有限原生数值")
    return float(value)


def _geometry(uid, geometry):
    required = {"original_shape", "input_shape", "resized_shape", "scale", "padding_ltrb",
                "source_uid", "channel_order", "letterbox_candidate_only"}
    if not isinstance(uid, str) or not uid or not isinstance(geometry, dict) or required-geometry.keys():
        raise ValueError("缺明确UID或既有完整letterbox geometry")
    g = deepcopy(geometry)
    if g["source_uid"] != uid or g["channel_order"] != "RGB":
        raise ValueError("UID/RGB几何绑定不符")
    if g["letterbox_candidate_only"] is not True:
        a=g.get("training_input",{})
        if (g["letterbox_candidate_only"] is not False or a.get("view")!="val"
            or a.get("role_policy")!="role_policy_v2_known_non_target_background_invalid_unknown_ignore_positive_priority"
            or a.get("matrix_from_letterbox")!=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            or a.get("hsv_additive_gains")!=[0.,0.,0.] or a.get("fliplr_applied") is not False
            or a.get("default_positive_filter") is not False or a.get("scale")!=1.):
            raise ValueError("正式推理仅接已批准的无增强val直接LetterBox，不能逆推训练仿射")
    if tuple(g["input_shape"]) != (960, 960) or len(g["original_shape"]) != 2:
        raise ValueError("只支持既有960 letterbox geometry")
    h, w = g["original_shape"]
    if any(type(v) is not int or v < 1 for v in (h, w)):
        raise ValueError("原shape须为正整数")
    ratio = min(960/h, 960/w)
    resized = (round(h*ratio), round(w*ratio))
    dw, dh = (960-resized[1])/2, (960-resized[0])/2
    padding = (round(dw-.1), round(dh-.1), round(dw+.1), round(dh+.1))
    if (tuple(g["resized_shape"]) != resized or tuple(g["scale"]) != (ratio, ratio)
            or tuple(g["padding_ltrb"]) != padding):
        raise ValueError("geometry不属于既有直接round letterbox，不推断ceil resize")
    return g, h, w, ratio, padding


def postprocess_decoded_person(uid, decoded_predictions, *, geometry):
    """候选单图结果，保留阈值前所有行及过滤原因；不修改输入，不读取文件。

    canonical FP32比较与NMS；原行和FP32决策值均保留。先NMS/max_det，再以传入
    float比例和整数pad逆变换并裁原图，裁剪退化框明确丢弃且不补名额。
    """
    g, h, w, ratio, padding = _geometry(uid, geometry)
    torch, nms = _backend()
    storage = "python_rows"
    if isinstance(decoded_predictions, torch.Tensor):
        if (decoded_predictions.device.type != "cpu" or not decoded_predictions.is_floating_point()
                or decoded_predictions.ndim != 2 or decoded_predictions.shape[1] != 6):
            raise ValueError("Tensor必须为CPU浮点Nx6已decode预测，不接受raw head或GPU")
        if decoded_predictions.shape[0] > 30000:
            raise ValueError("候选输入超过30000行上限，不静默截断原始记录")
        storage = str(decoded_predictions.dtype)
        rows = decoded_predictions.detach().tolist()
    else:
        if not isinstance(decoded_predictions, (list, tuple)) or len(decoded_predictions) > 30000:
            raise ValueError("只接受有界逐行预测或CPU Nx6 Tensor")
        rows = deepcopy(decoded_predictions)
    raw = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError("每行必须明确xyxy/score/class六项")
        numbers = [_number(v) for v in row]
        x1, y1, x2, y2, score, category = numbers
        if (x2 <= x1 or y2 <= y1 or not math.isfinite((x2-x1)*(y2-y1))
                or not 0 <= score <= 1 or not category.is_integer() or not 0 <= category < 80):
            raise ValueError("预测框须正面积、score在[0,1]且class为COCO80整数")
        raw.append({"input_prediction_index": index, "decoded_row": numbers,
                    "input_bbox": numbers[:4], "score": score, "class_id": int(category),
                    "filter_reasons": []})
    tensor = torch.tensor([r["decoded_row"] for r in raw], dtype=torch.float32).reshape(-1, 6)
    if not torch.isfinite(tensor).all() or (tensor[:, 2:4] <= tensor[:, :2]).any():
        raise ValueError("FP32决策转换后非有限或框退化")
    selected = []
    for index, record in enumerate(raw):
        record["decision_score_float32"] = float(tensor[index, 4])
        record["decision_bbox_float32"] = tensor[index, :4].tolist()
        if record["class_id"] != 0:
            record["filter_reasons"].append("non_person")
        if not bool(tensor[index, 4] > .001):
            record["filter_reasons"].append("score_below_or_equal_conf")
        if not record["filter_reasons"]:
            selected.append(index)
    # 固定原输入顺序交给native，不加epsilon、不强制改同分赢家。
    kept_local = nms(tensor[selected, :4], tensor[selected, 4], .7).tolist() if selected else []
    kept = [selected[i] for i in kept_local]
    kept_set, capped = set(kept), kept[:300]
    final = []
    for index in selected:
        if index not in kept_set:
            raw[index]["filter_reasons"].append("nms_suppressed")
        elif index not in set(capped):
            raw[index]["filter_reasons"].append("max_det")
    for index in capped:
        record = raw[index]
        box = record["decision_bbox_float32"]
        unclipped = [(box[0]-padding[0])/ratio, (box[1]-padding[1])/ratio,
                     (box[2]-padding[0])/ratio, (box[3]-padding[1])/ratio]
        restored = [max(0., min(w, unclipped[0])), max(0., min(h, unclipped[1])),
                    max(0., min(w, unclipped[2])), max(0., min(h, unclipped[3]))]
        record["unclipped_original_bbox"] = unclipped
        record["clipped_original_bbox"] = restored
        if restored[2] <= restored[0] or restored[3] <= restored[1]:
            record["filter_reasons"].append("degenerate_after_restore_clip")
            continue
        record["retained"] = True
        final.append({"bbox": restored, "score": record["decision_score_float32"],
                      "class_id": 0, "input_prediction_index": index})
    scores = [float(tensor[i, 4]) for i in selected]
    return {"schema_version": "candidate_person_postprocess_v1", "uid": uid,
            "candidate_only": True, "official_benchmark": False, "geometry": g,
            "binding": dict(VERSIONS), "decoded_input_storage": storage,
            "raw_predictions": raw, "predictions": final, "nms_kept_input_indices": kept,
            "same_score_candidates_present": len(set(scores)) != len(scores),
            "input_prediction_count": len(raw), "final_prediction_count": len(final),
            "definition": deepcopy(DEFINITION)}


def _safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("评价目录须明确绝对路径且不含上跳")
    for parent in (path, *path.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)&0x400:
            raise ValueError("评价路径拒绝symlink/junction/reparse")
    return path


def write_candidate_evaluation(directory, processed_images, ground_truth, *, bindings, scope):
    """一次性新目录/单JSON写入；仅fixture/candidate_val，拒正式test及覆盖。

    GT须逐UID显式positive_gt/ignore；AP是候选给定框计算，不是数据集成绩声明。
    不读取checkpoint/GT/媒体，不推理；bindings是caller提供的来源说明，不替其认证。
    不提供跨进程恶意改路径防护或完整评价生命周期事务。
    """
    if scope not in ("fixture", "candidate_val") or not isinstance(bindings, dict) or not bindings:
        raise ValueError("须明确候选scope与非空caller bindings，禁止正式test")
    if bindings.get("split") == "test":
        raise ValueError("本候选接口不开放test")
    if not isinstance(processed_images, dict) or not isinstance(ground_truth, dict) or set(processed_images) != set(ground_truth):
        raise ValueError("processed/GT UID集合必须明确且一致")
    from .metrics import player_ignore_ap
    inputs = {}
    for uid in sorted(processed_images):
        result, gt = processed_images[uid], ground_truth[uid]
        if (not isinstance(result, dict) or result.get("schema_version") != "candidate_person_postprocess_v1"
                or result.get("uid") != uid or result.get("candidate_only") is not True
                or result.get("official_benchmark") is not False or result.get("definition") != DEFINITION):
            raise ValueError("后处理结果UID/候选定义绑定错误")
        _geometry(uid, result["geometry"])
        if not isinstance(gt, dict) or not {"positive_gt", "ignore"} <= gt.keys():
            raise ValueError("GT必须显式提供positive_gt及ignore，空也不可省略")
        inputs[uid] = {"positive_gt": gt["positive_gt"], "ignore": gt["ignore"],
                       "predictions": result["predictions"]}
    metrics = player_ignore_ap(inputs, image_uid_order=sorted(inputs))
    document = {"schema_version": "candidate_given_boxes_evaluation_v1",
                "candidate_only": True, "official_benchmark": False, "scope": scope,
                "bindings": deepcopy(bindings), "definition": deepcopy(DEFINITION),
                "processed_images": deepcopy(processed_images), "ground_truth": deepcopy(ground_truth),
                "metrics": metrics, "model_inference_performed": False}
    encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode("utf-8")
    if len(encoded) > 64*1024*1024:
        raise ValueError("候选单次JSON超过64MiB，拒绝无界聚合，不丢原始预测")
    target = _safe_path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("拒绝覆盖评价目录")
    if not target.parent.is_dir():
        raise FileNotFoundError("caller须已建立明确评价父目录")
    target.mkdir(exist_ok=False)
    _safe_path(target)
    output = target/"candidate_evaluation.json"
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _safe_path(output)
    return {"path": str(output), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
            "candidate_only": True, "official_benchmark": False}
