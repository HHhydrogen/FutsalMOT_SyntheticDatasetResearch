"""Ultralytics 8.3.237检测ignore适配；仅criterion，不含训练/评价循环。

复用Ultralytics AGPL-3.0实现与TAL/BboxLoss，不复制完整loss分支：
https://github.com/ultralytics/ultralytics/blob/v8.3.237/ultralytics/utils/loss.py
正目标仍由原生TAL分配；本地只在明确ignore框内屏蔽未匹配anchor的全部类BCE。
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import threading

import torch
from ultralytics.utils import loss as _upstream_loss
from ultralytics.utils import tal as _upstream_tal
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

UPSTREAM_BINDING = {
    "version": "8.3.237",
    "loss.py": "475b82d342255d51c559cf08f27823b5f45fb948ffb13fd9c39f4e9582b09856",
    "tal.py": "c3ba22e9fa743662fa84787ecaf0825f5701bae07826e5fbc35c22478fd2319b",
}


def validate_upstream_binding() -> dict:
    """对安装的精确版本/源码字节绑定，不修改site-packages。"""
    if version("ultralytics") != UPSTREAM_BINDING["version"]:
        raise RuntimeError("ignore criterion仅核对过Ultralytics 8.3.237")
    for name, module in (("loss.py", _upstream_loss), ("tal.py", _upstream_tal)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM_BINDING[name]:
            raise RuntimeError(f"Ultralytics {name}源码与核对SHA不符")
    return dict(UPSTREAM_BINDING)


def ignored_anchor_mask(anchor_centers: torch.Tensor, *, input_hw: tuple[int, int],
                        batch_size: int, ignore_bboxes: torch.Tensor,
                        ignore_batch_idx: torch.Tensor) -> torch.Tensor:
    """框为变换后input像素半开XYXY；返回B×A bool，显式空Tensor同样校验。

    不推测缺失的裁判框，不按bbox IoU/人物中心替代anchor中心；不修改原张量。
    """
    if not isinstance(anchor_centers, torch.Tensor) or anchor_centers.ndim != 2 or anchor_centers.shape[1] != 2 or not anchor_centers.is_floating_point() or not torch.isfinite(anchor_centers).all():
        raise ValueError("anchor中心必须为有限A×2浮点Tensor")
    if type(batch_size) is not int or batch_size < 1 or len(input_hw) != 2 or any(type(v) is not int or v <= 0 for v in input_hw):
        raise ValueError("batch_size和input高宽必须为正整数")
    if not isinstance(ignore_bboxes, torch.Tensor) or ignore_bboxes.ndim != 2 or ignore_bboxes.shape[1] != 4:
        raise ValueError("ignore_bboxes必须显式为M×4 Tensor，空时shape=(0,4)")
    if not isinstance(ignore_batch_idx, torch.Tensor) or ignore_batch_idx.ndim != 1 or ignore_batch_idx.shape[0] != ignore_bboxes.shape[0]:
        raise ValueError("ignore_batch_idx必须显式为M维Tensor")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if ignore_batch_idx.dtype not in integer_types:
        raise ValueError("ignore_batch_idx必须为整数dtype，不能截断浮点索引")
    if ignore_bboxes.dtype not in integer_types and not ignore_bboxes.is_floating_point():
        raise ValueError("ignore框必须为实数坐标")
    if ignore_bboxes.device != anchor_centers.device or ignore_batch_idx.device != anchor_centers.device:
        raise ValueError("ignore与anchor须已显式传输至同一设备")
    if not torch.isfinite(ignore_bboxes).all() or ((ignore_batch_idx < 0) | (ignore_batch_idx >= batch_size)).any():
        raise ValueError("ignore包含非有限坐标或越界batch索引")
    boxes = ignore_bboxes.to(dtype=torch.float32)
    h, w = input_hw
    if ((boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > w) | (boxes[:, 3] > h)
        | (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
        raise ValueError("ignore必须是裁边后图内的正面积半开XYXY")
    result = torch.zeros((batch_size, len(anchor_centers)), dtype=torch.bool, device=anchor_centers.device)
    # 每框只分配A元素临时向量，避免M×A×2广播峰值；CPU/吞吐优化另做准入。
    centers = anchor_centers.to(dtype=torch.float32)
    for box, image_idx in zip(boxes, ignore_batch_idx):
        inside = ((centers[:, 0] >= box[0]) & (centers[:, 0] < box[2])
                  & (centers[:, 1] >= box[1]) & (centers[:, 1] < box[3]))
        result[image_idx.long()] |= inside
    return result


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    """复用原生forward，只替换当前调用的BCE元素，完整保留TAL/box/DFL。

    batch额外必需ignore_bboxes(M,4)、ignore_batch_idx(M,)，包括显式空张量。
    返回与8.3.237完全同型：(乘microbatch的3项loss向量, detached3项向量)。
    未来loop必须先sum，再按有效batch16缩放，不能当作已归一化标量。
    本criterion不接入Trainer、DDP/compile、Pose/Seg或ignore-aware评价。
    同一实例禁止并发/重入；原生assigner需保留nn.Module hook接口。
    """
    def __init__(self, model, tal_topk: int = 10):
        self.upstream_binding = validate_upstream_binding()
        super().__init__(model, tal_topk=tal_topk)
        if self.nc != 80 or (self.hyp.box, self.hyp.cls, self.hyp.dfl) != (7.5, .5, 1.5):
            raise ValueError("本协议固定COCO80类及box7.5/cls0.5/dfl1.5")
        self._call_lock = threading.Lock()

    def __call__(self, preds, batch):
        missing = {"ignore_bboxes", "ignore_batch_idx"} - batch.keys()
        if missing:
            raise ValueError("缺少显式ignore字段: " + ",".join(sorted(missing)))
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("同一ignore criterion禁止并发/重入调用")
        handle, original_bce = None, self.bce
        try:
            feats = preds[1] if isinstance(preds, tuple) else preds
            if not isinstance(feats, list) or not feats or any(f.ndim != 4 or f.shape[0] != feats[0].shape[0] for f in feats):
                raise ValueError("需要原生detect训练输出的非空4维特征list")
            points, strides = make_anchors(feats, self.stride, .5)
            input_hw = (int(feats[0].shape[2] * self.stride[0]), int(feats[0].shape[3] * self.stride[0]))
            ignored = ignored_anchor_mask(points * strides, input_hw=input_hw,
                                           batch_size=feats[0].shape[0],
                                           ignore_bboxes=batch["ignore_bboxes"],
                                           ignore_batch_idx=batch["ignore_batch_idx"])
            if batch["ignore_bboxes"].shape[0] == 0:
                return super().__call__(preds, batch)
            captured = {}

            def assignment_hook(module, inputs, output):
                fg = output[3]
                if fg.shape != ignored.shape:
                    raise RuntimeError("上游fg_mask形状变化，拒绝猜测ignore对齐")
                captured["fg_mask"] = fg.bool()

            def masked_bce(scores, targets):
                if "fg_mask" not in captured:
                    raise RuntimeError("没有本次原生TAL分配结果，拒绝静默忽略")
                keep = ~ignored | captured["fg_mask"]
                if scores.shape[:2] != keep.shape or scores.shape[-1] != 80:
                    raise RuntimeError("BCE与anchor/COCO80维度不匹配")
                values = original_bce(scores, targets)
                return torch.where(keep.unsqueeze(-1), values, torch.zeros_like(values))

            if not isinstance(self.assigner, torch.nn.Module):
                raise RuntimeError("assigner不支持已核对的原生forward hook")
            handle = self.assigner.register_forward_hook(assignment_hook)
            self.bce = masked_bce
            # ignore不送入正目标预处理/assigner，target_scores_sum完全由原生实现计算。
            return super().__call__(preds, batch)
        finally:
            self.bce = original_bce
            if handle is not None:
                handle.remove()
            self._call_lock.release()
