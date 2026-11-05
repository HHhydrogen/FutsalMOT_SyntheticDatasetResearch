"""精确预算与数值调度预计算；正式YOLO网络循环尚未实现。"""
from __future__ import annotations
import math

STRATEGIES = ("R", "S", "S->R", "R->R", "R+S")
MODEL_BACKEND_IMPLEMENTED = False

def budget_updates(n_images: int, epochs: int, batch: int = 16) -> int:
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (n_images, epochs, batch)):
        raise ValueError("图数、轮次与batch必须为正整数")
    return (n_images * epochs + batch - 1) // batch

def domain_at(strategy: str, step: int, u_s: int, u_r: int) -> tuple[str, int]:
    """step和每域更新ordinal均从1起；R_extra是独立的额外R流。"""
    if strategy not in STRATEGIES or min(u_s, u_r) < 1:
        raise ValueError("非法策略或预算")
    total = u_r if strategy == "R" else u_s if strategy == "S" else u_s + u_r
    if not 1 <= step <= total:
        raise ValueError("step越界")
    if strategy == "R": return "R", step
    if strategy == "S": return "S", step
    if strategy == "S->R": return ("S", step) if step <= u_s else ("R", step - u_s)
    if strategy == "R->R": return ("R_extra", step) if step <= u_s else ("R", step - u_s)
    r_now = step * u_r // total
    r_before = (step - 1) * u_r // total
    return ("R", r_now) if r_now > r_before else ("S", step - r_now)

def warmup_length(updates: int, *, final_stage: bool) -> int:
    if updates < 2: raise ValueError("本候选协议阶段至少2次更新")
    return min(100 if final_stage else 500, max(1, updates // 20), updates - 1)

def schedule_at(step: int, updates: int, *, final_stage: bool) -> dict:
    if not 0 <= step <= updates: raise ValueError("step越界")
    peak = 0.001 if final_stage else 0.01
    k = warmup_length(updates, final_stage=final_stage)
    if step <= k:
        lr = peak * step / k
        momentum = 0.8 + (0.937 - 0.8) * step / k
    else:
        lr = peak * (0.01 + 0.99 * (1 + math.cos(math.pi * (step-k) / (updates-k))) / 2)
        momentum = 0.937
    return {"step": step, "updates": updates, "warmup_updates": k, "lr": lr, "momentum": momentum, "ema_decay": 0.9999 * (1 - math.exp(-step / 2000))}

def update_spec(strategy: str, step: int, u_s: int, u_r: int) -> dict:
    domain, ordinal = domain_at(strategy, step, u_s, u_r)
    final = strategy == "R" or (strategy not in ("R", "S") and step > u_s)
    phase_step = step if strategy in ("R", "S") or step <= u_s else step-u_s
    phase_updates = u_r if final else u_s
    return {"global_step": step, "phase": "final" if final else "first", "domain": domain, "domain_update": ordinal, **schedule_at(phase_step, phase_updates, final_stage=final)}

def plan(n_s: int, n_r: int, *, epochs_s=20, epochs_r=50, batch=16) -> dict:
    u_s, u_r = budget_updates(n_s, epochs_s, batch), budget_updates(n_r, epochs_r, batch)
    if min(u_s, u_r) < 2: raise ValueError("阶段预算小于本协议最低2次更新")
    rows = {}
    for strategy in STRATEGIES:
        total = u_r if strategy == "R" else u_s if strategy == "S" else u_s+u_r
        r = u_r if strategy in ("R", "S->R", "R+S") else u_s+u_r if strategy == "R->R" else 0
        s = u_s if strategy in ("S", "S->R", "R+S") else 0
        preview_steps = sorted({1, min(8, total), total, max(1, total-7), *([u_s, u_s+1] if strategy not in ("R", "S") else [])})
        rows[strategy] = {"updates": total, "R_exposures": batch*r, "S_exposures": batch*s, "phase_reset_after_update": u_s if strategy not in ("R", "S") else None, "examples": [update_spec(strategy, j, u_s, u_r) for j in preview_steps]}
    return {"N_S": n_s, "N_R": n_r, "effective_batch": batch, "nominal_epochs_S": epochs_s, "nominal_epochs_R": epochs_r, "U_S": u_s, "U_R": u_r, "padding_S": batch*u_s-epochs_s*n_s, "padding_R": batch*u_r-epochs_r*n_r, "equivalent_epochs_S": batch*u_s/n_s, "equivalent_epochs_R": batch*u_r/n_r, "strategies": rows, "model_training_executed": False, "budget_is_candidate_not_calibrated_or_frozen": True}

def training_blockers(config: dict) -> list[str]:
    gate = config.get("gate", {})
    blockers = []
    for field, reason in [("protocol_frozen", "数值/作业协议未冻结"), ("data_frozen", "真实与合成数据绑定未冻结"), ("synthetic_complete", "合成全版未验收"), ("gpu_released", "GPU未确认由生产释放"), ("resource_precheck_passed", "GPU资源预检未通过"), ("numeric_precheck_passed", "数值与累积loss预检未通过"), ("training_authorized", "本次真实训练未授权")]:
        if gate.get(field) is not True: blockers.append(reason)
    if not config.get("numeric", {}).get("mode_frozen"):
        blockers.append("FP32/AMP策略尚未冻结")
    blockers.append("正式YOLO标签/ignore/pose/seg适配器与网络训练循环尚未实现")
    return blockers

def train(config: dict) -> dict:
    return {"status": "BLOCKED", "command": "train", "blockers": training_blockers(config), "model_loaded": False, "GPU_used": False, "updates": 0}


def _torch():
    """只有显式数值primitive才导入Torch；inspect/dry-run不受影响。"""
    import torch
    return torch


def prepare_detection_batch(batch, *, device, non_blocking=False):
    """collate的CPU RGB uint8批次→指定设备FP32/255，输出不共享输入可写存储。

    不是DataLoader/Trainer，不赋予CUDA资源许可。CUDA未初始化时明确拒绝，
    不调用is_available、初始化或设备探测。metadata保留，目标/ignore不重新归一化。
    """
    from copy import deepcopy
    from collections.abc import Mapping
    torch = _torch()
    if not isinstance(batch, Mapping) or type(non_blocking) is not bool:
        raise ValueError("需要collate映射及明确non_blocking bool")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("只支持显式cpu或已初始化cuda设备")
    if target.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("CUDA尚未初始化；本准备函数禁止探测或自动初始化")
    img = batch.get("img")
    if (not isinstance(img, torch.Tensor) or img.device.type != "cpu" or img.dtype != torch.uint8
            or img.ndim != 4 or tuple(img.shape[1:]) != (3, 960, 960) or img.shape[0] < 1
            or not img.is_contiguous()
            or batch.get("numeric_input") != "uint8_RGB_CHW_not_yet_divided_by_255"):
        raise ValueError("只接受collate的CPU连续RGB NCHW uint8，拒绝重复/255或float输入")
    size = img.shape[0]
    for key in ("uids", "domains", "geometry"):
        if not isinstance(batch.get(key), (list, tuple)) or len(batch[key]) != size:
            raise ValueError("批次UID/域/geometry行数不一致")
    if any(not isinstance(uid, str) or not uid for uid in batch["uids"]):
        raise ValueError("缺样本UID")
    if any(domain not in ("R", "S") for domain in batch["domains"]):
        raise ValueError("非法R/S域")
    if any(g.get("channel_order") != "RGB" or tuple(g.get("input_shape", ())) != (960, 960)
           for g in batch["geometry"]):
        raise ValueError("geometry必须明确RGB及960输入")
    for key, columns in (("cls", 1), ("bboxes", 4), ("ignore_bboxes", 4)):
        value = batch.get(key)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != columns
                or not torch.isfinite(value).all()):
            raise ValueError(f"{key}须为CPU有限float32 Nx{columns}")
    n, ni = len(batch["bboxes"]), len(batch["ignore_bboxes"])
    if len(batch["cls"]) != n or (batch["cls"] != 0).any():
        raise ValueError("目标类别/框不匹配或非person0")
    for key, count in (("batch_idx", n), ("ignore_batch_idx", ni)):
        value = batch.get(key)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.int64 or value.ndim != 1 or len(value) != count
                or ((value < 0) | (value >= size)).any()):
            raise ValueError(f"{key}须为合法CPU int64图索引")
    boxes, ignores = batch["bboxes"], batch["ignore_bboxes"]
    if ((boxes[:, :2] < 0) | (boxes[:, :2] > 1)).any() or ((boxes[:, 2:] <= 0) | (boxes[:, 2:] > 1)).any():
        raise ValueError("目标归一化cxcywh非法")
    corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
    if (corners < -1e-7).any() or (corners > 1 + 1e-7).any():
        raise ValueError("目标框越出输入")
    if ((ignores[:, :2] < 0) | (ignores[:, 2:] > 960) | (ignores[:, 2:] <= ignores[:, :2])).any():
        raise ValueError("ignore须为input像素正面积半开XYXY")
    numeric_keys = {"img", "cls", "bboxes", "ignore_bboxes", "batch_idx", "ignore_batch_idx"}
    result = {key: deepcopy(value) for key, value in batch.items() if key not in numeric_keys}
    # 顺序明确：uint8先传输，再float32，再且只做一次/255。
    result["img"] = img.to(device=target, non_blocking=non_blocking).to(dtype=torch.float32).div(255)
    for key in ("cls", "bboxes", "ignore_bboxes"):
        result[key] = batch[key].to(device=target, dtype=torch.float32, non_blocking=non_blocking, copy=True)
    for key in ("batch_idx", "ignore_batch_idx"):
        result[key] = batch[key].to(device=target, dtype=torch.int64, non_blocking=non_blocking, copy=True)
    result["numeric_input"] = "float32_RGB_NCHW_divided_by_255_once"
    return result


def criterion_backward_loss(criterion_output):
    """8.3.237检测criterion三项向量→sum/16；不额外乘microbatch或除累积次数。

    返回loss保留梯度；components及native_loss_items为可记录的detached副本。
    仅构造反传标量，不执行backward、优化器step或参数更新。
    """
    torch = _torch()
    if not isinstance(criterion_output, (tuple, list)) or len(criterion_output) != 2:
        raise ValueError("检测criterion必须返回两个三项向量")
    first, items = criterion_output
    for value in (first, items):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (3,)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("检测criterion各返回值须为有限浮点shape[3]")
    if first.device != items.device or first.dtype != items.dtype or items.requires_grad:
        raise ValueError("criterion设备/dtype不一致或第二向量未detach")
    scalar = first.sum() / 16
    if not torch.isfinite(scalar):
        raise ValueError("criterion sum/16非有限")
    return {"loss": scalar,
            "components": {name: (first[index].detach() / 16).clone()
                           for index, name in enumerate(("box", "cls", "dfl"))},
            "native_loss_items": items.detach().clone(),
            "effective_batch": 16, "normalization": "sum(returned_microbatch_scaled_vector)/16"}


def _validate_ema_raw(raw, torch):
    from collections.abc import Mapping
    if not isinstance(raw, Mapping) or not raw or any(not isinstance(k, str) or not k for k in raw):
        raise ValueError("需要非空完整Tensor state_dict及字符串key")
    integer_types = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool}
    for key, tensor in raw.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("EMA只支持完整普通Tensor state_dict: " + key)
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError("EMA不支持meta或其他设备")
        if tensor.device.type == "cuda" and not torch.cuda.is_initialized():
            raise RuntimeError("EMA不自动初始化CUDA")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("raw/EMA浮点参数及buffer须有限FP32，禁止静默half/转换: " + key)
        elif tensor.dtype not in integer_types:
            raise ValueError("EMA非浮点buffer仅支持明确整数/bool dtype: " + key)


def initialize_ema_stage(raw_state):
    """阶段0从完整raw state_dict独立复制，计数归零；不修改或持有其存储引用。"""
    torch = _torch()
    _validate_ema_raw(raw_state, torch)
    with torch.no_grad():
        return {"schema_version": "futsalmot_ema_state_v1", "updates": 0,
                "state_dict": {key: value.detach().clone() for key, value in raw_state.items()}}


def update_ema_state(ema_state, raw_state, *, update):
    """功能式EMA更新：返回全新状态，不原地修改raw或旧EMA；必须逐更新递增。

    raw/EMA全部浮点Tensor严格FP32；全部整数/bool buffer从当前raw复制。
    调用方必须提供完整state_dict（含buffer）并保证是一次已成功optimizer更新的
    同一快照；本helper不运行模型、优化器或提交训练记录。
    """
    torch = _torch()
    if (not isinstance(ema_state, dict)
            or set(ema_state) != {"schema_version", "updates", "state_dict"}
            or ema_state["schema_version"] != "futsalmot_ema_state_v1"
            or type(ema_state["updates"]) is not int or ema_state["updates"] < 0
            or type(update) is not int or update != ema_state["updates"] + 1):
        raise ValueError("EMA schema/阶段更新边界非法，必须从0顺序递增")
    old = ema_state["state_dict"]
    _validate_ema_raw(raw_state, torch)
    _validate_ema_raw(old, torch)
    if set(old) != set(raw_state):
        raise ValueError("EMA与raw keys不一致，禁止漏buffer")
    for key in raw_state:
        if (old[key].shape != raw_state[key].shape or old[key].dtype != raw_state[key].dtype
                or old[key].device != raw_state[key].device):
            raise ValueError("EMA与raw shape/dtype/device不一致: " + key)
    decay = 0.9999 * (1 - math.exp(-update / 2000))
    result = {}
    with torch.no_grad():
        for key, raw in raw_state.items():
            result[key] = (old[key].detach().mul(decay).add(raw.detach(), alpha=1 - decay)
                           if raw.is_floating_point() else raw.detach().clone())
    return {"schema_version": "futsalmot_ema_state_v1", "updates": update, "state_dict": result}
