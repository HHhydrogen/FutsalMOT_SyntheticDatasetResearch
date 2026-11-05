"""显式SGD与逐更新组调度候选helper；不加载模型、自动冻结、step或初始化GPU。

分类参考锁定Ultralytics8.3.237 Trainer.build_optimizer：
全文参数名包含bias优先无decay；nn中名字包含Norm的模块参数无decay；其它weight
显式decay。上游另有logit_scale无decay例外，本检测候选尚未定义其归属，遇到时
明确拒绝，不偷偷扩展为对比模型协议。共享/重复可训练Parameter身份也明确拒绝。
"""
from __future__ import annotations
import math

_BINDING = "futsalmot_explicit_sgd_v1"


def _torch():
    import torch
    return torch


def _scalar(value, name, *, positive=False, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name + "须为合法有限显式标量")
    if upper is not None and value >= upper:
        raise ValueError(name + "超允许范围")
    return float(value)


def _parameter(parameter, torch, *, check_values=False):
    if isinstance(parameter, torch.nn.parameter.UninitializedParameter):
        raise ValueError("未初始化的lazy参数不可进入本候选SGD")
    if not isinstance(parameter, torch.nn.Parameter) or parameter.layout != torch.strided:
        raise ValueError("只支持普通Parameter")
    if parameter.device.type not in ("cpu", "cuda"):
        raise ValueError("只支持CPU或已初始化CUDA参数")
    if parameter.device.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("本helper不探测或初始化CUDA")
    if parameter.dtype != torch.float32:
        raise ValueError("全部可训练浮点参数必须FP32，不静默转换")
    if check_values and not torch.isfinite(parameter).all():
        raise ValueError("参数含非有限值")


def make_sgd_optimizer(caller_model, *, lr, momentum, weight_decay):
    """仅为caller已提供模块创建零state SGD；不构造真实模型、不改变requires_grad。

    全部可训练参数必须唯一覆盖；requires_grad=False的DFL等保持原样且排除。
    保留bias/weight/norm三个具名组（允许空组），所有组同LR/momentum；
    无auto optimizer、nbs缩放或额外bias warmup。构造器不执行step。
    """
    torch = _torch()
    if not isinstance(caller_model, torch.nn.Module):
        raise ValueError("需要调用方已提供的nn.Module")
    lr = _scalar(lr, "lr")
    momentum = _scalar(momentum, "momentum", positive=True, upper=1)
    weight_decay = _scalar(weight_decay, "weight_decay")
    norm_types = tuple(value for name, value in vars(torch.nn).items()
                       if "Norm" in name and isinstance(value, type) and issubclass(value, torch.nn.Module))
    groups = {name: {"name": name, "params": [], "parameter_names": [],
                     "weight_decay": weight_decay if name == "weight" else 0.0,
                     "lr": lr, "momentum": momentum}
              for name in ("bias", "weight", "norm")}
    seen = set()
    for fullname, parameter in caller_model.named_parameters(recurse=True, remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("共享/重复可训练Parameter身份不能无歧义分组: " + fullname)
        seen.add(id(parameter))
        _parameter(parameter, torch, check_values=True)
        if "logit_scale" in fullname:
            raise ValueError("上游logit_scale例外尚未由本检测候选定义，不自动扩展")
        module_name, _, _ = fullname.rpartition(".")
        module = caller_model.get_submodule(module_name) if module_name else caller_model
        kind = "bias" if "bias" in fullname else "norm" if isinstance(module, norm_types) else "weight"
        groups[kind]["params"].append(parameter)
        groups[kind]["parameter_names"].append(fullname)
    expected = {id(p) for p in caller_model.parameters() if p.requires_grad}
    if not seen or seen != expected:
        raise ValueError("可训练参数为空或完整覆盖失败")
    optimizer = torch.optim.SGD(list(groups.values()), lr=lr, momentum=momentum,
                                weight_decay=0.0, dampening=0.0, nesterov=True, maximize=False)
    optimizer._futsalmot_sgd_binding = {
        "schema": _BINDING,
        "groups": tuple((g["name"], tuple(id(p) for p in g["params"]),
                         tuple(g["parameter_names"]), g["weight_decay"]) for g in optimizer.param_groups),
    }
    return optimizer


def _check_optimizer(optimizer, torch):
    if type(optimizer) is not torch.optim.SGD:
        raise ValueError("只接受本helper建立的精确SGD实例")
    binding = getattr(optimizer, "_futsalmot_sgd_binding", None)
    if not isinstance(binding, dict) or binding.get("schema") != _BINDING:
        raise ValueError("SGD缺本helper参数组绑定")
    expected = binding.get("groups")
    if not isinstance(expected, tuple) or len(expected) != 3 or len(optimizer.param_groups) != 3:
        raise ValueError("SGD组集合变化")
    seen = set()
    for group, (name, ids, names, decay) in zip(optimizer.param_groups, expected):
        if (group.get("name") != name or tuple(id(p) for p in group["params"]) != ids
                or tuple(group.get("parameter_names", ())) != names or group["weight_decay"] != decay
                or group.get("nesterov") is not True or group.get("dampening") != 0
                or group.get("maximize") is not False):
            raise ValueError("SGD参数组身份/decay/Nesterov绑定变化")
        _scalar(group["lr"], "existing lr")
        _scalar(group["momentum"], "existing momentum", positive=True, upper=1)
        for parameter in group["params"]:
            if id(parameter) in seen or not parameter.requires_grad:
                raise ValueError("SGD组重复参数或冻结状态发生变化")
            seen.add(id(parameter))
            _parameter(parameter, torch)


def apply_sgd_schedule(optimizer, schedule):
    """caller显式传schedule_at结果，同步所有组LR/momentum；不自动step/推进计数。

    这里只验证有界step及数值和原组绑定，不声称该schedule已由正式协议/资源Gate
    审批，也不证明一次optimizer更新成功。无隐式高bias LR或weight_decay重缩放。
    """
    torch = _torch()
    _check_optimizer(optimizer, torch)
    if not isinstance(schedule, dict) or not {"step", "updates", "lr", "momentum"} <= schedule.keys():
        raise ValueError("schedule缺step/updates/lr/momentum")
    step, updates = schedule["step"], schedule["updates"]
    if type(step) is not int or type(updates) is not int or updates < 2 or not 0 <= step <= updates:
        raise ValueError("schedule更新边界非法")
    lr = _scalar(schedule["lr"], "schedule lr")
    momentum = _scalar(schedule["momentum"], "schedule momentum", positive=True, upper=1)
    # 所有验证完成后才一起修改组标量；参数值/梯度和optimizer state不动。
    for group in optimizer.param_groups:
        group["lr"], group["momentum"] = lr, momentum
    return {"step": step, "updates": updates,
            "groups": [{"name": group["name"], "lr": group["lr"], "momentum": group["momentum"],
                        "weight_decay": group["weight_decay"], "parameter_tensors": len(group["params"])}
                       for group in optimizer.param_groups]}
