"""长预算首阶段→终阶段的有界候选边界准备，不是完整生命周期提交。

caller须先停止prefetch并建立外层barrier。本helper只读live sampler的分离锁快照，
无跨模型/optimizer/EMA/多sampler原子锁；检查不证明实际optimizer更新已经发生。
不保存checkpoint、不记录/commit/cancel，不修改raw/旧stage/采样流或累计曝光。
"""
from __future__ import annotations
from copy import deepcopy
from .training import schedule_at, initialize_ema_stage, _validate_ema_raw
from .optimizers import make_sgd_optimizer, _check_optimizer
from .data import PrefetchSampler


def _positive(value, name):
    if type(value) is not int or value < 2:
        raise ValueError(name + "须为既定协议至少2次更新")
    return value


def _boundary_counts(strategy, u_s, u_r):
    if strategy == "S->R":
        return {"R": 0, "S": 16*u_s, "R_extra": 0}
    if strategy == "R->R":
        return {"R": 0, "S": 0, "R_extra": 16*u_s}
    if strategy == "R+S":
        r = u_s*u_r//(u_s+u_r)
        return {"R": 16*r, "S": 16*(u_s-r), "R_extra": 0}
    raise ValueError("只有S->R/R->R/R+S长策略允许重置；R/S单阶段不重置")


def _sampler_snapshot(samplers, expected, strategy):
    required = {"R", "R_extra"} if strategy == "R->R" else {"R", "S"}
    if not isinstance(samplers, dict) or not required <= samplers.keys() or set(samplers)-{"R", "S", "R_extra"}:
        raise ValueError("缺实际阶段所需live采样流或存在未知流")
    snapshots = {}
    seeds = set()
    for stream, sampler in samplers.items():
        if type(sampler) is not PrefetchSampler:
            raise ValueError("阶段barrier须接实际PrefetchSampler，不接受手填游标/队列计数")
        if sampler.pending_batches != 0:
            raise ValueError("阶段边界仍有pending reservation，caller须先停止prefetch")
        state = sampler.state_dict()
        current = state["sampler"]
        if (state["schema_version"] != PrefetchSampler.SCHEMA or current["domain"] != stream
                or current["task"] != "detect" or current["seed"] not in (17,29,43)
                or type(current["draws"]) is not int or current["draws"] != expected[stream]):
            raise ValueError("采样流身份/seed或首阶段实际提交曝光不符")
        seeds.add(current["seed"])
        snapshots[stream] = state
    if len(seeds) != 1:
        raise ValueError("同run采样流训练seed不一致")
    if "R_extra" in snapshots and snapshots["R_extra"]["sampler"]["pool_sha256"] != snapshots["R"]["sampler"]["pool_sha256"]:
        raise ValueError("R_extra与主R必须绑定同一真实UID池")
    return snapshots


def start_final_phase(raw_model, *, strategy, u_s, u_r, committed_updates, current_phase,
                      phase_updates, samplers, cumulative_exposures, current_optimizer, current_ema):
    """只在global==u_s、首阶段全提交且live队列为空时创建独立终阶段状态。

    新SGD使用schedule_at(0,u_r,final_stage=True)：LR=0、momentum=.8，
    decay固定.0005（bias/norm仍0），新state为空；EMA从完整当前raw重置0。
    optimizer仍指向同一raw参数，独立的是optimizer/EMA状态，不复制或续用旧momentum。
    原sampler流原样继续，包括R/S/R_extra namespace与已提交RNG/游标。
    """
    u_s, u_r = _positive(u_s,"u_s"), _positive(u_r,"u_r")
    expected = _boundary_counts(strategy,u_s,u_r)
    if (type(committed_updates) is not int or committed_updates != u_s or current_phase != "first"
            or type(phase_updates) is not int or phase_updates != u_s):
        raise ValueError("早/晚/重复阶段边界或first阶段提交计数不完整")
    actual = {"R": expected["R"]+expected["R_extra"], "S": expected["S"]}
    if (not isinstance(cumulative_exposures,dict) or set(cumulative_exposures) != {"R","S"}
            or any(type(v) is not int or v < 0 for v in cumulative_exposures.values())
            or cumulative_exposures != actual or sum(actual.values()) != 16*committed_updates):
        raise ValueError("实际累计R/S曝光与首阶段冻结策略不符")
    snapshots = _sampler_snapshot(samplers,expected,strategy)
    import torch
    if not isinstance(raw_model,torch.nn.Module):
        raise ValueError("需要caller提供的raw CPU nn.Module")
    raw = raw_model.state_dict()
    if any(not isinstance(v,torch.Tensor) or v.device.type != "cpu" for v in raw.values()):
        raise ValueError("本阶段准备仅接完整CPU raw状态，不迁移或初始化CUDA")
    _validate_ema_raw(raw,torch)
    _check_optimizer(current_optimizer,torch)
    optimizer_ids = {id(p) for g in current_optimizer.param_groups for p in g["params"]}
    if optimizer_ids != {id(p) for p in raw_model.parameters() if p.requires_grad}:
        raise ValueError("旧SGD不绑定当前raw可训练参数")
    if any(g["weight_decay"] != (.0005 if g["name"]=="weight" else 0.)
           for g in current_optimizer.param_groups):
        raise ValueError("旧SGD decay与既定.0005/0规则不符")
    if (type(current_ema) is not dict or set(current_ema) != {"schema_version","updates","state_dict"}
            or current_ema["schema_version"] != "futsalmot_ema_state_v1"
            or type(current_ema["updates"]) is not int or current_ema["updates"] != u_s):
        raise ValueError("旧EMA阶段成功更新数与首阶段边界不符")
    old_ema = current_ema["state_dict"]
    _validate_ema_raw(old_ema,torch)
    if set(old_ema) != set(raw) or any(old_ema[k].shape != raw[k].shape
            or old_ema[k].dtype != raw[k].dtype or old_ema[k].device != raw[k].device for k in raw):
        raise ValueError("旧EMA与完整raw keys/shape/dtype/device不一致")
    initial = schedule_at(0,u_r,final_stage=True)
    optimizer = make_sgd_optimizer(raw_model,lr=initial["lr"],momentum=initial["momentum"],weight_decay=.0005)
    ema = initialize_ema_stage(raw)
    if len(optimizer.state) != 0:
        raise ValueError("新阶段SGD必须零state，不继承momentum")
    # 只能发现观测窗口内变化，外层caller仍负责barrier及生命周期原子性。
    for stream,sampler in samplers.items():
        if sampler.pending_batches != 0 or sampler.state_dict() != snapshots[stream]:
            raise ValueError("准备期间sampler变化；外层barrier未建立")
    return {"optimizer":optimizer,"ema":ema,"metadata":{
        "schema_version":"candidate_final_phase_v1","candidate_only":True,"execution_authorized":False,
        "strategy":strategy,"u_s":u_s,"u_r":u_r,"committed_updates":committed_updates,
        "phase":"final","phase_updates":0,"previous_phase":"first",
        "cumulative_exposures":deepcopy(cumulative_exposures),
        "committed_samplers":deepcopy(snapshots),"first_stage_stream_exposures":expected,
        "initial_schedule":initial,"new_stage_checkpoint_required":True,
        "samplers_reset":False,"lifecycle_committed":False}}
