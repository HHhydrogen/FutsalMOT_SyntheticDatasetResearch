"""冻结检测候选矩阵的纯数学资源预览；不测硬件、不读文件、不授权执行。"""
from __future__ import annotations

from collections.abc import Mapping

from .training import STRATEGIES, budget_updates, plan

MODEL_NAMES = tuple("yolo11" + size for size in ("s", "n", "m", "l", "x"))
TRAINING_SEEDS = (17, 29, 43)
CHECKPOINT_INTERVAL = 50


def checkpoint_count(updates: int) -> int:
    """每阶段独立保存0、每50更新、末步；末步与50倍数重合只计一份。"""
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("阶段更新数必须为正整数")
    return 1 + updates // CHECKPOINT_INTERVAL + int(updates % CHECKPOINT_INTERVAL != 0)


def _stage(phase, updates, r_updates, s_updates, *, starts_after_update=0):
    return {"phase": phase, "updates": updates, "checkpoint_count": checkpoint_count(updates),
            "R_exposures": 16 * r_updates, "S_exposures": 16 * s_updates,
            "global_start_update": starts_after_update,
            "global_final_update": starts_after_update + updates,
            "save_initial_state": True, "save_final_state": True,
            "initial_separate_from_previous_final": phase == "final" and starts_after_update > 0}


def _stages(strategy, u_s, u_r):
    if strategy == "R":
        return [_stage("final", u_r, u_r, 0)]
    if strategy == "S":
        return [_stage("first", u_s, 0, u_s)]
    if strategy == "S->R":
        first_r, first_s, last_r, last_s = 0, u_s, u_r, 0
    elif strategy == "R->R":
        first_r, first_s, last_r, last_s = u_s, 0, u_r, 0
    elif strategy == "R+S":
        # 与training.domain_at相同的精确interleave前缀计数，不创建逐步大清单。
        first_r = u_s * u_r // (u_s + u_r)
        first_s = u_s - first_r
        last_r, last_s = u_r - first_r, u_s - first_s
    else:
        raise ValueError("未知既定策略")
    return [_stage("first", u_s, first_r, first_s),
            _stage("final", u_r, last_r, last_s, starts_after_update=u_s)]


def _measurements(values):
    if values is None:
        values = {}
    if not isinstance(values, Mapping) or any(name not in MODEL_NAMES for name in values):
        raise ValueError("实测大小必须以既定yolo11n/s/m/l/x型号为键")
    result = {}
    for name in MODEL_NAMES:
        size = values.get(name)
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise ValueError("完整状态实测峰值字节必须是正整数或None，不能用模型路径/文件代替")
        result[name] = size
    return result


def _job(kind, model, strategy, seed, stages, expected, measurements, *, epochs_r):
    updates = sum(s["updates"] for s in stages)
    r = sum(s["R_exposures"] for s in stages)
    synth = sum(s["S_exposures"] for s in stages)
    if (updates, r, synth) != (expected["updates"], expected["R_exposures"], expected["S_exposures"]):
        raise ValueError("阶段汇总与现training.plan不一致")
    count = sum(s["checkpoint_count"] for s in stages)
    size = measurements[model]
    code = strategy.replace("->", "_to_").replace("+", "_plus_")
    label = f"R{epochs_r}" if kind == "calibration" else code
    return {"candidate_job_id": f"{kind}_detect_{model}_{label}_seed{seed}",
            "kind": kind, "task": "detect", "model": model, "strategy": strategy,
            "training_seed": seed, "epochs_r": epochs_r, "stages": stages,
            "total_updates": updates, "R_exposures": r, "S_exposures": synth,
            "complete_checkpoint_count": count,
            "complete_state_peak_bytes": size,
            "measurement_status": "CALLER_MEASURED" if size is not None else "UNMEASURED",
            "checkpoint_only_bytes": count * size if size is not None else None}


def _totals(jobs):
    by_model = {}
    for name in MODEL_NAMES:
        group = [job for job in jobs if job["model"] == name]
        if not group:
            continue
        known = all(job["checkpoint_only_bytes"] is not None for job in group)
        by_model[name] = {
            "jobs": len(group),
            "total_updates": sum(job["total_updates"] for job in group),
            "complete_checkpoint_count": sum(job["complete_checkpoint_count"] for job in group),
            "checkpoint_only_bytes": sum(job["checkpoint_only_bytes"] for job in group) if known else None,
            "measurement_status": "CALLER_MEASURED" if known else "UNMEASURED",
        }
    all_known = all(job["checkpoint_only_bytes"] is not None for job in jobs)
    return {
        "jobs": len(jobs), "total_updates": sum(j["total_updates"] for j in jobs),
        "R_exposures": sum(j["R_exposures"] for j in jobs),
        "S_exposures": sum(j["S_exposures"] for j in jobs),
        "complete_checkpoint_count": sum(j["complete_checkpoint_count"] for j in jobs),
        "checkpoint_only_bytes": sum(j["checkpoint_only_bytes"] for j in jobs) if all_known else None,
        "measurement_status": "CALLER_MEASURED" if all_known else "UNMEASURED",
        "unmeasured_models": [name for name, row in by_model.items()
                              if row["measurement_status"] == "UNMEASURED"],
        "by_model": by_model,
    }


def candidate_resource_plan(n_s: int, n_r: int, *, epochs_r=50,
                            complete_state_peak_bytes=None) -> dict:
    """枚举一个来源面板的39次检测候选及两次独立R预算校准，始终不可执行。

    N_S/N_R没有实际量默认值，须显式传入正整数。复用training.plan的S20、
    有效batch16；R候选只允许冻结的50或100轮。两次校准固定s/seed17，分别
    独立R50与R100，不承接权重或混入正式三seed结果。

    complete_state_peak_bytes形如{"yolo11s": 123456, ...}；调用方提供的值
    必须来自完整raw+EMA+optimizer+RNG/sampler等状态的实测峰值，而不是公开
    .pt或纯模型state大小。本函数不验证测量来源，不读取任何路径或测模型。
    缺测量保持UNMEASURED/None，不代入0。

    峰值字节乘保留份数只是统一的checkpoint保留基项估计，不声称每个文件等大
    或未来状态不会更大。完整资源预算还需日志/预测/诊断及失败记录、序列化与
    异步writer临时峰值、时间、磁盘余量等；本输出不构成资源PASS或训练授权。
    """
    if isinstance(epochs_r, bool) or not isinstance(epochs_r, int) or epochs_r not in (50, 100):
        raise ValueError("本冻结候选仅允许R50或R100，不另扩预算")
    base = plan(n_s, n_r, epochs_r=epochs_r)
    measurements = _measurements(complete_state_peak_bytes)
    u_s, u_r = base["U_S"], base["U_R"]
    formal = []
    for model in MODEL_NAMES:
        strategies = STRATEGIES if model == "yolo11s" else ("R", "S->R")
        for strategy in strategies:
            for seed in TRAINING_SEEDS:
                formal.append(_job("formal", model, strategy, seed, _stages(strategy, u_s, u_r),
                                   base["strategies"][strategy], measurements, epochs_r=epochs_r))
    calibration = []
    for epochs in (50, 100):
        updates = budget_updates(n_r, epochs, base["effective_batch"])
        calibration.append(_job(
            "calibration", "yolo11s", "R", 17, [_stage("final", updates, updates, 0)],
            {"updates": updates, "R_exposures": base["effective_batch"] * updates, "S_exposures": 0},
            measurements, epochs_r=epochs))
    totals = _totals(formal + calibration)
    measured = not totals["unmeasured_models"]
    return {
        "status": "CANDIDATE_CONDITIONAL_CHECKPOINT_ESTIMATE" if measured else "CANDIDATE_UNMEASURED",
        "candidate_only": True, "execution_authorized": False, "resource_pass": False,
        "hardware_measured": False, "measurement_provenance_verified": False,
        "files_created": False,
        "budget": {key: base[key] for key in (
            "N_S", "N_R", "effective_batch", "nominal_epochs_S", "nominal_epochs_R", "U_S", "U_R",
            "padding_S", "padding_R", "equivalent_epochs_S", "equivalent_epochs_R")},
        "checkpoint_rule": {"interval_updates": CHECKPOINT_INTERVAL, "save_stage_zero": True,
                            "save_stage_final": True, "deduplicate_final_with_interval": True,
                            "phase_reset_zero_is_separate": True},
        "formal_jobs": formal, "calibration_jobs": calibration,
        "formal_totals": _totals(formal), "calibration_totals": _totals(calibration),
        "combined_totals": totals,
        "complete_run_bytes": None, "wall_time_seconds": None, "storage_path": None,
        "excluded_resource_items": [
            "logs_and_step_records", "predictions_and_evaluation",
            "diagnostics_failures_and_interruptions", "serialization_and_async_writer_peak",
            "code_environment_and_data_cache", "retention_headroom_and_time"],
    }
