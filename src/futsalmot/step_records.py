"""微批到逐更新记录的纯metadata桥接；不读取数据、不导入Torch、不提交采样。

builder只核对调用方提供的metadata一致性，不能证明optimizer已经成功。未来只有
调用方确实成功更新后才可将结果交RunRecorder并按既定事务规则commit；本模块
不实现这些操作、不把materialized曝光称为已提交，也不补造缺失测量。
"""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import re

_FIELDS = {"request", "source_reference", "image_source_binding", "sampling_stream", "data_domain"}


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + "须为明确整数且不小于" + str(minimum))
    return value


def _number(value, name, minimum=0, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(name + "须为合法有限标量")
    if upper is not None and value >= upper:
        raise ValueError(name + "超过允许范围")
    return float(value)


def _pure(value, depth=0, budget=None):
    budget = [25000] if budget is None else budget
    budget[0] -= 1
    if depth > 20 or budget[0] < 0:
        raise ValueError("metadata超有界深度/节点数")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata含非有限标量")
        return
    if type(value) in (list, tuple):
        for child in value:
            _pure(child, depth + 1, budget)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _pure(child, depth + 1, budget)
        return
    raise ValueError("metadata仅接受原生JSON式值；禁止Tensor/RGB/自定义对象")


def _lineage(item, uid, domain):
    if type(item) is not dict or set(item) != _FIELDS:
        raise ValueError("需要完整非None五字段lineage")
    _pure(item)
    request, reference, binding = (item[k] for k in ("request", "source_reference", "image_source_binding"))
    if not all(type(v) is dict for v in (request, reference, binding)):
        raise ValueError("请求/来源/图像绑定必须为dict")
    stream = item["sampling_stream"]
    physical = "R" if stream == "R_extra" else stream
    if stream not in ("R", "S", "R_extra") or physical != domain or item["data_domain"] != domain:
        raise ValueError("lineage stream/data_domain不一致")
    if not {"uid", "domain", "stream", "data_domain", "epoch", "exposure_index", "augmentation_seed"} <= request.keys():
        raise ValueError("请求缺明确身份/曝光/增强字段，不补默认")
    if (request["uid"] != uid or request["domain"] not in (stream, domain)
            or request["stream"] != stream or request["data_domain"] != domain):
        raise ValueError("lineage请求与UID/stream/domain不一致")
    for key in ("epoch", "exposure_index", "augmentation_seed"):
        _int(request[key], key)
    allowed = ("train", "val") if domain == "R" else ("train", "dev")
    if reference.get("domain") != domain or reference.get("split") not in allowed:
        raise ValueError("来源域/split非法")
    if "split" in request and request["split"] != reference["split"]:
        raise ValueError("请求与来源split不一致")
    if "view" in request:
        view = request["view"]
        expected = view if domain == "R" else ("train" if view == "train" else "dev")
        if view not in ("train", "val") or reference["split"] != expected:
            raise ValueError("请求view与来源split不一致")
    for key in (("index_path",) if domain == "R" else ("seed_dir", "camera_id")):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise ValueError("缺来源引用: " + key)
    if domain == "S":
        _int(reference.get("frame_index"), "frame_index", 1)
    if not isinstance(binding.get("path"), str) or not Path(binding["path"]).is_absolute():
        raise ValueError("缺原绝对图像来源路径")
    identity = binding.get("stat")
    if not isinstance(identity, (list, tuple)) or len(identity) != 4:
        raise ValueError("缺原图像stat绑定")
    for value in identity:
        _int(value, "source_stat")
    if identity[2] == 0:
        raise ValueError("原图像stat字节为空")
    return stream


def trace_microbatch(collated):
    """只复制必要metadata，绝不复制或持有img/Tensor；不证明更新或commit。"""
    required = {"reservation_token", "effective_batch_size", "microbatch_offset", "uids", "domains", "sample_lineage"}
    if type(collated) is not dict or required - collated.keys():
        raise ValueError("微批缺reservation/offset/UID/完整lineage字段")
    selected = {key: collated[key] for key in required}
    _pure(selected)
    token = selected["reservation_token"]
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}:(0|[1-9][0-9]*):(0|[1-9][0-9]*)", token) is None:
        raise ValueError("reservation token格式非法")
    if type(selected["effective_batch_size"]) is not int or selected["effective_batch_size"] != 16:
        raise ValueError("有效batch必须16")
    uids, domains, lineage = (selected[key] for key in ("uids", "domains", "sample_lineage"))
    if not all(type(v) is list for v in (uids, domains, lineage)) or len(uids) not in (1, 2, 4, 8, 16):
        raise ValueError("微批metadata长度非法")
    count = len(uids)
    offset = _int(selected["microbatch_offset"], "microbatch_offset")
    if offset % count or offset + count > 16 or len(domains) != count or len(lineage) != count:
        raise ValueError("offset与微批长度/范围不一致")
    streams = []
    for uid, domain, item in zip(uids, domains, lineage):
        if not isinstance(uid, str) or not uid or domain not in ("R", "S"):
            raise ValueError("微批UID/domain非法")
        streams.append(_lineage(item, uid, domain))
    if len(set(streams)) != 1:
        raise ValueError("同微批混合采样流")
    rows = [item["request"] for item in lineage]
    for before, after in zip(rows, rows[1:]):
        if after["exposure_index"] != before["exposure_index"] + 1 or after["epoch"] - before["epoch"] not in (0, 1):
            raise ValueError("微批请求曝光/epoch错序")
    return {"schema_version": "microbatch_trace_v1", **deepcopy(selected)}


def _components(value, label):
    if type(value) is not dict or set(value) != {"box", "cls", "dfl"}:
        raise ValueError(label + "须显式提供box/cls/dfl实测分量")
    result = {key: _number(v, label + "." + key) for key, v in value.items()}
    return result


def _exposures(value, label):
    if type(value) is not dict or set(value) != {"R", "S"}:
        raise ValueError(label + "须明确R/S累计")
    return {key: _int(number, label + "." + key) for key, number in value.items()}


def build_update_record(traces, *, update, phase, phase_update, loss, loss_components, loss_definition,
                        backward_loss, backward_components, lr, momentum, parameter_groups,
                        gradient_norm_before_clip, gradient_norm_after_clip,
                        previous_exposures, cumulative_exposures, elapsed_seconds):
    """构造兼容RunRecorder.step的候选记录；所有测量必须由caller显式给出。

    loss/components采用caller声明的实测三项总量，backward_*独立保留实际sum/16
    归一化量，不根据microbatch大小反推。lr/momentum对应首参数组；每组须提供
    唯一name及实际lr/momentum。gradient_norm保留clip前范数，并另留clip后值。
    本函数不证明更新成功、不写文件、不commit。未来只有真实成功后caller才可
    持久记录。纯CPU夹具用假测量验证序列化，不是实际训练结果。
    """
    if not isinstance(traces, (list, tuple)) or not 1 <= len(traces) <= 16:
        raise ValueError("需要有界完整trace组")
    normalized = []
    for value in traces:
        if type(value) is not dict or value.get("schema_version") != "microbatch_trace_v1":
            raise ValueError("需要trace_microbatch产生的明确schema")
        normalized.append(trace_microbatch(value))
    first = normalized[0]
    token, offset, size = first["reservation_token"], 0, len(first["uids"])
    stream = first["sample_lineage"][0]["sampling_stream"]
    domain = first["domains"][0]
    lineage, uids = [], []
    for value in normalized:
        if (value["reservation_token"] != token or value["microbatch_offset"] != offset
                or len(value["uids"]) != size
                or any(v != domain for v in value["domains"])
                or any(item["sampling_stream"] != stream for item in value["sample_lineage"])):
            raise ValueError("trace缺组/错序/混token/混流/微批尺寸变化")
        offset += len(value["uids"])
        uids.extend(value["uids"])
        lineage.extend(value["sample_lineage"])
    if offset != 16:
        raise ValueError("trace未完整覆盖0..15")
    requests = [item["request"] for item in lineage]
    for before, after in zip(requests, requests[1:]):
        if after["exposure_index"] != before["exposure_index"] + 1 or after["epoch"] - before["epoch"] not in (0, 1):
            raise ValueError("有效batch请求曝光重复或错序")
    update = _int(update, "update", 1)
    phase_update = _int(phase_update, "phase_update", 1)
    if phase not in ("first", "final") or phase_update > update:
        raise ValueError("phase/phase_update非法")
    previous = _exposures(previous_exposures, "previous_exposures")
    cumulative = _exposures(cumulative_exposures, "cumulative_exposures")
    increments = {"R": 16 if domain == "R" else 0, "S": 16 if domain == "S" else 0}
    if (sum(previous.values()) != 16 * (update - 1)
            or any(cumulative[key] != previous[key] + increments[key] for key in ("R", "S"))):
        raise ValueError("累计曝光须与前次及本步所属域16张完全一致")
    measured_loss = _number(loss, "loss")
    measured_backward = _number(backward_loss, "backward_loss")
    components = _components(loss_components, "loss_components")
    backward = _components(backward_components, "backward_components")
    if not isinstance(loss_definition, str) or not loss_definition.strip():
        raise ValueError("须明确实测loss及分量的聚合定义")
    measured_lr = _number(lr, "lr")
    measured_momentum = _number(momentum, "momentum", upper=1)
    before = _number(gradient_norm_before_clip, "gradient_norm_before_clip")
    after = _number(gradient_norm_after_clip, "gradient_norm_after_clip")
    elapsed = _number(elapsed_seconds, "elapsed_seconds")
    _pure(parameter_groups)
    if not isinstance(parameter_groups, (list, tuple)) or not parameter_groups:
        raise ValueError("须给出各参数组实测LR/momentum")
    names = set()
    for group in parameter_groups:
        if type(group) is not dict or not {"name", "lr", "momentum"} <= group.keys():
            raise ValueError("参数组缺name/lr/momentum")
        name = group["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("参数组名称必须明确且唯一")
        names.add(name)
        _number(group["lr"], "group.lr")
        _number(group["momentum"], "group.momentum", upper=1)
    if parameter_groups[0]["lr"] != measured_lr or parameter_groups[0]["momentum"] != measured_momentum:
        raise ValueError("主lr/momentum须与首参数组实测值一致")
    return {"record_schema": "candidate_update_record_v1", "update": update, "phase": phase,
            "phase_update": phase_update, "domain": stream, "data_domain": domain, "uids": uids,
            "reservation_token": token, "effective_batch_size": 16, "sample_lineage": deepcopy(lineage),
            "microbatch_offsets": [value["microbatch_offset"] for value in normalized],
            "loss": measured_loss, "loss_components": components, "loss_definition": loss_definition, "backward_loss": measured_backward,
            "backward_components": backward, "backward_normalization": "sum(returned_microbatch_scaled_vector)/16",
            "lr": measured_lr, "momentum": measured_momentum, "parameter_groups": deepcopy(parameter_groups),
            "gradient_norm": before, "gradient_norm_before_clip": before, "gradient_norm_after_clip": after,
            "R_exposures": increments["R"], "S_exposures": increments["S"],
            "previous_exposures": previous, "cumulative_exposures": cumulative, "elapsed_seconds": elapsed}
