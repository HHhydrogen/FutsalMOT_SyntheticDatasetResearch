"""完整检查点的提交边界CPU快照/候选恢复载荷；不是Trainer恢复。

caller须先冻结同一成功提交边界，停止并发raw/EMA/optimizer/sampler修改。
本模块只能验证显式状态的一致性，不能证明跨对象原子性或optimizer已成功。
不重建优化器、不加载模型、不提交采样、不改变全局RNG。阶段0允许承接上阶段
raw及全局累计，要求EMA从当前raw重新初始化、计数0。历史流须保留累计游标。
"""
from __future__ import annotations
from collections import OrderedDict
import math
import random

import numpy as np
import torch

from .data import CycleSampler,PrefetchSampler,_tuples
from .records import CHECKPOINT_SCHEMA,capture_rng_state,validate_checkpoint_contract,_binding_digest

SNAPSHOT_SCHEMA="futsalmot_boundary_snapshot_v1"
_EXPECTED_KEYS={"phase","phase_updates","committed_updates","numeric_mode","cumulative_exposures","effective_batch"}
_STREAMS={"R","S","R_extra"}


def _integer(value,name):
    if type(value) is not int or value < 0:
        raise ValueError(name+"须为明确非负整数")
    return value


def _cuda_permission(allow_cuda,cuda_initialized):
    if type(allow_cuda) is not bool or type(cuda_initialized) is not bool:
        raise ValueError("CUDA许可和已初始化声明必须为bool")
    if not allow_cuda or not cuda_initialized:
        raise ValueError("CUDA状态只允许caller明确授权且声明已初始化的分支")


def _cuda_context(allow_cuda,cuda_initialized):
    if type(allow_cuda) is not bool or type(cuda_initialized) is not bool:
        raise ValueError("CUDA许可和已初始化声明必须为bool")
    if allow_cuda != cuda_initialized:
        raise ValueError("CUDA上下文声明必须同时明确授权及已初始化")
    return allow_cuda


def _clone_cpu(value,*,allow_cuda=False,cuda_initialized=False,active=None):
    if isinstance(value,torch.Tensor):
        if value.layout != torch.strided or value.device.type not in ("cpu","cuda"):
            raise ValueError("检查点只支持普通CPU/CUDA Tensor，不支持meta/稀疏状态")
        if value.device.type == "cuda":
            _cuda_permission(allow_cuda,cuda_initialized)
            if not torch.cuda.is_initialized():
                raise RuntimeError("拒绝为快照初始化CUDA")
        return value.detach().to(device="cpu",copy=True)
    if value is None or type(value) in (bool,int,float,str,bytes):
        return value
    if type(value) not in (dict,OrderedDict,list,tuple):
        raise ValueError("状态仅接受原生容器/标量/Tensor，不保存模型或任意对象")
    active=set() if active is None else active
    if id(value) in active:raise ValueError("状态容器不能形成循环")
    active.add(id(value))
    def clone(v):return _clone_cpu(v,allow_cuda=allow_cuda,cuda_initialized=cuda_initialized,active=active)
    if type(value) in (dict,OrderedDict):
        result=OrderedDict() if type(value) is OrderedDict else {}
        for key,item in value.items():
            if type(key) not in (str,int,float,bool,bytes,tuple):raise ValueError("不支持的state_dict键")
            result[clone(key)]=clone(item)
        if type(value) is OrderedDict:
            for name,item in vars(value).items():setattr(result,name,clone(item)) # 含原生state_dict._metadata。
    elif type(value) is list:result=[clone(v) for v in value]
    else:result=tuple(clone(v) for v in value)
    active.remove(id(value))
    return result


def _model_states(raw,ema,phase_updates):
    if not isinstance(raw,dict) or not raw or any(not isinstance(k,str) or not k for k in raw):
        raise ValueError("raw必须为非空完整Tensor state_dict")
    if not isinstance(ema,dict) or set(ema) != {"schema_version","updates","state_dict"} or ema["schema_version"] != "futsalmot_ema_state_v1":
        raise ValueError("需要现有完整EMA状态schema，不接受只有EMA权重")
    if type(ema["updates"]) is not int or ema["updates"] != phase_updates:
        raise ValueError("EMA updates与阶段已提交更新不一致")
    values=ema["state_dict"]
    if not isinstance(values,dict) or set(values) != set(raw):raise ValueError("raw/EMA完整keys不一致")
    integer_types={torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.bool}
    for key,a in raw.items():
        b=values[key]
        for tensor in (a,b):
            if not isinstance(tensor,torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided:
                raise ValueError("快照raw/EMA须为完整普通CPU Tensor")
            if tensor.is_floating_point():
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("raw/EMA浮点参数和buffer必须保留有限FP32")
            elif tensor.dtype not in integer_types:raise ValueError("不支持的raw/EMA buffer dtype")
        if a.shape != b.shape or a.dtype != b.dtype:raise ValueError("raw/EMA shape或dtype不一致")
        if phase_updates == 0 and not torch.equal(a,b):
            raise ValueError("阶段0 EMA必须从当前承接raw重新初始化")


def _optimizer_protocol(value,active=None):
    """本实验bundle准入：optimizer浮点Tensor有限FP32，整数buffer保真。

    通用records序列化仍保留任意支持dtype；此处不静默转换不合规优化器状态。
    """
    if isinstance(value,torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("optimizer快照须为普通CPU Tensor")
        if value.is_floating_point():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("optimizer浮点Tensor须为有限FP32，拒绝FP16/FP64/非有限状态")
        elif value.dtype not in {getattr(torch,name) for name in ("bool","uint8","uint16","uint32","uint64","int8","int16","int32","int64")}:
            raise ValueError("optimizer只支持FP32或原整数/bool buffer")
        return
    if type(value) is float:
        if not math.isfinite(value):raise ValueError("optimizer Python数值hyperparams须有限")
        return
    if value is None or type(value) in (bool,int,str,bytes):return
    if type(value) not in (dict,OrderedDict,list,tuple):
        raise ValueError("optimizer含不支持的非原生状态对象")
    active=set() if active is None else active
    if id(value) in active:raise ValueError("optimizer状态容器不能循环")
    active.add(id(value))
    if isinstance(value,dict):
        for key,child in value.items():
            _optimizer_protocol(key,active);_optimizer_protocol(child,active)
        if type(value) is OrderedDict:
            for child in vars(value).values():_optimizer_protocol(child,active)
    else:
        for child in value:_optimizer_protocol(child,active)
    active.remove(id(value))


def _rng(rng,*,allow_cuda=False,cuda_initialized=False):
    cuda_context=_cuda_context(allow_cuda,cuda_initialized)
    if not isinstance(rng,dict) or set(rng) != {"schema_version","python","numpy","torch_cpu","cuda"} or rng["schema_version"] != "futsalmot_rng_v1":
        raise ValueError("检查点缺失/未知完整RNG状态")
    if cuda_context and rng["cuda"] is None:
        raise ValueError("声明CUDA上下文时缺少完整CUDA RNG，不能作为完整恢复状态")
    try:
        random.Random(0).setstate(_tuples(rng["python"]))
        n=rng["numpy"]
        if not isinstance(n,dict) or set(n) != {"bit_generator","keys","position","has_gauss","cached_gaussian"}:
            raise ValueError("不完整NumPy全局RNG")
        keys=n["keys"]
        if n["bit_generator"] != "MT19937" or not isinstance(keys,list) or len(keys) != 624 or any(type(k) is not int or not 0 <= k < 2**32 for k in keys):
            raise ValueError("NumPy RNG key状态不合法")
        if type(n["position"]) is not int or not 0 <= n["position"] <= 624 or type(n["has_gauss"]) is not int or n["has_gauss"] not in (0,1):
            raise ValueError("NumPy RNG游标不合法")
        if type(n["cached_gaussian"]) not in (int,float) or not math.isfinite(n["cached_gaussian"]):raise ValueError("NumPy RNG缓存不合法")
        np.random.RandomState(0).set_state((n["bit_generator"],np.array(keys,dtype=np.uint32),n["position"],n["has_gauss"],n["cached_gaussian"]))
        cpu=rng["torch_cpu"]
        if not isinstance(cpu,torch.Tensor) or cpu.device.type != "cpu" or cpu.dtype != torch.uint8:
            raise ValueError("Torch CPU RNG须为CPU ByteTensor")
        torch.Generator(device="cpu").set_state(cpu)
    except (KeyError,TypeError,IndexError,RuntimeError) as exc:
        raise ValueError("RNG候选状态不可恢复") from exc
    if rng["cuda"] is not None:
        _cuda_permission(allow_cuda,cuda_initialized)
        if not isinstance(rng["cuda"],list) or not rng["cuda"] or any(not isinstance(s,torch.Tensor) or s.device.type != "cpu" or s.dtype != torch.uint8 or s.ndim != 1 for s in rng["cuda"]):
            raise ValueError("CUDA RNG载荷须为明确CPU ByteTensor列表")
        # 仅验证载荷，不查询GPU数量、不调用CUDA恢复；该分支尚需未来资源准入实测。


def _sampler_states(states,expected_samplers):
    if not isinstance(expected_samplers,dict) or not expected_samplers or set(expected_samplers)-_STREAMS:
        raise ValueError("必须给出本run实际R/S/R_extra sampler对象集合")
    if not isinstance(states,dict) or set(states) != set(expected_samplers):
        raise ValueError("检查点sampler流集合与本run绑定不一致")
    totals={"R":0,"S":0}
    for stream,expected in expected_samplers.items():
        if type(expected) is not PrefetchSampler:raise ValueError("需要实际PrefetchSampler，不接受派发请求或伪造state代理")
        saved=states[stream]
        if not isinstance(saved,dict) or set(saved) != {"schema_version","sampler"} or saved["schema_version"] != PrefetchSampler.SCHEMA:
            raise ValueError("只允许已提交sampler状态，不保存pending/派发token")
        live=expected.state_dict()["sampler"]
        inner=saved["sampler"]
        if not isinstance(inner,dict) or set(inner) != set(live) or inner.get("domain") != stream:
            raise ValueError("sampler内部schema/流身份不一致")
        # 本地接口的UID模板用于独立候选；绝不load到caller真实流或开放旧token。
        uids,task,domain,seed=expected._template
        if domain != stream:raise ValueError("本run sampler对象与流key不一致")
        candidate=CycleSampler(uids,task=task,domain=domain,seed=seed)
        candidate.load_state_dict(inner)
        draws=_integer(inner["draws"],"sampler committed draws")
        if draws % 16:raise ValueError("sampler已提交曝光不在完整16图更新边界")
        totals["R" if stream == "R_extra" else stream]+=draws
    return totals


def _metadata(bundle):
    return {key:bundle[key] for key in _EXPECTED_KEYS}


def validate_checkpoint_bundle(bundle,*,expected_bindings,expected_metadata,expected_samplers,
                               allow_cuda=False,cuda_initialized=False):
    """验证已可信读回的bundle，返回None；不修改模型/优化器/采样器/全局RNG。

    expected_samplers必须来自当前run实际UID池，独立CycleSampler候选验证其绑定。
    这不证明历史优化器执行真实性；expected_metadata须来自调用者可信run边界。
    """
    _cuda_context(allow_cuda,cuda_initialized)
    validate_checkpoint_contract(bundle)
    if bundle["schema_version"] != CHECKPOINT_SCHEMA or bundle.get("snapshot_schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("不是本工具提交边界快照schema")
    if not expected_bindings or _binding_digest(bundle["source_bindings"]) != _binding_digest(expected_bindings):
        raise ValueError("source_bindings与本run期望不一致")
    if not isinstance(expected_metadata,dict) or set(expected_metadata) != _EXPECTED_KEYS:
        raise ValueError("期望metadata须明确phase/阶段/全局/数值/曝光/effective_batch")
    for name in ("phase_updates","committed_updates"):_integer(bundle[name],name)
    if bundle["phase"] not in ("first","final") or bundle.get("effective_batch") != 16 or type(bundle.get("effective_batch")) is not int:
        raise ValueError("阶段及有效batch16边界必须明确")
    counters=bundle.get("cumulative_exposures")
    if not isinstance(counters,dict) or set(counters) != {"R","S"}:raise ValueError("缺明确累计R/S曝光")
    for key,value in counters.items():_integer(value,key+"累计曝光")
    if counters["R"]+counters["S"] != 16*bundle["committed_updates"]:
        raise ValueError("累计R+S不等于16×全局已提交更新")
    if _binding_digest(_metadata(bundle)) != _binding_digest(expected_metadata):
        raise ValueError("快照metadata与可信调用者边界不一致")
    _model_states(bundle["raw_model"],bundle["ema"],bundle["phase_updates"])
    optimizer=bundle["optimizer"]
    if not {"state","param_groups"} <= optimizer.keys() or not isinstance(optimizer["state"],dict) or not isinstance(optimizer["param_groups"],list) or not optimizer["param_groups"] or any(not isinstance(g,dict) for g in optimizer["param_groups"]):
        raise ValueError("必须保留完整optimizer.state_dict，不重建缺失优化器")
    _optimizer_protocol(optimizer)
    if bundle["numeric_mode"] == "amp_fp16" and not bundle["scaler"]:
        raise ValueError("AMP FP16快照必须显式提供已启用scaler完整状态")
    if _sampler_states(bundle["sampler"],expected_samplers) != counters:
        raise ValueError("sampler已提交游标（R_extra计入R）与累计曝光不齐")
    _rng(bundle["rng"],allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)


def snapshot_checkpoint(*,raw_state,ema_state,optimizer_state,phase,phase_updates,committed_updates,
                        source_bindings,numeric_mode,scaler_state,samplers,cumulative_exposures,
                        extra_state=None,allow_cuda=False,cuda_initialized=False,include_cuda_rng=False):
    """caller冻结同一提交边界后调用；pending可有，但state_dict仅取committed。

    所有Tensor独立detach复制到CPU，原dtype保真，不共享可写容器或Tensor。
    raw/EMA/optimizer浮点Tensor严格有限FP32，整数buffer保真；不静默转换。
    RNG在本次调用中显式capture；CUDA默认不触及。声明CUDA上下文时，
    allow_cuda/cuda_initialized/include_cuda_rng三者必须同时为True，不能漏采CUDA RNG。
    """
    if type(include_cuda_rng) is not bool:raise ValueError("include_cuda_rng须为显式bool")
    cuda_context=_cuda_context(allow_cuda,cuda_initialized)
    if include_cuda_rng != cuda_context:
        raise ValueError("CUDA上下文授权/已初始化/采集CUDA RNG三项必须一致")
    if not isinstance(samplers,dict) or not samplers or set(samplers)-_STREAMS:
        raise ValueError("需要明确已提交PrefetchSampler流集合")
    saved={}
    for name,sampler in samplers.items():
        if type(sampler) is not PrefetchSampler:raise ValueError("samplers必须是PrefetchSampler对象，不是pending请求")
        saved[name]=sampler.state_dict()
    state={"schema_version":CHECKPOINT_SCHEMA,"snapshot_schema_version":SNAPSHOT_SCHEMA,
           "raw_model":raw_state,"ema":ema_state,"optimizer":optimizer_state,
           "phase":phase,"phase_updates":phase_updates,"committed_updates":committed_updates,
           "source_bindings":source_bindings,"numeric_mode":numeric_mode,"scaler":scaler_state,
           "sampler":saved,"cumulative_exposures":cumulative_exposures,"effective_batch":16}
    if extra_state is not None:
        if not isinstance(extra_state,dict) or set(extra_state)&set(state) or "rng" in extra_state:
            raise ValueError("extra_state不能覆盖完整检查点必需字段")
        state.update(extra_state)
    state=_clone_cpu(state,allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)
    state["rng"]=_clone_cpu(capture_rng_state(include_cuda=include_cuda_rng),allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)
    validate_checkpoint_bundle(state,expected_bindings=source_bindings,expected_metadata=_metadata(state),
                               expected_samplers=samplers,allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)
    if any(samplers[name].state_dict() != saved[name] for name in samplers):
        raise ValueError("快照期间committed sampler变化，拒绝跨边界；caller须停止并发修改")
    return state


def prepare_restore_payload(bundle,*,expected_bindings,expected_metadata,expected_samplers,
                            allow_cuda=False,cuda_initialized=False):
    """严格验证后返回全新CPU载荷，不热改对象；声明CUDA上下文必须有CUDA RNG。"""
    validate_checkpoint_bundle(bundle,expected_bindings=expected_bindings,expected_metadata=expected_metadata,
                               expected_samplers=expected_samplers,allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)
    return _clone_cpu(bundle,allow_cuda=allow_cuda,cuda_initialized=cuda_initialized)
