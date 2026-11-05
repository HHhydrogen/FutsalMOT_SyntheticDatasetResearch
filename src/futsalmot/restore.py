"""可信factory新建CPU对象上的候选恢复；不是Trainer生命周期或运行授权。

caller负责factory确实新建且独占对象，不能仅凭无状态API证明factory没返回外部
既有对象；不得传复用已有模型的factory。本模块检查本次raw/EMA模块/参数/存储
彼此及bundle不共享，不重复构建模型以猜测fresh。没有跨进程锁、崩溃事务或GPU
恢复；任何factory副作用不在本函数可回滚范围内，失败仅保证全局CPU RNG恢复。
"""
from __future__ import annotations
from copy import deepcopy
import math

import torch

from .checkpoint_state import prepare_restore_payload,_clone_cpu
from .data import CycleSampler,PrefetchSampler
from .optimizers import _check_optimizer
from .records import capture_rng_state,restore_rng_state


def _tensors(value):
    if isinstance(value,torch.Tensor):
        yield value
    elif isinstance(value,dict):
        for item in value.values():yield from _tensors(item)
        for item in vars(value).values() if hasattr(value,"__dict__") else ():
            yield from _tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from _tensors(item)


def _storage(tensors):
    result=set()
    for t in tensors:
        if t.device.type != "cpu" or t.layout != torch.strided:
            raise ValueError("BLOCKED: 恢复只接受普通CPU Tensor")
        if t.numel():result.add(t.untyped_storage().data_ptr())
    return result


def _module(module,expected):
    if not isinstance(module,torch.nn.Module):raise ValueError("factory须返回新建CPU nn.Module")
    actual=module.state_dict()
    if set(actual) != set(expected):raise ValueError("factory与检查点strict state_dict keys不一致")
    for key,t in actual.items():
        saved=expected[key]
        if not isinstance(t,torch.Tensor) or t.device.type != "cpu" or t.layout != torch.strided:
            raise ValueError("BLOCKED: factory必须直接建立CPU模块，不自动搬设备")
        if t.is_floating_point() and t.dtype != torch.float32:
            raise ValueError("factory浮点参数/buffer必须FP32，不静默转换")
        if t.shape != saved.shape or t.dtype != saved.dtype:
            raise ValueError("factory与检查点shape/dtype不一致: "+key)
    for p in module.parameters():
        if p.device.type != "cpu" or p.dtype != torch.float32 or p.layout != torch.strided:
            raise ValueError("全部factory参数（含冻结参数）须CPU FP32")
    return actual


def _no_sharing(raw,ema,payload,bundle):
    if {id(m) for m in raw.modules()} & {id(m) for m in ema.modules()}:
        raise ValueError("raw/EMA factory共享Module对象，拒绝写回")
    a=list(raw.state_dict().values());b=list(ema.state_dict().values())
    source=list(_tensors(payload))+list(_tensors(bundle))
    sa,sb,sc=_storage(a),_storage(b),_storage(source)
    if sa&sb or (sa|sb)&sc or {id(p) for p in raw.parameters()}&{id(p) for p in ema.parameters()}:
        raise ValueError("factory参数/buffer与raw/EMA/bundle存在共享存储")


def _equal(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def _group_scalar_tree(value):
    if value is None or type(value) in (bool,str,int):return
    if type(value) is float and math.isfinite(value):return
    if isinstance(value,(list,tuple)):
        for x in value:_group_scalar_tree(x)
        return
    raise ValueError("SGD组配置须为有限原生标量/名字序列")


def _optimizer_topology(optimizer,raw,saved):
    _check_optimizer(optimizer,torch) # 精确SGD及现有helper的三组/身份/冻结绑定。
    if set(saved) != {"state","param_groups"}:raise ValueError("只恢复原生SGD state_dict，不猜测额外optimizer字段")
    fresh=optimizer.state_dict()
    if len(saved["param_groups"]) != len(fresh["param_groups"]):raise ValueError("SGD组数不一致")
    parameters={};seen=set()
    for current,native,stored in zip(optimizer.param_groups,fresh["param_groups"],saved["param_groups"]):
        if not isinstance(stored,dict) or not {"params","name","parameter_names","lr","momentum"} <= stored.keys():
            raise ValueError("SGD检查点缺具名参数组及配置")
        if stored["params"] != native["params"] or any(type(k) is not int or k < 0 for k in stored["params"]):
            raise ValueError("SGD参数ID/顺序与新对象不一致")
        for key,value in stored.items():
            if key != "params":_group_scalar_tree(value)
        # LR/momentum是已提交更新的运行状态，factory阶段0初值可不同，不能阻止合法恢复。
        for key,value in native.items():
            if key not in ("params","lr","momentum") and (key not in stored or not _equal(value,stored[key])):
                raise ValueError("SGD静态组name/parameter_names/顺序/decay/类型不一致: "+key)
        if len(current["params"]) != len(stored["params"]):raise ValueError("SGD组内参数数目不一致")
        for saved_id,param in zip(stored["params"],current["params"]):
            if saved_id in seen:raise ValueError("SGD保存参数ID重复")
            seen.add(saved_id);parameters[saved_id]=param
    expected={id(p) for p in raw.parameters() if p.requires_grad}
    if {id(p) for p in parameters.values()} != expected:
        raise ValueError("optimizer factory未绑定本次raw可训练参数")
    if set(saved["state"])-set(parameters):raise ValueError("SGD state引用未知参数ID")
    for key,state in saved["state"].items():
        if not isinstance(state,dict) or set(state)-{"momentum_buffer"}:
            raise ValueError("SGD state含未定义字段，拒绝猜测形状/遗漏恢复")
        if "momentum_buffer" in state:
            t=state["momentum_buffer"];p=parameters[key]
            if not isinstance(t,torch.Tensor) or t.device.type != "cpu" or t.dtype != p.dtype or t.shape != p.shape:
                raise ValueError("SGD momentum shape/dtype与参数不一致")


def restore_cpu_checkpoint(bundle,*,expected_bindings,expected_metadata,expected_samplers,
                           raw_factory,ema_factory,sgd_factory):
    """只建立新CPU raw/EMA/SGD/sampler，所有校验通过后才恢复检查点CPU RNG。

    两个Module factory须是真正fresh、无外部共享的可信caller函数；sgd_factory(raw)
    须调用现make_sgd_optimizer建立绑定三组。raw保留factory requires_grad，设置
    train；EMA设置eval且关闭梯度。失败不对caller模型/采样器load，CPU RNG还原。
    FP16/scaler与任何CUDA bundle具体拒绝；未实现Trainer/锁/崩溃重放边界。
    """
    before=capture_rng_state(include_cuda=False)
    try:
        payload=prepare_restore_payload(bundle,expected_bindings=expected_bindings,
                                        expected_metadata=expected_metadata,expected_samplers=expected_samplers)
        if payload["rng"]["cuda"] is not None:
            raise ValueError("BLOCKED: CUDA RNG恢复尚未实现")
        if payload["numeric_mode"] == "amp_fp16" or payload["scaler"] is not None:
            raise ValueError("BLOCKED: CPU候选未实现FP16 GradScaler对象恢复")
        if not all(callable(f) for f in (raw_factory,ema_factory,sgd_factory)):
            raise ValueError("需要明确可信的新Module/SGD factory")
        raw=raw_factory();ema=ema_factory()
        _module(raw,payload["raw_model"]);_module(ema,payload["ema"]["state_dict"])
        _no_sharing(raw,ema,payload,bundle)
        frozen={name:p.requires_grad for name,p in raw.named_parameters()}
        # assign=False复制到新参数，保持factory Parameter身份/冻结标记，不借用payload存储。
        raw.load_state_dict(payload["raw_model"],strict=True,assign=False)
        ema.load_state_dict(payload["ema"]["state_dict"],strict=True,assign=False)
        if frozen != {name:p.requires_grad for name,p in raw.named_parameters()}:
            raise ValueError("factory冻结参数标记在恢复过程中发生变化")
        optimizer=sgd_factory(raw)
        _optimizer_topology(optimizer,raw,payload["optimizer"])
        # PyTorch load_state_dict可能复用CPU同dtype输入；必须另复制一份给新optimizer。
        optimizer.load_state_dict(_clone_cpu(payload["optimizer"]))
        _check_optimizer(optimizer,torch)
        if not _equal(optimizer.state_dict(),payload["optimizer"]):
            raise ValueError("SGD完整state/groups载入后与检查点不一致")
        opt_storage=_storage(_tensors(optimizer.state_dict()["state"]))
        if opt_storage & (_storage(_tensors(payload))|_storage(raw.state_dict().values())|_storage(ema.state_dict().values())|_storage(_tensors(bundle))):
            raise ValueError("恢复后的optimizer状态存在跨对象共享存储")
        restored_samplers={}
        for stream,expected in expected_samplers.items():
            uids,task,domain,seed=expected._template
            candidate=PrefetchSampler(CycleSampler(uids,task=task,domain=domain,seed=seed),
                                      max_pending_batches=expected.max_pending_batches)
            candidate.load_state_dict(payload["sampler"][stream])
            restored_samplers[stream]=candidate
        raw.train(True);ema.eval();ema.requires_grad_(False)
        _no_sharing(raw,ema,payload,bundle)
        metadata={key:deepcopy(payload[key]) for key in ("phase","phase_updates","committed_updates","cumulative_exposures","source_bindings","numeric_mode","effective_batch")}
        structural={"schema_version","snapshot_schema_version","raw_model","ema","optimizer","sampler","rng","scaler",*metadata}
        result={"raw_model":raw,"ema_model":ema,"ema_updates":payload["ema"]["updates"],
                "optimizer":optimizer,"samplers":restored_samplers,"metadata":metadata,
                "rng_state":_clone_cpu(payload["rng"]),
                "auxiliary_state":{key:_clone_cpu(value) for key,value in payload.items() if key not in structural},
                "candidate_only":True,"trainer_restored":False}
        restore_rng_state(payload["rng"],allow_cuda=False) # 必须是最后一个有状态恢复动作。
        return result
    except BaseException:
        restore_rng_state(before,allow_cuda=False)
        raise
