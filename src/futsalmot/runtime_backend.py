"""真正YOLO检测有效16更新/完整状态接线；不授予作业或GPU准入。

调用方拥有数据、协议/资源锁、成功更新后的sampler commit与持久记录。失败可能已
改变BN/参数，实例立即FAIL_STOP，不回滚或继续；只能新建backend从完整边界恢复。
"""
from __future__ import annotations
from copy import deepcopy
from contextlib import nullcontext
import math
from types import SimpleNamespace
import torch

from .detectors import NativeDetectionNetwork
from .pretrained import load_local_yolo11_detect
from .losses import IgnoreAwareDetectionLoss
from .optimizers import make_sgd_optimizer,apply_sgd_schedule,_check_optimizer
from .training import schedule_at,criterion_backward_loss,initialize_ema_stage
from .checkpoint_state import snapshot_checkpoint,prepare_restore_payload,_clone_cpu
from .records import capture_rng_state,restore_rng_state
from .restore import _optimizer_topology

_SCHEMA="futsalmot_native_bbox_runtime_v1"
_NAMES=("box","cls","dfl")
_METADATA=("phase","phase_updates","committed_updates","numeric_mode","cumulative_exposures","effective_batch")


def _require(value,message):
    if not value:raise ValueError(message)


def _finite(tensors):
    values=[t.detach() for t in tensors if t.is_floating_point() and t.numel()]
    if not values:return True
    return bool(torch.isfinite(torch.stack(torch._foreach_norm(values,float("inf")))).all().item())


def _grad_norm(parameters):
    grads=[p.grad.detach() for p in parameters if p.grad is not None]
    _require(bool(grads),"没有实际梯度")
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads,2.0)),2.0)


class NativeBBoxBackend:
    """FP32 raw、SGD、FP32 EMA；FP32或已准入CUDA AMP BF16，无GradScaler。

    输入是已/255的prepared batch，不读图/增强；micro必须恰好累积16图。公共起点
    只准960，随机YAML fixture可显式64；不会自动加载旧训练权重或初始化CUDA。
    """
    def __init__(self,network,*,stage_updates,final_stage=True,microbatch_size=2,
                 numeric_mode="fp32",device="cpu",input_size=960,allow_initialized_cuda=False):
        _require(type(network) is NativeDetectionNetwork,"需要明确原生检测网络包装")
        _require(type(stage_updates) is int and stage_updates>=2,"阶段至少2个更新")
        _require(type(final_stage) is bool,"final_stage须明确bool")
        _require(type(microbatch_size) is int and microbatch_size in (1,2,4,8,16),"microbatch须为16的约定因数")
        _require(numeric_mode in ("fp32","amp_bf16"),"仅支持FP32/AMP BF16，不使用FP16 scaler")
        _require(type(input_size) is int and input_size>=32 and input_size%32==0,"明确32倍数输入")
        target=torch.device(device)
        _require(target.type in ("cpu","cuda"),"仅CPU/已初始化CUDA")
        if target.type=="cuda":
            if allow_initialized_cuda is not True or not torch.cuda.is_initialized():
                raise RuntimeError("CUDA须外部已初始化及显式准入，本组件不初始化/探测")
            if numeric_mode=="amp_bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("已初始化设备不支持BF16")
            network.model.to(device=target)
            network.device=next(network.model.parameters()).device
        else:
            _require(numeric_mode=="fp32","CPU fixture仅FP32，不借CPU BF16替代GPU预检")
            _require(all(p.device.type=="cpu" for p in network.model.parameters()),"CPU分支不静默搬回GPU模型")
        init=network.initialization_binding
        if init.get("public_pretraining_verified") is True:_require(input_size==960,"公开模型固定960，不降尺寸绕资源")
        elif input_size!=960:_require(init.get("kind")=="random_installed_yaml_fixture_not_public_pretrained","非960仅允许明确随机YAML夹具")
        self.network=network;self.model=network.model;self.device=next(self.model.parameters()).device
        self.stage_updates=stage_updates;self.final_stage=final_stage;self.microbatch_size=microbatch_size
        self.numeric_mode=numeric_mode;self.input_size=input_size;self.stage_update=0
        self.failed=False;self.failure=None;self.optimizer_steps_attempted=0;self.optimizer_steps_returned=0
        args=vars(self.model.args).copy() if isinstance(getattr(self.model,"args",None),SimpleNamespace) else dict(getattr(self.model,"args",{}) or {})
        args.update(box=7.5,cls=.5,dfl=1.5)
        self.model.args=SimpleNamespace(**args)
        self.criterion=IgnoreAwareDetectionLoss(self.model)
        self.optimizer=make_sgd_optimizer(self.model,lr=0.,momentum=.8,weight_decay=.0005)
        self.ema_state=initialize_ema_stage(self.model.state_dict())
        self._parameters=[p for p in self.model.parameters() if p.requires_grad]
        self._state_keys=tuple(self.model.state_dict())
        self.model.train(True)

    @classmethod
    def from_public(cls,weights_directory,*,model_size,**kwargs):
        return cls(load_local_yolo11_detect(weights_directory,model_size=model_size),**kwargs)

    def _config(self):
        return {"schema_version":_SCHEMA,"stage_updates":self.stage_updates,"final_stage":self.final_stage,
                "microbatch_size":self.microbatch_size,"numeric_mode":self.numeric_mode,"input_size":self.input_size,
                "device":str(self.device),"initialization_binding":deepcopy(self.network.initialization_binding)}

    def _batch(self,batch):
        _require(isinstance(batch,dict),"prepared batch须dict")
        img=batch.get("img");m=self.microbatch_size
        _require(isinstance(img,torch.Tensor) and img.device==self.device and img.dtype==torch.float32
                 and tuple(img.shape)==(m,3,self.input_size,self.input_size),"prepared RGB NCHW FP32尺寸/设备或microbatch不一致")
        _require(batch.get("numeric_input")=="float32_RGB_NCHW_divided_by_255_once","必须是已明确/255的prepared输入")
        _require(bool(torch.isfinite(img).all()) and not bool(((img<0)|(img>1)).any()),"输入非有限或不在[0,1]")
        for key,n in (("cls",1),("bboxes",4),("ignore_bboxes",4)):
            x=batch.get(key)
            _require(isinstance(x,torch.Tensor) and x.device==self.device and x.dtype==torch.float32
                     and x.ndim==2 and x.shape[1]==n and bool(torch.isfinite(x).all()),key+"不是同设备有限FP32二维Tensor")
        _require(len(batch["cls"])==len(batch["bboxes"]) and not bool((batch["cls"]!=0).any()),"目标须COCO person0")
        for key,count in (("batch_idx",len(batch["bboxes"])),("ignore_batch_idx",len(batch["ignore_bboxes"]))):
            x=batch.get(key)
            _require(isinstance(x,torch.Tensor) and x.device==self.device and x.dtype==torch.int64 and x.ndim==1
                     and len(x)==count and not bool(((x<0)|(x>=m)).any()),key+"图索引错误")
        boxes=batch["bboxes"]
        corners=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),dim=1)
        _require(not bool(((boxes[:,2:]<=0)|(boxes[:,2:]>1)).any()) and not bool(((corners< -1e-7)|(corners>1+1e-7)).any()),"训练目标须归一化正面积图内框")
        for key in ("uids","domains","sample_lineage"):_require(isinstance(batch.get(key),(list,tuple)) and len(batch[key])==m,key+"须逐图保留")
        _require(all(isinstance(u,str) and u for u in batch["uids"]) and all(d in ("R","S") for d in batch["domains"]),"UID/来源域不合法")
        for uid,domain,lineage in zip(batch["uids"],batch["domains"],batch["sample_lineage"]):
            _require(isinstance(lineage,dict) and isinstance(lineage.get("request"),dict)
                     and lineage["request"].get("uid")==uid and lineage.get("data_domain")==domain,"缺少实际请求/域血缘")
            _require(all(k in lineage for k in ("source_reference","image_source_binding","sampling_stream")),"血缘字段不完整")
        return img

    def _ema_update(self,update):
        raw=self.model.state_dict();old=self.ema_state["state_dict"]
        _require(tuple(raw)==self._state_keys and tuple(old)==self._state_keys,"raw/EMA state keys变化")
        _require(self.ema_state["updates"]+1==update,"EMA更新数不连续")
        d=.9999*(1-math.exp(-update/2000))
        floating=[key for key,v in raw.items() if v.is_floating_point()]
        with torch.no_grad():
            torch._foreach_mul_([old[k] for k in floating],d)
            torch._foreach_add_([old[k] for k in floating],[raw[k] for k in floating],alpha=1-d)
            for key,value in raw.items():
                if not value.is_floating_point():old[key].copy_(value)
        self.ema_state["updates"]=update
        _require(_finite(raw.values()) and _finite(old.values()),"更新后raw/EMA存在非有限状态")

    def step_update(self,microbatches,*,update):
        """实际更新；返回前不提交任何sampler。失败后禁止继续或快照。

        返回的每micro梯度范数是到该micro为止累计梯度；只在16图齐后统一clip一次。
        """
        _require(not self.failed,"FAIL_STOP backend不能接续；须fresh完整恢复")
        _require(type(update) is int and update==self.stage_update+1 and update<=self.stage_updates,"更新边界必须连续且不超预算")
        images=0;step_returned=False;micro=[];components=[];native=[];grads=[];uids=[];domains=[];lineage=[]
        schedule=schedule_at(update,self.stage_updates,final_stage=self.final_stage)
        try:
            groups=apply_sgd_schedule(self.optimizer,schedule)["groups"]
            self.optimizer.zero_grad(set_to_none=True)
            iterator=iter(microbatches)
            for index in range(16//self.microbatch_size):
                try:batch=next(iterator)
                except StopIteration as exc:raise ValueError("有效更新不足16图") from exc
                image=self._batch(batch);images+=len(image)
                context=torch.autocast(device_type="cuda",dtype=torch.bfloat16) if self.numeric_mode=="amp_bf16" else nullcontext()
                with context:
                    output=self.network.training_heads(image)
                    returned=self.criterion(output,batch)
                    loss=criterion_backward_loss(returned)
                loss["loss"].backward()
                components.append(torch.stack([loss["components"][name] for name in _NAMES]))
                native.append(loss["native_loss_items"])
                grads.append(_grad_norm(self._parameters).detach())
                uids.extend(batch["uids"]);domains.extend(batch["domains"]);lineage.extend(deepcopy(batch["sample_lineage"]))
                micro.append({"microbatch_index":index,"images":len(image),"uids":list(batch["uids"]),"domains":list(batch["domains"]),
                              "sample_lineage":deepcopy(batch["sample_lineage"]),"lr":schedule["lr"],"momentum":schedule["momentum"]})
            try:next(iterator)
            except StopIteration:pass
            else:raise ValueError("一次更新多于16图，不偷偷消费下一批")
            _require(images==16,"有效batch必须恰为16")
            before=torch.nn.utils.clip_grad_norm_(self._parameters,10.,error_if_nonfinite=True)
            after=_grad_norm(self._parameters)
            self.optimizer_steps_attempted+=1
            self.optimizer.step();step_returned=True;self.optimizer_steps_returned+=1
            self._ema_update(update)
            self.stage_update=update
            c=torch.stack(components).detach().to("cpu");n=torch.stack(native).detach().to("cpu")
            g=torch.stack(grads).detach().to("cpu")
            for index,record in enumerate(micro):
                record.update(backward_components={key:float(c[index,j]) for j,key in enumerate(_NAMES)},
                              native_loss_items={key:float(n[index,j]) for j,key in enumerate(_NAMES)},
                              gradient_norm_cumulative=float(g[index]))
            summed=c.sum(0);numbers={key:float(summed[j]) for j,key in enumerate(_NAMES)}
            return {"schema_version":_SCHEMA,"update":update,"optimizer_step_returned":True,"requires_caller_sampler_commit":True,
                    "images":16,"uids":uids,"domains":domains,"sample_lineage":lineage,"microbatches":micro,
                    "loss":float(summed.sum()),"loss_components":numbers,"backward_loss":float(summed.sum()),
                    "backward_components":deepcopy(numbers),"loss_definition":"sum(native_microbatch_scaled_three_component_vectors)/16",
                    "lr":schedule["lr"],"momentum":schedule["momentum"],"parameter_groups":groups,
                    "gradient_norm_before_clip":float(before.detach().cpu()),"gradient_norm_after_clip":float(after.detach().cpu()),
                    "ema_updates":self.ema_state["updates"],"numeric_mode":self.numeric_mode,"scaler":None}
        except BaseException as exc:
            self.failed=True
            self.failure={"type":type(exc).__name__,"message":str(exc),"requested_update":update,
                          "consumed_images":images,"optimizer_step_returned_not_committed":step_returned,
                          "optimizer_steps_attempted":self.optimizer_steps_attempted,"optimizer_steps_returned":self.optimizer_steps_returned}
            raise

    def ema_network_for_eval(self):
        _require(not self.failed,"失败backend不可评价")
        model=deepcopy(self.model)
        model.load_state_dict(self.ema_state["state_dict"],strict=True)
        model.eval();model.requires_grad_(False)
        # 原生wrapper要求DFL冻结；EMA本身不是新公开起点，用其独立来源链而非冒称验证。
        return NativeDetectionNetwork(model,device=str(self.device),allow_initialized_cuda=self.device.type=="cuda",
            initialization_binding={"kind":"stage_ema_evaluation","public_pretraining_verified":False,
                                    "origin":deepcopy(self.network.initialization_binding),"ema_updates":self.stage_update})

    def snapshot(self,*,samplers,source_bindings,committed_updates,cumulative_exposures,phase=None,extra_state=None):
        _require(not self.failed,"FAIL_STOP不能将部分更新伪装成完整检查点")
        phase=phase or ("final" if self.final_stage else "first")
        _require(phase==("final" if self.final_stage else "first"),"阶段与backend固定配置不符")
        extra=deepcopy(extra_state or {})
        _require("runtime_backend" not in extra,"extra_state不能覆盖backend配置")
        extra["runtime_backend"]=self._config()
        cuda=self.device.type=="cuda"
        return snapshot_checkpoint(raw_state=self.model.state_dict(),ema_state=self.ema_state,
            optimizer_state=self.optimizer.state_dict(),phase=phase,phase_updates=self.stage_update,
            committed_updates=committed_updates,source_bindings=source_bindings,numeric_mode=self.numeric_mode,
            scaler_state=None,samplers=samplers,cumulative_exposures=cumulative_exposures,extra_state=extra,
            allow_cuda=cuda,cuda_initialized=cuda,include_cuda_rng=cuda)

    def restore_bundle(self,bundle,*,samplers,expected_bindings,expected_metadata=None):
        """仅fresh backend恢复全部状态；不恢复活动/失败实例。RNG严格最后恢复。

        caller须持有可信checkpoint收据及run日志边界；expected_metadata省略时只核
        bundle内部metadata一致性，不声称独立认证历史提交。loader Generator等extra
        返回caller，须在派发下一数据窗口前还原，不得消费全局RNG后再当作已续接。
        """
        _require(not self.failed and self.stage_update==0 and not self.optimizer.state,"只能fresh backend做完整恢复")
        cuda=self.device.type=="cuda";before=capture_rng_state(include_cuda=cuda)
        try:
            expected_metadata=expected_metadata or {key:deepcopy(bundle[key]) for key in _METADATA}
            payload=prepare_restore_payload(bundle,expected_bindings=expected_bindings,expected_metadata=expected_metadata,
                expected_samplers=samplers,allow_cuda=cuda,cuda_initialized=cuda)
            _require(payload.get("runtime_backend")==self._config(),"backend阶段/数值/micro/输入/起点配置与检查点不符")
            _require(payload["numeric_mode"]==self.numeric_mode and payload["phase"]==("final" if self.final_stage else "first"),"恢复数值或阶段不同")
            _optimizer_topology(self.optimizer,self.model,payload["optimizer"])
            self.model.load_state_dict(payload["raw_model"],strict=True,assign=False)
            self.optimizer.load_state_dict(_clone_cpu(payload["optimizer"]))
            _check_optimizer(self.optimizer,torch)
            self.ema_state={"schema_version":"futsalmot_ema_state_v1","updates":payload["ema"]["updates"],
                            "state_dict":{k:v.to(device=self.device,copy=True) for k,v in payload["ema"]["state_dict"].items()}}
            self.stage_update=payload["phase_updates"]
            self.optimizer_steps_attempted=self.optimizer_steps_returned=self.stage_update
            for name,sampler in samplers.items():sampler.load_state_dict(payload["sampler"][name])
            self.model.train(True)
            restore_rng_state(payload["rng"],allow_cuda=cuda)
            excluded={"raw_model","ema","optimizer","rng","sampler",*_METADATA,"schema_version","snapshot_schema_version","source_bindings","scaler"}
            return {"phase_updates":self.stage_update,"committed_updates":payload["committed_updates"],
                    "cumulative_exposures":deepcopy(payload["cumulative_exposures"]),
                    "extra_state":{k:_clone_cpu(v) for k,v in payload.items() if k not in excluded},
                    "optimizer_restored":True,"samplers_restored":True,"RNG_restored_last":True}
        except BaseException:
            self.failed=True
            restore_rng_state(before,allow_cuda=cuda)
            raise
