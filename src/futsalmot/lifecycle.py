"""有限CPU_fixture_no_parameter_updates生命周期夹具驱动，不是YOLO Trainer。

只准系统临时根下的微小CPU模块/自造数据，不接正式run或训练入口。假backend的
success只表示fixture控制流；真实forward/backward/optimizer.step均为0。阶段/记录/
检查点计数保留相同结构并逐项标nonformal，不可作为任何真实模型实验。
没有单run跨进程锁、崩溃事务或自动恢复；任何部分提交失败保持证据并立即STOP。
"""
from __future__ import annotations
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import traceback

import torch

from .data import PrefetchSampler
from .training import plan,update_spec,schedule_at,initialize_ema_stage,update_ema_state
from .optimizers import make_sgd_optimizer,apply_sgd_schedule
from .phases import start_final_phase
from .loaders import ReservationDataLoader,loader_generator_state
from .execution import _reservation_requests,MICROBATCH_SIZES
from .step_records import trace_microbatch,build_update_record
from .records import RunRecorder,save_checkpoint
from .checkpoint_state import snapshot_checkpoint,_clone_cpu

SCOPE="CPU_fixture_no_parameter_updates"


class FixtureBackendFailure(RuntimeError):pass


class CPUFixtureBackend:
    """显式注入fixture测量/成功失败，仅接纯metadata，不接model/optimizer。"""
    scope=SCOPE
    def __init__(self,measurements,*,fail_update=None,cancel_update=None):
        if not isinstance(measurements,dict) and not callable(measurements):
            raise ValueError("必须显式提供fixture测量字典或callback")
        self.measurements=deepcopy(measurements) if isinstance(measurements,dict) else measurements
        self.fail_update=fail_update;self.cancel_update=cancel_update

    def run_update(self,context,traces):
        if context["update"] == self.cancel_update:raise KeyboardInterrupt("fixture人工取消")
        if context["update"] == self.fail_update:return {"scope":SCOPE,"success":False,"actual_parameter_updates":0}
        value=self.measurements(deepcopy(context),deepcopy(traces)) if callable(self.measurements) else deepcopy(self.measurements)
        return {"scope":SCOPE,"success":True,"actual_parameter_updates":0,"measurements":value}


def _temporary(path):
    target=Path(os.path.abspath(path));root=Path(tempfile.gettempdir()).resolve()
    if target == root or not target.is_relative_to(root) or not target.resolve().is_relative_to(root):
        raise ValueError("fixture只允许系统临时根下目录，不创建正式run")
    return target


def _unchanged(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(_unchanged(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(_unchanged(x,y) for x,y in zip(a,b))
    return a==b


def _fixture_lineage(trace):
    for item in trace["sample_lineage"]:
        ref=item["source_reference"]
        if ref["split"] != "train":raise ValueError("fixture更新不读val/dev/test")
        for path in (item["image_source_binding"]["path"],ref.get("index_path",ref.get("seed_dir"))):
            _temporary(path) # 仅路径检查，不读媒体，不把metadata loader当真实GT验收。


def _evaluation_role(phase_updates,reasons):
    # 仅修正标签：中间1000点也用当前EMA，但不是阶段末步模型。
    return ("stage0_reference" if phase_updates==0 else
            "terminal_EMA" if "stage_final" in reasons else "EMA_at_update")


def run_cpu_fixture(*,directory,raw_model,samplers,strategy,training_seed,n_s,n_r,epochs_s,epochs_r,
                    microbatch_size,generator,backend,source_bindings,max_updates,evaluation_basis=None,
                    dataset=None,loader_factory=None,evaluation_callback=None):
    """执行单个有限fixture，不重试/扩量；只有caller显式提供global/phase评价basis。

    loader_factory(reservation,microbatch_size,generator)可为长计数返回metadata微批，
    其trace仍走完整16图规范；默认真实ReservationDataLoader0worker。backend不做
    参数更新，LR/momentum在消费微批/进入backend前设置；EMA只在假success后推进。
    caller仍负责单进程独占和可信fixture callbacks。本函数不提供resume。
    """
    directory=_temporary(directory)
    if type(microbatch_size) is not int or microbatch_size not in MICROBATCH_SIZES:
        raise ValueError("microbatch_size须与现有1/2/4/8/16契约一致")
    loader_generator_state(generator) # 显式专属CPU Generator，坏参数不创建run或派发。
    if evaluation_basis not in ("global","phase"):
        raise ValueError("必须明确候选1000评价basis=global或phase，不默认正式定义")
    if type(backend) is not CPUFixtureBackend or backend.scope != SCOPE:
        raise ValueError("只允许明确CPU_fixture_no_parameter_updates backend，禁止真实训练backend")
    if training_seed not in (17,29,43) or type(training_seed) is not int:
        raise ValueError("fixture seed只接既定17/29/43")
    budget=plan(n_s,n_r,epochs_s=epochs_s,epochs_r=epochs_r)
    if strategy not in budget["strategies"]:raise ValueError("未知策略")
    total=budget["strategies"][strategy]["updates"];u_s,u_r=budget["U_S"],budget["U_R"]
    if type(max_updates) is not int or not total <= max_updates <= 2048:
        raise ValueError("caller有限fixture上限必须覆盖本run且不超过2048")
    required={"R"} if strategy=="R" else {"S"} if strategy=="S" else {"R","R_extra"} if strategy=="R->R" else {"R","S"}
    if not isinstance(samplers,dict) or set(samplers) != required:raise ValueError("实际sampler流与既定策略不符")
    for stream,q in samplers.items():
        if type(q) is not PrefetchSampler or q.pending_batches or q.committed_draws:
            raise ValueError("fresh单run fixture需要无pending/零提交的真实sampler")
        state=q.state_dict()["sampler"]
        if state["seed"] != training_seed or state["task"] != "detect" or state["domain"] != stream:
            raise ValueError("sampler身份/seed与run不一致")
        if len(q._template[0]) != (n_s if stream=="S" else n_r):raise ValueError("预算N与显式UID池数量不一致")
    if "R_extra" in samplers and samplers["R"].state_dict()["sampler"]["pool_sha256"] != samplers["R_extra"].state_dict()["sampler"]["pool_sha256"]:
        raise ValueError("R_extra须与R同一真实UID池")
    if not isinstance(raw_model,torch.nn.Module):raise ValueError("需要自造微小CPU Module")
    raw=raw_model.state_dict()
    if sum(v.numel() for v in raw.values())>4096 or any(v.device.type != "cpu" for v in raw.values()):
        raise ValueError("fixture仅接最多4096元素CPU state，不接真实YOLO或GPU")
    initialize_ema_stage(raw) # 复用完整有限FP32/buffer校验，不运行模型。
    if loader_factory is None:
        from .dataset import RequestBBoxDataset
        if not isinstance(dataset,RequestBBoxDataset) or dataset.view != "train":raise ValueError("默认loader需要临时train RequestBBoxDataset")
        for ref in dataset._catalogue.values():_temporary(ref.get("index_path",ref.get("seed_dir")))
    elif not callable(loader_factory):raise ValueError("loader_factory必须明确为fixture callback")
    if evaluation_callback is not None and not callable(evaluation_callback):raise ValueError("fixture评价callback非法")
    if not isinstance(source_bindings,dict) or not source_bindings:raise ValueError("缺明确fixture来源绑定")
    binding={"caller":deepcopy(source_bindings),"execution_scope":SCOPE,"nonformal":True}
    manifest={"execution_scope":SCOPE,"nonformal":True,"actual_parameter_updates":0,"strategy":strategy,
              "training_seed":training_seed,"N_S":n_s,"N_R":n_r,"epochs_S":epochs_s,"epochs_R":epochs_r,
              "U_S":u_s,"U_R":u_r,"total_fixture_updates":total,"microbatch_size":microbatch_size,
              "evaluation_basis":evaluation_basis,"checkpoint_interval":50,"evaluation_interval":1000,
              "source_bindings":binding,"loader_kind":"metadata_fixture_callback" if loader_factory else "DataLoader0worker_temporary_images"}
    recorder=RunRecorder(directory,manifest)
    (directory/"checkpoints").mkdir()
    committed=recorded=last_saved=0;counts={"R":0,"S":0};phase=update_spec(strategy,1,u_s,u_r)["phase"]
    phase_updates=0;ema=None;optimizer=None;receipts=[];saved_keys=set();current_token=None
    def event(name,**fields):recorder.event({"event":name,"execution_scope":SCOPE,"nonformal":True,"actual_parameter_updates":0,**fields})
    def checkpoint(reason):
        nonlocal last_saved
        key=(phase,phase_updates,committed)
        if key in saved_keys:return
        state=snapshot_checkpoint(raw_state=raw_model.state_dict(),ema_state=ema,optimizer_state=optimizer.state_dict(),
                                  phase=phase,phase_updates=phase_updates,committed_updates=committed,
                                  source_bindings=binding,numeric_mode="fp32",scaler_state=None,samplers=samplers,
                                  cumulative_exposures=counts,extra_state={"execution_scope":SCOPE,"nonformal":True,
                                  "actual_parameter_updates":0,"loader_generator":loader_generator_state(generator),
                                  "strategy":strategy,"training_seed":training_seed,"evaluation_basis":evaluation_basis})
        path=directory/"checkpoints"/f"{phase}_p{phase_updates:06d}_g{committed:06d}.pt"
        receipt=save_checkpoint(path,state);receipts.append(asdict(receipt));saved_keys.add(key);last_saved=committed
        event("checkpoint_saved",phase=phase,phase_updates=phase_updates,committed_updates=committed,reason=reason,receipt=asdict(receipt))
    def evaluate(reasons):
        payload={"execution_scope":SCOPE,"nonformal":True,"actual_parameter_updates":0,"phase":phase,
                 "phase_updates":phase_updates,"committed_updates":committed,"evaluation_basis":evaluation_basis,
                 "reasons":reasons,"model_role":_evaluation_role(phase_updates,reasons),
                 "ema":_clone_cpu(ema),"metrics":None}
        event("fixture_evaluation_started",**{k:v for k,v in payload.items() if k not in ("ema","execution_scope","nonformal","actual_parameter_updates")})
        if evaluation_callback is not None and evaluation_callback(payload) is not None:
            raise ValueError("fixture评价callback只能返回None，不生成真实分数")
        event("fixture_evaluation_completed",phase=phase,phase_updates=phase_updates,committed_updates=committed,
              evaluation_basis=evaluation_basis,reasons=reasons,model_role=payload["model_role"],metrics=None)
    with (directory/"stdout.log").open("x",encoding="utf8") as stdout,(directory/"stderr.log").open("x",encoding="utf8") as stderr:
        with redirect_stdout(stdout),redirect_stderr(stderr):
            try:
                event("fixture_run_started",committed_updates=0,total_fixture_updates=total,evaluation_basis=evaluation_basis)
                stage_budget=u_r if phase=="final" else u_s
                initial=schedule_at(0,stage_budget,final_stage=phase=="final")
                optimizer=make_sgd_optimizer(raw_model,lr=initial["lr"],momentum=initial["momentum"],weight_decay=.0005)
                ema=initialize_ema_stage(raw_model.state_dict());checkpoint("initial_stage0");evaluate(["PT_reference_fixture_only"])
                for update in range(1,total+1):
                    spec=update_spec(strategy,update,u_s,u_r)
                    stream=spec["domain"];q=samplers[stream]
                    group_schedule=apply_sgd_schedule(optimizer,spec)
                    reservation=q.reserve(16);current_token=reservation["token"]
                    event("update_started",update=update,phase=phase,planned_phase_update=spec["step"],
                          committed_updates=committed,recorded_updates=recorded,reservation_token=current_token,
                          schedule_applied_before_backend=True,groups=group_schedule["groups"])
                    _,_,domain,expected=_reservation_requests(reservation,microbatch_size)
                    loader=(loader_factory(deepcopy(reservation),microbatch_size,generator) if loader_factory else
                            ReservationDataLoader([reservation],dataset,microbatch_size=microbatch_size,generator=generator))
                    traces=[];offset=0
                    for output in loader:
                        trace=trace_microbatch(output);size=len(trace["uids"])
                        if size != microbatch_size or trace["reservation_token"] != current_token or trace["microbatch_offset"] != offset or trace["sample_lineage"] and [x["request"] for x in trace["sample_lineage"]] != expected[offset:offset+size]:
                            raise ValueError("loader trace与明确reservation不一致")
                        _fixture_lineage(trace);traces.append(trace);offset+=size
                        del output
                    if offset != 16:raise ValueError("loader未完整覆盖16图，立即STOP")
                    raw_before=_clone_cpu(raw_model.state_dict());opt_before=_clone_cpu(optimizer.state_dict())
                    outcome=backend.run_update({"execution_scope":SCOPE,"update":update,"phase":phase,
                                               "phase_update":spec["step"],"schedule":deepcopy(spec),"groups":deepcopy(group_schedule["groups"])},traces)
                    if not _unchanged(raw_before,raw_model.state_dict()) or not _unchanged(opt_before,optimizer.state_dict()):
                        raise ValueError("fixture backend修改了参数/optimizer，拒绝作为无更新夹具")
                    if outcome.get("scope") != SCOPE or outcome.get("actual_parameter_updates") != 0 or outcome.get("success") is not True:
                        raise FixtureBackendFailure("CPU fixture backend未成功；不commit当前token")
                    measured=outcome["measurements"]
                    if not isinstance(measured,dict) or "fixture" not in measured.get("loss_definition","").lower():
                        raise ValueError("必须显式标记fixture测量，不能冒充真实loss")
                    following=deepcopy(counts);following[domain]+=16
                    record=build_update_record(traces,update=update,phase=phase,phase_update=spec["step"],
                                               lr=optimizer.param_groups[0]["lr"],momentum=optimizer.param_groups[0]["momentum"],
                                               parameter_groups=group_schedule["groups"],previous_exposures=counts,
                                               cumulative_exposures=following,**measured)
                    record.update(execution_scope=SCOPE,nonformal=True,actual_parameter_updates=0,fixture_backend_success=True,
                                  evaluation_basis=evaluation_basis)
                    proposed=update_ema_state(ema,raw_model.state_dict(),update=spec["step"])
                    q.commit(current_token);committed=update;counts=following;phase_updates=spec["step"];ema=proposed
                    recorder.step(record);recorded=update
                    event("update_committed",committed_updates=committed,recorded_updates=recorded,phase=phase,
                          phase_updates=phase_updates,reservation_token=current_token,cumulative_exposures=counts)
                    stage_budget=u_r if phase=="final" else u_s
                    final=phase_updates==stage_budget
                    if phase_updates%50==0 or final:checkpoint("stage_final" if final else "interval50")
                    tick=committed if evaluation_basis=="global" else phase_updates
                    reasons=(["interval1000"] if tick%1000==0 else [])+(["stage_final"] if final else [])
                    if reasons:evaluate(reasons)
                    if final and committed < total:
                        reset=start_final_phase(raw_model,strategy=strategy,u_s=u_s,u_r=u_r,committed_updates=committed,
                                                current_phase=phase,phase_updates=phase_updates,samplers=samplers,
                                                cumulative_exposures=counts,current_optimizer=optimizer,current_ema=ema)
                        optimizer,ema=reset["optimizer"],reset["ema"];phase="final";phase_updates=0
                        event("phase_reset",committed_updates=committed,phase=phase,phase_updates=0,
                              sampler_positions_preserved=True,optimizer_state_entries=len(optimizer.state))
                        checkpoint("reset_stage0")
                event("fixture_run_completed",committed_updates=committed,recorded_updates=recorded,
                      actual_parameter_updates=0,cumulative_exposures=counts,complete_checkpoint_count=len(receipts))
                return {"status":"CPU_FIXTURE_COMPLETE","execution_scope":SCOPE,"nonformal":True,
                        "actual_parameter_updates":0,"fixture_committed_updates":committed,"cumulative_exposures":counts,
                        "checkpoints":receipts,"evaluation_basis":evaluation_basis,"trainer_ready":False}
            except BaseException as exc:
                traceback.print_exc(file=stderr)
                observed_draws={};observation_errors={}
                for stream,sampler in samplers.items():
                    try:
                        draw=sampler.state_dict()["sampler"]["draws"]
                        if type(draw) is not int or draw < 0:raise ValueError("未知committed draws")
                        observed_draws[stream]=draw
                    except BaseException as observation_error:
                        observed_draws[stream]=None;observation_errors[stream]=repr(observation_error)
                observed_unknown=bool(observation_errors)
                observed_exposures=(None if observed_unknown else
                                    {"R":observed_draws.get("R",0)+observed_draws.get("R_extra",0),
                                     "S":observed_draws.get("S",0)})
                observed_updates=(None if observed_unknown or sum(observed_exposures.values())%16
                                  else sum(observed_exposures.values())//16)
                stop={"event":"fixture_run_stopped","execution_scope":SCOPE,"nonformal":True,"actual_parameter_updates":0,
                      "exception_type":type(exc).__name__,"reason":str(exc),"committed_updates":committed,
                      "recorded_updates":recorded,"last_checkpoint_committed_updates":last_saved,
                      "reservation_token":current_token,"cumulative_exposures":counts,
                      "observed_sampler_draws":observed_draws,"observed_sampler_exposures":observed_exposures,
                      "observed_sampler_committed_updates":observed_updates,"observed_unknown":observed_unknown,
                      "sampler_observation_errors":observation_errors,
                      "inconsistent_boundary":observed_unknown or observed_exposures!=counts or observed_updates!=committed
                                              or committed!=recorded or last_saved!=recorded or not receipts,
                      "automatic_resume_allowed":False,"retry_performed":False}
                try:recorder.event(stop)
                except BaseException as record_error:
                    stop["event_log_append_failed"]=repr(record_error);stop["inconsistent_boundary"]=True
                    try:
                        with (directory/"STOP.json").open("xb") as f:
                            f.write((json.dumps(stop,ensure_ascii=False,allow_nan=False)+"\n").encode("utf8"));f.flush();os.fsync(f.fileno())
                    except BaseException as final_error:exc.add_note("STOP证据落盘也失败: "+repr(final_error))
                raise
            finally:
                stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
