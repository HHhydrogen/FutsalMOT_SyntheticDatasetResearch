"""正式合成输入/首阶段/完整续接的有界机械准入；导入不初始化CUDA。

本函数不选择超参、不计算val/test成绩，也不自动派发正式训练。FP32/micro8/workers2
直接继承已测R协议。PASS只补足S输入与首阶段机械缺口，容量和有限作业准入由外层负责。
"""
from __future__ import annotations
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from .preflight import _safe_path, _write_once, _same

GIB = 1024**3
S_READER_OPTIONS = {"max_indexed_cameras":7,"max_indexed_frames":3000}
SCHEMA = "formal_bbox_s_input_admission_v1"
U_S, U_R = 30187, 6563


def representative_requests(catalogue):
    """每种已存在train机位一张，不按标注大小或模型分数筛选。"""
    cameras={}
    for uid, ref in catalogue.items():
        if ref.get("domain") != "S" or ref.get("split") != "train":
            raise ValueError("代表样本只能来自完整S train catalogue")
        cameras.setdefault(ref["camera_id"],uid)
    if not {"C1","C2","C3","C4","C5"} <= set(cameras):
        raise ValueError("S train缺少五个必选机位类型")
    return [{"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,
             "exposure_index":i,"augmentation_seed":17+i} for i,uid in enumerate(cameras.values())]


def _validate_prior(value):
    if value.get("status")!="PASS":raise ValueError("原R数值准入未PASS")
    if (value.get("numeric_mode"),value.get("microbatch_size"),value.get("num_workers"))!=("fp32",8,2):
        raise ValueError("原R准入不是既定FP32/micro8/workers2")
    fraction=value.get("cuda_memory_fraction")
    if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not 0<fraction<=.8:
        raise ValueError("原R实际allocator额度缺失")


def preflight_formal(group_dir, synthetic_index, real_index, weights_dir, *, prior_admission,
                     gpu_release_confirmed=False, max_seconds=600):
    """唯一admission.json；实测失败留FAIL_STOP，不覆盖/重放，不派发训练。

    外层须已持有同PID的research_gpu锁并核生产退出；本函数再次核实际锁/生产进程。
    诊断权重仅TemporaryDirectory，打印其测量后精确删除；原始标量steps与小收据留存。
    600s为每微批/阶段前后的协作截止，不强杀内核。调用必须在spawn安全的主入口。
    """
    if type(max_seconds) is not int or not 1<=max_seconds<=600:raise ValueError("max_seconds须1..600整数")
    root=_safe_path(group_dir);root.mkdir(parents=True,exist_ok=True);target=root/"admission.json"
    if target.exists():raise FileExistsError("已有正式输入准入收据，禁止覆盖/自动重放")
    started=time.monotonic();backend=None;torch=None
    result={"schema_version":SCHEMA,"status":"RUNNING","scope":"YOLO11s_S_train_first_stage_and_resume_only",
            "numeric_mode":"fp32","microbatch_size":8,"num_workers":2,"input_size":960,"effective_batch":16,
            "S_reader_options":deepcopy(S_READER_OPTIONS),"diagnostic_updates":0,"source_bindings":{},
            "measurements":{},"checkpoint_bytes":None,"raw_upper_per_image_bytes":None,
            "test_read":False,"validation_scored":False,"formal_jobs_admitted":False,
            "deadline_seconds":max_seconds,"deadline_semantics":"cooperative_microbatch_boundary_no_kernel_kill"}
    def check():
        if time.monotonic()-started>=max_seconds:raise TimeoutError("正式输入预检有界时间已耗尽")
    def emit(value):
        payload=(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n").encode()
        with (root/"preflight_steps.jsonl").open("ab") as f:f.write(payload);f.flush();os.fsync(f.fileno())
        print(json.dumps({k:value[k] for k in ("kind","phase","update","domain","seconds") if k in value},ensure_ascii=False),flush=True)
    def clean_backend():
        nonlocal backend
        backend=None;gc.collect()
        if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()
    try:
        if gpu_release_confirmed is not True:raise RuntimeError("gpu_release_confirmed=True须由caller实际核验")
        if os.environ.get("YOLO_OFFLINE","").lower()!="true":raise RuntimeError("YOLO_OFFLINE=true required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_WORKSPACE_CONFIG必须固定:4096:8")
        from .runner import RESEARCH, production_closed
        import psutil
        lock=json.loads((RESEARCH/".research_gpu.lock").read_bytes())
        if lock["pid"]!=os.getpid() or abs(lock["create_time"]-psutil.Process().create_time())>.001:
            raise RuntimeError("预检须持有当前PID精确身份的GPU锁")
        production_closed()
        old_path=_safe_path(prior_admission);old_bytes=old_path.read_bytes();old=json.loads(old_bytes);_validate_prior(old)
        result["prior_admission"]={"path":str(old_path),"sha256":hashlib.sha256(old_bytes).hexdigest()}
        result["cuda_memory_fraction"]=old["cuda_memory_fraction"]
        result["raw_upper_per_image_bytes"]=old["raw_upper_per_image_bytes"]
        if shutil.disk_usage(root).free<101*GIB or shutil.disk_usage(RESEARCH).free<100*GIB+32*1024**2:
            raise RuntimeError("预检D/G100GiB余量或诊断1GiB容量不足")
        import torch as torch_module
        torch=torch_module
        if torch.cuda.is_initialized():raise RuntimeError("须独占新进程做S输入预检，不复用已初始化CUDA")
        from .catalogues import synthetic_bbox_catalogue
        from .runner_data import real_panel, RequestPlanDataset, make_loader
        from .runtime_backend import NativeBBoxBackend
        from .training import prepare_detection_batch
        from .data import CycleSampler, PrefetchSampler
        from .records import save_checkpoint,load_checkpoint
        from .runner import setup_cuda,seed_all,_shutdown
        from .formal import transition_backend
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
        setup_cuda(old);seed_all(17)
        synthetic=synthetic_bbox_catalogue(synthetic_index,split="train");real=real_panel(real_index)
        if len(synthetic["ordered_uids"])!=24149 or len(real["train_uids"])!=2100 or len(real["val_uids"])!=750:
            raise ValueError("既定NR/NS/Nval不符，不改池绕过")
        result["source_bindings"]={"real_panel":real["binding"],"synthetic_panel":synthetic["binding"]}
        check()
        reps=representative_requests(synthetic["catalogue"])
        repdata=RequestPlanDataset(synthetic["catalogue"],reps,view="train",synthetic_reader_options=S_READER_OPTIONS)
        represent=[]
        for i,req in enumerate(reps):
            check();sample=repdata[i]
            if tuple(sample["geometry"]["original_shape"])!=(2160,3840):raise ValueError("S原图非3840x2160")
            represent.append({"uid":req["uid"],"camera":synthetic["catalogue"][req["uid"]]["camera_id"],
                              "original_shape":[2160,3840],"targets_after_augmentation":len(sample["targets"])})
        result["measurements"]["representative_cameras"]=represent
        del repdata,sample
        check()
        backend=NativeBBoxBackend.from_public(weights_dir,model_size="s",stage_updates=U_S,final_stage=False,
            microbatch_size=8,numeric_mode="fp32",device="cuda:0",input_size=960,allow_initialized_cuda=True)
        result["source_bindings"]["public_initialization"]=deepcopy(backend.network.initialization_binding)
        bindings=deepcopy(result["source_bindings"])
        def sampler(domain):
            uids=synthetic["ordered_uids"] if domain=="S" else real["train_uids"]
            return PrefetchSampler(CycleSampler(uids,task="detect",domain=domain,seed=17))
        live=sampler("S");generator=torch.Generator(device="cpu").manual_seed(17)
        catalogue={**real["train_catalogue"],**synthetic["catalogue"]}
        def dataset(requests):return RequestPlanDataset(catalogue,requests,view="train",synthetic_reader_options=S_READER_OPTIONS,
            real_reader_options={"max_cached_clips":20,"max_cached_rows":600000})
        def prepared(iterator):
            for _ in range(2):
                check();yield prepare_detection_batch(next(iterator),device="cuda:0",non_blocking=True);check()
        def actual_step(update,stream,domain):
            reservation=stream.reserve(16);rows=deepcopy(reservation["samples"])
            for req in rows:req.update(stream=domain,data_domain="S" if domain=="S" else "R")
            loader=make_loader(dataset(rows),batch_size=8,num_workers=2,generator=generator,pin_memory=True,allow_initialized_cuda=True)
            iterator=iter(loader);before=time.monotonic()
            try:
                out=backend.step_update(prepared(iterator),update=update)
                if out["uids"]!=[x["uid"] for x in rows]:raise RuntimeError("实际worker UID顺序不一致")
                stream.commit(reservation["token"])
                torch.cuda.synchronize();elapsed=time.monotonic()-before;result["diagnostic_updates"]+=1
                emit({"kind":"diagnostic_update","phase":"final" if backend.final_stage else "first","update":update,
                      "domain":domain,"seconds":elapsed,"output":out})
                check();return out
            finally:_shutdown(iterator)
        with tempfile.TemporaryDirectory(prefix=".formal-preflight-",dir=root) as temporary:
            temporary=Path(temporary);checkpoints=[]
            def snap(update):
                check();before=time.monotonic()
                state=backend.snapshot(samplers={"S":live},source_bindings=bindings,committed_updates=update,
                    cumulative_exposures={"R":0,"S":update*16},extra_state={"loader_generator_state":generator.get_state()})
                snapshot_s=time.monotonic()-before;before=time.monotonic()
                receipt=save_checkpoint(temporary/f"diagnostic_{len(checkpoints)}.pt",state)
                write_s=time.monotonic()-before;before=time.monotonic();loaded=load_checkpoint(receipt,expected_bindings=bindings)
                checkpoints.append({"update":update,"bytes":receipt.bytes,"sha256":receipt.sha256,
                    "snapshot_seconds":snapshot_s,"write_seconds":write_s,"read_seconds":time.monotonic()-before})
                check();return loaded
            snap(0);out1=actual_step(1,live,"S");state1=snap(1);out2=actual_step(2,live,"S");state2=snap(2)
            next_rows=live.reserve(16)["samples"];live.cancel_pending();clean_backend()
            backend=NativeBBoxBackend.from_public(weights_dir,model_size="s",stage_updates=U_S,final_stage=False,
                microbatch_size=8,numeric_mode="fp32",device="cuda:0",input_size=960,allow_initialized_cuda=True)
            live=sampler("S");restored=backend.restore_bundle(state1,samplers={"S":live},expected_bindings=bindings)
            generator.set_state(restored["extra_state"]["loader_generator_state"])
            replay=actual_step(2,live,"S")
            state2_replay=backend.snapshot(samplers={"S":live},source_bindings=bindings,committed_updates=2,
                cumulative_exposures={"R":0,"S":32},extra_state={"loader_generator_state":generator.get_state()})
            equality={key:_same(state2[key],state2_replay[key],torch) for key in
                      ("raw_model","ema","optimizer","rng","sampler","loader_generator_state")}
            equality["next_sampler_requests"]=_same(next_rows,live.reserve(16)["samples"],torch);live.cancel_pending()
            equality["next_update_loss_and_gradient"]=(out2["loss"]==replay["loss"] and out2["gradient_norm_before_clip"]==replay["gradient_norm_before_clip"])
            result["measurements"]["exact_restore"]=equality
            if not all(equality.values()):raise RuntimeError("S first-stage全状态恢复后下一更新不精确")
            result["checkpoint_bytes"]=max(x["bytes"] for x in checkpoints)
            result["measurements"]["checkpoints"]=checkpoints
            del state1,state2,state2_replay,restored
            # 单次persistent-worker窗口真实测量，避免每更新冷启worker被当作正式吞吐。
            clone=CycleSampler(synthetic["ordered_uids"],task="detect",domain="S",seed=17)
            clone.load_state_dict(live.state_dict()["sampler"]);rows=clone.take(8*16)
            for req in rows:req.update(stream="S",data_domain="S")
            loader=make_loader(dataset(rows),batch_size=8,num_workers=2,generator=generator,pin_memory=True,allow_initialized_cuda=True)
            iterator=iter(loader);times=[]
            try:
                for update in range(3,11):
                    reservation=live.reserve(16);before=time.monotonic();out=backend.step_update(prepared(iterator),update=update)
                    if out["uids"]!=[x["uid"] for x in reservation["samples"]]:raise RuntimeError("窗口请求和实际更新不符")
                    live.commit(reservation["token"]);torch.cuda.synchronize();elapsed=time.monotonic()-before;times.append(elapsed)
                    result["diagnostic_updates"]+=1;emit({"kind":"timed_diagnostic_update","phase":"first","update":update,"seconds":elapsed,"output":out});check()
            finally:_shutdown(iterator)
            result["measurements"]["S_loader_window"]={"updates":8,"seconds":times,"warm_first_excluded_mean_seconds":sum(times[1:])/7,
                "includes_decode_augment_transfer_backward_optimizer_and_sync":True,"workers":2,"prefetch_factor":1}
            # 实际正式转换函数承接raw，确认SGD/EMA独立重建，并在真实R输入上更新。
            raw={k:v.detach().cpu().clone() for k,v in backend.model.state_dict().items()};old_optimizer=backend.optimizer
            backend=transition_backend(backend,u_r=U_R)
            reset={"raw_exact":all(torch.equal(v,backend.model.state_dict()[k].detach().cpu()) for k,v in raw.items()),
                   "new_optimizer":backend.optimizer is not old_optimizer,"empty_optimizer_state":not bool(backend.optimizer.state),
                   "EMA_update_zero":backend.ema_state["updates"]==0,
                   "EMA_equals_raw":all(torch.equal(v,backend.ema_state["state_dict"][k].detach().cpu()) for k,v in raw.items())}
            if not all(reset.values()):raise RuntimeError("首阶段到R阶段重建改变raw或继承旧状态")
            actual_step(1,sampler("R"),"R");result["measurements"]["transition_to_R"]=reset;del raw,old_optimizer
            free,total=torch.cuda.mem_get_info()
            result["measurements"]["memory"]={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"free_bytes":free,"total_bytes":total,
                "process_RSS_bytes":psutil.Process().memory_info().rss}
            if free<GIB or total-torch.cuda.max_memory_reserved()<GIB:raise RuntimeError("实际显存不足1GiB余量")
            result["measurements"]["step_record_max_bytes"]=max(len(json.dumps(out1).encode()),len(json.dumps(out2).encode()))
            result["temporary_artifacts"]={"full_diagnostic_models":len(checkpoints),"bytes":sum(x["bytes"] for x in checkpoints),
                "lifecycle":"printed_then_exact_TemporaryDirectory_cleanup_not_formal_models"}
            print(json.dumps({"temporary_artifacts":result["temporary_artifacts"]},ensure_ascii=False),flush=True)
        check()
        current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
        if current!=code:raise RuntimeError("预检期间源码变化，不能签测量准入")
        result["executed_code_sha256"]=code;result["status"]="PASS";result["elapsed_seconds"]=time.monotonic()-started
        _write_once(target,result);return result
    except BaseException as exc:
        result.update(status="FAIL_STOP",elapsed_seconds=time.monotonic()-started,
                      failure={"type":type(exc).__name__,"message":str(exc)})
        if not target.exists():_write_once(target,result)
        raise
    finally:clean_backend()
