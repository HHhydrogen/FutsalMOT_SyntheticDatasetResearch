"""有界R50准入实测；调用本函数会读取公开权重/train RGB并实际使用CUDA。

模块导入无Torch/GPU/数据副作用。只有caller已核UE/生产释放且显式传True后
才初始化CUDA。诊断更新不属于论文训练：全部临时完整state/预测在结束时删除，
只永久写一次admission.json。PASS仅覆盖s/seed17/SoccerNet/R50，不准入R100、
15/39矩阵或test。600s为协作截止：逐micro/阶段前后检查，不强杀CUDA内核。
"""
from __future__ import annotations
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

GIB = 1024 ** 3
SCHEMA = "r50_calibration_admission_v1"


def _safe_path(path):
    p = Path(path)
    if ".." in p.parts or str(path).startswith(("\\\\", "//")):
        raise ValueError("拒绝越界/UNC路径")
    p = p.absolute()
    for q in (p, *p.parents):
        if q.exists() or q.is_symlink():
            s = q.lstat()
            if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 0x400:
                raise ValueError("拒绝reparse路径")
    return p


def _write_once(path, result):
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _storage_budget(checkpoint_bytes, raw_upper, *, n_train, n_val):
    if any(type(v) is not int or v <= 0 for v in (checkpoint_bytes, raw_upper, n_train, n_val)):
        raise ValueError("容量预算需要实际正整数测量/图数")
    updates = (n_train * 50 + 15) // 16
    checkpoints = len({0, updates, *range(50, updates + 1, 50)})
    val_passes = 2 + updates // 1000
    # PT+每1000+final，若final恰好1000倍数不重复。
    if updates % 1000 == 0:
        val_passes -= 1
    required = math.ceil(checkpoint_bytes * checkpoints * 1.1) + raw_upper * n_val * val_passes + 9 * GIB
    return {"updates": updates, "checkpoints": checkpoints, "validation_passes": val_passes,
            "prediction_images": n_val * val_passes, "required_bytes": required,
            "checkpoint_margin": 1.1, "logs_bytes": 8 * GIB, "transient_bytes": GIB,
            "minimum_free_after_budget_bytes": 100 * GIB}


def _same(a, b, torch):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu()))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, torch) for x, y in zip(a, b))
    return a == b


def _numeric_comparison(reference_loss, candidate_loss, reference_gradient, candidate_gradient, torch):
    if (reference_gradient.shape != candidate_gradient.shape or reference_gradient.numel() == 0
            or not torch.isfinite(reference_gradient).all() or not torch.isfinite(candidate_gradient).all()
            or not math.isfinite(reference_loss) or not math.isfinite(candidate_loss)):
        raise ValueError("数值比较需要同shape有限实际loss/gradient")
    rn = float(torch.linalg.vector_norm(reference_gradient))
    cn = float(torch.linalg.vector_norm(candidate_gradient))
    if rn <= 0 or cn <= 0:
        raise ValueError("梯度范数必须正数，不能以零梯度签准入")
    relative = abs(candidate_loss - reference_loss) / max(abs(reference_loss), 1e-12)
    cosine = float(torch.dot(reference_gradient / rn, candidate_gradient / cn))
    norm_relative = abs(cn - rn) / rn
    return {"reference_loss": reference_loss, "candidate_loss": candidate_loss,
            "loss_relative_error": relative, "gradient_cosine": cosine,
            "reference_gradient_norm": rn, "candidate_gradient_norm": cn,
            "gradient_norm_relative_error": norm_relative,
            "accepted": relative <= .05 and cosine >= .98 and norm_relative <= .15,
            "threshold_scope": "engineering_numeric_gate_not_paper_noninferiority"}


def preflight_calibration(group_dir, index_path, weights_dir, *,
                          gpu_release_confirmed=False, max_seconds=600):
    """返回并exclusive写group/admission.json；失败留FAIL_STOP后重抛。

    gpu_release_confirmed是caller持有生产/UE/训练独占锁后的明确声明，本函数
    不替caller操作UE/进程/监控。仅真实测量可签PASS；不读取test、不算val成绩。
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 600:
        raise ValueError("预检总预算须为1..600秒")
    root = _safe_path(group_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "admission.json"
    if target.exists():
        raise FileExistsError("已有准入收据禁止覆盖/自动重放")
    executed_code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}
    started = time.monotonic()
    result = {"schema_version": SCHEMA, "status": "RUNNING", "scope": "YOLO11s_seed17_SoccerNet_R50_only",
              "diagnostic_updates": 0, "diagnostic_backward_batches": 0,
              "measurements": {"microbatch_trials": [], "numeric_comparisons": [], "worker_trials": []},
              "numeric_mode": None, "microbatch_size": None, "num_workers": None,
              "checkpoint_bytes": None, "raw_upper_per_image_bytes": None,
              "R50_required_bytes": None, "R50_max_run_seconds": None,
              "deadline_seconds": max_seconds,
              "deadline_semantics": "cooperative_before_after_each_microbatch_and_stage_no_kernel_kill",
              "source_bindings": {}, "test_read": False, "validation_scored": False}
    torch = None
    backend = None

    def check():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("预检600s以内声明预算已耗尽，不签准入PASS")

    def clean():
        nonlocal backend
        backend = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    try:
        if gpu_release_confirmed is not True:
            raise RuntimeError("BLOCKED: caller须先核UE/生产释放并显式声明gpu_release_confirmed=True")
        if os.environ.get("YOLO_OFFLINE", "").lower() != "true":
            raise RuntimeError("BLOCKED: caller须在首次原生导入前设置YOLO_OFFLINE=true")
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in (None, ":4096:8"):
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG与固定:4096:8冲突")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import random
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from .runner_data import real_panel, RequestPlanDataset, make_loader
        from .data import CycleSampler, PrefetchSampler
        from .batches import collate_bbox_samples
        from .training import prepare_detection_batch, criterion_backward_loss
        from .pretrained import load_local_yolo11_detect
        from .runtime_backend import NativeBBoxBackend
        from .records import save_checkpoint, load_checkpoint

        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA已被其它执行上下文初始化；要求本次独占新预检")
        if shutil.disk_usage(root).free < 101 * GIB:
            raise RuntimeError("诊断开始前D盘不足100GiB保留加1GiB临时预算")
        torch.cuda.init()
        free_at_start,total_memory=torch.cuda.mem_get_info()
        memory_fraction=min(.80,(free_at_start-int(1.5*GIB))/total_memory)
        if memory_fraction<=0 or total_memory*memory_fraction<4*GIB:raise RuntimeError("不足安全CUDA分配额度")
        torch.cuda.set_per_process_memory_fraction(memory_fraction,device=0)
        result["cuda_memory_fraction"]=memory_fraction
        result["cuda_allocator_limit_bytes"]=int(total_memory*memory_fraction)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(4)

        def seed():
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)

        seed()
        panel = real_panel(index_path)
        if len(panel["train_uids"]) != 2100 or len(panel["val_uids"]) != 750:
            raise ValueError("首SoccerNet已冻结2100train/750val池不符")
        if any(row.get("ignore") for row in panel["val_ground_truth"].values()):
            raise ValueError("标准COCO首面板存在真实ignore，不能冒充crowd")
        check()
        template = load_local_yolo11_detect(weights_dir, model_size="s")
        bindings = {"real_panel": panel["binding"], "model": template.initialization_binding,
                    "input": 960, "effective_batch": 16, "seed": 17}
        result["source_bindings"] = deepcopy(bindings)
        result["measurements"]["device"] = {
            "name": torch.cuda.get_device_name(0), "total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "deterministic_algorithms": True, "TF32": False,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        updates = (len(panel["train_uids"]) * 50 + 15) // 16
        requests = CycleSampler(panel["train_uids"], task="detect", domain="R", seed=17).take(64)
        for request in requests:
            request.update(stream="R", data_domain="R")
        dataset = RequestPlanDataset(panel["train_catalogue"], requests, view="train",
                                     real_reader_options={"max_cached_clips": 20, "max_cached_rows": 600_000})
        # 仅固定前三个有效batch的CPU输入复用；不缓存完整训练池RGB。
        cpu_inputs = [dataset[i] for i in range(48)]
        result["measurements"]["numeric_request_uids"] = [r["uid"] for r in requests[:48]]

        def factory(mb, mode):
            check()
            seed()
            return NativeBBoxBackend(deepcopy(template), stage_updates=updates,
                                     microbatch_size=mb, numeric_mode=mode, device="cuda:0",
                                     allow_initialized_cuda=True, input_size=960)

        def prepared(samples, mb):
            for offset in range(0, 16, mb):
                check()
                batch = collate_bbox_samples(samples[offset:offset + mb])
                yield prepare_detection_batch(batch, device="cuda:0", non_blocking=False)
                check()

        def step(samples, update):
            check()
            torch.cuda.synchronize()
            before = time.monotonic()
            output = backend.step_update(prepared(samples, backend.microbatch_size), update=update)
            torch.cuda.synchronize()
            result["diagnostic_updates"] += 1
            elapsed = time.monotonic() - before
            check()
            return elapsed, output

        def memory():
            free, total = torch.cuda.mem_get_info()
            return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "free_bytes": free, "total_bytes": total,
                    "safe_vram": free >= GIB and total - torch.cuda.max_memory_reserved() >= GIB}

        def trials(mode):
            nonlocal backend
            accepted = []
            for mb in (1, 2, 4, 8, 16):
                clean()
                check()
                row = {"numeric_mode": mode, "microbatch_size": mb, "warmup_updates": 1,
                       "timed_updates": 2, "status": "STARTED"}
                result["measurements"]["microbatch_trials"].append(row)
                print(json.dumps({"stage":"microbatch_begin","mode":mode,"microbatch":mb}),flush=True)
                try:
                    backend = factory(mb, mode)
                    torch.cuda.reset_peak_memory_stats()
                    times = [step(cpu_inputs[i * 16:(i + 1) * 16], i + 1)[0] for i in range(3)]
                    row.update(status="MEASURED", update_seconds=times,
                               timed_mean_seconds=sum(times[1:]) / 2, **memory())
                    if row["safe_vram"]:
                        accepted.append(row)
                    else:
                        row["status"] = "UNSAFE_VRAM"
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status="OOM", error=str(exc), **memory())
                finally:
                    clean()
                print(json.dumps({"stage":"microbatch_end",**row}),flush=True)
                check()
            return sorted(accepted, key=lambda item: (item["timed_mean_seconds"], item["microbatch_size"]))

        def loss_gradient(samples, mb, mode):
            nonlocal backend
            clean()
            backend = factory(mb, mode)
            backend.optimizer.zero_grad(set_to_none=True)
            loss_number = 0.
            for batch in prepared(samples, mb):
                context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mode == "amp_bf16" else nullcontext()
                with context:
                    output = backend.criterion(backend.network.training_heads(batch["img"]), batch)
                    native = criterion_backward_loss(output)
                native["loss"].backward()
                loss_number += float(native["loss"].detach().cpu())
            names, vectors = [], []
            for name, parameter in backend.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    names.append(name)
                    vectors.append(parameter.grad.detach().float().cpu().reshape(-1))
            gradient = torch.cat(vectors)
            result["diagnostic_backward_batches"] += 1
            clean()
            check()
            return loss_number, gradient, names

        bf16_rows = trials("amp_bf16") if torch.cuda.is_bf16_supported() else []
        selected = None
        for row in bf16_rows:
            mb = row["microbatch_size"]
            comparisons = []
            try:
                for index in range(3):
                    samples = cpu_inputs[index * 16:(index + 1) * 16]
                    ref, rg, rn = loss_gradient(samples, mb, "fp32")
                    try:
                        value, cg, cn = loss_gradient(samples, mb, "amp_bf16")
                    except (ValueError, RuntimeError) as exc:
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            raise
                        comparisons.append({"accepted": False})
                        result["measurements"]["numeric_comparisons"].append({
                            "microbatch_size": mb, "effective_batch_ordinal": index + 1,
                            "accepted": False, "status": "BF16_NUMERIC_FAILURE", "error": str(exc)})
                        clean()
                        break
                    if rn != cn:
                        raise ValueError("两模式实际梯度参数顺序不同")
                    try:
                        compared = _numeric_comparison(ref, value, rg, cg, torch)
                    except ValueError as exc:
                        compared={"accepted":False,"status":"BF16_NONFINITE_OR_INVALID","error":str(exc)}
                    compared.update(microbatch_size=mb, effective_batch_ordinal=index + 1)
                    comparisons.append(compared)
                    result["measurements"]["numeric_comparisons"].append(compared)
                if all(item["accepted"] for item in comparisons):
                    selected = row
                    break
            except torch.cuda.OutOfMemoryError as exc:
                result["measurements"]["numeric_comparisons"].append(
                    {"microbatch_size": mb, "status": "FP32_REFERENCE_OOM", "error": str(exc)})
                clean()
        if selected is None:
            fp32_rows = trials("fp32")
            if not fp32_rows:
                raise RuntimeError("没有满足960与1GiB余量的FP32候选")
            selected = fp32_rows[0]
            result["measurements"]["numeric_fallback"] = "BF16 unsupported/unsafe/gate_failed; actual FP32 trial selected"
        mode, mb = selected["numeric_mode"], selected["microbatch_size"]
        result.update(numeric_mode=mode, microbatch_size=mb)

        # 真DataLoader端到端4次更新，首更新warmup；每候选使用相同64请求。
        worker_rows = []
        for workers in (2, 4):
            clean()
            backend = factory(mb, mode)
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device="cpu").manual_seed(17)
            loader = make_loader(dataset, batch_size=mb, num_workers=workers, generator=generator,
                                 pin_memory=True, allow_initialized_cuda=True)
            loader.timeout = max(.1, max_seconds - (time.monotonic() - started))
            iterator = None
            print(json.dumps({"stage":"worker_begin","workers":workers,"microbatch":mb,"mode":mode}),flush=True)
            row = {"num_workers": workers, "updates": 4, "warmup_updates": 1, "status": "STARTED"}
            result["measurements"]["worker_trials"].append(row)
            try:
                iterator = iter(loader)
                times = []
                for update in range(1, 5):
                    check()
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    def microbatches():
                        for _ in range(16 // mb):
                            check()
                            yield prepare_detection_batch(next(iterator), device="cuda:0", non_blocking=True)
                    out = backend.step_update(microbatches(), update=update)
                    result["diagnostic_updates"] += 1
                    if out["uids"] != [r["uid"] for r in requests[(update - 1) * 16:update * 16]]:
                        raise ValueError("真实DataLoader改变冻结请求顺序")
                    torch.cuda.synchronize()
                    times.append(time.monotonic() - before)
                    check()
                row.update(status="MEASURED", update_seconds=times,
                           timed_mean_seconds=sum(times[1:]) / 3, **memory())
                if row["safe_vram"]:
                    worker_rows.append(row)
            finally:
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
                del iterator, loader
                clean()
        if not worker_rows:
            raise RuntimeError("workers2/4均未满足实际准入")
        chosen_worker = min(worker_rows, key=lambda row: (row["timed_mean_seconds"], row["num_workers"]))
        result["num_workers"] = chosen_worker["num_workers"]

        print(json.dumps({"stage":"restore_check","mode":mode,"microbatch":mb,"workers":result["num_workers"]}),flush=True)
        with tempfile.TemporaryDirectory(prefix=".r50-preflight-", dir=root) as temporary:
            temporary = Path(temporary)
            # 完整state1→state2，与fresh restore(state1)→state2精确比较。
            backend = factory(mb, mode)
            sampler = PrefetchSampler(CycleSampler(panel["train_uids"], task="detect", domain="R", seed=17))
            loader_generator = torch.Generator(device="cpu").manual_seed(17)
            checkpoint_rows = []

            def committed_step(update, reservation):
                # 恢复链真正消耗独立loader generator，而非保存一个永远不变的占位状态。
                rows = deepcopy(reservation["samples"])
                for request in rows:
                    request.update(stream="R", data_domain="R")
                actual_dataset = RequestPlanDataset(panel["train_catalogue"], rows, view="train",
                    real_reader_options={"max_cached_clips": 20, "max_cached_rows": 600_000})
                actual_loader = make_loader(actual_dataset, batch_size=mb, num_workers=0,
                    generator=loader_generator, pin_memory=False, allow_initialized_cuda=True)
                actual_iterator = iter(actual_loader)
                def actual_microbatches():
                    for batch in actual_iterator:
                        check()
                        yield prepare_detection_batch(batch, device="cuda:0")
                outcome = backend.step_update(actual_microbatches(), update=update)
                result["diagnostic_updates"] += 1
                if outcome["uids"] != [r["uid"] for r in rows]:
                    raise ValueError("恢复验证真实DataLoader请求顺序不符")
                sampler.commit(reservation["token"])
                check()

            def snapshot_at(update):
                check()
                before = time.monotonic()
                state = backend.snapshot(samplers={"R": sampler}, source_bindings=bindings,
                    committed_updates=update, cumulative_exposures={"R": update * 16, "S": 0},
                    extra_state={"loader_generator_state": loader_generator.get_state()})
                snapshot_seconds = time.monotonic() - before
                before = time.monotonic()
                receipt = save_checkpoint(temporary / ("full_" + str(len(checkpoint_rows)) + ".pt"), state)
                write_seconds = time.monotonic() - before
                before = time.monotonic()
                loaded = load_checkpoint(receipt, expected_bindings=bindings)
                read_seconds = time.monotonic() - before
                checkpoint_rows.append({"update": update, "bytes": receipt.bytes,
                    "snapshot_seconds": snapshot_seconds, "write_seconds": write_seconds,
                    "read_seconds": read_seconds, "sha256": receipt.sha256})
                check()
                return loaded

            snapshot_at(0)
            for update in (1, 2):
                reservation = sampler.reserve(16)
                if [x["uid"] for x in reservation["samples"]] != [x["uid"] for x in requests[(update - 1) * 16:update * 16]]:
                    raise ValueError("诊断实际提交流与冻结请求不同")
                committed_step(update, reservation)
                state = snapshot_at(update)
                if update == 1:
                    state1 = state
            state2 = state
            next_request = sampler.reserve(16)["samples"]
            sampler.cancel_pending()
            clean()
            backend = factory(mb, mode)
            sampler = PrefetchSampler(CycleSampler(panel["train_uids"], task="detect", domain="R", seed=17))
            restored = backend.restore_bundle(state1, samplers={"R": sampler}, expected_bindings=bindings)
            loader_generator.set_state(restored["extra_state"]["loader_generator_state"])
            reservation = sampler.reserve(16)
            committed_step(2, reservation)
            restored_state2 = backend.snapshot(samplers={"R": sampler}, source_bindings=bindings,
                committed_updates=2, cumulative_exposures={"R": 32, "S": 0},
                extra_state={"loader_generator_state": loader_generator.get_state()})
            keys = ("raw_model", "ema", "optimizer", "rng", "sampler", "loader_generator_state")
            equality = {key: _same(state2[key], restored_state2[key], torch) for key in keys}
            equality["next_sampler_requests"] = _same(next_request, sampler.reserve(16)["samples"], torch)
            sampler.cancel_pending()
            result["measurements"]["restore"] = {"state1_update": 1, "state2_update": 2,
                                                 "exact_equal": equality, "actual_loader_generator_consumed": True,
                                                 "resume_loader_workers": 0,
                                                 "worker_performance_measured_separately": result["num_workers"]}
            result["measurements"]["checkpoints"] = checkpoint_rows
            if not all(equality.values()):
                raise RuntimeError("完整state→fresh恢复后下一更新不精确，拒绝准入")
            result["checkpoint_bytes"] = max(row["bytes"] for row in checkpoint_rows)

            # 一张train原生全84×18900 FP32预测无损npz；不计算任何成绩。
            check()
            from .batches import collate_bbox_samples
            prepared_one = prepare_detection_batch(collate_bbox_samples(cpu_inputs[:1]), device="cuda:0")
            torch.cuda.synchronize()
            before = time.monotonic()
            native = backend.network.evaluation_output(prepared_one["img"])[0].detach().float().cpu().numpy()
            torch.cuda.synchronize()
            inference_seconds = time.monotonic() - before
            if native.shape != (1, 84, 18900) or not np.isfinite(native).all():
                raise ValueError("960原生eval必须是完整有限1×84×18900预测")
            flags = np.zeros(18900, dtype=np.uint16)
            scores = native[0, 4:].max(axis=0)
            classes = native[0, 4:].argmax(axis=0)
            flags[classes != 0] |= 1
            flags[scores <= .001] |= 2
            before = time.monotonic()
            prediction_path = temporary / "train_native.npz"
            np.savez_compressed(prediction_path, native_fp32=native, filterflags_u16=flags)
            npz_write = time.monotonic() - before
            before = time.monotonic()
            with np.load(prediction_path, allow_pickle=False) as recovered:
                exact = np.array_equal(native, recovered["native_fp32"]) and np.array_equal(flags, recovered["filterflags_u16"])
            npz_read = time.monotonic() - before
            if not exact:
                raise RuntimeError("原生全预测npz无损读回失败")
            raw_upper = int(native.nbytes + flags.nbytes + 262144)
            result["raw_upper_per_image_bytes"] = raw_upper
            result["measurements"]["raw_prediction"] = {
                "train_uid": cpu_inputs[0]["request"]["uid"], "native_shape": list(native.shape),
                "native_dtype": str(native.dtype), "filterflags_dtype": str(flags.dtype),
                "filterflags_meaning": {"1": "argmax_class_not_person0", "2": "argmax_score_le_0.001"},
                "native_bytes": int(native.nbytes), "filterflags_bytes": int(flags.nbytes),
                "npz_actual_bytes": prediction_path.stat().st_size, "upper_per_image_bytes": raw_upper,
                "upper_formula": "native_fp32.nbytes+filterflags_u16.nbytes+262144",
                "inference_seconds": inference_seconds, "npz_write_seconds": npz_write,
                "npz_read_seconds": npz_read, "lossless_roundtrip": True,
                "score_evaluation_executed": False}
            clean()
            check()

        budget = _storage_budget(result["checkpoint_bytes"], raw_upper,
                                 n_train=len(panel["train_uids"]), n_val=len(panel["val_uids"]))
        free = shutil.disk_usage(root).free
        result["measurements"]["R50_storage"] = {**budget, "actual_free_bytes": free}
        result["R50_required_bytes"] = budget["required_bytes"]
        if free < budget["required_bytes"] + 100 * GIB:
            raise RuntimeError("R50完整证据容量不足且须保留D100GiB，不缩减保存")
        cp_seconds = max(row["snapshot_seconds"] + row["write_seconds"] for row in checkpoint_rows)
        raw_seconds = inference_seconds + npz_write + npz_read
        estimated = (updates * chosen_worker["timed_mean_seconds"]
                     + budget["prediction_images"] * raw_seconds
                     + budget["checkpoints"] * cp_seconds)
        result["estimated_time_seconds"] = estimated
        result["R50_max_run_seconds"] = max(4*3600, math.ceil(estimated * 1.5 + 600))
        result["measurements"]["time_estimate"] = {
            "measured_training_seconds_per_update": chosen_worker["timed_mean_seconds"],
            "measured_raw_prediction_seconds_per_image": raw_seconds,
            "measured_checkpoint_snapshot_write_seconds": cp_seconds,
            "unmeasured_COCO_NMS_and_full_validation_overhead": True,
            "run_timeout_formula": "max(4h,ceil(measured_component_estimate*1.5+600))",
            "R100_and_formal_matrix_admitted": False}
        current_code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}
        if current_code!=executed_code:raise RuntimeError("预检期间源码变化，拒绝复用测量")
        result["executed_code_sha256"]=executed_code
        result["status"] = "PASS"
        result["elapsed_seconds"] = time.monotonic() - started
        check()
        _write_once(target, result)
        return result
    except BaseException as exc:
        result.update(status="FAIL_STOP", elapsed_seconds=time.monotonic() - started,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        try:
            if not target.exists():
                _write_once(target, result)
        except BaseException as receipt_error:
            exc.add_note("admission写入失败: " + repr(receipt_error))
        raise
    finally:
        clean()
