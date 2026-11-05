"""未来独立cached S->R的有界GPU准入；不会接入/重启活动基线队列。"""
from __future__ import annotations
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from .preflight import _safe_path, _write_once, _same
from .formal_preflight import _validate_prior, representative_requests
from .formal import cache_specification

SCHEMA = 'formal_bbox_cached_s_input_admission_v1'
GIB = 1024**3
MAX_SECONDS = 240
PREVIOUS_GPU_PRECHECK_SECONDS = 330
TOTAL_GPU_PRECHECK_SECONDS = 600
U_S, U_R = 30187, 6563


def _validate_closed_results(table, status):
    jobs = table.get('jobs', [])
    if (table.get('schema_version') != 'formal_bbox_five_strategy_v1' or len(jobs) != 2
            or {(j.get('strategy'), j.get('seed')) for j in jobs} != {('R', 17), ('R->R', 17)}):
        raise ValueError('必须引用本次明确R/R->R seed17有限基线块')
    results = status.get('results', [])
    if status.get('state') != 'FORMAL_BLOCK_COMPLETE' or len(results) != 2 or any(r.get('state') != 'COMPLETE' for r in results):
        raise RuntimeError('当前R/R->R块尚未完全闭合，缓存GPU预检不得开始')
    by_name = {Path(r.get('directory', '')).name: r for r in results}
    for job in jobs:
        result = by_name.get(job['job_id'])
        if result is None or result.get('update') != job['updates']:
            raise RuntimeError('基线块终点/作业身份不齐')


def _closed_group_binding(group):
    from .runner import RESEARCH, ARCHIVE_ROOT
    from .output_storage import resolve_output_group
    import psutil
    group = _safe_path(group)
    if group.parent != RESEARCH / 'runs':
        raise ValueError('基线控制组必须为研究runs直属目录')
    output = resolve_output_group(group, research_root=RESEARCH, output_base=ARCHIVE_ROOT / 'runs')
    table = json.loads((group / 'job_table.json').read_bytes())
    status = json.loads((output / 'status.json').read_bytes())
    _validate_closed_results(table, status)
    receipt = json.loads((output / 'control/CONTROL_MANIFEST.json').read_bytes())
    if receipt.get('scope') != 'control_snapshot_only' or receipt.get('state') != 'SEALED':
        raise RuntimeError('R/R->R owned退出后小控制尚未SEALED')
    for row in receipt['files']:
        p = output / 'control' / row['path']
        raw = p.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise RuntimeError('闭合基线小控制清单变化')
    launch = str(group / 'launch.py').replace('\\', '/').lower()
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if launch in {x.replace('\\', '/').lower() for x in process.info['cmdline'] or []}:
                raise RuntimeError('基线控制器仍活动，不能开始cache GPU预检')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    files = [group / 'job_table.json', output / 'status.json', output / 'control/CONTROL_MANIFEST.json']
    return {'control_group': str(group), 'output_group': str(output),
            'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def _validate_unchanged_numeric_source(prior, current):
    previous = prior.get('executed_code_sha256')
    if not isinstance(previous, dict) or not previous:
        raise ValueError('原S实际准入缺执行代码绑定')
    mechanical = {'formal.py', 'input_cache.py', 'formal_cache_preflight.py'}
    for name, digest in previous.items():
        if name not in mechanical and current.get(name) != digest:
            raise ValueError('缓存接线不能更改已测数值/数据实现: ' + name)
    audited_CPU_only = {'cache_resume_v2.py': '481b54d63eca9d399a7cb790cc53466ef03db9a4831bdb122695bdd4feeb5c1f'}
    for name, digest in audited_CPU_only.items():
        if name in current and name not in previous and current[name] != digest:
            raise ValueError('已审CPU缓存辅助源码摘要变化: ' + name)
    if set(current) - set(previous) - mechanical - set(audited_CPU_only):
        raise ValueError('缓存准入出现其他未审执行模块')


def _validate_source_bindings(previous, current):
    """JSON收据与内存绑定按同一规范表示比较，所有键和值仍精确绑定。"""
    from .runner import canonical
    for key, binding in current.items():
        if key not in previous or canonical(previous[key]) != canonical(binding):
            raise ValueError('与原S准入来源不一致: ' + key)


def _validate_previous_budget(prior, warm, requested_seconds):
    values = [prior.get('elapsed_seconds'), warm.get('elapsed_seconds'),
              warm.get('original_preflight_elapsed_seconds'), warm.get('combined_GPU_preparation_seconds')]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('原GPU准备耗时收据字段缺失或非法')
    original, extension, repeated, declared = values
    times = warm.get('per_update_seconds')
    if (warm.get('diagnostic_only') is not True or warm.get('same_numeric_and_input_protocol') is not True
            or warm.get('maximum_updates') != 64 or warm.get('updates') != 64
            or not isinstance(times, list) or len(times) != 64
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in times)):
        raise ValueError('原warm64更新同协议诊断证据不齐')
    used = original + extension
    if (abs(original - repeated) > 1e-6 or abs(used - declared) > 1e-6
            or abs(used - PREVIOUS_GPU_PRECHECK_SECONDS) > 1e-6
            or sum(times) > extension + 1e-6):
        raise ValueError('原GPU准备115.765+214.235=330秒预算收据不一致')
    if used + requested_seconds > TOTAL_GPU_PRECHECK_SECONDS:
        raise ValueError('已花实际预算加cached预检超过600秒')
    return used


def preflight_cached_formal(group_dir, synthetic_index, real_index, weights_dir, *, input_cache,
                            prior_admission, closed_training_group, gpu_release_confirmed=False,
                            max_seconds=MAX_SECONDS, prior_warm_measurement=None):
    """root仅在R/R->R已闭合且GPU释放后显式调用；不派发正式run。

    原330秒预算加本次至多240秒，合计不超过600秒；固定FP32/micro8/workers2，
    不扫描AMP。admission及原始标量日志为准入证据，临时诊断checkpoint打印后精确清理。
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError('cached预检max_seconds须1..240整数')
    if PREVIOUS_GPU_PRECHECK_SECONDS + max_seconds > TOTAL_GPU_PRECHECK_SECONDS:
        raise ValueError('原始加cached预检将超过600秒累计预算')
    spec = cache_specification(input_cache)
    root = _safe_path(group_dir); root.mkdir(parents=True, exist_ok=True); target = root / 'admission.json'
    if target.exists():
        raise FileExistsError('已有cached准入收据，禁止覆盖/自动重放')
    start = time.monotonic(); torch = None; backend = None; iterator = None
    result = {'schema_version': SCHEMA, 'status': 'RUNNING', 'scope': 'future_single_S_to_R_seed17_only',
              'input_cache': spec, 'numeric_mode': 'fp32', 'microbatch_size': 8, 'num_workers': 2,
              'input_size': 960, 'effective_batch': 16, 'diagnostic_updates': 0, 'source_bindings': {},
              'measurements': {}, 'checkpoint_bytes': None, 'raw_upper_per_image_bytes': None,
              'test_read': False, 'validation_scored': False, 'formal_jobs_admitted': False,
              'deadline_seconds': max_seconds, 'previous_gpu_precheck_budget_seconds': PREVIOUS_GPU_PRECHECK_SECONDS,
              'combined_gpu_precheck_budget_seconds': PREVIOUS_GPU_PRECHECK_SECONDS + max_seconds,
              'deadline_semantics': 'cooperative_microbatch_boundary_no_kernel_kill'}

    def check():
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError('cached输入预检240秒有限预算耗尽')

    def emit(row):
        data = (json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode()
        with (root / 'preflight_steps.jsonl').open('ab') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        print(json.dumps({k: row[k] for k in ('kind', 'phase', 'update', 'domain', 'seconds') if k in row}), flush=True)

    def clean_backend():
        nonlocal backend
        backend = None; gc.collect()
        if torch is not None and torch.cuda.is_initialized(): torch.cuda.empty_cache()

    try:
        if gpu_release_confirmed is not True:
            raise RuntimeError('gpu_release_confirmed=True须由root核R/RR完全退出')
        if os.environ.get('YOLO_OFFLINE', '').lower() != 'true': raise RuntimeError('YOLO_OFFLINE=true required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise RuntimeError('CUBLAS_WORKSPACE_CONFIG必须固定:4096:8')
        from .runner import RESEARCH, ARCHIVE_ROOT, production_closed, setup_cuda, seed_all, _shutdown
        import psutil
        lock = json.loads((RESEARCH / '.research_gpu.lock').read_bytes())
        if lock['pid'] != os.getpid() or abs(lock['create_time'] - psutil.Process().create_time()) > .001:
            raise RuntimeError('cached预检须持有当前PID准确GPU锁')
        production_closed()
        result['closed_training_group'] = _closed_group_binding(closed_training_group)
        old_path = _safe_path(prior_admission); old_bytes = old_path.read_bytes(); old = json.loads(old_bytes); _validate_prior(old)
        if old_path != Path(result['closed_training_group']['control_group']) / 'admission.json':
            raise ValueError('必须继承已闭合R/RR块的原始S准入')
        sealed_admission = Path(result['closed_training_group']['output_group']) / 'control/admission.json'
        if sealed_admission.read_bytes() != old_bytes:
            raise ValueError('D原始准入与G已封闭收据不同，不能继承')
        warm_path = Path(result['closed_training_group']['output_group']) / 'warm_measurement.json'
        if prior_warm_measurement is not None and _safe_path(prior_warm_measurement) != warm_path:
            raise ValueError('必须读取既定Gcore/warm_measurement.json预算收据')
        warm_bytes = warm_path.read_bytes(); warm = json.loads(warm_bytes)
        used_seconds = _validate_previous_budget(old, warm, max_seconds)
        result['prior_warm_measurement'] = {'path': str(warm_path), 'sha256': hashlib.sha256(warm_bytes).hexdigest()}
        result['previous_gpu_precheck_budget_seconds'] = used_seconds
        result['combined_gpu_precheck_budget_seconds'] = used_seconds + max_seconds
        result['prior_admission'] = {'path': str(old_path), 'sha256': hashlib.sha256(old_bytes).hexdigest()}
        result['cuda_memory_fraction'] = old['cuda_memory_fraction']; result['raw_upper_per_image_bytes'] = old['raw_upper_per_image_bytes']
        if shutil.disk_usage(RESEARCH).free < 101 * GIB or shutil.disk_usage(ARCHIVE_ROOT).free < 100 * GIB:
            raise RuntimeError('cached预检D/G100GiB余量或临时1GiB容量不足')
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized(): raise RuntimeError('须独占新进程执行cached预检，不复用活动CUDA')
        from .catalogues import synthetic_bbox_catalogue
        from .runner_data import real_panel, RequestPlanDataset, make_loader
        from .input_cache import validate_ready_cache
        from .formal import make_training_dataset, make_samplers, transition_backend, make_phase_iterator
        from .runtime_backend import NativeBBoxBackend
        from .training import prepare_detection_batch
        from .data import CycleSampler
        from .records import save_checkpoint, load_checkpoint
        code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        _validate_unchanged_numeric_source(old, code)
        synthetic = synthetic_bbox_catalogue(synthetic_index, split='train'); panel = real_panel(real_index)
        if len(synthetic['ordered_uids']) != 24149 or len(panel['train_uids']) != 2100 or len(panel['val_uids']) != 750:
            raise ValueError('既定N_S/N_R/N_val不符')
        _validate_source_bindings(old['source_bindings'],
            {'real_panel': panel['binding'], 'synthetic_panel': synthetic['binding']})
        ready = validate_ready_cache(spec, synthetic['binding']); check()
        result['input_cache_validation'] = ready
        result['source_bindings'] = {'real_panel': panel['binding'], 'synthetic_panel': synthetic['binding']}
        config = {'input_cache': spec}
        catalogue = {**panel['train_catalogue'], **synthetic['catalogue']}
        reps = representative_requests(synthetic['catalogue'])
        cached = make_training_dataset(catalogue, reps, config)
        direct = RequestPlanDataset(catalogue, reps, view='train',
            synthetic_reader_options={'max_indexed_cameras': 7, 'max_indexed_frames': 3000},
            real_reader_options={'max_cached_clips': 20, 'max_cached_rows': 600000})
        compared = []
        for i, request in enumerate(reps):
            check(); a, b = cached[i], direct[i]
            if not _same(a, b, torch): raise RuntimeError('cached与即时确定性输入+同seed增强/标签不逐值相同')
            compared.append({'uid': request['uid'], 'camera': synthetic['catalogue'][request['uid']]['camera_id'],
                             'pixels_targets_geometry_lineage_exact': True})
        result['measurements']['cached_direct_equivalence'] = compared
        del cached, direct, a, b; check()
        setup_cuda(old); seed_all(17)
        backend = NativeBBoxBackend.from_public(weights_dir, model_size='s', stage_updates=U_S, final_stage=False,
            microbatch_size=8, numeric_mode='fp32', device='cuda:0', input_size=960, allow_initialized_cuda=True)
        public = deepcopy(backend.network.initialization_binding)
        _validate_source_bindings(old['source_bindings'], {'public_initialization': public})
        result['source_bindings']['public_initialization'] = public
        bindings = {**result['source_bindings'], 'input_cache': spec}
        samplers = make_samplers(panel, synthetic, 17); gen = torch.Generator(device='cpu').manual_seed(170017)
        loader_started = False

        def requests(stream, count):
            live = samplers[stream]; uids, task, domain, seed = live._template
            clone = CycleSampler(uids, task=task, domain=domain, seed=seed)
            clone.load_state_dict(live.state_dict()['sampler']); return clone.take(count)

        def start_loader(stream, count, resume=False):
            nonlocal iterator, loader_started
            loader = make_loader(make_training_dataset(catalogue, requests(stream, count), config), batch_size=8,
                num_workers=2, generator=gen, pin_memory=True, allow_initialized_cuda=True)
            loader.timeout = 120
            iterator = make_phase_iterator(loader, gen, iteration_already_started=resume)
            loader_started = True
            return loader

        def prepared():
            for _ in range(2):
                check(); yield prepare_detection_batch(next(iterator), device='cuda:0', non_blocking=True); check()

        def step(update, stream):
            reservation = samplers[stream].reserve(16); begin = time.monotonic()
            output = backend.step_update(prepared(), update=update)
            if [row['request'] for row in output['sample_lineage']] != reservation['samples']:
                raise RuntimeError('cached worker与已提交采样窗口不符')
            samplers[stream].commit(reservation['token']); torch.cuda.synchronize(); seconds = time.monotonic() - begin
            result['diagnostic_updates'] += 1
            emit({'kind': 'cached_diagnostic_update', 'phase': 'final' if backend.final_stage else 'first',
                  'domain': stream, 'update': update, 'seconds': seconds, 'output': output})
            check(); return output, seconds

        with tempfile.TemporaryDirectory(prefix='.cached-preflight-', dir=root) as temporary:
            temporary = Path(temporary); receipts = []
            def snapshot(global_update):
                check(); begin = time.monotonic()
                exposures = {'R': samplers['R'].committed_draws + samplers['R_extra'].committed_draws,
                             'S': samplers['S'].committed_draws}
                state = backend.snapshot(samplers=samplers, source_bindings=bindings, committed_updates=global_update,
                    cumulative_exposures=exposures, extra_state={'loader_generator_state': gen.get_state().clone(),
                        'loader_iteration_started': loader_started, 'run_scope': 'cached_preflight_not_formal_score'})
                receipt = save_checkpoint(temporary / f'diagnostic_{len(receipts)}.pt', state)
                loaded = load_checkpoint(receipt, expected_bindings=bindings)
                receipts.append({'update': global_update, 'phase_update': backend.stage_update, 'bytes': receipt.bytes,
                    'sha256': receipt.sha256, 'snapshot_write_read_seconds': time.monotonic() - begin})
                check(); return loaded
            snapshot(0)
            loader = start_loader('S', 32)
            out1, _ = step(1, 'S'); state1 = snapshot(1)
            out2, _ = step(2, 'S'); state2 = snapshot(2)
            _shutdown(iterator); iterator = None; del loader; clean_backend()
            backend = NativeBBoxBackend.from_public(weights_dir, model_size='s', stage_updates=U_S, final_stage=False,
                microbatch_size=8, numeric_mode='fp32', device='cuda:0', input_size=960, allow_initialized_cuda=True)
            samplers = make_samplers(panel, synthetic, 17)
            restored = backend.restore_bundle(state1, samplers=samplers, expected_bindings=bindings)
            gen.set_state(restored['extra_state']['loader_generator_state'])
            loader = start_loader('S', 80, resume=restored['extra_state']['loader_iteration_started'])
            replay, _ = step(2, 'S')
            replay_state = backend.snapshot(samplers=samplers, source_bindings=bindings, committed_updates=2,
                cumulative_exposures={'R': 0, 'S': 32}, extra_state={'loader_generator_state': gen.get_state().clone(),
                    'loader_iteration_started': True, 'run_scope': 'cached_preflight_not_formal_score'})
            equal = {key: _same(state2[key], replay_state[key], torch) for key in
                     ('raw_model', 'ema', 'optimizer', 'rng', 'sampler', 'loader_generator_state', 'loader_iteration_started')}
            equal['loss_gradient'] = out2['loss'] == replay['loss'] and out2['gradient_norm_before_clip'] == replay['gradient_norm_before_clip']
            result['measurements']['exact_restore'] = equal
            if not all(equal.values()): raise RuntimeError('cached完整状态恢复后下一更新不精确')
            times = []
            for update in range(3, 7):
                _, seconds = step(update, 'S'); times.append(seconds)
            _shutdown(iterator); iterator = None; del loader
            result['measurements']['S_loader_window'] = {'updates': 4, 'seconds': times,
                'warm_mean_seconds': sum(times) / len(times), 'includes_cached_read_augment_transfer_backward_optimizer_and_sync': True,
                'workers': 2, 'prefetch_factor': 1}
            del state1, state2, replay_state, restored
            raw = {key: value.detach().cpu().clone() for key, value in backend.model.state_dict().items()}
            old_optimizer = backend.optimizer; backend = transition_backend(backend, U_R)
            reset = {'raw_exact': all(torch.equal(v, backend.model.state_dict()[k].detach().cpu()) for k, v in raw.items()),
                     'new_optimizer': backend.optimizer is not old_optimizer, 'empty_optimizer_state': not bool(backend.optimizer.state),
                     'EMA_update_zero': backend.ema_state['updates'] == 0,
                     'EMA_equals_raw': all(torch.equal(v, backend.ema_state['state_dict'][k].detach().cpu()) for k, v in raw.items())}
            if not all(reset.values()): raise RuntimeError('cached S到R重建改变raw或继承旧状态')
            loader = start_loader('R', 16); step(1, 'R'); _shutdown(iterator); iterator = None; del loader
            snapshot(7); result['measurements']['transition_to_R'] = reset; del raw, old_optimizer
            result['checkpoint_bytes'] = max(x['bytes'] for x in receipts); result['measurements']['checkpoints'] = receipts
            free, total = torch.cuda.mem_get_info()
            result['measurements']['memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'free_bytes': free, 'total_bytes': total,
                'process_RSS_bytes': psutil.Process().memory_info().rss}
            if free < GIB or total - torch.cuda.max_memory_reserved() < GIB: raise RuntimeError('cached实测GPU不足1GiB余量')
            result['temporary_artifacts'] = {'full_diagnostic_models': len(receipts), 'bytes': sum(x['bytes'] for x in receipts),
                'lifecycle': 'printed_then_exact_TemporaryDirectory_cleanup_not_formal_models'}
            print(json.dumps({'temporary_artifacts': result['temporary_artifacts']}), flush=True)
        check()
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        if current != code: raise RuntimeError('cached预检运行期间源码变化')
        validate_ready_cache(spec, synthetic['binding']); check()
        result['executed_code_sha256'] = code; result['status'] = 'PASS'; result['elapsed_seconds'] = time.monotonic() - start
        result['combined_gpu_precheck_elapsed_upper_seconds'] = PREVIOUS_GPU_PRECHECK_SECONDS + result['elapsed_seconds']
        _write_once(target, result); return result
    except BaseException as exc:
        result.update(status='FAIL_STOP', elapsed_seconds=time.monotonic() - start,
                      failure={'type': type(exc).__name__, 'message': str(exc)})
        if not target.exists(): _write_once(target, result)
        raise
    finally:
        if iterator is not None:
            from .runner import _shutdown
            _shutdown(iterator)
        clean_backend()
