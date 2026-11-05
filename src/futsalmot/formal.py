"""正式BBox五策略执行器：全局提交、分阶段raw承接与完整原始证据。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from .training import domain_at, STRATEGIES

SCHEMA = 'formal_bbox_five_strategy_v1'
FORMAL_PLAN_SHA = '3814e51504548a39e6b88fdf073a4377ba22b6ddf7b1607330bfb1664c6eac67'
U_S = 30187
U_R = 6563
SYNTHETIC = Path('G:/FutsalMOT_Dataset/synthetic/dataset.json')
SYNTHETIC_READER_OPTIONS = {'max_indexed_cameras': 7, 'max_indexed_frames': 3000}
REAL_READER_OPTIONS = {'max_cached_clips': 20, 'max_cached_rows': 600000}


def _require(value, message):
    if not value:
        raise ValueError(message)


def phases(strategy, u_s=U_S, u_r=U_R):
    _require(strategy in STRATEGIES, '未知五策略')
    _require(type(u_s) is int and type(u_r) is int and min(u_s, u_r) >= 2, '阶段须至少两更新')
    if strategy == 'R':
        return [{'name': 'final', 'offset': 0, 'updates': u_r, 'final_stage': True}]
    first = {'name': 'first', 'offset': 0, 'updates': u_s, 'final_stage': False}
    return [first] if strategy == 'S' else [first, {'name': 'final', 'offset': u_s, 'updates': u_r, 'final_stage': True}]


def exposure_counts(samplers):
    return {'R': sum(s.committed_draws for k, s in samplers.items() if k in ('R', 'R_extra')),
            'S': samplers['S'].committed_draws}


def make_samplers(panel, synthetic, seed):
    from .data import CycleSampler, PrefetchSampler
    _require(type(seed) is int and seed in (17, 29, 43), 'seed仅17/29/43')
    return {stream: PrefetchSampler(CycleSampler(
        synthetic['ordered_uids'] if stream == 'S' else panel['train_uids'],
        task='detect', domain=stream, seed=seed)) for stream in ('R', 'S', 'R_extra')}


def phase_requests(strategy, samplers, *, start, end, u_s=U_S, u_r=U_R):
    """只生成当前阶段尚未提交的请求，不推进真实sampler。"""
    from .data import CycleSampler
    _require(type(start) is int and type(end) is int and 0 <= start <= end, '全局区间非法')
    clones = {}
    for key, live in samplers.items():
        uids, task, domain, seed = live._template
        clone = CycleSampler(uids, task=task, domain=domain, seed=seed)
        clone.load_state_dict(live.state_dict()['sampler'])
        clones[key] = clone
    requests = []
    for step in range(start + 1, end + 1):
        stream, _ = domain_at(strategy, step, u_s, u_r)
        requests.extend(clones[stream].take(16))
    return requests


def transition_backend(backend, u_r=U_R):
    """承接同一raw网络，重建SGD/EMA；不读公开权重、不重播种。"""
    from .runtime_backend import NativeBBoxBackend
    _require(not backend.failed and not backend.final_stage, '只能从非失败第一阶段重建；正式边界由caller独立核验')
    return NativeBBoxBackend(backend.network, stage_updates=u_r, final_stage=True,
        microbatch_size=backend.microbatch_size, numeric_mode=backend.numeric_mode,
        device=str(backend.device), input_size=backend.input_size,
        allow_initialized_cuda=backend.device.type == 'cuda')


def evaluation_tag(global_update, phase, phase_update, phase_updates):
    suffix = 'PT' if global_update == 0 else 'initial' if phase_update == 0 else 'final' if phase_update == phase_updates else 'periodic'
    return f'step{global_update:06d}_{phase}_{suffix}'


def evaluation_schedule(strategy, u_s=U_S, u_r=U_R):
    out = []
    for phase in phases(strategy, u_s, u_r):
        points = [0, phase['updates']]
        for local in points:
            global_update = phase['offset'] + local
            out.append({'tag': evaluation_tag(global_update, phase['name'], local, phase['updates']),
                        'global_update': global_update, 'phase_update': local, 'phase': phase['name']})
    return out


def evaluation_due(config, phase, phase_update):
    """新训练/显式恢复共用冻结端点计划；交界的两种EMA状态不合并。"""
    return any(p['phase'] == phase and p['phase_update'] == phase_update
               for p in config['evaluation_schedule'])


def _evaluate_if_due(directory, backend, panel, config, gen, recorder, phase, global_update, deadline):
    if not evaluation_due(config, phase, backend.stage_update):
        return None
    return _evaluate(directory, backend, panel, config, gen, recorder, phase, global_update, deadline)


def cache_specification(value):
    """只规范新run显式缓存绑定，不读取/创建缓存或修改历史协议。"""
    _require(type(value) is dict and set(value) == {'path', 'manifest_sha256'}, '缓存绑定须仅path/manifest_sha256')
    path = value['path']; digest = value['manifest_sha256']
    _require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts,
             '缓存必须明确绝对路径且无上跳')
    _require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
             '缓存manifest缺小写SHA256')
    return {'path': str(Path(path).absolute()), 'manifest_sha256': digest}


def make_training_dataset(catalogue, requests, config):
    """相同显式请求/增强；仅未来显式缓存run替换确定性输入读取。"""
    from .runner_data import RequestPlanDataset
    options = dict(view='train', real_reader_options=REAL_READER_OPTIONS,
                   synthetic_reader_options=SYNTHETIC_READER_OPTIONS)
    if config.get('input_cache') is None:
        return RequestPlanDataset(catalogue, requests, **options)
    from .input_cache import CachedRequestPlanDataset
    spec = cache_specification(config['input_cache'])
    return CachedRequestPlanDataset(catalogue, requests, cache_root=spec['path'],
        expected_manifest_sha256=spec['manifest_sha256'], **options)


def protocol(strategy, seed, *, max_run_seconds, additional_space_bound_bytes,
             raw_upper_per_image_bytes, checkpoint_bytes, admission_sha256, input_cache=None):
    from .runner import DEADLINE, RESERVE, PAPER_SHA
    _require(strategy in STRATEGIES and type(seed) is int and seed in (17, 29, 43), '非法正式策略/seed')
    for name, value in [('max_run_seconds', max_run_seconds), ('additional_space_bound_bytes', additional_space_bound_bytes),
                        ('raw_upper_per_image_bytes', raw_upper_per_image_bytes), ('checkpoint_bytes', checkpoint_bytes)]:
        _require(type(value) is int and value > 0, name + '须实际准入正整数')
    _require(max_run_seconds <= 16 * 3600, '单job不能超过16h')
    _require(isinstance(admission_sha256, str) and len(admission_sha256) == 64, '缺准入收据SHA')
    spec = phases(strategy)
    result = {'schema_version': SCHEMA, 'job_id': strategy.replace('->', '_to_').replace('+', '_plus_') + f'_seed{seed}',
            'task': 'detect', 'model': 'yolo11s', 'strategy': strategy, 'seed': seed,
            'N_R': 2100, 'N_S': 24149, 'N_val': 750, 'nominal_epochs_R': 50, 'nominal_epochs_S': 20,
            'U_R': U_R, 'U_S': U_S, 'updates': sum(p['updates'] for p in spec), 'phases': spec,
            'input_size': 960, 'numeric_mode': 'fp32', 'microbatch_size': 8, 'effective_batch_size': 16,
            'num_workers': 2, 'checkpoint_every': 500, 'evaluation_every': None,
            'evaluation_policy': 'phase_endpoints_v1',
            'save_phase_zero': True, 'evaluate_stage0': True, 'evaluation_schedule': evaluation_schedule(strategy),
            'complete_checkpoint_count': sum(1 + p['updates']//500 + int(p['updates']%500 != 0) for p in spec), 'main_metric': 'COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset',
            'best_selection': False, 'test_enabled': False, 'optimizer': 'SGD_Nesterov',
            'first_lr_peak': .01, 'final_lr_peak': .001, 'lrf': .01, 'momentum': .937, 'weight_decay': .0005,
            'gradient_clip_norm': 10., 'mainline_all_logs_checkpoints_predictions_retained': True,
            'synthetic_reader_options': deepcopy(SYNTHETIC_READER_OPTIONS), 'real_reader_options': deepcopy(REAL_READER_OPTIONS),
            'source_admission_sha256': admission_sha256, 'EXPERIMENTS_sha256': FORMAL_PLAN_SHA, 'PAPER_sha256': PAPER_SHA,
            'storage_mode': 'direct_g', 'control_space_bound_bytes': 32 * 1024**2, 'disk_reserve_bytes': RESERVE,
            'max_run_seconds': max_run_seconds, 'additional_space_bound_bytes': additional_space_bound_bytes,
            'raw_upper_per_image_bytes': raw_upper_per_image_bytes, 'checkpoint_bytes': checkpoint_bytes,
            'deadline_utc': DEADLINE, 'infrastructure_retry_limit': 1, 'automatic_resume': False}
    if input_cache is not None:
        _require(strategy == 'S->R' and seed == 17, '本次缓存只准独立S->R seed17新run')
        result['input_cache'] = cache_specification(input_cache)
    return result


def _check_protocol(config):
    expected = protocol(config['strategy'], config['seed'], max_run_seconds=config['max_run_seconds'],
        additional_space_bound_bytes=config['additional_space_bound_bytes'], raw_upper_per_image_bytes=config['raw_upper_per_image_bytes'],
        checkpoint_bytes=config['checkpoint_bytes'], admission_sha256=config['source_admission_sha256'], input_cache=config.get('input_cache'))
    _require({k: config[k] for k in expected} == expected, '正式协议数值/范围与冻结定义不同')
    _require(set(config) == set(expected) | {'control_group'}, '协议含未知或缺控制路径字段')


def prepare_group(group, *, jobs, admission=None, code_source=None):
    """仅冻结已测有限jobs和小控制/G路径，不启动CUDA或训练。"""
    from . import runner
    from .runner_data import real_panel
    from .catalogues import synthetic_bbox_catalogue
    from .output_storage import create_output_binding, capture_control
    group = Path(group).absolute()
    _require(group.parent == runner.RESEARCH / 'runs', 'group须研究runs直属目录')
    _require(group.is_dir() and not (group / 'job_table.json').exists(), '先建独立group/准入收据，拒绝重准备')
    for name, digest in [('EXPERIMENTS.md', FORMAL_PLAN_SHA), ('PAPER.md', runner.PAPER_SHA)]:
        _require(runner.sha(runner.RESEARCH / 'docs/plan' / name) == digest, '封存方案变化')
    actual = json.loads((group / 'admission.json').read_bytes())
    _require(admission is None or admission == actual, 'caller准入与已落盘收据不一致')
    _require(actual.get('status') == 'PASS', '缺实际PASS准入')
    _require(actual.get('numeric_mode') == 'fp32' and actual.get('microbatch_size') == 8 and actual.get('num_workers') == 2,
             '准入数值/worker不同')
    panel = real_panel(runner.REAL)
    synthetic = synthetic_bbox_catalogue(SYNTHETIC, split='train')
    _require(len(panel['train_uids']) == 2100 and len(panel['val_uids']) == 750 and len(synthetic['ordered_uids']) == 24149,
             '来源池数量变化')
    for key, value in [('real_panel', panel['binding']), ('synthetic_panel', synthetic['binding'])]:
        _require(runner.canonical(actual['source_bindings'][key]) == runner.canonical(value), '准入来源变化:' + key)
    _require(isinstance(jobs, list) and jobs and len({j['job_id'] for j in jobs}) == len(jobs), '缺明确非重复有限jobs')
    frozen = []
    for job in jobs:
        cfg = deepcopy(job); cfg['control_group'] = str(group); _check_protocol(cfg)
        _require(cfg['source_admission_sha256'] == runner.sha(group / 'admission.json'), 'job准入SHA不同')
        frozen.append(cfg)
    cache_jobs = [job for job in frozen if 'input_cache' in job]
    if cache_jobs:
        _require(len(frozen) == 1 and len(cache_jobs) == 1, '缓存准入仅明确一个独立S->R新job')
        spec = cache_jobs[0]['input_cache']
        _require(actual.get('schema_version') == 'formal_bbox_cached_s_input_admission_v1'
                 and actual.get('input_cache') == spec, '缺本缓存manifest的实际GPU准入')
        from .input_cache import validate_ready_cache
        validate_ready_cache(spec, synthetic['binding'])
    else:
        _require(actual.get('input_cache') is None, '缓存准入不能冒充原4K输入准入')
    remaining = datetime.fromisoformat(runner.DEADLINE).timestamp() - time.time()
    _require(sum(j['max_run_seconds'] for j in frozen) + 600 <= remaining, '整个有限块无法在硬截止前闭合')
    runner.production_closed(); runner.check_disk(32 * 1024**2)
    runner.check_archive_space(sum(j['additional_space_bound_bytes'] for j in frozen))
    source = Path(code_source or Path(__file__).parent)
    executed = {p.name: runner.sha(p) for p in sorted(source.glob('*.py'))}
    _require(actual.get('executed_code_sha256') == executed, '预检实际执行代码集合/SHA与当前源码不同')
    target = group / 'code/futsalmot'; target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for p in sorted(source.glob('*.py')):
        data = p.read_bytes(); (target / p.name).write_bytes(data); hashes[p.name] = hashlib.sha256(data).hexdigest()
    _require(hashes == executed and {p.name: runner.sha(p) for p in source.glob('*.py')} == executed
             and {p.name: runner.sha(p) for p in target.glob('*.py')} == executed, '冻结复制期间源码变化')
    shutil.copyfile(runner.RESEARCH / 'requirements.lock', group / 'code/requirements.lock')
    runner.write_once(group / 'code_bindings.json', hashes)
    bindings = {'real_panel': panel['binding'], 'synthetic_panel': synthetic['binding'],
                'public_model': {'path': str(runner.WEIGHTS / 'yolo11s.pt'), 'sha256': runner.sha(runner.WEIGHTS / 'yolo11s.pt')},
                'EXPERIMENTS_sha256': FORMAL_PLAN_SHA, 'PAPER_sha256': runner.PAPER_SHA,
                'requirements_sha256': runner.sha(group / 'code/requirements.lock'), 'source_code': hashes}
    if cache_jobs:
        bindings['input_cache'] = deepcopy(cache_jobs[0]['input_cache'])
    table = {'schema_version': SCHEMA, 'scope': 'explicit_finite_formal_jobs_only', 'jobs': frozen,
             'source_bindings': bindings, 'test_enabled': False, 'extra_jobs_authorized': False,
             'total_bound_bytes': sum(j['additional_space_bound_bytes'] for j in frozen),
             'total_max_run_seconds': sum(j['max_run_seconds'] for j in frozen)}
    runner.write_once(group / 'job_table.json', table)
    launch = "import os,sys\nfrom pathlib import Path\nos.environ['YOLO_OFFLINE']='true'\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\ngroup=Path(__file__).absolute().parent\nsys.path.insert(0,str(group/'code'))\nif __name__=='__main__':\n from futsalmot.formal import controller\n controller(group)\n"
    with (group / 'launch.py').open('x', encoding='utf-8', newline='\n') as f: f.write(launch)
    location = create_output_binding(group, research_root=runner.RESEARCH, output_base=runner.ARCHIVE_ROOT / 'runs')
    snapshot = capture_control(group, research_root=runner.RESEARCH, output_base=runner.ARCHIVE_ROOT / 'runs')
    return {'group': str(group), 'output_group': location['output_group'], 'jobs': [j['job_id'] for j in frozen],
            'job_table_sha256': runner.sha(group / 'job_table.json'), 'source_bindings': bindings, 'control_snapshot': snapshot}


def make_phase_iterator(loader, generator, *, iteration_already_started):
    """恢复已有iterator时撤销重建的base-seed消耗，新阶段保持正常推进。"""
    _require(type(iteration_already_started) is bool, '须明确已保存iterator创建边界')
    before = generator.get_state()
    iterator = iter(loader)
    if iteration_already_started:
        generator.set_state(before)
    return iterator


def _save_boundary(directory, backend, samplers, bindings, gen, recorder, *, global_update, phase, tag, loader_iteration_started=False):
    from . import runner
    from .records import save_checkpoint
    bundle = backend.snapshot(samplers=samplers, source_bindings=bindings,
        committed_updates=global_update, cumulative_exposures=exposure_counts(samplers), phase=phase,
        extra_state={'loader_generator_state': gen.get_state().clone(), 'run_scope': SCHEMA,
                     'global_update': global_update, 'formal_phase': phase, 'loader_iteration_started': loader_iteration_started})
    path = Path(directory) / 'checkpoints' / f'{global_update:06d}_{phase}_{backend.stage_update:06d}_{tag}.pt'
    receipt = save_checkpoint(path, bundle)
    row = {'kind': 'checkpoint', 'update': global_update, 'phase': phase, 'phase_update': backend.stage_update,
           'tag': tag, 'receipt': asdict(receipt), 'saved_at_utc': runner.now()}
    recorder.event(row); runner.append(Path(directory) / 'checkpoints.jsonl', row)
    return receipt


class _EvaluationRecorder:
    """保留原评价局部update，同时向全局events附明确定位。"""
    def __init__(self, recorder, global_update, phase):
        self.recorder, self.global_update, self.phase = recorder, global_update, phase
    def event(self, record):
        self.recorder.event({**record, 'phase_update': record.get('update'),
                             'update': self.global_update, 'global_update': self.global_update, 'phase': self.phase})


def _evaluate(directory, backend, panel, config, gen, recorder, phase, global_update, deadline):
    from . import runner
    tag = evaluation_tag(global_update, phase, backend.stage_update, backend.stage_updates)
    # 原runner已保护正常评价RNG；这里也覆盖其loader/EMA初始化失败路径。
    from .records import capture_rng_state, restore_rng_state
    include_cuda = backend.device.type == 'cuda'
    before = capture_rng_state(include_cuda=include_cuda)
    loader_state = gen.get_state().clone()
    modes = [(module, module.training) for module in backend.model.modules()]
    try:
        return runner.evaluate(directory, backend, panel, config, gen,
            _EvaluationRecorder(recorder, global_update, phase), tag, hard_stop_unix=deadline)
    finally:
        gen.set_state(loader_state)
        for module, training in modes:
            module.training = training
        restore_rng_state(before, allow_cuda=include_cuda)


def _journal(directory, config, state, global_update, exposures, result=None):
    from . import runner
    text = (f"\n\n## FORMAL：{config['job_id']} {state}\n\nUTC {runner.now()}。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed{config['seed']}，"
        f"SoccerNet train2100/val750，synthetic train24149，策略{config['strategy']}。实际已提交{global_update}/{config['updates']}更新，"
        f"R/S曝光{exposures['R']}/{exposures['S']}；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。"
        f"第一阶段{U_S}步峰LR.01，真实终阶段{U_R}步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；"
        f"独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，每阶段0/阶段末750图原生预测与全部负结果保留。"
        f"不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。\n\n"
        f"状态/原始结果：{json.dumps(runner.plain(result), ensure_ascii=False)}。证据：{Path(directory).as_posix()}。\n")
    with (runner.RESEARCH / 'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:
        f.write(text.encode()); f.flush(); os.fsync(f.fileno())


def _verify_bindings(group, table):
    from . import runner
    from .runner_data import real_panel
    from .catalogues import synthetic_bbox_catalogue
    bindings = table['source_bindings']
    for name, digest in bindings['source_code'].items():
        _require(runner.sha(Path(__file__).parent / name) == digest, '冻结训练源码变化:' + name)
    _require(runner.sha(Path(__file__).parent.parent / 'requirements.lock') == bindings['requirements_sha256'], '冻结依赖锁变化')
    for name, key in [('EXPERIMENTS.md', 'EXPERIMENTS_sha256'), ('PAPER.md', 'PAPER_sha256')]:
        _require(runner.sha(runner.RESEARCH / 'docs/plan' / name) == bindings[key], '封存方案变化')
    _require(runner.sha(runner.WEIGHTS / 'yolo11s.pt') == bindings['public_model']['sha256'], '公开输入权重变化')
    panel = real_panel(runner.REAL); synthetic = synthetic_bbox_catalogue(SYNTHETIC, split='train')
    _require(runner.canonical(panel['binding']) == runner.canonical(bindings['real_panel']), '真实来源绑定变化')
    _require(runner.canonical(synthetic['binding']) == runner.canonical(bindings['synthetic_panel']), '合成来源绑定变化')
    if 'input_cache' in bindings:
        from .input_cache import validate_ready_cache
        validate_ready_cache(cache_specification(bindings['input_cache']), synthetic['binding'])
    return panel, synthetic


def run_job(group, config, *, resume_receipt=None):
    """调用方须已持GPU锁/初始化CUDA；resume仅由明确恢复调用提供可信收据。"""
    import torch
    import psutil
    from . import runner
    from .runner_data import RequestPlanDataset, make_loader
    from .runtime_backend import NativeBBoxBackend
    from .records import RunRecorder, CheckpointReceipt, load_checkpoint
    group = Path(group).absolute(); _check_protocol(config)
    lock = json.loads((runner.RESEARCH / '.research_gpu.lock').read_bytes())
    _require(lock['pid'] == os.getpid() and lock['create_time'] == psutil.Process().create_time()
             and Path(lock['group']) == group, '调用者未持本组GPU锁')
    table = json.loads((group / 'job_table.json').read_bytes())
    _require(config in table['jobs'], 'job不在已冻结有限清单')
    panel, synthetic = _verify_bindings(group, table)
    _require(runner.sha(group / 'admission.json') == config['source_admission_sha256'], '准入收据变化')
    runner.check_write_space(config, config['additional_space_bound_bytes'])
    directory = runner.output_group(group) / config['job_id']
    _require(config.get('input_cache') == table['source_bindings'].get('input_cache'), 'job缓存与冻结来源绑定不同')
    bindings = {**table['source_bindings'], 'protocol': deepcopy(config)}
    manifest = {'schema_version': SCHEMA, 'run_id': config['job_id'], 'source_bindings': bindings,
                'protocol': config, 'created_under_group': str(group), 'output_group': str(directory.parent), 'storage_mode': 'direct_g'}
    schedule = phases(config['strategy']); samplers = make_samplers(panel, synthetic, config['seed'])
    runner.seed_all(config['seed']); gen = torch.Generator(device='cpu').manual_seed(170000 + config['seed'])
    start = 0; restored_phase = None; bundle = None
    if resume_receipt is not None:
        receipt = CheckpointReceipt(**resume_receipt); bundle = load_checkpoint(receipt, expected_bindings=bindings)
        start = bundle['committed_updates']; restored_phase = bundle['phase']
        _require(bundle['extra_state']['run_scope'] == SCHEMA if 'extra_state' in bundle else bundle.get('run_scope') == SCHEMA,
                 '检查点不属正式协议')
        recorder = RunRecorder.resume(directory, expected_manifest=manifest, checkpoint_committed_updates=start)
    else:
        recorder = RunRecorder(directory, manifest); (directory / 'checkpoints').mkdir()
    # runtime deadline永久记录，显式恢复也不能重置预算。
    limits = directory / 'runtime_limit.json'
    if resume_receipt is None:
        deadline = min(time.time() + config['max_run_seconds'], datetime.fromisoformat(config['deadline_utc']).timestamp())
        runner.write_once(limits, {'started_at_utc': runner.now(), 'deadline_unix': deadline, 'max_run_seconds': config['max_run_seconds']})
    else:
        deadline = json.loads(limits.read_bytes())['deadline_unix']
    backend = None; iterator = None; latest = start; active_phase = None; end = None; resumed_iterator_pending = False
    try:
        for spec in schedule:
            offset, count, phase = spec['offset'], spec['updates'], spec['name']
            if start > offset + count or (start == offset + count and restored_phase == 'final' and phase == 'first'):
                continue
            restored_this_phase = False
            if backend is None:
                backend = NativeBBoxBackend.from_public(runner.WEIGHTS, model_size='s', stage_updates=count,
                    final_stage=spec['final_stage'], microbatch_size=8, numeric_mode='fp32', device='cuda:0', allow_initialized_cuda=True)
                if bundle is not None:
                    _require(restored_phase == phase and start == offset + bundle['phase_updates'], '检查点全局/阶段边界不齐')
                    restored = backend.restore_bundle(bundle, samplers=samplers, expected_bindings=bindings)
                    gen.set_state(restored['extra_state']['loader_generator_state'])
                    resumed_iterator_pending = restored['extra_state']['loader_iteration_started']
                    bundle = None; restored_this_phase = True
            elif phase == 'final':
                _require(backend.stage_update == backend.stage_updates and latest == U_S, '第一阶段未完整提交，不能转换')
                previous = backend; backend = transition_backend(previous, U_R); del previous; torch.cuda.empty_cache()
            active_phase = phase
            runner.status(directory, {'state': 'RUNNING', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                'update': latest, 'updates': config['updates'], 'phase': phase, 'phase_update': backend.stage_update})
            if backend.stage_update == 0:
                if not restored_this_phase:
                    _save_boundary(directory, backend, samplers, bindings, gen, recorder, global_update=offset, phase=phase, tag='initial')
                if offset == 0 and not restored_this_phase:
                    _journal(directory, config, 'STARTED', 0, exposure_counts(samplers), {'state': 'public_initialized_zero_updates'})
                elif offset > 0 and not restored_this_phase:
                    recorder.event({'kind': 'phase_reset', 'update': offset, 'phase': phase,
                                    'optimizer_reset': True, 'EMA_reset': True, 'raw_preserved': True})
                elif restored_this_phase:
                    recorder.event({'kind': 'explicit_resume_at_phase_zero', 'update': offset, 'phase': phase})
                # 阶段0均评价，末阶段0为重置EMA后的实际raw。
                _evaluate_if_due(directory, backend, panel, config, gen, recorder, phase, offset, deadline)
            else:
                evaluated = _evaluate_if_due(directory, backend, panel, config, gen, recorder, phase, latest, deadline)
                if evaluated is not None:
                    end = evaluated
            if backend.stage_update == count:
                resumed_iterator_pending = False
            if backend.stage_update < count:
                requests = phase_requests(config['strategy'], samplers, start=latest, end=offset + count)
                catalog = {**panel['train_catalogue'], **synthetic['catalogue']}
                dataset = make_training_dataset(catalog, requests, config)
                loader = make_loader(dataset, batch_size=8, num_workers=2, generator=gen, pin_memory=True, allow_initialized_cuda=True)
                loader.timeout = 300
                iterator = make_phase_iterator(loader, gen, iteration_already_started=resumed_iterator_pending)
                resumed_iterator_pending = False
                for local in range(backend.stage_update + 1, count + 1):
                    global_step = offset + local
                    if runner.stop_requested(group, directory):
                        _save_boundary(directory, backend, samplers, bindings, gen, recorder, global_update=latest, phase=phase, tag='user_stop', loader_iteration_started=True)
                        runner.status(directory, {'state': 'USER_STOP', 'update': latest, 'updates': config['updates'], 'phase': phase})
                        _journal(directory, config, 'USER_STOP', latest, exposure_counts(samplers)); return {'state': 'USER_STOP', 'directory': str(directory), 'update': latest}
                    if time.time() >= deadline: raise TimeoutError('LIMIT_STOP: original formal run deadline reached')
                    runner.check_write_space(config, 2 * config['checkpoint_bytes'])
                    if torch.cuda.mem_get_info()[0] < 1024**3: raise RuntimeError('LIMIT_STOP: GPU free below1GiB')
                    stream, ordinal = domain_at(config['strategy'], global_step, U_S, U_R)
                    live = samplers[stream]; window = live.reserve(16); began = time.monotonic()
                    batches = runner._microbatches(iterator, 2)
                    actual = [x['request'] for batch in batches for x in batch['sample_lineage']]
                    _require(actual == window['samples'], '已派发请求与事务提交窗口不符')
                    metrics = backend.step_update(batches, update=local)
                    live.commit(window['token']); latest = global_step
                    exposures = exposure_counts(samplers)
                    metrics.update(update=global_step, phase_update=local, phase=phase, domain='R' if stream == 'R_extra' else stream,
                        sampling_stream=stream, domain_update=ordinal, gradient_norm=metrics['gradient_norm_before_clip'],
                        R_exposures=exposures['R'], S_exposures=exposures['S'], wall_seconds=time.monotonic() - began,
                        geometry=[g for batch in batches for g in batch['geometry']], maximum_cuda_allocated=torch.cuda.max_memory_allocated(),
                        maximum_cuda_reserved=torch.cuda.max_memory_reserved())
                    recorder.step(metrics)
                    if runner.checkpoint_due(local, count, 500):
                        _save_boundary(directory, backend, samplers, bindings, gen, recorder, global_update=latest, phase=phase,
                                       tag='final' if local == count else 'interval', loader_iteration_started=True)
                    if local % 25 == 0 or local == count:
                        runner.production_closed()
                        runner.status(directory, {'state': 'RUNNING', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                            'update': latest, 'updates': config['updates'], 'phase': phase, 'phase_update': local, 'loss': metrics['loss']})
                        print(json.dumps({'event': 'update', 'run': directory.name, 'update': latest, 'updates': config['updates'],
                            'phase': phase, 'phase_update': local, 'loss': metrics['loss'], 'update_s': metrics['wall_seconds']}), flush=True)
                    del batches
                    evaluated = _evaluate_if_due(directory, backend, panel, config, gen, recorder, phase, latest, deadline)
                    if evaluated is not None:
                        end = evaluated
                runner._shutdown(iterator); iterator = None; del loader, dataset, requests, catalog
            start = latest
        _require(latest == config['updates'] and end is not None, '未到正式终点')
        result = {'state': 'COMPLETE', 'directory': str(directory), 'update': latest,
                  'exposures': exposure_counts(samplers), 'terminal': end, 'phase': active_phase}
        runner.status(directory, {**result, 'updates': config['updates'], 'terminal_evaluation': end})
        recorder.event({'kind': 'run_complete', 'update': latest, 'finished_at_utc': runner.now()})
        _journal(directory, config, 'COMPLETE', latest, exposure_counts(samplers), result)
        return result
    except BaseException as exc:
        error = {'state': 'LIMIT_STOP' if isinstance(exc, TimeoutError) else 'USER_STOP' if runner.stop_requested(group, directory) else 'FAIL_STOP',
                 'update': latest, 'phase': active_phase, 'backend_stage_update': None if backend is None else backend.stage_update,
                 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        try:
            for live in samplers.values(): live.cancel_pending()
            if backend is not None and not backend.failed and backend.stage_update + next(x['offset'] for x in schedule if x['name'] == active_phase) == latest:
                _save_boundary(directory, backend, samplers, bindings, gen, recorder, global_update=latest, phase=active_phase, tag='controlled_failure', loader_iteration_started=iterator is not None)
            elif backend is not None:
                from .checkpoint_state import _clone_cpu
                from .records import capture_rng_state
                evidence = {'schema_version': 'uncommitted_failure_evidence_v1', 'resume_allowed': False,
                    'failure': backend.failure, 'last_committed_update': latest, 'raw': backend.model.state_dict(),
                    'ema': backend.ema_state, 'optimizer': backend.optimizer.state_dict(), 'rng': capture_rng_state(include_cuda=True),
                    'samplers': {k: v.state_dict() for k, v in samplers.items()}, 'source_bindings': bindings}
                with (directory / 'failed_uncommitted_state.pt').open('xb') as f:
                    torch.save(_clone_cpu(evidence, allow_cuda=True, cuda_initialized=True), f); f.flush(); os.fsync(f.fileno())
        except BaseException as save_error: error['failure_evidence_error'] = repr(save_error)
        runner.status(directory, error); _journal(directory, config, error['state'], latest, exposure_counts(samplers), error)
        try: recorder.event({'kind': 'failure', **error})
        except BaseException: pass
        if error['state'] in ('USER_STOP', 'LIMIT_STOP'): return {**error, 'directory': str(directory)}
        raise
    finally:
        runner._shutdown(iterator)
        if backend is not None: del backend
        torch.cuda.empty_cache()


def controller(group):
    """只执行已冻结有限清单；任何失败/取消/超限立即停止后续派发。"""
    from . import runner
    from .output_storage import capture_control
    group = Path(group).absolute(); table = json.loads((group / 'job_table.json').read_bytes())
    _require(table.get('schema_version') == SCHEMA and table['test_enabled'] is False, '不是本正式有限块')
    admission = json.loads((group / 'admission.json').read_bytes()); _require(admission.get('status') == 'PASS', '准入未通过')
    _require(not (runner.output_group(group) / 'control/CONTROL_MANIFEST.json').exists(), '已封闭组不能重新派发')
    _require(not (group / 'status.json').exists(), '已派发组不能重放controller；恢复须明确可信收据入口')
    results = []; owned = False; active_config = None
    try:
        with runner.gpu_lock(group):
            owned = True
            runner.setup_cuda(admission)
            for index, config in enumerate(table['jobs']):
                active_config = config
                if runner.stop_requested(group, runner.output_group(group)):
                    runner.group_status(group, {'state': 'USER_STOP', 'results': results}); return
                runner.check_write_space(config, sum(j['additional_space_bound_bytes'] for j in table['jobs'][index:]))
                _require(time.time() + sum(j['max_run_seconds'] for j in table['jobs'][index:]) + 600 <= datetime.fromisoformat(runner.DEADLINE).timestamp(),
                         '有限块剩余job无法在硬截止内闭合')
                runner.group_status(group, {'state': 'RUNNING', 'pid': os.getpid(),
                    'create_time': __import__('psutil').Process().create_time(), 'active_job': config['job_id'], 'results': results})
                result = run_job(group, config); results.append(result)
                if result['state'] != 'COMPLETE':
                    runner.group_status(group, {'state': result['state'], 'results': results}); return
            runner.group_status(group, {'state': 'FORMAL_BLOCK_COMPLETE', 'results': results, 'test_enabled': False})
    except BaseException as exc:
        error = {'state': 'FAIL_STOP', 'error_type': type(exc).__name__, 'error': str(exc),
                 'traceback': traceback.format_exc(), 'results': results}
        if owned:
            runner.group_status(group, error)
            if active_config is not None:
                directory = runner.output_group(group) / active_config['job_id']
                if not (directory / 'status.json').exists():
                    _journal(directory, active_config, 'FAIL_STOP', 0, {'R': 0, 'S': 0}, error)
        raise
    finally:
        # context退出后GPU锁退休、run_job finally结束其worker；只封小control。
        if owned:
            captured = capture_control(group, research_root=runner.RESEARCH, output_base=runner.ARCHIVE_ROOT / 'runs', final=True)
            print(json.dumps({'event': 'control_sealed', **captured}), flush=True)
