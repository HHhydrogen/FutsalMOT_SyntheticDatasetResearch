"""独立公开起点的短SR/RR配对：1500+5063，固定100warmup/1000完整检查点。

沿原NativeBBoxBackend的同一step_update代码，仅明确替换schedule_at全局依赖；
原正式循环仅将backend导入改成本派生后端、检查点间隔改1000，其余循环原样。
不初始化GPU，不自动派发，不修改已有任何数值源码或历史冻包。
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import FunctionType
import hashlib,json,math,os,shutil,time,traceback
from . import formal
from .runtime_backend import NativeBBoxBackend
from .training import domain_at

SCHEMA='bbox_short_pair_v1'
SCHEDULE_POLICY='both_stages_lr001_warm100_cosine_v1'
U_S=1500
U_R=5063
UPDATES=6563
SOURCE_GROUP=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a01')
BASELINE_STATUS=Path('G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_seed17/status.json')


def require(value,message):
    if not value:raise ValueError(message)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'缺小写SHA256')
    return value


def short_schedule(step,updates,*,final_stage):
    require(type(step) is int and type(updates) is int and type(final_stage) is bool,'调度类型须明确')
    require(updates==(U_R if final_stage else U_S) and 0<=step<=updates,'短阶段调度边界非法')
    k=100
    if step<=k:
        lr=.001*step/k;momentum=.8+(.937-.8)*step/k
    else:
        lr=.001*(.01+.99*(1+math.cos(math.pi*(step-k)/(updates-k)))/2);momentum=.937
    return {'step':step,'updates':updates,'warmup_updates':k,'lr':lr,'momentum':momentum,'ema_decay':.9999*(1-math.exp(-step/2000))}


# 原方法code对象完全复用，只有schedule_at绑定为本独立协议；不改原module globals。
_step_globals={**NativeBBoxBackend.step_update.__globals__,'schedule_at':short_schedule}
_short_step=FunctionType(NativeBBoxBackend.step_update.__code__,_step_globals,NativeBBoxBackend.step_update.__name__,NativeBBoxBackend.step_update.__defaults__,NativeBBoxBackend.step_update.__closure__)
_short_step.__kwdefaults__=NativeBBoxBackend.step_update.__kwdefaults__


class ShortPairBackend(NativeBBoxBackend):
    step_update=_short_step
    def __init__(self,*args,**kwargs):
        require(kwargs.get('stage_updates')==(U_R if kwargs.get('final_stage',True) else U_S),'后端只接受1500/5063绝对短阶段')
        super().__init__(*args,**kwargs)
    def _config(self):
        return {**super()._config(),'schedule_policy':SCHEDULE_POLICY}


def phases(strategy):
    require(strategy in ('S->R','R->R'),'只准短SR/RR')
    return formal.phases(strategy,U_S,U_R)


def evaluation_schedule(strategy):return formal.evaluation_schedule(strategy,U_S,U_R)


def exposure_plan(strategy,update):
    require(strategy in ('S->R','R->R') and type(update) is int and 0<=update<=UPDATES,'策略或边界非法')
    s=min(update,U_S) if strategy=='S->R' else 0
    return {'R':16*(update-s),'S':16*s}


def phase_requests(strategy,samplers,*,start,end):
    return formal.phase_requests(strategy,samplers,start=start,end=end,u_s=U_S,u_r=U_R)


def transition_backend(backend,u_r=U_R):
    require(u_r==U_R and not backend.failed and not backend.final_stage and backend.stage_update==U_S,'第一阶段未完整提交')
    return ShortPairBackend(backend.network,stage_updates=u_r,final_stage=True,microbatch_size=backend.microbatch_size,numeric_mode=backend.numeric_mode,device=str(backend.device),input_size=backend.input_size,allow_initialized_cuda=backend.device.type=='cuda')


def protocol(strategy,seed=17,*,plan_sha256,input_cache,baseline_R_status_reference,additional_space_bound_bytes,raw_upper_per_image_bytes,checkpoint_bytes,admission_sha256,max_run_seconds=10800):
    from . import runner
    require(strategy in ('S->R','R->R') and type(seed) is int and seed==17,'本次只两个短阶段seed17')
    _digest(plan_sha256);_digest(admission_sha256)
    ref=baseline_R_status_reference
    require(type(ref) is dict and set(ref)=={'path','sha256'} and Path(ref['path'])==BASELINE_STATUS,'缺强R精确status引用');_digest(ref['sha256'])
    for value in (additional_space_bound_bytes,raw_upper_per_image_bytes,checkpoint_bytes,max_run_seconds):require(type(value) is int and value>0,'资源须实际正整数')
    require(max_run_seconds<=10800,'单项最多3小时')
    return {'schema_version':SCHEMA,'scope':'independent_public_short_pair','job_id':('short2_S_to_R' if strategy=='S->R' else 'short2_R_to_R')+'_seed17','strategy':strategy,'task':'detect','model':'yolo11s','seed':17,'updates':UPDATES,'U_S':U_S,'U_R':U_R,'phases':phases(strategy),'terminal_exposures':exposure_plan(strategy,UPDATES),'total_image_exposures':105008,
      'N_R':2100,'N_S':24149,'N_val':750,'input_size':960,'numeric_mode':'fp32','microbatch_size':8,'effective_batch_size':16,'num_workers':2,'optimizer':'SGD_Nesterov','schedule_policy':SCHEDULE_POLICY,'first_lr_peak':.001,'final_lr_peak':.001,'warmup_updates_per_stage':100,'warmup_momentum':.8,'lrf':.01,'momentum':.937,'weight_decay':.0005,'gradient_clip_norm':10.,'initialization':'independent_public_yolo11s','phase_transition':'raw_then_new_SGD_and_EMA','final_R_stream_starts_at_zero':True,
      'checkpoint_every':1000,'complete_checkpoint_count':10,'evaluation_every':None,'evaluation_policy':'phase_endpoints_v1','evaluation_schedule':evaluation_schedule(strategy),'save_phase_zero':True,'evaluate_stage0':True,'main_metric':'COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset','best_selection':False,'test_enabled':False,'mainline_all_logs_checkpoints_predictions_retained':True,'baseline_R_status_reference':deepcopy(ref),'input_cache':formal.cache_specification(input_cache),
      'EXPERIMENTS_sha256':plan_sha256,'PAPER_sha256':runner.PAPER_SHA,'source_admission_sha256':admission_sha256,'storage_mode':'direct_g','control_space_bound_bytes':32*1024**2,'disk_reserve_bytes':runner.RESERVE,'max_run_seconds':max_run_seconds,'additional_space_bound_bytes':additional_space_bound_bytes,'raw_upper_per_image_bytes':raw_upper_per_image_bytes,'checkpoint_bytes':checkpoint_bytes,'deadline_utc':runner.DEADLINE,'automatic_resume':False,'infrastructure_retry_limit':1}


def check_protocol(config):
    keys=('input_cache','baseline_R_status_reference','additional_space_bound_bytes','raw_upper_per_image_bytes','checkpoint_bytes','max_run_seconds')
    expected=protocol(config['strategy'],config['seed'],plan_sha256=config['EXPERIMENTS_sha256'],admission_sha256=config['source_admission_sha256'],**{k:config[k] for k in keys})
    require(set(config)==set(expected)|{'control_group'} and {k:config[k] for k in expected}==expected,'短两阶段冻结协议变化')


def execution_code(code_source=SOURCE_GROUP/'code/futsalmot'):
    source=Path(code_source);known=json.loads((SOURCE_GROUP/'code_bindings.json').read_bytes())
    require(len(known)==40 and {p.name:_sha(p) for p in source.glob('*.py')}==known,'只准旧A精确40数值源码作为来源，不允许Seg/Pose或其它新增模块')
    return {**known,'short_pair.py':_sha(Path(__file__))}


def _check_baseline(ref):
    require(Path(ref['path'])==BASELINE_STATUS and _sha(BASELINE_STATUS)==ref['sha256'],'强R参考变化')
    status=json.loads(BASELINE_STATUS.read_bytes());require(status['state']=='COMPLETE' and status['update']==6563,'强R终点不完整')


def prepare_group(group,*,config,admission=None,code_source=SOURCE_GROUP/'code/futsalmot'):
    from . import runner
    from .runner_data import real_panel
    from .catalogues import synthetic_bbox_catalogue
    from .input_cache import validate_ready_cache
    from .output_storage import create_output_binding,capture_control
    group=Path(group).absolute();require(group.parent==runner.RESEARCH/'runs' and group.is_dir() and not (group/'job_table.json').exists(),'组位置非法或已准备')
    config=deepcopy(config);config['control_group']=str(group);check_protocol(config);_check_baseline(config['baseline_R_status_reference'])
    actual=json.loads((group/'admission.json').read_bytes());require(admission is None or actual==admission,'实际准入收据变化')
    require(actual.get('schema_version')=='bbox_short_pair_admission_v1' and actual.get('status')=='PASS' and actual.get('strategy')==config['strategy'],'缺对应策略实际PASS准入')
    require((actual.get('numeric_mode'),actual.get('microbatch_size'),actual.get('num_workers'),actual.get('effective_batch'))==('fp32',8,2,16),'准入数值/loader不同')
    require(actual.get('schedule_policy')==SCHEDULE_POLICY and actual.get('checkpoint_every')==1000,'准入调度或保存频率不同')
    require(actual.get('input_cache')==config['input_cache'] and actual.get('baseline_R_status_reference')==config['baseline_R_status_reference'],'准入缓存/R引用不同')
    require(_sha(group/'admission.json')==config['source_admission_sha256'],'准入SHA不同')
    for name,key in [('EXPERIMENTS.md','EXPERIMENTS_sha256'),('PAPER.md','PAPER_sha256')]:require(_sha(runner.RESEARCH/'docs/plan'/name)==config[key],'封存方案变化')
    panel=real_panel(runner.REAL);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');require((len(panel['train_uids']),len(panel['val_uids']),len(synthetic['ordered_uids']))==(2100,750,24149),'数据数量不同')
    for key,value in [('real_panel',panel['binding']),('synthetic_panel',synthetic['binding'])]:require(runner.canonical(actual['source_bindings'][key])==runner.canonical(value),'实际准入来源变化:'+key)
    validate_ready_cache(config['input_cache'],synthetic['binding']);hashes=execution_code(code_source);require(actual.get('executed_code_sha256')==hashes,'实际准入源码集合/SHA与冻结不同')
    require(actual['source_bindings'].get('requirements_sha256')==_sha(runner.RESEARCH/'requirements.lock'),'预检依赖锁变化')
    public={'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':_sha(runner.WEIGHTS/'yolo11s.pt')};require(actual['source_bindings'].get('public_model')==public,'公开起点不同')
    require(time.time()+config['max_run_seconds']+600<datetime.fromisoformat(config['deadline_utc']).timestamp(),'全局界限无法闭合');runner.production_closed();runner.check_disk(32*1024**2);runner.check_archive_space(config['additional_space_bound_bytes'])
    target=group/'code/futsalmot';target.mkdir(parents=True,exist_ok=False)
    for name in hashes:shutil.copyfile(Path(__file__) if name=='short_pair.py' else Path(code_source)/name,target/name)
    require({p.name:_sha(p) for p in target.glob('*.py')}==hashes,'冻结复制期间源码变化');shutil.copyfile(runner.RESEARCH/'requirements.lock',group/'code/requirements.lock')
    bindings={'real_panel':panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':public,'source_code':hashes,'requirements_sha256':_sha(group/'code/requirements.lock'),'EXPERIMENTS_sha256':config['EXPERIMENTS_sha256'],'PAPER_sha256':config['PAPER_sha256'],'input_cache':config['input_cache'],'baseline_R_status_reference':config['baseline_R_status_reference']}
    runner.write_once(group/'code_bindings.json',hashes);runner.write_once(group/'job_table.json',{'schema_version':SCHEMA,'jobs':[config],'source_bindings':bindings,'test_enabled':False,'extra_jobs_authorized':False})
    launch="import os,sys\nfrom pathlib import Path\nos.environ['YOLO_OFFLINE']='true'\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\ngroup=Path(__file__).absolute().parent\nsys.path.insert(0,str(group/'code'))\nif __name__=='__main__':\n from futsalmot.short_pair import controller\n controller(group)\n"
    with (group/'launch.py').open('x',encoding='utf8',newline='\n') as f:f.write(launch)
    location=create_output_binding(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs');snapshot=capture_control(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs')
    return {'group':str(group),'output_group':location['output_group'],'source_bindings':bindings,'job_table_sha256':_sha(group/'job_table.json'),'snapshot':snapshot}


def _journal(directory,config,state,global_update,exposures,result=None):
    from . import runner
    text=f"\n\n## SHORT-PAIR：{config['job_id']} {state}\n\nUTC {runner.now()}。独立公开YOLO11s/seed17，{config['strategy']}1500+5063/总6563；实际已提交{global_update}/6563，R/S曝光{exposures['R']}/{exposures['S']}。960FP32/micro8有效16/workers2，两阶段各warm100/LR.001→.00001，原SGD/EMA/损失/增强；raw承接、新SGD EMA。0/每1000/每阶段末共10计划完整cp，4端点真实750验证及逐步/原生预测/全部失败保留，不best/test，不因SR低分取消RR。结果：{json.dumps(runner.plain(result),ensure_ascii=False)}。证据：{Path(directory).as_posix()}。\n"
    with (runner.RESEARCH/'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())


def _save_boundary(directory,backend,samplers,bindings,gen,recorder,*,global_update,phase,tag,loader_iteration_started=False):
    from . import runner
    from .records import save_checkpoint
    require(exposure_counts(samplers)==exposure_plan(bindings['protocol']['strategy'],global_update),'完整边界sampler曝光不匹配')
    bundle=backend.snapshot(samplers=samplers,source_bindings=bindings,committed_updates=global_update,cumulative_exposures=exposure_counts(samplers),phase=phase,extra_state={'loader_generator_state':gen.get_state().clone(),'run_scope':SCHEMA,'global_update':global_update,'formal_phase':phase,'loader_iteration_started':loader_iteration_started})
    path=Path(directory)/'checkpoints'/f'{global_update:06d}_{phase}_{backend.stage_update:06d}_{tag}.pt';receipt=save_checkpoint(path,bundle)
    row={'kind':'checkpoint','update':global_update,'phase':phase,'phase_update':backend.stage_update,'tag':tag,'receipt':asdict(receipt),'saved_at_utc':runner.now()};recorder.event(row);runner.append(Path(directory)/'checkpoints.jsonl',row)
    return receipt


_require=require
_check_protocol=check_protocol
_verify_bindings=formal._verify_bindings
make_samplers=formal.make_samplers
make_training_dataset=formal.make_training_dataset
make_phase_iterator=formal.make_phase_iterator
exposure_counts=formal.exposure_counts
_evaluate_if_due=formal._evaluate_if_due

def run_job(group, config, *, resume_receipt=None):
    """调用方须已持GPU锁/初始化CUDA；resume仅由明确恢复调用提供可信收据。"""
    import torch
    import psutil
    from . import runner
    from .runner_data import RequestPlanDataset, make_loader
    from .short_pair import ShortPairBackend as NativeBBoxBackend
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
                    if runner.checkpoint_due(local, count, 1000):
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
            runner.group_status(group, {'state': 'SHORT_PAIR_COMPLETE', 'results': results, 'test_enabled': False})
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
