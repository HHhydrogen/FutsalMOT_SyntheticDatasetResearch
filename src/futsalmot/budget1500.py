"""独立公开起点的1500步R/批内混合：完整短程调度、同一路径与完整恢复。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from contextlib import ExitStack
from pathlib import Path
import hashlib,json,math,os,shutil,time,traceback
from types import FunctionType
from .runtime_backend import NativeBBoxBackend

SCHEMA='bbox_budget1500_v1'
SNAPSHOT_SCHEMA='bbox_budget1500_boundary_v1'
UPDATES=1500
SCHEDULE_POLICY='single_stage_lr001_warm100_cosine_v1'
def sampler_contract(strategy):
    require(strategy in ('R','R+S'),'只准独立1500 R/混合')
    mixed=strategy=='R+S'
    return {'R_per_update':8 if mixed else 16,'S_per_update':8 if mixed else 0,'R_extra_per_update':0,'microbatch_domains':['R','S']*4 if mixed else ['R']*8,'microbatches_per_update':2}


SOURCE_GROUP=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a01')
BASELINE_STATUS=Path('G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_seed17/status.json')

def require(value,message):
    if not value:raise ValueError(message)

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _digest(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'缺精确SHA256')
    return value

def evaluation_schedule():
    return [{'tag':'step000000_final_PT','phase':'final','phase_update':0,'global_update':0},
            {'tag':'step001500_final_final','phase':'final','phase_update':1500,'global_update':1500}]

def exposure_plan(strategy,update):
    require(type(update) is int and 0<=update<=UPDATES,'1500更新边界非法')
    c=sampler_contract(strategy)
    return {'R':c['R_per_update']*update,'S':c['S_per_update']*update}


def protocol(strategy,seed=17,*,plan_sha256,input_cache,baseline_R_status_reference,additional_space_bound_bytes,
             raw_upper_per_image_bytes,checkpoint_bytes,admission_sha256,max_run_seconds=5400):
    from . import runner,formal
    require(seed==17 and type(seed) is int,"本有限对照只准seed17");sampler_contract(strategy)
    _digest(plan_sha256);_digest(admission_sha256)
    ref=baseline_R_status_reference
    require(type(ref) is dict and set(ref)=={'path','sha256'} and Path(ref['path'])==BASELINE_STATUS,'必须明确既有R基线status引用')
    _digest(ref['sha256'])
    for value in (additional_space_bound_bytes,raw_upper_per_image_bytes,checkpoint_bytes,max_run_seconds):require(type(value) is int and value>0,'资源准入值须正整数')
    require(max_run_seconds<=5400,'1500独立单项最多90min')
    cache=formal.cache_specification(input_cache)
    return {'schema_version':SCHEMA,'job_id':('budget1500_R' if strategy=='R' else 'budget1500_R_plus_S')+'_seed17','strategy':strategy,'task':'detect','model':'yolo11s','seed':17,
      'updates':UPDATES,'phases':[{'name':'final','offset':0,'updates':UPDATES,'final_stage':True}],
      'N_R':2100,'N_S':24149,'N_val':750,'input_size':960,'numeric_mode':'fp32','microbatch_size':8,'effective_batch_size':16,'num_workers':2,
      'sampler_contract':sampler_contract(strategy),'terminal_exposures':exposure_plan(strategy,UPDATES),'total_image_exposures':24000,
      'initialization':'independent_public_yolo11s','optimizer':'SGD_Nesterov','schedule_policy':SCHEDULE_POLICY,'final_lr_peak':.001,'lrf':.01,'warmup_updates':100,
      'momentum':.937,'warmup_momentum':.8,'weight_decay':.0005,'gradient_clip_norm':10.,'domain_loss_reweighting':False,'stage_reset':False,
      'checkpoint_every':1000,'complete_checkpoint_count':3,'evaluation_every':None,'evaluation_policy':'phase_endpoints_v1','evaluation_schedule':evaluation_schedule(),
      'main_metric':'COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset','best_selection':False,'test_enabled':False,
      'mainline_all_logs_checkpoints_predictions_retained':True,'baseline_R_status_reference':deepcopy(ref),'input_cache':cache,
      'EXPERIMENTS_sha256':plan_sha256,'PAPER_sha256':runner.PAPER_SHA,'source_admission_sha256':admission_sha256,
      'storage_mode':'direct_g','control_space_bound_bytes':32*1024**2,'disk_reserve_bytes':runner.RESERVE,
      'max_run_seconds':max_run_seconds,'additional_space_bound_bytes':additional_space_bound_bytes,'raw_upper_per_image_bytes':raw_upper_per_image_bytes,
      'checkpoint_bytes':checkpoint_bytes,'deadline_utc':runner.DEADLINE,'automatic_resume':False,'infrastructure_retry_limit':1}

def check_protocol(config):
    args={k:config[k] for k in ('input_cache','baseline_R_status_reference','additional_space_bound_bytes','raw_upper_per_image_bytes','checkpoint_bytes','max_run_seconds')}
    expected=protocol(config['strategy'],config['seed'],plan_sha256=config['EXPERIMENTS_sha256'],admission_sha256=config['source_admission_sha256'],**args)
    require({k:config[k] for k in expected}==expected and set(config)==set(expected)|{'control_group'},'固定混合冻结协议变化')

def make_samplers(panel,synthetic,seed=17):
    from .formal import make_samplers as original
    require(seed==17,'只准seed17')
    return original(panel,synthetic,seed)

def _interleave(r,s):
    require(len(r)==len(s)==8,'每步必须8R8S')
    return [x for pair in zip(r,s) for x in pair]

def phase_requests(strategy,samplers,*,start,end):
    from .data import CycleSampler
    require(type(start) is int and type(end) is int and 0<=start<=end<=UPDATES,'请求区间非法')
    require(exposure_counts(samplers)==exposure_plan(strategy,start),'请求须从已提交边界生成')
    c=sampler_contract(strategy);clones={}
    for k in ('R','S'):
        uids,task,domain,seed=samplers[k]._template
        clones[k]=CycleSampler(uids,task=task,domain=domain,seed=seed);clones[k].load_state_dict(samplers[k].state_dict()['sampler'])
    result=[]
    for _ in range(start,end):
        r=clones['R'].take(c['R_per_update'])
        result.extend(_interleave(r,clones['S'].take(8)) if strategy=='R+S' else r)
    return result


def reserve_windows(strategy,samplers):
    c=sampler_contract(strategy)
    require(all(s.pending_batches==0 for s in samplers.values()),'事务不接受旧pending')
    try:return {k:samplers[k].reserve(c[k+'_per_update']) for k in ('R','S') if c[k+'_per_update']}
    except BaseException:
        for sampler in samplers.values():sampler.cancel_pending()
        raise


def window_requests(strategy,windows):
    sampler_contract(strategy)
    require(set(windows)==({'R','S'} if strategy=='R+S' else {'R'}),'预留流集合变化')
    return _interleave(windows['R']['samples'],windows['S']['samples']) if strategy=='R+S' else windows['R']['samples']


def validate_batches(strategy,batches,windows):
    require(len(batches)==2 and all(len(b['sample_lineage'])==8 for b in batches),'必须两个micro8')
    actual=[x['request'] for b in batches for x in b['sample_lineage']]
    require(actual==window_requests(strategy,windows),'完整UID/域/增强请求与预留不齐')
    for batch in batches:require(batch['domains']==sampler_contract(strategy)['microbatch_domains'],'micro域分组变化')


def commit_windows(strategy,samplers,windows):
    c=sampler_contract(strategy);keys=tuple(k for k in ('R','S') if c[k+'_per_update'])
    require(set(windows)==set(keys),'提交流集合变化')
    with ExitStack() as stack:
        for k in keys:stack.enter_context(samplers[k]._lock)
        candidates={}
        for k in keys:
            live=samplers[k];require(len(live._pending)==1 and live._pending[0]['request']==windows[k],'提交token/完整窗口不齐')
            require(windows[k]['count']==c[k+'_per_update'] and windows[k]['stream']==k,'预留计数/流变化')
            candidates[k]=live._clone(live._pending[0]['end_state'])
        for k in keys:
            samplers[k]._committed=candidates[k];samplers[k]._pending.popleft()
    return exposure_counts(samplers)


def exposure_counts(samplers):
    require(samplers['R_extra'].committed_draws==0,'固定混合禁止R_extra消费')
    return {k:samplers[k].committed_draws for k in ('R','S')}

def make_dataset(catalogue,requests,config):
    from .formal import make_training_dataset
    return make_training_dataset(catalogue,requests,config)

def schedule_1500(step,updates,*,final_stage):
    require(type(step) is int and updates==1500 and final_stage is True and 0<=step<=1500,'固定1500完整调度边界非法')
    k=100
    if step<=k:lr=.001*step/k;momentum=.8+(.937-.8)*step/k
    else:lr=.001*(.01+.99*(1+math.cos(math.pi*(step-k)/(updates-k)))/2);momentum=.937
    return {'step':step,'updates':updates,'warmup_updates':100,'lr':lr,'momentum':momentum,'ema_decay':.9999*(1-math.exp(-step/2000))}


# 原step_update代码原样复用，仅本函数独立globals固定warm100；不修改原模块。
_step_globals={**NativeBBoxBackend.step_update.__globals__,'schedule_at':schedule_1500}
_step=FunctionType(NativeBBoxBackend.step_update.__code__,_step_globals,NativeBBoxBackend.step_update.__name__,NativeBBoxBackend.step_update.__defaults__,NativeBBoxBackend.step_update.__closure__)
_step.__kwdefaults__=NativeBBoxBackend.step_update.__kwdefaults__


class Budget1500Backend(NativeBBoxBackend):
    step_update=_step
    def __init__(self,*args,**kwargs):
        require(kwargs.get('stage_updates')==1500 and kwargs.get('final_stage',True) is True,'只准独立单阶段1500')
        super().__init__(*args,**kwargs)
    def _config(self):return {**super()._config(),'schedule_policy':SCHEDULE_POLICY}


def make_backend(*,device='cuda:0'):
    from . import runner
    return Budget1500Backend.from_public(runner.WEIGHTS,model_size='s',stage_updates=UPDATES,final_stage=True,microbatch_size=8,numeric_mode='fp32',device=device,input_size=960,allow_initialized_cuda=device.startswith('cuda'))


def validate_snapshot(bundle,*,strategy,samplers,expected_bindings,expected_update,allow_cuda=False):
    import torch
    from .records import validate_checkpoint_contract,_binding_digest,CHECKPOINT_SCHEMA
    from .checkpoint_state import _model_states,_optimizer_protocol,_rng
    from .data import CycleSampler,PrefetchSampler
    validate_checkpoint_contract(bundle);require(type(expected_update) is int and 0<=expected_update<=UPDATES,'缺可信混合边界')
    require(bundle['schema_version']==CHECKPOINT_SCHEMA,'容器schema变化')
    require(bundle.get('run_scope')==SCHEMA and type(bundle.get('loader_iteration_started')) is bool,'缺训练loader边界与run_scope')
    loader_state=bundle.get('loader_generator_state')
    require(isinstance(loader_state,torch.Tensor) and loader_state.device.type=='cpu' and loader_state.dtype==torch.uint8 and loader_state.ndim==1,'缺CPU loader generator完整状态')
    torch.Generator(device='cpu').set_state(loader_state)
    require(bundle.get('snapshot_schema_version')==SNAPSHOT_SCHEMA and bundle.get('sampler_contract')==sampler_contract(strategy),'不是混合独立快照')
    require(_binding_digest(bundle['source_bindings'])==_binding_digest(expected_bindings),'来源绑定变化')
    require((bundle['phase'],bundle['phase_updates'],bundle['committed_updates'],bundle['numeric_mode'],bundle.get('effective_batch'))==('final',expected_update,expected_update,'fp32',16),'混合metadata边界不齐')
    require(bundle.get('cumulative_exposures')==exposure_plan(strategy,expected_update),'混合曝光不齐')
    require(bundle.get('runtime_backend',{}).get('schedule_policy')==SCHEDULE_POLICY and bundle['runtime_backend'].get('stage_updates')==UPDATES,'快照调度或完整长度变化')
    require(set(bundle['sampler'])==set(samplers)=={'R','S','R_extra'},'混合流集合变化')
    for key,live in samplers.items():
        require(type(live) is PrefetchSampler,'需要真实PrefetchSampler')
        saved=bundle['sampler'][key];require(set(saved)=={'schema_version','sampler'} and saved['schema_version']==PrefetchSampler.SCHEMA,'不能保存pending')
        uids,task,domain,seed=live._template;require(domain==key,'流身份变化')
        candidate=CycleSampler(uids,task=task,domain=domain,seed=seed);candidate.load_state_dict(saved['sampler'])
        require(candidate.draws==(0 if key=='R_extra' else sampler_contract(strategy)[key+'_per_update']*expected_update),'实际sampler游标不是8倍边界')
    _model_states(bundle['raw_model'],bundle['ema'],expected_update)
    opt=bundle['optimizer'];require(set(opt)>={'state','param_groups'} and isinstance(opt['state'],dict) and isinstance(opt['param_groups'],list) and opt['param_groups'],'缺完整SGD状态');_optimizer_protocol(opt)
    _rng(bundle['rng'],allow_cuda=allow_cuda,cuda_initialized=allow_cuda)

def snapshot(backend,*,strategy,samplers,source_bindings,committed_updates,extra_state=None):
    from .checkpoint_state import _clone_cpu
    from .records import CHECKPOINT_SCHEMA,capture_rng_state
    require(not backend.failed and backend.final_stage and backend.stage_update==committed_updates and backend.numeric_mode=='fp32','不能保存失败/非提交边界')
    require(backend._config()['stage_updates']==UPDATES and backend._config().get('schedule_policy')==SCHEDULE_POLICY and backend.microbatch_size==8,'backend配置变化')
    saved={k:v.state_dict() for k,v in samplers.items()};extra=deepcopy(extra_state or {});cuda=backend.device.type=='cuda'
    state={'schema_version':CHECKPOINT_SCHEMA,'snapshot_schema_version':SNAPSHOT_SCHEMA,'raw_model':backend.model.state_dict(),'ema':backend.ema_state,
      'optimizer':backend.optimizer.state_dict(),'phase':'final','phase_updates':committed_updates,'committed_updates':committed_updates,'source_bindings':source_bindings,
      'numeric_mode':'fp32','scaler':None,'sampler':saved,'cumulative_exposures':exposure_counts(samplers),'effective_batch':16,'sampler_contract':sampler_contract(strategy),'runtime_backend':backend._config()}
    require(not set(extra)&set(state) and 'rng' not in extra,'extra不能覆盖完整字段');state.update(extra)
    state=_clone_cpu(state,allow_cuda=cuda,cuda_initialized=cuda);state['rng']=capture_rng_state(include_cuda=cuda)
    validate_snapshot(state,strategy=strategy,samplers=samplers,expected_bindings=source_bindings,expected_update=committed_updates,allow_cuda=cuda)
    require(all(v.state_dict()==saved[k] for k,v in samplers.items()),'快照期间采样提交变化')
    return state

def restore(backend,bundle,*,strategy,samplers,expected_bindings,expected_update):
    import torch
    from .checkpoint_state import _clone_cpu
    from .records import capture_rng_state,restore_rng_state
    from .restore import _optimizer_topology
    from .optimizers import _check_optimizer
    require(not backend.failed and backend.stage_update==0 and not backend.optimizer.state,'只能fresh backend恢复')
    cuda=backend.device.type=='cuda';before=capture_rng_state(include_cuda=cuda)
    try:
        validate_snapshot(bundle,strategy=strategy,samplers=samplers,expected_bindings=expected_bindings,expected_update=expected_update,allow_cuda=cuda)
        require(bundle['runtime_backend']==backend._config(),'恢复backend配置不同');payload=_clone_cpu(bundle)
        _optimizer_topology(backend.optimizer,backend.model,payload['optimizer'])
        backend.model.load_state_dict(payload['raw_model'],strict=True,assign=False);backend.optimizer.load_state_dict(payload['optimizer']);_check_optimizer(backend.optimizer,torch)
        backend.ema_state={'schema_version':'futsalmot_ema_state_v1','updates':expected_update,'state_dict':{k:v.to(device=backend.device,copy=True) for k,v in payload['ema']['state_dict'].items()}}
        backend.stage_update=expected_update;backend.optimizer_steps_attempted=backend.optimizer_steps_returned=expected_update
        for k,v in samplers.items():v.load_state_dict(payload['sampler'][k])
        backend.model.train(True);restore_rng_state(payload['rng'],allow_cuda=cuda)
        excluded={'schema_version','snapshot_schema_version','raw_model','ema','optimizer','rng','sampler','phase','phase_updates','committed_updates','source_bindings','numeric_mode','scaler','cumulative_exposures','effective_batch','sampler_contract'}
        return {'phase_updates':expected_update,'committed_updates':expected_update,'cumulative_exposures':exposure_plan(strategy,expected_update),'extra_state':{k:v for k,v in payload.items() if k not in excluded},'RNG_restored_last':True}
    except BaseException:
        backend.failed=True;restore_rng_state(before,allow_cuda=cuda);raise

def execution_code(code_source=SOURCE_GROUP/'code/futsalmot'):
    source=Path(code_source);known=json.loads((SOURCE_GROUP/'code_bindings.json').read_bytes())
    require(len(known)==40 and {p.name:_sha(p) for p in source.glob('*.py')}==known,'必须精确A已冻40模块，不含Seg/Pose')
    return {**known,'budget1500.py':_sha(Path(__file__))}

def check_baseline(reference):
    require(Path(reference['path'])==BASELINE_STATUS and _sha(BASELINE_STATUS)==reference['sha256'],'原R基线status变化')
    status=json.loads(BASELINE_STATUS.read_bytes());require(status['state']=='COMPLETE' and status['update']==6563,'原R基线非完整6563')
    return status

def prepare_group(group,*,config,admission=None,code_source=SOURCE_GROUP/'code/futsalmot'):
    from . import runner,formal
    from .runner_data import real_panel
    from .catalogues import synthetic_bbox_catalogue
    from .input_cache import validate_ready_cache
    from .output_storage import create_output_binding,capture_control
    group=Path(group).absolute();require(group.parent==runner.RESEARCH/'runs' and group.is_dir() and not (group/'job_table.json').exists(),'组位置非法或已准备')
    config=deepcopy(config);config['control_group']=str(group);check_protocol(config);check_baseline(config['baseline_R_status_reference'])
    actual=json.loads((group/'admission.json').read_bytes());require(admission is None or actual==admission,'准入收据不一致')
    require(actual.get('schema_version')=='bbox_budget1500_admission_v1' and actual.get('status')=='PASS','缺实际mixed PASS准入')
    require((actual.get('numeric_mode'),actual.get('microbatch_size'),actual.get('num_workers'),actual.get('effective_batch'))==('fp32',8,2,16),'准入数值不同')
    require(actual.get('strategy')==config['strategy'] and actual.get('updates')==UPDATES and actual.get('sampler_contract')==sampler_contract(config['strategy']) and actual.get('input_cache')==config['input_cache'] and actual.get('baseline_R_status_reference')==config['baseline_R_status_reference'],'准入mixed/缓存/R引用不同')
    require(actual.get('checkpoint_every')==1000 and actual.get('diagnostic_updates')==3,'准入检查点/真实诊断次数不符')
    require(actual.get('measurements',{}).get('temporary_models_removed') is True and actual.get('schedule_policy')==SCHEDULE_POLICY,'诊断模型未清或调度变化')
    exact=actual.get('measurements',{}).get('exact_restore',{})
    require(exact and all(value is True for value in exact.values()) and exact.get('loss_and_metrics') is True and exact.get('actual_gradient_tensors') is True,'未实际通过完整恢复/下一步loss及梯度exact')
    require(_sha(group/'admission.json')==config['source_admission_sha256'],'准入SHA不同')
    for name,key in [('EXPERIMENTS.md','EXPERIMENTS_sha256'),('PAPER.md','PAPER_sha256')]:require(_sha(runner.RESEARCH/'docs/plan'/name)==config[key],'封存方案变化')
    panel=real_panel(runner.REAL);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');require((len(panel['train_uids']),len(panel['val_uids']),len(synthetic['ordered_uids']))==(2100,750,24149),'数据数量变化')
    for key,value in [('real_panel',panel['binding']),('synthetic_panel',synthetic['binding'])]:require(runner.canonical(actual['source_bindings'][key])==runner.canonical(value),'实际准入来源不同:'+key)
    validate_ready_cache(config['input_cache'],synthetic['binding']);hashes=execution_code(code_source);require(actual.get('executed_code_sha256')==hashes,'实际执行代码与冻结代码不同')
    require(actual['source_bindings'].get('requirements_sha256')==_sha(runner.RESEARCH/'requirements.lock'),'预检依赖锁变化')
    public={'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':_sha(runner.WEIGHTS/'yolo11s.pt')};require(actual['source_bindings'].get('public_model')==public,'公开起点绑定不同')
    require(time.time()+config['max_run_seconds']+600<datetime.fromisoformat(config['deadline_utc']).timestamp(),'全局截止无法闭合');runner.production_closed();runner.check_disk(32*1024**2);runner.check_archive_space(config['additional_space_bound_bytes'])
    target=group/'code/futsalmot';target.mkdir(parents=True,exist_ok=False)
    for name in hashes:shutil.copyfile(Path(__file__) if name=='budget1500.py' else Path(code_source)/name,target/name)
    require({p.name:_sha(p) for p in target.glob('*.py')}==hashes,'冻结期间源码变化');shutil.copyfile(runner.RESEARCH/'requirements.lock',group/'code/requirements.lock')
    bindings={'real_panel':panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':public,'source_code':hashes,'requirements_sha256':_sha(group/'code/requirements.lock'),'EXPERIMENTS_sha256':config['EXPERIMENTS_sha256'],'PAPER_sha256':config['PAPER_sha256'],'input_cache':config['input_cache'],'baseline_R_status_reference':config['baseline_R_status_reference']}
    runner.write_once(group/'code_bindings.json',hashes);runner.write_once(group/'job_table.json',{'schema_version':SCHEMA,'jobs':[config],'source_bindings':bindings,'test_enabled':False,'extra_jobs_authorized':False})
    launch="import os,sys\nfrom pathlib import Path\nos.environ['YOLO_OFFLINE']='true'\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\ngroup=Path(__file__).absolute().parent\nsys.path.insert(0,str(group/'code'))\nif __name__=='__main__':\n from futsalmot.budget1500 import controller\n controller(group)\n"
    with (group/'launch.py').open('x',encoding='utf8',newline='\n') as f:f.write(launch)
    location=create_output_binding(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs');snapshot=capture_control(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs')
    return {'group':str(group),'output_group':location['output_group'],'source_bindings':bindings,'job_table_sha256':_sha(group/'job_table.json'),'snapshot':snapshot}

def _journal(directory,config,state,update,result=None):
    from . import runner
    text=f"\n\n## BUDGET1500：{config['job_id']} {state}\n\nUTC {runner.now()}。独立公开YOLO11s/seed17，策略{config['strategy']}，{update}/1500成功更新；实际曝光{exposure_plan(config['strategy'],update)}。SoccerNet train2100/val750/S24149，960FP32/micro8×2/workers2，单SGD/EMA，100步warmup .001完整cosine至.00001；0/1000/1500三份完整cp、0/末两次真实750验证，全部逐步/原生预测/失败保留，不best/test。结果：{json.dumps(runner.plain(result),ensure_ascii=False)}。证据：{Path(directory).as_posix()}。\n"
    with (runner.RESEARCH/'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())


def run_job(group,config):
    import torch,psutil
    from . import runner,formal
    from .runner_data import make_loader
    from .records import RunRecorder,save_checkpoint,capture_rng_state
    from .checkpoint_state import _clone_cpu
    strategy=config['strategy'];group=Path(group).absolute();check_protocol(config);table=json.loads((group/'job_table.json').read_bytes());require(config in table['jobs'],'混合作业不在冻结表')
    lock=json.loads((runner.RESEARCH/'.research_gpu.lock').read_bytes());require(lock['pid']==os.getpid() and abs(lock['create_time']-psutil.Process().create_time())<.001 and Path(lock['group'])==group,'须持本组准确GPU锁')
    require(config['input_cache']==table['source_bindings']['input_cache'],'协议与source缓存绑定不同')
    panel,synthetic=formal._verify_bindings(group,table);check_baseline(config['baseline_R_status_reference']);require(_sha(group/'admission.json')==config['source_admission_sha256'],'准入变化');runner.check_write_space(config,config['additional_space_bound_bytes'])
    directory=runner.output_group(group)/config['job_id'];bindings={**table['source_bindings'],'protocol':deepcopy(config)}
    recorder=RunRecorder(directory,{'schema_version':SCHEMA,'run_id':config['job_id'],'source_bindings':bindings,'protocol':config,'storage_mode':'direct_g'});(directory/'checkpoints').mkdir()
    deadline=min(time.time()+config['max_run_seconds'],datetime.fromisoformat(config['deadline_utc']).timestamp());runner.write_once(directory/'runtime_limit.json',{'started_at_utc':runner.now(),'deadline_unix':deadline,'max_run_seconds':config['max_run_seconds']})
    backend=None;samplers=None;gen=None;iterator=None;latest=0
    def state(value):return {'state':value,'pid':os.getpid(),'create_time':psutil.Process().create_time(),'update':latest,'updates':UPDATES,'phase':'final','phase_update':latest,'exposures':exposure_plan(strategy,latest)}
    def save_snapshot(tag,started):
        bundle=snapshot(backend,strategy=strategy,samplers=samplers,source_bindings=bindings,committed_updates=latest,extra_state={'loader_generator_state':gen.get_state().clone(),'loader_iteration_started':started,'run_scope':SCHEMA})
        receipt=save_checkpoint(directory/'checkpoints'/f'{latest:06d}_final_{latest:06d}_{tag}.pt',bundle);row={'kind':'checkpoint','update':latest,'phase':'final','phase_update':latest,'tag':tag,'receipt':asdict(receipt),'saved_at_utc':runner.now()};recorder.event(row);runner.append(directory/'checkpoints.jsonl',row)
    try:
        runner.seed_all(17);backend=make_backend();samplers=make_samplers(panel,synthetic);gen=torch.Generator(device='cpu').manual_seed(170017)
        save_snapshot('initial',False);runner.status(directory,state('RUNNING'));_journal(directory,config,'STARTED',0)
        formal._evaluate(directory,backend,panel,config,gen,recorder,'final',0,deadline)
        requests=phase_requests(strategy,samplers,start=0,end=UPDATES);dataset=make_dataset({**panel['train_catalogue'],**synthetic['catalogue']},requests,config)
        loader=make_loader(dataset,batch_size=8,num_workers=2,generator=gen,pin_memory=True,allow_initialized_cuda=True);loader.timeout=300;iterator=formal.make_phase_iterator(loader,gen,iteration_already_started=False)
        for local in range(1,UPDATES+1):
            if runner.stop_requested(group,directory):save_snapshot('user_stop',True);runner.status(directory,state('USER_STOP'));_journal(directory,config,'USER_STOP',latest);return {**state('USER_STOP'),'directory':str(directory)}
            if time.time()>=deadline:raise TimeoutError('LIMIT_STOP budget1500 finite runtime')
            runner.check_write_space(config,2*config['checkpoint_bytes'])
            if torch.cuda.mem_get_info()[0]<1024**3:raise RuntimeError('GPU free below1GiB')
            windows=reserve_windows(strategy,samplers);begin=time.monotonic();batches=runner._microbatches(iterator,2);validate_batches(strategy,batches,windows)
            metrics=backend.step_update(batches,update=local);total=commit_windows(strategy,samplers,windows);latest=local;require(total==exposure_plan(strategy,latest),'混合提交曝光不齐')
            metrics.update(update=local,phase_update=local,phase='final',domain=strategy,sampling_streams=list(windows),gradient_norm=metrics['gradient_norm_before_clip'],R_exposures=total['R'],S_exposures=total['S'],wall_seconds=time.monotonic()-begin,geometry=[x for b in batches for x in b['geometry']],microbatch_domains=[b['domains'] for b in batches],sampler_contract=sampler_contract(strategy),maximum_cuda_allocated=torch.cuda.max_memory_allocated(),maximum_cuda_reserved=torch.cuda.max_memory_reserved())
            recorder.step(metrics)
            if runner.checkpoint_due(local,UPDATES,1000):save_snapshot('final' if local==UPDATES else 'interval',True)
            if local%25==0 or local==UPDATES:runner.production_closed();runner.status(directory,{**state('RUNNING'),'loss':metrics['loss']});print(json.dumps({'event':'update','run':config['job_id'],'update':local,'updates':UPDATES,'loss':metrics['loss'],'update_s':metrics['wall_seconds']}),flush=True)
            del batches
        runner._shutdown(iterator);iterator=None;del loader,dataset,requests
        terminal=formal._evaluate(directory,backend,panel,config,gen,recorder,'final',UPDATES,deadline)
        result={**state('COMPLETE'),'schema_version':SCHEMA,'job_id':config['job_id'],'directory':str(directory),'terminal':terminal,'terminal_evaluation':terminal};runner.status(directory,result);recorder.event({'kind':'run_complete','update':latest});_journal(directory,config,'COMPLETE',latest,result);return result
    except BaseException as exc:
        error={**state('LIMIT_STOP' if isinstance(exc,TimeoutError) else 'USER_STOP' if runner.stop_requested(group,directory) else 'FAIL_STOP'),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
        try:
            if samplers is not None:
                for live in samplers.values():live.cancel_pending()
            if backend is not None and not backend.failed and backend.stage_update==latest:save_snapshot('controlled_failure',iterator is not None)
            elif backend is not None:
                with (directory/'failed_state_evidence.pt').open('xb') as f:torch.save(_clone_cpu({'schema_version':'budget1500_failure_evidence_v1','resume_allowed':False,'raw':backend.model.state_dict(),'ema':backend.ema_state,'optimizer':backend.optimizer.state_dict(),'rng':capture_rng_state(include_cuda=True),'samplers':{k:v.state_dict() for k,v in samplers.items()},'failure':backend.failure,'source_bindings':bindings,'committed_updates':latest},allow_cuda=True,cuda_initialized=True),f);f.flush();os.fsync(f.fileno())
        except BaseException as save_error:error['failure_evidence_error']=repr(save_error)
        runner.status(directory,error);_journal(directory,config,error['state'],latest,error)
        if error['state'] in ('USER_STOP','LIMIT_STOP'):return {**error,'directory':str(directory)}
        raise
    finally:
        runner._shutdown(iterator)
        if backend is not None:del backend
        torch.cuda.empty_cache()

def controller(group):
    from . import runner
    from .output_storage import capture_control
    group=Path(group).absolute();table=json.loads((group/'job_table.json').read_bytes());require(table['schema_version']==SCHEMA and len(table['jobs'])==1 and not table['test_enabled'],'不是单个固定混合作业')
    require(not (group/'status.json').exists() and not (runner.output_group(group)/'control/CONTROL_MANIFEST.json').exists(),'已派发/封闭组不可重放')
    admission=json.loads((group/'admission.json').read_bytes());require(admission['status']=='PASS','准入未通过');owned=False;results=[]
    try:
        with runner.gpu_lock(group):
            owned=True;runner.setup_cuda(admission);cfg=table['jobs'][0];runner.group_status(group,{'state':'RUNNING','pid':os.getpid(),'create_time':__import__('psutil').Process().create_time(),'active_job':cfg['job_id'],'results':[]})
            result=run_job(group,cfg);results.append(result);runner.group_status(group,{'state':'BUDGET1500_COMPLETE' if result['state']=='COMPLETE' else result['state'],'results':results,'test_enabled':False})
    except BaseException as exc:
        if owned:runner.group_status(group,{'state':'FAIL_STOP','error':str(exc),'traceback':traceback.format_exc(),'results':results})
        raise
    finally:
        if owned:
            sealed=capture_control(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs',final=True);print(json.dumps({'event':'control_sealed',**sealed}),flush=True)
