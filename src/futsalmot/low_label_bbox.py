"""N25=525真实图的独立R/混合前端；保留原1500数值与完整记录，不提供准入捷径。"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import FunctionType
import ast,hashlib,inspect,json,os,shutil,time,traceback
from . import budget1500 as base

SCHEMA='bbox_low_label25_v1'
SNAPSHOT_SCHEMA='bbox_low_label25_boundary_v1'
SUBSET_SCHEMA='bbox_real_subset_v1'
UPDATES=base.UPDATES
SCHEDULE_POLICY=base.SCHEDULE_POLICY
SOURCE_GROUP=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01')
BASELINE_STATUS=base.BASELINE_STATUS
BASE_MODULE_SHA='659d949099537e3fbfcbfb74936a77d4f83e8c6284d83058871a5bd348f47833'
require=base.require
_sha=base._sha
_digest=base._digest
sampler_contract=base.sampler_contract
evaluation_schedule=base.evaluation_schedule
exposure_plan=base.exposure_plan
phase_requests=base.phase_requests
reserve_windows=base.reserve_windows
window_requests=base.window_requests
validate_batches=base.validate_batches
commit_windows=base.commit_windows
exposure_counts=base.exposure_counts
make_dataset=base.make_dataset
make_backend=base.make_backend
Budget1500Backend=base.Budget1500Backend
schedule_1500=base.schedule_1500
check_baseline=base.check_baseline


def _pool_sha(uids):return hashlib.sha256(json.dumps(tuple(uids),ensure_ascii=True).encode()).hexdigest()

def subset_reference(reference):
    require(type(reference) is dict and set(reference)=={'path','sha256'},'须精确subset绝对path/SHA引用')
    p=Path(reference['path']);require(p.is_absolute() and '..' not in p.parts and not str(p).startswith(('\\\\','//')),'subset必须本地绝对普通路径')
    _digest(reference['sha256'])
    return {'path':str(p),'sha256':reference['sha256']}

def load_subset(reference,full_panel):
    from . import runner
    from .records import _strict_json
    ref=subset_reference(reference);path=Path(ref['path'])
    require(not path.is_symlink() and not path.is_junction(),'subset不能是链接')
    before=path.stat();require(0<before.st_size<=16*1024**2,'subset小manifest超界')
    raw=path.read_bytes();after=path.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and hashlib.sha256(raw).hexdigest()==ref['sha256'],'subset字节/摘要发生变化')
    d=_strict_json(raw);require(type(d) is dict and d.get('schema_version')==SUBSET_SCHEMA and d.get('status')=='FROZEN','subset未冻结或schema不符')
    require((d.get('full_N_R'),d.get('N_R'),d.get('real_fraction'))==(2100,525,.25),'仅准入N25=525/2100')
    full=list(full_panel['train_uids']);require(len(full)==len(set(full))==2100 and set(full_panel['train_catalogue'])==set(full),'full real train源非2100唯一池')
    require(runner.canonical(d.get('source_binding'))==runner.canonical(full_panel['binding']),'subset原real panel完整绑定不同')
    require(d.get('full_ordered_train_pool_sha256')==_pool_sha(full)==full_panel['binding']['ordered_train_pool_sha256'],'原2100有序池SHA变化')
    chosen=d.get('ordered_train_uids');require(type(chosen) is list and len(chosen)==len(set(chosen))==525 and all(type(u) is str and u in full_panel['train_catalogue'] for u in chosen),'subset UID重复/非train/数量不符')
    require(chosen==[u for u in full if u in set(chosen)] and d.get('ordered_train_pool_sha256')==_pool_sha(chosen),'subset顺序或有序UID SHA不符')
    policy=d.get('selection_policy');require(type(policy) is dict and type(policy.get('algorithm')) is str and policy['algorithm'] and policy.get('uses_validation_scores') is False and policy.get('uses_test') is False,'缺事前群组选择算法或违规使用val/test')
    blocks=d.get('group_blocks');require(type(blocks) is list and len(blocks)==420,'须完整420个冻结连续5图block')
    all_uids=[];selected=[];ids=set();clip_counts={};clip_selected={};selected50=[];clip50={}
    for row in blocks:
        require(type(row) is dict and type(row.get('block_id')) is str and row['block_id'] not in ids and type(row.get('match_id')) is str and row['match_id'] and type(row.get('clip_uid')) is str and row['clip_uid'] and type(row.get('selected')) is bool,'block身份/比赛/clip缺失或重复')
        ids.add(row['block_id']);uids=row.get('ordered_uids');require(type(uids) is list and uids and len(uids)==len(set(uids)) and all(type(u) is str for u in uids),'block须非空唯一有序UID')
        require(len(uids)==5,'本N25冻结block必须完整5图')
        clip=row['clip_uid'];clip_counts[clip]=clip_counts.get(clip,0)+1;clip_selected[clip]=clip_selected.get(clip,0)+int(row['selected'])
        if 'selected_50' in row:
            require(type(row['selected_50']) is bool and (not row['selected'] or row['selected_50']),'50%设计须bool并嵌套25%')
            clip50[clip]=clip50.get(clip,0)+int(row['selected_50'])
            if row['selected_50']:selected50+=uids
        all_uids+=uids
        if row['selected']:selected+=uids
    require(all_uids==full and len(set(all_uids))==2100,'blocks须按原顺序互斥完整覆盖2100，不得丢帧/重叠')
    require(selected==chosen,'只能选择完整冻结block，禁止块内挑帧')
    require(len(clip_counts)==14 and set(clip_counts.values())=={30} and set(clip_selected.values())<={7,8},'每clip必须30整block且选7或8，原14clip结构不符')
    if clip50:require(len(clip50)==14 and set(clip50.values())=={15} and len(selected50)==1050,'50%嵌套设计每clip须15整block/1050图')
    return d

def subset_binding(full_panel,reference):
    d=load_subset(reference,full_panel)
    return {'subset_manifest_reference':subset_reference(reference),'ordered_subset_train_pool_sha256':d['ordered_train_pool_sha256'],'subset_N_R':525,'full_N_R':2100}

def filter_train_panel(full_panel,reference):
    """真实筛数据，不改原source binding和val，不让未选1575图进入训练catalogue。"""
    d=load_subset(reference,full_panel);uids=list(d['ordered_train_uids']);result={**full_panel}
    result['train_uids']=uids;result['train_catalogue']={u:deepcopy(full_panel['train_catalogue'][u]) for u in uids}
    result['statistics']=deepcopy(full_panel.get('statistics',{}));result['statistics']['original_full_train']=result['statistics'].get('train')
    result['statistics']['train']={'images':525,'full_pool_images':2100,'scope':'selected_UIDs_only; detailed_GT_counts_require_explicit_selected_audit'}
    result['subset_manifest_reference']=subset_reference(reference);result['ordered_subset_train_pool_sha256']=d['ordered_train_pool_sha256']
    require(result['val_uids'] is full_panel['val_uids'] and result['val_catalogue'] is full_panel['val_catalogue'] and result['val_ground_truth'] is full_panel['val_ground_truth'] and result['binding'] is full_panel['binding'],'val与完整源绑定必须原样保留')
    return result

def make_samplers(panel,synthetic,seed=17):
    require(len(panel['train_uids'])==525 and set(panel['train_catalogue'])==set(panel['train_uids']) and panel.get('ordered_subset_train_pool_sha256')==_pool_sha(panel['train_uids']),'须先实际filter到525，再构建采样器')
    require('subset_manifest_reference' in panel,'缺子集血缘')
    return base.make_samplers(panel,synthetic,seed)

def protocol(strategy,seed=17,*,subset_manifest_reference,**kwargs):
    ref=subset_reference(subset_manifest_reference);cfg=base.protocol(strategy,seed,**kwargs)
    cfg.update(schema_version=SCHEMA,job_id=('lowlabel25_R' if strategy=='R' else 'lowlabel25_R_plus_S')+'_seed17',N_R=525,full_N_R=2100,real_fraction=.25,subset_manifest_reference=ref,unselected_real_images_allowed=False,label_savings_claim_automatic=False)
    return cfg

def check_protocol(config):
    keys=('input_cache','baseline_R_status_reference','additional_space_bound_bytes','raw_upper_per_image_bytes','checkpoint_bytes','max_run_seconds')
    expected=protocol(config['strategy'],config['seed'],subset_manifest_reference=config['subset_manifest_reference'],plan_sha256=config['EXPERIMENTS_sha256'],admission_sha256=config['source_admission_sha256'],**{k:config[k] for k in keys})
    require({k:config[k] for k in expected}==expected and set(config)==set(expected)|{'control_group'},'低标注冻结协议变化')

def execution_code(code_source=SOURCE_GROUP/'code/futsalmot'):
    source=Path(code_source);known=json.loads((SOURCE_GROUP/'code_bindings.json').read_bytes())
    require(len(known)==41 and known.get('budget1500.py')==BASE_MODULE_SHA and _sha(Path(base.__file__))==BASE_MODULE_SHA,'必须复用原R1500已冻41数值集合')
    require({p.name:_sha(p) for p in source.glob('*.py')}==known,'原41来源缺失/新增/变化，不可root glob Seg/Pose')
    return {**known,'low_label_bbox.py':_sha(Path(__file__))}

def _verify_low_label_bindings(group,table):
    from . import formal,runner
    full,synthetic=formal._verify_bindings(group,table);cfg=table['jobs'][0];check_protocol(cfg)
    expected=subset_binding(full,cfg['subset_manifest_reference'])
    require(all(runner.canonical(table['source_bindings'].get(k))==runner.canonical(v) for k,v in expected.items()),'冻结来源的subset引用/有序池不同')
    return filter_train_panel(full,cfg['subset_manifest_reference']),synthetic

def _journal(directory,config,state,update,result=None):
    from . import runner
    value=f"\n\n## LOW-LABEL25：{config['job_id']} {state}\n\nUTC {runner.now()}。公开YOLO11s独立起点seed17，唯一真实525/原2100，subset={json.dumps(config['subset_manifest_reference'],ensure_ascii=False)}。策略{config['strategy']}，{update}/1500成功更新，实际曝光{exposure_plan(config['strategy'],update)}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{json.dumps(runner.plain(result),ensure_ascii=False)}。证据：{Path(directory).as_posix()}。\n"
    with (runner.RESEARCH/'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:f.write(value.encode());f.flush();os.fsync(f.fileno())

# snapshot/restore使用同一原代码对象，仅独立schema/global绑定；原模块一字不改。
_NAMESPACE={**base.__dict__,**globals()}
for _name in ('validate_snapshot','snapshot','restore'):
    _original=getattr(base,_name);_function=FunctionType(_original.__code__,_NAMESPACE,_name,_original.__defaults__,_original.__closure__);_function.__kwdefaults__=_original.__kwdefaults__;globals()[_name]=_function;_NAMESPACE[_name]=_function

# 训练循环仅替换完整来源验证后的筛选调用；其余AST必须精确可逆。
RUN_LOOP_SOURCE=inspect.getsource(base.run_job)
_old='panel,synthetic=formal._verify_bindings(group,table)'
_new='panel,synthetic=_verify_low_label_bindings(group,table)'
require(RUN_LOOP_SOURCE.count(_old)==1,'原运行循环筛选接点变化')
RUN_LOOP_SOURCE=RUN_LOOP_SOURCE.replace(_old,_new)
_NAMESPACE.update(globals());exec(compile(RUN_LOOP_SOURCE,str(Path(__file__).absolute())+':derived_run_job','exec'),_NAMESPACE);run_job=_NAMESPACE['run_job']
require(ast.dump(ast.parse(RUN_LOOP_SOURCE.replace(_new,_old)),include_attributes=False)==ast.dump(ast.parse(inspect.getsource(base.run_job)),include_attributes=False),'低标注运行循环除一处数据入口外发生变化')


def prepare_group(group,*,config,admission=None,code_source=SOURCE_GROUP/'code/futsalmot'):
    from . import runner,formal
    from .runner_data import real_panel
    from .catalogues import synthetic_bbox_catalogue
    from .input_cache import validate_ready_cache
    from .output_storage import create_output_binding,capture_control
    group=Path(group).absolute();require(group.parent==runner.RESEARCH/'runs' and group.is_dir() and not (group/'job_table.json').exists(),'组位置非法或已准备')
    config=deepcopy(config);config['control_group']=str(group);check_protocol(config);check_baseline(config['baseline_R_status_reference'])
    actual=json.loads((group/'admission.json').read_bytes());require(admission is None or actual==admission,'准入收据不一致')
    require(actual.get('schema_version')=='bbox_low_label25_admission_v1' and actual.get('status')=='PASS','缺实际mixed PASS准入')
    require((actual.get('numeric_mode'),actual.get('microbatch_size'),actual.get('num_workers'),actual.get('effective_batch'))==('fp32',8,2,16),'准入数值不同')
    require(actual.get('strategy')==config['strategy'] and actual.get('updates')==UPDATES and actual.get('sampler_contract')==sampler_contract(config['strategy']) and actual.get('input_cache')==config['input_cache'] and actual.get('baseline_R_status_reference')==config['baseline_R_status_reference'],'准入mixed/缓存/R引用不同')
    require(actual.get('checkpoint_every')==1000 and actual.get('diagnostic_updates')==3,'准入检查点/真实诊断次数不符')
    require(actual.get('measurements',{}).get('temporary_models_removed') is True and actual.get('schedule_policy')==SCHEDULE_POLICY,'诊断模型未清或调度变化')
    exact=actual.get('measurements',{}).get('exact_restore',{})
    require(exact and all(value is True for value in exact.values()) and exact.get('loss_and_metrics') is True and exact.get('actual_gradient_tensors') is True,'未实际通过完整恢复/下一步loss及梯度exact')
    require(_sha(group/'admission.json')==config['source_admission_sha256'],'准入SHA不同')
    for name,key in [('EXPERIMENTS.md','EXPERIMENTS_sha256'),('PAPER.md','PAPER_sha256')]:require(_sha(runner.RESEARCH/'docs/plan'/name)==config[key],'封存方案变化')
    panel=real_panel(runner.REAL);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');require((len(panel['train_uids']),len(panel['val_uids']),len(synthetic['ordered_uids']))==(2100,750,24149),'数据数量变化')
    selected=filter_train_panel(panel,config['subset_manifest_reference']);subset=subset_binding(panel,config['subset_manifest_reference'])
    require(actual.get('N_R')==525 and actual.get('full_N_R')==2100 and actual.get('subset_manifest_reference')==config['subset_manifest_reference'],'实际准入没有独立525子集')
    require(all(runner.canonical(actual['source_bindings'].get(k))==runner.canonical(v) for k,v in subset.items()),'实际准入subset绑定不同')
    for key in ('EXPERIMENTS_sha256','PAPER_sha256','input_cache','baseline_R_status_reference','subset_manifest_reference'):
        require(runner.canonical(actual['source_bindings'].get(key))==runner.canonical(config[key]),'实际准入计划/缓存/基线/子集不同:'+key)
    for key,value in [('real_panel',panel['binding']),('synthetic_panel',synthetic['binding'])]:require(runner.canonical(actual['source_bindings'][key])==runner.canonical(value),'实际准入来源不同:'+key)
    validate_ready_cache(config['input_cache'],synthetic['binding']);hashes=execution_code(code_source);require(actual.get('executed_code_sha256')==hashes,'实际执行代码与冻结代码不同')
    require(actual['source_bindings'].get('requirements_sha256')==_sha(runner.RESEARCH/'requirements.lock'),'预检依赖锁变化')
    public={'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':_sha(runner.WEIGHTS/'yolo11s.pt')};require(actual['source_bindings'].get('public_model')==public,'公开起点绑定不同')
    require(time.time()+config['max_run_seconds']+600<datetime.fromisoformat(config['deadline_utc']).timestamp(),'全局截止无法闭合');runner.production_closed();runner.check_disk(32*1024**2);runner.check_archive_space(config['additional_space_bound_bytes'])
    target=group/'code/futsalmot';target.mkdir(parents=True,exist_ok=False)
    for name in hashes:shutil.copyfile(Path(__file__) if name=='low_label_bbox.py' else Path(code_source)/name,target/name)
    require({p.name:_sha(p) for p in target.glob('*.py')}==hashes,'冻结期间源码变化');shutil.copyfile(runner.RESEARCH/'requirements.lock',group/'code/requirements.lock')
    bindings={'real_panel':panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':public,'source_code':hashes,'requirements_sha256':_sha(group/'code/requirements.lock'),'EXPERIMENTS_sha256':config['EXPERIMENTS_sha256'],'PAPER_sha256':config['PAPER_sha256'],'input_cache':config['input_cache'],'baseline_R_status_reference':config['baseline_R_status_reference'],**subset}
    runner.write_once(group/'code_bindings.json',hashes);runner.write_once(group/'job_table.json',{'schema_version':SCHEMA,'jobs':[config],'source_bindings':bindings,'test_enabled':False,'extra_jobs_authorized':False})
    launch="import os,sys\nfrom pathlib import Path\nos.environ['YOLO_OFFLINE']='true'\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\ngroup=Path(__file__).absolute().parent\nsys.path.insert(0,str(group/'code'))\nif __name__=='__main__':\n from futsalmot.low_label_bbox import controller\n controller(group)\n"
    with (group/'launch.py').open('x',encoding='utf8',newline='\n') as f:f.write(launch)
    location=create_output_binding(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs');snapshot=capture_control(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs')
    return {'group':str(group),'output_group':location['output_group'],'source_bindings':bindings,'job_table_sha256':_sha(group/'job_table.json'),'snapshot':snapshot}

def controller(group):
    from . import runner
    from .output_storage import capture_control
    group=Path(group).absolute();table=json.loads((group/'job_table.json').read_bytes());require(table['schema_version']==SCHEMA and len(table['jobs'])==1 and not table['test_enabled'],'不是单个固定混合作业')
    require(not (group/'status.json').exists() and not (runner.output_group(group)/'control/CONTROL_MANIFEST.json').exists(),'已派发/封闭组不可重放')
    admission=json.loads((group/'admission.json').read_bytes());require(admission['status']=='PASS','准入未通过');owned=False;results=[]
    try:
        with runner.gpu_lock(group):
            owned=True;runner.setup_cuda(admission);cfg=table['jobs'][0];runner.group_status(group,{'state':'RUNNING','pid':os.getpid(),'create_time':__import__('psutil').Process().create_time(),'active_job':cfg['job_id'],'results':[]})
            result=run_job(group,cfg);results.append(result);runner.group_status(group,{'state':'LOW_LABEL25_COMPLETE' if result['state']=='COMPLETE' else result['state'],'results':results,'test_enabled':False})
    except BaseException as exc:
        if owned:runner.group_status(group,{'state':'FAIL_STOP','error':str(exc),'traceback':traceback.format_exc(),'results':results})
        raise
    finally:
        if owned:
            sealed=capture_control(group,research_root=runner.RESEARCH,output_base=runner.ARCHIVE_ROOT/'runs',final=True);print(json.dumps({'event':'control_sealed',**sealed}),flush=True)

_NAMESPACE.update(globals())
