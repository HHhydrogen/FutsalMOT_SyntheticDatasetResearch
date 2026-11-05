"""独立比赛确认的有限外围：仅显式调用才读授权源/启动子进程；不接入训练源码。"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import FunctionType
import ctypes,hashlib,json,math,os,re,shutil,stat,subprocess,sys,time,traceback

RESEARCH=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch')
OUTPUT=Path('G:/FutsalMOT_Training/runs')
GLOBAL_DEADLINE='2026-11-07T23:55:00Z'
RAW_PER_IMAGE=6650344
JSON_PER_IMAGE=1024**2
RESERVE=100*1024**3
CONTROL_BOUND=32*1024**2
PER_JOB=750*(RAW_PER_IMAGE+JSON_PER_IMAGE)
PREDICTIONS_48=48*PER_JOB
PARTIAL=PER_JOB
TRANSIENT=3*1024**3
SCHEMA='bbox_confirmation_suite_v1'
JOB_SCHEMA='bbox_confirmation_job_v1'


def require(value,message):
    if not value:raise ValueError(message)

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)

def no_links(path,*,exists=True):
    path=Path(path).absolute();require('..' not in path.parts and not str(path).startswith(('\\\\','//')),'只准本机绝对普通路径')
    for part in reversed((path,*path.parents)):
        try:s=part.lstat()
        except FileNotFoundError:
            if exists:raise
            continue
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,'禁止reparse/junction')
    return path

def write_new(path,value):
    path=no_links(path,exists=False);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())

def read_reference(reference,*,maximum=32*1024**2):
    require(type(reference) is dict and type(reference.get('path')) is str and type(reference.get('sha256')) is str,'须精确path/SHA')
    path=no_links(reference['path']);s=path.stat();require(0<s.st_size<=maximum,'小文件为空或超界');raw=path.read_bytes();after=path.stat()
    require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and hashlib.sha256(raw).hexdigest()==reference['sha256'],'小文件身份/摘要变化')
    return json.loads(raw)

def reference(path):return {'path':str(Path(path).absolute()),'sha256':sha(path)}

def deadline_unix():return datetime.fromisoformat(GLOBAL_DEADLINE.replace('Z','+00:00')).timestamp()

def resource_bound(remaining=48):
    require(type(remaining) is int and 0<=remaining<=48,'只准0..48剩余评价')
    return {'remaining_model_matches':remaining,'prediction_bytes':remaining*PER_JOB,'one_partial_bytes':PARTIAL,'transient_bytes':TRANSIENT,'additional_G_bytes':remaining*PER_JOB+PARTIAL+TRANSIENT,'G_reserve_bytes':RESERVE,'D_control_bound_bytes':CONTROL_BOUND,'D_reserve_bytes':RESERVE,'full_predictions48_bytes':PREDICTIONS_48,'formula':'remaining*750*(6650344+1048576) +750*(6650344+1048576) +3GiB'}

def check_resources(*,research=RESEARCH,output=OUTPUT,remaining=48,now=None):
    now=time.time() if now is None else now;bound=resource_bound(remaining);freeD=shutil.disk_usage(research).free;freeG=shutil.disk_usage(output).free
    require(freeD>=RESERVE+CONTROL_BOUND and freeG>=RESERVE+bound['additional_G_bytes'],'确认实际卷空间不足完整剩余输出+保留')
    require(now+remaining*900+600<deadline_unix(),'剩余确认按900秒/项不能在原Nov7全局闭合')
    return {**bound,'actual_D_free_bytes':freeD,'actual_G_free_bytes':freeG,'observed_at_utc':utc(),'time_bound_seconds':remaining*900+600,'remaining_full_lifecycle_seconds_per_job':900}


def validate_runtime(runtime,*,now=None):
    now=time.time() if now is None else now
    require(type(runtime) is dict and runtime.get('schema_version')=='bbox_confirmation_runtime_v1','缺唯一原生进程runtime')
    start=runtime.get('dispatch_started_unix');deadline=runtime.get('deadline_unix')
    require(type(start) in (int,float) and type(deadline) in (int,float) and math.isfinite(start) and math.isfinite(deadline),'非法绝对时间')
    require(runtime.get('max_seconds')==900 and abs(deadline-min(start+900,deadline_unix()))<1e-6,'900秒deadline被延长或重置')
    require(now<deadline,'确认绝对900秒/全局截止已到')
    return deadline

def stop_paths(control,job,output):
    return (Path(control)/'STOP.json',Path(job)/'STOP.json',Path(output)/'STOP.json',Path(output)/'jobs'/Path(job).name/'STOP.json')

def check_stop_deadline(runtime,paths,*,now=None):
    if any(Path(p).exists() for p in paths):raise RuntimeError('USER_STOP: confirmation request retained')
    return validate_runtime(runtime,now=now)


def safe_job_id(model_id,match_uid):
    require(type(model_id) is str and re.fullmatch(r'[A-Za-z0-9_.+:-]{1,100}',model_id) is not None,'模型ID不是固定安全token')
    require(match_uid in ('match:soccernet-tracking:game:8','match:soccernet-tracking:game:9'),'只准game8/game9')
    return model_id.replace(':','_')+'__game'+match_uid.rsplit(':',1)[1]


def fixed_jobs(spec):
    """只读已绑定spec元数据；不得借此调用真实GT/模型授权。"""
    models=spec.get('models',[]);matches=spec.get('matches',[])
    require(len(models)==24 and len({x['model_id'] for x in models})==24,'spec不是唯一24模型')
    require([x['match_uid'] for x in matches]==['match:soccernet-tracking:game:8','match:soccernet-tracking:game:9'],'spec比赛/顺序不是固定game8/game9')
    jobs=[{'job_id':safe_job_id(model['model_id'],match['match_uid']),'model_id':model['model_id'],'match_uid':match['match_uid']} for model in models for match in matches]
    require(len({row['job_id'] for row in jobs})==48,'固定48job身份冲突')
    return jobs


def suite_paths(control_group,*,research=RESEARCH,output_base=OUTPUT):
    control=Path(control_group).absolute();require(control.parent==Path(research)/'runs' and control.name.startswith('bbox_confirmation_'),'只准独立确认D小控制组')
    no_links(control,exists=False);no_links(research);no_links(output_base)
    return control,Path(output_base)/control.name


def freeze_code(control,*,confirmation_module=None):
    from . import bbox_confirmation as confirmation
    confirmation=confirmation_module or confirmation
    numeric=confirmation.execution_code();source=confirmation.SOURCE_GROUP/'code/futsalmot';target=Path(control)/'code/futsalmot';target.mkdir(parents=True)
    for name in numeric:shutil.copyfile(Path(confirmation.__file__) if name=='bbox_confirmation.py' else source/name,target/name)
    shutil.copyfile(Path(__file__),target/'bbox_confirmation_lifecycle.py')
    hashes={p.name:sha(p) for p in target.glob('*.py')};require(len(hashes)==43 and all(hashes[k]==v for k,v in numeric.items()),'确认43代码冻结不符')
    return hashes


def prepare_suite(control_group,*,spec_reference,completion_reference,plan_reference,observer_source,research=RESEARCH,output_base=OUTPUT):
    """只能在完整24模型授权后显式调用；导入本模块不读取test/模型。"""
    from . import bbox_confirmation as confirmation
    control,out=suite_paths(control_group,research=research,output_base=output_base)
    require(not control.exists() and not out.exists(),'确认组已存在，禁止覆盖/猜重放')
    permit=confirmation.authorize_confirmation(spec_reference,completion_reference=completion_reference,plan_reference=plan_reference)
    resource=check_resources(research=research,output=output_base,remaining=48)
    control.mkdir();out.mkdir();hashes=freeze_code(control);shutil.copyfile(observer_source,control/'observe.py')
    spec=permit.spec();jobs=fixed_jobs(spec)
    require(len({r['job_id'] for r in jobs})==48,'确认job身份冲突')
    request={'schema_version':SCHEMA,'state':'FROZEN','control_group':str(control),'output_group':str(out),'spec_reference':deepcopy(spec_reference),'completion_reference':deepcopy(completion_reference),'plan_reference':deepcopy(plan_reference),'jobs':jobs,'source_code_sha256':hashes,'observer_sha256':sha(control/'observe.py'),'resources':resource,'max_seconds_per_job':900,'global_deadline_utc':GLOBAL_DEADLINE,'inference_only':True,'optimizer_updates':0,'automatic_resume':False,'automatic_retry':False,'allowed_evaluations':48,'created_at_utc':utc(),'research_root':str(Path(research)),'output_base':str(Path(output_base))}
    write_new(control/'suite_request.json',request);write_new(control/'code_bindings.json',hashes)
    write_new(out/'RUN_LOCATION.json',{'control_group':str(control),'output_group':str(out),'schema':'bbox_confirmation_direct_output_v1'})
    return {'control_group':str(control),'output_group':str(out),'suite_reference':reference(control/'suite_request.json'),'resources':resource}


def load_suite(control_group):
    control=Path(control_group).absolute();no_links(control);p=control/'suite_request.json';suite=json.loads(p.read_bytes())
    require(suite.get('schema_version')==SCHEMA and suite.get('state')=='FROZEN' and suite.get('control_group')==str(control),'不是冻结确认suite')
    require(suite.get('max_seconds_per_job')==900 and suite.get('global_deadline_utc')==GLOBAL_DEADLINE and suite.get('allowed_evaluations')==48 and suite.get('automatic_retry') is False and suite.get('automatic_resume') is False,'确认scope/时间/重放限制变化')
    require(suite.get('jobs')==fixed_jobs(read_reference(suite['spec_reference'])),'确认job列表不是spec固定24模型×game8/game9顺序')
    require(json.loads((control/'code_bindings.json').read_bytes())==suite['source_code_sha256'],'确认code清单不同')
    actual={p.name:sha(p) for p in (control/'code/futsalmot').glob('*.py')};require(actual==suite['source_code_sha256'] and len(actual)==43,'已冻结确认43源码变化')
    require(sha(control/'observe.py')==suite['observer_sha256'],'观察器源码变化')
    return suite


def validate_job_request(control,suite,job_id,req):
    control=Path(control);rows=[(i,row) for i,row in enumerate(suite['jobs']) if row['job_id']==job_id]
    require(len(rows)==1,'确认job不在固定48列表');index,row=rows[0];job=control/'jobs'/job_id;out=Path(suite['output_group'])/'jobs'/job_id
    expected={'schema_version':JOB_SCHEMA,**row,'control_group':str(control),'job_control':str(job),'output_directory':str(out),'tag':'confirmation_game'+row['match_uid'].rsplit(':',1)[1]+'_terminal_EMA','max_seconds':900,'automatic_retry':False,'automatic_resume':False}
    require(all(req.get(k)==v for k,v in expected.items()),'确认job模型/比赛/tag/路径/scope不是固定suite行')
    require(read_reference(req['suite_reference'])==suite,'冻结suite引用不同')
    bound=resource_bound(48-index);resources=req.get('resources',{})
    require(all(resources.get(k)==v for k,v in bound.items()),'确认job剩余资源公式/项数不是固定suite行')
    if 'time_bound_seconds' in resources:require(resources['time_bound_seconds']==(48-index)*900+600,'确认job剩余时间公式不同')
    return row


def remaining_jobs(control,suite):
    complete=[]
    for row in suite['jobs']:
        job=Path(control)/'jobs'/row['job_id'];result=job/'result.json';life=job/'lifetime_receipt.json';seal=Path(suite['output_group'])/'jobs'/row['job_id']/'control/CONTROL_MANIFEST.json'
        if not result.exists():break
        r=json.loads(result.read_bytes());require(r.get('state')=='COMPLETE','前项失败，不自动跳过/重放')
        require((r.get('job_id'),r.get('model_id'),r.get('match_uid'))==(row['job_id'],row['model_id'],row['match_uid']),'前项模型/比赛身份不同')
        require(r.get('summary',{}).get('images')==750 and r['summary'].get('update')==1500 and r['summary'].get('evaluated_model')=='stage_EMA_at_recorded_update','前项不是750图末EMA')
        require(read_reference(r['summary_reference'])==r['summary'],'前项原生summary变化')
        require(life.exists() and seal.exists(),'前项尚未真实退出/封闭')
        l=json.loads(life.read_bytes());m=json.loads(seal.read_bytes());require(l.get('state')=='VERIFIED_EXIT' and l.get('returncode')==0 and l.get('owned_still_alive')==[] and l.get('GPU_lock_retired') is True and m.get('state')=='SEALED','前项owned/GPU/小控制未闭合')
        for f in m.get('files',[]):
            p=seal.parent/f['path'];require(p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'前项小控制快照变化')
        complete.append(row['job_id'])
    return suite['jobs'][len(complete):],complete


def prepare_job(control_group,job_id):
    from . import bbox_confirmation as confirmation
    control=Path(control_group).absolute();suite=load_suite(control);remaining,done=remaining_jobs(control,suite)
    require(remaining and remaining[0]['job_id']==job_id,'只能顺序启动下一唯一model-match')
    require(not (control/'STOP.json').exists() and not (Path(suite['output_group'])/'STOP.json').exists(),'USER_STOP: suite cancelled before next job')
    require(not (Path(suite['research_root'])/'.research_gpu.lock').exists(),'研究GPU未释放')
    permit=confirmation.authorize_confirmation(suite['spec_reference'],completion_reference=suite['completion_reference'],plan_reference=suite['plan_reference'])
    resource=check_resources(research=Path(suite['research_root']),output=Path(suite['output_base']),remaining=len(remaining))
    job=control/'jobs'/job_id;out=Path(suite['output_group'])/'jobs'/job_id
    require(not job.exists() and not out.exists(),'model-match已存在，禁止任何隐式retry')
    row=remaining[0];permit.model(row['model_id']);permit.match(row['match_uid']);job.mkdir(parents=True);out.mkdir(parents=True)
    req={'schema_version':JOB_SCHEMA,**row,'suite_reference':reference(control/'suite_request.json'),'control_group':str(control),'job_control':str(job),'output_directory':str(out),'tag':'confirmation_game'+row['match_uid'].rsplit(':',1)[1]+'_terminal_EMA','resources':resource,'max_seconds':900,'automatic_retry':False,'automatic_resume':False,'created_at_utc':utc()}
    write_new(job/'job_request.json',req)
    return {'job':str(job),'output_directory':str(out),'job_reference':reference(job/'job_request.json')}


def verified_ema_backend(authorization,model_id,weights_directory):
    """只构造原s架构并使用可信末EMA，不构建或加载优化器，不接raw训练权重。"""
    from . import bbox_confirmation as confirmation
    from .pretrained import load_local_yolo11_detect
    from .runtime_backend import NativeBBoxBackend
    import torch
    payload=confirmation.load_terminal_ema(authorization,model_id)
    network=load_local_yolo11_detect(weights_directory,model_size='s')
    original=payload['expected_source_bindings']['public_model'];require(network.initialization_binding['sha256']==original['sha256'] and Path(network.initialization_binding['path'])==Path(original['path']),'确认公开s架构来源与训练起点不同')
    class EMAOnly:
        # 原函数代码对象；只复制原模型结构并load ema_state，不使用raw或SGD。
        ema_network_for_eval=NativeBBoxBackend.ema_network_for_eval
    backend=EMAOnly();backend.network=network;backend.model=network.model;backend.ema_state={'updates':1500,'state_dict':payload['state_dict']};backend.stage_update=1500;backend.failed=False;backend.device=torch.device('cuda:0');backend.confirmation_model_id=model_id;backend.confirmation_checkpoint_receipt=payload['checkpoint_receipt']
    return backend


def scoped_evaluator(candidate,paths):
    """仅新增一个iterator生命周期接点；网络/GT/解码/NMS/AP/native数值不改。"""
    from . import bbox_confirmation as confirmation,runner
    original=candidate['candidate_function'];scope=dict(original.__globals__);prior=scope['stop_requested'];tracked=[];attempted=False
    def tracked_iterator(loader):
        iterator=iter(loader);tracked.append(iterator);return iterator
    scope['stop_requested']=lambda group,directory:any(Path(p).exists() for p in paths) or prior(group,directory)
    scope['_confirmation_tracked_iterator']=tracked_iterator
    existing=list(candidate['mechanical_binding']['substitutions'])
    extra=('iterator=iter(loader)','iterator=_confirmation_tracked_iterator(loader)')
    # candidate的source由原runner.evaluate加已审Dataset接点生成；从原函数再精确复现并追加唯一iterator接点。
    function,binding=confirmation._derive(runner.evaluate,existing+[extra],scope)
    def evaluate(*args,**kwargs):
        nonlocal attempted
        require(not attempted,'确认函数已调用，禁止隐式retry');attempted=True
        try:return function(*args,**kwargs)
        finally:
            # 原evaluate的try之前EMA构造失败也覆盖；正常路径原shutdown幂等，再核一次。
            errors=[]
            for iterator in tracked:
                try:runner._shutdown(iterator)
                except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
            if errors:raise RuntimeError('确认最外iterator shutdown失败: '+repr(errors))
    evaluate.confirmation_lifecycle_binding={'schema_version':'bbox_confirmation_iterator_finally_v1','candidate_binding':deepcopy(candidate['mechanical_binding']),'additional_substitution':extra,'combined_inverse_AST_binding':binding,'outside_finally_uses_original_runner_shutdown':True,'normal_and_initialization_failure_covered':True,'no_numeric_or_postprocess_change':True}
    return evaluate


def wait_owned(rows,*,seconds=3):
    import psutil
    until=time.monotonic()+seconds
    while True:
        live=[]
        for row in rows:
            try:
                p=psutil.Process(row['pid'])
                if abs(p.create_time()-row['create_time'])<.01:live.append(row)
            except psutil.NoSuchProcess:pass
            except psutil.AccessDenied:live.append({**row,'identity_unverifiable':True})
        if not live or time.monotonic()>=until:return live
        time.sleep(.05)


class ConfirmGPULock:
    """沿原锁身份协议，若worker残留则保留自有锁，不因子进程退出提前退休。"""
    def __init__(self,runner,group):self.runner=runner;self.group=Path(group);self.value=None;self.identity=None;self.cleanup=None
    def __enter__(self):
        import psutil
        self.runner.production_closed();self.path=self.runner.RESEARCH/'.research_gpu.lock';self.value={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'group':str(self.group),'created_at_utc':utc()};self.runner.write_once(self.path,self.value);s=self.path.stat();self.identity=(s.st_dev,s.st_ino);return self
    def __exit__(self,*exc):
        import psutil
        rows=[]
        for p in psutil.Process().children(recursive=True):
            try:rows.append({'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()})
            except psutil.NoSuchProcess:pass
        live=wait_owned(rows,seconds=3);retired=False
        if not live and self.path.exists():
            s=self.path.stat()
            if (s.st_dev,s.st_ino)==self.identity and json.loads(self.path.read_bytes())==self.value:self.path.unlink();retired=True
        self.cleanup={'owned_children':rows,'owned_children_still_alive':live,'GPU_lock_retired':retired,'lock_retained_if_workers_or_identity_unknown':not retired}
        return False


def verify_dispatch_identity(dispatch,control,job_id):
    import psutil
    current=psutil.Process();chain=[current]+current.parents()
    require(dispatch.get('schema_version')=='bbox_confirmation_observed_dispatch_v1','不是原生观察器dispatch')
    require(any(p.pid==dispatch['pid'] and abs(p.create_time()-dispatch['create_time'])<.01 and p.cmdline()==dispatch['command'] for p in chain),'确认持句柄wrapper PID/CT/完整命令与实际祖先链不同')
    require(current.cmdline()[1:]==dispatch['command'][1:],'确认实际controller脚本参数与唯一wrapper不同')
    observer=dispatch.get('observer',{})
    require(any(p.pid==observer.get('pid') and abs(p.create_time()-observer.get('create_time',-1))<.01 and p.cmdline()==observer.get('command') for p in chain[1:]),'确认父观察器PID/CT/完整命令不同')
    require(sys.argv[-2:]==['--child',job_id] and Path(sys.argv[0]).absolute()==Path(control)/'observe.py','确认只准冻结--child入口')


def wait_dispatch(job,runtime,paths):
    """父Popen后写小收据的至多5秒就绪窗口，仍受原绝对900秒和STOP约束。"""
    until=time.monotonic()+5
    while True:
        check_stop_deadline(runtime,paths)
        try:return json.loads(no_links(Path(job)/'dispatch_wrapper.json').read_bytes())
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=until:raise RuntimeError('父观察器dispatch未在5秒就绪；保留失败不重放')
            time.sleep(.02)


def run_job(control_group,job_id):
    """仅冻结observe.py子进程显式调用；实际48评价仍待root准入。"""
    control=Path(control_group).absolute();suite=load_suite(control);job=control/'jobs'/job_id;req=json.loads((job/'job_request.json').read_bytes());runtime=json.loads((job/'runtime_limit.json').read_bytes());out=Path(suite['output_group'])/'jobs'/job_id;paths=stop_paths(control,job,suite['output_group'])
    hard=runtime.get('deadline_unix',0);result={'state':'FAIL_STOP','job_id':job_id,'model_id':req.get('model_id'),'match_uid':req.get('match_uid'),'started_at_utc':utc(),'pid':os.getpid(),'create_time':__import__('psutil').Process().create_time(),'command':list(sys.argv),'inference_only':True,'optimizer_updates':0};lock=None
    try:
        require(req.get('schema_version')==JOB_SCHEMA and req.get('job_id')==job_id and req.get('automatic_retry') is False and req.get('automatic_resume') is False,'不是唯一确认job')
        validate_job_request(control,suite,job_id,req);hard=check_stop_deadline(runtime,paths)
        require(not (job/'result.json').exists() and not (out/'evaluation').exists(),'已执行/partial保留，禁止runner隐式retry')
        dispatch=wait_dispatch(job,runtime,paths);require(read_reference(dispatch['request_reference'])==req and read_reference(dispatch['runtime_reference'])==runtime,'观察器请求/runtime绑定变化');require(dispatch.get('observer_sha256')==suite['observer_sha256'] and dispatch.get('started_unix')==runtime['dispatch_started_unix'] and dispatch.get('deadline_unix')==runtime['deadline_unix'],'dispatch观察器/绝对时间绑定不同');verify_dispatch_identity(dispatch,control,job_id)
        from . import bbox_confirmation as confirmation,runner
        import torch
        require(control.parent==runner.RESEARCH/'runs' and Path(suite['research_root'])==runner.RESEARCH and Path(suite['output_base'])==runner.ARCHIVE_ROOT/'runs','实际确认执行根不是既定D/G研究根')
        require(Path(suite['output_group'])==runner.ARCHIVE_ROOT/'runs'/control.name and Path(req['job_control'])==job and out==Path(suite['output_group'])/'jobs'/job_id,'实际确认D/G及job路径绑定不同')
        permit=confirmation.authorize_confirmation(suite['spec_reference'],completion_reference=suite['completion_reference'],plan_reference=suite['plan_reference']);check_stop_deadline(runtime,paths)
        check_resources(research=Path(suite['research_root']),output=Path(suite['output_base']),remaining=req['resources']['remaining_model_matches']);runner.production_closed();check_stop_deadline(runtime,paths)
        with ConfirmGPULock(runner,job) as lock:
            # 全24完工/GT授权已通过，且已持GPU锁后才初始化CUDA/读取模型。
            check_stop_deadline(runtime,paths);runner.setup_cuda({'cuda_memory_fraction':.8});check_stop_deadline(runtime,paths);require(torch.cuda.mem_get_info()[0]>=1024**3,'确认初始化GPU余量低于1GiB')
            backend=verified_ema_backend(permit,req['model_id'],runner.WEIGHTS);check_stop_deadline(runtime,paths)
            panel=confirmation.confirmation_panel(permit,req['match_uid']);check_stop_deadline(runtime,paths)
            require(not any(x['ignore'] for x in panel['val_ground_truth'].values()),'真实test true-ignore不允许当前严格标准COCO，不得删除GT绕过')
            candidate=confirmation.make_confirmation_evaluator(permit,req['match_uid']);evaluate=scoped_evaluator(candidate,paths)
            class Recorder:
                def event(self,value):runner.append(out/'events.jsonl',value)
            cfg={'microbatch_size':8,'num_workers':2,'numeric_mode':'fp32','storage_mode':'direct_g','raw_upper_per_image_bytes':RAW_PER_IMAGE,'control_group':str(job)}
            summary=evaluate(out,backend,panel,cfg,None,Recorder(),req['tag'],hard_stop_unix=hard);check_stop_deadline(runtime,paths)
            require(summary['images']==750 and summary['update']==1500 and summary['evaluated_model']=='stage_EMA_at_recorded_update','确认结果不是750末EMA')
            free,total=torch.cuda.mem_get_info();require(free>=1024**3,'确认结束GPU余量低于1GiB')
            result['GPU_measurement']={'free_bytes':free,'total_bytes':total,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            require(json.loads((out/'evaluation'/req['tag']/'summary.json').read_bytes())==summary,'原生summary未实际写入或不符')
            result.update(state='COMPLETE',summary=summary,summary_reference=reference(out/'evaluation'/req['tag']/'summary.json'),checkpoint_receipt=backend.confirmation_checkpoint_receipt,numeric_mechanical_binding=candidate['mechanical_binding'],iterator_lifecycle_binding=deepcopy(getattr(evaluate,'confirmation_lifecycle_binding',{})))
            del backend,panel,evaluate;torch.cuda.empty_cache()
    except BaseException as exc:
        result.update(state='USER_STOP' if 'USER_STOP' in str(exc) else 'LIMIT_STOP' if isinstance(exc,TimeoutError) or time.time()>=hard else 'FAIL_STOP',error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    finally:
        if lock is not None and lock.cleanup is not None:result['worker_cleanup']=lock.cleanup
        if lock is not None and lock.cleanup is not None and not lock.cleanup['GPU_lock_retired']:result['state']='ATTENTION'
        result.update(finished_at_utc=utc(),deadline_unix=hard,within_absolute_deadline=time.time()<=hard)
        if not result['within_absolute_deadline'] and result['state']=='COMPLETE':result['state']='LIMIT_STOP'
        # 已存在result属于未知提交/重放；绝不覆盖原始结果。
        if not (job/'result.json').exists():write_new(job/'result.json',result)
        if not (out/'result.json').exists():write_new(out/'result.json',result)
    return 0 if result['state']=='COMPLETE' else 1


def process_times(handle):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;values=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(wintypes.HANDLE(int(handle)),*(ctypes.byref(v) for v in values)):raise ctypes.WinError(ctypes.get_last_error())
    def stamp(v):return ((v.dwHighDateTime<<32)|v.dwLowDateTime)/1e7-11644473600
    return {'create_time':stamp(values[0]),'exit_time':stamp(values[1])}


def owned_tree(pid):
    import psutil
    rows=[]
    try:
        p=psutil.Process(pid)
        for child in [p]+p.children(recursive=True):
            try:rows.append({'pid':child.pid,'create_time':child.create_time(),'command':child.cmdline()})
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    except psutil.NoSuchProcess:pass
    return rows


def seal_job(job,out):
    """父观察器确认真实退出后，精确小控制快照单次封闭；不扫描预测/媒体。"""
    job=Path(job);out=Path(out);target=out/'control';manifest=target/'CONTROL_MANIFEST.json';require(not target.exists(),'小控制已存在，禁止覆盖/刷新')
    allowed={'job_request.json','runtime_limit.json','dispatch_wrapper.json','lifetime_receipt.json','result.json','STOP.json'};rows=[];total=0
    for path in job.iterdir():
        require(path.is_file() and path.name in allowed,'确认job含未知控制文件/目录')
        no_links(path);size=path.stat().st_size;require(size<=16*1024**2,'小控制单文件超界');total+=size;rows.append({'path':path.name,'bytes':size,'sha256':sha(path)})
    require(total<=CONTROL_BOUND and {'job_request.json','runtime_limit.json','dispatch_wrapper.json','lifetime_receipt.json','result.json'}<={r['path'] for r in rows},'小控制总量/必要文件不符')
    target.mkdir()
    for row in rows:
        shutil.copyfile(job/row['path'],target/row['path']);require(sha(target/row['path'])==row['sha256'],'小控制复制读回不符')
    value={'schema_version':'bbox_confirmation_job_control_v1','state':'SEALED','scope':'control_snapshot_only','files':rows,'file_count':len(rows),'total_bytes':total,'control_group':str(job),'output_directory':str(out),'created_at_utc':utc()};write_new(manifest,value);return reference(manifest)


def observe_job(control_group,job_id):
    """纯CPU父进程持原生句柄，不初始化CUDA；唯一child包含导入/初始化900秒计时。"""
    control=Path(control_group).absolute();suite=load_suite(control);job=control/'jobs'/job_id;req=json.loads((job/'job_request.json').read_bytes());validate_job_request(control,suite,job_id,req);out=Path(req['output_directory'])
    require(not any((job/n).exists() for n in ('dispatch_wrapper.json','runtime_limit.json','lifetime_receipt.json','result.json')) and not (out/'evaluation').exists(),'已派发/partial不可重放')
    require(not (Path(suite['research_root'])/'.research_gpu.lock').exists(),'GPU未释放')
    start=time.time();hard=min(start+900,deadline_unix());runtime={'schema_version':'bbox_confirmation_runtime_v1','dispatch_started_unix':start,'deadline_unix':hard,'max_seconds':900};check_stop_deadline(runtime,stop_paths(control,job,suite['output_group']));write_new(job/'runtime_limit.json',runtime)
    command=[str(Path(suite['research_root'])/'.venv/Scripts/python.exe'),'-B','-X','utf8','-u',str(control/'observe.py'),str(control),'--child',job_id]
    import psutil
    parent=psutil.Process();observer={'pid':parent.pid,'create_time':parent.create_time(),'command':parent.cmdline()}
    env=os.environ.copy();env.update(YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');owned={};stop_written=False
    with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,cwd=suite['research_root'],env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));times=process_times(child._handle)
        dispatch={'schema_version':'bbox_confirmation_observed_dispatch_v1','pid':child.pid,'create_time':times['create_time'],'command':command,'observer':observer,'request_reference':reference(job/'job_request.json'),'runtime_reference':reference(job/'runtime_limit.json'),'observer_sha256':suite['observer_sha256'],'started_unix':start,'deadline_unix':hard,'process_lifetime_isolation_verified':False};write_new(job/'dispatch_wrapper.json',dispatch)
        while child.poll() is None:
            for row in owned_tree(child.pid):owned[(row['pid'],row['create_time'])]=row
            now=time.time()
            if now>=hard and not stop_written:
                if not (job/'STOP.json').exists():write_new(job/'STOP.json',{'reason':'confirmation_absolute_900s_or_global_deadline','at_unix':now})
                stop_written=True
            if now>=hard+30:
                value={**dispatch,'state':'ATTENTION_STILL_EXITING','owned':list(owned.values()),'observed_seconds':now-start,'no_force_kill':True};write_new(job/'lifetime_receipt.json',value);return 1
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        times=process_times(child._handle);elapsed=times['exit_time']-start
    retirement_started=time.monotonic();live=wait_owned(list(owned.values()),seconds=3);retirement_wait=time.monotonic()-retirement_started;result=json.loads((job/'result.json').read_bytes()) if (job/'result.json').exists() else {};retired=not (Path(suite['research_root'])/'.research_gpu.lock').exists();success=child.returncode==0 and result.get('state')=='COMPLETE' and result.get('within_absolute_deadline') is True and not live and retired and 0<elapsed<=900 and times['exit_time']<=hard
    life={**dispatch,'state':'VERIFIED_EXIT' if success else 'ATTENTION','returncode':child.returncode,'precise_GetProcessTimes':True,'actual_exit_unix':times['exit_time'],'full_process_seconds':elapsed,'owned':list(owned.values()),'owned_still_alive':live,'GPU_lock_retired':retired,'result_state':result.get('state'),'full_process_budget_includes_import_initialization_cleanup':True,'outer_owned_retirement_wait_seconds':retirement_wait,'native_exit_measurement_excludes_parent_retirement_wait':True};write_new(job/'lifetime_receipt.json',life)
    if not live and retired and result:
        try:seal_job(job,out)
        except BaseException as exc:write_new(control/(job_id+'_seal_failure.json'),{'state':'ATTENTION','error':str(exc),'at_utc':utc()});return 1
    return 0 if success else 1


def seal_suite(control,suite):
    """48子项全部真实退出及封闭后封唯一suite小控制；不扫描G预测树。"""
    control=Path(control);out=Path(suite['output_group']);remaining,done=remaining_jobs(control,suite);require(not remaining and len(done)==48,'suite未全48闭合')
    require(not (Path(suite['research_root'])/'.research_gpu.lock').exists(),'suite封闭前GPU锁未退休')
    destination=out/'control';require(not destination.exists(),'suite小控制已存在，不覆盖')
    roots={'suite_request.json','code_bindings.json','observe.py','suite_dispatch.json','suite_result.json','STOP.json'};rows=[]
    for p in control.iterdir():require(p.name in roots|{'code','jobs'},'suite含未知小控制路径')
    for name in roots:
        p=control/name
        if p.exists():rows.append((p,name))
    for name in suite['source_code_sha256']:rows.append((control/'code/futsalmot'/name,'code/futsalmot/'+name))
    allowed_job={'job_request.json','runtime_limit.json','dispatch_wrapper.json','lifetime_receipt.json','result.json','STOP.json'}
    for job_id in done:
        for p in (control/'jobs'/job_id).iterdir():
            require(p.is_file() and p.name in allowed_job,'job含未知小控制');rows.append((p,'jobs/'+job_id+'/'+p.name))
    total=sum(p.stat().st_size for p,rel in rows);require(total<=CONTROL_BOUND,'suite小控制超过32MiB')
    records=[];destination.mkdir()
    for p,rel in rows:
        no_links(p);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);digest=sha(p);require(sha(target)==digest,'suite小控制写后SHA不符');records.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest})
    value={'schema_version':'bbox_confirmation_suite_control_v1','state':'SEALED','scope':'control_snapshot_only','files':records,'file_count':len(records),'total_bytes':total,'completed_model_matches':48,'created_at_utc':utc()};write_new(destination/'CONTROL_MANIFEST.json',value);return reference(destination/'CONTROL_MANIFEST.json')


def observe_suite(control_group):
    """固定48序列，无评分分支/重试；只一项完成并退出封闭后接下一项。"""
    import psutil
    control=Path(control_group).absolute();suite=load_suite(control);require(not (control/'suite_dispatch.json').exists(),'suite已启动，不自动恢复重放')
    write_new(control/'suite_dispatch.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'started_at_utc':utc(),'suite_reference':reference(control/'suite_request.json'),'fixed_evaluations':48,'automatic_retry':False})
    completed=[]
    try:
        for row in suite['jobs']:
            prepare_job(control,row['job_id'])
            if observe_job(control,row['job_id'])!=0:raise RuntimeError('确认子项失败/超界/未闭合，不继续后项')
            completed.append(row['job_id'])
        write_new(control/'suite_result.json',{'state':'COMPLETE','completed':completed,'finished_at_utc':utc(),'inference_only':True,'optimizer_updates':0});seal_suite(control,suite);return 0
    except BaseException as exc:
        if not (control/'suite_result.json').exists():write_new(control/'suite_result.json',{'state':'FAIL_STOP','completed':completed,'error':str(exc),'traceback':traceback.format_exc(),'finished_at_utc':utc(),'no_retry':True})
        else:write_new(control/'suite_seal_failure.json',{'state':'ATTENTION','error':str(exc),'finished_at_utc':utc()})
        return 1
