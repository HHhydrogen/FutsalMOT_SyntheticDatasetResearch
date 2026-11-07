"""已批准16项BBox配对复现的CPU串行生命周期；复用准入不伪称新数值预检。"""
from __future__ import annotations
import argparse,ast,ctypes,hashlib,importlib.util,inspect,json,os,re,shutil,subprocess,sys,time,traceback,uuid
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from ctypes import wintypes

R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch')
G=Path('G:/FutsalMOT_Training/runs')
BASE=R/'runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01'
SOURCE_SHA='c6eaf4117441747aee277bcc4ffcbdd79c7b99b4d8341aecc2eb4a25f3c37e83'
PAPER_SHA='d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17'
DEADLINE='2026-11-07T23:55:00+00:00'
SPACE=24295280650
RESERVE=100*1024**3
CONTROL=32*1024**2
MAX_JOB=5400
CLOSE_ALLOWANCE=600
SCHEMA='bbox_label_curve_batch_v1'
REUSE_SCHEMA='bbox_label_curve_reuse_admission_v1'
ORDER=[{'seed':seed,'real_fraction':fraction,'strategy':strategy} for seed in (29,43) for fraction in (1.,.5,.25,.1) for strategy in ('R','R+S')]


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def digest(value):require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'缺SHA256');return value


def ordinary(path,*,exists=True):
    path=Path(path).absolute();require(path.is_absolute() and '..' not in path.parts and not str(path).startswith(('\\\\','//')),'仅本地明确绝对路径')
    for part in reversed((path,*path.parents)):
        if not part.exists():
            if exists:raise FileNotFoundError(str(part))
            continue
        require(not part.is_symlink() and not part.is_junction(),'拒绝链接路径:'+str(part))
    return path


def read(path,maximum=32*1024**2):
    path=ordinary(path);require(0<path.stat().st_size<=maximum,'小JSON越界:'+str(path))
    return json.loads(path.read_bytes())


def reference(path):return {'path':str(Path(path).absolute()),'sha256':sha(path)}
def read_ref(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'必须精确path/SHA小引用');digest(ref['sha256']);path=ordinary(ref['path']);require(path.stat().st_size<=32*1024**2 and sha(path)==ref['sha256'],'小引用已变:'+str(path));return read(path)


def write_new(path,value):
    path=Path(path);ordinary(path,exists=False)
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())


def atomic(path,value):
    path=Path(path);tmp=path.parent/('.state_'+uuid.uuid4().hex+'.tmp');write_new(tmp,value);os.replace(tmp,path)


def append_log(value):
    with (R/'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:f.write(value.encode('utf8'));f.flush();os.fsync(f.fileno())


def jobs_equal(jobs):return jobs==ORDER and all(type(x.get('seed')) is int and type(x.get('real_fraction')) in (int,float) and type(x.get('strategy')) is str for x in jobs)


def validate_request(request):
    required={'plan_sha256','subset_manifest_references','input_cache','baseline_R_status_reference','validated_family','closed_predecessor','source_code_reference'}
    require(type(request) is dict and required<=set(request),'队列请求缺必要绑定')
    digest(request['plan_sha256']);require(request.get('jobs',ORDER)==ORDER,'只能固定seed29再43四标注量R/混合16项')
    require(set(request['subset_manifest_references'])=={'100','50','25','10'},'缺四冻结子集')
    require(type(request['validated_family']) is list and len(request['validated_family'])==4,'必须四项实际已完成配置家族')
    require(request['source_code_reference'].get('sha256')==SOURCE_SHA,'42源前端不符')
    for row in request['validated_family']:
        require(set(row)>={'admission_reference','status_reference','control_manifest_reference'},'家族引用不齐')
    require(set(request['closed_predecessor'])>={'status_reference','control_manifest_reference'},'前项未绑定')
    return True


def prepare_queue(batch_dir,request):
    """只冻结G小队列，不初始化CUDA、不派发；root另行执行返回launch命令。"""
    validate_request(request);batch=Path(batch_dir).absolute();require(batch.parent==G and batch.name.startswith('bbox_label_curve_batch_'),'队列只准G直接子目录')
    ordinary(batch,exists=False);require(not batch.exists(),'既有队列不可覆写/恢复')
    require(sha(R/'docs/plan/EXPERIMENTS.md')==request['plan_sha256'] and sha(R/'docs/plan/PAPER.md')==PAPER_SHA,'方案基线不同')
    source=ordinary(request['source_code_reference']['path']);require(sha(source)==SOURCE_SHA,'数值前端已变')
    # 这里只核源码、小元数据引用，不load模型/读取图像或运行数值测试。
    for refs in request['validated_family']:
        for key in ('admission_reference','status_reference','control_manifest_reference'):read_ref(refs[key])
    for ref in request['subset_manifest_references'].values():read_ref(ref)
    batch.mkdir();(batch/'code').mkdir();shutil.copyfile(Path(__file__),batch/'batch.py');shutil.copyfile(source,batch/'code/label_curve_bbox.py')
    monitor=R/'tools/training_monitor/server.py';tree=ast.parse(monitor.read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_group_name');function=ast.get_source_segment(monitor.read_text(encoding='utf8'),node)
    naming='import re\nfrom datetime import datetime,timedelta,timezone\n\n'+function+'\n';(batch/'naming.py').write_text(naming,encoding='utf8',newline='\n')
    queue={**deepcopy(request),'schema_version':SCHEMA,'state':'FROZEN','created_at_utc':now(),'jobs':deepcopy(ORDER),'fresh_cpu_tests':False,'fresh_gpu_probe':False,'admission_policy':'validated_configuration_reuse','per_job_max_seconds':MAX_JOB,'per_job_additional_space_bound_bytes':SPACE,'global_deadline_utc':DEADLINE,'code_references':{'batch':reference(batch/'batch.py'),'label_curve':reference(batch/'code/label_curve_bbox.py'),'naming':reference(batch/'naming.py'),'base_code_bindings':reference(BASE/'code_bindings.json')},'known_closed_groups':deepcopy(request.get('known_closed_groups',{}))}
    write_new(batch/'queue.json',queue)
    launch="import sys\nfrom pathlib import Path\np=Path(__file__).absolute().parent\nsys.path.insert(0,str(p))\nif __name__=='__main__':\n import batch\n raise SystemExit(batch.run(p))\n"
    (batch/'launch.py').write_text(launch,encoding='utf8',newline='\n')
    return {'batch_dir':str(batch),'queue_reference':reference(batch/'queue.json'),'launch_reference':reference(batch/'launch.py'),'command':[str(R/'.venv/Scripts/python.exe'),'-B','-X','utf8','-u',str(batch/'launch.py')],'jobs':len(ORDER),'fresh_cpu_tests':False,'fresh_gpu_probe':False,'dispatched':False}


def _load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def load_training_module(batch,queue):
    for ref in queue['code_references'].values():
        path=ordinary(ref['path']);require(sha(path)==ref['sha256'],'冻结CPU/数值源码已变')
    sys.path.insert(0,str(BASE/'code'))
    module=_load_module(batch/'code/label_curve_bbox.py','futsalmot.label_curve_bbox')
    require(sha(module.__file__)==SOURCE_SHA,'训练前端字节已变');return module


def _field_equal(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def verify_control(ref):
    m=read_ref(ref);require(m.get('state')=='SEALED' and m.get('scope')=='control_snapshot_only','控制清单未封闭')
    folder=Path(ref['path']).parent;total=0
    require(len(m['files'])<=4096,'控制条目越界')
    for row in m['files']:
        rel=Path(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'控制相对路径越界');p=ordinary(folder/rel);require(row['bytes']<=16*1024**2 and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'控制小文件已变:'+str(p));total+=row['bytes']
    require(total==m['total_bytes']<=CONTROL and len(m['files'])==m['file_count'],'控制大小/计数不同');return m


def validate_family(queue,code):
    actual=[]
    for item in queue['validated_family']:
        admission=read_ref(item['admission_reference']);status=read_ref(item['status_reference']);verify_control(item['control_manifest_reference'])
        require(admission.get('schema_version')=='bbox_label_curve_admission_v1' and admission.get('status')=='PASS' and admission.get('seed')==17 and admission.get('diagnostic_updates')==3,'继承必须真实seed17机械PASS')
        require(status.get('state')=='COMPLETE' and status.get('update')==1500,'继承作业未完整结束')
        require(admission.get('executed_code_sha256')==code and admission['source_bindings']['source_code']==code,'继承42实际源码不同')
        exact=admission.get('measurements',{}).get('exact_restore',{});require(exact and all(x is True for x in exact.values()) and exact.get('loss_and_metrics') is True and exact.get('actual_gradient_tensors') is True,'继承全状态/下一步exact证据不齐')
        require(admission.get('cuda_memory_fraction')==.8,'继承显存策略不同')
        actual.append((admission['real_fraction'],admission['strategy']))
    require(set(actual)=={(.5,'R'),(.5,'R+S'),(.1,'R'),(.1,'R+S')},'继承不是已通过的四配置家族')
    predecessor=read_ref(queue['closed_predecessor']['status_reference']);verify_control(queue['closed_predecessor']['control_manifest_reference']);require(predecessor.get('state')=='COMPLETE','队列前项未COMPLETE')
    return True


def _validate_reuse_admission(actual,config,queue,queue_ref,code):
    require(actual.get('schema_version')==REUSE_SCHEMA and actual.get('status')=='PASS' and actual.get('scope')=='validated_configuration_reuse','仅明确复用准入，不伪称新实测')
    require(actual.get('fresh_cpu_tests') is False and actual.get('fresh_gpu_probe') is False and actual.get('diagnostic_updates')==0 and actual.get('measured_probe_seconds')==0,'不得伪造新诊断')
    require(actual.get('queue_reference')==queue_ref and actual.get('inherited_configuration_family')==queue['validated_family'],'复用授权/原证据链不同')
    require(actual.get('source_code_sha256')==code and 'executed_code_sha256' not in actual,'复用只能绑定实际将冻结source集合')
    require(actual.get('seed') in (29,43) and actual['seed']==config['seed'] and actual.get('checkpoint_every')==1000 and actual.get('schedule_policy')=='single_stage_lr001_warm100_cosine_v1','不是固定复现种子/调度')
    require(actual.get('EXPERIMENTS_sha256')==queue['plan_sha256']==config['EXPERIMENTS_sha256'],'新方案复用范围不同')
    require(sha(queue_ref['path'])==queue_ref['sha256'],'队列已变')


def derive_reuse_prepare(module,queue,queue_ref,code):
    """只改准备端准入证据类型；原数值42与原prepare原文件不改。"""
    original=inspect.getsource(module.prepare_group);source=original
    pairs=[
      ("require(actual.get('schema_version')=='bbox_label_curve_admission_v1' and actual.get('status')=='PASS','缺实际mixed PASS准入')","require(actual.get('schema_version')=='bbox_label_curve_reuse_admission_v1' and actual.get('status')=='PASS','缺明确配置复用准入')"),
      ("require(actual.get('checkpoint_every')==1000 and actual.get('diagnostic_updates')==3,'准入检查点/真实诊断次数不符')","require(actual.get('checkpoint_every')==1000 and actual.get('diagnostic_updates')==0,'复用检查点/零新诊断不同')"),
      ("require(actual.get('measurements',{}).get('temporary_models_removed') is True and actual.get('schedule_policy')==SCHEDULE_POLICY,'诊断模型未清或调度变化')","require(actual.get('fresh_gpu_probe') is False and actual.get('schedule_policy')==SCHEDULE_POLICY,'复用声明/调度不同')"),
      ("exact=actual.get('measurements',{}).get('exact_restore',{})\n    require(exact and all(value is True for value in exact.values()) and exact.get('loss_and_metrics') is True and exact.get('actual_gradient_tensors') is True,'未实际通过完整恢复/下一步loss及梯度exact')","_batch_validate_reuse(actual,config)"),
      ("actual.get('executed_code_sha256')==hashes","actual.get('source_code_sha256')==hashes")]
    for before,after in pairs:require(source.count(before)==1,'原准备接点不同');source=source.replace(before,after)
    reverse=source
    for before,after in reversed(pairs):require(reverse.count(after)==1,'逆准备接点不同');reverse=reverse.replace(after,before)
    require(ast.dump(ast.parse(reverse),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False),'准备端除明确复用准入外AST变化')
    namespace=dict(module.__dict__);namespace['_batch_validate_reuse']=lambda actual,config:_validate_reuse_admission(actual,config,queue,queue_ref,code)
    exec(compile(source,str(Path(__file__).absolute())+':reuse_prepare_only','exec'),namespace)
    return namespace['prepare_group'],{'original_prepare_source_sha256':hashlib.sha256(original.encode()).hexdigest(),'derived_prepare_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'AST_inverse_exact':True,'substitution_count':len(pairs),'scope':'prepare_only_validated_configuration_reuse; training_source42_unchanged'}


def space_and_time(remaining):
    require(type(remaining) is int and 1<=remaining<=16,'剩余数量非法')
    freeD=shutil.disk_usage(R).free;freeG=shutil.disk_usage(G).free
    require(freeD>=RESERVE+remaining*CONTROL,'D完整剩余控制预算不足')
    require(freeG>=RESERVE+remaining*SPACE,'G完整剩余训练预算不足')
    deadline=datetime.fromisoformat(DEADLINE).timestamp();require(time.time()+remaining*(MAX_JOB+CLOSE_ALLOWANCE)<deadline,'剩余全部90分钟训练及收口不能在原Nov7截止内闭合')
    return {'D_free_bytes':freeD,'G_free_bytes':freeG,'remaining_jobs':remaining,'remaining_G_upper_bytes':remaining*SPACE,'remaining_control_upper_bytes':remaining*CONTROL,'G_reserve_bytes':RESERVE,'D_reserve_bytes':RESERVE,'per_job_max_seconds':MAX_JOB,'per_job_additional_space_bound_bytes':SPACE,'global_deadline_utc':DEADLINE}


def process_row(process):return {'pid':process.pid,'create_time':process.create_time(),'command':process.cmdline()}

def matching_alive(row):
    import psutil
    try:
        p=psutil.Process(row['pid']);return abs(p.create_time()-row['create_time'])<.01
    except psutil.NoSuchProcess:return False
    except psutil.AccessDenied:raise RuntimeError('owned身份无法核验')


def unknown_writers(batch):
    import psutil
    own=psutil.Process();excluded={own.pid,*[p.pid for p in own.parents()]};unknown=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.info['pid'] in excluded:continue
        n=(p.info['name'] or '').lower();cmd=p.info['cmdline'] or []
        if n.startswith('unrealeditor') or ('python' in n and (any(str(x).replace('\\','/').lower().endswith('/launch.py') for x in cmd[1:]) or '--multiprocessing-fork' in cmd)):unknown.append(p.info)
    return unknown


def choose_group(batch,queue,job,completed,naming):
    at=datetime.now(timezone.utc);allowed={x['group_name'] for x in completed}
    known=queue.get('known_closed_groups',{})
    for attempt in range(1,100):
        name=naming.canonical_group_name(purpose='main',task='detect',model='yolo11s',panel='soccernet',strategy=job['strategy'],seed=job['seed'],at=at,attempt=attempt)
        d=R/'runs'/name;g=G/name
        if not d.exists() and not g.exists():return d,g
        if name in allowed:continue
        failed=queue.get('known_preparation_failures',{})
        if name in failed:
            receipt=read_ref(failed[name]);require(receipt.get('schema')=='bbox_batch_prepare_failure_v1' and receipt.get('state')=='VERIFIED_PREPARATION_ONLY_FAILURE' and Path(receipt.get('group_path',''))==d,'仅明确绑定的零派发准备失败可消解命名')
            result=read_ref(receipt['batch_result_reference']);require(result.get('state')=='FAIL_STOP' and result.get('completed')==[] and result.get('error_type')=='ValueError' and result.get('error')=='准入收据不一致','原准备失败身份不同')
            require(d.is_dir() and not g.exists() and {p.name for p in d.iterdir()}=={'admission.json','storage_revision.json'},'准备失败组已有未知产物，拒绝重放')
            for key,file in [('admission_reference','admission.json'),('storage_reference','storage_revision.json')]:
                require(Path(receipt[key]['path'])==d/file,'失败小收据越组');read_ref(receipt[key])
            continue
        if name in known:
            receipt=read_ref(known[name]);require(receipt.get('state')=='VERIFIED_COMPLETE','已知碰名组未完整闭合');read_ref(receipt['status_reference']);verify_control(receipt['control_manifest_reference']);continue
        raise FileExistsError('未知D/G候选不得覆盖或猜增attempt:'+name)
    raise RuntimeError('同日命名空间耗尽')


def make_reuse_admission(module,queue,queue_ref,job,group,code):
    from futsalmot import runner,formal
    from futsalmot.runner_data import real_panel
    from futsalmot.catalogues import synthetic_bbox_catalogue
    from futsalmot.input_cache import validate_ready_cache
    full=real_panel(runner.REAL);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');require((len(full['train_uids']),len(full['val_uids']),len(synthetic['ordered_uids']))==(2100,750,24149),'原数据数量变化')
    label=str(round(job['real_fraction']*100));subset_ref=queue['subset_manifest_references'][label];selected=module.filter_train_panel(full,subset_ref);subset=module.subset_binding(full,subset_ref);require(len(selected['train_uids'])==round(2100*job['real_fraction']),'实际标注池不同')
    cache=formal.cache_specification(queue['input_cache']);ready=validate_ready_cache(cache,synthetic['binding']);family=read_ref(queue['validated_family'][0]['admission_reference'])
    public=family['source_bindings']['public_model']
    bindings={**subset,'real_panel':full['binding'],'synthetic_panel':synthetic['binding'],'public_model':public,'requirements_sha256':sha(R/'requirements.lock'),'source_code':code,'EXPERIMENTS_sha256':queue['plan_sha256'],'PAPER_sha256':PAPER_SHA,'input_cache':cache,'baseline_R_status_reference':queue['baseline_R_status_reference'],'scope':'validated_configuration_reuse_metadata_not_new_probe'}
    value={'schema_version':REUSE_SCHEMA,'status':'PASS','scope':'validated_configuration_reuse','fresh_cpu_tests':False,'fresh_gpu_probe':False,'diagnostic_updates':0,'measured_probe_seconds':0,'seed':job['seed'],'strategy':job['strategy'],'real_fraction':job['real_fraction'],'N_R':len(selected['train_uids']),'full_N_R':2100,'updates':1500,'numeric_mode':'fp32','microbatch_size':8,'effective_batch':16,'num_workers':2,'sampler_contract':module.sampler_contract(job['strategy']),'checkpoint_every':1000,'schedule_policy':module.SCHEDULE_POLICY,'input_cache':cache,'baseline_R_status_reference':queue['baseline_R_status_reference'],'subset_manifest_reference':subset_ref,'ordered_subset_train_pool_sha256':subset['ordered_subset_train_pool_sha256'],'cuda_memory_fraction':.8,'raw_upper_per_image_bytes':6650344,'checkpoint_bytes':114293807,'source_code_sha256':code,'source_bindings':bindings,'queue_reference':queue_ref,'EXPERIMENTS_sha256':queue['plan_sha256'],'inherited_configuration_family':queue['validated_family'],'input_cache_validation':ready,'created_at_utc':now()}
    write_new(group/'admission.json',value);return read(group/'admission.json')


def get_process_times(handle):
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    values=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(wintypes.HANDLE(int(handle)),*(ctypes.byref(v) for v in values)):raise ctypes.WinError(ctypes.get_last_error())
    def seconds(v):return ((v.dwHighDateTime<<32)|v.dwLowDateTime)/1e7-11644473600
    return {'create_time':seconds(values[0]),'exit_time':seconds(values[1])}


def request_controlled_stop(batch,group,owned,reason):
    import psutil
    candidates=[]
    for row in owned.values():
        if str(group/'launch.py') in row['command'] and matching_alive(row):
            p=psutil.Process(row['pid']);require(str(group/'launch.py') in p.cmdline(),'停止命令身份改变');candidates.append(row)
    require(candidates,'无准确当前child可写STOP')
    path=group/'STOP.json'
    if not path.exists():write_new(path,{'schema':'bbox_batch_controlled_stop_v1','reason':reason,'batch_directory':str(batch),'requested_at_utc':now(),'owned_identities':candidates})
    return True


def dispatch_and_wait(batch,group,out,queue,index):
    import psutil
    cmd=[str(R/'.venv/Scripts/python.exe'),'-B','-X','utf8','-u',str(group/'launch.py')];env=os.environ.copy();env.update(YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    started=time.time();owned={};stop=False;stop_at=None
    with (out/'controller.stdout.log').open('xb') as stdout,(out/'controller.stderr.log').open('xb') as stderr:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        times=get_process_times(child._handle);dispatch={'schema':'bbox_batch_child_dispatch_v1','batch_queue_reference':reference(batch/'queue.json'),'index':index,'started_at_utc':now(),'wrapper_pid':child.pid,'wrapper_create_time':times['create_time'],'command':cmd,'launch_sha256':sha(group/'launch.py'),'job_table_sha256':sha(group/'job_table.json'),'code_bindings_sha256':sha(group/'code_bindings.json'),'admission_sha256':sha(group/'admission.json'),'storage_revision_sha256':sha(group/'storage_revision.json'),'process_lifetime_isolation_verified':False}
        # 持有句柄；首次短轮观察controller身份后写唯一dispatch，不等待PT0结束。
        for _ in range(30):
            try:
                for p in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                    row=process_row(p);owned[(row['pid'],row['create_time'])]=row
            except psutil.NoSuchProcess:pass
            if any(x['pid']!=child.pid and str(group/'launch.py') in x['command'] for x in owned.values()) or child.poll() is not None:break
            time.sleep(.1)
        dispatch['processes']=list(owned.values());write_new(group/'dispatch.json',dispatch)
        atomic(batch/'state.json',{'state':'RUNNING','index':index,'jobs':16,'active_group':str(group),'active_output':str(out),'active_job':read(group/'job_table.json')['jobs'][0]['job_id'],'child_dispatch_reference':reference(group/'dispatch.json'),'updated_at_utc':now()})
        while child.poll() is None:
            try:
                parent=psutil.Process(child.pid)
                for p in [parent,*parent.children(recursive=True)]:
                    try:row=process_row(p);owned[(row['pid'],row['create_time'])]=row
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            except psutil.NoSuchProcess:pass
            if (batch/'STOP.json').exists() and not stop:
                request_controlled_stop(batch,group,owned,'user_batch_STOP');stop=True;stop_at=time.time()
            if time.time()>started+MAX_JOB+CLOSE_ALLOWANCE and not stop:
                request_controlled_stop(batch,group,owned,'batch_child_runtime_and_cleanup_bound');stop=True;stop_at=time.time()
            if stop and time.time()>stop_at+300:raise RuntimeError('ATTENTION:准确child已请求受控停止但仍未退出，无force kill/续派')
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        end=get_process_times(child._handle)
    live=[];until=time.monotonic()+30
    while True:
        live=[x for x in owned.values() if matching_alive(x)]
        if not live or time.monotonic()>=until:break
        time.sleep(.1)
    lifetime={'state':'VERIFIED_EXIT' if not live else 'ATTENTION_OWNED_REMAIN','wrapper_returncode':child.returncode,'actual_exit_unix':end['exit_time'],'full_process_seconds':end['exit_time']-started,'precise_GetProcessTimes':True,'owned':list(owned.values()),'owned_still_alive':live,'batch_stop_requested':stop,'no_force_kill':True};write_new(out/'batch_lifetime_receipt.json',lifetime)
    require(not live,'owned仍活，禁止下一项');require(child.returncode==0,'child失败退出，禁止自动重试');require(not stop,'人工停止/运行界限，禁止下一项');return lifetime


def verify_job(group,out,job,lifetime,batch):
    leaf=out/('label'+str(round(job['real_fraction']*100))+('_R' if job['strategy']=='R' else '_R_plus_S')+f"_seed{job['seed']}")
    s=read(leaf/'status.json');gs=read(out/'status.json');exp={'R':24000,'S':0} if job['strategy']=='R' else {'R':12000,'S':12000}
    require((s.get('state'),s.get('update'),s.get('exposures'),gs.get('state'))==('COMPLETE',1500,exp,'LABEL_CURVE_COMPLETE'),'作业未完整成功，不自动恢复')
    cpfile=ordinary(leaf/'checkpoints.jsonl');require(cpfile.stat().st_size<=2*1024**2,'检查点小收据越界');cp=[json.loads(x) for x in cpfile.read_text().splitlines()];require({0,1000,1500}<={x['update'] for x in cp},'缺计划完整cp')
    for row in cp:
        path=ordinary(row['receipt']['path']);require(path.parent==leaf/'checkpoints' and path.stat().st_size==row['receipt']['bytes'],'cp收据bytes/路径不同')
    vals=[]
    for tag,u in [('step000000_final_PT',0),('step001500_final_final',1500)]:
        path=leaf/'evaluation'/tag/'summary.json';v=read(path);require((v.get('tag'),v.get('update'),v.get('images'))==(tag,u,750),'val端点不同');vals.append({'tag':tag,'reference':reference(path),'main_metrics':v['main_metrics']})
    require(v==s['terminal'],'末EMA summary与status不同')
    p=ordinary(leaf/'steps.jsonl')
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2*1024**2));last=json.loads(f.read().splitlines()[-1])
    require((last['update'],last['R_exposures'],last['S_exposures'])==(1500,exp['R'],exp['S']),'末提交/曝光不同')
    mref=reference(out/'control/CONTROL_MANIFEST.json');m=verify_control(mref)
    require(not (R/'.research_gpu.lock').exists(),'GPU锁尚在');require(not unknown_writers(batch),'未知训练/UE写者，不续派')
    receipt={'state':'VERIFIED_COMPLETE','verified_at_utc':now(),'batch_queue_reference':reference(batch/'queue.json'),'status_reference':reference(leaf/'status.json'),'control_manifest_reference':mref,'global_updates':1500,'exposures':exp,'N_R':round(2100*job['real_fraction']),'seed':job['seed'],'strategy':job['strategy'],'checkpoints':len(cp),'checkpoint_receipt_bytes_verified':True,'validation_summaries':vals,'control_files':m['file_count'],'control_bytes':m['total_bytes'],'terminal_step_bounded_tail_verified':True,'owned':lifetime['owned'],'GPU_lock_retired':True,'unknown_training_writers':[],'main_metrics':s['terminal']['main_metrics'],'PT_NPZ_media_not_hashed':True}
    write_new(out/'completion_verification.json',receipt)
    append_log(f"\n\n## LABEL-CURVE-BATCH 验收：{leaf.name}\n\nUTC {now()}。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R={receipt['N_R']}/曝光{exp}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={json.dumps(receipt['main_metrics'],ensure_ascii=False)}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据{out/'completion_verification.json'}；无best/test/新策略。\n")
    return {'group_name':group.name,'group':str(group),'output':str(out),'job':deepcopy(job),'completion_reference':reference(out/'completion_verification.json'),'status_reference':receipt['status_reference'],'main_metrics':receipt['main_metrics']}


def run(batch_dir):
    import psutil
    batch=ordinary(batch_dir);require(batch.parent==G and batch.name.startswith('bbox_label_curve_batch_'),'非法队列目录')
    queue=read(batch/'queue.json');validate_request(queue);require(queue.get('schema_version')==SCHEMA and queue.get('state')=='FROZEN' and jobs_equal(queue['jobs']),'队列未冻结或顺序变更')
    require(not any((batch/x).exists() for x in ('dispatch.json','result.json','state.json')),'本队列已调用，禁止重放/恢复')
    queue_ref=reference(batch/'queue.json');completed=[];lock_path=G/'label_curve_batch.lock';lock=None
    try:
        own=process_row(psutil.Process());lock={'schema':SCHEMA,'batch_directory':str(batch),'pid':own['pid'],'create_time':own['create_time'],'created_at_utc':now()};write_new(lock_path,lock)
        write_new(batch/'dispatch.json',{'schema':SCHEMA,'started_at_utc':now(),'queue_reference':queue_ref,'runtime_identity':own,'process_lifetime_isolation_verified':False});write_new(batch/'runtime_identity.json',own)
        require(not (batch/'STOP.json').exists(),'队列已被人工STOP');require(sha(R/'docs/plan/EXPERIMENTS.md')==queue['plan_sha256'] and sha(R/'docs/plan/PAPER.md')==PAPER_SHA,'方案已变')
        require(not unknown_writers(batch) and not (R/'.research_gpu.lock').exists(),'GPU/未知训练或UE未释放')
        module=load_training_module(batch,queue);code=module.execution_code();require(len(code)==42 and code.get('label_curve_bbox.py')==SOURCE_SHA,'非精确42数值集合')
        validate_family(queue,code);naming=_load_module(batch/'naming.py','batch_frozen_naming');prepare,proof=derive_reuse_prepare(module,queue,queue_ref,code);write_new(batch/'prepare_derivation.json',proof)
        for index,job in enumerate(queue['jobs']):
            require(not (batch/'STOP.json').exists(),'人工STOP，停止队列');require(sha(batch/'queue.json')==queue_ref['sha256'],'队列被改')
            require(sha(R/'docs/plan/EXPERIMENTS.md')==queue['plan_sha256'] and sha(R/'docs/plan/PAPER.md')==PAPER_SHA,'方案变化');require(not (R/'.research_gpu.lock').exists() and not unknown_writers(batch),'前项GPU/未知写者未释放')
            resource=space_and_time(16-index);group,out=choose_group(batch,queue,job,completed,naming);ordinary(group,exists=False);group.mkdir()
            admission=make_reuse_admission(module,queue,queue_ref,job,group,code)
            storage={'schema':'bbox_label_curve_reuse_storage_v1',**resource,'additional_space_bound_bytes':SPACE,'control_space_bound_bytes':CONTROL,'checkpoint_bytes':114293807,'raw_upper_per_image_bytes':6650344,'formula':'3cp*ceil114293807*1.1 +1500*2MiB +2*750*(6650344+1MiB)+one750partial+3GiB+2cp','inherited_actual_configuration_family':queue['validated_family'],'fresh_measurements':False,'created_at_utc':now()};write_new(group/'storage_revision.json',storage)
            cfg=module.protocol(job['strategy'],job['seed'],real_fraction=job['real_fraction'],subset_manifest_reference=admission['subset_manifest_reference'],plan_sha256=queue['plan_sha256'],input_cache=admission['input_cache'],baseline_R_status_reference=queue['baseline_R_status_reference'],additional_space_bound_bytes=SPACE,raw_upper_per_image_bytes=6650344,checkpoint_bytes=114293807,admission_sha256=sha(group/'admission.json'),max_run_seconds=MAX_JOB)
            prepared=prepare(group,config=cfg,admission=admission,code_source=BASE/'code/futsalmot');require(Path(prepared['output_group'])==out,'G输出绑定不同')
            lifetime=dispatch_and_wait(batch,group,out,queue,index);completed.append(verify_job(group,out,job,lifetime,batch));write_new(batch/f'job_{index:02d}_completion.json',completed[-1])
            atomic(batch/'state.json',{'state':'BETWEEN_JOBS' if index<15 else 'COMPLETE','completed':len(completed),'jobs':16,'last_completed':completed[-1],'updated_at_utc':now()})
        result={'schema':SCHEMA,'state':'COMPLETE','completed':16,'jobs':completed,'finished_at_utc':now(),'queue_reference':queue_ref,'fresh_cpu_tests':False,'fresh_gpu_probe':False,'no_further_training_authorized_by_batch':True,'independent_match_evaluation_not_dispatched':True};write_new(batch/'result.json',result)
        append_log(f"\n\n## LABEL-CURVE-BATCH COMPLETE\n\nUTC {now()}。seed29/43、100/50/25/10四固定标注量xR/混合共16项已逐项完整验收；复用准入无新CPU/GPU数值预检，原42数值代码保持。全部结果与小控制收据见{batch/'result.json'}。本队列不自动启动独立比赛评价或更多训练。\n")
        return 0
    except BaseException as exc:
        value={'schema':SCHEMA,'state':'USER_STOP' if (batch/'STOP.json').exists() else 'FAIL_STOP','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'completed':completed,'at_utc':now(),'queue_reference':queue_ref,'automatic_retry':False,'force_kill':False}
        if not (batch/'result.json').exists():write_new(batch/'result.json',value)
        atomic(batch/'state.json',{k:value[k] for k in ('state','error_type','error','at_utc','automatic_retry')})
        append_log(f"\n\n## LABEL-CURVE-BATCH {value['state']}\n\nUTC {now()}。已完整闭合{len(completed)}/16，具体{type(exc).__name__}: {exc}。保留{batch/'result.json'}，无自动重试/恢复/强杀或跳过。\n")
        return 1
    finally:
        if lock is not None and lock_path.exists():
            try:
                if read(lock_path)==lock:lock_path.unlink()
            except BaseException:pass


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True);p=sub.add_parser('prepare');p.add_argument('batch_dir');p.add_argument('request_json');p=sub.add_parser('run');p.add_argument('batch_dir');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare_queue(args.batch_dir,read(args.request_json)),ensure_ascii=False));return 0
    return run(args.batch_dir)


if __name__=='__main__':raise SystemExit(main())
