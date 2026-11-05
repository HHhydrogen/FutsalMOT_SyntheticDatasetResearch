"""仅root显式调用的29/43训练seed四标注档R1500/mixed1500复现预检观察器：保留进程句柄，完整退出计时，不初始化CUDA。"""
from pathlib import Path
from datetime import datetime,timezone
import ctypes,json,os,subprocess,sys,time,math
from ctypes import wintypes
import psutil
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch')
G=Path('G:/FutsalMOT_Training/runs')

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())

def process_times(handle):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    values=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(wintypes.HANDLE(int(handle)),*(ctypes.byref(v) for v in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    def seconds(v):return ((v.dwHighDateTime<<32)|v.dwLowDateTime)/1e7-11644473600
    return {'create_time':seconds(values[0]),'exit_time':seconds(values[1])}

def is_console_host(row):
    command=row.get('cmdline') or []
    return bool(command) and Path(command[0]).name.lower()=='conhost.exe'


def partition_live_owned(rows):
    return ([row for row in rows if not is_console_host(row)],[row for row in rows if is_console_host(row)])


def live_owned(owned):
    live=[]
    for row in owned.values():
        try:
            p=psutil.Process(row['pid'])
            if abs(p.create_time()-row['create_time'])<.01:live.append(row)
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:live.append({**row,'identity_exit_unverifiable':True})
    return live


def _number(value):return type(value) in (int,float) and math.isfinite(value)

def _read_reference(reference):
    if type(reference) is not dict or set(reference)!={'path','sha256'}:raise ValueError('须精确path/SHA小收据引用')
    p=Path(reference['path']);digest=reference['sha256']
    if not p.is_absolute() or '..' in p.parts or str(p).startswith(('\\\\','//')) or type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('非法本地绝对引用或SHA')
    if any(x.is_symlink() or x.is_junction() for x in (p,*p.parents)):raise ValueError('不得跟随小收据链接')
    before=p.stat()
    if not 0<before.st_size<=1024**2:raise ValueError('小收据超出1MiB')
    raw=p.read_bytes();after=p.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or __import__('hashlib').sha256(raw).hexdigest()!=digest:raise ValueError('小收据字节或SHA变化')
    def pairs(rows):
        value={}
        for key,item in rows:
            if key in value:raise ValueError('小收据重复JSON键')
            value[key]=item
        return value
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('非有限JSON数值')))
    if type(value) is not dict:raise ValueError('小收据必须JSON对象')
    return value

def _failed_curve4_seconds(closed,lifetime_paths,passed_n50_control):
    """仅允许一次已闭合N50纯R协作超时；失败原样保留并计入实际账。"""
    attempts=closed.get('failed_preparation_attempts')
    if type(attempts) is not list or len(attempts)>1:raise ValueError('失败准备必须显式列表且最多一次同协议基础设施修订')
    if not attempts:return 0.
    row=attempts[0]
    if type(row) is not dict or row.get('job_id')!='label50_R_seed17':raise ValueError('仅N50_R_seed17一次失败可计入此既定修订')
    life=_read_reference(row.get('lifetime_reference'));admission=_read_reference(row.get('admission_reference'));lp=str(Path(row['lifetime_reference']['path']))
    if lp in lifetime_paths:raise ValueError('失败不能复用PASS lifetime')
    seconds=life.get('conservative_full_process_seconds')
    if life.get('state')!='ATTENTION' or type(life.get('returncode')) is not int or life['returncode']!=1 or life.get('precise_GetProcessTimes') is not True or not _number(seconds) or not 0<seconds<=60 or life.get('owned_still_alive')!=[] or life.get('GPU_lock_retired') is not True or life.get('admission_status')!='FAIL_STOP':raise ValueError('失败准备未在60秒内精确完整退出')
    failure=admission.get('failure');elapsed=admission.get('elapsed_seconds')
    if admission.get('schema_version')!='bbox_label_curve_admission_v1' or admission.get('status')!='FAIL_STOP' or admission.get('seed')!=17 or admission.get('real_fraction')!=.5 or admission.get('N_R')!=1050 or admission.get('strategy')!='R' or admission.get('diagnostic_updates')!=3 or type(failure) is not dict or failure.get('type')!='TimeoutError' or failure.get('message')!='labelcurve finite mechanical preparation exhausted' or not _number(elapsed) or not 50<=elapsed<=60:raise ValueError('只能保留明确3诊断步后协作时间超界，不能包装数值或未知失败')
    request_path=Path(row['lifetime_reference']['path']).parent/'preflight_request.json'
    q=_read_reference({'path':str(request_path),'sha256':life.get('request_sha256')})
    if q.get('seed')!=17 or q.get('strategy')!='R' or q.get('real_fraction')!=.5 or q.get('N_R')!=1050 or q.get('max_seconds')!=50 or q.get('full_process_limit_seconds')!=60 or q.get('budget_policy')!='label_curve4_240_v1' or type(q.get('group')) is not str or Path(row['admission_reference']['path'])!=Path(q['group'])/'admission.json':raise ValueError('失败lifetime原请求与原admission身份/协议不符')
    control_ref,control=passed_n50_control;files=control.get('files')
    if type(files) is not list:raise ValueError('N50已通过控制缺admission引用')
    receipt=[x for x in files if type(x) is dict and x.get('path')=='admission.json']
    if len(receipt)!=1:raise ValueError('N50已通过控制admission引用不唯一')
    passed=_read_reference({'path':str(Path(control_ref['path']).parent/'admission.json'),'sha256':receipt[0].get('sha256')})
    if passed.get('status')!='PASS' or passed.get('seed')!=17 or passed.get('strategy')!='R' or passed.get('real_fraction')!=.5 or passed.get('N_R')!=1050:raise ValueError('一次修订必须最终同协议N50_R实际PASS')
    fields=('seed','strategy','updates','schedule_policy','checkpoint_every','numeric_mode','microbatch_size','num_workers','effective_batch','sampler_contract','input_cache','baseline_R_status_reference','executed_code_sha256','source_bindings','subset_manifest_reference','ordered_subset_train_pool_sha256','N_R','full_N_R','real_fraction')
    if any(key not in admission or key not in passed or admission[key]!=passed[key] for key in fields):raise ValueError('一次修订改变训练数值/来源/代码/子集协议')
    if type(admission['executed_code_sha256']) is not dict or len(admission['executed_code_sha256'])!=42:raise ValueError('同协议失败与PASS必须保留42执行源码')
    return seconds

def validate_curve4_closed(config):
    """只读前四项完整退出链的小引用；不替代root逐小控制SHA终验。"""
    declared=config.get('completed_curve4_preparation_seconds')
    if not _number(declared) or not 0<=declared<=240:raise ValueError('前四项实际准备账非法')
    closed=_read_reference(config.get('completed_curve4_reference'))
    if closed.get('schema')!='bbox_labelcurve4_closed_v1' or closed.get('state')!='VERIFIED_COMPLETE':raise ValueError('前四项未完整闭合')
    rows=closed.get('four_jobs');expected={'label50_R_seed17':{'R':24000,'S':0},'label50_R_plus_S_seed17':{'R':12000,'S':12000},'label10_R_seed17':{'R':24000,'S':0},'label10_R_plus_S_seed17':{'R':12000,'S':12000}}
    if type(rows) is not list or len(rows)!=4:raise ValueError('必须四项真实闭合收据')
    seen=set();lifetime_paths=set();measured=0.;passed_n50_control=None
    for row in rows:
        if type(row) is not dict or row.get('job_id') not in expected or row['job_id'] in seen:raise ValueError('前四项身份缺失/重复/替换')
        seen.add(row['job_id']);status=_read_reference(row.get('status_reference'));control=_read_reference(row.get('control_manifest_reference'));life=_read_reference(row.get('lifetime_reference'))
        if status.get('state')!='COMPLETE' or status.get('schema_version')!='bbox_label_curve_v1' or status.get('job_id')!=row['job_id'] or status.get('update')!=1500 or status.get('exposures')!=expected[row['job_id']]:raise ValueError('前四项真实作业状态/更新/曝光不符')
        if control.get('state')!='SEALED' or control.get('scope')!='control_snapshot_only':raise ValueError('前四项控制未封闭')
        if row['job_id']=='label50_R_seed17':passed_n50_control=(row['control_manifest_reference'],control)
        lp=str(Path(row['lifetime_reference']['path']))
        if lp in lifetime_paths:raise ValueError('四项不能复制同一次机械收据')
        lifetime_paths.add(lp);seconds=life.get('conservative_full_process_seconds')
        if life.get('state')!='VERIFIED_EXIT' or type(life.get('returncode')) is not int or life['returncode']!=0 or life.get('precise_GetProcessTimes') is not True or not _number(seconds) or not 0<seconds<=60 or life.get('owned_still_alive')!=[] or life.get('GPU_lock_retired') is not True or life.get('admission_status')!='PASS':raise ValueError('前四项机械未精确60秒内完整退出')
        request_path=Path(row['lifetime_reference']['path']).parent/'preflight_request.json'
        prior_request=_read_reference({'path':str(request_path),'sha256':life.get('request_sha256')})
        ratio=.5 if row['job_id'].startswith('label50_') else .1
        strategy='R+S' if '_plus_S_' in row['job_id'] else 'R'
        if prior_request.get('seed')!=17 or prior_request.get('real_fraction')!=ratio or prior_request.get('N_R')!={.5:1050,.1:210}[ratio] or prior_request.get('strategy')!=strategy:raise ValueError('四项lifetime不对应声明策略/比例/seed')
        if type(prior_request.get('group')) is not str or type(control.get('control_group')) is not str or type(control.get('output_group')) is not str or type(status.get('directory')) is not str or Path(prior_request['group'])!=Path(control['control_group']) or Path(status['directory'])!=Path(control['output_group'])/row['job_id']:raise ValueError('四项lifetime/正式status/control不属于同一作业')
        measured+=seconds
    failed_seconds=_failed_curve4_seconds(closed,lifetime_paths,passed_n50_control);measured+=failed_seconds
    summary=closed.get('measured_preparation_seconds')
    if seen!=set(expected) or not _number(summary) or not math.isclose(summary,measured,rel_tol=0,abs_tol=1e-6) or not math.isclose(declared,measured,rel_tol=0,abs_tol=1e-6):raise ValueError('前四项实际准备账与逐项lifetime不符')
    return {'completed_curve4_reference':dict(config['completed_curve4_reference']),'completed_curve4_preparation_seconds':measured,'failed_curve4_preparation_seconds':failed_seconds,'failed_curve4_attempts':len(closed['failed_preparation_attempts']),'jobs':sorted(seen)}

def _validate_scope_budget(config):
    if type(config.get('seed')) is not int or config['seed'] not in (29,43):raise ValueError('本stage2仅seed29/43，不含17或新seed')
    fraction=config.get('real_fraction');count=config.get('N_R')
    if type(fraction) not in (int,float) or fraction not in (1.,.5,.25,.1) or type(count) is not int or count!={1.:2100,.5:1050,.25:525,.1:210}[fraction]:raise ValueError('本stage2须明确100/50/25/10%对应真实池')
    if config.get('strategy') not in ('R','R+S') or config.get('budget_policy')!='label_curve16_960_v1':raise ValueError('不是本16项策略/许可')
    if config.get('full_process_limit_seconds')!=60 or config.get('block_preparation_cap_seconds')!=960 or not _number(config.get('total_preparation_cap_seconds')) or not math.isclose(config['total_preparation_cap_seconds'],1995.8882010710282,rel_tol=0,abs_tol=1e-6):raise ValueError('stage2预算必须60/960/1995.8882010710282')
    previous=config.get('previous_gpu_preparation_seconds');block=config.get('previous_block_preparation_seconds');completed=config.get('completed_curve4_preparation_seconds')
    if not all(_number(x) for x in (previous,block,completed)) or block<0 or not 0<=completed<=240 or block+60>960:raise ValueError('stage2预算不足或账非有限')
    if not math.isclose(previous,795.8882010710282+completed+block,rel_tol=0,abs_tol=1e-6) or previous+60>1995.8882010710282+1e-6:raise ValueError('stage2未衔接前四项实际账或超累计界限')
    return {'seed':config['seed'],'real_fraction':fraction,'N_R':count,'previous_block_preparation_seconds':block}

def validate_request(config):
    """stage2启动前CPU核账及前四项完整闭合引用。"""
    scope=_validate_scope_budget(config)
    if config.get('max_seconds')!=50:raise ValueError('stage2协作窗口固定50秒')
    return {**scope,**validate_curve4_closed(config)}


def main(prep):
    prep=Path(prep).absolute()
    if prep.parent!=G or not prep.name.startswith('bbox_labelcurve_replicate_prepare_') or prep.is_symlink() or prep.is_junction():raise ValueError('只准独立G预算1500准备目录')
    request_path=prep/'preflight_request.json';config=json.loads(request_path.read_bytes());helper=prep/'preflight.py'
    validate_request(config)
    if config.get('preflight_helper_sha256')!=__import__('hashlib').sha256(helper.read_bytes()).hexdigest():raise ValueError('机械helper SHA变化')
    if (R/'.research_gpu.lock').exists():raise RuntimeError('前项GPU未释放')
    if any((prep/n).exists() for n in ('dispatch_wrapper.json','runtime_identity.json','preflight_config.json','lifetime_receipt.json','STOP_PREFLIGHT.json')):raise FileExistsError('本准备已调用，禁止重放')
    started=time.time();config['full_process_deadline_unix']=started+60;config['dispatch_started_unix']=started
    write_new(prep/'preflight_config.json',config)
    command=[str(R/'.venv/Scripts/python.exe'),'-B','-X','utf8','-u',str(helper)]
    owned={};stop_written=False
    with (prep/'stdout.log').open('xb') as stdout,(prep/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,cwd=str(R),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        initial=process_times(child._handle)
        dispatch={'schema':'labelcurve_replicate_preflight_observed_dispatch_v1','seed':config['seed'],'real_fraction':config['real_fraction'],'N_R':config['N_R'],'completed_curve4_reference':config['completed_curve4_reference'],'completed_curve4_preparation_seconds':config['completed_curve4_preparation_seconds'],'started_unix':started,'wrapper_pid':child.pid,'wrapper_create_time':initial['create_time'],'command':command,'helper_sha256':config['preflight_helper_sha256'],'request_sha256':__import__('hashlib').sha256(request_path.read_bytes()).hexdigest(),'full_process_limit_seconds':60,'cooperative_max_seconds':50}
        write_new(prep/'dispatch_wrapper.json',dispatch)
        print(json.dumps({'state':'STARTED','pid':child.pid,'full_process_deadline_unix':started+60}),flush=True)
        while child.poll() is None:
            try:
                parent=psutil.Process(child.pid)
                for p in [parent]+parent.children(recursive=True):
                    try:owned[(p.pid,p.create_time())]={'pid':p.pid,'create_time':p.create_time(),'cmdline':p.cmdline()}
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            except psutil.NoSuchProcess:pass
            now=time.time()
            if now>=started+60 and not stop_written:
                write_new(prep/'STOP_PREFLIGHT.json',{'reason':'approved_full_process_60_seconds_exhausted','at_unix':now});stop_written=True
            if now>=started+75:
                value={**dispatch,'state':'ATTENTION_STILL_EXITING','observed_seconds':now-started,'owned':list(owned.values()),'no_force_kill':True};write_new(prep/'lifetime_receipt.json',value);print(json.dumps(value),flush=True);return 1
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        times=process_times(child._handle);elapsed=times['exit_time']-started
    # helper ExitTime独立计账；Windows conhost退出可能稍晚，不把它当训练Python。
    wait_started=time.monotonic();initial_live=live_owned(owned);live=initial_live
    _,hosts=partition_live_owned(live)
    while hosts and time.monotonic()-wait_started<2:
        time.sleep(.05);live=live_owned(owned);_,hosts=partition_live_owned(live)
    cleanup_wait=time.monotonic()-wait_started
    training_live,hosts=partition_live_owned(live)
    admission_path=Path(config['group'])/'admission.json';admission=json.loads(admission_path.read_bytes()) if admission_path.exists() else {}
    success=child.returncode==0 and 0<elapsed<=60 and not live and not (R/'.research_gpu.lock').exists() and admission.get('status')=='PASS'
    value={**dispatch,'state':'VERIFIED_EXIT' if success else 'ATTENTION','actual_exit_unix':times['exit_time'],'conservative_full_process_seconds':elapsed,'precise_GetProcessTimes':True,'returncode':child.returncode,'owned':list(owned.values()),'owned_still_alive':live,'owned_training_or_unknown_still_alive':training_live,'owned_console_hosts_still_alive':hosts,'owned_alive_at_helper_exit':initial_live,'console_host_wait_seconds':cleanup_wait,'console_host_exact_exit_time_measured':False,'helper_exit_time_accounting_excludes_console_host_wait':True,'GPU_lock_retired':not (R/'.research_gpu.lock').exists(),'admission_status':admission.get('status'),'previous_gpu_preparation_seconds':config['previous_gpu_preparation_seconds'],'combined_GPU_preparation_seconds_upper':config['previous_gpu_preparation_seconds']+elapsed,'block_preparation_seconds_upper':config['previous_block_preparation_seconds']+elapsed,'max_seconds':60}
    write_new(prep/'runtime_identity.json',{'processes':list(owned.values())});write_new(prep/'lifetime_receipt.json',value)
    print(json.dumps({k:value[k] for k in ('state','conservative_full_process_seconds','returncode','admission_status','combined_GPU_preparation_seconds_upper')}),flush=True)
    return 0 if success else 1

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: label_curve_replicate_preflight_observer.py G:/.../prepare_dir')
    raise SystemExit(main(sys.argv[1]))
