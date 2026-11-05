"""仅root显式调用的50%/10%低标注四项R1500/mixed1500预检观察器：保留进程句柄，完整退出计时，不初始化CUDA。"""
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


def validate_request(config):
    """派发前CPU门禁，禁止旧账/比例/数据量与单项上限被偷换。"""
    if type(config.get('seed')) is not int or config['seed']!=17:raise ValueError('本四项仅seed17，不含后续复现预算')
    fraction=config.get('real_fraction');count=config.get('N_R')
    if type(fraction) not in (int,float) or fraction not in (.5,.1) or type(count) is not int or count!={.5:1050,.1:210}[fraction]:raise ValueError('仅50%/1050或10%/210')
    if config.get('strategy') not in ('R','R+S') or config.get('budget_policy')!='label_curve4_240_v1' or config.get('max_seconds')!=50 or config.get('full_process_limit_seconds')!=60:raise ValueError('仅本四项各60秒/协作50秒')
    if not all(type(config.get(k)) in (int,float) and math.isfinite(config[k]) for k in ('previous_block_preparation_seconds','previous_gpu_preparation_seconds','total_preparation_cap_seconds')):raise ValueError('预算账必须有限数值')
    block=config['previous_block_preparation_seconds'];previous=config['previous_gpu_preparation_seconds']
    if block<0 or block+60>240 or abs(previous-(795.8882010710282+block))>1e-6 or config.get('block_preparation_cap_seconds')!=240 or abs(config['total_preparation_cap_seconds']-1035.8882010710282)>1e-6:raise ValueError('历史/新增预算账非法')
    return {'N_R':count,'real_fraction':fraction,'previous_block_preparation_seconds':block}


def main(prep):
    prep=Path(prep).absolute()
    if prep.parent!=G or not prep.name.startswith('bbox_labelcurve_prepare_') or prep.is_symlink() or prep.is_junction():raise ValueError('只准独立G预算1500准备目录')
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
        dispatch={'schema':'labelcurve_preflight_observed_dispatch_v1','started_unix':started,'wrapper_pid':child.pid,'wrapper_create_time':initial['create_time'],'command':command,'helper_sha256':config['preflight_helper_sha256'],'request_sha256':__import__('hashlib').sha256(request_path.read_bytes()).hexdigest(),'full_process_limit_seconds':60,'cooperative_max_seconds':50}
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
    if len(sys.argv)!=2:raise SystemExit('usage: label_curve_preflight_observer.py G:/.../prepare_dir')
    raise SystemExit(main(sys.argv[1]))
