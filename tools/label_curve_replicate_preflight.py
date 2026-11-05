"""仅root显式运行的29/43训练seed四标注档R1500/mixed1500复现机械预检；导入不初始化CUDA。"""
import os,sys,json,time,gc,tempfile,traceback,importlib.util,math
from pathlib import Path
os.environ.update(YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch');D=Path(__file__).absolute().parent
A=R/'runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01'
sys.path.insert(0,str(A/'code'))
import futsalmot
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

def validate_preparation_budget(config):
    """新stage2的16项，每项60秒；前四项按真实已用承接，不能借未用额度。"""
    scope=_validate_scope_budget(config);maximum=config.get('max_seconds');deadline=config.get('full_process_deadline_unix')
    if not _number(maximum) or not 0<maximum<=50 or not _number(deadline) or deadline<=0:raise ValueError('非法stage2协作窗口或绝对截止')
    closed=validate_curve4_closed(config)
    return {'policy':'label_curve16_960_v1',**scope,**closed,'deadline_unix':deadline,'full_process_limit_seconds':60,'cooperative_window_seconds':maximum,'block_cap_seconds':960,'historical_preparation_seconds_upper':795.8882010710282,'total_cap_seconds':1995.8882010710282}


def main():
 spec=importlib.util.spec_from_file_location('futsalmot.label_curve_bbox',R/'src/futsalmot/label_curve_bbox.py')
 mix=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mix;spec.loader.exec_module(mix)
 from futsalmot import runner,formal
 from futsalmot.runner_data import real_panel,make_loader,RequestPlanDataset
 from futsalmot.catalogues import synthetic_bbox_catalogue
 from futsalmot.input_cache import validate_ready_cache
 from futsalmot.records import save_checkpoint,load_checkpoint
 from futsalmot.preflight import _same
 import torch,psutil
 config=json.loads((D/'preflight_config.json').read_bytes());group=Path(config['group']);strategy=config['strategy'];budget=validate_preparation_budget(config);seed=config['seed'];fraction=budget['real_fraction'];expected_N=budget['N_R'];cache=formal.cache_specification(config['input_cache']);ref=config['baseline_R_status_reference'];code=mix.execution_code();started=time.monotonic();backend=None;iterator=None
 result={'schema_version':'bbox_label_curve_admission_v1','status':'RUNNING','seed':seed,'completed_curve4_reference':config['completed_curve4_reference'],'completed_curve4_preparation_seconds':config['completed_curve4_preparation_seconds'],'strategy':strategy,'updates':1500,'schedule_policy':'single_stage_lr001_warm100_cosine_v1','checkpoint_every':1000,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'effective_batch':16,'sampler_contract':mix.sampler_contract(strategy),'input_cache':cache,'baseline_R_status_reference':ref,'executed_code_sha256':code,'cuda_memory_fraction':.8,'raw_upper_per_image_bytes':6650344,'previous_gpu_preparation_seconds':config['previous_gpu_preparation_seconds'],'max_seconds':config['max_seconds'],'preparation_budget':budget,'diagnostic_updates':0,'test_read':False,'validation_scored':False,'measurements':{}}
 def check():
  if time.monotonic()-started>=config['max_seconds'] or time.time()>=budget['deadline_unix'] or (D/'STOP_PREFLIGHT.json').exists():raise TimeoutError('labelcurve finite mechanical preparation exhausted')
 def log(value):runner.append(D/'preflight_steps.jsonl',value)
 def cleanup():
  nonlocal backend,iterator
  runner._shutdown(iterator);iterator=None;backend=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
 try:
  with runner.gpu_lock(group):
   runner.production_closed();mix.check_baseline(ref);runner.check_disk(2**30);runner.check_archive_space(0)
   predecessor=config['closed_predecessor'];result['closed_predecessor']=predecessor
   for key in ('status_reference','control_manifest_reference'):
    reference=predecessor[key]
    if runner.sha(Path(reference['path']))!=reference['sha256']:raise ValueError('predecessor small receipt changed:'+key)
   prior_status=json.loads(Path(predecessor['status_reference']['path']).read_bytes());prior_control=json.loads(Path(predecessor['control_manifest_reference']['path']).read_bytes())
   if prior_status.get('state')!='COMPLETE' or prior_control.get('state')!='SEALED':raise ValueError('predecessor not COMPLETE/SEALED')
   result['predecessor_small_evidence_verified']=True
   if runner.sha(R/'docs/plan/EXPERIMENTS.md')!=config['plan_sha256']:raise ValueError('current plan changed')
   full_panel=real_panel(runner.REAL);subset_ref=mix.subset_reference(config['subset_manifest_reference']);panel=mix.filter_train_panel(full_panel,subset_ref);subset=mix.subset_binding(full_panel,subset_ref);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');ready=validate_ready_cache(cache,synthetic['binding']);check()
   if (len(full_panel['train_uids']),len(panel['train_uids']),len(panel['val_uids']),len(synthetic['ordered_uids']))!=(2100,expected_N,750,24149):raise ValueError('full/actual selected/val/synthetic data counts changed')
   if set(panel['train_catalogue'])!=set(panel['train_uids']) or panel['ordered_subset_train_pool_sha256']!=subset['ordered_subset_train_pool_sha256']:raise ValueError('actual training catalogue not selected declared N_R exactly')
   if subset['subset_N_R']!=expected_N:raise ValueError('manifest/subset binding does not match declared fraction/N_R')
   selected_real_uids=frozenset(panel['train_uids']);actual_real_uids=set()
   result.update(N_R=expected_N,full_N_R=2100,real_fraction=fraction,subset_manifest_reference=subset_ref,ordered_subset_train_pool_sha256=subset['ordered_subset_train_pool_sha256'])
   bindings={**subset,'real_panel':full_panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':{'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':runner.sha(runner.WEIGHTS/'yolo11s.pt')},'requirements_sha256':runner.sha(R/'requirements.lock'),'source_code':code,'EXPERIMENTS_sha256':config['plan_sha256'],'PAPER_sha256':runner.PAPER_SHA,'input_cache':cache,'baseline_R_status_reference':ref,'scope':'labelcurve_replicate_mechanical_not_formal_scored','completed_curve4_reference':config['completed_curve4_reference']}
   result['source_bindings']=bindings;result['input_cache_validation']=ready
   catalogue={**panel['train_catalogue'],**synthetic['catalogue']};
   if any(u in catalogue for u in set(full_panel['train_uids'])-selected_real_uids):raise ValueError('unselected real images entered training catalogue')
   samplers=mix.make_samplers(panel,synthetic,seed=seed);representatives=[samplers[k].reserve(1)['samples'][0] for k in ('R','S')]
   for live in samplers.values():live.cancel_pending()
   requests=mix.phase_requests(strategy,samplers,start=0,end=2)
   cached=mix.make_dataset(catalogue,representatives,{'input_cache':cache});direct=RequestPlanDataset(catalogue,representatives,view='train',synthetic_reader_options=formal.SYNTHETIC_READER_OPTIONS,real_reader_options=formal.REAL_READER_OPTIONS)
   compared=[]
   for i in range(2):
    a,b=cached[i],direct[i]
    if not _same(a,b,torch):raise RuntimeError('mixed representative cached/direct pixels labels geometry lineage mismatch')
    compared.append({'uid':representatives[i]['uid'],'domain':representatives[i]['domain'],'all_sample_fields_exact':True});check()
   result['measurements']['cached_direct_equivalence']=compared;del cached,direct,a,b
   runner.setup_cuda(result);runner.seed_all(seed);backend=mix.make_backend();samplers=mix.make_samplers(panel,synthetic,seed=seed);gen=torch.Generator(device='cpu').manual_seed(seed*10000+17)
   def start_loader(end,resume=False):
    nonlocal iterator
    req=mix.phase_requests(strategy,samplers,start=backend.stage_update,end=end);
    if any(row['domain']=='R' and row['uid'] not in selected_real_uids for row in req):raise ValueError('unselected R UID entered loader requests')
    dataset=mix.make_dataset(catalogue,req,{'input_cache':cache});loader=make_loader(dataset,batch_size=8,num_workers=2,generator=gen,pin_memory=True,allow_initialized_cuda=True);loader.timeout=30;iterator=formal.make_phase_iterator(loader,gen,iteration_already_started=resume);return loader
   def step(local):
    windows=mix.reserve_windows(strategy,samplers);begin=time.monotonic();batches=runner._microbatches(iterator,2);mix.validate_batches(strategy,batches,windows);
    for batch in batches:
     for lineage in batch['sample_lineage']:
      row=lineage['request']
      if row['domain']=='R':
       if row['uid'] not in selected_real_uids:raise ValueError('unselected real UID was consumed')
       actual_real_uids.add(row['uid'])
    metrics=backend.step_update(batches,update=local);counts=mix.commit_windows(strategy,samplers,windows);torch.cuda.synchronize();seconds=time.monotonic()-begin
    if counts!=mix.exposure_plan(strategy,local):raise ValueError('actual committed R/S exposures mismatch')
    result['diagnostic_updates']+=1;log({'update':local,'metrics':metrics,'exposures':counts,'microbatch_domains':[b['domains'] for b in batches],'seconds':seconds});check();return metrics,seconds
   def snapshot(local):return mix.snapshot(backend,strategy=strategy,samplers=samplers,source_bindings=bindings,committed_updates=local,extra_state={'loader_generator_state':gen.get_state().clone(),'loader_iteration_started':True,'run_scope':mix.SCHEMA})
   with tempfile.TemporaryDirectory(prefix='.labelcurve-probe-',dir=group) as temporary:
    loader=start_loader(2);out1,t1=step(1);state1=snapshot(1);r1=save_checkpoint(Path(temporary)/'step1.pt',state1);state1=load_checkpoint(r1,expected_bindings=bindings);check();out2,t2=step(2);gradient2=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in backend.model.parameters()];state2=snapshot(2);r2=save_checkpoint(Path(temporary)/'step2.pt',state2);runner._shutdown(iterator);iterator=None;del loader;cleanup();check()
    backend=mix.make_backend();samplers=mix.make_samplers(panel,synthetic,seed=seed);restored=mix.restore(backend,state1,strategy=strategy,samplers=samplers,expected_bindings=bindings,expected_update=1);gen=torch.Generator(device='cpu');gen.set_state(restored['extra_state']['loader_generator_state']);loader=start_loader(2,resume=restored['extra_state']['loader_iteration_started']);replay,t3=step(2);gradient_again=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in backend.model.parameters()];again=snapshot(2)
    equal={k:_same(state2[k],again[k],torch) for k in state2};equal['loss_and_metrics']=_same(out2,replay,torch);equal['actual_gradient_tensors']=_same(gradient2,gradient_again,torch)
    if not all(equal.values()):raise RuntimeError('mixed odd disk checkpoint next-step mismatch:'+repr([k for k,v in equal.items() if not v]))
    runner._shutdown(iterator);iterator=None;del loader;result['measurements']['exact_restore']=equal;result['measurements']['actual_selected_real_UIDs']=sorted(actual_real_uids);result['measurements']['selected_training_catalogue_N_R']=len(panel['train_catalogue']);result['measurements']['unselected_real_training_catalogue_entries']=0;result['measurements']['update_seconds']=[t1,t2,t3];result['checkpoint_bytes']=max(r1.bytes,r2.bytes);result['measurements']['temporary_checkpoints']=[{'path':x.path,'bytes':x.bytes,'sha256':x.sha256} for x in (r1,r2)]
    print(json.dumps({'temporary_diagnostic_models':2,'bytes':r1.bytes+r2.bytes,'receipts':result['measurements']['temporary_checkpoints']}),flush=True)
   result['measurements']['temporary_models_removed']=all(not Path(row['path']).exists() for row in result['measurements']['temporary_checkpoints'])
   free,total=torch.cuda.mem_get_info();result['measurements']['memory']={'free_bytes':free,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'total_bytes':total}
   if min(free,total-torch.cuda.max_memory_reserved())<2**30:raise RuntimeError('GPU reserve below1GiB')
   check()
   if mix.execution_code()!=code:raise RuntimeError('source changed during preflight')
   result.update(status='PASS',elapsed_seconds=time.monotonic()-started);result['combined_GPU_preparation_seconds']=config['previous_gpu_preparation_seconds']+result['elapsed_seconds'];runner.write_once(group/'admission.json',result)
  print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'checkpoint_bytes':result['checkpoint_bytes'],'exact_restore':result['measurements']['exact_restore']}),flush=True)
 except BaseException as exc:
  result.update(status='FAIL_STOP',elapsed_seconds=time.monotonic()-started,failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
  if not (group/'admission.json').exists():runner.write_once(group/'admission.json',result)
  raise
 finally:cleanup()
if __name__=='__main__':main()