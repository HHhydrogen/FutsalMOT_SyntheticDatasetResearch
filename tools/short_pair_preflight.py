"""根显式运行的短SR/RR有界机械预检；导入不初始化CUDA，不自行派发。"""
import os,sys,json,time,gc,tempfile,traceback,importlib.util,math
from pathlib import Path
os.environ.update(YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch');D=Path(__file__).absolute().parent
A=R/'runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a01'
sys.path.insert(0,str(A/'code'))
import futsalmot
spec=importlib.util.spec_from_file_location('futsalmot.short_pair',R/'src/futsalmot/short_pair.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)


def validate_preparation_budget(config):
    """生命周期预算门禁；不初始化CUDA，不改训练数值。"""
    strategy=config.get('strategy');maximum=config.get('max_seconds');previous=config.get('previous_gpu_preparation_seconds')
    def number(x):return type(x) in (int,float) and math.isfinite(x)
    if strategy not in ('S->R','R->R') or not number(maximum) or not 0<maximum<=60 or not number(previous) or previous<0:
        raise ValueError('非法短配对策略/有限预算')
    policy=config.get('budget_policy','original_pair_120_v1')
    if policy=='original_pair_120_v1':
        if previous+maximum>600:raise ValueError('原累计600秒预算不足')
        return {'policy':policy,'global_cap_seconds':600,'deadline_unix':None,'full_process_limit_seconds':None}
    if policy!='RR_additional_60_v1' or strategy!='R->R':raise ValueError('新增许可仅RR独立一次')
    pair=config.get('previous_pair_preparation_seconds');deadline=config.get('full_process_deadline_unix')
    if not number(pair) or pair<0 or not math.isclose(previous,475.376+pair,abs_tol=1e-6):raise ValueError('已用总量/配对预算必须衔接原账')
    if config.get('total_preparation_cap_seconds')!=660 or config.get('pair_preparation_cap_seconds')!=180 or config.get('full_process_limit_seconds')!=60:
        raise ValueError('新增预算必须精确660/180/60秒')
    if previous+60>660 or pair+60>180 or maximum>50:raise ValueError('RR全进程或协作窗口无余量')
    if not number(deadline) or deadline<=0:raise ValueError('缺启动时冻结的完整进程deadline')
    return {'policy':policy,'global_cap_seconds':660,'pair_cap_seconds':180,'deadline_unix':deadline,'full_process_limit_seconds':60,'cooperative_window_seconds':maximum}


def main():
 from futsalmot import runner,formal
 from futsalmot.runner_data import real_panel,make_loader,RequestPlanDataset
 from futsalmot.catalogues import synthetic_bbox_catalogue
 from futsalmot.input_cache import validate_ready_cache
 from futsalmot.records import save_checkpoint,load_checkpoint
 from futsalmot.preflight import _same
 import torch
 config=json.loads((D/'preflight_config.json').read_bytes());group=Path(config['group']);strategy=config['strategy'];cache=config['input_cache'];ref=config['baseline_R_status_reference'];code=s.execution_code();maximum=config['max_seconds'];previous=config['previous_gpu_preparation_seconds'];started=time.monotonic();backend=None;iterator=None
 budget=validate_preparation_budget(config)
 result={'schema_version':'bbox_short_pair_admission_v1','status':'RUNNING','strategy':strategy,'schedule_policy':s.SCHEDULE_POLICY,'checkpoint_every':1000,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'effective_batch':16,'input_cache':cache,'baseline_R_status_reference':ref,'executed_code_sha256':code,'cuda_memory_fraction':.8,'raw_upper_per_image_bytes':6650344,'previous_gpu_preparation_seconds':previous,'max_seconds':maximum,'diagnostic_updates':0,'test_read':False,'validation_scored':False,'measurements':{}}
 result['preparation_budget']=budget
 def check():
  if time.monotonic()-started>=maximum or (budget['deadline_unix'] is not None and time.time()>=budget['deadline_unix']) or (D/'STOP_PREFLIGHT.json').exists():raise TimeoutError('short pair mechanical preparation budget exhausted')
 def log(value):runner.append(D/'preflight_steps.jsonl',value)
 def cleanup():
  nonlocal backend,iterator
  runner._shutdown(iterator);iterator=None;backend=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
 def public_backend():
  return s.ShortPairBackend.from_public(runner.WEIGHTS,model_size='s',stage_updates=1500,final_stage=False,microbatch_size=8,numeric_mode='fp32',device='cuda:0',input_size=960,allow_initialized_cuda=True)
 try:
  with runner.gpu_lock(group):
   runner.production_closed();s._check_baseline(ref);runner.check_disk(2**30);runner.check_archive_space(0)
   predecessor=config['closed_predecessor'];result['closed_predecessor']=predecessor
   for key in ('status_reference','control_manifest_reference'):
    reference=predecessor[key]
    if runner.sha(Path(reference['path']))!=reference['sha256']:raise ValueError('predecessor small receipt changed:'+key)
   prior_status=json.loads(Path(predecessor['status_reference']['path']).read_bytes());prior_control=json.loads(Path(predecessor['control_manifest_reference']['path']).read_bytes())
   if prior_status.get('state')!='COMPLETE' or prior_control.get('state')!='SEALED':raise ValueError('前项未完成/小control未SEALED')
   result['predecessor_small_evidence_verified']=True
   if runner.sha(R/'docs/plan/EXPERIMENTS.md')!=config['plan_sha256']:raise ValueError('current plan changed')
   panel=real_panel(runner.REAL);synthetic=synthetic_bbox_catalogue(formal.SYNTHETIC,split='train');ready=validate_ready_cache(cache,synthetic['binding']);check()
   if (len(panel['train_uids']),len(panel['val_uids']),len(synthetic['ordered_uids']))!=(2100,750,24149):raise ValueError('data counts changed')
   bindings={'real_panel':panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':{'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':runner.sha(runner.WEIGHTS/'yolo11s.pt')},'requirements_sha256':runner.sha(R/'requirements.lock'),'source_code':code,'EXPERIMENTS_sha256':config['plan_sha256'],'PAPER_sha256':runner.PAPER_SHA,'input_cache':cache,'baseline_R_status_reference':ref,'scope':'short_pair_mechanical_not_formal_scored'}
   result['source_bindings']=bindings;result['input_cache_validation']=ready
   catalogue={**panel['train_catalogue'],**synthetic['catalogue']};samplers=s.make_samplers(panel,synthetic,17)
   # 每次实际核一张R与一张S；不是重复全4K或全数组hash。
   representatives=[samplers[k].reserve(1)['samples'][0] for k in ('R','S')]
   for live in samplers.values():live.cancel_pending()
   cached=s.make_training_dataset(catalogue,representatives,{'input_cache':cache});direct=RequestPlanDataset(catalogue,representatives,view='train',synthetic_reader_options=formal.SYNTHETIC_READER_OPTIONS,real_reader_options=formal.REAL_READER_OPTIONS)
   compared=[]
   for i in range(2):
    x,y=cached[i],direct[i]
    if not _same(x,y,torch):raise RuntimeError('representative cached/direct pixels labels geometry lineage mismatch')
    compared.append({'uid':representatives[i]['uid'],'domain':representatives[i]['domain'],'all_sample_fields_exact':True});check()
   result['measurements']['cached_direct_equivalence']=compared;del cached,direct,x,y
   runner.setup_cuda(result);runner.seed_all(17);backend=public_backend();samplers=s.make_samplers(panel,synthetic,17);gen=torch.Generator(device='cpu').manual_seed(170017);stream='S' if strategy=='S->R' else 'R_extra'
   def start_loader(end,resume=False):
    nonlocal iterator
    requests=s.phase_requests(strategy,samplers,start=backend.stage_update,end=end);dataset=s.make_training_dataset(catalogue,requests,{'input_cache':cache});loader=make_loader(dataset,batch_size=8,num_workers=2,generator=gen,pin_memory=True,allow_initialized_cuda=True);loader.timeout=20;iterator=formal.make_phase_iterator(loader,gen,iteration_already_started=resume);return loader
   def step(local):
    window=samplers[stream].reserve(16);begin=time.monotonic();batches=runner._microbatches(iterator,2)
    if [v['request'] for batch in batches for v in batch['sample_lineage']]!=window['samples']:raise ValueError('实际派发与sampler提交窗口不符')
    metrics=backend.step_update(batches,update=local);samplers[stream].commit(window['token']);torch.cuda.synchronize();seconds=time.monotonic()-begin
    if s.exposure_counts(samplers)!=s.exposure_plan(strategy,local):raise ValueError('diagnostic first-phase exposures mismatch')
    result['diagnostic_updates']+=1;log({'phase':'first','update':local,'metrics':metrics,'exposures':s.exposure_counts(samplers),'seconds':seconds});check();return metrics,seconds
   def snapshot(local):
    return backend.snapshot(samplers=samplers,source_bindings=bindings,committed_updates=local,cumulative_exposures=s.exposure_counts(samplers),phase='first',extra_state={'loader_generator_state':gen.get_state().clone(),'loader_iteration_started':True,'run_scope':s.SCHEMA})
   with tempfile.TemporaryDirectory(prefix='.short-pair-probe-',dir=group) as temporary:
    loader=start_loader(2);out1,t1=step(1);state1=snapshot(1);receipt1=save_checkpoint(Path(temporary)/'step1.pt',state1);state1=load_checkpoint(receipt1,expected_bindings=bindings);check();out2,t2=step(2);gradients2=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in backend.model.parameters()];state2=snapshot(2);receipt2=save_checkpoint(Path(temporary)/'step2.pt',state2);runner._shutdown(iterator);iterator=None;del loader;cleanup();check()
    backend=public_backend();samplers=s.make_samplers(panel,synthetic,17);restored=backend.restore_bundle(state1,samplers=samplers,expected_bindings=bindings);gen=torch.Generator(device='cpu');gen.set_state(restored['extra_state']['loader_generator_state']);loader=start_loader(2,resume=restored['extra_state']['loader_iteration_started']);replay,t3=step(2);again_gradients=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in backend.model.parameters()];again=snapshot(2)
    exact={k:_same(state2[k],again[k],torch) for k in state2};exact['loss_and_metrics']=_same(out2,replay,torch);exact['actual_gradient_tensors']=_same(gradients2,again_gradients,torch)
    if not all(exact.values()):raise RuntimeError('disk checkpoint next-step mismatch:'+repr([k for k,v in exact.items() if not v]))
    runner._shutdown(iterator);iterator=None;del loader;result['measurements']['exact_restore']=exact
    # 机械转换只检验真实raw承接、新SGD/EMA与首个真实R更新，不冒称已跑满1500。
    # 完整1500边界的production guard由CPU检查与实际run独立保证，不伪造计数。
    old_backend=backend;backend=s.ShortPairBackend(old_backend.network,stage_updates=5063,final_stage=True,microbatch_size=8,numeric_mode='fp32',device='cuda:0',input_size=960,allow_initialized_cuda=True)
    raw=old_backend.model.state_dict();transition={'raw_preserved':all(torch.equal(raw[k],backend.model.state_dict()[k]) for k in raw),'new_EMA_from_raw':all(torch.equal(raw[k],backend.ema_state['state_dict'][k]) for k in raw),'SGD_empty':len(backend.optimizer.state)==0,'EMA_updates_zero':backend.ema_state['updates']==0,'stage_updates':backend.stage_updates,'schedule_policy':backend._config()['schedule_policy'],'diagnostic_first_updates':2,'production_first_horizon':1500,'full_first_stage_execution_claimed':False}
    if not all(transition[k] for k in ('raw_preserved','new_EMA_from_raw','SGD_empty','EMA_updates_zero')):raise RuntimeError('raw/newSGD/EMA rebuild mismatch')
    del old_backend,raw
    requests=s.phase_requests(strategy,samplers,start=1500,end=1501);dataset=s.make_training_dataset(catalogue,requests,{'input_cache':cache});loader=make_loader(dataset,batch_size=8,num_workers=2,generator=gen,pin_memory=True,allow_initialized_cuda=True);loader.timeout=20;iterator=formal.make_phase_iterator(loader,gen,iteration_already_started=False)
    w=samplers['R'].reserve(16);begin=time.monotonic();data=runner._microbatches(iterator,2)
    if [v['request'] for b in data for v in b['sample_lineage']]!=w['samples']:raise RuntimeError('R-rebuild request mismatch')
    final_first=backend.step_update(data,update=1);samplers['R'].commit(w['token']);torch.cuda.synchronize();t4=time.monotonic()-begin;result['diagnostic_updates']+=1
    if final_first['lr']!=.00001 or final_first['momentum']!=.80137:raise RuntimeError('actual final LR/momentum not100warmup')
    log({'phase':'diagnostic_final_after_two_first_updates','update':1,'metrics':final_first,'actual_exposures':s.exposure_counts(samplers),'seconds':t4,'not_formal_score':True});check()
    runner._shutdown(iterator);iterator=None;del loader,dataset,data
    result['measurements']['phase_rebuild']=transition;result['measurements']['update_seconds']=[t1,t2,t3,t4];result['checkpoint_bytes']=max(receipt1.bytes,receipt2.bytes);result['measurements']['temporary_checkpoints']=[{'path':r.path,'bytes':r.bytes,'sha256':r.sha256} for r in (receipt1,receipt2)]
    print(json.dumps({'temporary_diagnostic_models':2,'bytes':receipt1.bytes+receipt2.bytes,'receipts':result['measurements']['temporary_checkpoints']}),flush=True)
   result['measurements']['temporary_models_removed']=all(not Path(r['path']).exists() for r in result['measurements']['temporary_checkpoints'])
   free,total=torch.cuda.mem_get_info();result['measurements']['memory']={'free_bytes':free,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'total_bytes':total}
   if min(free,total-torch.cuda.max_memory_reserved())<2**30:raise RuntimeError('GPU reserve below1GiB')
   check()
   if s.execution_code()!=code:raise RuntimeError('source changed during preflight')
   result.update(status='PASS',elapsed_seconds=time.monotonic()-started);result['combined_GPU_preparation_seconds']=previous+result['elapsed_seconds'];runner.write_once(group/'admission.json',result)
  print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'checkpoint_bytes':result['checkpoint_bytes'],'exact_restore':result['measurements']['exact_restore']}),flush=True)
 except BaseException as exc:
  result.update(status='FAIL_STOP',elapsed_seconds=time.monotonic()-started,failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
  if not (group/'admission.json').exists():runner.write_once(group/'admission.json',result)
  raise
 finally:cleanup()

if __name__=='__main__':main()