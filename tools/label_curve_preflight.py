"""仅root显式运行的50%/10%低标注R1500/mixed1500机械预检；导入不初始化CUDA。"""
import os,sys,json,time,gc,tempfile,traceback,importlib.util,math
from pathlib import Path
os.environ.update(YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch');D=Path(__file__).absolute().parent
A=R/'runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01'
sys.path.insert(0,str(A/'code'))
import futsalmot
def validate_preparation_budget(config):
 """仅新增50%/10%四项，每项60秒，历史账原样承接。"""
 def number(x):return type(x) in (int,float) and math.isfinite(x)
 if type(config.get('seed')) is not int or config['seed']!=17:raise ValueError('本四项仅seed17，不含后续复现预算')
 fraction=config.get('real_fraction');count=config.get('N_R')
 if type(fraction) not in (int,float) or fraction not in (.5,.1) or type(count) is not int or count!={.5:1050,.1:210}[fraction]:raise ValueError('只准50%/1050或10%/210明确真实图池')
 strategy=config.get('strategy');maximum=config.get('max_seconds');previous=config.get('previous_gpu_preparation_seconds');block=config.get('previous_block_preparation_seconds');deadline=config.get('full_process_deadline_unix')
 if strategy not in ('R','R+S') or config.get('budget_policy')!='label_curve4_240_v1':raise ValueError('不是本四项策略/许可')
 if not all(number(x) for x in (maximum,previous,block,deadline)) or not 0<maximum<=50 or block<0 or deadline<=0:raise ValueError('非法有限时间字段')
 if not math.isclose(previous,795.8882010710282+block,rel_tol=0,abs_tol=1e-6):raise ValueError('历史准备账未衔接')
 total=config.get('total_preparation_cap_seconds')
 if config.get('full_process_limit_seconds')!=60 or config.get('block_preparation_cap_seconds')!=240 or not number(total) or not math.isclose(total,1035.8882010710282,rel_tol=0,abs_tol=1e-6):raise ValueError('新块预算必须60/240/1035.8882010710282')
 if block+60>240 or previous+60>1035.8882010710282+1e-6:raise ValueError('准备预算不足')
 return {'policy':'label_curve4_240_v1','real_fraction':fraction,'N_R':count,'deadline_unix':deadline,'full_process_limit_seconds':60,'cooperative_window_seconds':maximum,'block_cap_seconds':240,'historical_preparation_seconds_upper':795.8882010710282,'total_cap_seconds':1035.8882010710282}


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
 result={'schema_version':'bbox_label_curve_admission_v1','status':'RUNNING','seed':seed,'strategy':strategy,'updates':1500,'schedule_policy':'single_stage_lr001_warm100_cosine_v1','checkpoint_every':1000,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'effective_batch':16,'sampler_contract':mix.sampler_contract(strategy),'input_cache':cache,'baseline_R_status_reference':ref,'executed_code_sha256':code,'cuda_memory_fraction':.8,'raw_upper_per_image_bytes':6650344,'previous_gpu_preparation_seconds':config['previous_gpu_preparation_seconds'],'max_seconds':config['max_seconds'],'preparation_budget':budget,'diagnostic_updates':0,'test_read':False,'validation_scored':False,'measurements':{}}
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
   bindings={**subset,'real_panel':full_panel['binding'],'synthetic_panel':synthetic['binding'],'public_model':{'path':str(runner.WEIGHTS/'yolo11s.pt'),'sha256':runner.sha(runner.WEIGHTS/'yolo11s.pt')},'requirements_sha256':runner.sha(R/'requirements.lock'),'source_code':code,'EXPERIMENTS_sha256':config['plan_sha256'],'PAPER_sha256':runner.PAPER_SHA,'input_cache':cache,'baseline_R_status_reference':ref,'scope':'labelcurve_mechanical_not_formal_scored'}
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
   runner.setup_cuda(result);runner.seed_all(seed);backend=mix.make_backend();samplers=mix.make_samplers(panel,synthetic,seed=seed);gen=torch.Generator(device='cpu').manual_seed(170017)
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