"""只读24组BBox末步证据，验收批量退出并生成开发集汇总；不训练/推理。"""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,statistics,os
import psutil
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch');B=Path('G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02');G=B.parent
METRICS=('AP50_95','AP50','AP75');PLAN={'EXPERIMENTS.md':'32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2','PAPER.md':'d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(p.read_bytes())
def checkref(v):
 p=Path(v['path']);assert p.is_file() and sha(p)==v['sha256'],p
 if 'bytes' in v:assert p.stat().st_size==v['bytes']
 return read(p)
def new(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def now():return datetime.now(timezone.utc).isoformat()
def main():
 for n,h in PLAN.items():assert sha(R/'docs/plan'/n)==h
 assert not (B/'completion_verification.json').exists() and not (B/'label_curve_development_comparison.json').exists() and not (B/'control').exists(),'拒绝覆盖闭合证据'
 result=read(B/'result.json');state=read(B/'state.json');assert result['state']==state['state']=='COMPLETE' and result['completed']==state['completed']==16 and len(result['jobs'])==16
 root=read(B/'root_dispatch.json');runtime=read(B/'runtime_identity.json');assert sha(B/'queue.json')==root['queue_reference']['sha256'] and sha(B/'launch.py')==root['launch_sha256'] and sha(B/'batch.py')==root['batch_source_sha256'];assert result['independent_match_evaluation_not_dispatched'] is True
 oldpath=G/'main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02/label_budget_comparison.json';old=read(oldpath);assert len(old['rows'])==8
 rows=[];known=[];controls=[];expected_jobs=[(seed,n,strategy) for seed in [29,43] for n in [2100,1050,525,210] for strategy in ['R','R+S']]
 assert [(x['job']['seed'],round(x['job']['real_fraction']*2100),x['job']['strategy']) for x in result['jobs']]==expected_jobs
 entries=[{'seed':17,'N_R':x['N_R'],'strategy':x['strategy'],'status_reference':x['status_reference'],'metrics':x['main_metrics'],'new_batch':False} for x in old['rows']]+[{'seed':x['job']['seed'],'N_R':round(x['job']['real_fraction']*2100),'strategy':x['job']['strategy'],'status_reference':x['status_reference'],'metrics':x['main_metrics'],'new_batch':True,'batch_entry':x} for x in result['jobs']]
 for index,e in enumerate(entries):
  s=checkref(e['status_reference']);leaf=Path(e['status_reference']['path']).parent;output=leaf.parent;group=R/'runs'/output.name;table=read(group/'job_table.json');job=next(x for x in table['jobs'] if x['job_id']==s['job_id']);exp={'R':24000,'S':0} if e['strategy']=='R' else {'R':12000,'S':12000}
  assert s['state']=='COMPLETE' and s['update']==1500 and s['exposures']==exp and job['seed']==e['seed'] and job['strategy']==e['strategy'] and job['N_R']==e['N_R'] and job['updates']==1500 and job['test_enabled'] is False
  assert job['input_size']==960 and job['numeric_mode']=='fp32' and job['effective_batch_size']==16 and job['microbatch_size']==8 and job['num_workers']==2
  terminal=read(leaf/'evaluation/step001500_final_final/summary.json');initial=read(leaf/'evaluation/step000000_final_PT/summary.json');assert terminal==s['terminal'] and terminal['images']==initial['images']==750 and terminal['update']==1500 and initial['update']==0 and terminal['evaluated_model']=='stage_EMA_at_recorded_update';assert all(terminal['main_metrics'][m]==e['metrics'][m] for m in METRICS)
  completion=read(output/'completion_verification.json');assert completion['state']=='VERIFIED_COMPLETE' and completion['GPU_lock_retired'] is True
  if completion.get('schema')=='labelcurve_completion_verification_v1':
   assert completion['last_step']=={'update':1500,'R_exposures':exp['R'],'S_exposures':exp['S']} and completion['owned_exit_verified'] is True and not completion['unknown_writers']
   assert {0,1000,1500}<={x['update'] for x in completion['checkpoint_receipts']}
   for ev in completion['evaluations']:assert checkref(ev['summary_reference'])==ev['summary']
  else:
   assert completion['global_updates']==1500 and completion['exposures']==exp and completion['checkpoint_receipt_bytes_verified'] is True and completion['terminal_step_bounded_tail_verified'] is True and not completion['unknown_training_writers']
  checkref(completion['status_reference']);manifest=checkref(completion['control_manifest_reference']);assert manifest['state']=='SEALED' and manifest['scope']=='control_snapshot_only'
  total=0
  if e['new_batch']:
   assert e['batch_entry']['completion_reference']['sha256']==sha(output/'completion_verification.json')
   for f in manifest['files']:
    rel=Path(f['path']);assert not rel.is_absolute() and '..' not in rel.parts and f['bytes']<=16*1024**2;p=output/'control'/rel;assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];total+=f['bytes']
   assert total==manifest['total_bytes']<=32*1024**2 and len(manifest['files'])==manifest['file_count']
   life=read(output/'batch_lifetime_receipt.json');assert life['state']=='VERIFIED_EXIT' and life['wrapper_returncode']==0 and life['precise_GetProcessTimes'] is True and not life['owned_still_alive'];known.extend(life['owned'])
  controls.append({'group':group.as_posix(),'manifest_reference':completion['control_manifest_reference'],'small_files_rehashed_this_root_review':e['new_batch'],'files':manifest['file_count'],'bytes':manifest['total_bytes']})
  cpfile=leaf/'checkpoints.jsonl';assert cpfile.stat().st_size<=2*1024**2;cps=[json.loads(x) for x in cpfile.read_text().splitlines()];assert {0,1000,1500}<={x['update'] for x in cps}
  for cp in cps:assert Path(cp['receipt']['path']).stat().st_size==cp['receipt']['bytes']
  finalcp=[x for x in cps if x['update']==1500];assert len(finalcp)==1
  rows.append({'seed':e['seed'],'real_fraction':e['N_R']/2100,'N_R':e['N_R'],'strategy':e['strategy'],'job_id':s['job_id'],'schema_version':s['schema_version'],'updates':1500,'exposures':exp,'metrics':terminal['main_metrics'],'status_reference':e['status_reference'],'job_table_reference':ref(group/'job_table.json'),'code_bindings_reference':ref(group/'code_bindings.json'),'completion_reference':ref(output/'completion_verification.json'),'control_manifest_reference':completion['control_manifest_reference'],'terminal_summary_reference':ref(leaf/'evaluation/step001500_final_final/summary.json'),'checkpoint_index_reference':ref(cpfile),'terminal_checkpoint_receipt':finalcp[0]['receipt']})
  if (index+1)%8==0:print(json.dumps({'verified_models':index+1,'total':24}),flush=True)
 assert len({(x['seed'],x['N_R'],x['strategy']) for x in rows})==24
 known.extend(root['processes']);alive=[]
 for row in known:
  try:
   p=psutil.Process(row['pid'])
   if abs(p.create_time()-row['create_time'])<.001:alive.append(row)
  except psutil.NoSuchProcess:pass
 assert not alive,'仍有原owned身份';assert not (B.parent/'label_curve_batch.lock').exists() and not (R/'.research_gpu.lock').exists()
 targets={os.path.normcase(str(Path(x['job_table_reference']['path']).parent/'launch.py')) for x in rows}|{os.path.normcase(str(B/'launch.py'))};unknown=[];selfproc=psutil.Process();anc={selfproc.pid}|{x.pid for x in selfproc.parents()}
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.pid in anc:continue
  cmd=p.info['cmdline'] or []
  if p.info['name']=='UnrealEditor.exe' or any(os.path.normcase(a) in targets for a in cmd):unknown.append({'pid':p.pid,'name':p.info['name'],'cmd':cmd})
 assert not unknown,'未知写者或UE活动'
 grouped=[];paired=[]
 for n in [210,525,1050,2100]:
  pair={st:sorted([x for x in rows if x['N_R']==n and x['strategy']==st],key=lambda x:x['seed']) for st in ['R','R+S']};assert all([x['seed'] for x in p]==[17,29,43] for p in pair.values())
  d={'N_R':n,'real_fraction':n/2100,'methods':{},'paired':{}}
  for st,p in pair.items():d['methods'][st]={m:{'values_percent':[x['metrics'][m]*100 for x in p],'mean_percent':statistics.mean(x['metrics'][m]*100 for x in p),'sample_std_pp':statistics.stdev(x['metrics'][m]*100 for x in p)} for m in METRICS}
  for m in METRICS:
   vals=[100*(s['metrics'][m]-r['metrics'][m]) for r,s in zip(pair['R'],pair['R+S'])];d['paired'][m]={'seed_order':[17,29,43],'RS_minus_R_pp':vals,'mean_pp':statistics.mean(vals),'sample_std_pp':statistics.stdev(vals),'positive_seeds':sum(x>0 for x in vals)}
  grouped.append(d)
 trade=[]
 for low,high in [(210,525),(525,1050),(1050,2100)]:
  vals=[100*(next(x for x in rows if (x['seed'],x['N_R'],x['strategy'])==(seed,low,'R+S'))['metrics']['AP50_95']-next(x for x in rows if (x['seed'],x['N_R'],x['strategy'])==(seed,high,'R'))['metrics']['AP50_95']) for seed in [17,29,43]];trade.append({'mixed_N_R':low,'R_N_R':high,'per_seed_difference_pp':vals,'mean_pp':statistics.mean(vals),'equivalence_or_label_savings_PASS':False})
 comparison={'schema':'bbox3seed4label_development_curve_v1','at_utc':now(),'state':'ALL24_TERMINAL_EMA_VERIFIED','source_batch_result_reference':ref(B/'result.json'),'seed17_comparison_reference':ref(oldpath),'rows':rows,'grouped':grouped,'smaller_pool_mixed_vs_larger_pool_R':trade,'numeric_units':'AP percent; differences and sample std in percentagepoints','std_definition':'sample std ddof1 across3 trainingseeds, notCI and not independentmatch variability','all_models_same_updates':1500,'R_exposures':24000,'RS_exposures':{'R':12000,'S':12000},'limitations':['One repeatedly developed750-frame game6 validation panel, not independentconfirmation.','Trainingseeds17/29/43 vary trainingrandomness, fixednested subset selectionseed17; no subsetselection uncertainty measured.','Mix halvesreal exposure within same totalbudget; effects belong complete recipe, not isolated causalmechanism.','No predefined equivalence/noninferiority margin or targetq; no automatic percentage annotation-savings claim.'],'independent_match_evaluation_executed':False,'Seg_status':'SHELVED_BY_LATEST_USER_NO_TRAINING_OR_GPU','code_reference':ref(Path(__file__))}
 new(B/'label_curve_development_comparison.json',comparison)
 with (B/'label_curve_development_results.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','real_fraction','N_R','strategy','updates','R_exposures','S_exposures',*METRICS,'status_path'])
  for x in sorted(rows,key=lambda x:(x['N_R'],x['seed'],x['strategy'])):w.writerow([x['seed'],x['real_fraction'],x['N_R'],x['strategy'],1500,x['exposures']['R'],x['exposures']['S'],*[100*x['metrics'][m] for m in METRICS],x['status_reference']['path']])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');xs=np.array([10,25,50,100])
 for st,color in [('R','#2864b4'),('R+S','#e77724')]:
  means=[d['methods'][st]['AP50_95']['mean_percent'] for d in grouped];std=[d['methods'][st]['AP50_95']['sample_std_pp'] for d in grouped];axes[0].errorbar(xs,means,yerr=std,fmt='o-',capsize=4,label=st,color=color)
 axes[0].set(xlabel='Real annotation pool (%)',ylabel='Terminal EMA COCO AP50:95 (%)',xticks=xs,title='1500 updates; mean +/- sample SD');axes[0].grid(alpha=.2);axes[0].legend()
 for j,seed in enumerate([17,29,43]):axes[1].scatter(xs+(j-1)*1.2,[d['paired']['AP50_95']['RS_minus_R_pp'][j] for d in grouped],s=27,label=f'seed {seed}')
 axes[1].errorbar(xs,[d['paired']['AP50_95']['mean_pp'] for d in grouped],yerr=[d['paired']['AP50_95']['sample_std_pp'] for d in grouped],fmt='kD-',capsize=4,label='paired mean +/- SD');axes[1].axhline(0,color='gray',linewidth=1);axes[1].set(xlabel='Real annotation pool (%)',ylabel='R+S minus R (percentage points)',xticks=xs,title='Same labeled pool within each pair');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
 fig.suptitle('YOLO11s / SoccerNet game6 development validation / seeds 17, 29, 43',fontsize=11);fig.savefig(B/'label_curve_development.png',dpi=180);fig.savefig(B/'label_curve_development.svg');plt.close(fig)
 completion={'schema':'bbox_label_curve_batch_root_closed_v1','state':'VERIFIED_COMPLETE','at_utc':now(),'batch_jobs':16,'all_models_with_seed17':24,'result_reference':ref(B/'result.json'),'comparison_reference':ref(B/'label_curve_development_comparison.json'),'CSV_reference':ref(B/'label_curve_development_results.csv'),'plot_reference':ref(B/'label_curve_development.png'),'per_model_control_verification':controls,'batch_and_known_owned_exit_verified':True,'owned_checked':len(known),'owned_still_alive':[],'GPU_lock_retired':True,'batch_lock_retired':True,'unknown_writers':[],'PT_NPZ_media_not_hashed':True,'independent_match_evaluation_executed':False,'Seg_shelved':True};new(B/'completion_verification.json',completion)
 control=B/'control';control.mkdir();names=['batch.py','launch.py','naming.py','queue.json','root_dispatch.json','dispatch.json','runtime_identity.json','prepare_derivation.json','startup_verification.json','state.json','result.json','completion_verification.json','label_curve_development_comparison.json','label_curve_development_results.csv']+[f'job_{j:02d}_completion.json' for j in range(16)];sealed=[]
 for name in names:
  p=B/name;raw=p.read_bytes();assert len(raw)<=16*1024**2
  with (control/name).open('xb') as f:f.write(raw)
  sealed.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 manifest={'schema':'bbox_batch_control_snapshot_v1','state':'SEALED','scope':'control_snapshot_only','at_utc':now(),'file_count':len(sealed),'total_bytes':sum(x['bytes'] for x in sealed),'files':sealed};assert manifest['total_bytes']<=32*1024**2;new(control/'CONTROL_MANIFEST.json',manifest)
 for n,h in PLAN.items():assert sha(R/'docs/plan'/n)==h
 print(json.dumps({'verified_models':24,'batch_owned_exited':True,'sealed_control_files':len(sealed),'sealed_control_bytes':manifest['total_bytes'],'comparison_reference':ref(B/'label_curve_development_comparison.json'),'batch_completion_reference':ref(B/'completion_verification.json'),'control_manifest_reference':ref(control/'CONTROL_MANIFEST.json'),'grouped':grouped,'tradeoffs':trade},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
