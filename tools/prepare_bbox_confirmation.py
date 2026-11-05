"""显式冻结24末EMA及两场原test确认输入；不派发、不加载权重或CUDA。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from futsalmot import bbox_confirmation as c,bbox_confirmation_lifecycle as life
B=Path('G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02')

def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);assert p.stat().st_size<=32*1024**2
 return {'path':str(p.absolute()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(v):
 assert ref(v['path'])['sha256']==v['sha256'],v['path']
 return read(v['path'])
def main():
 day=datetime.now(timezone.utc).astimezone().strftime('%Y%m%d')
 inputs=R/'runs'/('bbox_confirmation_inputs_'+day+'_a01')
 control=R/'runs'/('bbox_confirmation_suite_'+day+'_a01')
 assert not inputs.exists() and not control.exists() and not (life.OUTPUT/control.name).exists(),'已有确认输入或派发，禁止覆盖/重放'
 plan=ref(R/'docs/plan/EXPERIMENTS.md');assert plan['sha256']=='32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2'
 root=read(B/'completion_verification.json');assert root['state']=='VERIFIED_COMPLETE' and root['all_models_with_seed17']==24 and root['batch_and_known_owned_exit_verified'] and root['GPU_lock_retired'] and root['batch_lock_retired'] and not root['unknown_writers']
 comparison=checked(root['comparison_reference']);rows=comparison['rows'];assert len(rows)==24
 models=[];done=[]
 for r in sorted(rows,key=lambda x:(x['seed'],-x['N_R'],x['strategy'])):
  for k in ('status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference','completion_reference'):checked(r[k]) if k!='checkpoint_index_reference' else None
  assert ref(r['checkpoint_index_reference']['path'])['sha256']==r['checkpoint_index_reference']['sha256']
  t=checked(r['job_table_reference']);cfg=t['jobs'][0];co=checked(r['completion_reference']);assert co['state']=='VERIFIED_COMPLETE' and co['GPU_lock_retired']
  model={'model_id':f"seed{r['seed']}_N{round(r['real_fraction']*100)}_{'R' if r['strategy']=='R' else 'RS'}",'seed':r['seed'],'real_fraction':r['real_fraction'],'strategy':r['strategy'],**{k:r[k] for k in ('status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference')},'checkpoint_receipt':r['terminal_checkpoint_receipt'],'expected_source_bindings':{**t['source_bindings'],'protocol':cfg},'original_completion_reference':r['completion_reference'],'original_code_bindings_reference':r['code_bindings_reference']}
  models.append(model);done.append({**{k:model[k] for k in ('model_id','status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference','checkpoint_receipt')},'owned_exit_verified':True,'GPU_lock_retired':True,'verification_reference':r['completion_reference'],'root_verification_reference':ref(B/'completion_verification.json')})
 assert {(x['seed'],x['real_fraction'],x['strategy']) for x in models}==c.EXPECTED_KEYS
 # 此前仅闭合训练小证据；先落不可覆盖模型注册，再读取原test GT小文件。
 resources=life.check_resources(remaining=48);inputs.mkdir()
 life.write_new(inputs/'models_registry.json',{'state':'FROZEN','models':models,'source_comparison_reference':root['comparison_reference'],'models24_verified_before_confirmation_GT':True})
 life.write_new(inputs/'completion.json',{'schema_version':c.COMPLETION_SCHEMA,'state':'COMPLETE','models':done,'root_verification_reference':ref(B/'completion_verification.json')})
 index_ref=models[0]['expected_source_bindings']['real_panel']['index'];index_path=Path(index_ref['path']);index=checked(index_ref);base=index_path.parent
 matches=[];trainuids=set();trainmatches=set()
 for seq in index['sequences']:
  if seq['split'] in ('train','val'):
   trainmatches.add(seq['match_uid']);trainuids.update(seq['detection_sample_uid_pattern'].format(frame_uid=seq['frame_uid_pattern'].format(frame_number=f)) for f in seq['detection_frames'])
 for uid in c.MATCHES:
  seqs=[s for s in index['sequences'] if s['match_uid']==uid];assert len(seqs)==5 and all(s['split']=='test' for s in seqs) and uid not in trainmatches
  uids=[];files=[]
  for seq in seqs:
   assert len(seq['detection_frames'])==150
   uids.extend(seq['detection_sample_uid_pattern'].format(frame_uid=seq['frame_uid_pattern'].format(frame_number=f)) for f in seq['detection_frames'])
   for name in ('gt','gameinfo','seqinfo'):
    v=ref(base/seq['sequence_path']/seq['files'][name]);assert name!='gt' or v['sha256']==seq['gt_sha256'];files.append(v)
  assert len(uids)==750 and not trainuids.intersection(uids)
  matches.append({'match_uid':uid,'ordered_uids':uids,'ordered_uid_sha256':c._pool_sha(uids),'files':files})
 spec={'schema_version':c.SCHEMA,'state':'FROZEN','inference_only':True,'execution':c.EXECUTION,'evaluation_count':48,'deadline_utc':life.GLOBAL_DEADLINE,'plan_reference':plan,'index_reference':ref(index_path),'evaluation_code_sha256':c.execution_code(),'models':models,'matches':matches,'match_observation_history':'game9 previously observed; game8 prior download/GT structure access; not first blind test','no_test_model_selection':True}
 c.validate_spec(spec);life.write_new(inputs/'confirmation_spec.json',spec)
 permit=c.authorize_confirmation(ref(inputs/'confirmation_spec.json'),completion_reference=ref(inputs/'completion.json'),plan_reference=plan)
 panels=[]
 for match in c.MATCHES:
  panel=c.confirmation_panel(permit,match);trueignore=sum(len(x['ignore']) for x in panel['val_ground_truth'].values());assert trueignore==0,'原test true-ignore当前标准COCO不支持；原GT保留，阻塞不派发'
  panels.append({'match_uid':match,'images':len(panel['val_uids']),'true_ignore':trueignore,'target_boxes':sum(len(x['positive_gt']) for x in panel['val_ground_truth'].values()),'non_target_role_boxes':sum(len(x['known_non_target_auxiliary_ignore']) for x in panel['val_ground_truth'].values())})
 life.write_new(inputs/'admission.json',{'state':'ADMITTED','resources':resources,'panels':panels,'models_registry_reference':ref(inputs/'models_registry.json'),'CPU_lifecycle_review':'32 tests OK root6.520s, fictitious fixtures; no GPU probe','inference_only':True,'fresh_training_preflight':False,'optimizer_updates':0})
 prepared=life.prepare_suite(control,spec_reference=ref(inputs/'confirmation_spec.json'),completion_reference=ref(inputs/'completion.json'),plan_reference=plan,observer_source=R/'tools/bbox_confirmation_observer.py')
 print(json.dumps({'prepared':prepared,'inputs':str(inputs),'panels':panels},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
