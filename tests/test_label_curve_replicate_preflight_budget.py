"""Stage2十六项CPU预算与前四项真实小收据链检查；不运行CUDA/模型。"""
import ast,hashlib,importlib.util,inspect,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def load(name,rel):
 s=importlib.util.spec_from_file_location(name,R/rel);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
H=load('labelcurve_replicate_helper','tools/label_curve_replicate_preflight.py')
O=load('labelcurve_replicate_observer','tools/label_curve_replicate_preflight_observer.py')
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False),encoding='utf8')
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

class ReplicateBudgetTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.jobs=[];self.requests={};self.lives={};self.statuses={};self.controls={}
  for i,(label,ratio,n,strategy) in enumerate((('50',.5,1050,'R'),('50',.5,1050,'R+S'),('10',.1,210,'R'),('10',.1,210,'R+S'))):
   job='label'+label+('_R' if strategy=='R' else '_R_plus_S')+'_seed17';p=self.root/job;p.mkdir();out=self.root/('output_'+job);group=self.root/('control_'+job);statuspath=p/'status.json';controlpath=p/'control.json';lifepath=p/'lifetime_receipt.json';requestpath=p/'preflight_request.json'
   q={'seed':17,'real_fraction':ratio,'N_R':n,'strategy':strategy,'group':str(group)};status={'state':'COMPLETE','schema_version':'bbox_label_curve_v1','job_id':job,'update':1500,'exposures':{'R':24000,'S':0} if strategy=='R' else {'R':12000,'S':12000},'directory':str(out/job)};control={'state':'SEALED','scope':'control_snapshot_only','control_group':str(group),'output_group':str(out)}
   write(requestpath,q);life={'state':'VERIFIED_EXIT','returncode':0,'precise_GetProcessTimes':True,'conservative_full_process_seconds':40.+i/10,'owned_still_alive':[],'GPU_lock_retired':True,'admission_status':'PASS','request_sha256':ref(requestpath)['sha256']}
   write(statuspath,status);write(controlpath,control);write(lifepath,life)
   row={'job_id':job,'status_reference':ref(statuspath),'control_manifest_reference':ref(controlpath),'lifetime_reference':ref(lifepath)};self.jobs.append(row);self.requests[job]=(requestpath,q);self.lives[job]=(lifepath,life);self.statuses[job]=(statuspath,status);self.controls[job]=(controlpath,control)
  self.closed={'schema':'bbox_labelcurve4_closed_v1','state':'VERIFIED_COMPLETE','four_jobs':self.jobs,'failed_preparation_attempts':[],'measured_preparation_seconds':sum(x[1]['conservative_full_process_seconds'] for x in self.lives.values())};self.closedpath=self.root/'closed.json';write(self.closedpath,self.closed)
 def tearDown(self):self.tmp.cleanup()
 def request(self,**changes):
  completed=self.closed['measured_preparation_seconds'];value={'seed':29,'real_fraction':1.,'N_R':2100,'strategy':'R','max_seconds':50,'full_process_limit_seconds':60,'budget_policy':'label_curve16_960_v1','completed_curve4_preparation_seconds':completed,'completed_curve4_reference':ref(self.closedpath),'previous_block_preparation_seconds':0.,'previous_gpu_preparation_seconds':795.8882010710282+completed,'block_preparation_cap_seconds':960,'total_preparation_cap_seconds':1995.8882010710282,'full_process_deadline_unix':2000000000.};value.update(changes);return value
 def reject(self,q):
  with self.assertRaises(ValueError):H.validate_preparation_budget(q)
  with self.assertRaises(ValueError):O.validate_request(q)
 def refresh(self):write(self.closedpath,self.closed)
 def replace_row(self,kind,job,data):
  mapping={'status_reference':self.statuses,'control_manifest_reference':self.controls,'lifetime_reference':self.lives};p= mapping[kind][job][0];write(p,data)
  for row in self.jobs:
   if row['job_id']==job:row[kind]=ref(p)
  self.refresh()
 def add_failed_preparation(self):
  job='label50_R_seed17';code={f'm{i}.py':'a'*64 for i in range(42)}
  same={'schema_version':'bbox_label_curve_admission_v1','seed':17,'strategy':'R','real_fraction':.5,'N_R':1050,'full_N_R':2100,'updates':1500,'schedule_policy':'single_stage_lr001_warm100_cosine_v1','checkpoint_every':1000,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'effective_batch':16,'sampler_contract':{'R_per_update':16,'S_per_update':0},'input_cache':{'path':'D:/fixed_cache','manifest_sha256':'a'*64},'baseline_R_status_reference':{'path':'G:/fixed_R_status.json','sha256':'b'*64},'executed_code_sha256':code,'source_bindings':{'fixed_source':'same'},'subset_manifest_reference':{'path':'D:/fixed_N50.json','sha256':'c'*64},'ordered_subset_train_pool_sha256':'d'*64}
  passed_path=Path(self.jobs[0]['control_manifest_reference']['path']).parent/'admission.json';write(passed_path,{**same,'status':'PASS'})
  control={**self.controls[job][1],'files':[{'path':'admission.json','sha256':ref(passed_path)['sha256']}]};self.replace_row('control_manifest_reference',job,control)
  prep=self.root/'failed_N50_R';prep.mkdir();group=self.root/'failed_N50_R_control';group.mkdir();q={'seed':17,'strategy':'R','real_fraction':.5,'N_R':1050,'group':str(group),'max_seconds':50,'full_process_limit_seconds':60,'budget_policy':'label_curve4_240_v1'};write(prep/'preflight_request.json',q)
  admission={**same,'status':'FAIL_STOP','diagnostic_updates':3,'elapsed_seconds':52.266,'failure':{'type':'TimeoutError','message':'labelcurve finite mechanical preparation exhausted'}};write(group/'admission.json',admission)
  life={'state':'ATTENTION','returncode':1,'precise_GetProcessTimes':True,'conservative_full_process_seconds':56.02125358581543,'owned_still_alive':[],'GPU_lock_retired':True,'admission_status':'FAIL_STOP','request_sha256':ref(prep/'preflight_request.json')['sha256']};write(prep/'lifetime_receipt.json',life)
  row={'job_id':job,'lifetime_reference':ref(prep/'lifetime_receipt.json'),'admission_reference':ref(group/'admission.json')};self.closed['failed_preparation_attempts']=[row];self.closed['measured_preparation_seconds']+=life['conservative_full_process_seconds'];self.refresh()
  return row,admission,life,q
 def test_one_real_timeout_is_retained_and_counted(self):
  self.add_failed_preparation();q=self.request();a=H.validate_preparation_budget(q);b=O.validate_request(q);self.assertAlmostEqual(a['completed_curve4_preparation_seconds'],self.closed['measured_preparation_seconds']);self.assertEqual(a['completed_curve4_preparation_seconds'],b['completed_curve4_preparation_seconds']);self.assertEqual(a['failed_curve4_attempts'],1);self.assertEqual(a['failed_curve4_preparation_seconds'],56.02125358581543)
 def test_failed_preparation_cannot_be_omitted_or_unknown(self):
  self.add_failed_preparation();rows=list(self.closed['failed_preparation_attempts']);self.closed['failed_preparation_attempts']=[];self.refresh();self.reject(self.request())
  self.closed['failed_preparation_attempts']=rows*2;self.refresh();self.reject(self.request())
  self.closed['failed_preparation_attempts']=[{**rows[0],'job_id':'label10_R_seed17'}];self.refresh();self.reject(self.request())
 def test_failure_must_remain_timeout_not_numeric_or_changed_protocol(self):
  row,admission,life,q=self.add_failed_preparation();path=Path(row['admission_reference']['path'])
  for changes in ({'failure':{'type':'RuntimeError','message':'gradient mismatch'}},{'failure':{'type':'TimeoutError','message':'unknown timeout'}},{'seed':29},{'diagnostic_updates':2},{'status':'PASS'},{'numeric_mode':'amp'},{'source_bindings':{'fixed_source':'changed'}}):
   with self.subTest(changes=changes):
    write(path,{**admission,**changes});row['admission_reference']=ref(path);self.refresh();self.reject(self.request())
 def test_failed_lifetime_request_must_bind_original_admission(self):
  row,admission,life,q=self.add_failed_preparation();prep=Path(row['lifetime_reference']['path']).parent;write(prep/'preflight_request.json',{**q,'group':str(self.root/'other_group')});write(prep/'lifetime_receipt.json',{**life,'request_sha256':ref(prep/'preflight_request.json')['sha256']});row['lifetime_reference']=ref(prep/'lifetime_receipt.json');self.refresh();self.reject(self.request())
 def test_failed_lifetime_requires_closed_exact_exit_within60(self):
  row,admission,life,q=self.add_failed_preparation();path=Path(row['lifetime_reference']['path'])
  for changes in ({'state':'VERIFIED_EXIT'},{'returncode':0},{'conservative_full_process_seconds':61.},{'owned_still_alive':[{'pid':321}]},{'GPU_lock_retired':False},{'precise_GetProcessTimes':False}):
   with self.subTest(changes=changes):write(path,{**life,**changes});row['lifetime_reference']=ref(path);self.refresh();self.reject(self.request())
 def test_all_sixteen_exact_scopes(self):
  for seed in (29,43):
   for ratio,n in ((1.,2100),(.5,1050),(.25,525),(.1,210)):
    for strategy in ('R','R+S'):
     with self.subTest(seed=seed,ratio=ratio,strategy=strategy):
      q=self.request(seed=seed,real_fraction=ratio,N_R=n,strategy=strategy);a=H.validate_preparation_budget(q);b=O.validate_request(q);self.assertEqual((a['N_R'],b['real_fraction']),(n,ratio))
 def test_unapproved_seed_fraction_and_count_rejected(self):
  for d in ({'seed':17},{'seed':45},{'seed':True},{'real_fraction':.2,'N_R':420},{'real_fraction':.25,'N_R':2100},{'N_R':2100.}):
   with self.subTest(d=d):self.reject(self.request(**d))
 def test_history_cannot_reset_or_borrow_unused_first4(self):
  for d in ({'previous_gpu_preparation_seconds':795.8882010710282},{'previous_gpu_preparation_seconds':0.},{'block_preparation_cap_seconds':1200},{'total_preparation_cap_seconds':2000.},{'budget_policy':'label_curve4_240_v1'}):
   with self.subTest(d=d):self.reject(self.request(**d))
  q=self.request(previous_block_preparation_seconds=900.,previous_gpu_preparation_seconds=795.8882010710282+self.closed['measured_preparation_seconds']+900.);H.validate_preparation_budget(q);O.validate_request(q)
  self.reject(self.request(previous_block_preparation_seconds=900.001,previous_gpu_preparation_seconds=795.8882010710282+self.closed['measured_preparation_seconds']+900.001))
 def test_nonfinite_and_invalid_caps(self):
  for key in ('previous_gpu_preparation_seconds','previous_block_preparation_seconds','completed_curve4_preparation_seconds','total_preparation_cap_seconds'):
   for v in (math.nan,math.inf,-math.inf):
    with self.subTest(key=key,v=v):self.reject(self.request(**{key:v}))
  for d in ({'completed_curve4_preparation_seconds':241.},{'completed_curve4_preparation_seconds':-1.},{'full_process_limit_seconds':61},{'max_seconds':51},{'max_seconds':0}):self.reject(self.request(**d))
 def test_corrupted_closed_sha_and_schema_rejected(self):
  q=self.request();q['completed_curve4_reference']['sha256']='0'*64;self.reject(q)
  self.closed['schema']='old_schema';self.refresh();self.reject(self.request())
 def test_missing_or_duplicate_arm_rejected(self):
  saved=list(self.jobs);self.closed['four_jobs']=self.jobs[:3];self.refresh();self.reject(self.request())
  self.closed['four_jobs']=[saved[0],saved[0],saved[2],saved[3]];self.refresh();self.reject(self.request())
 def test_status_counts_and_control_seal_rejected(self):
  job=self.jobs[0]['job_id'];original=dict(self.statuses[job][1]);bad={**original,'exposures':{'R':12000,'S':0}};self.replace_row('status_reference',job,bad);self.reject(self.request());self.replace_row('status_reference',job,original)
  bad={**self.controls[job][1],'state':'RUNNING'};self.replace_row('control_manifest_reference',job,bad);self.reject(self.request())
 def test_lifetime_must_be_precise_full_exit_within60(self):
  job=self.jobs[0]['job_id'];original=self.lives[job][1]
  for d in ({'state':'ATTENTION'},{'returncode':True},{'returncode':1},{'precise_GetProcessTimes':False},{'conservative_full_process_seconds':60.001},{'owned_still_alive':[{'pid':123}]},{'GPU_lock_retired':False},{'admission_status':'FAIL_STOP'}):
   with self.subTest(d=d):self.replace_row('lifetime_reference',job,{**original,**d});self.reject(self.request())
 def test_sum_must_match_each_lifetime_and_declared(self):
  self.reject(self.request(completed_curve4_preparation_seconds=self.closed['measured_preparation_seconds']+.1,previous_gpu_preparation_seconds=795.8882010710282+self.closed['measured_preparation_seconds']+.1))
  self.closed['measured_preparation_seconds']+=1;self.refresh();self.reject(self.request())
 def test_lifetime_request_binds_same_arm_and_group(self):
  job=self.jobs[0]['job_id'];p,q=self.requests[job];write(p,{**q,'seed':29});life={**self.lives[job][1],'request_sha256':ref(p)['sha256']};self.replace_row('lifetime_reference',job,life);self.reject(self.request())
 def test_lifetime_cannot_be_reused_or_group_substituted(self):
  self.jobs[1]['lifetime_reference']=self.jobs[0]['lifetime_reference'];self.refresh();self.reject(self.request())
 def test_helper_absolute_deadline_required(self):
  for v in (0.,math.nan,math.inf):
   with self.assertRaises(ValueError):H.validate_preparation_budget(self.request(full_process_deadline_unix=v))
 def test_two_tools_share_exact_gate_code(self):
  for name in ('_number','_read_reference','_failed_curve4_seconds','validate_curve4_closed','_validate_scope_budget'):
   self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(H,name)))),ast.dump(ast.parse(inspect.getsource(getattr(O,name)))))
 def test_fresh_import_no_torch_and_first4_bytes_unchanged(self):
  expected={'tools/label_curve_preflight.py':'24dc443a48d835149d6db20266256de046386446c7219ae6db725858ccb60b1f','tools/label_curve_preflight_observer.py':'f1999a70d23b4c19c72dcbd49f77038e53a3c08779255744c386f324e2844b3f'}
  for rel,s in expected.items():self.assertEqual(hashlib.sha256((R/rel).read_bytes()).hexdigest(),s)
  code="import importlib.util,sys,json;from pathlib import Path;r=Path("+repr(str(R))+");\nfor name in ('label_curve_replicate_preflight','label_curve_replicate_preflight_observer'):\n s=importlib.util.spec_from_file_location(name,r/'tools'/str(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nprint(json.dumps({'torch_imported':'torch' in sys.modules}))"
  p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],capture_output=True,text=True,check=True);self.assertFalse(json.loads(p.stdout)['torch_imported'])
if __name__=='__main__':unittest.main()
