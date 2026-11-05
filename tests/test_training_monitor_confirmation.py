"""确认监视：只读小记录，partial不当完成，失败不补分数。"""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
sp=importlib.util.spec_from_file_location('monitor_confirmation_test',Path(__file__).parents[1]/'tools/training_monitor.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf8')
class ConfirmationMonitor(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);base=Path(self.temp.name);self.r=base/'D';self.g=base/'G';self.c=self.r/'runs/bbox_confirmation_suite_fixture';self.o=self.g/'runs'/self.c.name;self.o.mkdir(parents=True)
  models=[{'model_id':f'm{i}','seed':[17,29,43][i//8],'real_fraction':1.,'strategy':'R'} for i in range(24)];jobs=[{'job_id':f"{x['model_id']}__game{g}",'model_id':x['model_id'],'match_uid':f'match:soccernet-tracking:game:{g}'} for x in models for g in [8,9]];self.jobs=jobs
  spec=self.r/'runs/inputs/spec.json';put(spec,{'models':models});put(self.c/'suite_request.json',{'schema_version':'bbox_confirmation_suite_v1','jobs':jobs,'spec_reference':{'path':str(spec)}});put(self.r/'.bbox_confirmation_monitor.json',{'control_group':str(self.c),'output_group':str(self.o),'suite_sha256':hashlib.sha256((self.c/'suite_request.json').read_bytes()).hexdigest()})
 def snapshot(self):return m.confirmation_snapshot(self.r,self.g)
 def test_all48_pending(self):
  v=self.snapshot();self.assertEqual((v['state'],len(v['jobs']),v['completed'],v['images']),('READY',48,0,0));self.assertIsNone(v['eta_seconds'])
 def test_result_before_lifetime_is_finalizing(self):
  j=self.jobs[0]['job_id'];put(self.o/'jobs'/j/'result.json',{'state':'COMPLETE','summary':{'images':750,'main_metrics':{'AP50_95':.4}}});v=self.snapshot();self.assertEqual(v['completed'],0);self.assertEqual(v['jobs'][0]['state'],'FINALIZING')
 def test_completed_requires_exit_seal_and_counts_models_after_both(self):
  for row in self.jobs[:2]:
   j=row['job_id'];put(self.o/'jobs'/j/'result.json',{'state':'COMPLETE','summary':{'images':750,'main_metrics':{'AP50_95':.4}}});put(self.c/'jobs'/j/'lifetime_receipt.json',{'state':'VERIFIED_EXIT','owned_still_alive':[],'GPU_lock_retired':True,'full_process_seconds':300});put(self.o/'jobs'/j/'control/CONTROL_MANIFEST.json',{'state':'SEALED'})
  v=self.snapshot();self.assertEqual((v['completed'],v['completed_models'],v['images']),(2,1,1500));self.assertEqual(v['eta_seconds'],46*300)
 def test_error_and_path_escape_are_visible(self):
  j=self.jobs[0]['job_id'];put(self.o/'jobs'/j/'result.json',{'state':'FAIL_STOP','error':{'message':'original failure'}});v=self.snapshot();self.assertEqual(v['completed'],0);self.assertIsNone(v['jobs'][0]['metrics']);self.assertEqual(v['active']['error'],'original failure');p=self.r/'.bbox_confirmation_monitor.json';b=json.loads(p.read_bytes());b['output_group']=str(self.g/'outside');put(p,b);self.assertEqual(self.snapshot()['state'],'UNAVAILABLE')
if __name__=='__main__':unittest.main()
