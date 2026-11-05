"""确认外围的CPU mock合同；不spawn、不读真实test/权重、不初始化CUDA。"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import json,os,time,unittest
import torch
from futsalmot import bbox_confirmation_lifecycle as m,bbox_confirmation as confirmation,runner

class Permit:
    def spec(self):return {'models':[{'model_id':f'm{i:02d}'} for i in range(24)],'matches':[{'match_uid':x} for x in confirmation.MATCHES]}
    def model(self,key):
        if key not in {f'm{i:02d}' for i in range(24)}:raise ValueError('unknownmodel')
        return {'model_id':key}
    def match(self,key):
        if key not in confirmation.MATCHES:raise ValueError('unknownmatch')
        return {'match_uid':key}

class FakeChild:
    pid=454545
    def __init__(self,returncode=0):self.returncode=returncode;self._handle=object()
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode
    def kill(self):raise AssertionError('不得force kill')
    def terminate(self):raise AssertionError('不得terminate')


def suite_fixture(root):
    research=root/'D';output=root/'G/runs';(research/'runs').mkdir(parents=True);output.mkdir(parents=True)
    control=research/'runs/bbox_confirmation_fixture';out=output/control.name;control.mkdir();out.mkdir();code=control/'code/futsalmot';code.mkdir(parents=True)
    hashes={}
    for i in range(43):
        p=code/f'module{i:02d}.py';p.write_text('# CPU fixture\n',encoding='utf8');hashes[p.name]=m.sha(p)
    observer=control/'observe.py';observer.write_text('# CPU fixture not executable\n',encoding='utf8')
    jobs=[{'model_id':f'm{i:02d}','match_uid':match,'job_id':m.safe_job_id(f'm{i:02d}',match)} for i in range(24) for match in confirmation.MATCHES]
    dummy=root/'ref.json';m.write_new(dummy,Permit().spec());ref=m.reference(dummy)
    suite={'schema_version':m.SCHEMA,'state':'FROZEN','control_group':str(control),'output_group':str(out),'spec_reference':ref,'completion_reference':ref,'plan_reference':ref,'jobs':jobs,'source_code_sha256':hashes,'observer_sha256':m.sha(observer),'resources':m.resource_bound(48),'max_seconds_per_job':900,'global_deadline_utc':m.GLOBAL_DEADLINE,'inference_only':True,'optimizer_updates':0,'automatic_resume':False,'automatic_retry':False,'allowed_evaluations':48,'research_root':str(research),'output_base':str(output)}
    m.write_new(control/'suite_request.json',suite);m.write_new(control/'code_bindings.json',hashes);return control,out,suite


class ConfirmationLifecycleCPU(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(prefix='confirmation-lifecycle-CPU-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.control,self.out,self.suite=suite_fixture(self.root);self.job_id=self.suite['jobs'][0]['job_id']
        self.guards=[patch.object(torch.cuda,name,side_effect=AssertionError('真实GPU禁止')) for name in ('init','_lazy_init','is_available')]
        for guard in self.guards:guard.start();self.addCleanup(guard.stop)
    def prepare(self):
        with patch.object(confirmation,'authorize_confirmation',return_value=Permit()),patch.object(m,'check_resources',return_value=m.resource_bound(48)):
            return m.prepare_job(self.control,self.job_id)
    def runtime_and_dispatch(self):
        self.prepare();job=self.control/'jobs'/self.job_id;start=time.time();runtime={'schema_version':'bbox_confirmation_runtime_v1','dispatch_started_unix':start,'deadline_unix':min(start+900,m.deadline_unix()),'max_seconds':900};m.write_new(job/'runtime_limit.json',runtime);m.write_new(job/'dispatch_wrapper.json',{'pid':os.getpid(),'create_time':0.,'observer_sha256':self.suite['observer_sha256'],'started_unix':start,'deadline_unix':runtime['deadline_unix'],'request_reference':m.reference(job/'job_request.json'),'runtime_reference':m.reference(job/'runtime_limit.json')});return job,runtime
    def run_mocked(self,*,true_ignore=False,evaluate_error=None,model_error=None,dirty_workers=False,dispatch_delay=False):
        job,runtime=self.runtime_and_dispatch();events=[]
        dispatch_path=job/'dispatch_wrapper.json';dispatch_bytes=dispatch_path.read_bytes()
        if dispatch_delay:dispatch_path.unlink()
        def delayed_parent_write(seconds):
            events.append('parent_dispatch_ready');dispatch_path.write_bytes(dispatch_bytes)
        class Lock:
            def __init__(self,*a):self.cleanup={'GPU_lock_retired':not dirty_workers,'owned_children_still_alive':[{'pid':4}] if dirty_workers else []}
            def __enter__(self):events.append('lock');return self
            def __exit__(self,*a):events.append('cleanup');return False
        def auth(*args,**kwargs):events.append('authorize24');return Permit()
        backend=SimpleNamespace(confirmation_checkpoint_receipt={'selected':'EMA'})
        def load(*args):
            events.append('loadEMA')
            if model_error:raise model_error
            return backend
        def evaluate(*args,**kwargs):
            events.append('evaluate');self.assertEqual(kwargs['hard_stop_unix'],runtime['deadline_unix'])
            if evaluate_error:raise evaluate_error
            summary={'images':750,'update':1500,'evaluated_model':'stage_EMA_at_recorded_update','main_metrics':{'AP50_95':.5}};req=json.loads((job/'job_request.json').read_bytes());m.write_new(Path(req['output_directory'])/'evaluation'/req['tag']/'summary.json',summary);return summary
        with ExitStack() as stack:
            if dispatch_delay:stack.enter_context(patch.object(m.time,'sleep',side_effect=delayed_parent_write))
            for obj,name,value in [(runner,'RESEARCH',Path(self.suite['research_root'])),(runner,'ARCHIVE_ROOT',Path(self.suite['output_base']).parent),(m,'verify_dispatch_identity',lambda *a:None),(confirmation,'authorize_confirmation',auth),(m,'check_resources',lambda **kw:m.resource_bound(48)),(runner,'production_closed',lambda:None),(m,'ConfirmGPULock',Lock),(runner,'setup_cuda',lambda *a:events.append('CUDAsetup_mock')),(m,'verified_ema_backend',load),(confirmation,'confirmation_panel',lambda *a:{'val_ground_truth':{'fixture':{'ignore':[[0,0,1,1]] if true_ignore else []}}}),(confirmation,'make_confirmation_evaluator',lambda *a:{'mechanical_binding':{'inverse_AST_exact':True}}),(m,'scoped_evaluator',lambda *a:evaluate),(torch.cuda,'mem_get_info',lambda:(2*1024**3,4*1024**3)),(torch.cuda,'empty_cache',lambda:None),(torch.cuda,'max_memory_reserved',lambda:0),(torch.cuda,'max_memory_allocated',lambda:0)]:stack.enter_context(patch.object(obj,name,value))
            code=m.run_job(self.control,self.job_id)
        return code,json.loads((job/'result.json').read_bytes()),events
    def test_duplicate_model_match_with_unique_job_id_is_rejected(self):
        suite=deepcopy(self.suite);suite['jobs'][1]={**suite['jobs'][0],'job_id':'different_unique_token'}
        (self.control/'suite_request.json').write_text(json.dumps(suite),encoding='utf8')
        with self.assertRaises(ValueError):m.load_suite(self.control)
    def test_job_request_must_match_frozen_row_tag_and_resource(self):
        self.prepare();p=self.control/'jobs'/self.job_id/'job_request.json';original=json.loads(p.read_bytes())
        for change in ({'model_id':'m01'},{'match_uid':confirmation.MATCHES[1]},{'tag':'different_tag'},{'resources':{**original['resources'],'remaining_model_matches':47}}):
            with self.subTest(change=change):
                p.write_text(json.dumps({**original,**change}),encoding='utf8')
                with patch.object(m.subprocess,'Popen',side_effect=AssertionError('invalid request must not spawn')):
                    with self.assertRaises(ValueError):m.observe_job(self.control,self.job_id)
        p.write_text(json.dumps(original),encoding='utf8')
    def test_delayed_parent_dispatch_is_waited_before_any_authorized_IO(self):
        code,result,events=self.run_mocked(dispatch_delay=True)
        self.assertEqual(code,0);self.assertEqual(result['state'],'COMPLETE');self.assertLess(events.index('parent_dispatch_ready'),events.index('authorize24'));self.assertLess(events.index('parent_dispatch_ready'),events.index('CUDAsetup_mock'))
    def test_reordered_jobs_and_changed_spec_reference_are_rejected(self):
        suite=deepcopy(self.suite);suite['jobs'][0],suite['jobs'][1]=suite['jobs'][1],suite['jobs'][0]
        p=self.control/'suite_request.json';p.write_text(json.dumps(suite),encoding='utf8')
        with self.assertRaises(ValueError):m.load_suite(self.control)
        p.write_text(json.dumps(self.suite),encoding='utf8');Path(self.suite['spec_reference']['path']).write_text('{}',encoding='utf8')
        with self.assertRaises(ValueError):m.load_suite(self.control)
    def test_tampered_child_job_row_fails_before_test_model_CUDA_and_keeps_fixed_output(self):
        job,runtime=self.runtime_and_dispatch();p=job/'job_request.json';req=json.loads(p.read_bytes());foreign=self.root/'unapproved_output';req.update(model_id='m01',output_directory=str(foreign));p.write_text(json.dumps(req),encoding='utf8')
        with patch.object(confirmation,'authorize_confirmation') as auth,patch.object(m,'verified_ema_backend') as model,patch.object(runner,'setup_cuda') as cuda:
            self.assertEqual(m.run_job(self.control,self.job_id),1);auth.assert_not_called();model.assert_not_called();cuda.assert_not_called()
        self.assertFalse(foreign.exists());self.assertTrue((self.out/'jobs'/self.job_id/'result.json').exists())
    def test_dispatch_wait_partial_json_short_bound_and_absolute_deadline(self):
        job=self.root/'waiting';job.mkdir();runtime={'schema_version':'bbox_confirmation_runtime_v1','dispatch_started_unix':100.,'deadline_unix':1000.,'max_seconds':900};clock={'mono':0.,'wall':200.}
        def tick(seconds):clock['mono']+=1.;clock['wall']+=seconds
        with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']),patch.object(m.time,'time',side_effect=lambda:clock['wall']),patch.object(m.time,'sleep',side_effect=tick):
            with self.assertRaisesRegex(RuntimeError,'5秒'):m.wait_dispatch(job,runtime,[])
        self.assertEqual(clock['mono'],5.);self.assertEqual(runtime['deadline_unix'],1000.)
        dispatch=job/'dispatch_wrapper.json';dispatch.write_text('{',encoding='utf8');clock.update(mono=0.,wall=200.)
        def ready(seconds):clock['mono']+=.02;dispatch.write_text('{"CPUfixture":"complete"}',encoding='utf8')
        with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']),patch.object(m.time,'time',side_effect=lambda:clock['wall']),patch.object(m.time,'sleep',side_effect=ready):self.assertEqual(m.wait_dispatch(job,runtime,[]),{'CPUfixture':'complete'})
        dispatch.unlink();clock.update(mono=0.,wall=999.99)
        with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']),patch.object(m.time,'time',side_effect=lambda:clock['wall']),patch.object(m.time,'sleep',side_effect=tick):
            with self.assertRaisesRegex(ValueError,'截止'):m.wait_dispatch(job,runtime,[])
        self.assertEqual(runtime['deadline_unix'],1000.)
    def test_dispatch_wait_honors_STOP_before_parent_ready(self):
        job=self.root/'waiting';job.mkdir();runtime={'schema_version':'bbox_confirmation_runtime_v1','dispatch_started_unix':100.,'deadline_unix':1000.,'max_seconds':900};stop=job/'STOP.json'
        def stop_now(seconds):m.write_new(stop,{'CPUfixture':True})
        with patch.object(m.time,'time',return_value=200.),patch.object(m.time,'sleep',side_effect=stop_now):
            with self.assertRaisesRegex(RuntimeError,'USER_STOP'):m.wait_dispatch(job,runtime,[stop])
    def test_dispatch_child_and_parent_PID_CT_command_mismatch_not_waited_or_accepted(self):
        argv=[str(self.control/'observe.py'),str(self.control),'--child',self.job_id];command=['python','-B',*argv];parent_command=['python','observer','--suite'];parent=SimpleNamespace(pid=44,create_time=lambda:66.,cmdline=lambda:parent_command)
        current=SimpleNamespace(pid=55,create_time=lambda:77.,cmdline=lambda:command,parents=lambda:[parent]);dispatch={'schema_version':'bbox_confirmation_observed_dispatch_v1','pid':55,'create_time':77.,'command':command,'observer':{'pid':44,'create_time':66.,'command':parent_command}}
        with patch('psutil.Process',return_value=current),patch.object(m.sys,'argv',argv),patch.object(m.time,'sleep',side_effect=AssertionError('identity mismatch must not wait')):
            m.verify_dispatch_identity(dispatch,self.control,self.job_id)
            for bad in ({**dispatch,'pid':56},{**dispatch,'create_time':78.},{**dispatch,'command':['wrong']},{**dispatch,'observer':{**dispatch['observer'],'create_time':67.}},{**dispatch,'observer':{**dispatch['observer'],'command':['wrong']}}):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):m.verify_dispatch_identity(bad,self.control,self.job_id)
    def test_dispatch_accepts_verified_venv_wrapper_then_actual_controller(self):
        argv=[str(self.control/'observe.py'),str(self.control),'--child',self.job_id];command=['venv_python.exe','-B',*argv];observer_command=['python','observer','--suite']
        observer=SimpleNamespace(pid=44,create_time=lambda:66.,cmdline=lambda:observer_command);wrapper=SimpleNamespace(pid=55,create_time=lambda:77.,cmdline=lambda:command)
        current=SimpleNamespace(pid=56,create_time=lambda:77.01,cmdline=lambda:['base_python.exe',*command[1:]],parents=lambda:[wrapper,observer]);dispatch={'schema_version':'bbox_confirmation_observed_dispatch_v1','pid':55,'create_time':77.,'command':command,'observer':{'pid':44,'create_time':66.,'command':observer_command}}
        with patch('psutil.Process',return_value=current),patch.object(m.sys,'argv',argv):m.verify_dispatch_identity(dispatch,self.control,self.job_id)
        wrong_current=SimpleNamespace(pid=56,create_time=lambda:77.01,cmdline=lambda:['base_python.exe','wrong_script'],parents=lambda:[wrapper,observer])
        with patch('psutil.Process',return_value=wrong_current),patch.object(m.sys,'argv',argv):
            with self.assertRaises(ValueError):m.verify_dispatch_identity(dispatch,self.control,self.job_id)
    def test_suite_preparation_freezes_exact43_and_does_not_spawn_or_load_model(self):
        research=self.root/'freshD';output=self.root/'freshG/runs';(research/'runs').mkdir(parents=True);output.mkdir(parents=True);control=research/'runs/bbox_confirmation_frozen';observer=Path(__file__).parents[1]/'tools/bbox_confirmation_observer.py'
        with patch.object(confirmation,'authorize_confirmation',return_value=Permit()),patch.object(m,'check_resources',return_value=m.resource_bound(48)),patch.object(m.subprocess,'Popen',side_effect=AssertionError('prepare must not spawn')):
            result=m.prepare_suite(control,spec_reference=self.suite['spec_reference'],completion_reference=self.suite['completion_reference'],plan_reference=self.suite['plan_reference'],observer_source=observer,research=research,output_base=output)
        suite=m.load_suite(control);self.assertEqual(len(suite['source_code_sha256']),43);self.assertEqual(len(suite['jobs']),48);self.assertEqual(suite['observer_sha256'],m.sha(control/'observe.py'));self.assertTrue(Path(result['output_group']).is_dir())
        for path in (control/'code/futsalmot').glob('*.py'):compile(path.read_bytes(),str(path),'exec')
        with self.assertRaises(ValueError):m.prepare_suite(control,spec_reference=self.suite['spec_reference'],completion_reference=self.suite['completion_reference'],plan_reference=self.suite['plan_reference'],observer_source=observer,research=research,output_base=output)
    def test_fixed48_resource_formula_and_actual_free_and_global_deadline(self):
        bound=m.resource_bound();self.assertEqual(bound['prediction_bytes'],277161120000);self.assertEqual(bound['additional_G_bytes'],286156535472)
        with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)):
            self.assertEqual(m.check_resources(research=self.root,output=self.root,now=m.deadline_unix()-50000)['remaining_model_matches'],48)
            with self.assertRaises(ValueError):m.check_resources(research=self.root,output=self.root,now=m.deadline_unix()-40000)
        with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=m.RESERVE)):
            with self.assertRaises(ValueError):m.check_resources(research=self.root,output=self.root,now=0)
    def test_runtime_absolute_deadline_not_reset_and_STOP(self):
        runtime={'schema_version':'bbox_confirmation_runtime_v1','dispatch_started_unix':100.,'deadline_unix':1000.,'max_seconds':900};self.assertEqual(m.validate_runtime(runtime,now=999),1000)
        with self.assertRaises(ValueError):m.validate_runtime({**runtime,'deadline_unix':1001},now=101)
        with self.assertRaises(ValueError):m.validate_runtime(runtime,now=1000)
        p=self.root/'STOP.json';m.write_new(p,{})
        with self.assertRaises(RuntimeError):m.check_stop_deadline(runtime,[p],now=200)
    def test_prepare_refuses_scope_order_replay_and_STOP_before_auth(self):
        with patch.object(confirmation,'authorize_confirmation') as auth:
            m.write_new(self.control/'STOP.json',{})
            with self.assertRaises(ValueError):m.prepare_job(self.control,self.job_id)
            auth.assert_not_called()
        (self.control/'STOP.json').unlink();self.prepare()
        with patch.object(confirmation,'authorize_confirmation',return_value=Permit()),patch.object(m,'check_resources',return_value=m.resource_bound(48)):
            with self.assertRaises(ValueError):m.prepare_job(self.control,self.job_id)
            with self.assertRaises(ValueError):m.prepare_job(self.control,self.suite['jobs'][1]['job_id'])
    def test_ema_backend_uses_trusted_EMA_factory_and_no_optimizer(self):
        from futsalmot import pretrained,runtime_backend
        network=SimpleNamespace(model=torch.nn.Linear(1,1,bias=False),initialization_binding={'sha256':'a'*64,'path':str(self.root/'public.pt')});payload={'state_dict':{'weight':torch.tensor([[2.]])},'expected_source_bindings':{'public_model':{'sha256':'a'*64,'path':str(self.root/'public.pt')}},'checkpoint_receipt':{'sha256':'b'*64}}
        with patch.object(confirmation,'load_terminal_ema',return_value=payload) as read,patch.object(pretrained,'load_local_yolo11_detect',return_value=network) as factory,patch.object(torch.optim,'SGD',side_effect=AssertionError('确认不得构建SGD')):
            backend=m.verified_ema_backend(Permit(),'m00',self.root)
        read.assert_called_once();factory.assert_called_once_with(self.root,model_size='s');self.assertEqual(backend.ema_state['state_dict']['weight'].item(),2.);self.assertEqual(backend.stage_update,1500);self.assertIs(backend.ema_network_for_eval.__func__,runtime_backend.NativeBBoxBackend.ema_network_for_eval)
    def test_run_success_gate_order_and_900_passed_to_original_candidate(self):
        code,result,events=self.run_mocked();self.assertEqual(code,0);self.assertEqual(result['state'],'COMPLETE');self.assertEqual(result['optimizer_updates'],0);self.assertLess(events.index('authorize24'),events.index('lock'));self.assertLess(events.index('lock'),events.index('loadEMA'));self.assertEqual(events[-1],'cleanup')
    def test_initialize_failure_and_true_ignore_keep_failure_evidence(self):
        code,result,events=self.run_mocked(model_error=RuntimeError('CPU injected loader fail'));self.assertEqual(code,1);self.assertEqual(result['state'],'FAIL_STOP');self.assertEqual(events[-1],'cleanup');self.assertNotIn('evaluate',events)
    def test_true_ignore_refused_without_deleting_GT_or_running_evaluation(self):
        code,result,events=self.run_mocked(true_ignore=True);self.assertEqual(code,1);self.assertIn('true-ignore',result['error']['message']);self.assertNotIn('evaluate',events);self.assertTrue(result['worker_cleanup']['GPU_lock_retired'])
    def test_evaluation_failure_retains_error_and_never_retries(self):
        code,result,events=self.run_mocked(evaluate_error=RuntimeError('CPU injected eval failure'));self.assertEqual(code,1);self.assertEqual(result['state'],'FAIL_STOP');self.assertEqual(events.count('evaluate'),1)
    def test_dirty_workers_not_COMPLETE(self):
        code,result,events=self.run_mocked(dirty_workers=True);self.assertEqual(code,1);self.assertEqual(result['state'],'ATTENTION');self.assertFalse(result['worker_cleanup']['GPU_lock_retired'])
    def test_observer_exact_exit_receipt_seal_and_refuses_second_dispatch(self):
        self.prepare();job=self.control/'jobs'/self.job_id;child=FakeChild();starts=[]
        def spawn(*args,**kwargs):
            runtime=json.loads((job/'runtime_limit.json').read_bytes());starts.append(runtime['dispatch_started_unix']);m.write_new(job/'result.json',{'state':'COMPLETE','within_absolute_deadline':True});return child
        def times(handle):self.assertIs(handle,child._handle);return {'create_time':starts[0]+.01,'exit_time':starts[0]+2.}
        with patch.object(m.subprocess,'Popen',side_effect=spawn) as popen,patch.object(m,'process_times',side_effect=times),patch.object(m,'wait_owned',return_value=[]):
            self.assertEqual(m.observe_job(self.control,self.job_id),0)
            with self.assertRaises(ValueError):m.observe_job(self.control,self.job_id)
            self.assertEqual(popen.call_count,1);self.assertEqual(popen.call_args.kwargs['creationflags'],getattr(m.subprocess,'CREATE_NO_WINDOW',0))
        life=json.loads((job/'lifetime_receipt.json').read_bytes());self.assertEqual(life['state'],'VERIFIED_EXIT');self.assertEqual(life['full_process_seconds'],2.);self.assertTrue(life['precise_GetProcessTimes']);self.assertTrue((self.out/'jobs'/self.job_id/'control/CONTROL_MANIFEST.json').exists())
    def test_observer_over_budget_is_ATTENTION_even_complete_child(self):
        self.prepare();job=self.control/'jobs'/self.job_id;child=FakeChild();starts=[]
        def spawn(*args,**kwargs):
            runtime=json.loads((job/'runtime_limit.json').read_bytes());starts.append(runtime['dispatch_started_unix']);m.write_new(job/'result.json',{'state':'COMPLETE','within_absolute_deadline':True});return child
        with patch.object(m.subprocess,'Popen',side_effect=spawn),patch.object(m,'process_times',side_effect=lambda h:{'create_time':starts[0]+.01,'exit_time':starts[0]+901}),patch.object(m,'wait_owned',return_value=[]):self.assertEqual(m.observe_job(self.control,self.job_id),1)
        self.assertEqual(json.loads((job/'lifetime_receipt.json').read_bytes())['state'],'ATTENTION')
    def test_observer_at900_only_requests_STOP_and_records_limit_without_kill(self):
        self.prepare();job=self.control/'jobs'/self.job_id;clock={'now':100.}
        class SlowChild(FakeChild):
            def __init__(self):super().__init__(None);self.first=True
            def poll(self):
                if self.first:self.first=False;clock['now']=1000.5;return None
                return self.returncode
            def wait(self,timeout=None):self.returncode=1;return 1
        child=SlowChild()
        def spawn(*args,**kwargs):m.write_new(job/'result.json',{'state':'LIMIT_STOP','within_absolute_deadline':False});return child
        with patch.object(m.time,'time',side_effect=lambda:clock['now']),patch.object(m.subprocess,'Popen',side_effect=spawn),patch.object(m,'process_times',return_value={'create_time':100.01,'exit_time':1001.}),patch.object(m,'owned_tree',return_value=[]),patch.object(m,'wait_owned',return_value=[]):self.assertEqual(m.observe_job(self.control,self.job_id),1)
        self.assertTrue((job/'STOP.json').exists());self.assertEqual(json.loads((job/'lifetime_receipt.json').read_bytes())['state'],'ATTENTION')
    def test_observer_after930_preserves_ATTENTION_without_force_or_seal(self):
        self.prepare();job=self.control/'jobs'/self.job_id;clock={'now':100.}
        class HungChild(FakeChild):
            def __init__(self):super().__init__(None)
            def poll(self):clock['now']=1031.;return None
        child=HungChild()
        with patch.object(m.time,'time',side_effect=lambda:clock['now']),patch.object(m.subprocess,'Popen',return_value=child),patch.object(m,'process_times',return_value={'create_time':100.01,'exit_time':0.}),patch.object(m,'owned_tree',return_value=[{'pid':child.pid,'create_time':100.01,'command':['python','CPUfixture']}]):self.assertEqual(m.observe_job(self.control,self.job_id),1)
        self.assertTrue((job/'STOP.json').exists());life=json.loads((job/'lifetime_receipt.json').read_bytes());self.assertEqual(life['state'],'ATTENTION_STILL_EXITING');self.assertTrue(life['no_force_kill']);self.assertFalse((self.out/'jobs'/self.job_id/'control').exists())
    def test_suite_stops_after_one_failed_job_no_score_branch(self):
        with patch.object(m,'prepare_job') as prepare,patch.object(m,'observe_job',return_value=1) as observe:
            self.assertEqual(m.observe_suite(self.control),1);self.assertEqual(prepare.call_count,1);self.assertEqual(observe.call_count,1)
        state=json.loads((self.control/'suite_result.json').read_bytes());self.assertEqual(state['state'],'FAIL_STOP');self.assertEqual(state['completed'],[])
        with self.assertRaises(ValueError):m.observe_suite(self.control)
    def test_seal_refuses_unknown_or_overwrite(self):
        self.prepare();job=self.control/'jobs'/self.job_id;out=self.out/'jobs'/self.job_id;m.write_new(job/'unknown.json',{})
        with self.assertRaises(ValueError):m.seal_job(job,out)
    def test_existing_partial_never_calls_authorization_or_original_retry(self):
        job,runtime=self.runtime_and_dispatch();req=json.loads((job/'job_request.json').read_bytes());partial=Path(req['output_directory'])/'evaluation/previous_partial/raw.json';m.write_new(partial,{'retained':True});before=partial.read_bytes()
        with patch.object(confirmation,'authorize_confirmation') as authorize:
            self.assertEqual(m.run_job(self.control,self.job_id),1);authorize.assert_not_called()
        self.assertEqual(partial.read_bytes(),before);self.assertEqual(json.loads((job/'result.json').read_bytes())['state'],'FAIL_STOP')
    def test_STOP_before_entry_does_not_read_confirmation_or_model(self):
        job,runtime=self.runtime_and_dispatch();m.write_new(self.control/'STOP.json',{})
        with patch.object(confirmation,'authorize_confirmation') as authorize,patch.object(m,'verified_ema_backend') as model:
            self.assertEqual(m.run_job(self.control,self.job_id),1);authorize.assert_not_called();model.assert_not_called()
        self.assertEqual(json.loads((job/'result.json').read_bytes())['state'],'USER_STOP')
    def test_observer_live_owned_never_seals_even_complete_child(self):
        self.prepare();job=self.control/'jobs'/self.job_id;child=FakeChild();starts=[]
        def spawn(*args,**kwargs):
            runtime=json.loads((job/'runtime_limit.json').read_bytes());starts.append(runtime['dispatch_started_unix']);m.write_new(job/'result.json',{'state':'COMPLETE','within_absolute_deadline':True});return child
        with patch.object(m.subprocess,'Popen',side_effect=spawn),patch.object(m,'process_times',side_effect=lambda h:{'create_time':starts[0]+.01,'exit_time':starts[0]+2}),patch.object(m,'wait_owned',return_value=[{'pid':123,'create_time':77}]):self.assertEqual(m.observe_job(self.control,self.job_id),1)
        self.assertFalse((self.out/'jobs'/self.job_id/'control').exists());self.assertEqual(json.loads((job/'lifetime_receipt.json').read_bytes())['state'],'ATTENTION')
    def test_fixed48_mock_suite_closes_once_without_any_process(self):
        called=[]
        def observe(control,job_id):
            called.append(job_id);job=Path(control)/'jobs'/job_id;req=json.loads((job/'job_request.json').read_bytes());out=Path(req['output_directory']);summary={'images':750,'update':1500,'evaluated_model':'stage_EMA_at_recorded_update','main_metrics':{'AP50_95':.5}};summary_path=out/'evaluation'/req['tag']/'summary.json';m.write_new(summary_path,summary)
            for name,value in [('runtime_limit.json',{'fixture':True}),('dispatch_wrapper.json',{'fixture':True}),('result.json',{'state':'COMPLETE','job_id':job_id,'model_id':req['model_id'],'match_uid':req['match_uid'],'summary':summary,'summary_reference':m.reference(summary_path)}),('lifetime_receipt.json',{'state':'VERIFIED_EXIT','returncode':0,'owned_still_alive':[],'GPU_lock_retired':True})]:m.write_new(job/name,value)
            m.seal_job(job,out);return 0
        with patch.object(confirmation,'authorize_confirmation',return_value=Permit()),patch.object(m,'check_resources',side_effect=lambda **kw:m.resource_bound(kw['remaining'])),patch.object(m,'observe_job',side_effect=observe),patch.object(m.subprocess,'Popen',side_effect=AssertionError('no actual process')):
            self.assertEqual(m.observe_suite(self.control),0)
        self.assertEqual(len(called),48);self.assertEqual(len(set(called)),48);manifest=self.out/'control/CONTROL_MANIFEST.json';before=manifest.read_bytes();value=json.loads(before);self.assertEqual(value['completed_model_matches'],48);self.assertEqual(value['state'],'SEALED')
        with self.assertRaises(ValueError):m.seal_suite(self.control,self.suite)
        self.assertEqual(manifest.read_bytes(),before)
    def test_tracked_iterator_finally_covers_EMA_initialization_failure_before_original_try(self):
        from futsalmot import records,runner_data
        class Iterator:
            closed=False
            def __iter__(self):return self
            def __next__(self):raise StopIteration
            def _shutdown_workers(self):self.closed=True
        iterator=Iterator();loader=SimpleNamespace(__iter__=lambda:iterator)
        class Loader:
            timeout=0
            def __iter__(self):return iterator
        permit=SimpleNamespace(check_live=lambda:None,match=lambda uid:{'match_uid':uid});candidate=confirmation.make_confirmation_evaluator(permit,confirmation.MATCHES[0]);candidate['candidate_function'].__globals__['ConfirmationRequestPlanDataset']=lambda *a,**k:object()
        evaluate=m.scoped_evaluator(candidate,[]);backend=SimpleNamespace(stage_update=1500,ema_network_for_eval=Mock(side_effect=RuntimeError('after iterator EMA constructor failure')));panel={'val_catalogue':{},'val_uids':['CPUfixture']};config={'microbatch_size':8,'num_workers':2,'numeric_mode':'fp32'}
        with patch.object(confirmation,'make_confirmation_loader',return_value=Loader()),patch.object(records,'capture_rng_state',return_value={'CPU_fixture':True}),patch.object(records,'restore_rng_state'),patch.object(runner,'_shutdown',wraps=runner._shutdown) as shutdown,patch.object(torch.cuda,'empty_cache',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'EMA constructor failure'):evaluate(self.root/'cpu_eval',backend,panel,config,None,SimpleNamespace(event=lambda x:None),'CPUfixture')
        self.assertTrue(iterator.closed);shutdown.assert_called_once_with(iterator);self.assertTrue(evaluate.confirmation_lifecycle_binding['combined_inverse_AST_binding']['inverse_AST_exact']);self.assertEqual(evaluate.confirmation_lifecycle_binding['additional_substitution'][0],'iterator=iter(loader)')
        with self.assertRaises(ValueError):evaluate(self.root/'cpu_eval',backend,panel,config,None,None,'CPUfixture')
    def test_tracked_shutdown_finishes_before_GPU_lock_retirement_normal_or_failure(self):
        from futsalmot import records,runner_data
        events=[]
        class Iterator:
            closed=False
            def __iter__(self):return self
            def __next__(self):raise StopIteration
            def _shutdown_workers(self):self.closed=True;events.append('shutdown')
        iterator=Iterator()
        class Loader:
            timeout=0
            def __iter__(self):return iterator
        permit=SimpleNamespace(check_live=lambda:None,match=lambda uid:{'match_uid':uid});candidate=confirmation.make_confirmation_evaluator(permit,confirmation.MATCHES[0]);candidate['candidate_function'].__globals__['ConfirmationRequestPlanDataset']=lambda *a,**k:object();evaluate=m.scoped_evaluator(candidate,[])
        backend=SimpleNamespace(stage_update=1500,ema_network_for_eval=lambda:object());panel={'val_catalogue':{},'val_uids':['CPUfixture']};config={'microbatch_size':8,'num_workers':2,'numeric_mode':'fp32'};proc=SimpleNamespace(create_time=lambda:77.,children=lambda recursive:[]);fake_runner=SimpleNamespace(RESEARCH=self.root,production_closed=lambda:None,write_once=m.write_new)
        def waited(rows,seconds):events.append('lock_barrier');return [] if iterator.closed else [{'pid':4}]
        with patch.object(confirmation,'make_confirmation_loader',return_value=Loader()),patch.object(records,'capture_rng_state',return_value={'CPU_fixture':True}),patch.object(records,'restore_rng_state'),patch.object(torch.cuda,'empty_cache',return_value=None),patch('psutil.Process',return_value=proc),patch.object(m,'wait_owned',side_effect=waited):
            with m.ConfirmGPULock(fake_runner,self.control) as lock:
                with self.assertRaisesRegex(RuntimeError,'Incomplete validation images'):evaluate(self.root/'cpu_eval',backend,panel,config,None,SimpleNamespace(event=lambda x:None),'CPUfixture')
        self.assertTrue(iterator.closed);self.assertLess(events.index('shutdown'),events.index('lock_barrier'));self.assertTrue(lock.cleanup['GPU_lock_retired']);self.assertFalse((self.root/'.research_gpu.lock').exists())
    def test_GPU_lock_worker_barrier_and_unknown_lock_preserved(self):
        fake_runner=SimpleNamespace(RESEARCH=self.root,production_closed=lambda:None,write_once=m.write_new);process=SimpleNamespace(create_time=lambda:77.,children=lambda recursive:[])
        with patch('psutil.Process',return_value=process),patch.object(m,'wait_owned',return_value=[]):
            with m.ConfirmGPULock(fake_runner,self.control) as lock:pass
        self.assertTrue(lock.cleanup['GPU_lock_retired']);self.assertFalse((self.root/'.research_gpu.lock').exists())
        with patch('psutil.Process',return_value=process),patch.object(m,'wait_owned',return_value=[{'pid':9}]):
            with m.ConfirmGPULock(fake_runner,self.control) as dirty:pass
        self.assertFalse(dirty.cleanup['GPU_lock_retired']);self.assertTrue((self.root/'.research_gpu.lock').exists())

if __name__=='__main__':unittest.main()
