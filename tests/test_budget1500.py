"""1500步R/混合的CPU事务、实际完整快照恢复及统一短程协议检查。"""
import os
os.environ['YOLO_OFFLINE']='true'
from copy import deepcopy
from pathlib import Path
import ast,inspect,tempfile,unittest
from unittest.mock import patch
import torch
from futsalmot import budget1500 as m
from futsalmot.records import save_checkpoint,load_checkpoint,capture_rng_state,restore_rng_state
from futsalmot.optimizers import make_sgd_optimizer,apply_sgd_schedule
from futsalmot.training import initialize_ema_stage,update_ema_state,schedule_at
from futsalmot.restore import _equal

def samplers():return m.make_samplers({'train_uids':['r'+str(i) for i in range(21)]},{'ordered_uids':['s'+str(i) for i in range(23)]})
def batches(strategy,windows):
    rows=deepcopy(m.window_requests(strategy,windows))
    return [{'sample_lineage':[{'request':r} for r in rows[i:i+8]],'domains':[r['domain'] for r in rows[i:i+8]]} for i in (0,8)]

class TinyBackend:
    def __init__(self):
        self.model=torch.nn.Linear(2,2);self.optimizer=make_sgd_optimizer(self.model,lr=0.,momentum=.8,weight_decay=.0005)
        self.ema_state=initialize_ema_stage(self.model.state_dict());self.stage_update=0;self.final_stage=True;self.failed=False;self.numeric_mode='fp32';self.microbatch_size=8;self.device=torch.device('cpu')
    def _config(self):return {'stage_updates':1500,'final_stage':True,'microbatch_size':8,'numeric_mode':'fp32','input_size':960,'device':'cpu','initialization_binding':{'CPU_fixture':True},'schedule_policy':m.SCHEDULE_POLICY}
    def step(self):
        step=self.stage_update+1;apply_sgd_schedule(self.optimizer,m.schedule_1500(step,1500,final_stage=True));self.optimizer.zero_grad()
        loss=self.model(torch.rand((16,2))).square().mean();loss.backward();self.optimizer.step();self.ema_state=update_ema_state(self.ema_state,self.model.state_dict(),update=step);self.stage_update=step;return float(loss.detach())

class Budget1500Tests(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng_state();self.guards=[patch.object(torch.cuda,n,side_effect=AssertionError('CPU only')) for n in ('init','_lazy_init','is_available')]
        for g in self.guards:g.start()
    def tearDown(self):
        restore_rng_state(self.rng)
        for g in self.guards:g.stop()
    def test_full_1500_request_table_matches_actual_transactions_both_strategies(self):
        for strategy in ('R','R+S'):
            with self.subTest(strategy=strategy):
                live=samplers();requests=m.phase_requests(strategy,live,start=0,end=1500)
                self.assertEqual(m.exposure_counts(live),{'R':0,'S':0})
                for step in range(1500):
                    window=m.reserve_windows(strategy,live);data=batches(strategy,window);m.validate_batches(strategy,data,window)
                    self.assertEqual(m.window_requests(strategy,window),requests[step*16:(step+1)*16])
                    self.assertEqual(m.commit_windows(strategy,live,window),m.exposure_plan(strategy,step+1))
                self.assertEqual(m.exposure_counts(live),{'R':24000,'S':0} if strategy=='R' else {'R':12000,'S':12000})
    def test_invalid_last_token_is_atomic_for_one_or_two_domains(self):
        for strategy in ('R','R+S'):
            live=samplers();window=m.reserve_windows(strategy,live);window['S' if strategy=='R+S' else 'R']['token']='bad'
            with self.assertRaises(ValueError):m.commit_windows(strategy,live,window)
            self.assertEqual(m.exposure_counts(live),{'R':0,'S':0})
    def test_micro_validation_failure_does_not_commit_or_silently_change_order(self):
        for strategy in ('R','R+S'):
            live=samplers();window=m.reserve_windows(strategy,live);data=batches(strategy,window);data[1]['sample_lineage'][2]['request']['uid']='bad'
            with self.assertRaises(ValueError):m.validate_batches(strategy,data,window)
            self.assertEqual(m.exposure_counts(live),{'R':0,'S':0})
            for v in live.values():v.cancel_pending()
            self.assertEqual(m.window_requests(strategy,m.reserve_windows(strategy,live)),m.phase_requests(strategy,live,start=0,end=1))
    def test_disk_restore_replays_next_optimizer_state_loss_and_rng_exact_both_arms(self):
        for strategy in ('R','R+S'):
            with self.subTest(strategy=strategy):
                live=samplers();backend=TinyBackend();backend.step();m.commit_windows(strategy,live,m.reserve_windows(strategy,live))
                bundle=m.snapshot(backend,strategy=strategy,samplers=live,source_bindings={'fixture':1},committed_updates=1,extra_state={'loader_generator_state':torch.Generator().manual_seed(170017).get_state(),'loader_iteration_started':True,'run_scope':m.SCHEMA})
                with tempfile.TemporaryDirectory() as d:
                    receipt=save_checkpoint(Path(d)/'state.pt',bundle);loaded=load_checkpoint(receipt,expected_bindings={'fixture':1})
                loss=backend.step();gradients=[v.grad.clone() for v in backend.model.parameters()];rng=torch.get_rng_state().clone();fresh=TinyBackend();fresh_s=samplers()
                restored=m.restore(fresh,loaded,strategy=strategy,samplers=fresh_s,expected_bindings={'fixture':1},expected_update=1)
                self.assertTrue(restored['RNG_restored_last']);self.assertEqual(loss,fresh.step())
                self.assertTrue(_equal(backend.model.state_dict(),fresh.model.state_dict()));self.assertTrue(_equal(backend.optimizer.state_dict(),fresh.optimizer.state_dict()));self.assertTrue(_equal(backend.ema_state,fresh.ema_state));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a,b.grad) for a,b in zip(gradients,fresh.model.parameters())))
                self.assertEqual(m.exposure_counts(fresh_s),m.exposure_plan(strategy,1))
                for other in ('R','R+S'):
                    if other!=strategy:
                        with self.assertRaises(ValueError):m.validate_snapshot(loaded,strategy=other,samplers=fresh_s,expected_bindings={'fixture':1},expected_update=1)
    def test_snapshot_rejects_missing_loader_or_boundary_and_actual_sampler_drift(self):
        for strategy in ('R','R+S'):
            live=samplers();backend=TinyBackend();backend.step();m.commit_windows(strategy,live,m.reserve_windows(strategy,live))
            bundle=m.snapshot(backend,strategy=strategy,samplers=live,source_bindings={'fixture':1},committed_updates=1,extra_state={'loader_generator_state':torch.Generator().get_state(),'loader_iteration_started':True,'run_scope':m.SCHEMA})
            for field in ('loader_generator_state','loader_iteration_started','run_scope'):
                bad=deepcopy(bundle);bad.pop(field)
                with self.assertRaises(ValueError):m.validate_snapshot(bad,strategy=strategy,samplers=live,expected_bindings={'fixture':1},expected_update=1)
            bad=deepcopy(bundle);bad['sampler']['S']['sampler']['draws']+=8
            with self.assertRaises(ValueError):m.validate_snapshot(bad,strategy=strategy,samplers=live,expected_bindings={'fixture':1},expected_update=1)
    def test_shared_backend_has_original_code_and_complete_1500_cosine(self):
        from futsalmot.runtime_backend import NativeBBoxBackend
        self.assertIs(NativeBBoxBackend.step_update.__globals__['schedule_at'],schedule_at)
        self.assertIs(m.Budget1500Backend.step_update.__code__,NativeBBoxBackend.step_update.__code__)
        self.assertIs(m.Budget1500Backend.step_update.__globals__['schedule_at'],m.schedule_1500)
        with patch.object(m.Budget1500Backend,'from_public',return_value='fixture') as public:
            self.assertEqual(m.make_backend(device='cpu'),'fixture')
            self.assertEqual(public.call_args.kwargs['stage_updates'],1500);self.assertTrue(public.call_args.kwargs['final_stage'])
        self.assertEqual(m.schedule_1500(100,1500,final_stage=True)['lr'],.001)
        self.assertAlmostEqual(m.schedule_1500(1500,1500,final_stage=True)['lr'],.00001)
        self.assertLess(m.schedule_1500(500,1500,final_stage=True)['lr'],schedule_at(500,6563,final_stage=True)['lr'])
    def test_protocol_exact_counts_endpoints_and_rejects_mutation(self):
        from futsalmot import runner
        for strategy in ('R','R+S'):
            cfg=m.protocol(strategy,plan_sha256='a'*64,input_cache={'path':'D:/cache','manifest_sha256':'b'*64},baseline_R_status_reference={'path':str(m.BASELINE_STATUS),'sha256':'c'*64},additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)
            m.check_protocol({**cfg,'control_group':'D:/fixture'})
            self.assertEqual(cfg['total_image_exposures'],24000);self.assertEqual(cfg['complete_checkpoint_count'],sum(runner.checkpoint_due(i,1500,1000) for i in range(1501)));self.assertEqual(len(cfg['evaluation_schedule']),2)
            self.assertEqual(cfg['evaluation_schedule'][-1]['tag'],'step001500_final_final');self.assertFalse(cfg['stage_reset']);self.assertEqual(cfg['warmup_updates'],100)
            for key,value in [('updates',6563),('stage_reset',True),('checkpoint_every',500),('strategy','S->R')]:
                with self.assertRaises(ValueError):m.check_protocol({**cfg,'control_group':'D:/fixture',key:value})
        self.assertFalse(torch.cuda.is_initialized())
    def test_real_loop_uses_public_snapshot_not_shadowed_and_one_dispatch_path(self):
        tree=ast.parse(inspect.getsource(m.run_job));nested={n.name for n in tree.body[0].body if isinstance(n,ast.FunctionDef)}
        self.assertNotIn('snapshot',nested);self.assertIn('save_snapshot',nested)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='make_backend' for n in calls),1)
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='phase_requests' for n in calls),1)
        due=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='checkpoint_due')
        self.assertEqual(due.args[2].value,1000)

if __name__=='__main__':unittest.main()
