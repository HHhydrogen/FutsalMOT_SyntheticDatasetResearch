"""固定混合CPU事务/奇数完整边界/确定性续接验证；不访问GPU或公开权重。"""
import os
os.environ['YOLO_OFFLINE']='true'
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch
from futsalmot import fixed_mix as mix
from futsalmot.records import save_checkpoint,load_checkpoint,capture_rng_state,restore_rng_state
from futsalmot.optimizers import make_sgd_optimizer,apply_sgd_schedule
from futsalmot.training import initialize_ema_stage,update_ema_state,schedule_at
from futsalmot.restore import _equal

def samplers():return mix.make_samplers({'train_uids':['r'+str(i) for i in range(21)]},{'ordered_uids':['s'+str(i) for i in range(23)]})

def batches(window):
    rows=deepcopy(mix.window_requests(window))
    return [{'sample_lineage':[{'request':r} for r in rows[i:i+8]],'domains':[r['domain'] for r in rows[i:i+8]]} for i in (0,8)]

class TinyBackend:
    def __init__(self):
        self.model=torch.nn.Linear(2,2);self.optimizer=make_sgd_optimizer(self.model,lr=0.,momentum=.8,weight_decay=.0005)
        self.ema_state=initialize_ema_stage(self.model.state_dict());self.stage_update=0;self.final_stage=True;self.failed=False;self.numeric_mode='fp32';self.microbatch_size=8;self.device=torch.device('cpu')
    def _config(self):return {'stage_updates':6563,'final_stage':True,'microbatch_size':8,'numeric_mode':'fp32','input_size':960,'device':'cpu','initialization_binding':{'CPU_fixture':True}}
    def step(self):
        step=self.stage_update+1;apply_sgd_schedule(self.optimizer,schedule_at(step,6563,final_stage=True));self.optimizer.zero_grad()
        loss=self.model(torch.rand((16,2))).square().mean();loss.backward();self.optimizer.step();self.ema_state=update_ema_state(self.ema_state,self.model.state_dict(),update=step);self.stage_update=step;return float(loss.detach())

class FixedMixTests(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng_state();self.guards=[patch.object(torch.cuda,n,side_effect=AssertionError('CPU only')) for n in ('init','_lazy_init','is_available')]
        for g in self.guards:g.start()
    def tearDown(self):
        restore_rng_state(self.rng)
        for g in self.guards:g.stop()
    def test_every_micro_matches_independent_R_S_seeded_streams_and_budget(self):
        live=samplers();requests=mix.mix_requests(live,start=0,end=17)
        self.assertEqual(mix.exposure_counts(live),{'R':0,'S':0})
        for step in range(17):
            win=mix.reserve_windows(live);data=batches(win);mix.validate_batches(data,win)
            self.assertEqual([r for b in data for r in [x['request'] for x in b['sample_lineage']]],requests[step*16:(step+1)*16])
            self.assertEqual([b['domains'] for b in data],[['R','S']*4]*2)
            self.assertEqual(mix.commit_windows(live,win),{'R':8*(step+1),'S':8*(step+1)})
        self.assertEqual(mix.exposure_plan(6563),{'R':52504,'S':52504})
    def test_second_token_bad_cannot_partially_commit_R(self):
        live=samplers();win=mix.reserve_windows(live);win['S']['token']='bad'
        with self.assertRaises(ValueError):mix.commit_windows(live,win)
        self.assertEqual(mix.exposure_counts(live),{'R':0,'S':0});self.assertEqual(live['R'].pending_batches,1)
    def test_failed_micro_validation_keeps_both_committed_streams(self):
        live=samplers();win=mix.reserve_windows(live);bad=batches(win);bad[1]['sample_lineage'][2]['request']['uid']='bad'
        with self.assertRaises(ValueError):mix.validate_batches(bad,win)
        self.assertEqual(mix.exposure_counts(live),{'R':0,'S':0})
        for v in live.values():v.cancel_pending()
        self.assertEqual(mix.window_requests(mix.reserve_windows(live)),mix.mix_requests(live,start=0,end=1))
    def test_restore_request_table_ignores_pending_and_cannot_commit_old_token(self):
        live=samplers();mix.commit_windows(live,mix.reserve_windows(live));saved={k:v.state_dict() for k,v in live.items()};old=mix.reserve_windows(live)
        fresh=samplers()
        for k,v in fresh.items():v.load_state_dict(saved[k])
        self.assertEqual(mix.mix_requests(fresh,start=1,end=5),mix.mix_requests(live,start=1,end=5))
        with self.assertRaises(ValueError):mix.commit_windows(fresh,old)
        self.assertEqual(mix.exposure_counts(fresh),{'R':8,'S':8})
    def test_odd_snapshot_disk_fresh_restore_next_optimizer_step_and_rng_exact(self):
        live=samplers();original=TinyBackend();original.step();mix.commit_windows(live,mix.reserve_windows(live));gen=torch.Generator().manual_seed(170017)
        bundle=mix.snapshot_mixed(original,samplers=live,source_bindings={'CPU_fixture':1},committed_updates=1,extra_state={'loader_generator_state':gen.get_state(),'loader_iteration_started':True,'run_scope':mix.SCHEMA})
        with tempfile.TemporaryDirectory() as d:
            receipt=save_checkpoint(Path(d)/'full.pt',bundle);loaded=load_checkpoint(receipt,expected_bindings={'CPU_fixture':1})
        self.assertEqual(loaded['sampler']['R']['sampler']['draws'],8);self.assertEqual(loaded['sampler']['S']['sampler']['draws'],8)
        expected_loss=original.step();expected_rng=torch.get_rng_state().clone();fresh=TinyBackend();fresh_s=samplers()
        restored=mix.restore_mixed(fresh,loaded,samplers=fresh_s,expected_bindings={'CPU_fixture':1},expected_update=1)
        self.assertTrue(restored['RNG_restored_last']);self.assertEqual(fresh.step(),expected_loss)
        self.assertTrue(_equal(fresh.model.state_dict(),original.model.state_dict()));self.assertTrue(_equal(fresh.optimizer.state_dict(),original.optimizer.state_dict()));self.assertTrue(_equal(fresh.ema_state,original.ema_state));self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assertEqual(mix.exposure_counts(fresh_s),{'R':8,'S':8})
    def test_mixed_snapshot_rejects_wrong_ratios_and_metadata(self):
        live=samplers();backend=TinyBackend();backend.step();mix.commit_windows(live,mix.reserve_windows(live));bundle=mix.snapshot_mixed(backend,samplers=live,source_bindings={'fixture':1},committed_updates=1,extra_state={'loader_generator_state':torch.Generator().get_state(),'loader_iteration_started':True,'run_scope':mix.SCHEMA})
        for field in ('loader_generator_state','loader_iteration_started','run_scope'):
            bad=deepcopy(bundle);bad.pop(field)
            with self.assertRaises(ValueError):mix.validate_mixed(bad,samplers=live,expected_bindings={'fixture':1},expected_update=1)
        for key in ('phase_updates','committed_updates'):
            bad=deepcopy(bundle);bad[key]=2
            with self.assertRaises(ValueError):mix.validate_mixed(bad,samplers=live,expected_bindings={'fixture':1},expected_update=1)
        bad=deepcopy(bundle);bad['sampler']['S']['sampler']['draws']=16
        with self.assertRaises(ValueError):mix.validate_mixed(bad,samplers=live,expected_bindings={'fixture':1},expected_update=1)
        with self.assertRaises(ValueError):mix.validate_mixed(bundle,samplers=live,expected_bindings={'other':1},expected_update=1)
    def test_loader_iterator_rebuild_rewinds_only_saved_iteration_seed(self):
        from futsalmot.formal import make_phase_iterator
        gen=torch.Generator().manual_seed(170017)
        class Loader:
            def __iter__(self):torch.randint(100,(1,),generator=gen);return iter([1])
        before=gen.get_state().clone();make_phase_iterator(Loader(),gen,iteration_already_started=False);self.assertFalse(torch.equal(before,gen.get_state()))
        before=gen.get_state().clone();make_phase_iterator(Loader(),gen,iteration_already_started=True);self.assertTrue(torch.equal(before,gen.get_state()))
    def test_protocol_is_independent_single_stage_same_R_total_and_endpoints(self):
        from futsalmot import runner
        cfg=mix.protocol(plan_sha256='a'*64,input_cache={'path':'D:/cache','manifest_sha256':'b'*64},baseline_R_status_reference={'path':str(mix.BASELINE_STATUS),'sha256':'c'*64},additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)
        mix.check_protocol({**cfg,'control_group':'D:/fixture'})
        self.assertEqual(cfg['schema_version'],'bbox_fixed_budget_mix_v1');self.assertEqual(cfg['updates'],6563);self.assertEqual(cfg['complete_checkpoint_count'],sum(runner.checkpoint_due(i,6563,500) for i in range(6564)));self.assertEqual(len(cfg['evaluation_schedule']),2)
        self.assertEqual(cfg['evaluation_schedule'][0]['tag'],'step000000_final_PT');self.assertNotIn('pt0_summary_reference',cfg)
        for key,value in [('strategy','fixed_mix_R8_S8'),('updates',36750),('stage_reset',True),('sampler_contract',{'R_per_update':16})]:
            with self.assertRaises(ValueError):mix.check_protocol({**cfg,'control_group':'D:/fixture',key:value})
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
