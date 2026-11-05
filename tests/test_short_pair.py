"""短两阶段CPU协议、调度、真实原生梯度与完整恢复检查；不访问GPU/公开权重。"""
import os
os.environ['YOLO_OFFLINE']='true'
from copy import deepcopy
from pathlib import Path
import ast,inspect,random,tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
from futsalmot import short_pair as s,formal
from futsalmot.runtime_backend import NativeBBoxBackend
from futsalmot.detectors import build_random_yolo11
from futsalmot.records import capture_rng_state,restore_rng_state,save_checkpoint,load_checkpoint
from futsalmot.restore import _equal


def args():
    return dict(plan_sha256='a'*64,input_cache={'path':'D:/cache','manifest_sha256':'b'*64},baseline_R_status_reference={'path':str(s.BASELINE_STATUS),'sha256':'c'*64},additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)


def samplers():return s.make_samplers({'train_uids':['r'+str(i) for i in range(21)]},{'ordered_uids':['s'+str(i) for i in range(23)]},17)


def batches(window):
    rows=window['samples'];out=[]
    for start in (0,8):
        req=rows[start:start+8];uids=[r['uid'] for r in req];domains=['S' if r['domain']=='S' else 'R' for r in req]
        out.append({'img':torch.rand((8,3,64,64)),'numeric_input':'float32_RGB_NCHW_divided_by_255_once','cls':torch.zeros((8,1)),'bboxes':torch.tensor([[.5,.5,.4,.5]]*8),'batch_idx':torch.arange(8),'ignore_bboxes':torch.empty((0,4)),'ignore_batch_idx':torch.empty((0,),dtype=torch.int64),'uids':uids,'domains':domains,'sample_lineage':[{'request':r,'data_domain':d,'sampling_stream':r['domain'],'source_reference':{'CPU_fixture':True},'image_source_binding':{'CPU_fixture':True}} for r,d in zip(req,domains)]})
    return out


class ShortPairTests(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng_state();self.threads=torch.get_num_threads();torch.set_num_threads(2)
        self.guards=[patch.object(torch.cuda,n,side_effect=AssertionError('CPU fixture禁止GPU')) for n in ('init','_lazy_init','is_available')]
        for g in self.guards:g.start()
    def tearDown(self):
        restore_rng_state(self.rng);torch.set_num_threads(self.threads)
        for g in reversed(self.guards):g.stop()
    def test_protocol_both_independent_equal_total_and_no_conditional_AP(self):
        for strategy,total in [('S->R',{'R':81008,'S':24000}),('R->R',{'R':105008,'S':0})]:
            cfg=s.protocol(strategy,**args());s.check_protocol({**cfg,'control_group':'D:/fixture'})
            self.assertEqual(cfg['updates'],6563);self.assertEqual(cfg['terminal_exposures'],total);self.assertEqual(sum(total.values()),105008)
            self.assertEqual(cfg['initialization'],'independent_public_yolo11s');self.assertNotIn('parent_checkpoint_receipt',cfg);self.assertNotIn('conditional_A_result',cfg)
            self.assertFalse(cfg['test_enabled']);self.assertFalse(cfg['best_selection']);self.assertFalse(cfg['automatic_resume'])
    def test_protocol_rejects_changed_lr_horizon_frequency_or_missing_cache(self):
        cfg={**s.protocol('S->R',**args()),'control_group':'D:/fixture'}
        for k,v in [('first_lr_peak',.01),('U_S',3000),('U_R',6563),('updates',9563),('checkpoint_every',500),('warmup_updates_per_stage',75),('complete_checkpoint_count',15),('strategy','R+S')]:
            with self.subTest(key=k),self.assertRaises(ValueError):s.check_protocol({**cfg,k:v})
        for invalid in ('A','R','S'):
            with self.assertRaises(ValueError):s.protocol(invalid,**args())
        with self.assertRaises(ValueError):s.protocol('S->R',max_run_seconds=10801,**args())
    def test_schedule_warm100_and_each_full_cosine_endpoints(self):
        for final,U in [(False,1500),(True,5063)]:
            self.assertEqual(s.short_schedule(0,U,final_stage=final)['lr'],0)
            self.assertAlmostEqual(s.short_schedule(100,U,final_stage=final)['lr'],.001)
            self.assertAlmostEqual(s.short_schedule(75,U,final_stage=final)['lr'],.00075)
            self.assertAlmostEqual(s.short_schedule(U,U,final_stage=final)['lr'],.00001)
            self.assertEqual(s.short_schedule(100,U,final_stage=final)['momentum'],.937)
            values=[s.short_schedule(i,U,final_stage=final)['lr'] for i in range(100,U+1)]
            self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        with self.assertRaises(ValueError):s.short_schedule(1,1500,final_stage=True)
    def test_backend_reuses_same_original_code_only_schedule_global_changes(self):
        original=NativeBBoxBackend.step_update;derived=s.ShortPairBackend.step_update
        self.assertIs(original.__code__,derived.__code__)
        self.assertEqual(set(original.__globals__),set(derived.__globals__))
        for k,v in original.__globals__.items():
            self.assertIs(derived.__globals__[k],s.short_schedule if k=='schedule_at' else v)
        self.assertIsNot(original.__globals__['schedule_at'],s.short_schedule)
    def test_actual_loop_inverse_changes_only_backend_import_and_checkpoint_interval(self):
        candidate=inspect.getsource(s.run_job).replace('from .short_pair import ShortPairBackend as NativeBBoxBackend','from .runtime_backend import NativeBBoxBackend').replace('runner.checkpoint_due(local, count, 1000)','runner.checkpoint_due(local, count, 500)')
        self.assertEqual(ast.dump(ast.parse(candidate),include_attributes=False),ast.dump(ast.parse(inspect.getsource(formal.run_job)),include_attributes=False))
    def test_four_tags_keep_phase_reset_separate_and_ten_checkpoints(self):
        from futsalmot.runner import checkpoint_due
        cfg=s.protocol('S->R',**args());schedule=cfg['evaluation_schedule']
        self.assertEqual([r['global_update'] for r in schedule],[0,1500,1500,6563])
        self.assertEqual(len({r['tag'] for r in schedule}),4)
        points=[[i for i in range(p['updates']+1) if checkpoint_due(i,p['updates'],1000)] for p in cfg['phases']]
        self.assertEqual(points,[[0,1000,1500],[0,1000,2000,3000,4000,5000,5063]])
        self.assertEqual(sum(map(len,points)),cfg['complete_checkpoint_count'])
    def test_final_R_stream_matches_and_first_stream_separate_no_live_consumption(self):
        sr=samplers();rr=samplers()
        a=s.phase_requests('S->R',sr,start=0,end=1500);b=s.phase_requests('R->R',rr,start=0,end=1500)
        self.assertTrue(all(x['domain']=='S' for x in a));self.assertTrue(all(x['domain']=='R_extra' for x in b))
        self.assertEqual(s.exposure_counts(sr),{'R':0,'S':0});self.assertEqual(s.exposure_counts(rr),{'R':0,'S':0})
        sr['S'].commit(sr['S'].reserve(24000)['token']);rr['R_extra'].commit(rr['R_extra'].reserve(24000)['token'])
        self.assertEqual(s.phase_requests('S->R',sr,start=1500,end=1503),s.phase_requests('R->R',rr,start=1500,end=1503))
    def test_full_disk_checkpoint_native_cpu_restore_next_step_gradients_and_rng_exact(self):
        original=s.ShortPairBackend(build_random_yolo11(model_size='n'),stage_updates=1500,final_stage=False,microbatch_size=8,input_size=64)
        live=samplers();w=live['S'].reserve(16);m=original.step_update(batches(w),update=1);live['S'].commit(w['token'])
        self.assertEqual(m['lr'],.00001);self.assertEqual(m['momentum'],.80137)
        gen=torch.Generator().manual_seed(170017);binding={'CPU_fixture':'short_pair_native'}
        bundle=original.snapshot(samplers=live,source_bindings=binding,committed_updates=1,cumulative_exposures={'R':0,'S':16},phase='first',extra_state={'loader_generator_state':gen.get_state(),'loader_iteration_started':True,'run_scope':s.SCHEMA})
        before=capture_rng_state()
        with tempfile.TemporaryDirectory() as temp:
            receipt=save_checkpoint(Path(temp)/'full.pt',bundle);loaded=load_checkpoint(receipt,expected_bindings=binding)
        self.assertTrue(torch.equal(before['torch_cpu'],capture_rng_state()['torch_cpu']))
        wa=live['S'].reserve(16);expected=original.step_update(batches(wa),update=2);live['S'].commit(wa['token']);expected_grad=[p.grad.detach().clone() for p in original._parameters if p.grad is not None];expected_rng=torch.get_rng_state().clone()
        fresh=s.ShortPairBackend(build_random_yolo11(model_size='n'),stage_updates=1500,final_stage=False,microbatch_size=8,input_size=64);restored_s=samplers()
        restored=fresh.restore_bundle(loaded,samplers=restored_s,expected_bindings=binding)
        self.assertTrue(restored['RNG_restored_last']);self.assertEqual(restored['extra_state']['run_scope'],s.SCHEMA)
        wb=restored_s['S'].reserve(16);actual=fresh.step_update(batches(wb),update=2);restored_s['S'].commit(wb['token'])
        self.assertEqual(expected['loss'],actual['loss']);self.assertEqual(expected['lr'],actual['lr'])
        self.assertTrue(_equal(original.model.state_dict(),fresh.model.state_dict()));self.assertTrue(_equal(original.optimizer.state_dict(),fresh.optimizer.state_dict()));self.assertTrue(_equal(original.ema_state,fresh.ema_state))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected_grad,[p.grad for p in fresh._parameters if p.grad is not None])))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()));self.assertEqual(s.exposure_counts(live),s.exposure_counts(restored_s))
    def test_transition_preserves_raw_and_rebuilds_optimizer_ema_with_new_horizon(self):
        original=s.ShortPairBackend(build_random_yolo11(model_size='n'),stage_updates=1500,final_stage=False,microbatch_size=8,input_size=64)
        with self.assertRaises(ValueError):s.transition_backend(original)
        original.stage_update=1500
        raw=deepcopy(original.model.state_dict());new=s.transition_backend(original)
        self.assertTrue(_equal(raw,new.model.state_dict()));self.assertEqual(set(raw),set(new.ema_state['state_dict']));self.assertTrue(all(torch.equal(v,new.ema_state['state_dict'][k]) for k,v in raw.items()))
        self.assertEqual(new.stage_update,0);self.assertEqual(new.ema_state['updates'],0);self.assertEqual(new.optimizer.state,{})
        self.assertEqual(new.stage_updates,5063);self.assertTrue(new.final_stage);self.assertEqual(new._config()['schedule_policy'],s.SCHEDULE_POLICY)

if __name__=='__main__':unittest.main()