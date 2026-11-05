"""正式五策略的采样/边界/全状态CPU检查；随机小网络不代表主线成绩。"""
import os
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from futsalmot import formal
from futsalmot.data import CycleSampler, PrefetchSampler
from futsalmot.training import domain_at


def pools():
    return {'train_uids': [f'r{i}' for i in range(19)]}, {'ordered_uids': [f's{i}' for i in range(23)]}


class FormalPlanTests(unittest.TestCase):
    def test_budget_and_all_stage_evaluation_counts(self):
        for strategy, cp, val, total in [('R', 15, 2, 6563), ('S', 62, 2, 30187),
                                         ('S->R', 77, 4, 36750), ('R->R', 77, 4, 36750), ('R+S', 77, 4, 36750)]:
            p = formal.protocol(strategy, 17, max_run_seconds=14400, additional_space_bound_bytes=10,
                raw_upper_per_image_bytes=10, checkpoint_bytes=10, admission_sha256='a'*64)
            self.assertEqual((p['complete_checkpoint_count'], len(p['evaluation_schedule']), p['updates']), (cp, val, total))
            self.assertEqual(len({x['tag'] for x in p['evaluation_schedule']}), val)
        points = formal.evaluation_schedule('S->R')
        boundary = [x for x in points if x['global_update'] == formal.U_S]
        self.assertEqual([x['phase_update'] for x in boundary], [formal.U_S, 0])
        self.assertEqual([x['tag'] for x in boundary], ['step030187_first_final', 'step030187_final_initial'])

    def test_protocol_rejects_scope_and_parameter_changes(self):
        p = formal.protocol('R', 17, max_run_seconds=14400, additional_space_bound_bytes=10,
            raw_upper_per_image_bytes=10, checkpoint_bytes=10, admission_sha256='a'*64)
        p['control_group'] = '/fixture'
        formal._check_protocol(p)
        for key, value in [('numeric_mode', 'amp_bf16'), ('seed', 9), ('checkpoint_every', 50), ('test_enabled', True), ('evaluation_every', 1000), ('evaluation_policy', 'periodic')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                formal._check_protocol({**p, key:value})

    def test_pairing_R_and_S_sequences_and_extra_R_namespace(self):
        panel, synth = pools()
        r = formal.phase_requests('R', formal.make_samplers(panel,synth,17), start=0,end=3,u_s=5,u_r=3)
        sr = formal.phase_requests('S->R', formal.make_samplers(panel,synth,17), start=5,end=8,u_s=5,u_r=3)
        rr = formal.phase_requests('R->R', formal.make_samplers(panel,synth,17), start=0,end=3,u_s=5,u_r=3)
        self.assertEqual(r, sr)
        self.assertEqual({x['domain'] for x in rr}, {'R_extra'})
        self.assertNotEqual(r, rr)
        ss = formal.phase_requests('S', formal.make_samplers(panel,synth,17), start=0,end=5,u_s=5,u_r=3)
        mixed = formal.phase_requests('R+S', formal.make_samplers(panel,synth,17), start=0,end=8,u_s=5,u_r=3)
        self.assertEqual([x for x in mixed if x['domain']=='R'],r)
        self.assertEqual([x for x in mixed if x['domain']=='S'],ss)

    def test_requests_never_advance_committed_sampler_and_resume_exact(self):
        panel, synth = pools()
        for strategy in ('S->R','R->R','R+S'):
            live = formal.make_samplers(panel,synth,29)
            whole = formal.phase_requests(strategy, live, start=0,end=9,u_s=5,u_r=4)
            self.assertEqual(formal.exposure_counts(live), {'R':0,'S':0})
            for j in range(1,7):
                stream,_ = domain_at(strategy,j,5,4)
                window=live[stream].reserve(16)
                self.assertEqual(window['samples'],whole[(j-1)*16:j*16])
                live[stream].commit(window['token'])
            state={k:v.state_dict() for k,v in live.items()}
            restored=formal.make_samplers(panel,synth,29)
            for k,v in restored.items():v.load_state_dict(state[k])
            self.assertEqual(formal.phase_requests(strategy,restored,start=6,end=9,u_s=5,u_r=4),whole[6*16:])
            self.assertEqual(sum(formal.exposure_counts(restored).values()),96)

    def test_evaluation_event_has_global_and_phase_local_update(self):
        class Recorder:
            def event(self,row):self.row=row
        recorder=Recorder();bridge=formal._EvaluationRecorder(recorder,36750,'final')
        bridge.event({'kind':'evaluation','update':6563,'images':750})
        self.assertEqual((recorder.row['update'],recorder.row['global_update'],recorder.row['phase_update']), (36750,36750,6563))


class EndpointEvaluationExecutionTests(unittest.TestCase):
    def config(self,strategy='S->R'):
        return formal.protocol(strategy,17,max_run_seconds=14400,additional_space_bound_bytes=10,
            raw_upper_per_image_bytes=10,checkpoint_bytes=10,admission_sha256='a'*64)

    def test_actual_endpoint_executor_skips_resumed_and_live_intermediate_points(self):
        from types import SimpleNamespace
        config=self.config();calls=[]
        def evaluated(*args):
            calls.append((args[6],args[1].stage_update,args[7]));return {'evaluated':len(calls)}
        with patch('futsalmot.formal._evaluate',side_effect=evaluated):
            for phase,count,offset in [('first',formal.U_S,0),('final',formal.U_R,formal.U_S)]:
                for local in sorted({0,1,500,1000,1500,count}):
                    backend=SimpleNamespace(stage_update=local,stage_updates=count)
                    result=formal._evaluate_if_due(None,backend,None,config,None,None,phase,offset+local,None)
                    self.assertEqual(result is not None,local in (0,count))
        self.assertEqual(calls,[('first',0,0),('first',30187,30187),('final',0,30187),('final',6563,36750)])

    def test_evaluation_restores_all_CPU_random_streams_loader_generator_and_modes_on_success_and_failure(self):
        import random,numpy as np,torch
        from types import SimpleNamespace
        for fail in (False,True):
            with self.subTest(fail=fail):
                model=torch.nn.Sequential(torch.nn.Linear(2,1),torch.nn.Dropout(.2));model.train();model[1].eval()
                gen=torch.Generator().manual_seed(123);backend=SimpleNamespace(device=torch.device('cpu'),model=model,stage_update=0,stage_updates=2)
                before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),gen.get_state().clone(),[m.training for m in model.modules()])
                def consume(*args,**kwargs):
                    random.random();np.random.rand();torch.rand(3);torch.rand(3,generator=gen);model.eval()
                    if fail:raise RuntimeError('fixture validation initialization failed')
                    return {'fixture_only':True}
                with patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU fixture forbids CUDA')),patch('futsalmot.runner.evaluate',side_effect=consume):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'initialization failed'):
                            formal._evaluate(None,backend,None,self.config(),gen,None,'first',0,None)
                    else:self.assertEqual(formal._evaluate(None,backend,None,self.config(),gen,None,'first',0,None),{'fixture_only':True})
                self.assertEqual(random.getstate(),before[0]);now=np.random.get_state();self.assertEqual(now[0],before[1][0]);np.testing.assert_array_equal(now[1],before[1][1]);self.assertEqual(now[2:],before[1][2:])
                self.assertTrue(torch.equal(torch.get_rng_state(),before[2]));self.assertTrue(torch.equal(gen.get_state(),before[3]));self.assertEqual([m.training for m in model.modules()],before[4])

    def test_different_validation_cadence_preserves_next_training_losses_and_parameters(self):
        # 仅微型CPU数值夹具，验证评价边界随机性隔离，不代表数据集成绩。
        import random,numpy as np,torch
        from types import SimpleNamespace
        def train(points):
            random.seed(11);np.random.seed(12);torch.manual_seed(13)
            model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.2),torch.nn.Linear(3,1));model.train()
            optimizer=torch.optim.SGD(model.parameters(),lr=.01);gen=torch.Generator().manual_seed(14)
            backend=SimpleNamespace(device=torch.device('cpu'),model=model,stage_update=0,stage_updates=6)
            config=self.config('R');config['evaluation_schedule']=[{'phase':'final','phase_update':x} for x in points]
            def consume(*args,**kwargs):
                random.random();np.random.rand();torch.rand(5);torch.rand(5,generator=gen);model.eval();return {'fixture_only':True}
            losses=[]
            with patch('futsalmot.runner.evaluate',side_effect=consume):
                formal._evaluate_if_due(None,backend,None,config,gen,None,'final',0,None)
                for i in range(1,7):
                    x=torch.rand((4,2),generator=gen)+random.random()+float(np.random.rand());target=torch.rand((4,1))
                    optimizer.zero_grad();loss=(model(x)-target).square().mean();loss.backward();optimizer.step();losses.append(loss.item());backend.stage_update=i
                    formal._evaluate_if_due(None,backend,None,config,gen,None,'final',i,None)
            return losses,{k:v.detach().clone() for k,v in model.state_dict().items()}
        with patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU fixture forbids CUDA')):
            direct=train([]);old_cadence=train([0,2,4,6]);new_cadence=train([0,6])
        self.assertEqual(direct[0],old_cadence[0]);self.assertEqual(direct[0],new_cadence[0])
        for key in direct[1]:
            self.assertTrue(torch.equal(direct[1][key],old_cadence[1][key]));self.assertTrue(torch.equal(direct[1][key],new_cadence[1][key]))


class FormalTorchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='formal-cpu-')
        cls.env={k:os.environ.get(k) for k in ('YOLO_OFFLINE','YOLO_CONFIG_DIR','MPLCONFIGDIR')}
        os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_CONFIG_DIR']=str(Path(cls.temp.name)/'yolo');os.environ['MPLCONFIGDIR']=str(Path(cls.temp.name)/'mpl')
        import torch
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(2)
        cls.guards=[]
        for name in ('_lazy_init','init','is_available'):
            guard=patch.object(torch.cuda,name,side_effect=AssertionError('CPU夹具禁止CUDA'));guard.start();cls.guards.append(guard)
    @classmethod
    def tearDownClass(cls):
        for x in reversed(cls.guards):x.stop()
        cls.torch.set_num_threads(cls.threads);cls.temp.cleanup()
        for k,v in cls.env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

    def backend(self):
        from futsalmot.detectors import build_random_yolo11
        from futsalmot.runtime_backend import NativeBBoxBackend
        return NativeBBoxBackend(build_random_yolo11(model_size='n'),stage_updates=2,final_stage=False,
                                 microbatch_size=2,input_size=64)

    def batches(self):
        torch=self.torch;gen=torch.Generator().manual_seed(27);out=[]
        for j in range(8):
            uids=[f's{j*2}',f's{j*2+1}']
            out.append({'img':torch.rand((2,3,64,64),generator=gen),'numeric_input':'float32_RGB_NCHW_divided_by_255_once',
                'cls':torch.zeros((2,1)),'bboxes':torch.tensor([[.5,.5,.4,.5],[.4,.4,.3,.3]]),'batch_idx':torch.tensor([0,1]),
                'ignore_bboxes':torch.empty((0,4)),'ignore_batch_idx':torch.empty((0,),dtype=torch.int64),
                'uids':uids,'domains':['S','S'],'sample_lineage':[{'request':{'uid':u,'domain':'S','epoch':0,'exposure_index':j*2+i,'augmentation_seed':3},
                    'data_domain':'S','sampling_stream':'S','source_reference':{'fixture':True},'image_source_binding':{'fixture':True}} for i,u in enumerate(uids)]})
        return out

    def test_actual_transition_preserves_raw_resets_SGD_EMA_without_public_reload(self):
        torch=self.torch;backend=self.backend();backend.step_update(self.batches(),update=1);backend.step_update(self.batches(),update=2)
        raw={k:v.clone() for k,v in backend.model.state_dict().items()};rng=torch.get_rng_state().clone()
        self.assertTrue(backend.optimizer.state)
        with patch('futsalmot.runtime_backend.load_local_yolo11_detect', side_effect=AssertionError('不重新加载公开权重')):
            next_backend=formal.transition_backend(backend,3)
        self.assertIs(next_backend.network,backend.network);self.assertEqual(next_backend.stage_update,0)
        self.assertEqual(next_backend.optimizer.state,{});self.assertEqual(next_backend.ema_state['updates'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in raw.items():
            self.assertTrue(torch.equal(v,next_backend.model.state_dict()[k]));self.assertTrue(torch.equal(v,next_backend.ema_state['state_dict'][k]))
        self.assertEqual(next_backend.step_update(self.batches(),update=1)['lr'],.001)

    def test_actual_DataLoader_generator_initial_midphase_and_new_phase_resume(self):
        torch=self.torch
        from torch.utils.data import DataLoader, TensorDataset
        def loader(gen):return DataLoader(TensorDataset(torch.zeros(16,1)),batch_size=2,num_workers=0,generator=gen)
        gen=torch.Generator().manual_seed(170017);initial=gen.get_state().clone()
        first=formal.make_phase_iterator(loader(gen),gen,iteration_already_started=False)
        next(first);started=gen.get_state().clone();self.assertFalse(torch.equal(initial,started))
        # initial checkpoint在首次iter之前；恢复后应正常消耗一次base-seed。
        resumed=torch.Generator();resumed.set_state(initial)
        formal.make_phase_iterator(loader(resumed),resumed,iteration_already_started=False)
        self.assertTrue(torch.equal(resumed.get_state(),started))
        # 中途（也含iter已创建但尚未step的USER_STOP）重建iterator不得再次推进。
        resumed.set_state(started)
        formal.make_phase_iterator(loader(resumed),resumed,iteration_already_started=True)
        self.assertTrue(torch.equal(resumed.get_state(),started))
        # 第一阶段末续入全新阶段，与原连续执行均正常再消耗一次。
        formal.make_phase_iterator(loader(gen),gen,iteration_already_started=False)
        second=gen.get_state().clone();self.assertFalse(torch.equal(second,started))
        resumed.set_state(started)
        formal.make_phase_iterator(loader(resumed),resumed,iteration_already_started=False)
        self.assertTrue(torch.equal(resumed.get_state(),second))

    def test_boundary_zero_full_snapshot_global_counts_and_restore(self):
        torch=self.torch
        from futsalmot.records import save_checkpoint, load_checkpoint
        from futsalmot.runtime_backend import NativeBBoxBackend
        from futsalmot.detectors import NativeDetectionNetwork
        panel,synth=pools();samplers=formal.make_samplers(panel,synth,17)
        for _ in range(2):
            w=samplers['S'].reserve(16);samplers['S'].commit(w['token'])
        first=self.backend();first.step_update(self.batches(),update=1);first.step_update(self.batches(),update=2)
        final=formal.transition_backend(first,3);gen=torch.Generator().manual_seed(17)
        bundle=final.snapshot(samplers=samplers,source_bindings={'fixture':'formal_boundary'},committed_updates=2,
            cumulative_exposures=formal.exposure_counts(samplers),extra_state={'loader_generator_state':gen.get_state().clone(),'run_scope':formal.SCHEMA,'global_update':2,'formal_phase':'final'})
        path=Path(self.temp.name)/'boundary.pt';receipt=save_checkpoint(path,bundle);loaded=load_checkpoint(receipt,expected_bindings={'fixture':'formal_boundary'})
        network=NativeDetectionNetwork(deepcopy(final.model),initialization_binding=deepcopy(final.network.initialization_binding))
        fresh=NativeBBoxBackend(network,stage_updates=3,final_stage=True,microbatch_size=2,input_size=64)
        restored_samplers=formal.make_samplers(panel,synth,17)
        restored=fresh.restore_bundle(loaded,samplers=restored_samplers,expected_bindings={'fixture':'formal_boundary'})
        self.assertEqual((restored['committed_updates'],restored['phase_updates']), (2,0))
        self.assertEqual(formal.exposure_counts(restored_samplers), {'R':0,'S':32})
        self.assertTrue(torch.equal(restored['extra_state']['loader_generator_state'],gen.get_state()))
        self.assertEqual(formal.phase_requests('S->R',restored_samplers,start=2,end=5,u_s=2,u_r=3),
            formal.phase_requests('R',formal.make_samplers(panel,synth,17),start=0,end=3,u_s=2,u_r=3))


if __name__=='__main__':unittest.main()
