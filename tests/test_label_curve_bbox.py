"""50%/10%真实筛池、四臂完整采样与磁盘恢复；只CPU。"""
from copy import deepcopy
from pathlib import Path
import ast,hashlib,inspect,json,tempfile,unittest
from unittest.mock import patch
import torch
from futsalmot import label_curve_bbox as m,budget1500 as original
from futsalmot.records import save_checkpoint,load_checkpoint,capture_rng_state,restore_rng_state
from futsalmot.restore import _equal
from test_budget1500 import TinyBackend


def source_fixture(fraction):
    n,_,quotas=m.label_spec(fraction)
    uids=[f'detect:soccernet_tracking:C{c:02d}:{i:06d}' for c in range(14) for i in range(1,151)]
    full={'train_uids':uids,'train_catalogue':{u:{'domain':'R','index_path':'D:/fixture/index.json','split':'train'} for u in uids},'val_uids':['val'+str(i) for i in range(750)],'val_catalogue':{'val'+str(i):{} for i in range(750)},'val_ground_truth':{'val'+str(i):{'fixture':True} for i in range(750)},'statistics':{'train':{'images':2100},'val':{'images':750}},'binding':{'ordered_train_pool_sha256':m._pool_sha(uids),'ordered_val_pool_sha256':'b'*64,'source':'CPU_fixture'}}
    blocks=[];chosen=[]
    for c in range(14):
        quota=(8 if c<7 else 7) if fraction==.25 else quotas[0]
        for k in range(30):
            row={'block_id':f'C{c:02d}:{k:02d}','match_id':'M'+str(0 if c<4 else 1 if c<9 else 2),'clip_uid':f'C{c:02d}','selected':k<quota,'selected_50':k<15,'ordered_uids':uids[c*150+k*5:c*150+k*5+5]};blocks.append(row)
            if row['selected']:chosen+=row['ordered_uids']
    doc={'schema_version':m.SUBSET_SCHEMA,'status':'FROZEN','full_N_R':2100,'N_R':n,'real_fraction':fraction,'source_binding':deepcopy(full['binding']),'full_ordered_train_pool_sha256':m._pool_sha(uids),'ordered_train_uids':chosen,'ordered_train_pool_sha256':m._pool_sha(chosen),'selection_policy':{'algorithm':'CPU_fixture_complete_five_UID_blocks','uses_validation_scores':False,'uses_test':False},'group_blocks':blocks}
    return full,doc


def write_ref(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');return {'path':str(path.absolute()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def cfg(reference,strategy,fraction,seed=17):return m.protocol(strategy,seed,real_fraction=fraction,subset_manifest_reference=reference,plan_sha256='a'*64,input_cache={'path':'D:/cache','manifest_sha256':'b'*64},baseline_R_status_reference={'path':str(m.BASELINE_STATUS),'sha256':'c'*64},additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)


class LabelCurveCPU(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng_state();self.guards=[patch.object(torch.cuda,n,side_effect=AssertionError('CPU only')) for n in ('init','_lazy_init','is_available')]
        for g in self.guards:g.start()
        self.tmp=tempfile.TemporaryDirectory(prefix='label-curve-CPU-');self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'subset.json'
    def tearDown(self):
        restore_rng_state(self.rng)
        for g in self.guards:g.stop()
    def fixture(self,fraction):
        full,doc=source_fixture(fraction);ref=write_ref(self.path,doc);return full,doc,ref
    def test_actual_filter_two_counts_preserves_full_binding_val_original_and_nested_pools(self):
        selections={}
        for fraction,n in ((.5,1050),(.1,210)):
            full,doc,ref=self.fixture(fraction);before=deepcopy(full);selected=m.filter_train_panel(full,ref)
            self.assertEqual(len(selected['train_uids']),n);self.assertEqual(len(selected['train_catalogue']),n);self.assertEqual(full,before)
            self.assertEqual(len(set(full['train_uids'])-set(selected['train_catalogue'])),2100-n)
            for key in ('binding','val_uids','val_catalogue','val_ground_truth'):self.assertIs(selected[key],full[key])
            self.assertEqual(m.subset_binding(full,ref)['subset_N_R'],n);selections[fraction]=set(selected['train_uids'])
        self.assertTrue(selections[.1]<selections[.5])
    def test_full1500_requests_all_four_arms_never_use_unselected_real_UIDs(self):
        for fraction in (.5,.1):
            full,doc,ref=self.fixture(fraction);selected=m.filter_train_panel(full,ref);synthetic={'ordered_uids':['s'+str(i) for i in range(31)]};chosen=set(selected['train_uids']);excluded=set(full['train_uids'])-chosen
            for strategy in ('R','R+S'):
                with self.subTest(fraction=fraction,strategy=strategy):
                    samplers=m.make_samplers(selected,synthetic);requests=m.phase_requests(strategy,samplers,start=0,end=1500);self.assertEqual(len(requests),24000)
                    for step in range(1500):
                        windows=m.reserve_windows(strategy,samplers);actual=m.window_requests(strategy,windows);self.assertEqual(actual,requests[step*16:(step+1)*16]);m.commit_windows(strategy,samplers,windows)
                    real=[x['uid'] for x in requests if x['domain']=='R'];self.assertEqual(set(real),chosen);self.assertFalse(set(real)&excluded);self.assertEqual(len(real),24000 if strategy=='R' else 12000)
                    self.assertEqual(m.exposure_counts(samplers),m.exposure_plan(strategy,1500));self.assertEqual(samplers['R_extra'].committed_draws,0)
            with self.assertRaises(ValueError):m.make_samplers(full,synthetic)
    def test_subset_hash_source_uid_fraction_and_whole_block_changes_rejected(self):
        for fraction in (.5,.1):
            full,doc,ref=self.fixture(fraction);self.path.write_bytes(self.path.read_bytes()+b' ')
            with self.assertRaises(ValueError):m.load_subset(ref,full)
            def duplicate(d):d['ordered_train_uids'][0]=d['ordered_train_uids'][1]
            def source(d):d['source_binding']['source']='other'
            def partial(d):d['group_blocks'][0]['ordered_uids'].pop()
            def selected(d):d['group_blocks'][-1]['selected']=True
            def val(d):d['ordered_train_uids'][0]='val0'
            def method(d):d['selection_policy']['uses_validation_scores']=True
            def nested(d):d['group_blocks'][0]['selected_50']=False
            def ratio(d):d['real_fraction']=.25
            for mutation in (duplicate,source,partial,selected,val,method,nested,ratio):
                value=deepcopy(doc);mutation(value);changed=write_ref(self.path,value)
                with self.subTest(fraction=fraction,mutation=mutation.__name__),self.assertRaises(ValueError):m.load_subset(changed,full)
    def test_protocol_four_unique_jobs_and_frozen_original_absolute_parameters(self):
        names=set()
        for fraction,n in ((.5,1050),(.1,210)):
            full,doc,ref=self.fixture(fraction)
            for strategy in ('R','R+S'):
                row=cfg(ref,strategy,fraction);names.add(row['job_id']);self.assertEqual((row['N_R'],row['full_N_R'],row['real_fraction']),(n,2100,fraction));self.assertEqual(row['subset_manifest_reference'],ref);self.assertFalse(row['unselected_real_images_allowed']);m.check_protocol({**row,'control_group':'D:/fixture'})
                baseline=original.protocol(strategy,plan_sha256='a'*64,input_cache=row['input_cache'],baseline_R_status_reference=row['baseline_R_status_reference'],additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)
                for key,value in baseline.items():
                    if key not in ('schema_version','job_id','N_R'):self.assertEqual(row[key],value,key)
                for key,value in [('N_R',2100),('real_fraction',.25),('checkpoint_every',500),('unselected_real_images_allowed',True)]:
                    with self.assertRaises(ValueError):m.check_protocol({**row,'control_group':'D:/fixture',key:value})
        self.assertEqual(names,{'label50_R_seed17','label50_R_plus_S_seed17','label10_R_seed17','label10_R_plus_S_seed17'})
        for fraction in (True,0.,.75,.2):
            with self.assertRaises(ValueError):cfg(ref,'R',fraction)
    def test_run_verifies_full_sources_then_actual_filter_and_count_config_match(self):
        from futsalmot import formal
        for fraction in (.5,.1):
            full,doc,ref=self.fixture(fraction);config={**cfg(ref,'R',fraction),'control_group':'D:/fixture'};bindings={'real_panel':full['binding'],**m.subset_binding(full,ref)};table={'jobs':[config],'source_bindings':bindings};synth={'ordered_uids':['s1']}
            with patch.object(formal,'_verify_bindings',return_value=(full,synth)) as check:selected,actual=m._verify_low_label_bindings(Path('D:/fixture'),table)
            check.assert_called_once();self.assertEqual(len(selected['train_uids']),doc['N_R']);self.assertIs(actual,synth)
            table['source_bindings'].pop('subset_manifest_reference')
            with patch.object(formal,'_verify_bindings',return_value=(full,synth)),self.assertRaises(ValueError):m._verify_low_label_bindings(Path('D:/fixture'),table)
            other=.1 if fraction==.5 else .5;table={'jobs':[{**cfg(ref,'R',other),'control_group':'D:/fixture'}],'source_bindings':bindings}
            with patch.object(formal,'_verify_bindings',return_value=(full,synth)),self.assertRaises(ValueError):m._verify_low_label_bindings(Path('D:/fixture'),table)
    def test_loop_backend_and_snapshot_original_except_filter_and_explicit_seed_initialization(self):
        inverse=m.RUN_LOOP_SOURCE.replace(m._new,m._old).replace(m._seed_new,m._seed_old)
        self.assertEqual(ast.dump(ast.parse(inverse),include_attributes=False),ast.dump(ast.parse(inspect.getsource(original.run_job)),include_attributes=False));self.assertIs(m.make_backend,original.make_backend);self.assertIs(m.Budget1500Backend,original.Budget1500Backend);self.assertIs(m.schedule_1500,original.schedule_1500)
        for name in ('snapshot','validate_snapshot','restore'):self.assertIs(getattr(m,name).__code__,getattr(original,name).__code__)
        self.assertEqual(original.SCHEMA,'bbox_budget1500_v1');self.assertEqual(m.snapshot.__globals__['SCHEMA'],m.SCHEMA)
        hashes=m.execution_code();self.assertEqual(len(hashes),42);self.assertEqual(hashes['budget1500.py'],m.BASE_MODULE_SHA);self.assertIn('label_curve_bbox.py',hashes);self.assertNotIn('low_label_bbox.py',hashes);self.assertNotIn('segmentation_backend.py',hashes)
    def test_all_supported_seed_protocols_keep_fixed_pool_and_change_actual_sampler_sequences(self):
        for fraction,n in ((1.,2100),(.5,1050),(.25,525),(.1,210)):
            full,doc,ref=self.fixture(fraction);selected=m.filter_train_panel(full,ref);sequences={}
            self.assertEqual(len(selected['train_uids']),n)
            for seed in (17,29,43):
                for strategy in ('R','R+S'):
                    row=cfg(ref,strategy,fraction,seed);self.assertEqual(row['seed'],seed);self.assertTrue(row['job_id'].endswith(f'_seed{seed}'));self.assertEqual(row['subset_manifest_reference'],ref);m.check_protocol({**row,'control_group':'D:/fixture'})
                samplers=m.make_samplers(selected,{'ordered_uids':['s'+str(i) for i in range(31)]},seed=seed)
                for stream in ('R','S','R_extra'):self.assertEqual(samplers[stream]._template[-1],seed)
                sequences[seed]=m.phase_requests('R+S',samplers,start=0,end=2)
                self.assertEqual(samplers['R']._template[0],tuple(selected['train_uids']))
            self.assertNotEqual(sequences[17],sequences[29]);self.assertNotEqual(sequences[29],sequences[43])
        self.assertIn("runner.seed_all(config['seed'])",m.RUN_LOOP_SOURCE)
        self.assertIn("make_samplers(panel,synthetic,seed=config['seed'])",m.RUN_LOOP_SOURCE)
        self.assertIn("manual_seed(config['seed']*10000+17)",m.RUN_LOOP_SOURCE)
        self.assertNotIn('runner.seed_all(17)',m.RUN_LOOP_SOURCE)
        self.assertNotIn('170017',m.RUN_LOOP_SOURCE)
        for seed in (True,1,18,29.):
            with self.assertRaises(ValueError):cfg(ref,'R',.1,seed)

    def test_new_seed29_and43_odd_mixed_disk_restore_with_identical_fixed210_pool(self):
        full,doc,ref=self.fixture(.1);selected=m.filter_train_panel(full,ref);synthetic={'ordered_uids':['s'+str(i) for i in range(31)]}
        for seed in (29,43):
            bindings={'full_real_panel':full['binding'],**m.subset_binding(full,ref),'training_seed':seed};torch.manual_seed(seed)
            live=m.make_samplers(selected,synthetic,seed);backend=TinyBackend();backend.step();m.commit_windows('R+S',live,m.reserve_windows('R+S',live))
            state=m.snapshot(backend,strategy='R+S',samplers=live,source_bindings=bindings,committed_updates=1,extra_state={'loader_generator_state':torch.Generator().manual_seed(seed*10000+17).get_state(),'loader_iteration_started':True,'run_scope':m.SCHEMA})
            with tempfile.TemporaryDirectory() as d:receipt=save_checkpoint(Path(d)/'fixture.pt',state);loaded=load_checkpoint(receipt,expected_bindings=bindings)
            wanted=backend.step();grads=[p.grad.clone() for p in backend.model.parameters()];rng=capture_rng_state();fresh=TinyBackend();other=m.make_samplers(selected,synthetic,seed);m.restore(fresh,loaded,strategy='R+S',samplers=other,expected_bindings=bindings,expected_update=1)
            self.assertEqual(wanted,fresh.step());self.assertTrue(_equal(backend.model.state_dict(),fresh.model.state_dict()));self.assertTrue(_equal(backend.optimizer.state_dict(),fresh.optimizer.state_dict()));self.assertTrue(_equal(backend.ema_state,fresh.ema_state));self.assertTrue(_equal(rng,capture_rng_state()));self.assertTrue(all(torch.equal(g,p.grad) for g,p in zip(grads,fresh.model.parameters())))
            self.assertEqual(other['R']._template[0],tuple(selected['train_uids']));self.assertEqual(other['R']._template[-1],seed)

    def test_odd_mixed_boundary_full_disk_restore_exact_for_each_subset(self):
        for fraction in (.5,.1):
            full,doc,ref=self.fixture(fraction);selected=m.filter_train_panel(full,ref);synthetic={'ordered_uids':['s'+str(i) for i in range(31)]};bindings={'full_real_panel':full['binding'],**m.subset_binding(full,ref)}
            for strategy in ('R','R+S'):
                live=m.make_samplers(selected,synthetic);backend=TinyBackend();backend.step();m.commit_windows(strategy,live,m.reserve_windows(strategy,live))
                state=m.snapshot(backend,strategy=strategy,samplers=live,source_bindings=bindings,committed_updates=1,extra_state={'loader_generator_state':torch.Generator().manual_seed(17).get_state(),'loader_iteration_started':True,'run_scope':m.SCHEMA})
                with tempfile.TemporaryDirectory() as d:receipt=save_checkpoint(Path(d)/'fixture.pt',state);loaded=load_checkpoint(receipt,expected_bindings=bindings)
                self.assertFalse(Path(receipt.path).exists());wanted=backend.step();grads=[p.grad.clone() for p in backend.model.parameters()];rng=capture_rng_state();fresh=TinyBackend();other=m.make_samplers(selected,synthetic);m.restore(fresh,loaded,strategy=strategy,samplers=other,expected_bindings=bindings,expected_update=1)
                self.assertEqual(wanted,fresh.step());self.assertTrue(_equal(backend.model.state_dict(),fresh.model.state_dict()));self.assertTrue(_equal(backend.optimizer.state_dict(),fresh.optimizer.state_dict()));self.assertTrue(_equal(backend.ema_state,fresh.ema_state));self.assertTrue(_equal(rng,capture_rng_state()));self.assertTrue(all(torch.equal(g,p.grad) for g,p in zip(grads,fresh.model.parameters())))
                self.assertEqual(other['R']._template[0],tuple(selected['train_uids']));self.assertEqual(loaded['source_bindings']['subset_manifest_reference'],ref)


if __name__=='__main__':unittest.main()
