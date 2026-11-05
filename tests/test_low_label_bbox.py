"""N525实际筛池/全1500采样/奇数混合完整恢复；只CPU不真实数据训练。"""
from copy import deepcopy
from pathlib import Path
import ast,hashlib,inspect,json,tempfile,unittest
from unittest.mock import patch
import torch
from futsalmot import low_label_bbox as m,budget1500 as original
from futsalmot.records import save_checkpoint,load_checkpoint,capture_rng_state,restore_rng_state
from futsalmot.restore import _equal
from test_budget1500 import TinyBackend

def source_fixture():
    uids=[f'detect:soccernet_tracking:C{c:02d}:{i:06d}' for c in range(14) for i in range(1,151)]
    full={'train_uids':uids,'train_catalogue':{u:{'domain':'R','index_path':'D:/fixture/index.json','split':'train'} for u in uids},'val_uids':['val'+str(i) for i in range(750)],'val_catalogue':{'val'+str(i):{} for i in range(750)},'val_ground_truth':{'val'+str(i):{'fixture':True} for i in range(750)},'statistics':{'train':{'images':2100},'val':{'images':750}},'binding':{'ordered_train_pool_sha256':m._pool_sha(uids),'ordered_val_pool_sha256':'b'*64,'source':'CPU_fixture'}}
    blocks=[];chosen=[]
    for c in range(14):
        selected_count=8 if c in (0,1,4,5,9,10,11) else 7
        for k in range(30):
            row={'block_id':f'C{c:02d}:{k:02d}','match_id':'M'+str(0 if c<4 else 1 if c<9 else 2),'clip_uid':f'C{c:02d}','selected':k<selected_count,'selected_50':k<15,'ordered_uids':uids[c*150+k*5:c*150+k*5+5]};blocks.append(row)
            if row['selected']:chosen+=row['ordered_uids']
    d={'schema_version':m.SUBSET_SCHEMA,'status':'FROZEN','full_N_R':2100,'N_R':525,'real_fraction':.25,'source_binding':deepcopy(full['binding']),'full_ordered_train_pool_sha256':m._pool_sha(uids),'ordered_train_uids':chosen,'ordered_train_pool_sha256':m._pool_sha(chosen),'selection_policy':{'algorithm':'CPU_fixture_complete_five_UID_blocks','uses_validation_scores':False,'uses_test':False},'group_blocks':blocks}
    return full,d

def write_ref(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');return {'path':str(path.absolute()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def cfg(reference,strategy):return m.protocol(strategy,subset_manifest_reference=reference,plan_sha256='a'*64,input_cache={'path':'D:/cache','manifest_sha256':'b'*64},baseline_R_status_reference={'path':str(m.BASELINE_STATUS),'sha256':'c'*64},additional_space_bound_bytes=1,raw_upper_per_image_bytes=1,checkpoint_bytes=1,admission_sha256='d'*64)

class LowLabelCPU(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng_state();self.guards=[patch.object(torch.cuda,n,side_effect=AssertionError('CPU only')) for n in ('init','_lazy_init','is_available')]
        for g in self.guards:g.start()
        self.tmp=tempfile.TemporaryDirectory(prefix='low-label-CPU-');self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'subset.json';self.full,self.document=source_fixture();self.ref=write_ref(self.path,self.document)
    def tearDown(self):
        restore_rng_state(self.rng)
        for g in self.guards:g.stop()
    def test_actual_filter_removes1575_and_preserves_full_binding_val_and_original(self):
        before=deepcopy(self.full);selected=m.filter_train_panel(self.full,self.ref)
        self.assertEqual(len(selected['train_uids']),525);self.assertEqual(len(selected['train_catalogue']),525);self.assertEqual(self.full,before)
        self.assertEqual(set(self.full['train_uids'])-set(selected['train_catalogue']),set(self.full['train_uids'])-set(self.document['ordered_train_uids']))
        for key in ('binding','val_uids','val_catalogue','val_ground_truth'):self.assertIs(selected[key],self.full[key])
        self.assertEqual(m.subset_binding(self.full,self.ref)['subset_manifest_reference'],self.ref)
    def test_full1500_real_requests_never_use_unselected_UID_both_arms(self):
        selected=m.filter_train_panel(self.full,self.ref);synthetic={'ordered_uids':['s'+str(i) for i in range(31)]};chosen=set(selected['train_uids']);excluded=set(self.full['train_uids'])-chosen
        for strategy in ('R','R+S'):
            samplers=m.make_samplers(selected,synthetic);requests=m.phase_requests(strategy,samplers,start=0,end=1500);self.assertEqual(len(requests),24000)
            for step in range(1500):
                windows=m.reserve_windows(strategy,samplers);actual=m.window_requests(strategy,windows);self.assertEqual(actual,requests[step*16:(step+1)*16]);m.commit_windows(strategy,samplers,windows)
            real=[x['uid'] for x in requests if x['domain']=='R'];self.assertEqual(set(real),chosen);self.assertFalse(set(real)&excluded);self.assertEqual(len(real),24000 if strategy=='R' else 12000)
            self.assertEqual(m.exposure_counts(samplers),m.exposure_plan(strategy,1500));self.assertEqual(samplers['R_extra'].committed_draws,0)
        with self.assertRaises(ValueError):m.make_samplers(self.full,synthetic)
    def test_subset_hash_source_group_and_uid_tampering_rejected(self):
        self.path.write_bytes(self.path.read_bytes()+b' ')
        with self.assertRaises(ValueError):m.load_subset(self.ref,self.full)
        mutations=[]
        def duplicate(d):d['ordered_train_uids'][0]=d['ordered_train_uids'][1]
        def source(d):d['source_binding']['source']='other'
        def partial(d):d['group_blocks'][0]['ordered_uids'].pop()
        def selected(d):d['group_blocks'][-1]['selected']=True
        def val(d):d['ordered_train_uids'][0]='val0'
        def method(d):d['selection_policy']['uses_validation_scores']=True
        def nested(d):d['group_blocks'][0]['selected_50']=False
        for mutation in (duplicate,source,partial,selected,val,method,nested):
            value=deepcopy(self.document);mutation(value);ref=write_ref(self.path,value)
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError):m.load_subset(ref,self.full)
    def test_protocol_subset_is_bound_identically_in_both_arms(self):
        rows=[cfg(self.ref,s) for s in ('R','R+S')]
        for row in rows:
            self.assertEqual((row['N_R'],row['full_N_R'],row['real_fraction']),(525,2100,.25));self.assertEqual(row['subset_manifest_reference'],self.ref);self.assertFalse(row['unselected_real_images_allowed']);self.assertEqual(row['updates'],1500)
            m.check_protocol({**row,'control_group':'D:/fixture'})
            for key,value in [('N_R',2100),('full_N_R',525),('checkpoint_every',500),('unselected_real_images_allowed',True)]:
                with self.assertRaises(ValueError):m.check_protocol({**row,'control_group':'D:/fixture',key:value})
        self.assertEqual(rows[0]['subset_manifest_reference'],rows[1]['subset_manifest_reference'])
    def test_run_calls_real_source_verifier_then_filter_with_bound_subset(self):
        from futsalmot import formal
        config=cfg(self.ref,'R');config['control_group']='D:/fixture';bindings={'real_panel':self.full['binding'],**m.subset_binding(self.full,self.ref)};table={'jobs':[config],'source_bindings':bindings};synth={'ordered_uids':['s1']}
        with patch.object(formal,'_verify_bindings',return_value=(self.full,synth)) as check:selected,actual=m._verify_low_label_bindings(Path('D:/fixture'),table)
        check.assert_called_once();self.assertEqual(len(selected['train_uids']),525);self.assertIs(actual,synth)
        table['source_bindings'].pop('subset_manifest_reference')
        with patch.object(formal,'_verify_bindings',return_value=(self.full,synth)),self.assertRaises(ValueError):m._verify_low_label_bindings(Path('D:/fixture'),table)
    def test_numeric_loop_and_backend_snapshot_code_are_original_except_filter(self):
        inverse=m.RUN_LOOP_SOURCE.replace('panel,synthetic=_verify_low_label_bindings(group,table)','panel,synthetic=formal._verify_bindings(group,table)')
        self.assertEqual(ast.dump(ast.parse(inverse),include_attributes=False),ast.dump(ast.parse(inspect.getsource(original.run_job)),include_attributes=False))
        self.assertIs(m.make_backend,original.make_backend);self.assertIs(m.Budget1500Backend,original.Budget1500Backend);self.assertIs(m.schedule_1500,original.schedule_1500)
        for name in ('snapshot','validate_snapshot','restore'):self.assertIs(getattr(m,name).__code__,getattr(original,name).__code__)
        self.assertEqual(original.SCHEMA,'bbox_budget1500_v1');self.assertEqual(m.snapshot.__globals__['SCHEMA'],m.SCHEMA)
        hashes=m.execution_code();self.assertEqual(len(hashes),42);self.assertEqual(hashes['budget1500.py'],m.BASE_MODULE_SHA);self.assertIn('low_label_bbox.py',hashes);self.assertNotIn('segmentation_backend.py',hashes)
    def test_odd_mixed_boundary_full_disk_restore_stays_exact_with_subset_binding(self):
        selected=m.filter_train_panel(self.full,self.ref);synthetic={'ordered_uids':['s'+str(i) for i in range(31)]};bindings={'full_real_panel':self.full['binding'],**m.subset_binding(self.full,self.ref)}
        for strategy in ('R','R+S'):
            live=m.make_samplers(selected,synthetic);backend=TinyBackend();backend.step();m.commit_windows(strategy,live,m.reserve_windows(strategy,live))
            state=m.snapshot(backend,strategy=strategy,samplers=live,source_bindings=bindings,committed_updates=1,extra_state={'loader_generator_state':torch.Generator().manual_seed(17).get_state(),'loader_iteration_started':True,'run_scope':m.SCHEMA})
            with tempfile.TemporaryDirectory() as d:receipt=save_checkpoint(Path(d)/'fixture.pt',state);loaded=load_checkpoint(receipt,expected_bindings=bindings)
            self.assertFalse(Path(receipt.path).exists());wanted=backend.step();grads=[p.grad.clone() for p in backend.model.parameters()];rng=capture_rng_state();fresh=TinyBackend();other=m.make_samplers(selected,synthetic);m.restore(fresh,loaded,strategy=strategy,samplers=other,expected_bindings=bindings,expected_update=1)
            self.assertEqual(wanted,fresh.step());self.assertTrue(_equal(backend.model.state_dict(),fresh.model.state_dict()));self.assertTrue(_equal(backend.optimizer.state_dict(),fresh.optimizer.state_dict()));self.assertTrue(_equal(backend.ema_state,fresh.ema_state));self.assertTrue(_equal(rng,capture_rng_state()));self.assertTrue(all(torch.equal(g,p.grad) for g,p in zip(grads,fresh.model.parameters())))
            self.assertEqual(other['R']._template[0],tuple(selected['train_uids']));self.assertEqual(loaded['source_bindings']['subset_manifest_reference'],self.ref)

if __name__=='__main__':unittest.main()
