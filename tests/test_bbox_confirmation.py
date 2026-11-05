"""确认集授权/读取/EMA选择的CPU虚构夹具；不读真实test，不模型/GPU。"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,hashlib,inspect,json,os,pickle,unittest
import numpy as np
from PIL import Image
import torch
from futsalmot import bbox_confirmation as m,adapters,runner,train_data
from futsalmot.records import _binding_digest


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=True).encode();path.write_bytes(raw)
    return {'path':str(path.absolute()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def file_ref(path):
    return {'path':str(path.absolute()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

def fixture(root):
    source=root/'source';source.mkdir();target,ignore=adapters._POLICIES['soccernet_tracking'];seqs=[];matches=[];match_specs=[]
    for game,clips in [(8,[133,137,141,145,149]),(9,[153,157,161,165,169])]:
        match=f'match:soccernet-tracking:game:{game}';matches.append({'match_uid':match,'split':'test'});uids=[];files=[]
        for clip in clips:
            name=f'SNMOT-{clip:03d}';base=source/'sequences'/name;(base/'gt').mkdir(parents=True)
            gt=base/'gt/gt.txt';gt.write_text('1,1,0,5,20,20,1,-1,-1,-1\n1,2,20,5,10,20,1,-1,-1,-1\n1,3,50,5,10,20,1,-1,-1,-1\n',encoding='utf8')
            (base/'gameinfo.ini').write_text(f'[Sequence]\nname={name}\ngameID={game}\ntrackletID_1= player team left;1\ntrackletID_2= goalkeepers team left;2\ntrackletID_3= referee;main\n',encoding='utf8');(base/'seqinfo.ini').write_text('[Sequence]\nseqLength=750\n',encoding='utf8')
            frames=list(range(1,750,5));row={'source_id':'soccernet_tracking','clip_uid':'soccernet_tracking:'+name,'match_uid':match,'split':'test','complete':True,'selected':True,'official_frame_count':750,'detection_frames':frames,'width':100,'height':50,'sequence_path':'sequences/'+name,'files':{'gt':'gt/gt.txt','gameinfo':'gameinfo.ini','seqinfo':'seqinfo.ini'},'frame_uid_pattern':'soccernet_tracking:'+name+':{frame_number:06d}','detection_sample_uid_pattern':'detect:{frame_uid}','image_file_pattern':'img1/{frame_number:06d}.jpg','track_namespace':name+':original_ID','target_policy':target,'ignore_policy':ignore,'gt_sha256':file_ref(gt)['sha256']};seqs.append(row)
            uids.extend('detect:soccernet_tracking:'+name+f':{frame:06d}' for frame in frames);files.extend(file_ref(base/row['files'][k]) for k in ('gt','gameinfo','seqinfo'))
        match_specs.append({'match_uid':match,'ordered_uids':uids,'ordered_uid_sha256':m._pool_sha(uids),'files':files})
    # 原train/val reader必须仍拒test；只给CPU一张必要JPEG。
    first=source/'sequences/SNMOT-133/img1/000001.jpg';first.parent.mkdir();Image.fromarray(np.full((50,100,3),[10,30,210],dtype=np.uint8)).save(first)
    index={'schema_version':'canonical_real_dataset_v1','status':'READY_SOURCE_LOCAL','source_panel_id':'soccernet_local','source':{'source_id':'soccernet_tracking','target_semantics':target,'ignore_policy':ignore},'matches':matches,'sequences':seqs,'limitations':['CPU_fixture'],'joint_use_allowed':False,'cross_source_disjoint_certified':False};index_ref=put(source/'dataset.json',index)
    plan=put(root/'plan.json',{'CPU_fixture':True});models=[];done=[]
    for seed in (17,29,43):
        for fraction in (1.,.5,.25,.1):
            for strategy in ('R','R+S'):
                key=f'{seed}-{fraction}-{strategy}';base=root/'models'/key;base.mkdir(parents=True)
                cfg={'seed':seed,'strategy':strategy,'N_R':int(fraction*2100),'updates':1500,'model':'yolo11s','input_size':960,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'job_id':key};source_bindings={'CPU_fixture':key,'source_schema':'any_original_41_or42_allowed'};bindings={**source_bindings,'protocol':cfg}
                table=put(base/'job_table.json',{'jobs':[cfg],'source_bindings':source_bindings})
                status=put(base/'status.json',{'state':'COMPLETE','update':1500,'phase_update':1500,'exposures':{'R':24000,'S':0} if strategy=='R' else {'R':12000,'S':12000},'terminal':{'tag':'step001500_final_final'},'directory':str(base)})
                control=put(base/'control.json',{'state':'SEALED','scope':'control_snapshot_only'});(base/'checkpoints').mkdir();cp=base/'checkpoints/001500_final_001500_final.pt';cp.write_bytes(('CPU dummy never-loaded '+key).encode());receipt={**file_ref(cp),'source_bindings_sha256':_binding_digest(bindings),'schema_version':'futsalmot_checkpoint_v1','producer':'futsalmot.records/v1'}
                idx=base/'checkpoints.jsonl';idx.write_text(json.dumps({'update':1500,'phase':'final','phase_update':1500,'tag':'final','receipt':receipt})+'\n',encoding='utf8')
                row={'model_id':key,'seed':seed,'real_fraction':fraction,'strategy':strategy,'status_reference':status,'control_manifest_reference':control,'job_table_reference':table,'checkpoint_index_reference':file_ref(idx),'checkpoint_receipt':receipt,'expected_source_bindings':bindings};models.append(row);done.append({**{k:deepcopy(row[k]) for k in ('model_id','status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference','checkpoint_receipt')},'owned_exit_verified':True,'GPU_lock_retired':True})
    spec={'schema_version':m.SCHEMA,'state':'FROZEN','inference_only':True,'execution':deepcopy(m.EXECUTION),'evaluation_count':48,'deadline_utc':'2026-11-07T23:55:00Z','plan_reference':plan,'index_reference':index_ref,'evaluation_code_sha256':m.execution_code(),'models':models,'matches':match_specs}
    return put(root/'confirmation.json',spec),put(root/'completion.json',{'schema_version':m.COMPLETION_SCHEMA,'state':'COMPLETE','models':done}),plan,spec


class ConfirmationCPU(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(prefix='confirmation-CPU-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.ref,self.done,self.plan,self.spec=fixture(self.root)
        self.guards=[patch.object(torch.cuda,x,side_effect=AssertionError('CPU tests forbid CUDA')) for x in ('_lazy_init','init','is_available')]
        for guard in self.guards:guard.start();self.addCleanup(guard.stop)
        env=patch.dict(os.environ,{'YOLO_OFFLINE':'true','YOLO_CONFIG_DIR':str(self.root/'yolo'),'MPLCONFIGDIR':str(self.root/'mpl')});env.start();self.addCleanup(env.stop)
    def permit(self):return m.authorize_confirmation(self.ref,completion_reference=self.done,plan_reference=self.plan)
    def test_spec_strict24x2_scope_no_data_access(self):
        self.assertEqual(len(m.validate_spec(self.spec)['models']),24)
        for field,value in [('evaluation_count',24),('models',self.spec['models'][:-1]),('inference_only',False)]:
            d=deepcopy(self.spec);d[field]=value
            with self.assertRaises(ValueError):m.validate_spec(d)
        d=deepcopy(self.spec);d['models'][0]['seed']=23
        with self.assertRaises(ValueError):m.validate_spec(d)
        d=deepcopy(self.spec);d['matches'][0]['match_uid']='match:soccernet-tracking:game:7'
        with self.assertRaises(ValueError):m.validate_spec(d)
    def test_no_test_GT_open_before_whole24_complete(self):
        completion=json.loads(Path(self.done['path']).read_bytes());completion['state']='RUNNING';done=put(Path(self.done['path']),completion);original=Path.open;opened=[]
        def guarded(path,*args,**kwargs):
            if path.name=='gt.txt':opened.append(str(path));raise AssertionError('testGT before24closed')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded),self.assertRaises(ValueError):m.authorize_confirmation(self.ref,completion_reference=done,plan_reference=self.plan)
        self.assertEqual(opened,[])
    def test_reader_only_authorized_test_and_original_reader_unchanged(self):
        p=self.permit();index=self.spec['index_reference']['path'];uid=self.spec['matches'][0]['ordered_uids'][0];other=self.spec['matches'][1]['ordered_uids'][0]
        original=adapters.RealBBoxReader(index)
        with self.assertRaises(ValueError):original.read(uid,split='test')
        reader=m.ConfirmationReader(index,authorization=p,match_uid=m.MATCHES[0]);sample=reader.read(uid)
        self.assertEqual(sample['split'],'test');self.assertEqual(sample['domain'],'R');self.assertEqual([x['track_id'] for x in sample['targets']],[1,2]);self.assertEqual(sample['ignore_targets'][0]['ignore_reason'],'non_target_role')
        with self.assertRaises(ValueError):reader.read(other)
        with self.assertRaises(ValueError):reader.read(uid,split='val')
        fresh=pickle.loads(pickle.dumps(reader));self.assertEqual(fresh.read(uid),sample)
    def test_confirmation_dataset_preserves_test_provenance_and_original_val_geometry(self):
        p=self.permit();match=self.spec['matches'][0];index=self.spec['index_reference']['path'];uids=match['ordered_uids'];catalogue={u:{'domain':'R','index_path':index,'split':'test'} for u in uids};requests=[{'uid':u,'domain':'R','epoch':0,'exposure_index':i,'augmentation_seed':0} for i,u in enumerate(uids)]
        ds=m.ConfirmationRequestPlanDataset(catalogue,requests,authorization=p,match_uid=m.MATCHES[0]);self.assertIsInstance(ds,m._data.RequestPlanDataset);a=ds[0]
        self.assertEqual(a['source_reference']['split'],'test');self.assertEqual(a['geometry']['training_input']['view'],'val');self.assertFalse(a['random_augmentation_implemented']);self.assertEqual(tuple(a['img'].shape),(3,960,960))
        prepared=m._dataset.RequestBBoxDataset.__getitem__(ds.dataset,requests[0]);val=deepcopy(prepared);val['source_reference']['split']='val';b=train_data.prepare_training_sample(val,view='val')
        self.assertTrue(torch.equal(a['img'],b['img']));self.assertTrue(torch.equal(a['bboxes'],b['bboxes']));self.assertTrue(torch.equal(a['ignore_bboxes'],b['ignore_bboxes']));self.assertEqual(a['geometry'],b['geometry'])
        wrong=deepcopy(requests);wrong[0]['augmentation_seed']=1
        with self.assertRaises(ValueError):m.ConfirmationRequestPlanDataset(catalogue,wrong,authorization=p,match_uid=m.MATCHES[0])
        with self.assertRaises(ValueError):m.ConfirmationRequestPlanDataset(catalogue,requests,authorization=p,match_uid=m.MATCHES[0],view='train')
    def test_panel_GT_uses_original_role_ignore_and750_UID(self):
        p=self.permit();panel=m.confirmation_panel(p,m.MATCHES[0]);self.assertEqual(len(panel['val_uids']),750);self.assertEqual(panel['source_split'],'test');first=panel['val_ground_truth'][panel['val_uids'][0]]
        self.assertEqual(first['positive_gt'],[[0.,5.,20.,25.],[20.,5.,30.,25.]]);self.assertEqual(first['ignore'],[]);self.assertEqual(first['known_non_target_auxiliary_ignore'],[[50.,5.,60.,25.]]);self.assertEqual(panel['val_catalogue'][panel['val_uids'][0]]['split'],'test')
    def test_bound_GT_and_status_tampering_rejected(self):
        p=self.permit();path=Path(self.spec['models'][0]['status_reference']['path']);path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):p.check_live()
        path=Path(self.spec['matches'][0]['files'][0]['path']);path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(ValueError):self.permit()
    def test_terminal_EMA_only_and_original_trusted_loader_arguments(self):
        p=self.permit();model=self.spec['models'][0];state={'numeric_mode':'fp32','committed_updates':1500,'phase_updates':1500,'phase':'final','raw_model':{'weight':torch.tensor([1.])},'cumulative_exposures':{'R':24000,'S':0},'sampler':{key:{'sampler':{'seed':17,'domain':key,'draws':24000 if key=='R' else 0}} for key in ('R','S','R_extra')},'ema':{'updates':1500,'state_dict':{'weight':torch.tensor([2.])}}}
        with patch.object(m,'load_checkpoint',return_value=state) as load:
            result=m.load_terminal_ema(p,model['model_id']);self.assertEqual(result['state_dict']['weight'].item(),2.);self.assertNotIn('raw_model',result);self.assertEqual(load.call_args.kwargs['expected_bindings'],model['expected_source_bindings']);self.assertEqual(load.call_args.args[0].sha256,model['checkpoint_receipt']['sha256'])
        state['ema']['updates']=1499
        with patch.object(m,'load_checkpoint',return_value=state),self.assertRaises(ValueError):m.load_terminal_ema(p,model['model_id'])
    def test_model_key_rejects_another_training_seed_before_GT(self):
        model=self.spec['models'][0];path=Path(model['job_table_reference']['path']);table=json.loads(path.read_bytes());table['jobs'][0]['seed']=29;ref=put(path,table);model['job_table_reference']=ref
        completion=json.loads(Path(self.done['path']).read_bytes());completion['models'][0]['job_table_reference']=ref;done=put(Path(self.done['path']),completion);spec_ref=put(Path(self.ref['path']),self.spec)
        original=Path.open;opened=[]
        def guard(path,*args,**kwargs):
            if path.name=='gt.txt':opened.append(str(path));raise AssertionError('GT before registry matches')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard),self.assertRaises(ValueError):m.authorize_confirmation(spec_ref,completion_reference=done,plan_reference=self.plan)
        self.assertEqual(opened,[])
    def test_terminal_receipt_requires_exact_own_leaf_even_when_index_agrees(self):
        model=self.spec['models'][0];original_cp=Path(model['checkpoint_receipt']['path']);wrong=original_cp.parent.parent/'unowned_final.pt';wrong.write_bytes(original_cp.read_bytes());model['checkpoint_receipt']['path']=str(wrong)
        idx=Path(model['checkpoint_index_reference']['path']);idx.write_text(json.dumps({'update':1500,'phase':'final','phase_update':1500,'tag':'final','receipt':model['checkpoint_receipt']})+'\n',encoding='utf8');model['checkpoint_index_reference']=file_ref(idx)
        completion=json.loads(Path(self.done['path']).read_bytes());completion['models'][0]['checkpoint_receipt']=deepcopy(model['checkpoint_receipt']);completion['models'][0]['checkpoint_index_reference']=deepcopy(model['checkpoint_index_reference']);done=put(Path(self.done['path']),completion);spec_ref=put(Path(self.ref['path']),self.spec)
        with self.assertRaises(ValueError):m.authorize_confirmation(spec_ref,completion_reference=done,plan_reference=self.plan)
    def test_authorization_requires_factory_and_checkpoint_actual_sampler_seed(self):
        with self.assertRaises(ValueError):m.ConfirmationAuthorization(b'{}',(),())
        p=self.permit();model=self.spec['models'][0];state={'numeric_mode':'fp32','committed_updates':1500,'phase_updates':1500,'phase':'final','raw_model':{'weight':torch.tensor([1.])},'cumulative_exposures':{'R':24000,'S':0},'sampler':{key:{'sampler':{'seed':29,'domain':key,'draws':24000 if key=='R' else 0}} for key in ('R','S','R_extra')},'ema':{'updates':1500,'state_dict':{'weight':torch.tensor([2.])}}}
        with patch.object(m,'load_checkpoint',return_value=state),self.assertRaises(ValueError):m.load_terminal_ema(p,model['model_id'])
    def test_confirmation_actual_collate_and_CPU_preparation_preserve_test_identity(self):
        from futsalmot import batches,training
        permit=self.permit();match=self.spec['matches'][0];index=self.spec['index_reference']['path'];uids=match['ordered_uids'];catalogue={u:{'domain':'R','index_path':index,'split':'test'} for u in uids};requests=[{'uid':u,'domain':'R','epoch':0,'exposure_index':i,'augmentation_seed':0} for i,u in enumerate(uids)]
        ds=m.ConfirmationRequestPlanDataset(catalogue,requests,authorization=permit,match_uid=m.MATCHES[0]);sample=ds[0]
        with self.assertRaisesRegex(ValueError,'split非法'):batches.collate_bbox_samples([sample])
        loader=m.make_confirmation_loader(ds,batch_size=1,num_workers=0,generator=torch.Generator().manual_seed(170001))
        actual=next(iter(loader));self.assertEqual(actual['sample_lineage'][0]['source_reference']['split'],'test')
        val=deepcopy(sample);val['source_reference']['split']='val';expected=batches.collate_bbox_samples([val])
        for key in ('img','cls','bboxes','batch_idx','ignore_bboxes','ignore_batch_idx'):self.assertTrue(torch.equal(actual[key],expected[key]),key)
        a=training.prepare_detection_batch(actual,device='cpu');b=training.prepare_detection_batch(expected,device='cpu');self.assertTrue(torch.equal(a['img'],b['img']));self.assertEqual(a['sample_lineage'][0]['source_reference']['split'],'test')
        collator=m.ConfirmationCollator(permit,m.MATCHES[0])
        with self.assertRaises(ValueError):collator([val])
        wrong=deepcopy(sample);wrong['uid']=self.spec['matches'][1]['ordered_uids'][0]
        with self.assertRaises(ValueError):collator([wrong])
        self.assertTrue(m.COLLATE_BINDING['inverse_AST_exact']);self.assertTrue(m.LINEAGE_BINDING['inverse_AST_exact'])

    def test_confirmation_collator_survives_actual_two_spawn_workers_and_shutdown(self):
        from torch.utils.data import DataLoader
        permit=self.permit();match=self.spec['matches'][0];index=self.spec['index_reference']['path'];uids=match['ordered_uids'];catalogue={u:{'domain':'R','index_path':index,'split':'test'} for u in uids};requests=[{'uid':u,'domain':'R','epoch':0,'exposure_index':i,'augmentation_seed':0} for i,u in enumerate(uids)]
        # 只有前40张虚构JPEG；不访问真实媒体或模型，不遍历750张。
        source=Path(index).parent
        for uid in uids[:40]:
            clip,frame=uid.split(':')[-2:];path=source/'sequences'/clip/'img1'/(frame+'.jpg');path.parent.mkdir(exist_ok=True);Image.fromarray(np.full((50,100,3),[10,30,210],dtype=np.uint8)).save(path)
        ds=m.ConfirmationRequestPlanDataset(catalogue,requests,authorization=permit,match_uid=m.MATCHES[0]);original=m.make_confirmation_loader(ds,batch_size=8,num_workers=2,generator=torch.Generator().manual_seed(170001))
        loader=DataLoader(ds,batch_size=8,num_workers=2,collate_fn=original.collate_fn,worker_init_fn=original.worker_init_fn,prefetch_factor=1,persistent_workers=True,multiprocessing_context='spawn')
        iterator=iter(loader)
        try:
            batch=next(iterator);self.assertEqual(batch['uids'],uids[:8]);self.assertTrue(all(x['source_reference']['split']=='test' for x in batch['sample_lineage']))
        finally:iterator._shutdown_workers()
        self.assertTrue(all(not w.is_alive() for w in iterator._workers))

    def test_all_derivations_inverse_AST_and_original_functions_unmodified(self):
        self.assertTrue(all(x['inverse_AST_exact'] for x in (m.READER_INIT_BINDING,m.READER_READ_BINDING,m.PREPROCESS_BINDING)))
        before=inspect.getsource(runner.evaluate);p=self.permit();candidate=m.make_confirmation_evaluator(p,m.MATCHES[0]);self.assertTrue(candidate['mechanical_binding']['inverse_AST_exact']);self.assertEqual(inspect.getsource(runner.evaluate),before);self.assertFalse(candidate['real_GPU_verified']);self.assertFalse(candidate['lifecycle_launcher_implemented'])
        self.assertIn('if split not in ("train", "val"):',inspect.getsource(adapters.RealBBoxReader.read));self.assertIn('split!=(view if domain=="R"',inspect.getsource(train_data.prepare_training_sample))

if __name__=='__main__':unittest.main()
