"""保存降频、原数值证据复用及跨组R预算比较的纯CPU机械验证。"""
import unittest,tempfile,json,hashlib
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
from types import SimpleNamespace
from futsalmot import runner
from futsalmot.output_storage import create_output_binding,resolve_output_group,capture_control

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class StorageRunnerTests(unittest.TestCase):
    def test_checkpoint_schedule_preserves_zero_final_and_failure_boundary_is_external(self):
        self.assertEqual([i for i in range(6564) if runner.checkpoint_due(i,6563,500)],[0]+list(range(500,6501,500))+[6563])
        self.assertEqual(sum(runner.checkpoint_due(i,13125,500) for i in range(13126)),28)
        self.assertFalse(runner.checkpoint_due(50,6563,500))
        self.assertTrue(runner.checkpoint_due(50,6563,50))
        with self.assertRaises(ValueError):runner.checkpoint_due(1,10,0)

    def test_archive_capacity_admission_does_not_borrow_reserve(self):
        with tempfile.TemporaryDirectory() as t, patch.object(runner,'ARCHIVE_ROOT',Path(t)):
            with patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=runner.RESERVE+9)):
                with self.assertRaisesRegex(RuntimeError,'G archive'):runner.check_archive_space(10)
                self.assertEqual(runner.check_archive_space(9),runner.RESERVE+9)

    def _revision(self,base):
        research=base/'research';group=research/'runs/new';group.mkdir(parents=True)
        archive=base/'archive';(archive/'runs').mkdir(parents=True)
        create_output_binding(group,research_root=research,output_base=archive/'runs')
        base=group
        old={'runner.py':'old','runtime_backend.py':'fixed_model','train_data.py':'fixed_data'}
        current={**old,'runner.py':'new_storage','run_archive.py':'new_archive','output_storage.py':'new_output'}
        ad={'executed_code_sha256':old};dump(base/'admission.json',ad)
        rev={'schema_version':'bbox_storage_only_revision_v1','source_admission_sha256':digest(base/'admission.json'),
             'new_plan_sha256':runner.PLAN_SHA,'current_source_sha256':current,'checkpoint_every':500,'archive_root':str(archive),'storage_mode':'direct_g','output_group':str(archive/'runs/new')}
        dump(base/'storage_revision.json',rev)
        return base,research,archive,ad,current,rev

    def test_original_gpu_measurement_can_only_cross_storage_revision(self):
        with tempfile.TemporaryDirectory() as t:
            base,research,archive,ad,current,rev=self._revision(Path(t))
            with patch.object(runner,'PREFLIGHT_SHA',digest(base/'admission.json')),patch.object(runner,'RESEARCH',research),patch.object(runner,'ARCHIVE_ROOT',archive):
                self.assertEqual(runner.validate_storage_revision(base,ad,current),digest(base/'storage_revision.json'))
                forged={**current,'runtime_backend.py':'changed_loss_or_optimizer'}
                rev['current_source_sha256']=forged;dump(base/'storage_revision.json',rev)
                with self.assertRaisesRegex(RuntimeError,'Numerical/data'):runner.validate_storage_revision(base,ad,forged)
                del forged['runtime_backend.py'];rev['current_source_sha256']=forged;dump(base/'storage_revision.json',rev)
                with self.assertRaisesRegex(RuntimeError,'Numerical/data'):runner.validate_storage_revision(base,ad,forged)

    def test_revision_binding_or_frequency_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base,research,archive,ad,current,rev=self._revision(Path(t))
            with patch.object(runner,'PREFLIGHT_SHA',digest(base/'admission.json')),patch.object(runner,'RESEARCH',research),patch.object(runner,'ARCHIVE_ROOT',archive):
                rev['checkpoint_every']=50;dump(base/'storage_revision.json',rev)
                with self.assertRaisesRegex(RuntimeError,'storage rule'):runner.validate_storage_revision(base,ad,current)
                rev['checkpoint_every']=500;rev['new_plan_sha256']='old';dump(base/'storage_revision.json',rev)
                with self.assertRaisesRegex(RuntimeError,'authorized'):runner.validate_storage_revision(base,ad,current)

    def _reference(self,root):
        archived=root/'archive';d=archived/'runs/old/R50_seed17';g=root/'new';g.mkdir()
        panel={'pool':'exact ordered UIDs'}
        cfg={'epochs':50,'updates':6563,'model':'yolo11s','seed':17,'N_R':2100,'N_val':750,'input_size':960,'effective_batch':16,
             'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'checkpoint_every':50,
             'main_metric':'COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset'}
        summary={'update':6563,'images':750,'main_metrics':{'AP50_95':.2}}
        dump(d/'run.json',{'protocol':cfg,'source_bindings':{'real_panel':panel,'public_model':{'sha256':'85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'}}})
        dump(d/'status.json',{'state':'COMPLETE','update':6563,'terminal_evaluation':summary})
        dump(d/'evaluation/step006563/summary.json',summary);dump(d.parent/'ARCHIVE_MANIFEST.json',{'status':'PASS'})
        ref={'schema_version':'archived_R50_terminal_reference_v1','directory':str(d),
             'files':[{'relative_path':rel,'sha256':digest(d/rel)} for rel in ('run.json','status.json','evaluation/step006563/summary.json')],
             'archive_manifest_sha256':digest(d.parent/'ARCHIVE_MANIFEST.json')}
        dump(g/'R50_reference.json',ref)
        ad={'numeric_mode':'fp32','microbatch_size':8,'num_workers':2,'source_bindings':{'real_panel':panel}}
        return archived,d,g,ad,ref

    def test_archived_legacy_R50_terminal_reference_is_accepted_without_retraining(self):
        with tempfile.TemporaryDirectory() as t:
            archive,d,g,ad,ref=self._reference(Path(t))
            with patch.object(runner,'ARCHIVE_ROOT',archive):
                got=runner.validate_R50_reference(g,ad)
                self.assertEqual(got['terminal']['main_metrics']['AP50_95'],.2)
                summary=d/'evaluation/step006563/summary.json';summary.write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'reference changed'):runner.validate_R50_reference(g,ad)

    def test_rebound_cancelled_R50_or_other_panel_cannot_become_valid_reference(self):
        with tempfile.TemporaryDirectory() as t:
            archive,d,g,ad,ref=self._reference(Path(t))
            state=json.loads((d/'status.json').read_bytes());state['state']='USER_STOP';dump(d/'status.json',state)
            next(x for x in ref['files'] if x['relative_path']=='status.json')['sha256']=digest(d/'status.json');dump(g/'R50_reference.json',ref)
            with patch.object(runner,'ARCHIVE_ROOT',archive):
                with self.assertRaisesRegex(RuntimeError,'not complete'):runner.validate_R50_reference(g,ad)
                state['state']='COMPLETE';dump(d/'status.json',state)
                next(x for x in ref['files'] if x['relative_path']=='status.json')['sha256']=digest(d/'status.json');dump(g/'R50_reference.json',ref)
                ad['source_bindings']['real_panel']={'pool':'different'}
                with self.assertRaisesRegex(RuntimeError,'source panel'):runner.validate_R50_reference(g,ad)

    def test_existing_R100_cannot_be_prepared_again(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);research=root/'research';(research/'runs').mkdir(parents=True)
            archive=root/'archive';old=archive/'runs/old';(old/'R100_seed17').mkdir(parents=True)
            with patch.object(runner,'RESEARCH',research),patch.object(runner,'ARCHIVE_ROOT',archive),patch.object(runner,'production_closed'):
                with self.assertRaisesRegex(RuntimeError,'never duplicate'):runner.prepare_R100_after_archive(old,research/'runs/new')
                self.assertFalse((research/'runs/new').exists())

    def test_archived_capacity_handoff_prepares_real_frozen_R100_without_GPU(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);archive,d,g,ad,ref=self._reference(root)
            research=root/'research';(research/'runs').mkdir(parents=True);(research/'docs/plan').mkdir(parents=True)
            # 同一正式方案/依赖bytes仅复制到临时夹具，不读取媒体、GT或公开模型。
            for name in ('EXPERIMENTS.md','PAPER.md'):
                (research/'docs/plan'/name).write_bytes((runner.RESEARCH/'docs/plan'/name).read_bytes())
            (research/'requirements.lock').write_bytes((runner.RESEARCH/'requirements.lock').read_bytes())
            current={f.name:digest(f) for f in Path(runner.__file__).parent.glob('*.py')}
            original={k:v for k,v in current.items() if k not in ('run_archive.py','output_storage.py')};original['runner.py']='old_storage_runner'
            ad.update(status='PASS',executed_code_sha256=original,checkpoint_bytes=114192047,raw_upper_per_image_bytes=6650344,R50_max_run_seconds=14400)
            old=d.parent;dump(old/'admission.json',ad)
            dump(old/'status.json',{'state':'FAIL_STOP','error':'BLOCKED: D free=1, reserve=100, additional_bound=200'})
            rows=[{'path':'R50_seed17/'+item['relative_path'],'bytes':(d/item['relative_path']).stat().st_size,'sha256':digest(d/item['relative_path'])} for item in ref['files']]
            dump(old/'ARCHIVE_MANIFEST.json',{'schema':'futsalmot_run_archive_v1','status':'PASS','target':str(old),'files':rows})
            target=research/'runs/followup'
            with patch.object(runner,'RESEARCH',research),patch.object(runner,'ARCHIVE_ROOT',archive),patch.object(runner,'PREFLIGHT_SHA',digest(old/'admission.json')),patch.object(runner,'production_closed'),patch.object(runner,'check_disk'),patch.object(runner,'check_archive_space'):
                got=runner.prepare_R100_after_archive(old,target)
            self.assertEqual(got['state'],'FROZEN_NOT_DISPATCHED')
            self.assertEqual((target/'admission.json').read_bytes(),(old/'admission.json').read_bytes())
            table=json.loads((target/'job_table.json').read_bytes())
            self.assertEqual([(x['epochs'],x['checkpoint_every'],x['checkpoint_count']) for x in table['jobs']],[(100,500,28)])
            self.assertTrue(table['R50_reference_sha256'])
            self.assertEqual(digest(target/'code/futsalmot/runtime_backend.py'),current['runtime_backend.py'])
            compile((target/'launch.py').read_text(encoding='utf-8'),str(target/'launch.py'),'exec')
            self.assertIn("if __name__=='__main__':main()",(target/'launch.py').read_text(encoding='utf-8'))
            self.assertFalse((target/'R100_seed17').exists())
            output=resolve_output_group(target,research_root=research,output_base=archive/'runs')
            self.assertEqual(output,archive/'runs/followup')
            self.assertEqual(json.loads((output/'control/storage_revision.json').read_bytes())['storage_mode'],'direct_g')
            self.assertFalse((output/'control/CONTROL_MANIFEST.json').exists())
            self.assertEqual(got['output_group'],str(output))
            self.assertEqual(table['jobs'][0]['control_space_bound_bytes'],32*1024**2)
            self.assertFalse(table['jobs'][0]['archive_after_owned_normal_exit'])

    def test_only_R100_is_dispatched_when_completed_R50_is_bound(self):
        with tempfile.TemporaryDirectory() as t:
            g=Path(t);dump(g/'admission.json',{'status':'PASS','checkpoint_bytes':114192047});dump(g/'R50_reference.json',{})
            dump(g/'job_table.json',{'jobs':[{'epochs':100}],'R50_reference_sha256':digest(g/'R50_reference.json')})
            prior={'state':'COMPLETE','terminal':{'main_metrics':{'AP50_95':.2}}}
            current={'state':'COMPLETE','terminal':{'main_metrics':{'AP50_95':.3}}}
            with patch.object(runner,'gpu_lock',return_value=nullcontext()),patch.object(runner,'setup_cuda'),patch.object(runner,'protocol',return_value={}),patch.object(runner,'validate_R50_reference',return_value=prior),patch.object(runner,'run_one',return_value=current) as run:
                runner.controller(g)
                run.assert_called_once();self.assertEqual(json.loads((g/'budget_choice.json').read_bytes())['chosen_R_epochs'],100)


    def test_direct_G_admission_reserves_D_only_for_small_control_and_G_full_run(self):
        ad={'checkpoint_bytes':114192047,'raw_upper_per_image_bytes':6650344,'numeric_mode':'fp32','microbatch_size':8,'num_workers':2}
        with patch.object(runner,'check_disk') as d,patch.object(runner,'check_archive_space') as g:
            cfg=runner.protocol(ad,100)
            d.assert_called_once_with(32*1024**2)
            g.assert_called_once_with(cfg['additional_space_bound_bytes'])
            self.assertEqual(cfg['additional_space_bound_bytes'],95741738325)
        with patch.object(runner,'check_disk') as d,patch.object(runner,'check_archive_space') as g:
            runner.check_write_space(cfg,123)
            d.assert_called_once_with(32*1024**2);g.assert_called_once_with(123)

    def test_direct_G_controller_mirrors_state_and_seals_tiny_control_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);research=root/'research';g=research/'runs/new';g.mkdir(parents=True)
            archive=root/'archive';(archive/'runs').mkdir(parents=True)
            create_output_binding(g,research_root=research,output_base=archive/'runs')
            dump(g/'admission.json',{'status':'PASS','checkpoint_bytes':114192047});dump(g/'R50_reference.json',{})
            dump(g/'job_table.json',{'jobs':[{'epochs':100}],'R50_reference_sha256':digest(g/'R50_reference.json')})
            (g/'code/futsalmot').mkdir(parents=True);(g/'code/futsalmot/runner.py').write_text('# fixed',encoding='utf8')
            output=archive/'runs/new';capture_control(g,research_root=research,output_base=archive/'runs')
            prior={'state':'COMPLETE','terminal':{'main_metrics':{'AP50_95':.2}}}
            current={'state':'COMPLETE','terminal':{'main_metrics':{'AP50_95':.1}}}
            def fake_run(group,cfg):
                self.assertEqual(cfg['control_group'],str(g));self.assertEqual(cfg['output_group'],str(output))
                run=runner.output_group(group)/'R100_seed17';run.mkdir();(run/'steps.jsonl').write_text('fixture',encoding='utf8')
                return current
            with patch.object(runner,'RESEARCH',research),patch.object(runner,'ARCHIVE_ROOT',archive),patch.object(runner,'gpu_lock',return_value=nullcontext()),patch.object(runner,'setup_cuda'),patch.object(runner,'protocol',return_value={'storage_mode':'direct_g'}),patch.object(runner,'validate_R50_reference',return_value=prior),patch.object(runner,'run_one',side_effect=fake_run):
                runner.controller(g)
                self.assertFalse((g/'R100_seed17').exists());self.assertFalse((g/'controller_events.jsonl').exists())
                self.assertTrue((output/'R100_seed17/steps.jsonl').exists())
                self.assertTrue((output/'controller_events.jsonl').exists())
                self.assertEqual(json.loads((g/'status.json').read_bytes())['chosen_R_epochs'],50)
                self.assertEqual(json.loads((output/'status.json').read_bytes())['chosen_R_epochs'],50)
                self.assertTrue(runner.stop_requested(g,output/'R100_seed17') is False)
                dump(output/'STOP.json',{'reason':'user'})
                self.assertTrue(runner.stop_requested(g,output/'R100_seed17'))
                sealed=capture_control(g,research_root=research,output_base=archive/'runs',final=True)
                self.assertTrue(sealed['final']);self.assertLess(sealed['total_bytes'],32*1024**2)
                self.assertEqual(capture_control(g,research_root=research,output_base=archive/'runs',final=True)['manifest_sha256'],sealed['manifest_sha256'])

if __name__=='__main__':unittest.main()
