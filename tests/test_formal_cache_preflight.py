"""未来缓存训练接线/准入边界的CPU夹具；不构建缓存或初始化CUDA。"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from futsalmot import formal
from futsalmot import formal_cache_preflight as cached


def base_args():
    return dict(max_run_seconds=14400,additional_space_bound_bytes=10,
                raw_upper_per_image_bytes=10,checkpoint_bytes=10,admission_sha256='a'*64)


def cache_spec():
    return {'path':str(Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/cache/synthetic_bbox_960_v1').absolute()),
            'manifest_sha256':'b'*64}


class FormalCacheTests(unittest.TestCase):
    def test_persisted_source_bindings_accept_json_roundtrip_reject_real_changes(self):
        live={'real_panel': {'index': {'identity': (1,2,3,4), 'sha256': 'a'*64},
               'ordered_uids': ('u1','u2'), 'reader_cache_info': {'cached_clips': ('c1',)}}}
        persisted=json.loads(json.dumps(live))
        self.assertNotEqual(live,persisted)
        cached._validate_source_bindings(persisted,live)
        for path,value in [('digest','b'*64),('identity',(1,2,3,5)),('uids',('u2','u1')),('missing',None)]:
            changed=deepcopy(live)
            if path=='digest':changed['real_panel']['index']['sha256']=value
            elif path=='identity':changed['real_panel']['index']['identity']=value
            elif path=='uids':changed['real_panel']['ordered_uids']=value
            else:del changed['real_panel']['index']['sha256']
            with self.subTest(change=path),self.assertRaises(ValueError):
                cached._validate_source_bindings(persisted,changed)

    def test_no_cache_protocol_differs_only_in_approved_evaluation_and_plan(self):
        # 只读活动代码；不导入其controller或修改冻结文件。
        path=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/bbox_s_soccernet_seed17_core_20261102_a01/code/futsalmot/formal.py')
        spec=importlib.util.spec_from_file_location('futsalmot._original_formal_fixture',path)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        for strategy in ('R','S','S->R','R->R','R+S'):
            current=formal.protocol(strategy,17,**base_args());original=legacy.protocol(strategy,17,**base_args())
            self.assertEqual(set(current)-set(original),{'evaluation_policy'})
            self.assertEqual({k for k in original if current[k]!=original[k]},
                             {'EXPERIMENTS_sha256','evaluation_every','evaluation_schedule'})
            self.assertEqual(current['EXPERIMENTS_sha256'],'3814e51504548a39e6b88fdf073a4377ba22b6ddf7b1607330bfb1664c6eac67')
            self.assertEqual(current['evaluation_policy'],'phase_endpoints_v1')
            self.assertIsNone(current['evaluation_every'])
            self.assertEqual(original['EXPERIMENTS_sha256'],'46aa2fe83daf99bc0a2e1880f593eb36b2c11297b95f8eaaf479de19c4401446')
            self.assertEqual({k:v for k,v in current.items() if k not in {'EXPERIMENTS_sha256','evaluation_every','evaluation_schedule','evaluation_policy'}},
                             {k:v for k,v in original.items() if k not in {'EXPERIMENTS_sha256','evaluation_every','evaluation_schedule'}})
            self.assertNotIn('input_cache',current)

    def test_only_explicit_S_to_R_seed17_can_bind_cache(self):
        spec=cache_spec();config=formal.protocol('S->R',17,input_cache=spec,**base_args())
        self.assertEqual(config['input_cache'],spec);config['control_group']='/fixture';formal._check_protocol(config)
        for strategy,seed in [('R',17),('S',17),('R->R',17),('R+S',17),('S->R',29)]:
            with self.subTest(strategy=strategy,seed=seed),self.assertRaises(ValueError):
                formal.protocol(strategy,seed,input_cache=spec,**base_args())

    def test_cache_specification_needs_exact_absolute_path_and_SHA(self):
        good=cache_spec();self.assertEqual(formal.cache_specification(good),good)
        for bad in [{**good,'path':'relative/cache'},{**good,'manifest_sha256':'B'*64},
                    {**good,'manifest_sha256':'b'*63},{**good,'status':'READY'}, {'path':good['path']}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):formal.cache_specification(bad)

    def test_factory_preserves_requests_and_passes_strict_manifest(self):
        cat={'u':{'domain':'S'}};rows=[{'uid':'u','domain':'S'}];before=deepcopy(rows);spec=cache_spec()
        with patch('futsalmot.runner_data.RequestPlanDataset') as direct,patch('futsalmot.input_cache.CachedRequestPlanDataset') as reader:
            actual=formal.make_training_dataset(cat,rows,{})
            self.assertIs(actual,direct.return_value);reader.assert_not_called()
            actual=formal.make_training_dataset(cat,rows,{'input_cache':spec})
            self.assertIs(actual,reader.return_value)
            reader.assert_called_once_with(cat,rows,cache_root=spec['path'],expected_manifest_sha256=spec['manifest_sha256'],
                view='train',real_reader_options=formal.REAL_READER_OPTIONS,synthetic_reader_options=formal.SYNTHETIC_READER_OPTIONS)
        self.assertEqual(rows,before)

    def test_cached_config_cannot_change_scientific_parameters(self):
        config=formal.protocol('S->R',17,input_cache=cache_spec(),**base_args());config['control_group']='/fixture'
        for key,value in [('numeric_mode','amp_bf16'),('checkpoint_every',1000),('num_workers',4),('updates',10)]:
            with self.subTest(key=key),self.assertRaises(ValueError):formal._check_protocol({**config,key:value})


class CachedPreflightBoundaryTests(unittest.TestCase):
    def closed(self):
        jobs=[{'strategy':'R','seed':17,'job_id':'R_seed17','updates':6563},
              {'strategy':'R->R','seed':17,'job_id':'R_to_R_seed17','updates':36750}]
        table={'schema_version':formal.SCHEMA,'jobs':jobs}
        status={'state':'FORMAL_BLOCK_COMPLETE','results':[{'state':'COMPLETE','directory':'G:/runs/group/'+j['job_id'],'update':j['updates']} for j in jobs]}
        return table,status

    def test_closed_baseline_requires_both_exact_terminal_results(self):
        table,status=self.closed();cached._validate_closed_results(table,status)
        for wrong in [{**status,'state':'RUNNING'}, {**status,'results':status['results'][:1]},
                      {**status,'results':[{**status['results'][0],'update':6562},status['results'][1]]},
                      {**status,'results':[{**status['results'][0],'state':'USER_STOP'},status['results'][1]]}]:
            with self.subTest(wrong=wrong),self.assertRaises(RuntimeError):cached._validate_closed_results(table,wrong)
        with self.assertRaises(ValueError):cached._validate_closed_results({**table,'jobs':table['jobs'][:1]},status)

    def test_numeric_source_only_mechanical_cache_modules_may_differ(self):
        prior={'executed_code_sha256':{'losses.py':'a','formal.py':'b','data.py':'c'}}
        current={'losses.py':'a','formal.py':'new','data.py':'c','input_cache.py':'d','formal_cache_preflight.py':'e'}
        cached._validate_unchanged_numeric_source(prior,current)
        audited='481b54d63eca9d399a7cb790cc53466ef03db9a4831bdb122695bdd4feeb5c1f'
        cached._validate_unchanged_numeric_source(prior,{**current,'cache_resume_v2.py':audited})
        for wrong in [{**current,'losses.py':'changed'},{**current,'extra_unreviewed.py':'x'},
                      {**current,'cache_resume_v2.py':'unreviewed'}, {'formal.py':'new'}]:
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):cached._validate_unchanged_numeric_source(prior,wrong)

    def test_previous_actual_preflight_and_warm_receipts_share_600s_budget(self):
        prior={'elapsed_seconds':115.765}
        warm={'diagnostic_only':True,'same_numeric_and_input_protocol':True,'maximum_updates':64,'updates':64,
              'elapsed_seconds':214.235,'original_preflight_elapsed_seconds':115.765,
              'combined_GPU_preparation_seconds':330.,'per_update_seconds':[3.2]*64}
        self.assertEqual(cached._validate_previous_budget(prior,warm,240),330.)
        for wrong in [{**warm,'combined_GPU_preparation_seconds':115.765},
                      {**warm,'original_preflight_elapsed_seconds':116.},
                      {**warm,'updates':63},{**warm,'per_update_seconds':[4.]*64}]:
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):cached._validate_previous_budget(prior,wrong,240)
        with self.assertRaises(ValueError):cached._validate_previous_budget(prior,warm,271)

    def test_finite_budget_rejects_before_creating_directory(self):
        for bad in (0,241,True,2.5):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                cached.preflight_cached_formal('uncreated','S','R','W',input_cache=cache_spec(),
                    prior_admission='prior',closed_training_group='old',max_seconds=bad)
        self.assertLessEqual(cached.PREVIOUS_GPU_PRECHECK_SECONDS+cached.MAX_SECONDS,cached.TOTAL_GPU_PRECHECK_SECONDS)

    def test_missing_GPU_release_creates_only_failure_receipt_and_no_torch_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict('sys.modules',{'torch':None,'futsalmot.runner':None}):
                with self.assertRaisesRegex(RuntimeError,'gpu_release_confirmed'):
                    cached.preflight_cached_formal(temporary,'S','R','W',input_cache=cache_spec(),
                        prior_admission='prior',closed_training_group='old')
            receipt=json.loads((Path(temporary)/'admission.json').read_bytes())
            self.assertEqual((receipt['status'],receipt['diagnostic_updates']),('FAIL_STOP',0))
            self.assertFalse(receipt['test_read']);self.assertFalse(receipt['validation_scored'])
            self.assertEqual([x.name for x in Path(temporary).iterdir()],['admission.json'])

    def test_failed_receipt_is_never_replayed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            kwargs=dict(input_cache=cache_spec(),prior_admission='prior',closed_training_group='old')
            with self.assertRaises(RuntimeError):cached.preflight_cached_formal(temporary,'S','R','W',**kwargs)
            before=(Path(temporary)/'admission.json').read_bytes()
            with self.assertRaises(FileExistsError):cached.preflight_cached_formal(temporary,'S','R','W',**kwargs)
            self.assertEqual((Path(temporary)/'admission.json').read_bytes(),before)

    def test_offline_environment_is_checked_before_reading_old_group_or_GPU(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ,{'YOLO_OFFLINE':'false'}),patch.dict('sys.modules',{'torch':None,'futsalmot.runner':None}):
                with self.assertRaisesRegex(RuntimeError,'YOLO_OFFLINE'):
                    cached.preflight_cached_formal(temporary,'S','R','W',input_cache=cache_spec(),
                        prior_admission='prior',closed_training_group='old',gpu_release_confirmed=True)


if __name__=='__main__':unittest.main()
