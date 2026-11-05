"""仅队列顺序/复用证据/AST适配的极小mock，不导入训练包或torch。"""
import ast,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).absolute().parents[1]
spec=importlib.util.spec_from_file_location('label_curve_batch_mock_target',ROOT/'tools/label_curve_batch.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class LifecycleMock(unittest.TestCase):
    def test_fixed_order_exactly16_and_no_score_condition(self):
        self.assertEqual(len(b.ORDER),16);self.assertEqual([j['seed'] for j in b.ORDER],[29]*8+[43]*8)
        self.assertEqual([(j['real_fraction'],j['strategy']) for j in b.ORDER[:8]],[(f,s) for f in (1.,.5,.25,.1) for s in ('R','R+S')])
        self.assertTrue(b.jobs_equal(b.ORDER));self.assertFalse(b.jobs_equal(b.ORDER[::-1]));self.assertFalse(b.jobs_equal([{**j,'seed':17} for j in b.ORDER]))
    def test_reuse_admission_explicit_zero_measurements_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'queue.json';p.write_text('{}');qref=b.reference(p);code={'label_curve_bbox.py':b.SOURCE_SHA};family=[{'fixture':i} for i in range(4)];q={'plan_sha256':'a'*64,'validated_family':family};cfg={'seed':29,'EXPERIMENTS_sha256':'a'*64}
            value={'schema_version':b.REUSE_SCHEMA,'status':'PASS','scope':'validated_configuration_reuse','fresh_cpu_tests':False,'fresh_gpu_probe':False,'diagnostic_updates':0,'measured_probe_seconds':0,'queue_reference':qref,'inherited_configuration_family':family,'source_code_sha256':code,'seed':29,'checkpoint_every':1000,'schedule_policy':'single_stage_lr001_warm100_cosine_v1','EXPERIMENTS_sha256':'a'*64}
            b._validate_reuse_admission(value,cfg,q,qref,code)
            for key,new in [('schema_version','bbox_label_curve_admission_v1'),('scope','actually_measured'),('fresh_cpu_tests',True),('fresh_gpu_probe',True),('diagnostic_updates',3),('measured_probe_seconds',1),('source_code_sha256',{}),('seed',17)]:
                with self.subTest(key=key),self.assertRaises(ValueError):b._validate_reuse_admission({**value,key:new},cfg,q,qref,code)
    def test_derived_prepare_only_reuses_exactly_five_auditable_gates(self):
        source_path=ROOT/'src/futsalmot/label_curve_bbox.py';whole=source_path.read_text(encoding='utf8');node=next(n for n in ast.parse(whole).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_group');source=ast.get_source_segment(whole,node)
        fake=SimpleNamespace(SOURCE_GROUP=b.BASE,prepare_group=object())
        with patch.object(b.inspect,'getsource',return_value=source):function,proof=b.derive_reuse_prepare(fake,{}, {}, {})
        self.assertTrue(callable(function));self.assertTrue(proof['AST_inverse_exact']);self.assertEqual(proof['substitution_count'],5);self.assertEqual(b.sha(source_path),b.SOURCE_SHA)
    def test_full_remaining_bounds_and_stop_have_no_force_kill_or_fresh_probe(self):
        with patch.object(b.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)),patch.object(b.time,'time',return_value=0):v=b.space_and_time(16)
        self.assertEqual(v['remaining_G_upper_bytes'],16*24295280650);self.assertEqual(v['per_job_max_seconds'],5400)
        with patch.object(b.shutil,'disk_usage',return_value=SimpleNamespace(free=1)),self.assertRaises(ValueError):b.space_and_time(16)
        whole=(ROOT/'tools/label_curve_batch.py').read_text(encoding='utf-8-sig')
        tree=ast.parse(whole)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('kill','terminate') for n in ast.walk(tree)))
        self.assertNotIn('torch',__import__('sys').modules)

if __name__=='__main__':unittest.main()
