"""四项标注效率预检CPU门禁；不执行模型、CUDA或训练。"""
import importlib.util,json,math,subprocess,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

H=module('labelcurve_preflight_budget',R/'tools/label_curve_preflight.py')
O=module('labelcurve_preflight_observer_budget',R/'tools/label_curve_preflight_observer.py')

def request(**changes):
    value={'seed':17,'strategy':'R','real_fraction':.5,'N_R':1050,'max_seconds':50,'full_process_limit_seconds':60,'budget_policy':'label_curve4_240_v1','previous_block_preparation_seconds':0.,'previous_gpu_preparation_seconds':795.8882010710282,'block_preparation_cap_seconds':240,'total_preparation_cap_seconds':1035.8882010710282,'full_process_deadline_unix':2000000000.}
    value.update(changes);return value

class LabelCurveBudgetTests(unittest.TestCase):
    def reject_both(self,config):
        with self.assertRaises(ValueError):H.validate_preparation_budget(config)
        with self.assertRaises(ValueError):O.validate_request(config)

    def test_four_arms_have_exact_scope(self):
        for fraction,count in ((.5,1050),(.1,210)):
            for strategy in ('R','R+S'):
                with self.subTest(fraction=fraction,strategy=strategy):
                    q=request(real_fraction=fraction,N_R=count,strategy=strategy)
                    a=H.validate_preparation_budget(q);b=O.validate_request(q)
                    self.assertEqual((a['N_R'],a['real_fraction']),(count,fraction));self.assertEqual(b['N_R'],count)

    def test_reject_wrong_fraction_count_and_bool(self):
        for changes in ({'real_fraction':.25,'N_R':525},{'real_fraction':.5,'N_R':210},{'real_fraction':.1,'N_R':1050},{'real_fraction':True,'N_R':210},{'real_fraction':.5,'N_R':1050.}):
            with self.subTest(changes=changes):self.reject_both(request(**changes))

    def test_historical_account_cannot_reset_or_change(self):
        for previous in (0.,702.3206997644944,754.7158580553574,795.8882010710282+1):
            with self.subTest(previous=previous):self.reject_both(request(previous_gpu_preparation_seconds=previous))

    def test_exact_budget_policy_limits_and_strategy(self):
        for changes in ({'seed':29},{'seed':43},{'seed':True},{'budget_policy':'lowlabel25_pair120_v1'},{'full_process_limit_seconds':61},{'block_preparation_cap_seconds':241},{'total_preparation_cap_seconds':1036.},{'strategy':'S->R'}):
            with self.subTest(changes=changes):self.reject_both(request(**changes))

    def test_remaining_full_process_60_required(self):
        q=request(previous_block_preparation_seconds=180,previous_gpu_preparation_seconds=975.8882010710282)
        H.validate_preparation_budget(q);O.validate_request(q)
        self.reject_both(request(previous_block_preparation_seconds=180.0001,previous_gpu_preparation_seconds=975.8883010710282))
        self.reject_both(request(previous_block_preparation_seconds=-1,previous_gpu_preparation_seconds=794.8882010710282))

    def test_nonfinite_numbers_rejected(self):
        for key in ('previous_block_preparation_seconds','previous_gpu_preparation_seconds','total_preparation_cap_seconds'):
            for value in (math.nan,math.inf,-math.inf):
                with self.subTest(key=key,value=value):self.reject_both(request(**{key:value}))

    def test_helper_deadline_and_cooperative_window(self):
        for changes in ({'full_process_deadline_unix':0},{'full_process_deadline_unix':math.nan},{'max_seconds':51},{'max_seconds':0}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):H.validate_preparation_budget(request(**changes))
        with self.assertRaises(ValueError):O.validate_request(request(max_seconds=49))

    def test_only_exact_console_host_is_host(self):
        real={'cmdline':['C:/WINDOWS/system32/conhost.exe','0x4']};fake={'cmdline':['python.exe','-c','conhost.exe']}
        self.assertTrue(O.is_console_host(real));self.assertFalse(O.is_console_host(fake));self.assertEqual(O.partition_live_owned([real,fake]),([fake],[real]))

    def test_fresh_import_does_not_import_torch_or_cuda(self):
        code="import importlib.util,sys,json;from pathlib import Path;r=Path("+repr(str(R))+");\nfor name in ('label_curve_preflight','label_curve_preflight_observer'):\n s=importlib.util.spec_from_file_location(name,r/'tools'/str(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nprint(json.dumps({'torch_imported':'torch' in sys.modules}))"
        p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],capture_output=True,text=True,check=True)
        self.assertFalse(json.loads(p.stdout)['torch_imported'])

if __name__=='__main__':unittest.main()
