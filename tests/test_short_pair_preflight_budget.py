"""RR额外有限预算边界与真实Windows进程退出计时；不加载训练/GPU。"""
from pathlib import Path
import ast,importlib.util,math,subprocess,sys,unittest
R=Path(__file__).absolute().parents[1]
source=ast.parse((R/'tools/short_pair_preflight.py').read_text(encoding='utf-8'))
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='validate_preparation_budget')
ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<budget-only>','exec'),ns);validate=ns['validate_preparation_budget']
def cfg():
    used=102.56875896453857
    return {'strategy':'R->R','budget_policy':'RR_additional_60_v1','max_seconds':50,'previous_pair_preparation_seconds':used,'previous_gpu_preparation_seconds':475.376+used,'total_preparation_cap_seconds':660,'pair_preparation_cap_seconds':180,'full_process_limit_seconds':60,'full_process_deadline_unix':1.0}
class BudgetTests(unittest.TestCase):
    def test_additional_RR_accepts_frozen_ledger_and_reserved_cleanup(self):
        result=validate(cfg());self.assertEqual(result['global_cap_seconds'],660);self.assertEqual(result['full_process_limit_seconds'],60);self.assertEqual(result['cooperative_window_seconds'],50)
    def test_rejects_transfer_restarts_nonfinite_or_cap_relaxation(self):
        changes=[('strategy','S->R'),('max_seconds',51),('max_seconds',float('nan')),('max_seconds',True),('previous_gpu_preparation_seconds',0),('previous_pair_preparation_seconds',float('inf')),('total_preparation_cap_seconds',661),('pair_preparation_cap_seconds',181),('full_process_limit_seconds',61),('full_process_deadline_unix',None),('budget_policy','unlimited')]
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises(ValueError):validate({**cfg(),key:value})
    def test_rejects_exhausted_pair_and_global_budgets(self):
        c=cfg();c.update(previous_pair_preparation_seconds=121,previous_gpu_preparation_seconds=596.376)
        with self.assertRaises(ValueError):validate(c)
    def test_legacy_600_gate_is_not_silently_widened(self):
        self.assertEqual(validate({'strategy':'S->R','max_seconds':60,'previous_gpu_preparation_seconds':540})['global_cap_seconds'],600)
        with self.assertRaises(ValueError):validate({'strategy':'R->R','max_seconds':60,'previous_gpu_preparation_seconds':577.9447589645385})
    def test_real_held_process_handle_keeps_exact_exit_time_after_wait(self):
        spec=importlib.util.spec_from_file_location('cpu_exit_observer',R/'tools/short_pair_preflight_observer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with subprocess.Popen([sys.executable,'-B','-c','pass'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW) as p:
            before=m.process_times(p._handle);self.assertLess(before['exit_time'],0);p.wait(timeout=5);after=m.process_times(p._handle)
            self.assertEqual(p.returncode,0);self.assertEqual(after['create_time'],before['create_time']);self.assertGreater(after['exit_time'],after['create_time']);self.assertLess(after['exit_time']-after['create_time'],5)
if __name__=='__main__':unittest.main()
