"""1500预算配对的历史账/新块边界CPU检查，不调用训练或CUDA。"""
from pathlib import Path
import ast,math,unittest
R=Path(__file__).absolute().parents[1]
source=ast.parse((R/'tools/budget1500_preflight.py').read_text(encoding='utf8'))
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='validate_preparation_budget')
ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<budget-only>','exec'),ns);validate=ns['validate_preparation_budget']
def cfg(strategy='R',used=0):
 return {'strategy':strategy,'budget_policy':'budget1500_pair120_v1','max_seconds':50,'previous_block_preparation_seconds':used,'previous_gpu_preparation_seconds':625.003047539+used,'block_preparation_cap_seconds':120,'total_preparation_cap_seconds':745.003047539,'full_process_limit_seconds':60,'full_process_deadline_unix':1.0}
class BudgetTests(unittest.TestCase):
 def test_each_strategy_uses_new_ledger_without_resetting_history(self):
  for strategy,used in [('R',0),('R+S',47.058288574)]:
   out=validate(cfg(strategy,used));self.assertEqual(out['historical_preparation_seconds_upper'],625.003047539);self.assertEqual(out['full_process_limit_seconds'],60);self.assertEqual(out['cooperative_window_seconds'],50)
 def test_rejects_other_strategies_reset_and_nonfinite_limits(self):
  for key,value in [('strategy','S->R'),('strategy','R->R'),('strategy','S'),('budget_policy','RR_additional_60_v1'),('max_seconds',51),('max_seconds',True),('max_seconds',float('nan')),('previous_gpu_preparation_seconds',0),('previous_block_preparation_seconds',float('inf')),('total_preparation_cap_seconds',746),('block_preparation_cap_seconds',121),('full_process_limit_seconds',61),('full_process_deadline_unix',None)]:
   with self.subTest(key=key),self.assertRaises(ValueError):validate({**cfg(),key:value})
 def test_second_full_process_must_fit_remaining_new_block(self):
  validate(cfg('R+S',60))
  with self.assertRaises(ValueError):validate(cfg('R+S',60.001))
 def test_observer_preserves_proven_native_handle_exit_timing(self):
  def fn(path,name):
   tree=ast.parse(path.read_text(encoding='utf8'));return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  self.assertEqual(fn(R/'tools/budget1500_preflight_observer.py','process_times'),fn(R/'tools/short_pair_preflight_observer.py','process_times'))
 def test_helper_syntax_and_three_update_replay_path(self):
  self.assertEqual(len([n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='step']),3)
  for p in ('tools/budget1500_preflight.py','tools/budget1500_preflight_observer.py'):compile((R/p).read_text(encoding='utf8'),p,'exec')
if __name__=='__main__':unittest.main()
