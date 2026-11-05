"""N25低标注机械预检的CPU预算/实际筛选/conhost分类检查，不运行GPU。"""
from pathlib import Path
import ast,math,unittest
R=Path(__file__).absolute().parents[1]
source=ast.parse((R/'tools/low_label_bbox_preflight.py').read_text(encoding='utf8'))
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='validate_preparation_budget')
ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<budget-only>','exec'),ns);validate=ns['validate_preparation_budget']

def cfg(strategy='R',used=0):
 return {'strategy':strategy,'budget_policy':'lowlabel25_pair120_v1','max_seconds':50,'previous_block_preparation_seconds':used,'previous_gpu_preparation_seconds':702.3206997644944+used,'block_preparation_cap_seconds':120,'total_preparation_cap_seconds':822.3206997644944,'full_process_limit_seconds':60,'full_process_deadline_unix':1.0}
class BudgetTests(unittest.TestCase):
 def test_ledger_preserves_old_preparation_and_exact_new_scope(self):
  for strategy,used in [('R',0),('R+S',47.058288574)]:
   out=validate(cfg(strategy,used));self.assertEqual(out['historical_preparation_seconds_upper'],702.3206997644944);self.assertEqual(out['full_process_limit_seconds'],60);self.assertEqual(out['cooperative_window_seconds'],50)
 def test_rejects_scope_budget_resets_nonfinite_and_bool(self):
  for key,value in [('strategy','S->R'),('strategy','R->R'),('budget_policy','budget1500_pair120_v1'),('max_seconds',51),('max_seconds',True),('max_seconds',float('nan')),('previous_gpu_preparation_seconds',0),('previous_block_preparation_seconds',float('inf')),('total_preparation_cap_seconds',823),('block_preparation_cap_seconds',121),('full_process_limit_seconds',61),('full_process_deadline_unix',None)]:
   with self.subTest(key=key),self.assertRaises(ValueError):validate({**cfg(),key:value})
 def test_second_full_helper_must_fit_remaining_block(self):
  validate(cfg('R+S',60))
  with self.assertRaises(ValueError):validate(cfg('R+S',60.001))
 def test_native_handle_timing_is_same_proven_implementation(self):
  def fn(path,name):
   tree=ast.parse(path.read_text(encoding='utf8'));return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  self.assertEqual(fn(R/'tools/low_label_bbox_preflight_observer.py','process_times'),fn(R/'tools/budget1500_preflight_observer.py','process_times'))
 def test_console_detection_uses_executable_not_command_text(self):
  tree=ast.parse((R/'tools/low_label_bbox_preflight_observer.py').read_text(encoding='utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_console_host','partition_live_owned')];env={'Path':Path};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<console-only>','exec'),env)
  rows=[{'cmdline':['C:/Windows/System32/conhost.exe','0x4']},{'cmdline':['python.exe','-c','print("conhost.exe")']},{'cmdline':[]}]
  training,hosts=env['partition_live_owned'](rows);self.assertEqual(hosts,rows[:1]);self.assertEqual(training,rows[1:])
 def test_helper_filters_actual_panel_before_sampling_and_three_update_replay(self):
  code=(R/'tools/low_label_bbox_preflight.py').read_text(encoding='utf8')
  self.assertLess(code.index('panel=mix.filter_train_panel(full_panel,subset_ref)'),code.index('samplers=mix.make_samplers(panel,synthetic)'))
  self.assertIn("bindings={**subset,'real_panel':full_panel['binding']",code)
  self.assertIn('unselected real images entered training catalogue',code);self.assertIn('unselected real UID was consumed',code)
  self.assertEqual(len([n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='step']),3)
  for p in ('tools/low_label_bbox_preflight.py','tools/low_label_bbox_preflight_observer.py'):compile((R/p).read_text(encoding='utf8'),p,'exec')
if __name__=='__main__':unittest.main()
