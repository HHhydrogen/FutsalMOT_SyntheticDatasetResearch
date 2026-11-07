"""观察台预算来自冻结协议，保留总量/阶段/真实唯一池区别。"""
import unittest
from tools.training_monitor import server as m
class BudgetTests(unittest.TestCase):
 def test_short_and_long_same_strategy_have_distinct_planned_budget(self):
  a=m.budget_summary({'protocol':{'updates':1500,'strategy':'R','effective_batch_size':16,'N_R':2100}})
  b=m.budget_summary({'protocol':{'updates':6563,'strategy':'R','effective_batch_size':16,'N_R':2100}})
  self.assertNotEqual(a['summary'],b['summary']);self.assertEqual(a['planned_exposures'],{'R':24000,'S':0});self.assertEqual(b['planned_exposures'],{'R':105008,'S':0})
 def test_same_updates_low_label_pool_is_explicit(self):
  p={'updates':1500,'strategy':'R+S','effective_batch_size':16,'terminal_exposures':{'R':12000,'S':12000},'N_R':525,'full_N_R':2100}
  a=m.budget_summary({'protocol':p});self.assertIn('真实图525/2100',a['summary']);self.assertEqual(a['planned_exposures'],p['terminal_exposures'])
 def test_two_phase_budget_and_long_source_counts(self):
  p={'updates':36750,'strategy':'S->R','effective_batch_size':16,'phases':[{'updates':30187},{'updates':6563}],'N_R':2100}
  a=m.budget_summary({'protocol':p});self.assertEqual(a['phase_updates'],[30187,6563]);self.assertIn('阶段30187+6563',a['summary']);self.assertEqual(a['planned_exposures'],{'R':105008,'S':482992})
 def test_prefix_is_not_mistaken_for_new_only_budget(self):
  a=m.budget_summary({'protocol':{'prefix_updates':3000,'updates':6563,'trajectory_updates':9563,'strategy':'S->R','effective_batch_size':16}})
  self.assertEqual(a['total_updates'],9563);self.assertIn('继承3000+新增6563',a['summary']);self.assertEqual(a['planned_exposures'],{'R':105008,'S':48000})
 def test_unknown_is_not_guessed_from_status_or_directory(self):
  a=m.budget_summary({'state':'COMPLETE','update':1500,'directory':'R500_seed17'});self.assertIsNone(a['total_updates']);self.assertEqual(a['summary'],'预算未登记')
 def test_new_formal_schemas_registered_without_changing_legacy_purpose(self):
  for schema in ('bbox_short_pair_v1','bbox_budget1500_v1','bbox_low_label25_v1','bbox_label_curve_v1'):
   p={'schema_version':schema,'task':'detect','model':'yolo11s','strategy':'R','seed':17};self.assertTrue(m.display_label({'protocol':p},'job').startswith('正式｜BBox｜YOLO11s｜'))
  self.assertTrue(m.display_label({'protocol':{'schema_version':'bbox_short_prefix_branch_v1'}},'job').startswith('开发诊断｜'))
 def test_label_curve_budgets_and_training_seeds_are_explicit(self):
  for n in (2100,1050,525,210):
   for seed in (17,29,43):
    p={'schema_version':'bbox_label_curve_v1','task':'detect','model':'yolo11s','strategy':'R+S','seed':seed,'updates':1500,'N_R':n,'full_N_R':2100,'terminal_exposures':{'R':12000,'S':12000}}
    label=m.display_label({'protocol':p},'job');budget=m.budget_summary({'protocol':p})
    self.assertTrue(label.startswith('正式｜BBox｜'));self.assertIn('seed'+str(seed),label);self.assertEqual(budget['unique_real_images'],n);self.assertIn('总1,500步',budget['summary']);self.assertEqual(budget['planned_exposures'],{'R':12000,'S':12000})
if __name__=='__main__':unittest.main()
