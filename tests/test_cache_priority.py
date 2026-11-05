import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sys
ROOT=Path(__file__).parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.cache_priority import RealTrainingPriority,recent_step_metrics


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def fixture(root,seconds=.5,update=50):
    write(root/'status.json',{'state':'RUNNING','active_job':'R_seed17'})
    d=root/'R_seed17';write(d/'run.json',{'protocol':{'strategy':'R','evaluation_schedule':[{'tag':'step001000_final_periodic','global_update':1000}]}})
    write(d/'status.json',{'state':'RUNNING','update':update})
    (d/'steps.jsonl').write_text(''.join(json.dumps({'update':i,'wall_seconds':seconds})+'\n' for i in range(update-15,update+1)),encoding='utf-8')
    return d


class PriorityTests(unittest.TestCase):
    def test_healthy_R_does_not_pause(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root);p=RealTrainingPriority(root,baseline_seconds=.5)
            self.assertFalse(p());self.assertEqual(p.reason,'real_training_healthy')

    def test_slowdown_pauses_and_cooldown_requires_recovery(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root,.7);now=[1.];p=RealTrainingPriority(root,baseline_seconds=.5,poll_seconds=0,clock=lambda:now[0])
            self.assertTrue(p());fixture(root,.5);now[0]=10;self.assertTrue(p());now[0]=32;self.assertFalse(p())

    def test_validation_pauses_before_first_stdout_progress(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=fixture(root,update=1000);p=RealTrainingPriority(root,baseline_seconds=.5,poll_seconds=0)
            self.assertTrue(p());self.assertEqual(p.reason,'real_validation_or_startup')
            write(d/'evaluation/step001000_final_periodic/summary.json',{'images':750});self.assertFalse(p())

    def test_missing_control_fail_safe_and_complete_block_allows(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=RealTrainingPriority(root,baseline_seconds=.5,poll_seconds=0);self.assertTrue(p())
            write(root/'status.json',{'state':'FORMAL_BLOCK_COMPLETE'});self.assertFalse(p())

    def test_partial_log_last_line_is_not_committed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'steps';p.write_bytes(b'{"update":1,"wall_seconds":0.5}\n{"update":2')
            self.assertEqual(recent_step_metrics(p),[{'update':1,'wall_seconds':.5}])

    def test_unregistered_job_pauses(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=fixture(root);write(d/'run.json',{'protocol':{'strategy':'S'}})
            self.assertTrue(RealTrainingPriority(root,baseline_seconds=.5)())


if __name__=='__main__':unittest.main()
