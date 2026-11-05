"""准入/记录与正式val几何边界；不跑GPU或真实数据。"""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from futsalmot import runner
from futsalmot.postprocess import _geometry

class RunnerTests(unittest.TestCase):
    def test_exact_R_budgets_and_independent_resource_admission(self):
        admission={"checkpoint_bytes":120000000,"raw_upper_per_image_bytes":6700000,"R50_max_run_seconds":14400,"numeric_mode":"amp_bf16","microbatch_size":4,"num_workers":2}
        with patch.object(runner,"check_disk") as check, patch.object(runner,"check_archive_space") as archive:
            a=runner.protocol(admission,50,admit=False);b=runner.protocol(admission,100,admit=False)
            check.assert_not_called();archive.assert_not_called()
            self.assertEqual((a['updates'],a['checkpoint_count'],a['evaluation_count']),(6563,15,8))
            self.assertEqual((b['updates'],b['checkpoint_count'],b['evaluation_count']),(13125,28,15))
            runner.protocol(admission,50);check.assert_called_once();archive.assert_called_once()
    def test_no_overwrite_and_atomic_owned_status(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"evidence.json";runner.write_once(path,{"state":"kept"})
            with self.assertRaises(FileExistsError):runner.write_once(path,{"state":"other"})
            runner.status(t,{"state":"RUNNING"});runner.status(t,{"state":"COMPLETE"})
            self.assertEqual(json.loads((Path(t)/"status.json").read_bytes())['state'],"COMPLETE")
            self.assertEqual(json.loads(path.read_bytes()),{"state":"kept"})
    def test_formal_val_geometry_allowed_but_training_matrix_rejected(self):
        g={"original_shape":[1080,1920],"input_shape":[960,960],"resized_shape":[540,960],"scale":[.5,.5],"padding_ltrb":[0,210,0,210],"source_uid":"u","channel_order":"RGB","letterbox_candidate_only":False,
           "training_input":{"view":"val","role_policy":"role_policy_v2_known_non_target_background_invalid_unknown_ignore_positive_priority","matrix_from_letterbox":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],"hsv_additive_gains":[0.,0.,0.],"fliplr_applied":False,"default_positive_filter":False,"scale":1.}}
        self.assertEqual(_geometry("u",g)[3],.5)
        for key,value in (("view","train"),("fliplr_applied",True),("default_positive_filter",True),("scale",1.1)):
            with self.subTest(key=key):
                old=g['training_input'][key];g['training_input'][key]=value
                with self.assertRaises(ValueError):_geometry('u',g)
                g['training_input'][key]=old
        g['training_input']['matrix_from_letterbox'][0][2]=1.
        with self.assertRaises(ValueError):_geometry('u',g)

if __name__=='__main__':unittest.main()
