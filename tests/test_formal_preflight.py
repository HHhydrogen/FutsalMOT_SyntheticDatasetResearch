"""正式S预检的纯CPU前置/样本选择检查；不冒充实际GPU准入。"""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import futsalmot.formal_preflight as module


class FormalPreflightTests(unittest.TestCase):
    def test_representatives_preserve_one_per_original_camera(self):
        cat={}
        for camera in ("C5","C1","C2","C3","C4","P02","P04"):
            for frame in (1,2):cat[camera+str(frame)]={"domain":"S","split":"train","camera_id":camera}
        result=module.representative_requests(cat)
        self.assertEqual(len(result),7)
        self.assertEqual([r["uid"] for r in result],[c+"1" for c in ("C5","C1","C2","C3","C4","P02","P04")])
        self.assertTrue(all(r["domain"]==r["data_domain"]==r["stream"]=="S" for r in result))
        self.assertEqual([r["exposure_index"] for r in result],list(range(7)))

    def test_representatives_refuse_partial_and_dev(self):
        with self.assertRaises(ValueError):module.representative_requests({"u":{"domain":"S","split":"train","camera_id":"C5"}})
        with self.assertRaises(ValueError):module.representative_requests({"u":{"domain":"S","split":"dev","camera_id":"C5"}})

    def test_inherited_numeric_mode_is_exact(self):
        value={"status":"PASS","numeric_mode":"fp32","microbatch_size":8,"num_workers":2,"cuda_memory_fraction":.8}
        module._validate_prior(value)
        for key,wrong in (("status","FAIL_STOP"),("numeric_mode","amp_bf16"),("microbatch_size",4),("num_workers",4),("cuda_memory_fraction",None)):
            with self.subTest(key=key),self.assertRaises(ValueError):module._validate_prior({**value,key:wrong})

    def test_missing_gpu_release_only_writes_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("sys.modules",{"torch":None,"futsalmot.runner":None}):
                with self.assertRaisesRegex(RuntimeError,"gpu_release_confirmed"):
                    module.preflight_formal(temporary,"not_read_S","not_read_R","not_read_weights",prior_admission="not_read_old")
            result=json.loads((Path(temporary)/"admission.json").read_bytes())
            self.assertEqual(result["status"],"FAIL_STOP")
            self.assertEqual(result["diagnostic_updates"],0)
            self.assertFalse(result["test_read"])
            self.assertFalse(result["validation_scored"])
            self.assertFalse(result["formal_jobs_admitted"])
            self.assertEqual([p.name for p in Path(temporary).iterdir()],["admission.json"])

    def test_failure_receipt_cannot_be_overwritten_or_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=(temporary,"S","R","weights")
            with self.assertRaises(RuntimeError):module.preflight_formal(*args,prior_admission="prior")
            before=(Path(temporary)/"admission.json").read_bytes()
            with self.assertRaises(FileExistsError):module.preflight_formal(*args,prior_admission="prior")
            self.assertEqual((Path(temporary)/"admission.json").read_bytes(),before)

    def test_budget_limits_before_creating_paths(self):
        for bad in (0,601,True,1.5):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                module.preflight_formal("uncreated","S","R","W",prior_admission="P",max_seconds=bad)

    def test_offline_environment_before_backend_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("os.environ",{"YOLO_OFFLINE":"false"}),patch.dict("sys.modules",{"torch":None,"futsalmot.runner":None}):
                with self.assertRaisesRegex(RuntimeError,"YOLO_OFFLINE"):
                    module.preflight_formal(temporary,"S","R","W",prior_admission="P",gpu_release_confirmed=True)
            self.assertEqual(json.loads((Path(temporary)/"admission.json").read_bytes())["diagnostic_updates"],0)

    def test_frozen_workload_and_bounded_offsets(self):
        self.assertEqual((module.U_S,module.U_R),(30187,6563))
        self.assertEqual(module.S_READER_OPTIONS,{"max_indexed_cameras":7,"max_indexed_frames":3000})


if __name__=="__main__":unittest.main()
