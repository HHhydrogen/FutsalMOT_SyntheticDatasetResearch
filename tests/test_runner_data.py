"""有限driver数据接线：临时小文本/媒体，禁止实际G/test和worker spawn。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-runner-data-")
_ENV={k:os.environ.get(k) for k in ("YOLO_OFFLINE","CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["YOLO_OFFLINE"]="true";os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo");os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import torch
import cv2
from futsalmot.runner_data import real_panel,RequestPlanDataset,make_loader,worker_init_fn
from futsalmot.data import CycleSampler
from test_adapters import real_fixture,write_json
_THREADS=None
_CV=None
_GUARD=None

def setUpModule():
    global _THREADS,_CV,_GUARD
    _THREADS=torch.get_num_threads();torch.set_num_threads(2)
    _CV=cv2.getNumThreads();cv2.setNumThreads(2)
    _GUARD=patch("torch.cuda._lazy_init",side_effect=AssertionError("禁止GPU"));_GUARD.start()

def tearDownModule():
    if _GUARD:_GUARD.stop()
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    if _CV is not None:cv2.setNumThreads(_CV)
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value

def panel_fixture(root):
    sequences=[];matches=[]
    for split in ("train","val"):
        _,data,gt=real_fixture(root/split,split=split)
        seq=data["sequences"][0];seq["clip_uid"]=split+":clip";seq["match_uid"]=split+":match"
        seq["sequence_path"]=split+"/"+seq["sequence_path"]
        seq["frame_uid_pattern"]=split+":clip:{frame_number:06d}"
        seq["files"]["seqinfo"]="seqinfo.ini"
        seqinfo=gt.parent.parent/"seqinfo.ini";seqinfo.write_text("[Sequence]\nimWidth=100\nimHeight=50\n",encoding="utf8")
        if split=="train":
            gt.write_text(gt.read_text().replace("3,1,20,10,5,5","3,1,20,10,0.1,0.1")+"3,7,110,10,10,10,1,-1,-1,-1\n")
            game=gt.parent.parent/"gameinfo.ini"
            game.write_text(game.read_text()+"trackletID_7= player team left;7\n")
        sequences.append(seq);matches.append({"match_uid":seq["match_uid"],"split":split})
    test=deepcopy(sequences[0]);test.update(split="test",clip_uid="test:clip",match_uid="test:match",sequence_path="forbidden_test")
    sequences.append(test);matches.append({"match_uid":"test:match","split":"test"})
    data.update(sequences=sequences,matches=matches)
    write_json(root/"dataset.json",data)
    return root/"dataset.json"

class RunnerDataTests(unittest.TestCase):
    def test_order_role_original_GT_pool_SHA_and_no_test_or_media_reads(self):
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as temp:
            root=Path(temp);index=panel_fixture(root);original_open=Path.open
            def guarded(path,*args,**kwargs):
                if "forbidden_test" in str(path) or path.suffix in (".jpg",".png"):raise AssertionError("不得读test/媒体")
                return original_open(path,*args,**kwargs)
            import futsalmot.runner_data as runner
            with patch.object(Path,"open",guarded),patch.object(runner,"_text_binding",wraps=runner._text_binding) as bound:
                result=real_panel(index)
            self.assertEqual(bound.call_count,6)
            self.assertEqual(result["train_uids"],["detect:train:clip:000001","detect:train:clip:000003"])
            self.assertEqual(len(result["val_uids"]),2);self.assertEqual(len(result["binding"]["clips"]),2)
            sampler=CycleSampler(result["train_uids"],task="detect",domain="R",seed=0)
            self.assertEqual(result["binding"]["ordered_train_pool_sha256"],sampler.pool_sha)
            gt=result["val_ground_truth"][result["val_uids"][0]]
            self.assertEqual(gt["positive_gt"][0],[-5.,5.,15.,25.])
            self.assertEqual(len(gt["ignore"]),2);self.assertEqual(len(gt["known_non_target_auxiliary_ignore"]),2)
            train=result["statistics"]["train"]
            self.assertEqual(train["visible_invalid_records"],1)
            self.assertEqual(train["positive_input_wh_lt_2"],1)
            self.assertEqual(train["true_ignore_records"],2)
            self.assertEqual(train["known_non_target_background_records"],2)
            self.assertEqual(result["statistics"]["val"]["visible_invalid_records"],0)
            self.assertEqual(result["statistics"]["val"]["positive_input_wh_lt_2"],0)
            self.assertTrue(all(file["sha256"] for clip in result["binding"]["clips"] for file in clip["files"].values()))
    def test_source_change_or_missing_seqinfo_stops_panel(self):
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as temp:
            index=panel_fixture(Path(temp))
            (Path(temp)/"val/sequences/clip/seqinfo.ini").unlink()
            with self.assertRaises(FileNotFoundError):real_panel(index)
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as temp:
            index=panel_fixture(Path(temp))
            import futsalmot.runner_data as runner
            original=runner._text_binding
            def mutate(path,**kwargs):
                result=original(path,**kwargs)
                if Path(path).name=="seqinfo.ini":
                    with Path(path).open("a") as f:f.write("#changed")
                return result
            with patch.object(runner,"_text_binding",side_effect=mutate),self.assertRaisesRegex(ValueError,"变化"):real_panel(index)
    def test_request_plan_loader0worker_explicit_order_pickle_and_independent_generator(self):
        from test_dataset import fixture,request,R_UID,S_UID
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as temp:
            catalogue,_,_=fixture(Path(temp));requests=[request(),request(S_UID,"S")]
            dataset=RequestPlanDataset(catalogue,requests)
            requests[0]["augmentation_seed"]=-1
            self.assertEqual(dataset.requests[0]["augmentation_seed"],12345)
            with self.assertRaises(IndexError):dataset[-1]
            with self.assertRaises(IndexError):dataset[True]
            clone=pickle.loads(pickle.dumps(dataset));generator=torch.Generator(device="cpu").manual_seed(42)
            global_before=torch.get_rng_state().clone();before=generator.get_state().clone()
            loader=make_loader(clone,batch_size=2,num_workers=0,generator=generator)
            batches=list(loader)
            self.assertEqual(len(batches),1);self.assertEqual(batches[0]["uids"],[R_UID,S_UID])
            self.assertTrue(all(g["training_input"]["view"]=="train" for g in batches[0]["geometry"]))
            self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
            self.assertFalse(torch.equal(before,generator.get_state()))
            with self.assertRaises(ValueError):make_loader(clone,batch_size=2,num_workers=1,generator=generator)
            with self.assertRaises(RuntimeError):make_loader(clone,batch_size=2,num_workers=0,generator=generator,pin_memory=True)
    def test_worker_init_only_sets_threads_and_never_seeds(self):
        with patch.object(torch,"set_num_threads") as t,patch.object(cv2,"setNumThreads") as c, \
             patch.object(torch,"manual_seed",side_effect=AssertionError("不播种")):
            worker_init_fn(0);t.assert_called_once_with(1);c.assert_called_once_with(0)
