"""正式detect输入CPU夹具；无模型/GPU/实际数据，临时PNG自动清理。"""
from copy import deepcopy
import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch
_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-train-input-")
_ENV={key:os.environ.get(key) for key in ("YOLO_OFFLINE","CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["YOLO_OFFLINE"]="true";os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo");os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import cv2
import numpy as np
import torch
from futsalmot.train_data import TrainingBBoxDataset,prepare_training_sample,_affine,_hsv_rgb,ROLE_POLICY
from futsalmot.batches import letterbox_bbox_sample,collate_bbox_samples
from ultralytics.data.augment import RandomPerspective,RandomHSV
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

def record(box,**extra):
    return {"source_bbox":list(box),"source_xyxy":list(box),"source_format":"xyxy_half_open",
            "clipped_xyxy":list(box),"bbox_supervision_valid":True,**extra}

def fixture(*,view="train",seed=317,stream="R"):
    domain="R" if stream=="R_extra" else stream;uid="fixture:"+domain
    source={"width":960,"height":960,"domain":domain,"sample_uid":uid,"derived_sample_uid":uid,
            "class_mapping":{"person":0},"model_head_classes":80,
            "targets":[record([350,350,550,550],class_id=0,track_id=1),
                       record([450,450,451,451],class_id=0,track_id=2)],
            "ignore_targets":[record([200,200,300,300],ignore_reason="non_target_role",role="referee"),
                              record([400,400,400.5,401],ignore_reason="unknown_role"),
                              record([600,600,700,700],ignore_reason="invalid_gt_flag")],
            "absent_targets":[],"supervision":{"requires_ignore_aware_loss":True}}
    rgb=np.zeros((960,960,3),np.uint8);rgb[:]=[220,50,10]
    prepared=letterbox_bbox_sample(source,rgb)
    split=view if domain=="R" else ("train" if view=="train" else "dev")
    reference={"domain":domain,"split":split}
    if domain=="R":reference["index_path"]="D:/fixture/dataset.json"
    else:reference.update(seed_dir="D:/fixture/seed",camera_id="C5",frame_index=1)
    prepared.update(request={"uid":uid,"domain":stream,"augmentation_seed":seed,"exposure_index":3,"epoch":0},
        source_reference=reference,image_source_binding={"path":"D:/fixture/image.png","stat":(1,2,3,4)},
        sampling_stream=stream,data_domain=domain)
    return prepared

class TrainingInputTests(unittest.TestCase):
    def test_role_policy_filter_targets_but_not_tiny_ignore_and_collate_lineage(self):
        original=fixture(stream="R_extra");before=deepcopy(original)
        got=prepare_training_sample(original,view="train")
        self.assertEqual(len(got["bboxes"]),1);self.assertEqual(len(got["ignore_bboxes"]),2)
        audit=got["geometry"]["training_input"]
        self.assertEqual(audit["role_policy"],ROLE_POLICY);self.assertEqual(audit["true_ignore_source_count"],2)
        self.assertEqual(audit["known_non_target_background_records"][0]["role"],"referee")
        self.assertIn("default_wh_not_gt_2",got["targets"][1]["training_filter_reasons"])
        self.assertEqual(got["targets"][0]["source_xyxy"],before["targets"][0]["source_xyxy"])
        self.assertEqual(original["ignore_targets"],before["ignore_targets"])
        self.assertTrue(torch.equal(original["img"],before["img"]))
        batch=collate_bbox_samples([got]);self.assertEqual(len(batch["ignore_bboxes"]),2)
        self.assertEqual(batch["sample_lineage"][0]["sampling_stream"],"R_extra")
        self.assertEqual(batch["geometry"][0]["training_input"],audit)
        from futsalmot.training import prepare_detection_batch
        numeric=prepare_detection_batch(batch,device="cpu")
        self.assertEqual(numeric["img"].dtype,torch.float32)
        self.assertEqual(numeric["geometry"][0]["training_input"],audit)
        self.assertTrue(torch.equal(numeric["ignore_bboxes"],batch["ignore_bboxes"]))
    def test_determinism_different_exposure_seeds_pickle_and_global_RNG_isolation(self):
        py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state().clone()
        first=prepare_training_sample(fixture(seed=13),view="train")
        second=prepare_training_sample(fixture(seed=13),view="train")
        third=prepare_training_sample(fixture(seed=14),view="train")
        self.assertTrue(torch.equal(first["img"],second["img"]));self.assertTrue(torch.equal(first["bboxes"],second["bboxes"]))
        self.assertNotEqual(first["geometry"]["training_input"]["matrix_from_letterbox"],third["geometry"]["training_input"]["matrix_from_letterbox"])
        self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],npstate[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),ts))
        restored=pickle.loads(pickle.dumps(first));self.assertTrue(torch.equal(restored["img"],first["img"]))
    def test_val_no_augmentation_or_training_filter_and_original_gt_preserved(self):
        for stream in ("R","S"):
            original=fixture(view="val",stream=stream)
            got=prepare_training_sample(original,view="val")
            self.assertTrue(torch.equal(got["img"],original["img"]));self.assertTrue(torch.equal(got["bboxes"],original["bboxes"]))
            self.assertEqual(len(got["bboxes"]),2);self.assertEqual(len(got["ignore_bboxes"]),2)
            self.assertEqual(got["geometry"]["training_input"]["true_ignore_source_count"],2)
            self.assertFalse(got["geometry"]["training_input"]["default_positive_filter"])
            self.assertEqual([r["source_xyxy"] for r in got["targets"]],[r["source_xyxy"] for r in original["targets"]])
    def test_matrix_native_parity_and_RGB_HSV_native_BGR_parity(self):
        seed=883;ours,scale=_affine(random.Random(seed));state=random.getstate()
        native=RandomPerspective(degrees=0.,translate=.05,scale=.25,shear=0.,perspective=0.);native.size=(960,960)
        try:
            random.seed(seed)
            _,reference,reference_scale=native.affine_transform(np.zeros((960,960,3),np.uint8),(0,0))
        finally:random.setstate(state)
        self.assertTrue(np.array_equal(ours,reference));self.assertEqual(scale,reference_scale)
        rgb=np.array([[[220,30,3],[255,255,255]],[[1,100,220],[0,0,0]]],dtype=np.uint8)
        got,gains=_hsv_rgb(rgb,np.random.Generator(np.random.PCG64(seed)))
        bgr=np.ascontiguousarray(rgb[...,::-1])
        with patch("numpy.random.uniform",return_value=gains/np.array([.015,.4,.4])):
            ref=RandomHSV(.015,.4,.4)({"img":bgr})["img"][...,::-1]
        self.assertTrue(np.array_equal(got,ref))
    def test_empty_targets_ignores_and_bad_role_seed_or_split_rejected(self):
        original=fixture();original["targets"]=[];original["ignore_targets"]=[]
        got=prepare_training_sample(original,view="train")
        self.assertEqual(tuple(got["bboxes"].shape),(0,4));self.assertEqual(tuple(got["ignore_bboxes"].shape),(0,4))
        for change in ("role","seed","split"):
            bad=fixture()
            if change=="role":bad["ignore_targets"][0]["ignore_reason"]="invented"
            if change=="seed":bad["request"]["augmentation_seed"]=True
            if change=="split":bad["source_reference"]["split"]="val"
            with self.subTest(change=change),self.assertRaises(ValueError):prepare_training_sample(bad,view="train")
    def test_real_temporary_R_S_image_chain_and_dataset_pickle(self):
        from test_dataset import fixture as image_fixture,request,R_UID,S_UID
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as t:
            catalogue,_,_=image_fixture(Path(t));dataset=TrainingBBoxDataset(catalogue)
            r=dataset[request()];s=dataset[request(S_UID,"S")]
            self.assertEqual(r["uid"],R_UID);self.assertEqual(s["uid"],S_UID)
            self.assertTrue(r["random_augmentation_implemented"]);self.assertTrue(s["img"].is_contiguous())
            cloned=pickle.loads(pickle.dumps(dataset));again=cloned[request()]
            self.assertTrue(torch.equal(again["img"],r["img"]));self.assertTrue(torch.equal(again["bboxes"],r["bboxes"]))
