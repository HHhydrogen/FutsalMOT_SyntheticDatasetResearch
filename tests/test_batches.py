"""960几何/批次CPU夹具；仅自造RGB，不读文件、模型、GT或GPU。"""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-batch-fixture-")
_ENV={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo")
os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances
from futsalmot.batches import letterbox_bbox_sample,collate_bbox_samples

_THREADS=None
_BLOCK=None

def setUpModule():
    global _THREADS,_BLOCK
    _THREADS=torch.get_num_threads();torch.set_num_threads(2)
    _BLOCK=mock.patch("torch.cuda._lazy_init",side_effect=AssertionError("CPU fixture禁止CUDA初始化"));_BLOCK.start()

def tearDownModule():
    if _BLOCK is not None:_BLOCK.stop()
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


def record(box,**extra):
    return {"source_bbox":list(box),"source_format":"xyxy_half_open", "clipped_xyxy":list(box),
            "bbox_supervision_valid":True, "track_id":1, **extra}


def sample(h=2,w=4,domain="R",empty=False):
    return {"width":w,"height":h,"domain":domain,
            "sample_uid":"detect:real:fixture" if domain=="R" else None,
            "derived_sample_uid":"detect:derived:synthetic:fixture" if domain=="S" else None,
            "class_mapping":{"person":0},"model_head_classes":80,
            "targets":[] if empty else [record([w/4,h/4,3*w/4,3*h/4],class_id=0)],
            "ignore_targets":[] if empty else [record([0,0,w/4,h/2],ignore_reason="referee")],
            "absent_targets":[{"entity_id":"R4","bbox_supervision_valid":False}],
            "supervision":{"native_yolo_training_ready":False}}


class LetterboxBatchTests(unittest.TestCase):
    def test_reference_pixel_and_box_parity_odd_padding_RGB_and_no_mutation(self):
        reference=LetterBox(new_shape=(960,960),auto=False,scale_fill=False,scaleup=True,center=True,
                            stride=32,padding_value=114)
        for h,w in [(3,7),(7,3),(5,5),(4,9)]:
            rgb=(np.arange(h*w*3,dtype=np.uint16)%251).astype(np.uint8).reshape(h,w,3)
            source=sample(h,w);before=deepcopy(source);pixels=rgb.copy()
            got=letterbox_bbox_sample(source,rgb)
            inst=Instances(np.array([source["targets"][0]["clipped_xyxy"]],dtype=np.float64),segments=np.empty((0,0,2)),bbox_format="xyxy",normalized=False)
            expected=reference(labels={"img":rgb.copy(),"instances":inst})
            self.assertTrue(np.array_equal(got["img"].numpy().transpose(1,2,0),expected["img"]))
            self.assertTrue(np.allclose(got["targets"][0]["input_xyxy"],expected["instances"].bboxes[0],rtol=0,atol=1e-8))
            self.assertTrue(np.array_equal(pixels,rgb));self.assertEqual(source,before)
            self.assertTrue(got["img"].is_contiguous());self.assertEqual(got["img"].dtype,torch.uint8)
        # 3/7产生549像素垂直padding，上274下275，不能均分后随意取整。
        odd=letterbox_bbox_sample(sample(3,7),np.zeros((3,7,3),dtype=np.uint8))
        self.assertEqual(odd["geometry"]["padding_ltrb"],(0,274,0,275))
        rgb=np.full((2,4,3),[230,10,3],dtype=np.uint8)
        got=letterbox_bbox_sample(sample(),rgb)
        self.assertEqual(got["img"][:,300,300].tolist(),[230,10,3])
        got["img"][0,300,300]=0
        self.assertEqual(rgb[0,0,0],230)

    def test_hand_geometry_sync_ignore_and_original_provenance(self):
        source=sample();got=letterbox_bbox_sample(source,np.zeros((2,4,3),dtype=np.uint8))
        self.assertEqual(got["geometry"]["original_shape"],(2,4))
        self.assertEqual(got["geometry"]["scale"],(240,240))
        self.assertEqual(got["geometry"]["padding_ltrb"],(0,240,0,240))
        self.assertEqual(got["targets"][0]["input_xyxy"],[240,360,720,600])
        self.assertEqual(got["bboxes"].tolist(),[[.5,.5,.5,.25]])
        self.assertEqual(got["ignore_bboxes"].tolist(),[[0,240,240,480]])
        self.assertEqual(got["targets"][0]["source_bbox"],[1,.5,3,1.5])
        self.assertFalse(got["supervision"]["native_yolo_training_ready"])

    def test_collate_preserves_order_empty_shapes_and_ignore_image_indices(self):
        rgb=np.zeros((2,4,3),dtype=np.uint8)
        a=letterbox_bbox_sample(sample(empty=True),rgb)
        b=letterbox_bbox_sample(sample(domain="S"),rgb)
        self.assertEqual(tuple(a["bboxes"].shape),(0,4));self.assertEqual(tuple(a["ignore_bboxes"].shape),(0,4))
        got=collate_bbox_samples([a,b])
        self.assertEqual(tuple(got["img"].shape),(2,3,960,960))
        self.assertEqual(got["img"].dtype,torch.uint8)
        self.assertEqual(got["batch_idx"].tolist(),[1]);self.assertEqual(got["ignore_batch_idx"].tolist(),[1])
        self.assertEqual(got["cls"].tolist(),[[0]])
        self.assertEqual(got["domains"],["R","S"])
        self.assertEqual(got["uids"],["detect:real:fixture","detect:derived:synthetic:fixture"])
        self.assertFalse(got["formal_train_augmentation_implemented"])
        empty=collate_bbox_samples([a,a])
        for field,shape in [("bboxes",(0,4)),("cls",(0,1)),("batch_idx",(0,)),("ignore_bboxes",(0,4)),("ignore_batch_idx",(0,))]:
            self.assertEqual(tuple(empty[field].shape),shape)

    def test_invalid_input_shape_labels_and_batches_refused(self):
        rgb=np.zeros((2,4,3),dtype=np.uint8)
        for wrong in [rgb.astype(np.float32),np.zeros((2,4),dtype=np.uint8),np.zeros((3,4,3),dtype=np.uint8)]:
            with self.assertRaises(ValueError):letterbox_bbox_sample(sample(),wrong)
        changes=[lambda s:s.update(original_shape=(4,2)), lambda s:s.pop("ignore_targets"),
                 lambda s:s["targets"][0].update(clipped_xyxy=[-1,0,1,1]),
                 lambda s:s["targets"][0].update(class_id=1),
                 lambda s:s["ignore_targets"][0].update(clipped_xyxy=[0,0,float("nan"),1]),
                 lambda s:s["targets"][0].update(bbox_supervision_valid=False)]
        for change in changes:
            bad=sample();change(bad)
            with self.assertRaises(ValueError):letterbox_bbox_sample(bad,rgb)
        with self.assertRaises(ValueError):collate_bbox_samples([])
        good=letterbox_bbox_sample(sample(),rgb)
        for key,value in [("img",torch.zeros((3,640,960),dtype=torch.uint8)),("cls",torch.tensor([[1.]])),
                          ("ignore_bboxes",torch.tensor([[0.,0.,961.,2.]])),("bboxes",torch.tensor([[.5,.5,-1.,.5]]))]:
            with self.assertRaises(ValueError):collate_bbox_samples([{**good,key:value}])

    def test_explicit_no_image_region_preserved_without_inventing_box(self):
        source=sample()
        source["targets"].append({"class_id":0,"track_id":2,"source_bbox":[20,0,30,1],
                                  "bbox_supervision_valid":False,"clipped_xyxy":None})
        got=letterbox_bbox_sample(source,np.zeros((2,4,3),dtype=np.uint8))
        self.assertEqual(len(got["bboxes"]),1)
        self.assertEqual(got["targets"][1]["source_bbox"],[20,0,30,1])
        self.assertIsNone(got["targets"][1]["input_xyxy"])

if __name__=="__main__":unittest.main()
