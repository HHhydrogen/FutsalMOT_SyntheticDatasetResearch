"""微小Tensor criterion检查；不建YOLO网络、不读媒体、不step，临时配置自动清理。"""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

_TEMP = tempfile.TemporaryDirectory(prefix="futsalmot-loss-fixture-")
_ENV = {key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo")
os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")

import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from futsalmot.losses import IgnoreAwareDetectionLoss, ignored_anchor_mask, validate_upstream_binding

_PREVIOUS_THREADS = None
_CUDA_BLOCK = None


def setUpModule():
    global _PREVIOUS_THREADS, _CUDA_BLOCK
    _PREVIOUS_THREADS=torch.get_num_threads()
    torch.set_num_threads(2)
    _CUDA_BLOCK=mock.patch("torch.cuda._lazy_init",side_effect=AssertionError("CPU fixture不得初始化CUDA"))
    _CUDA_BLOCK.start()


def tearDownModule():
    if _CUDA_BLOCK is not None:_CUDA_BLOCK.stop()
    if _PREVIOUS_THREADS is not None:torch.set_num_threads(_PREVIOUS_THREADS)
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


def metadata(nc=80):
    # 仅head元数据与device哨兵，不实例化nn.Module网络/YOLO。
    return SimpleNamespace(parameters=lambda:iter([torch.zeros(1,device="cpu")]),
                           args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5),
                           model=[SimpleNamespace(nc=nc,reg_max=4,stride=torch.tensor([8.]))])


def predictions():
    return [torch.linspace(-.3,.5,2*96*3*4,device="cpu").reshape(2,96,3,4).clone().requires_grad_(True)]


def batch(empty_targets=False):
    return {"batch_idx":torch.empty(0) if empty_targets else torch.tensor([0.,1.]),
            "cls":torch.empty((0,1)) if empty_targets else torch.tensor([[0.],[0.]]),
            "bboxes":torch.empty((0,4)) if empty_targets else torch.tensor([[.5,.5,.75,.75],[.5,.5,.75,.75]]),
            "ignore_bboxes":torch.empty((0,4)),"ignore_batch_idx":torch.empty(0,dtype=torch.long)}


class FixedAssignment(torch.nn.Module):
    def forward(self,scores,boxes,anchors,labels,gt_boxes,mask_gt):
        b,a,_=scores.shape
        target_boxes=torch.zeros((b,a,4),dtype=boxes.dtype)
        target_scores=torch.zeros_like(scores)
        fg=torch.zeros((b,a),dtype=torch.bool)
        target_boxes[0,0]=torch.tensor([1.,1.,7.,7.]);target_scores[0,0,0]=.75;fg[0,0]=True
        target_boxes[1,5]=torch.tensor([9.,9.,15.,15.]);target_scores[1,5,0]=.5;fg[1,5]=True
        return torch.zeros((b,a),dtype=torch.long),target_boxes,target_scores,fg,torch.zeros((b,a),dtype=torch.long)


class IgnoreCriterionTests(unittest.TestCase):
    def test_exact_upstream_binding_and_empty_ignore_value_gradient_parity(self):
        self.assertEqual(validate_upstream_binding()["version"],"8.3.237")
        for empty in (False,True):
            native=v8DetectionLoss(metadata())
            adapted=IgnoreAwareDetectionLoss(metadata())
            x,y=predictions(),predictions()
            original,original_items=native(x,batch(empty))
            actual,actual_items=adapted(y,batch(empty))
            self.assertEqual(tuple(actual.shape),(3,))
            self.assertTrue(torch.equal(actual,actual_items*2)) # 原生返回乘microbatch的向量。
            self.assertTrue(torch.equal(original,actual))
            self.assertTrue(torch.equal(original_items,actual_items))
            if not empty:
                self.assertGreater(actual[0].item(),0)
                self.assertGreater(actual[2].item(),0)
            original.sum().backward();actual.sum().backward()
            self.assertTrue(torch.equal(x[0].grad,y[0].grad))

    def test_ignore_negative_all80_only_positive_priority_box_dfl_unchanged(self):
        native,adapted=v8DetectionLoss(metadata()),IgnoreAwareDetectionLoss(metadata())
        native.assigner=FixedAssignment();adapted.assigner=FixedAssignment()
        x,y=predictions(),predictions()
        b=batch();b["ignore_bboxes"]=torch.tensor([[0.,0.,32.,24.],[8.,8.,16.,16.]])
        b["ignore_batch_idx"]=torch.tensor([0,1])
        old,_=native(x,b);new,_=adapted(y,b)
        self.assertTrue(torch.equal(old[[0,2]],new[[0,2]]))
        self.assertLess(new[1].item(),old[1].item())
        old.sum().backward();new.sum().backward()
        self.assertTrue(torch.equal(x[0].grad[:,:16],y[0].grad[:,:16]))
        old_cls=x[0].grad[:,16:].reshape(2,80,12)
        new_cls=y[0].grad[:,16:].reshape(2,80,12)
        self.assertTrue(torch.equal(old_cls[0,:,0],new_cls[0,:,0])) # ignore内已匹配正anchor完整保留80类BCE。
        self.assertTrue(torch.equal(old_cls[1],new_cls[1])) # image1唯一ignore anchor本身是正目标。
        self.assertTrue((old_cls[0,:,1:]!=0).all())
        self.assertTrue((new_cls[0,:,1:]==0).all())
        # 原分母仅来自.75+.5的TAL正目标分数；ignore不加入分母。
        logits=x[0].detach()[:,16:].reshape(2,80,12).permute(0,2,1)
        removed=torch.nn.functional.binary_cross_entropy_with_logits(logits[0,1:],torch.zeros_like(logits[0,1:]),reduction="sum")
        self.assertTrue(torch.allclose(old[1]-new[1],removed/1.25*.5*2,rtol=1e-6,atol=1e-5))

    def test_half_open_boundaries_per_image_and_union(self):
        centers=torch.tensor([[0.,0.],[1.,1.],[2.,1.],[1.,2.]])
        got=ignored_anchor_mask(centers,input_hw=(3,3),batch_size=2,
             ignore_bboxes=torch.tensor([[1.,1.,2.,2.],[0.,0.,1.,1.],[1.,1.,2.,2.]]),
             ignore_batch_idx=torch.tensor([0,1,0]))
        self.assertEqual(got.tolist(),[[False,True,False,False],[True,False,False,False]])

    def test_missing_and_malformed_ignore_fail_even_when_empty(self):
        criterion=IgnoreAwareDetectionLoss(metadata())
        missing=batch();missing.pop("ignore_batch_idx")
        with self.assertRaisesRegex(ValueError,"缺少"):criterion(predictions(),missing)
        bad=[(torch.empty(0),torch.empty(0,dtype=torch.long)),
             (torch.empty((0,4)),torch.empty((0,1),dtype=torch.long)),
             (torch.tensor([[0.,0.,1.,1.]]),torch.tensor([.9])),
             (torch.tensor([[0.,0.,1.,1.]]),torch.tensor([-1])),
             (torch.tensor([[0.,0.,1.,1.]]),torch.tensor([2])),
             (torch.tensor([[0.,0.,0.,1.]]),torch.tensor([0])),
             (torch.tensor([[-1.,0.,1.,1.]]),torch.tensor([0])),
             (torch.tensor([[0.,0.,33.,1.]]),torch.tensor([0])),
             (torch.tensor([[0.,0.,float("nan"),1.]]),torch.tensor([0]))]
        for boxes,idx in bad:
            b=batch();b.update(ignore_bboxes=boxes,ignore_batch_idx=idx)
            with self.assertRaises(ValueError):criterion(predictions(),b)
        # 拒绝错误类头；未运行模型。
        with self.assertRaisesRegex(ValueError,"COCO80"):IgnoreAwareDetectionLoss(metadata(nc=1))

    def test_failure_removes_hook_restores_BCE_and_allows_next_call(self):
        criterion=IgnoreAwareDetectionLoss(metadata())
        original=criterion.bce
        b=batch();b.update(ignore_bboxes=torch.tensor([[0.,0.,8.,8.]]),ignore_batch_idx=torch.tensor([0]))
        with mock.patch.object(criterion.assigner,"forward",side_effect=RuntimeError("fixture assignment failure")):
            with self.assertRaisesRegex(RuntimeError,"fixture assignment failure"):criterion(predictions(),b)
        self.assertIs(criterion.bce,original)
        self.assertEqual(len(criterion.assigner._forward_hooks),0)
        criterion(predictions(),batch())

if __name__=="__main__":unittest.main()
