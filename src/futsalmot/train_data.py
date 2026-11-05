"""已获用户授权的detect输入：role_policy_v2、直接960及逐请求确定性增强。

复用原RequestBBoxDataset/LetterBox，不改原GT/index或sampler。几何算法与
box_candidates复用锁定Ultralytics8.3.237（AGPL-3.0）；HSV LUT同版本但显式RGB。
每请求使用独立Python Random和NumPy PCG64流；不seed/消费全局worker RNG。
不包含网络/损失/optimizer/GPU；此模块可pickle，真实worker吞吐由外层冻结。
"""
from __future__ import annotations
from copy import deepcopy
import os
import random
import cv2
import numpy as np
import torch
from .dataset import RequestBBoxDataset
from .batches import _validate_binding,LETTERBOX_BINDING

ROLE_POLICY="role_policy_v2_known_non_target_background_invalid_unknown_ignore_positive_priority"
PARAMETERS={"hsv_h":.015,"hsv_s":.4,"hsv_v":.4,"degrees":0.,"translate":.05,"scale":.25,
            "shear":0.,"perspective":0.,"fliplr":.5,"flipud":0.,"mosaic":0.,"mixup":0.,"copy_paste":0.}
_TRUE_IGNORE={"invalid_gt_flag","unknown_role"}

def _native_geometry():
    if os.environ.get("YOLO_OFFLINE","").lower()!="true":
        raise RuntimeError("训练输入要求首次原生导入前YOLO_OFFLINE=true")
    _validate_binding()
    from ultralytics.data.augment import RandomPerspective
    return RandomPerspective(degrees=0.,translate=.05,scale=.25,shear=0.,perspective=0.)

def _affine(rng):
    # 与8.3.237 RandomPerspective.affine_transform同抽样次序；零幅度也消耗local流。
    center=np.eye(3,dtype=np.float32);center[0,2]=center[1,2]=-480.
    perspective=np.eye(3,dtype=np.float32)
    perspective[2,0]=rng.uniform(0.,0.);perspective[2,1]=rng.uniform(0.,0.)
    rotation=np.eye(3,dtype=np.float32);angle=rng.uniform(0.,0.);scale=rng.uniform(.75,1.25)
    rotation[:2]=cv2.getRotationMatrix2D((0,0),angle,scale)
    shear=np.eye(3,dtype=np.float32)
    shear[0,1]=rng.uniform(0.,0.);shear[1,0]=rng.uniform(0.,0.)
    translation=np.eye(3,dtype=np.float32)
    translation[0,2]=rng.uniform(.45,.55)*960.;translation[1,2]=rng.uniform(.45,.55)*960.
    return translation@shear@rotation@perspective@center,scale

def _hsv_rgb(image,rng):
    gains=rng.uniform(-1.,1.,3)*np.array([.015,.4,.4])
    x=np.arange(256,dtype=gains.dtype)
    hue=((x+gains[0]*180)%180).astype(np.uint8)
    saturation=np.clip(x*(gains[1]+1),0,255).astype(np.uint8);saturation[0]=0
    value=np.clip(x*(gains[2]+1),0,255).astype(np.uint8)
    h,s,v=cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2HSV))
    result=cv2.cvtColor(cv2.merge((cv2.LUT(h,hue),cv2.LUT(s,saturation),cv2.LUT(v,value))),cv2.COLOR_HSV2RGB)
    return result,gains

def _boxes(records):
    result=[];indices=[]
    for index,record in enumerate(records):
        box=record.get("input_xyxy")
        if box is None:
            if record.get("bbox_supervision_valid") is not False:raise ValueError("缺input框且监督标志不一致")
            continue
        array=np.asarray(box,dtype=np.float32)
        if array.shape!=(4,) or not np.isfinite(array).all() or np.any(array[:2]<0) or np.any(array[2:]>960) or np.any(array[2:]<=array[:2]):
            raise ValueError("输入框必须为960内有限正面积xyxy")
        result.append(array);indices.append(index)
    return np.asarray(result,dtype=np.float32).reshape(-1,4),indices

def _filter_reasons(before,after,scale):
    w1,h1=(before[:,2:]-before[:,:2]).T*scale
    w2,h2=(after[:,2:]-after[:,:2]).T
    reasons=[]
    for i in range(len(before)):
        row=[]
        if w2[i]<=2 or h2[i]<=2:row.append("default_wh_not_gt_2")
        if w2[i]*h2[i]/(w1[i]*h1[i]+1e-16)<=.1:row.append("default_area_ratio_not_gt_0.1")
        if max(w2[i]/(h2[i]+1e-16),h2[i]/(w2[i]+1e-16))>=100:row.append("default_aspect_not_lt_100")
        reasons.append(row)
    return reasons

def prepare_training_sample(prepared,*,view):
    """输入现Dataset已绑定的960样本；输出兼容现collate/prepare_detection_batch。

    val不增强、不以训练阈值过滤GT，保留原source_xyxy。几何metadata内完整记录
    参数/矩阵/过滤与角色策略；现collate保留geometry，原source策略不被覆盖。
    """
    if view not in ("train","val"):raise ValueError("只允许train/val，不读test")
    transform=_native_geometry()
    if not isinstance(prepared,dict):raise ValueError("需要已绑定Dataset样本")
    for field in ("request","source_reference","image_source_binding","sampling_stream","data_domain"):
        if field not in prepared:raise ValueError("缺Dataset请求/来源血缘")
    req=prepared["request"];seed=req.get("augmentation_seed")
    if type(seed) is not int or seed<0:raise ValueError("augmentation_seed须非负整数")
    split=prepared["source_reference"].get("split");domain=prepared.get("domain")
    if split!=(view if domain=="R" else ("train" if view=="train" else "dev")):
        raise ValueError("view与原source split不一致")
    image=prepared.get("img")
    if not isinstance(image,torch.Tensor) or image.device.type!="cpu" or image.dtype!=torch.uint8 or tuple(image.shape)!=(3,960,960):
        raise ValueError("需要原Dataset CPU uint8 RGB 3x960x960")
    output=deepcopy(prepared);background=[];ignores=[]
    for record in output["ignore_targets"]:
        reason=record.get("ignore_reason")
        if reason=="non_target_role":background.append(record)
        elif reason in _TRUE_IGNORE:ignores.append(record)
        else:raise ValueError("未知ignore角色原因，不静默作为背景")
    output["ignore_targets"]=ignores
    for record in output["targets"]:
        if "source_xyxy" not in record:raise ValueError("目标必须保留原source_xyxy供主AP")
    original_targets,target_indices=_boxes(output["targets"])
    original_ignore,ignore_indices=_boxes(ignores)
    # 不信任旧ignore监督数组；从原角色记录重建，避免known_non_target继续屏蔽BCE。
    matrix=np.eye(3,dtype=np.float32);scale=1.;flipped=False;gains=np.zeros(3)
    rgb=image.permute(1,2,0).numpy().copy()
    if view=="train":
        py_rng=random.Random(seed);np_rng=np.random.Generator(np.random.PCG64(seed))
        matrix,scale=_affine(py_rng)
        rgb=cv2.warpAffine(rgb,matrix[:2],(960,960),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))
        rgb,gains=_hsv_rgb(rgb,np_rng)
        py_rng.random() # 原flipud=0仍消耗一次local draw。
        flipped=py_rng.random()<.5
    mapped_targets=transform.apply_bboxes(original_targets.copy(),matrix).clip(0,960)
    mapped_ignore=transform.apply_bboxes(original_ignore.copy(),matrix).clip(0,960)
    keep=transform.box_candidates(box1=(original_targets*scale).T,box2=mapped_targets.T) if view=="train" else np.ones(len(original_targets),dtype=bool)
    reasons=_filter_reasons(original_targets,mapped_targets,scale) if view=="train" else [[] for _ in target_indices]
    ignore_keep=np.all(mapped_ignore[:,2:]>mapped_ignore[:,:2],axis=1)
    if flipped:
        rgb=np.fliplr(rgb)
        for boxes in (mapped_targets,mapped_ignore):boxes[:,[0,2]]=960-boxes[:,[2,0]]
        flip=np.array([[-1,0,960],[0,1,0],[0,0,1]],dtype=np.float32);matrix=flip@matrix
    for records in (output["targets"],ignores):
        for record in records:
            record["letterbox_xyxy"]=deepcopy(record.get("input_xyxy"))
            record["training_selected"]=False
            record["training_filter_reasons"]=["source_no_visible_area"] if record.get("input_xyxy") is None else []
    for position,index in enumerate(target_indices):
        record=output["targets"][index];record["input_xyxy"]=mapped_targets[position].tolist()
        record["training_selected"]=bool(keep[position]);record["training_filter_reasons"]=reasons[position]
    for position,index in enumerate(ignore_indices):
        record=ignores[index];record["input_xyxy"]=mapped_ignore[position].tolist()
        record["training_selected"]=bool(ignore_keep[position])
        record["training_filter_reasons"]=[] if ignore_keep[position] else ["zero_area_after_transform"]
    xyxy=mapped_targets[keep];normalized=np.empty_like(xyxy)
    normalized[:,:2]=(xyxy[:,:2]+xyxy[:,2:])/1920.;normalized[:,2:]=(xyxy[:,2:]-xyxy[:,:2])/960.
    output.update(img=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))),
        cls=torch.zeros((len(xyxy),1),dtype=torch.float32,device="cpu"),
        bboxes=torch.from_numpy(normalized),ignore_bboxes=torch.from_numpy(mapped_ignore[ignore_keep].copy()),
        candidate_only=False,random_augmentation_implemented=view=="train",
        formal_train_augmentation_implemented=view=="train")
    audit={"role_policy":ROLE_POLICY,"view":view,"parameters":dict(PARAMETERS),"augmentation_seed":seed,
           "rng":"independent_PythonRandom_and_NumPy_PCG64_per_request",
           "matrix_from_letterbox":matrix.tolist(),"scale":float(scale),"fliplr_applied":flipped,
           "hsv_additive_gains":gains.tolist(),"target_input_count":len(original_targets),
           "target_output_count":len(xyxy),"true_ignore_source_count":len(ignores),
           "true_ignore_output_count":int(ignore_keep.sum()),"known_non_target_background_records":deepcopy(background),
           "source_role_policy_preserved":True,"positive_assignment_priority_required":True,
           "default_positive_filter":view=="train","ignore_filter":"positive_area_only",
           "binding":dict(LETTERBOX_BINDING)}
    output["geometry"]["training_input"]=audit
    output["geometry"]["letterbox_candidate_only"]=False
    output["supervision"]["requires_ignore_aware_loss"]=bool(ignore_keep.any())
    return output

class TrainingBBoxDataset(RequestBBoxDataset):
    """可pickle正式输入适配；外层仍负责run/split冻结及真实worker/GPU Gate。"""
    def __getitem__(self,request):
        return prepare_training_sample(super().__getitem__(request),view=self.view)
