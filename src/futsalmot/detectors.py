"""Ultralytics8.3.237原生detect机械包装/80类eval解码候选，非训练就绪。

不实现公开.pt加载：15份起点未在本组件加载/验证，不隐式下载或信任任意pickle。
随机YAML构造仅供显式fixture；训练loss/ignore/AP/增强正式定义仍待用户讨论。
复用上游AGPL-3.0原生DetectionModel/Detect/DFL，不fuse、不重建单类头。
https://github.com/ultralytics/ultralytics/tree/v8.3.237/ultralytics/nn
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import distribution,version
from pathlib import Path

import torch
import yaml

SOURCE_BINDING={
    "ultralytics":"8.3.237",
    "nn/tasks.py":"2c106b48434a010d64a512dabe343891b060a4240faa24a47ce9b10addb1408c",
    "nn/modules/head.py":"1c02c485bd48e5536a09024507ce0e813f5da5102b20a59b8a0972a21ceb49f7",
    "nn/modules/block.py":"f92baff7a3a8d0b335b2a1c72ae76de652f96087162e4c70ad8541808024993b",
    "cfg/models/11/yolo11.yaml":"43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c",
}


@lru_cache(maxsize=1)
def _sources():
    if version("ultralytics") != SOURCE_BINDING["ultralytics"]:raise RuntimeError("仅核对Ultralytics8.3.237")
    package=distribution("ultralytics").locate_file("ultralytics")
    for relative,digest in SOURCE_BINDING.items():
        if relative=="ultralytics":continue
        if hashlib.sha256((package/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("原生源码/YAML SHA变化，停止机械适配: "+relative)
    return Path(package)


def _native():
    _sources()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect
    return DetectionModel,Detect


def _model(model):
    DetectionModel,Detect=_native()
    if type(model) is not DetectionModel:raise ValueError("只支持原生DetectionModel，不接受YOLO外层/pose/seg/其它网络")
    head=model.model[-1]
    if type(head) is not Detect or head.nc!=80 or head.reg_max!=16 or head.end2end or head.export or head.xyxy:
        raise ValueError("仅支持非export/end2end的COCO80 YOLO11 detect原生xywh头")
    for name,value in model.state_dict().items():
        if value.device.type not in ("cpu","cuda") or value.layout != torch.strided:
            raise ValueError("原生网络只支持普通CPU/已初始化CUDA状态")
        if value.is_floating_point() and value.dtype!=torch.float32:
            raise ValueError("原生raw参数/buffer须FP32，不静默转换")
    if head.dfl.conv.weight.requires_grad:
        raise ValueError("原生DFL积分权重必须保持冻结，不自动修复冻结状态")
    return head


def decode_native_eval_rows(output):
    """原生eval tuple(B,84,A probabilities+xywh,raw3heads)→逐图CPU FP32 A×6。

    score/class是全部80类argmax；原生Detect已sigmoid，不再sigmoid或乘objectness。
    不筛person/score、不NMS、不裁框、不逆letterbox，完整候选行交现postprocess。
    """
    if not isinstance(output,tuple) or len(output)!=2:
        raise ValueError("需要原生非export detect eval二元tuple")
    prediction,features=output
    if not isinstance(prediction,torch.Tensor) or prediction.dtype!=torch.float32 or prediction.ndim!=3 or prediction.shape[1]!=84 or prediction.shape[0]<1 or prediction.shape[2]<1:
        raise ValueError("原生eval须为FP32 B×84×A，不接受训练logits或其它任务")
    if not isinstance(features,list) or len(features)!=3:
        raise ValueError("原生eval须保留三个原始训练头，不能伪装其它输出")
    if any(not isinstance(f,torch.Tensor) or f.ndim!=4 or f.shape[:2]!=(prediction.shape[0],144)
           or f.device!=prediction.device or f.dtype!=torch.float32 for f in features):
        raise ValueError("原生三头batch/channel/device/dtype不一致")
    if sum(f.shape[2]*f.shape[3] for f in features)!=prediction.shape[2]:
        raise ValueError("原生eval anchor数量与三尺度不一致")
    if not torch.isfinite(prediction).all():raise ValueError("eval含非有限预测")
    xywh=prediction[:,:4].transpose(1,2)
    probabilities=prediction[:,4:].transpose(1,2)
    if (xywh[...,2:]<=0).any() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError("需要正面积decoded xywh及已经sigmoid的[0,1]概率")
    boxes=torch.cat((xywh[...,:2]-xywh[...,2:]/2,xywh[...,:2]+xywh[...,2:]/2),dim=-1)
    scores,classes=probabilities.max(dim=-1)
    rows=torch.cat((boxes,scores.unsqueeze(-1),classes.to(torch.float32).unsqueeze(-1)),dim=-1)
    return [image.detach().to(device="cpu",copy=True).contiguous() for image in rows]


def postprocess_native_eval(output,*,uids,geometries):
    """只交现候选给定框接口，不改其定义或声称AP/ignore已正式准入。"""
    rows=decode_native_eval_rows(output)
    # 现有postprocess仅定义960 letterbox；P3/8-P5/32真实网格必须对应960。
    if any(tuple(feature.shape[2:])!=(960//stride,960//stride) for feature,stride in zip(output[1],(8,16,32))):
        raise ValueError("现postprocess只接真实960输入网格，不可把64等输出贴960几何")
    if not isinstance(uids,(list,tuple)) or not isinstance(geometries,(list,tuple)) or len(uids)!=len(rows) or len(geometries)!=len(rows):
        raise ValueError("eval逐图UID/geometry须明确对齐batch")
    from .postprocess import postprocess_decoded_person
    return [postprocess_decoded_person(uid,row,geometry=geometry) for uid,row,geometry in zip(uids,rows,geometries)]


class NativeDetectionNetwork:
    """显式device wrapper，保留caller原生model state_dict/参数名，不隐藏外层前缀。

    CPU/FP32已做随机fixture；CUDA须caller明确授权且已初始化，不探测或初始化设备。
    接收已准备FP32 RGB NCHW [0,1]，不提供自动resize/归一化或正式loss/训练循环。
    """
    def __init__(self,model,*,device="cpu",allow_initialized_cuda=False,initialization_binding=None):
        _model(model)
        target=torch.device(device)
        if target.type not in ("cpu","cuda"):raise ValueError("仅支持显式cpu/已初始化cuda")
        if target.type=="cuda":
            if allow_initialized_cuda is not True or not torch.cuda.is_initialized():
                raise RuntimeError("CUDA须已初始化且显式授权；不自动初始化/探测")
        elif any(p.device.type!="cpu" for p in model.parameters()):
            raise ValueError("CPU包装不隐式读取/迁移既有非CPU参数")
        self.model=model.to(device=target) if target.type=="cuda" else model
        self.device=next(self.model.parameters()).device
        if initialization_binding is not None and not isinstance(initialization_binding,dict):
            from .pretrained import _verified_binding
            self.initialization_binding=_verified_binding(initialization_binding,model)
        else:
            self.initialization_binding=deepcopy(initialization_binding or {"kind":"caller_native_model","public_pretraining_verified":False})
            if self.initialization_binding.get("public_pretraining_verified") is not False:
                raise ValueError("不接受caller自称verified；须由固定本地公开起点加载器签发")
        self.source_binding=dict(SOURCE_BINDING)
        self.candidate_only=True
        _model(self.model)

    def _input(self,image):
        if not isinstance(image,torch.Tensor) or image.dtype!=torch.float32 or image.device!=self.device or image.ndim!=4 or image.shape[0]<1 or image.shape[1]!=3:
            raise ValueError("输入须为与模型同设备的FP32 RGB NCHW")
        if any(v<32 or v%32 for v in image.shape[2:]) or not torch.isfinite(image).all() or ((image<0)|(image>1)).any():
            raise ValueError("输入须已归一化[0,1]且H/W为至少32的32倍数")

    def training_heads(self,image):
        self._input(image);_model(self.model);self.model.train(True)
        output=self.model(image) # Tensor path不调用原生criterion，保持loss/ignore尚未选定。
        if not isinstance(output,list) or len(output)!=3 or any(x.ndim!=4 or x.shape[1]!=144 for x in output):
            raise RuntimeError("原生训练头API变化")
        return output

    def evaluation_output(self,image):
        self._input(image);_model(self.model);self.model.eval()
        with torch.inference_mode():output=self.model(image)
        # 保持原tuple，caller可再显式decode；不自动跑后处理或评价。
        return output


def build_random_yolo11(*,model_size="n",device="cpu",allow_initialized_cuda=False):
    """安装包本地YAML随机初始化fixture；不读任何.pt、不下载、不冒称公开起点。

    构造器原生包含256×256 stride探测forward；本组件外部只测试n/64×64 CPU，
    其他规格/已初始化CUDA的资源及数值尚未验证，不自动播种。
    """
    if model_size not in ("n","s","m","l","x"):raise ValueError("只支持YOLO11五种明确尺寸")
    if torch.get_default_dtype()!=torch.float32:raise ValueError("fixture构造要求默认FP32，不更改全局dtype")
    package=_sources();DetectionModel,_=_native()
    cfg=yaml.safe_load((package/"cfg/models/11/yolo11.yaml").read_text(encoding="utf8"));cfg["scale"]=model_size
    with torch.device("cpu"):
        model=DetectionModel(cfg=cfg,ch=3,nc=None,verbose=False)
    return NativeDetectionNetwork(model,device=device,allow_initialized_cuda=allow_initialized_cuda,
                                  initialization_binding={"kind":"random_installed_yaml_fixture_not_public_pretrained",
                                  "model_size":model_size,"public_pretraining_verified":False,
                                  "yaml_sha256":SOURCE_BINDING["cfg/models/11/yolo11.yaml"]})
