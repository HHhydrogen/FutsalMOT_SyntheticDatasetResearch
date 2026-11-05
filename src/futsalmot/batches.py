"""BBox确定性960 letterbox/CPU collate候选接口，真实train增强尚未接入。

仅实现auto=False/scale_fill=False/scaleup=True/center=True/stride32、
INTER_LINEAR与pad114；不新增HSV、随机仿射、翻转、抽样或正式方法定义。
尺寸/padding对齐Ultralytics 8.3.237 LetterBox（AGPL-3.0）：
https://github.com/ultralytics/ultralytics/blob/v8.3.237/ultralytics/data/augment.py
输入已经是RGB，输出仍RGB CHW uint8；未来loop再按/255转换。
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import distribution, version
from pathlib import Path

import cv2
import numpy as np
import torch

LETTERBOX_BINDING = {
    "ultralytics": "8.3.237", "opencv-python": "4.12.0.88",
    "augment.py_sha256": "b7991663a2239e71f3d25cb94534a22516169e8361d6b1624fc03ccd51055240",
}


@lru_cache(maxsize=1)
def _validate_binding():
    for package in ("ultralytics", "opencv-python"):
        if version(package) != LETTERBOX_BINDING[package]:
            raise RuntimeError("letterbox只核对过锁定的Ultralytics/OpenCV版本")
    source = Path(distribution("ultralytics").locate_file("ultralytics/data/augment.py"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != LETTERBOX_BINDING["augment.py_sha256"]:
        raise RuntimeError("LetterBox上游源码SHA变化，须重新核对")
    if cv2.__version__ != "4.12.0":
        raise RuntimeError("OpenCV运行时版本与安装锁不一致")


def _box(record, width, height):
    valid = record.get("bbox_supervision_valid")
    box = record.get("clipped_xyxy")
    if type(valid) is not bool:
        raise ValueError("目标/ignore必须保留适配器的明确bbox有效性")
    if not valid:
        if box is not None:
            raise ValueError("无效框与非空clipped_xyxy不一致")
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(x) not in (int, float) or not np.isfinite(x) for x in box):
        raise ValueError("有效clipped_xyxy必须为四个有限实数")
    x1,y1,x2,y2=box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("clipped_xyxy须为原图内正面积框；不悄悄裁边或修标")
    if "source_bbox" not in record:
        raise ValueError("缺原source bbox，不能丢原始几何来源")
    return np.array(box,dtype=np.float64)


def letterbox_bbox_sample(sample: dict, rgb: np.ndarray) -> dict:
    """已验证adapters样本+调用方RGB；不读文件、不改变原标注或输入图。

    明确无图内区域的记录保持原有效性/原框并排除于监督数组，不重新裁框。
    本接口既不证明整个数据版已发布，也不把候选预处理自动启用为正式增强。
    """
    _validate_binding()
    if not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("输入必须是非空HWC RGB uint8 NumPy图")
    h,w=rgb.shape[:2]
    if type(sample.get("width")) is not int or type(sample.get("height")) is not int or (sample["height"],sample["width"]) != (h,w):
        raise ValueError("原图与adapters原shape/宽高不一致")
    for key in ("original_shape","ori_shape"):
        if key in sample and tuple(sample[key]) != (h,w):
            raise ValueError("显式原shape与RGB不一致")
    domain=sample.get("domain")
    uid_key="sample_uid" if domain == "R" else "derived_sample_uid"
    if domain not in ("R","S") or not isinstance(sample.get(uid_key),str) or not sample[uid_key]:
        raise ValueError("需要原样本UID或明确派生UID及R/S域")
    if sample.get("class_mapping") != {"person":0} or sample.get("model_head_classes") != 80:
        raise ValueError("只支持person0/COCO80适配契约")
    if not isinstance(sample.get("targets"),list) or not isinstance(sample.get("ignore_targets"),list):
        raise ValueError("目标与ignore必须显式提供列表，空亦不能缺字段")

    ratio=min(960/h,960/w)
    new_unpad=(round(w*ratio),round(h*ratio))
    dw,dh=(960-new_unpad[0])/2,(960-new_unpad[1])/2
    left,right=round(dw-.1),round(dw+.1)
    top,bottom=round(dh-.1),round(dh+.1)
    image=cv2.resize(rgb,new_unpad,interpolation=cv2.INTER_LINEAR) if (w,h) != new_unpad else rgb
    image=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    if image.shape != (960,960,3):
        raise RuntimeError("候选letterbox输出尺寸异常")
    target_records,ignore_records,boxes,ignores=[],[],[],[]
    for key,destination,supervised in (("targets",target_records,boxes),("ignore_targets",ignore_records,ignores)):
        for record in sample[key]:
            if key == "targets" and record.get("class_id") != 0:
                raise ValueError("目标类必须为person0")
            source=_box(record,w,h)
            item=deepcopy(record)
            if source is not None:
                mapped=source*ratio+np.array([left,top,left,top],dtype=np.float64)
                item["input_xyxy"]=mapped.tolist()
                supervised.append(mapped)
            else:
                item["input_xyxy"]=None
            destination.append(item)
    xyxy=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    normalized=np.empty_like(xyxy)
    normalized[:,:2]=(xyxy[:,:2]+xyxy[:,2:])/2/960
    normalized[:,2:]=(xyxy[:,2:]-xyxy[:,:2])/960
    # 转置后明确copy：输出绝不与调用方原RGB共享可写存储。
    chw=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)))
    return {"img":chw,"cls":torch.zeros((len(boxes),1),dtype=torch.float32),
            "bboxes":torch.from_numpy(normalized),"ignore_bboxes":torch.from_numpy(np.asarray(ignores,dtype=np.float32).reshape(-1,4)),
            "uid":sample[uid_key],"uid_field":uid_key,"domain":domain,
            "targets":target_records,"ignore_targets":ignore_records,
            "absent_targets":deepcopy(sample.get("absent_targets",[])),
            "geometry":{"original_shape":(h,w),"input_shape":(960,960),
                        "resized_shape":(new_unpad[1],new_unpad[0]),"scale":(ratio,ratio),
                        "padding_ltrb":(left,top,right,bottom),
                        "source_uid":sample[uid_key],"channel_order":"RGB", "letterbox_candidate_only":True},
            "supervision":deepcopy(sample.get("supervision",{})),"implementation_binding":dict(LETTERBOX_BINDING)}


def _collate_lineage(sample):
    """保留Dataset请求血缘，不独立认证原GT split或来源文件。

    原split由Dataset的catalogue/reader核验；这里只检查合法split及请求中已有
    split/view的一致性。没有额外证据时不补默认train。五字段全缺仅供原纯几何
    夹具返回None；正式执行调用者必须另外拒绝None，不把它当已绑定样本。
    """
    fields=("request","source_reference","image_source_binding","sampling_stream","data_domain")
    present={key for key in fields if key in sample}
    if not present:
        return None
    if len(present) != len(fields):
        raise ValueError("Dataset血缘五字段必须完整提供，不能部分丢失")
    request,reference,binding=(sample[key] for key in fields[:3])
    if not all(isinstance(value,dict) for value in (request,reference,binding)):
        raise ValueError("请求、来源引用和图像绑定须为字典")
    stream=sample["sampling_stream"]
    if stream not in ("R","S","R_extra"):
        raise ValueError("血缘sampling_stream非法")
    domain="R" if stream == "R_extra" else stream
    if sample["data_domain"] != domain or sample["domain"] != domain:
        raise ValueError("血缘stream/data_domain与样本域不一致")
    if request.get("uid") != sample["uid"] or request.get("domain") not in (stream,domain):
        raise ValueError("血缘请求UID/domain与样本不一致")
    if request.get("stream",request["domain"]) != stream or request.get("data_domain",domain) != domain:
        raise ValueError("血缘请求stream/data_domain与样本不一致")
    for name in ("exposure_index","augmentation_seed","epoch"):
        if type(request.get(name)) is not int or request[name] < 0:
            raise ValueError("血缘请求曝光/增强seed/epoch必须为非负整数")
    if reference.get("domain") != domain:
        raise ValueError("来源引用domain与样本不一致")
    split=reference.get("split")
    allowed=("train","val") if domain == "R" else ("train","dev")
    if split not in allowed:
        raise ValueError("来源引用split非法，不允许test或跨任务改名")
    if ("split" in request and request["split"] != split) or ("split" in sample and sample["split"] != split):
        raise ValueError("来源引用split与请求/样本不一致")
    if "view" in request:
        view=request["view"]
        expected=view if domain == "R" else ("train" if view == "train" else "dev")
        if view not in ("train","val") or split != expected:
            raise ValueError("来源引用split与请求view不一致")
    if domain == "R":
        if not isinstance(reference.get("index_path"),str) or not reference["index_path"]:
            raise ValueError("真实来源引用缺index_path")
    else:
        if any(not isinstance(reference.get(key),str) or not reference[key] for key in ("seed_dir","camera_id")):
            raise ValueError("合成来源引用缺seed_dir/camera_id")
        if type(reference.get("frame_index")) is not int or reference["frame_index"] < 1:
            raise ValueError("合成来源引用frame_index须为正整数")
    if not isinstance(binding.get("path"),str) or not binding["path"] or not Path(binding["path"]).is_absolute():
        raise ValueError("图像来源绑定须保留原绝对path，不在collate重新解析文件")
    identity=binding.get("stat")
    if not isinstance(identity,(tuple,list)) or len(identity) != 4 or any(type(x) is not int or x < 0 for x in identity) or identity[2] == 0:
        raise ValueError("图像来源绑定须保留合法fileID/size/mtime stat")
    # 字段之间、同批重复UID之间、原输入之间均不共享可写血缘对象。
    return {key:deepcopy(sample[key]) for key in fields}


def collate_bbox_samples(samples: list[dict]) -> dict:
    """堆叠确定性样本并逐图保留血缘；正式调用者须拒绝sample_lineage的None。"""
    if not samples:
        raise ValueError("不能collate空批次")
    images,classes,boxes,indices,ignores,ignore_indices=[],[],[],[],[],[]
    sample_lineage=[]
    for i,s in enumerate(samples):
        img=s.get("img")
        if not isinstance(img,torch.Tensor) or img.device.type != "cpu" or img.dtype != torch.uint8 or tuple(img.shape) != (3,960,960) or not img.is_contiguous():
            raise ValueError("图片须为同尺寸CPU连续RGB CHW uint8")
        if s.get("domain") not in ("R","S") or not isinstance(s.get("uid"),str) or not s["uid"]:
            raise ValueError("批次缺UID/域身份")
        if s.get("geometry",{}).get("input_shape") != (960,960):
            raise ValueError("批次几何metadata与图片尺寸不一致")
        for key,columns in (("cls",1),("bboxes",4),("ignore_bboxes",4)):
            v=s.get(key)
            if not isinstance(v,torch.Tensor) or v.device.type != "cpu" or v.dtype != torch.float32 or v.ndim != 2 or v.shape[1] != columns or not torch.isfinite(v).all():
                raise ValueError(f"{key}必须显式为CPU float32 Nx{columns}")
        n=len(s["bboxes"])
        if s["cls"].shape[0] != n or (s["cls"] != 0).any():
            raise ValueError("类别/框行数不一致或目标非person0")
        b=s["bboxes"]
        if ((b[:,:2] < 0) | (b[:,:2] > 1)).any() or ((b[:,2:] <= 0) | (b[:,2:] > 1)).any():
            raise ValueError("归一化目标框非法")
        corners=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),dim=1)
        if (corners < -1e-7).any() or (corners > 1+1e-7).any():
            raise ValueError("目标框越出input范围")
        g=s["ignore_bboxes"]
        if ((g[:,:2] < 0) | (g[:,2:] > 960) | (g[:,2:] <= g[:,:2])).any():
            raise ValueError("ignore必须为图内正面积input XYXY")
        sample_lineage.append(_collate_lineage(s))
        images.append(img);classes.append(s["cls"]);boxes.append(b);indices.append(torch.full((n,),i,dtype=torch.long))
        ignores.append(g);ignore_indices.append(torch.full((len(g),),i,dtype=torch.long))
    return {"img":torch.stack(images),"cls":torch.cat(classes),"bboxes":torch.cat(boxes),
            "batch_idx":torch.cat(indices),"ignore_bboxes":torch.cat(ignores),
            "ignore_batch_idx":torch.cat(ignore_indices),
            "uids":[s["uid"] for s in samples],"domains":[s["domain"] for s in samples],
            "sample_lineage":sample_lineage,
            "geometry":[deepcopy(s["geometry"]) for s in samples],
            "sample_provenance":[{"targets":deepcopy(s["targets"]),"ignore_targets":deepcopy(s["ignore_targets"]),
                                  "absent_targets":deepcopy(s["absent_targets"]),"uid_field":s["uid_field"]} for s in samples],
            "numeric_input":"uint8_RGB_CHW_not_yet_divided_by_255","formal_train_augmentation_implemented":all(s.get("formal_train_augmentation_implemented") is True for s in samples)}
