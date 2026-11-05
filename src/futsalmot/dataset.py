"""显式请求驱动的候选BBox Dataset接口；不是正式DataLoader或训练授权。

catalogue由调用者冻结UID/来源/split/发布绑定；本模块不扫描目录、不重划数据，
不自行抽样、播种、更新sampler或commit。只接既有reader与确定性letterbox候选。
"""
from __future__ import annotations
from copy import deepcopy
import io
import os
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .adapters import RealBBoxReader,SyntheticBBoxReader,_source_stat,_stat_tuple
from .batches import letterbox_bbox_sample


def _positive_int(value,name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}必须为正整数")
    return value


def decode_rgb_image(path,*,expected_shape,max_encoded_bytes=64*1024*1024,
                     max_pixels=16*1024*1024,expected_stat=None):
    """Unicode安全单图解码，header先于OpenCV分配，返回RGB uint8及stat绑定。

    只接静态RGB PNG/JPEG；不静默把灰度/CMYK/alpha当RGB。EXIF旋转被忽略，
    以原标注像素坐标为准。stat绑定不是媒体内容hash或数据验收证明。
    """
    max_encoded_bytes=_positive_int(max_encoded_bytes,"max_encoded_bytes")
    max_pixels=_positive_int(max_pixels,"max_pixels")
    if not isinstance(expected_shape,(list,tuple)) or len(expected_shape) != 2:
        raise ValueError("expected_shape须为原(height,width)")
    h,w=expected_shape
    _positive_int(h,"height");_positive_int(w,"width")
    if h*w > max_pixels:
        raise ValueError("原标注像素数超解码预算")
    path=Path(path)
    before=_source_stat(path) # 含原始逻辑路径每层，拒绝symlink/junction/非普通文件。
    if expected_stat is not None and before != tuple(expected_stat):
        raise ValueError("图像来源身份/size/mtime变化，拒绝混版本")
    if before[2] <= 0 or before[2] > max_encoded_bytes:
        raise ValueError("图像编码字节数为空或超预算")
    with path.open("rb") as f:
        if _stat_tuple(os.fstat(f.fileno())) != before:
            raise ValueError("图像打开时来源变化")
        encoded=f.read(max_encoded_bytes+1)
        if len(encoded) != before[2] or _stat_tuple(os.fstat(f.fileno())) != before:
            raise ValueError("图像读取时来源变化")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(encoded)) as header:
                if header.format not in ("PNG","JPEG") or header.mode != "RGB":
                    raise ValueError("仅支持原始RGB PNG/JPEG")
                if header.size != (w,h) or header.width*header.height > max_pixels:
                    raise ValueError("图像header尺寸与原标注不符或超预算")
                if getattr(header,"is_animated",False):
                    raise ValueError("不支持动画图像")
                header.verify() # 检查PNG等完整性，不解压成大RGB数组。
    except (OSError,SyntaxError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise ValueError("图像header损坏或解码安全限制拒绝") from exc
    image=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.dtype != np.uint8 or image.shape != (h,w,3):
        raise ValueError("OpenCV解码损坏或原像素尺寸不一致")
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if _source_stat(path) != before:
        raise ValueError("图像解码期间来源变化")
    return np.ascontiguousarray(rgb),before


class RequestBBoxDataset:
    """按主进程的显式曝光请求读样本；__getitem__(整数)明确拒绝。

    view=train只接R train/S train，view=val只接R val/S dev，test始终拒绝。
    catalogue[uid] = {domain:R,index_path,split:train/val} 或
    {domain:S,seed_dir,camera_id,frame_index,split:train/dev}。
    所有引用仅来自此显式catalogue，reader按已指定来源复用；不缓存RGB/增强结果。
    Windows pickle交给reader自己的冷缓存协议，保留其已知来源绑定。
    """
    def __init__(self,catalogue:dict,*,view="train",real_reader_options=None,synthetic_reader_options=None,
                 max_encoded_bytes=64*1024*1024,max_pixels=16*1024*1024):
        if not isinstance(catalogue,dict) or not catalogue:
            raise ValueError("需要调用者明确冻结的非空UID catalogue")
        if view not in ("train","val"):
            raise ValueError("候选view只允许train/val，不开放test")
        self.view=view
        self._catalogue=deepcopy(catalogue)
        for uid,ref in self._catalogue.items():
            if not isinstance(uid,str) or not uid or not isinstance(ref,dict):
                raise ValueError("catalogue需要非空字符串UID和来源字典")
            domain=ref.get("domain")
            if domain == "R":
                if set(ref) != {"domain","index_path","split"} or ref["split"] not in ("train","val"):
                    raise ValueError("真实引用仅支持明确train/val，不开放test")
                if ref["split"] != view:
                    raise ValueError("真实source split与显式Dataset view不一致")
                ref["index_path"]=str(Path(ref["index_path"]).absolute())
            elif domain == "S":
                if set(ref) != {"domain","seed_dir","camera_id","frame_index","split"} or ref["split"] not in ("train","dev"):
                    raise ValueError("合成引用须明确seed/camera/frame及原split")
                if ref["split"] != ("train" if view == "train" else "dev"):
                    raise ValueError("合成source split与显式Dataset view不一致")
                _positive_int(ref["frame_index"],"frame_index")
                if not isinstance(ref["camera_id"],str) or not ref["camera_id"]:
                    raise ValueError("合成camera_id必须明确")
                ref["seed_dir"]=str(Path(ref["seed_dir"]).absolute())
            else:
                raise ValueError("catalogue数据域只允许R/S；R_extra是采样流")
        self.real_reader_options=deepcopy(real_reader_options or {})
        self.synthetic_reader_options=deepcopy(synthetic_reader_options or {})
        self.max_encoded_bytes=_positive_int(max_encoded_bytes,"max_encoded_bytes")
        self.max_pixels=_positive_int(max_pixels,"max_pixels")
        self._readers={}
        self._image_bindings={}

    def __len__(self):
        return len(self._catalogue)

    def _request(self,request):
        if not isinstance(request,dict):
            raise ValueError("只接受主进程明确请求，不支持整数顺序采样")
        required={"uid","domain","exposure_index","augmentation_seed"}
        if required-request.keys():
            raise ValueError("请求缺UID/stream-domain/曝光/增强seed")
        req=deepcopy(request)
        if "view" in req and req["view"] != self.view:
            raise ValueError("请求view与Dataset view不一致")
        stream=req.get("stream",req["domain"])
        if stream not in ("R","S","R_extra"):
            raise ValueError("未知采样stream")
        data_domain="R" if stream == "R_extra" else stream
        if req["domain"] not in (stream,data_domain) or req.get("data_domain",data_domain) != data_domain:
            raise ValueError("采样stream与实际数据域不一致")
        for name in ("exposure_index","augmentation_seed","epoch"):
            if name in req and (type(req[name]) is not int or req[name] < 0):
                raise ValueError("曝光/增强seed/epoch必须为非负整数")
        uid=req["uid"]
        if not isinstance(uid,str) or uid not in self._catalogue:
            raise ValueError("UID不在调用者显式catalogue中")
        ref=self._catalogue[uid]
        if ref["domain"] != data_domain or ("split" in req and req["split"] != ref["split"]):
            raise ValueError("请求数据域/split与catalogue不一致")
        return req,ref,stream,data_domain

    def _read(self,uid,ref):
        domain=ref["domain"]
        key=(domain,ref["index_path"]) if domain == "R" else (domain,ref["seed_dir"],ref["split"])
        if key not in self._readers:
            if domain == "R":
                self._readers[key]=RealBBoxReader(ref["index_path"],**self.real_reader_options)
            else:
                self._readers[key]=SyntheticBBoxReader(ref["seed_dir"],split=ref["split"],**self.synthetic_reader_options)
        reader=self._readers[key]
        if domain == "R":
            sample=reader.read(uid,split=ref["split"])
            # 保留reader当前索引中的逻辑路径用于reparse检查，避免resolve掩盖img1链接。
            clip,frame,_=reader._lookup[(ref["split"],uid)]
            seq=reader._clips[clip]
            logical=reader.index_path.parent/seq["sequence_path"]/seq["image_file_pattern"].format(frame_number=frame)
            actual_uid=sample.get("sample_uid")
        else:
            sample=reader.read(ref["camera_id"],ref["frame_index"],split=ref["split"])
            logical=Path(ref["seed_dir"])/ref["camera_id"]/"img1"/f"{ref['frame_index']:06d}.png"
            actual_uid=sample.get("derived_sample_uid")
        if actual_uid != uid or sample.get("domain") != domain or sample.get("split") != ref["split"]:
            raise ValueError("reader读回UID/domain/原split与catalogue不符")
        _source_stat(logical)
        if logical.resolve() != Path(sample["image_path"]).resolve():
            raise ValueError("reader图像引用与原逻辑路径不符")
        return sample,logical

    def __getitem__(self,request):
        req,ref,stream,data_domain=self._request(request)
        sample,path=self._read(req["uid"],ref)
        path_key=str(path.absolute())
        rgb,identity=decode_rgb_image(path,expected_shape=(sample["height"],sample["width"]),
                                      max_encoded_bytes=self.max_encoded_bytes,max_pixels=self.max_pixels,
                                      expected_stat=self._image_bindings.get(path_key))
        prepared=letterbox_bbox_sample(sample,rgb)
        if _source_stat(path) != identity:
            raise ValueError("letterbox期间图像来源变化")
        self._image_bindings[path_key]=identity
        prepared.update(request=deepcopy(req),source_reference=deepcopy(ref),
                        sampling_stream=stream,data_domain=data_domain,
                        image_source_binding={"path":path_key,"stat":identity},
                        candidate_only=True,random_augmentation_implemented=False,
                        caller_release_validation_required=True)
        return prepared

    def __getstate__(self):
        for path,identity in self._image_bindings.items():
            if _source_stat(path) != identity:
                raise ValueError("pickle前图像来源变化")
        # 不复制/压扁reader；pickle会调用其__getstate__保留绑定并丢弃索引缓存。
        return dict(self.__dict__)

    def __setstate__(self,state):
        self.__dict__.update(state)
        for path,identity in self._image_bindings.items():
            if _source_stat(path) != identity:
                raise ValueError("pickle冷重建时图像来源变化")
