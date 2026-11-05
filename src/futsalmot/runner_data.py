"""正式driver所需SoccerNet train/val元数据与显式请求DataLoader接线。

不读取test/图像做panel审计，不重划UID，不写来源。DataLoader仅按caller冻结
requests读单图；不reserve/commit/重播种或自行扩作业，workers实际准入由外层实测。
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
from .adapters import RealBBoxReader,_source_stat,_relative,_stat_tuple
from .batches import collate_bbox_samples
from .train_data import TrainingBBoxDataset,ROLE_POLICY

_TEXT_LIMIT=64*1024*1024

def _pool_sha(uids):
    # 与CycleSampler.__init__保持完全同编码、空格及ensure_ascii。
    return hashlib.sha256(json.dumps(tuple(uids),ensure_ascii=True).encode()).hexdigest()

def _text_binding(path,*,expected_identity=None):
    path=Path(path);before=_source_stat(path)
    if before[2]>_TEXT_LIMIT:raise ValueError("小文本绑定超过64MiB")
    if expected_identity is not None and before!=tuple(expected_identity):raise ValueError("已解析小文本来源变化")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        if _stat_tuple(os.fstat(stream.fileno()))!=before:raise ValueError("小文本打开身份变化")
        while block:=stream.read(1024*1024):digest.update(block)
        if _stat_tuple(os.fstat(stream.fileno()))!=before:raise ValueError("小文本读取身份变化")
    if _source_stat(path)!=before:raise ValueError("小文本绑定期间变化")
    return {"path":str(path.absolute()),"identity":before,"sha256":digest.hexdigest(),"bytes":before[2]}

def real_panel(index_path):
    """一次按原sequence/frame次序读取SoccerNet train/val GT；不打开媒体/test。

    返回有序UID/catalogue、val原source_xyxy与主/辅助ignore、角色/小框统计和
    小文本绑定。统计只审计，绝不以大小/可见性筛图或改原框。
    """
    reader=RealBBoxReader(index_path,max_cached_clips=20,max_cached_rows=600_000)
    if reader._source!="soccernet_tracking":raise ValueError("首面板只允许SoccerNet canonical来源")
    if len(reader._clips)>20:raise ValueError("train/val clip超过既定一次解析缓存20，不扩缓存")
    uids={"train":[],"val":[]};catalogues={"train":{},"val":{}};gt={}
    counters={s:Counter() for s in uids};roles={s:Counter() for s in uids};files=[];clip_bindings=[]
    seen=set()
    for sequence in reader._index["sequences"]:
        split=sequence["split"]
        if split not in ("train","val"):continue
        if "seqinfo" not in sequence["files"]:raise ValueError("canonical SoccerNet缺seqinfo原来源引用")
        clip=sequence["clip_uid"];counter=counters[split];counter["clips"]+=1
        for frame in sequence["detection_frames"]:
            frame_uid=sequence["frame_uid_pattern"].format(frame_number=frame)
            uid=sequence["detection_sample_uid_pattern"].format(frame_uid=frame_uid)
            if uid in seen:raise ValueError("train/val检测UID重复")
            seen.add(uid);sample=reader.read(uid,split=split)
            uids[split].append(uid);catalogues[split][uid]={"domain":"R","index_path":str(reader.index_path),"split":split}
            counter["images"]+=1;positive=[];ignore=[];auxiliary=[];role_audit=[]
            ratio=min(960/sample["height"],960/sample["width"])
            for kind,records in (("positive",sample["targets"]),("ignore",sample["ignore_targets"])):
                for record in records:
                    role=record.get("role");roles[split][str(role)]+=1
                    counter["source_records"]+=1
                    valid=record["bbox_supervision_valid"];counter["visible_invalid_records"]+=not valid
                    if kind=="positive":
                        counter["positive_records"]+=1;positive.append(deepcopy(record["source_xyxy"]))
                        if valid:
                            counter["positive_visible_valid_records"]+=1
                            x1,y1,x2,y2=record["clipped_xyxy"];w=(x2-x1)*ratio;h=(y2-y1)*ratio
                            counter["positive_input_wh_lt_2"]+=w<2 or h<2
                            counter["positive_input_wh_not_gt_2"]+=w<=2 or h<=2
                    else:
                        reason=record.get("ignore_reason");counter["role_reason:"+str(reason)]+=1
                        if reason=="non_target_role":
                            counter["known_non_target_background_records"]+=1;auxiliary.append(deepcopy(record["source_xyxy"]))
                        elif reason in ("invalid_gt_flag","unknown_role"):
                            counter["true_ignore_records"]+=1;ignore.append(deepcopy(record["source_xyxy"]))
                        else:raise ValueError("未知角色策略原因")
                    role_audit.append({"track_id":record["track_id"],"role":role,"gt_flag":record["gt_flag"],
                        "source_xyxy":deepcopy(record["source_xyxy"]),"bbox_supervision_valid":valid,
                        "ignore_reason":record.get("ignore_reason"),"derived_training_role":
                        "positive" if kind=="positive" else ("background" if record["ignore_reason"]=="non_target_role" else "ignore")})
            if split=="val":
                gt[uid]={"positive_gt":positive,"ignore":ignore,"known_non_target_auxiliary_ignore":auxiliary,
                         "width":sample["width"],"height":sample["height"],"role_audit":role_audit,"role_policy":ROLE_POLICY}
        # 每个已实际打开clip文本SHA只算一次；不hash媒体。
        if sequence["detection_frames"]:
            original=reader.index_path.parent/sequence["sequence_path"]
            entry={"clip_uid":clip,"split":split,"files":{}}
            for name in ("gt","gameinfo","seqinfo"):
                logical=original/sequence["files"][name];_source_stat(logical)
                path=_relative(reader.index_path.parent,sequence["sequence_path"])/sequence["files"][name]
                expected=reader._bindings[clip].get(name)
                binding=_text_binding(path,expected_identity=None if expected is None else expected[1])
                if name=="gt" and sequence.get("gt_sha256") and binding["sha256"]!=sequence["gt_sha256"]:
                    raise ValueError("原canonical GT SHA绑定不符")
                entry["files"][name]=binding;files.append(binding)
            clip_bindings.append(entry)
    if not uids["train"] or not uids["val"]:raise ValueError("面板要求原train/val均非空")
    reader._check_index()
    for clip in reader._bindings:reader._check_clip(clip)
    # seqinfo也属于冻结来源；最后再stat核验，不二次hash。
    for item in files:
        if _source_stat(item["path"])!=tuple(item["identity"]):raise ValueError("panel收口小文本来源变化")
    statistics={}
    for split in uids:
        counter=counters[split]
        for name in ("clips","images","source_records","visible_invalid_records","positive_records",
                     "positive_visible_valid_records","positive_input_wh_lt_2","positive_input_wh_not_gt_2",
                     "known_non_target_background_records","true_ignore_records",
                     "role_reason:non_target_role","role_reason:invalid_gt_flag","role_reason:unknown_role"):
            counter.setdefault(name,0)
        den=counter["positive_visible_valid_records"]
        statistics[split]={**dict(counter),"roles":dict(roles[split]),
            "positive_input_wh_lt_2_fraction":counter["positive_input_wh_lt_2"]/den if den else None,
            "positive_input_wh_not_gt_2_fraction":counter["positive_input_wh_not_gt_2"]/den if den else None}
    binding={"index":{"path":str(reader.index_path),"sha256":reader._index_sha,"identity":reader._index_stat},
             "ordered_train_pool_sha256":_pool_sha(uids["train"]),"ordered_val_pool_sha256":_pool_sha(uids["val"]),
             "clips":clip_bindings,"role_policy":ROLE_POLICY,"scope":"SoccerNet_original_train_val_only_no_test_no_media",
             "reader_cache_info":reader.cache_info()}
    return {"train_uids":uids["train"],"val_uids":uids["val"],"train_catalogue":catalogues["train"],
            "val_catalogue":catalogues["val"],"val_ground_truth":gt,"statistics":statistics,"binding":binding}

class RequestPlanDataset(Dataset):
    """整数索引只定位caller已冻结请求表，不在worker自行抽UID/计曝光。"""
    def __init__(self,catalogue,requests,*,view="train",**readeroptions):
        if not isinstance(requests,(list,tuple)) or not requests:raise ValueError("需要明确非空请求表")
        self.dataset=TrainingBBoxDataset(catalogue,view=view,**readeroptions)
        self.requests=deepcopy(list(requests))
        for request in self.requests:self.dataset._request(request)
    def __len__(self):return len(self.requests)
    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self.requests):raise IndexError("仅支持显式请求表内非负整数")
        return self.dataset[deepcopy(self.requests[index])]

def worker_init_fn(worker_id):
    """模块级spawn回调；只限制线程，不自己seed Python/NumPy/Torch。"""
    torch.set_num_threads(1);cv2.setNumThreads(0)

def make_loader(dataset,*,batch_size,num_workers,generator,pin_memory=False,allow_initialized_cuda=False):
    """独立CPU generator；workers2/4由caller实测，预取固定1，不自行启GPU。"""
    if not isinstance(dataset,RequestPlanDataset):raise ValueError("需要明确RequestPlanDataset")
    if type(batch_size) is not int or batch_size<1:raise ValueError("batch_size须正整数")
    if type(num_workers) is not int or num_workers not in (0,2,4):raise ValueError("只允许workers0/2/4")
    if not isinstance(generator,torch.Generator) or generator.device.type!="cpu":raise ValueError("须caller独立CPU Generator")
    if type(pin_memory) is not bool:raise ValueError("pin_memory须明确bool")
    if pin_memory and (allow_initialized_cuda is not True or not torch.cuda.is_initialized()):
        raise RuntimeError("pin_memory仅用于caller已授权且已初始化CUDA")
    kwargs={"dataset":dataset,"batch_size":batch_size,"shuffle":False,"drop_last":False,
            "num_workers":num_workers,"pin_memory":pin_memory,"generator":generator,
            "collate_fn":collate_bbox_samples,"worker_init_fn":worker_init_fn}
    if num_workers:kwargs.update(prefetch_factor=1,persistent_workers=True)
    return DataLoader(**kwargs)
