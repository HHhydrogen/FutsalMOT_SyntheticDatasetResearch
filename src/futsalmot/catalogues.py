"""canonical synthetic READY索引→既有BBox Dataset catalogue；只读metadata。

不扫描/打开RGB、mask、JSONL、trajectory或COCO；完整文件表只从既有RELOCATION
一次有界读取。这里复用发布/整理收据，不重做生成验收或授予训练许可。
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from .adapters import _source_stat, _stat_tuple
from .layout import VERSION, _canonical_sha, _expected_seed_files

MAX_RELOCATION_BYTES = 512*1024*1024
MAX_SMALL_METADATA_BYTES = 4*1024*1024


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None,"缺合法SHA绑定")
    return value


def _absolute(value):
    _require(isinstance(value,(str,Path)),"路径须明确")
    path=Path(value)
    _require(path.is_absolute() and ".." not in path.parts,"路径须绝对且不含上跳")
    return path


def _pairs(pairs):
    result={}
    for key,value in pairs:
        _require(key not in result,"metadata重复JSON key")
        result[key]=value
    return result


def _float(value):
    number=float(value)
    _require(math.isfinite(number),"metadata非有限JSON数值")
    return number


def _read(path, maximum, opened):
    path=_absolute(path)
    before=_source_stat(path)
    _require(0<before[2]<=maximum,"metadata大小超本类型上限或为空")
    with path.open("rb") as stream:
        _require(_stat_tuple(__import__("os").fstat(stream.fileno()))==before,"metadata打开期间变化")
        raw=stream.read(before[2]+1)
        _require(len(raw)==before[2] and _stat_tuple(__import__("os").fstat(stream.fileno()))==before,
                 "metadata读取期间变化")
    _require(_source_stat(path)==before,"metadata读取前后变化")
    value=json.loads(raw,object_pairs_hook=_pairs,parse_float=_float,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError("metadata非法常量")))
    _require(isinstance(value,dict),"metadata根须为object")
    binding={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"stat":list(before)}
    opened[str(path)]=binding
    return value,binding


def synthetic_bbox_catalogue(index_path, *, split="train"):
    """返回binding/ordered_uids/catalogue/counts；不读任何样本/GT，不改索引。

    仅train/dev，顺序为原索引seed→原camera_ids→frame升序。UID沿用adapter规则，
    使用各seed原dataset_id/version，保留旧v3捕获。所有RGB进入池，包括合法空目标图；
    不依据面积或目标数筛图。实际媒体等价性沿用PASS精确表，不重新stat媒体。
    """
    _require(split in ("train","dev"),"只允许原train/dev，不开放test")
    path=Path(index_path).absolute()
    _require(path.name=="dataset.json","入口须为canonical dataset.json")
    opened={}
    index,index_binding=_read(path,MAX_SMALL_METADATA_BYTES,opened)
    _require(index.get("schema_version")=="canonical_synthetic_dataset_v1"
             and index.get("status")=="READY" and index.get("readiness")=="READY"
             and index.get("relocation_status")=="PASS","canonical索引必须实际READY/PASS，拒BUILDING/规划")
    _require(index.get("relocation")=="RELOCATION.json","只接受同根明确RELOCATION.json")
    receipt,relocation_binding=_read(path.parent/"RELOCATION.json",MAX_RELOCATION_BYTES,opened)
    _require(receipt.get("schema_version")=="synthetic_relocation_v1"
             and receipt.get("status")=="PASS" and receipt.get("phase")=="ALL_LINKS_VERIFIED"
             and receipt.get("pilot",{}).get("status")=="PASS","整理收据未完整PASS")
    source=index["source_release"]
    _require(isinstance(source,dict) and receipt.get("source_binding")==source,"索引与整理来源绑定不一致")
    source_root=_absolute(source["release_root"])
    _require(source_root.name==VERSION and source_root.parent.name=="synthetic_research_v1"
             and path.parent==source_root.parent.parent/"synthetic"
             and _absolute(receipt["source_root"])==source_root
             and _absolute(receipt["target_root"])==path.parent,"来源/目标根不属于原canonical结构")
    sources={}
    for key,digest_key,name in (("dataset_manifest","manifest_sha256","dataset_manifest.json"),
                                ("validation_receipt","validation_sha256","validation_receipt.json"),
                                ("g5_receipt","g5_sha256","G5_complete.json")):
        source_path=_absolute(source[key])
        _require(source_path.name==name and (key=="g5_receipt" or source_path.parent==source_root),
                 "源绑定不是明确发布小metadata")
        value,binding=_read(source_path,MAX_SMALL_METADATA_BYTES,opened)
        _require(binding["sha256"]==_sha(source[digest_key]),"原发布metadata SHA变化")
        saved=receipt["source_metadata"].get(str(source_path))
        _require(saved is not None and saved["sha256"]==binding["sha256"]
                 and tuple(saved["stat"])==tuple(binding["stat"]),"整理时原metadata身份/SHA与当前不符")
        sources[key]=value
    manifest,validation,g5=(sources[key] for key in ("dataset_manifest","validation_receipt","g5_receipt"))
    _require(manifest["schema"]=="futsalmot_dataset_manifest_v1"
             and manifest["dataset_version"]==source["dataset_version"]==VERSION
             and manifest["diagnostic_only"] is False,"非绑定正式版本")
    _require(_canonical_sha({k:v for k,v in manifest.items() if k!="fingerprint"})==manifest["fingerprint"]
             ==source["fingerprint"]==validation["fingerprint"]==g5["fingerprint"],"发布fingerprint不符")
    _require(validation["status"]=="PASS" and validation["dataset_manifest_sha256"]==source["manifest_sha256"]
             and validation["all_owned_sessions_normal_exit"] is True
             and g5["status"]=="published" and _absolute(g5["release_path"])==source_root
             and g5["manifest_sha256"]==source["manifest_sha256"]
             and g5["validation_sha256"]==source["validation_sha256"],"根PASS/G5发布未闭合")
    rows=index["seeds"]
    _require(isinstance(rows,list) and len(rows)==20
             and len({r["seed_id"] for r in rows})==20
             and sum(r["split"]=="train" for r in rows)==16
             and sum(r["split"]=="dev" for r in rows)==4,"原20seed/16train4dev不齐")
    original={r["seed_id"]:r for r in manifest["seeds"]}
    _require(len(original)==20 and set(original)=={r["seed_id"] for r in rows},"canonical与发布seed集合不符")
    expected_files=set()
    counts={s:{"seeds":0,"cameras":0,"images":0} for s in ("train","dev")}
    total_files=total_bytes=0
    for row in rows:
        _require({k:row[k] for k in original[row["seed_id"]]}==original[row["seed_id"]],
                 "canonical改写原发布seed row")
        seed_id=row["seed_id"]
        _require(type(row["root_seed"]) is int and seed_id==f'seed_{row["root_seed"]:08x}',
                 "seed ID原数字身份不符")
        _require(row["split"] in counts and row["directory"]==row["split"]+"/"+seed_id,
                 "seed目录必须严格split/seed_id")
        _require(type(row["frames_per_camera"]) is int and row["frames_per_camera"]>0,
                 "实际帧数须正整数")
        cameras=row["camera_ids"]
        _require(isinstance(cameras,list) and len(cameras)==len(set(cameras))
                 and 6<=len(cameras)<=7 and {"C5","C1","C2","C3","C4"}<=set(cameras)
                 and set(cameras)<={"C5","C1","C2","C3","C4","P01","P02","P03","P04"},"原机位清单非法")
        f=row["frames_per_camera"]*len(cameras)
        c=row["counts"]
        _require(c["rgb_frames"]==c["instance_masks"]==c["number_masks"]==f
                 and c["files"]==3*f+2*len(cameras)+3,"seed原数量不齐")
        for name in _expected_seed_files(row):
            expected_files.add(row["directory"]+"/"+name)
        counts[row["split"]]["seeds"]+=1
        counts[row["split"]]["cameras"]+=len(cameras)
        counts[row["split"]]["images"]+=f
        total_files+=c["files"];total_bytes+=c["bytes"]
    _require(index["totals"]==manifest["totals"] and index["totals"]["files"]==total_files
             and index["totals"]["bytes"]==total_bytes
             and index["totals"]["rgb_frames"]==sum(c["images"] for c in counts.values()),
             "canonical totals与原发布不齐")
    files=receipt["files"]
    _require(isinstance(files,list) and len(files)==total_files==receipt["expected_linked_files"]
             ==receipt["created_links"]==receipt["verified_links"],"精确整理文件计数不齐")
    entries={}
    for entry in files:
        relative=entry["target_relative"]
        _require(relative==entry["source_relative"] and relative in expected_files and relative not in entries,
                 "精确表有额外/越界/重复或改名路径")
        _require(type(entry["bytes"]) is int and entry["bytes"]>=0
                 and type(entry["mtime_ns"]) is int and isinstance(entry["file_id"],dict)
                 and all(type(entry["file_id"].get(k)) is int and entry["file_id"][k]>=0 for k in ("dev","ino")),
                 "精确表stat字段非法")
        entries[relative]=entry
    _require(set(entries)==expected_files and sum(e["bytes"] for e in files)==total_bytes
             ==receipt["logical_linked_media_and_annotation_bytes"]==receipt["destination_logical_bytes"]
             and receipt["media_copy_bytes"]==0 and receipt["original_release_retained"] is True
             and receipt["original_receipts_rewritten"] is False
             and len(receipt["completed_seed_directories"])==20
             and set(receipt["completed_seed_directories"])=={r["directory"] for r in rows},"整理全表/字节/完成seed不齐")
    uids=[];catalogue={};seed_bindings={}
    for row in rows:
        seed_path=path.parent/row["directory"]/"seed_manifest.json"
        seed,binding=_read(seed_path,MAX_SMALL_METADATA_BYTES,opened)
        _require(binding["sha256"]==_sha(row["seed_manifest_sha256"]),"seed manifest原SHA不符")
        entry=entries[row["directory"]+"/seed_manifest.json"]
        _require(tuple(binding["stat"])==(entry["file_id"]["dev"],entry["file_id"]["ino"],entry["bytes"],entry["mtime_ns"]),
                 "seed manifest与整理时原fileID/size/mtime不符")
        _require(seed["schema"]=="futsalmot_seed_manifest" and seed["schema_version"]=="1.0.0"
                 and seed["seed_id"]==row["seed_id"] and seed["root_seed"]==f'{row["root_seed"]:08x}'
                 and seed["split"]==row["split"] and seed["dataset_id"]==manifest["dataset_id"]
                 and seed["dataset_version"]==row["original_seed_dataset_version"]
                 and seed["frame_count"]==row["frames_per_camera"] and seed["camera_ids"]==row["camera_ids"]
                 and seed["trajectory_uid"]==row["trajectory_uid"],"seed原身份/版本/帧/机位/trajectory绑定不符")
        seed_bindings[row["directory"]]=binding
        if row["split"]!=split:
            continue
        for camera in row["camera_ids"]:
            for frame in range(1,row["frames_per_camera"]+1):
                uid=f"detect:synthetic:{seed['dataset_id']}:{seed['dataset_version']}:{seed['seed_id']}:{camera}:{frame:06d}"
                _require(uid not in catalogue,"派生UID重复")
                uids.append(uid)
                catalogue[uid]={"domain":"S","seed_dir":str(seed_path.parent),"camera_id":camera,
                                "frame_index":frame,"split":split}
    _require(len(uids)==counts[split]["images"],"实际展开UID数与原整图数量不齐")
    for name,binding in opened.items():
        _require(_source_stat(name)==tuple(binding["stat"]),"metadata在catalogue构造期间变化")
    return {"binding":{"index":index_binding,"relocation":{**relocation_binding,"entries":len(files)},
                       "source_release":deepcopy(source),"seed_manifests":seed_bindings,
                       "ordered_pool_sha256":hashlib.sha256(json.dumps(tuple(uids),ensure_ascii=True).encode()).hexdigest(),
                       "order":"index_seed_then_original_camera_ids_then_ascending_frame",
                       "read_scope":"metadata_only_no_media_GT_JSONL_COCO_scan"},
            "ordered_uids":uids,"catalogue":catalogue,
            "counts":{"split":split,**counts[split],"unique_uids":len(catalogue),"all_splits":counts}}
