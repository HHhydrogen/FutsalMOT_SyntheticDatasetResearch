"""未来独立run可用的确定性LetterBox缓存；不自动构建或接入活动训练。

仅无损缓存随机增强前CPU uint8整图和prepared标注，读取后仍调用原随机增强。
v1为三文件pack；v2为arrays/16个原seed.npy加cache.json、samples.jsonl，保留旧v1。
生产固定全24149张S train、D目标、128GiB上限与100GiB余量；默认总活动预算4h，
v2仅在调用方另获明确许可并冻结approved_max_seconds及绝对截止后才可使用新预算。
常态写入/消费最多1个32行映射；v1前缀D→D复制最多2个32行映射（读、写各一个）。
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import base64
import builtins
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from .adapters import _source_stat
from .batches import LETTERBOX_BINDING
from .dataset import RequestBBoxDataset
from .runner_data import RequestPlanDataset
from .train_data import prepare_training_sample
from .preflight import _safe_path

RESEARCH = Path("D:/projects/FutsalMOT_SyntheticDatasetResearch")
DEFAULT_TARGET = RESEARCH/"cache/synthetic_bbox_960_v1"
SOURCE_INDEX = Path("G:/FutsalMOT_Dataset/synthetic/dataset.json")
SCHEMA = "futsalmot_prepared_letterbox_cache_v1"
COUNT = 24149
GIB = 1024**3
MAX_BYTES = 128*GIB
MIN_FREE = 100*GIB
MAX_SECONDS = 4*3600
MAX_MANIFEST = 64*1024**2
MAX_SAMPLES = 2*GIB
MAX_ROW = 4*1024**2
_DYNAMIC = frozenset(("request","sampling_stream","data_domain"))
_DTYPES = {"float32":np.dtype("float32"),"int64":np.dtype("int64"),"uint8":np.dtype("uint8")}


def cache_requirements(count=COUNT):
    if type(count) is not int or count<1:raise ValueError("缓存图数须正整数")
    header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{"descr":"|u1","fortran_order":False,"shape":(count,3,960,960)})
    images=count*3*960*960+len(header.getvalue())
    upper=images+MAX_SAMPLES+MAX_MANIFEST
    return {"count":count,"shape":[count,3,960,960],"dtype":"uint8","image_payload_bytes":count*3*960*960,
            "npy_header_bytes":len(header.getvalue()),"images_npy_bytes":images,
            "metadata_upper_bytes":MAX_SAMPLES+MAX_MANIFEST,"complete_upper_bytes":upper,
            "cache_ceiling_bytes":MAX_BYTES,"reserve_bytes":MIN_FREE,"required_free_bytes":upper+MIN_FREE}


def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _json(value):return (json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n").encode("utf-8")
def _stat(path):return list(_source_stat(path))


def _encode(value):
    if isinstance(value,torch.Tensor):
        if value.device.type!="cpu" or value.dtype not in (torch.float32,torch.int64,torch.uint8):raise ValueError("缓存只编码既定CPU Tensor类型")
        array=value.detach().contiguous().numpy()
        return {"__cache_type__":"tensor","dtype":str(array.dtype),"shape":list(array.shape),"base64":base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value,tuple):return {"__cache_type__":"tuple","items":[_encode(x) for x in value]}
    if isinstance(value,list):return [_encode(x) for x in value]
    if isinstance(value,dict):
        if "__cache_type__" in value or any(type(k) is not str for k in value):raise ValueError("缓存metadata键不合法")
        return {k:_encode(v) for k,v in value.items()}
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float and math.isfinite(value):return value
    raise ValueError("不支持或非有限的prepared metadata类型")


def _decode(value):
    if isinstance(value,list):return [_decode(x) for x in value]
    if isinstance(value,dict):
        kind=value.get("__cache_type__")
        if kind=="tuple" and set(value)=={"__cache_type__","items"}:return tuple(_decode(x) for x in value["items"])
        if kind=="tensor" and set(value)=={"__cache_type__","dtype","shape","base64"}:
            dtype=_DTYPES.get(value["dtype"]);shape=value["shape"]
            if dtype is None or not isinstance(shape,list) or len(shape)>4 or any(type(n) is not int or n<0 for n in shape):raise ValueError("缓存Tensor shape/dtype非法")
            raw=base64.b64decode(value["base64"],validate=True)
            if len(raw)!=math.prod(shape)*dtype.itemsize:raise ValueError("缓存Tensor字节与shape不齐")
            return torch.from_numpy(np.frombuffer(raw,dtype=dtype).reshape(shape).copy())
        if kind is not None:raise ValueError("未知缓存metadata类型标记")
        return {k:_decode(v) for k,v in value.items()}
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float and math.isfinite(value):return value
    raise ValueError("缓存metadata数值类型非法")


def _equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def _implementation():
    base=Path(__file__).parent
    return {"letterbox_binding":dict(LETTERBOX_BINDING),"code_sha256":{
        name:_sha((base/name).read_bytes()) for name in ("adapters.py","batches.py","dataset.py","train_data.py","input_cache.py")}}


def _file_binding(path,check=None):
    path=_safe_path(path);identity=_stat(path);digest=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(1024**2):
            if check is not None:check()
            digest.update(block)
    if _stat(path)!=identity:raise ValueError("来源在绑定期间变化")
    return {"path":str(path),"stat":identity,"sha256":digest.hexdigest()}


def _check_binding(row,*,hash_contents=False):
    path=Path(row["path"])
    if _stat(path)!=row["stat"]:raise ValueError("cache来源身份/size/mtime变化: "+str(path))
    if hash_contents and _file_binding(path)["sha256"]!=row["sha256"]:raise ValueError("cache来源metadata/GT SHA变化: "+str(path))


WINDOW_ROWS = 32
ROW_BYTES = 3*960*960


def _image_header(path,count):
    path=Path(path)
    with path.open("rb") as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError("缓存仅接受原NPY1.0 header")
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096)
        offset=f.tell()
    if shape!=(count,3,960,960) or fortran or dtype!=np.uint8 or offset!=128:
        raise ValueError("缓存NPY header/shape/dtype/布局改变")
    if path.stat().st_size!=offset+count*ROW_BYTES:raise ValueError("缓存预分配文件大小不符")
    return offset


def _create_images(path,count):
    header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{"descr":"|u1","fortran_order":False,"shape":(count,3,960,960)})
    with Path(path).open("xb") as f:
        f.write(header.getvalue());f.truncate(len(header.getvalue())+count*ROW_BYTES);f.flush();os.fsync(f.fileno())


class _WindowedArray:
    """最多32行view；flush不释放工作集，故每窗口明确close后才开下一view。"""
    def __init__(self,path,count,*,mode="r",window_rows=WINDOW_ROWS):
        if mode not in ("r","r+") or type(window_rows) is not int or not 1<=window_rows<=64:
            raise ValueError("缓存映射仅r/r+且窗口1..64行")
        self.path=Path(path);self.count=count;self.mode=mode;self.window_rows=window_rows
        self.offset=_image_header(self.path,count);self.shape=(count,3,960,960);self.dtype=np.dtype("uint8")
        self._view=None;self._start=0;self._end=0;self.max_mapped_bytes=0;self.windows_opened=0
    def _at(self,index):
        if type(index) is not int or not 0<=index<self.count:raise IndexError("缓存行号越界")
        if self._view is None or not self._start<=index<self._end:
            self.close();self._start=(index//self.window_rows)*self.window_rows;self._end=min(self.count,self._start+self.window_rows)
            self._view=np.memmap(self.path,mode=self.mode,dtype=np.uint8,offset=self.offset+self._start*ROW_BYTES,
                                 shape=(self._end-self._start,3,960,960),order="C")
            self.max_mapped_bytes=max(self.max_mapped_bytes,self._view.nbytes);self.windows_opened+=1
        return index-self._start
    def write_exact(self,index,pixels):
        if self.mode!="r+":raise ValueError("只读缓存view不能写")
        local=self._at(index);self._view[local]=pixels
        if not np.array_equal(self._view[local],pixels):raise RuntimeError("缓存960像素读回不一致")
    def __getitem__(self,index):
        local=self._at(index)
        return np.array(self._view[local],dtype=np.uint8,copy=True,order="C")
    def flush(self):
        if self._view is not None and self.mode=="r+":self._view.flush()
    def close(self):
        if self._view is not None:
            self.flush();self._view._mmap.close();self._view=None;self._start=self._end=0


def _close_array(array):
    if array is not None:
        if isinstance(array,_WindowedArray) or isinstance(array,_SeedArrayStore):array.close();return
        if hasattr(array,"flush"):array.flush()
        mmap=getattr(array,"_mmap",None)
        if mmap is not None:mmap.close()


def _write_manifest(path,value):
    raw=_json(value)
    if len(raw)>MAX_MANIFEST:raise ValueError("cache.json超过64MiB上限")
    with path.open("wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())


def _build_pack(target, ordered_uids, prepared_factory, *, source_binding, source_files,
                dependencies, production, max_seconds=MAX_SECONDS, free_reserve=MIN_FREE, guard=None,
                resume_prefix=None,window_rows=WINDOW_ROWS):
    """内部唯一writer；resume_prefix只能来自已完整核验的现存前缀。"""
    target=_safe_path(target);partial=target.with_name(target.name+".partial")
    if target.exists() or (partial.exists() and resume_prefix is None):raise FileExistsError("缓存或.partial已存在，禁止覆盖/自动重建")
    if type(production) is not bool or not ordered_uids or len(ordered_uids)!=len(set(ordered_uids)):raise ValueError("缓存UID须非空唯一")
    if production and len(ordered_uids)!=COUNT:raise ValueError("正式缓存必须含全部24149个S train UID")
    requirement=cache_requirements(len(ordered_uids))
    if requirement["complete_upper_bytes"]>MAX_BYTES:raise ValueError("完整缓存预算超过128GiB")
    additional=requirement["complete_upper_bytes"] if resume_prefix is None else MAX_SAMPLES+MAX_MANIFEST
    if shutil.disk_usage(target.parent).free<free_reserve+additional:raise RuntimeError("D盘缓存剩余完整上界加100GiB余量不足")
    for row in source_files.values():_check_binding(row)
    started=time.monotonic();array=None;prior_active=0.
    if resume_prefix is None:
        partial.mkdir();rows=[]
        manifest={"schema":SCHEMA,"state":"BUILDING","kind":"production" if production else "cpu_fixture",
                  "count":len(ordered_uids),"source_binding":deepcopy(source_binding),"source_files":deepcopy(source_files),
                  "implementation":_implementation(),"budget":requirement,"writer":{"pid":os.getpid()},
                  "format":{"images":"images.npy","samples":"samples.jsonl","layout":"NCHW_RGB_uint8_960_before_random_augmentation"}}
    else:
        if not partial.is_dir():raise ValueError("恢复只能使用现有精确.partial")
        if _stat(partial/"images.npy")!=resume_prefix["images_stat"] or _stat(partial/"samples.jsonl")!=resume_prefix["samples_stat"]:
            raise ValueError("前缀核验后缓存文件变化，拒绝append")
        manifest=deepcopy(resume_prefix["manifest"]);rows=deepcopy(resume_prefix["rows"])
        if manifest["source_binding"]!=source_binding or manifest["source_files"]!=source_files:
            raise ValueError("恢复来源绑定不符")
        if [r["uid"] for r in rows]!=list(ordered_uids[:len(rows)]):raise ValueError("恢复前缀不是原UID顺序")
        prior_active=float(manifest.get("active_elapsed_seconds_total",manifest["elapsed_seconds"]))
        recovery=deepcopy(resume_prefix["recovery"]);manifest.update(state="BUILDING",recovery=recovery,implementation=_implementation())
        manifest.pop("error",None)
    first=len(rows);manifest.update(writer={"pid":os.getpid(),"version":"windowed_v2","window_rows":window_rows},
        completed_rows=first,active_elapsed_seconds_prior=prior_active)
    def check():
        if guard is not None:guard()
        if time.monotonic()-started>=max_seconds:raise TimeoutError("缓存有界剩余时间耗尽")
        if shutil.disk_usage(partial).free<free_reserve:raise RuntimeError("缓存写入时D盘100GiB余量不足")
    try:
        _write_manifest(partial/"cache.json",manifest)
        if resume_prefix is None:_create_images(partial/"images.npy",len(ordered_uids))
        array=_WindowedArray(partial/"images.npy",len(ordered_uids),mode="r+",window_rows=window_rows)
        with (partial/"samples.jsonl").open("ab" if resume_prefix is not None else "xb") as out:
            for index in range(first,len(ordered_uids)):
                uid=ordered_uids[index];check();sample=prepared_factory(uid);image=sample.get("img")
                if (not isinstance(image,torch.Tensor) or image.device.type!="cpu" or image.dtype!=torch.uint8
                    or tuple(image.shape)!=(3,960,960) or sample.get("uid")!=uid or sample.get("domain")!="S"
                    or sample.get("source_reference",{}).get("split")!="train"
                    or sample.get("random_augmentation_implemented") is not False
                    or sample.get("geometry",{}).get("letterbox_candidate_only") is not True):
                    raise ValueError("只能缓存原Dataset的S train增强前LetterBox输出")
                binding=sample["image_source_binding"]
                if _stat(binding["path"])!=list(binding["stat"]):raise ValueError("原RGB在缓存时变化")
                metadata={k:deepcopy(v) for k,v in sample.items() if k!="img" and k not in _DYNAMIC}
                encoded=_encode(metadata);decoded=_decode(encoded)
                if not _equal(metadata,decoded):raise RuntimeError("prepared metadata无损读回不一致")
                raw=_json({"uid":uid,"row":index,"metadata":encoded})
                if len(raw)>MAX_ROW or out.tell()+len(raw)>MAX_SAMPLES:raise ValueError("samples.jsonl大小超界")
                pixels=image.detach().contiguous().numpy();array.write_exact(index,pixels)
                dep=dependencies[uid]
                if any(path not in source_files for path in dep):raise ValueError("样本GT/metadata来源绑定不完整")
                row={"uid":uid,"row":index,"offset":out.tell(),"bytes":len(raw),"sha256":_sha(raw),
                     "pixel_sha256":_sha(pixels.tobytes()),"source_files":list(dep)}
                # 行完整写出后才增加已提交计数；异常不猜测/截断未知尾部。
                out.write(raw);rows.append(row)
                if (index+1)%window_rows==0:
                    out.flush();array.close();manifest.update(completed_rows=index+1,elapsed_seconds=time.monotonic()-started,
                        active_elapsed_seconds_total=prior_active+time.monotonic()-started,
                        mapped_view_peak_bytes=array.max_mapped_bytes)
                    _write_manifest(partial/"cache.json",manifest)
            out.flush();os.fsync(out.fileno())
        mapped_peak=array.max_mapped_bytes;windows=array.windows_opened;_close_array(array);array=None;check()
        for row in source_files.values():_check_binding(row)
        if _implementation()!=manifest["implementation"]:raise RuntimeError("缓存生成期间预处理代码变化")
        if sum(p.stat().st_size for p in partial.iterdir())>MAX_BYTES:raise RuntimeError("缓存实际大小超过128GiB")
        manifest.update(state="READY",rows=rows,images_file={"stat":_stat(partial/"images.npy")},
            samples_file={"stat":_stat(partial/"samples.jsonl")},elapsed_seconds=time.monotonic()-started,
            active_elapsed_seconds_total=prior_active+time.monotonic()-started,mapped_view_peak_bytes=mapped_peak,
            acceptance={"new_rows_pixels_write_read_exact":True,"all_metadata_roundtrip_exact":True,
                        "random_augmentation_cached":False,"prefix_inherited":first})
        if manifest["images_file"]["stat"][2]!=requirement["images_npy_bytes"]:raise RuntimeError("images.npy精确大小不符")
        _write_manifest(partial/"cache.json",manifest)
        if target.exists():raise FileExistsError("发布前目标已存在，不覆盖")
        os.rename(partial,target)
        return {"state":"READY","target":str(target),"files":3,"count":len(rows),"reused_prefix_rows":first,
                "images_npy_bytes":requirement["images_npy_bytes"],"total_bytes":sum(p.stat().st_size for p in target.iterdir()),
                "manifest_sha256":_sha((target/"cache.json").read_bytes()),"elapsed_seconds":manifest["elapsed_seconds"],
                "mapped_view_peak_bytes":mapped_peak,"windows_opened":windows}
    except BaseException as exc:
        peak=None if array is None else array.max_mapped_bytes;_close_array(array);array=None
        manifest.pop("rows",None)
        manifest.update(state="FAIL_STOP",completed_rows=len(rows),elapsed_seconds=time.monotonic()-started,
                        active_elapsed_seconds_total=prior_active+time.monotonic()-started,mapped_view_peak_bytes=peak,
                        error={"type":type(exc).__name__,"message":str(exc)})
        _write_manifest(partial/"cache.json",manifest)
        raise
    finally:_close_array(array)


def _system_memory():
    """Windows物理available与系统commit余量；不改pagefile。"""
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong),
                  ("total_phys",ctypes.c_ulonglong),("avail_phys",ctypes.c_ulonglong),
                  ("total_pagefile",ctypes.c_ulonglong),("avail_pagefile",ctypes.c_ulonglong),
                  ("total_virtual",ctypes.c_ulonglong),("avail_virtual",ctypes.c_ulonglong),
                  ("avail_extended_virtual",ctypes.c_ulonglong)]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise OSError("GlobalMemoryStatusEx失败")
    return {"physical_available_bytes":int(status.avail_phys),"commit_available_bytes":int(status.avail_pagefile)}


def _guard_memory():
    memory=_system_memory()
    if memory["physical_available_bytes"]<4*GIB or memory["commit_available_bytes"]<8*GIB:
        raise RuntimeError("缓存CPU物理4GiB/commit8GiB余量不足: "+repr(memory))
    return memory


_WORKER_DATASET=None

def _worker_init(catalogue):
    global _WORKER_DATASET
    # 仅本缓存CPU worker；不影响并行的已冻结GPU训练进程环境。
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    import cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    _WORKER_DATASET=RequestBBoxDataset(catalogue,view="train",synthetic_reader_options={"max_indexed_cameras":7,"max_indexed_frames":3000})


def _prepare_worker(uid):
    return _WORKER_DATASET[{"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,"exposure_index":0,"augmentation_seed":0}]


def build_input_cache(index_path=SOURCE_INDEX, target=DEFAULT_TARGET, *, explicitly_authorized=False,
                      max_seconds=MAX_SECONDS,workers=1):
    """明确许可后独立CPU生产，可与已冻结R/RR并行；绝不接线或改动活动run。

    workers仅1/2，最多2*workers个待消费future，唯一父writer按原UID顺序落三文件。
    ready前不对消费者可见；失败.partial保留FAIL_STOP，不自动重建或删除。
    """
    if explicitly_authorized is not True:raise RuntimeError("缓存实际批量构建须本次明确许可")
    if type(max_seconds) is not int or not 1<=max_seconds<=MAX_SECONDS:raise ValueError("构建预算必须1..14400秒")
    if type(workers) is not int or workers not in (1,2):raise ValueError("CPU缓存workers只能1/2")
    index_path=_safe_path(index_path);target=_safe_path(target)
    if index_path!=SOURCE_INDEX.absolute() or target!=DEFAULT_TARGET.absolute():raise ValueError("正式缓存仅固定canonical S入口和D SSD目标")
    started=time.monotonic();deadline=started+max_seconds
    def guard():
        if time.monotonic()>=deadline:raise TimeoutError("缓存全流程4小时截止已到")
        _guard_memory()
    guard()
    from .catalogues import synthetic_bbox_catalogue
    from .runner import production_closed
    production_closed();torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    source=synthetic_bbox_catalogue(index_path,split="train");guard()
    if len(source["ordered_uids"])!=COUNT:raise ValueError("缓存必须完整24149图，禁止按成绩/大小筛图")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.with_name(target.name+".partial").exists():raise FileExistsError("缓存/partial已存在，不覆盖或自动重建")
    requirement=cache_requirements()
    if shutil.disk_usage(target.parent).free<requirement["required_free_bytes"]:raise RuntimeError("SSD完整缓存保守上界加100GiB余量不足")
    global_paths={str(index_path),str(index_path.parent/"RELOCATION.json")}
    release=source["binding"]["source_release"]
    global_paths.update(release[k] for k in ("dataset_manifest","validation_receipt","g5_receipt"))
    dependencies={};paths=set(global_paths)
    for uid,ref in source["catalogue"].items():
        seed=Path(ref["seed_dir"]);camera=seed/ref["camera_id"]
        local={str(seed/"seed_manifest.json"),str(camera/"camera.json"),str(camera/"annotations.jsonl")}
        dependencies[uid]=sorted(global_paths|local);paths.update(local)
    source_files={path:_file_binding(path,guard) for path in sorted(paths)};guard()
    pool=None;pending=deque();remaining=iter(source["ordered_uids"])
    if workers==1:
        original=RequestBBoxDataset(source["catalogue"],view="train",synthetic_reader_options={"max_indexed_cameras":7,"max_indexed_frames":3000})
        def prepared(uid):
            guard()
            return original[{"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,"exposure_index":0,"augmentation_seed":0}]
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init,initargs=(source["catalogue"],))
        for _ in range(2*workers):
            uid=next(remaining,None)
            if uid is not None:pending.append((uid,pool.submit(_prepare_worker,uid)))
        def prepared(uid):
            guard();expected,future=pending.popleft()
            if expected!=uid:raise RuntimeError("有界并行结果顺序与原UID不符")
            result=future.result(timeout=max(.001,deadline-time.monotonic()))
            guard();following=next(remaining,None)
            if following is not None:pending.append((following,pool.submit(_prepare_worker,following)))
            return result
    try:
        result=_build_pack(target,source["ordered_uids"],prepared,source_binding=source["binding"],source_files=source_files,
                           dependencies=dependencies,production=True,max_seconds=max(.001,deadline-time.monotonic()),
                           free_reserve=MIN_FREE,guard=guard)
        result.update(workers=workers,pending_futures_ceiling=2*workers if workers==2 else 0,
                      full_elapsed_seconds=time.monotonic()-started,memory_after=_system_memory())
        return result
    finally:
        if pool is not None:
            for _,future in pending:future.cancel()
            pool.shutdown(wait=True,cancel_futures=True)


def inspect_partial_cache(target, ordered_uids, *, expected_manifest_sha256, catalogue=None,
                          require_production=True,check_source_files=False,verify_source_hashes=False,
                          max_seconds=600,guard=None,window_rows=WINDOW_ROWS):
    """只读重建已完整提交前缀，不打开原RGB、也不删除未知尾部。

    旧FAIL没有持久pixel SHA：这里明确新建现存缓存前缀SHA，继承原冻结writer逐行
    write/read验收证据，不能宣称重新与4K真源像素比对。check_source_files=False可先
    做纯D前缀调查；实际resume强制重新核所有GT/metadata SHA与RGB stat。
    """
    target=_safe_path(target);partial=target.with_name(target.name+".partial");path=partial/"cache.json"
    raw=path.read_bytes()
    if not isinstance(expected_manifest_sha256,str) or len(expected_manifest_sha256)!=64 or _sha(raw)!=expected_manifest_sha256:
        raise ValueError("旧FAIL manifest SHA与调用方绑定不符")
    manifest=json.loads(raw);count=manifest.get("count");completed=manifest.get("completed_rows")
    if manifest.get("schema")!=SCHEMA or manifest.get("state")!="FAIL_STOP":raise ValueError("只恢复明确FAIL_STOP，不猜测活动/未知状态")
    if require_production and (target!=DEFAULT_TARGET.absolute() or manifest.get("kind")!="production" or count!=COUNT):
        raise ValueError("正式恢复目标/完整图数不符")
    if type(completed) is not int or not 0<=completed<=count or len(ordered_uids)!=count or len(set(ordered_uids))!=count:
        raise ValueError("旧已提交行数/完整有序UID池不符")
    current=_implementation();old=manifest["implementation"]
    if old.get("letterbox_binding")!=current["letterbox_binding"] or set(old.get("code_sha256",{}))!=set(current["code_sha256"]):
        raise ValueError("恢复实现集合/依赖绑定变化")
    if any(digest!=current["code_sha256"][name] for name,digest in old["code_sha256"].items() if name!="input_cache.py"):
        raise ValueError("除缓存writer外的预处理/标签代码变化，不能复用前缀")
    if set(p.name for p in partial.iterdir())!={"cache.json","images.npy","samples.jsonl"}:raise ValueError("partial包含未知文件，拒绝自动清理")
    before={name:_stat(partial/name) for name in ("cache.json","images.npy","samples.jsonl")}
    if before["samples.jsonl"][2]>MAX_SAMPLES:raise ValueError("既有JSONL超界")
    started=time.monotonic()
    def check():
        if guard is not None:guard()
        if time.monotonic()-started>=max_seconds:raise TimeoutError("前缀核验有界预算耗尽")
    if check_source_files:
        for binding in manifest["source_files"].values():check();_check_binding(binding,hash_contents=verify_source_hashes)
    common={path for path in manifest["source_files"] if Path(path).name in
            ("dataset.json","RELOCATION.json","dataset_manifest.json","validation_receipt.json","G5_complete.json")}
    array=_WindowedArray(partial/"images.npy",count,mode="r",window_rows=window_rows)
    rows=[];labels_sha=hashlib.sha256();pixels_sha=hashlib.sha256();offset=0
    try:
        with (partial/"samples.jsonl").open("rb") as f:
            while True:
                check();raw_line=f.readline(MAX_ROW+1)
                if not raw_line:break
                index=len(rows)
                if index>=completed or len(raw_line)>MAX_ROW or not raw_line.endswith(b"\n"):
                    raise ValueError("既有JSONL有未知/不完整尾部，不自动截断或猜提交")
                saved=json.loads(raw_line);uid=ordered_uids[index]
                if saved.get("uid")!=uid or saved.get("row")!=index:raise ValueError("前缀row/UID不是原精确有序池")
                metadata=_decode(saved["metadata"])
                if metadata.get("uid")!=uid or metadata.get("domain")!="S" or metadata.get("source_reference",{}).get("split")!="train":
                    raise ValueError("前缀metadata身份/来源不符")
                if not _equal(metadata,_decode(_encode(metadata))):raise ValueError("前缀metadata无损类型不符")
                ref=metadata["source_reference"]
                if catalogue is not None and ref!=catalogue[uid]:raise ValueError("前缀来源引用不是当前原catalogue")
                seed=Path(ref["seed_dir"]);camera=seed/ref["camera_id"]
                dependencies=sorted(common|{str(seed/"seed_manifest.json"),str(camera/"camera.json"),str(camera/"annotations.jsonl")})
                if any(p not in manifest["source_files"] for p in dependencies):raise ValueError("前缀GT/metadata来源绑定不完整")
                if check_source_files:
                    image=metadata["image_source_binding"]
                    if _stat(image["path"])!=list(image["stat"]):raise ValueError("前缀原RGB stat已变化")
                pixels=array[index];pixel_bytes=pixels.tobytes();pixels_sha.update(pixel_bytes);labels_sha.update(raw_line)
                rows.append({"uid":uid,"row":index,"offset":offset,"bytes":len(raw_line),"sha256":_sha(raw_line),
                             "pixel_sha256":_sha(pixel_bytes),"source_files":dependencies})
                offset+=len(raw_line)
        if len(rows)!=completed or offset!=before["samples.jsonl"][2]:raise ValueError("完整JSONL行数与FAIL已提交行数不齐")
        peak=array.max_mapped_bytes
    finally:array.close()
    for name,identity in before.items():
        if _stat(partial/name)!=identity:raise ValueError("只读前缀核验期间文件变化")
    return {"legacy_target":str(target),"partial_path":str(partial),"manifest":manifest,"rows":rows,"images_stat":before["images.npy"],"samples_stat":before["samples.jsonl"],
        "manifest_sha256":expected_manifest_sha256,"completed_rows":completed,"prefix_images_bytes":completed*ROW_BYTES,
        "prefix_images_sha256_new_baseline":pixels_sha.hexdigest(),"prefix_samples_sha256":labels_sha.hexdigest(),
        "old_pixel_sha_durable":False,"original_4K_read":False,"source_files_revalidated":bool(check_source_files),
        "source_file_hashes_revalidated":bool(check_source_files and verify_source_hashes),
        "mapped_view_peak_bytes":peak,"inspection_elapsed_seconds":time.monotonic()-started,
        "pixel_equality_evidence":"inherited_frozen_writer_per_row_write_read_then_complete_JSONL_commit; new_prefix_SHA_is_recovery_baseline"}


def resume_input_cache(*, expected_old_manifest_sha256, old_manifest_evidence_path,old_writer_code_path,
                       index_path=SOURCE_INDEX,target=DEFAULT_TARGET,explicitly_authorized=False,
                       max_seconds=13390,workers=1,absolute_deadline_utc="2026-11-07T23:55:00Z"):
    """明确单次resume；只复用核验前缀并append，不自动重建/截断。

    应由root在GPU队列自然闭合后调用；原已用构建时间从总4h扣除，等待不计构建。
    旧FAIL原bytes保存在指定G任务证据；新发布manifest明确old/new实现与新prefix SHA。
    """
    if explicitly_authorized is not True:raise RuntimeError("缓存恢复必须本次明确授权")
    if type(workers) is not int or workers!=1:raise ValueError("本次恢复固定一个CPU worker，最多2个future")
    if type(max_seconds) is not int or not 1<=max_seconds<=MAX_SECONDS:raise ValueError("恢复预算须有界整数")
    target=_safe_path(target);index_path=_safe_path(index_path)
    if target!=DEFAULT_TARGET.absolute() or index_path!=SOURCE_INDEX.absolute():raise ValueError("恢复只接既定canonical缓存目标/数据源")
    if (RESEARCH/".research_gpu.lock").exists():raise RuntimeError("当前GPU训练队列仍活动，延后缓存源读取而不取消训练")
    partial=target.with_name(target.name+".partial");old_raw=(partial/"cache.json").read_bytes()
    if _sha(old_raw)!=expected_old_manifest_sha256:raise ValueError("旧FAIL manifest绑定不符")
    old=json.loads(old_raw)
    if old.get("recovery"):raise RuntimeError("已存在一次恢复现场，不自动再次续建")
    elapsed=float(old.get("active_elapsed_seconds_total",old["elapsed_seconds"]))
    if not math.isfinite(elapsed) or elapsed<0 or max_seconds>math.floor(MAX_SECONDS-elapsed):raise ValueError("恢复预算超过原4小时扣除已用时间")
    import psutil
    if psutil.pid_exists(old["writer"]["pid"]):raise RuntimeError("旧writer PID仍存在/复用；外层须先核实真实退出")
    code_path=_safe_path(old_writer_code_path)
    old_writer_sha=old["implementation"]["code_sha256"]["input_cache.py"]
    if _sha(code_path.read_bytes())!=old_writer_sha:raise ValueError("原冻结writer代码SHA与失败manifest不符")
    global_limit=datetime(2026,11,7,23,55,tzinfo=timezone.utc)
    absolute=datetime.fromisoformat(absolute_deadline_utc.replace("Z","+00:00"))
    if absolute.tzinfo is None or absolute>global_limit:raise ValueError("恢复绝对截止不得超原全局界限")
    started=time.monotonic();deadline=min(started+max_seconds,time.monotonic()+(absolute-datetime.now(timezone.utc)).total_seconds())
    def guard():
        if time.monotonic()>=deadline:raise TimeoutError("缓存恢复有界活动/绝对预算耗尽")
        _guard_memory()
        if shutil.disk_usage(partial).free<MIN_FREE:raise RuntimeError("缓存恢复D盘100GiB余量不足")
    guard()
    from .catalogues import synthetic_bbox_catalogue
    from .runner import production_closed
    production_closed();torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    source=synthetic_bbox_catalogue(index_path,split="train");guard()
    if source["binding"]!=old["source_binding"]:raise ValueError("恢复前原S完整数据绑定变化")
    prefix=inspect_partial_cache(target,source["ordered_uids"],expected_manifest_sha256=expected_old_manifest_sha256,
        catalogue=source["catalogue"],require_production=True,check_source_files=True,verify_source_hashes=True,
        max_seconds=max(.001,deadline-time.monotonic()),guard=guard,window_rows=WINDOW_ROWS)
    evidence=_safe_path(old_manifest_evidence_path)
    evidence_root=Path("G:/FutsalMOT_Training").absolute()
    if not evidence.is_relative_to(evidence_root):raise ValueError("旧FAIL证据必须放G任务区域，不能塞入缓存三文件")
    evidence.parent.mkdir(parents=True,exist_ok=True)
    if evidence.exists():
        if evidence.read_bytes()!=old_raw:raise FileExistsError("旧FAIL证据目标有冲突，不覆盖")
    else:
        with evidence.open("xb") as f:f.write(old_raw);f.flush();os.fsync(f.fileno())
    recovery={k:deepcopy(v) for k,v in prefix.items() if k not in ("manifest","rows")}
    recovery.update(old_manifest_evidence_path=str(evidence),old_writer_code_path=str(code_path),
        old_implementation=old["implementation"],new_implementation=_implementation(),old_writer=old["writer"],
        max_active_seconds=max_seconds,prior_active_seconds=elapsed,absolute_deadline_utc=absolute_deadline_utc,
        original_JSONL_prefix_retained_bytes=prefix["samples_stat"][2],uncommitted_image_tail_policy="retain_then_write_only_original_remaining_UID_rows")
    receipt=evidence.with_name("prefix_resume_receipt.json")
    with receipt.open("xb") as f:f.write(_json(recovery));f.flush();os.fsync(f.fileno())
    prefix["recovery"]=recovery
    # 核验耗时亦计入原4h实际构建预算；等待GPU释放不在本函数活动时间内。
    prefix["manifest"]["active_elapsed_seconds_total"]=elapsed+time.monotonic()-started
    remaining_uids=source["ordered_uids"][prefix["completed_rows"]:];pending=deque();following=iter(remaining_uids)
    dependencies={}
    common={p for p in old["source_files"] if Path(p).name in ("dataset.json","RELOCATION.json","dataset_manifest.json","validation_receipt.json","G5_complete.json")}
    for uid,ref in source["catalogue"].items():
        seed=Path(ref["seed_dir"]);camera=seed/ref["camera_id"]
        dependencies[uid]=sorted(common|{str(seed/"seed_manifest.json"),str(camera/"camera.json"),str(camera/"annotations.jsonl")})
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),initializer=_worker_init,initargs=(source["catalogue"],))
    for _ in range(2):
        uid=next(following,None)
        if uid is not None:pending.append((uid,pool.submit(_prepare_worker,uid)))
    def prepared(uid):
        guard();expected,future=pending.popleft()
        if expected!=uid:raise RuntimeError("恢复有界future顺序不符")
        sample=future.result(timeout=max(.001,deadline-time.monotonic()));guard();next_uid=next(following,None)
        if next_uid is not None:pending.append((next_uid,pool.submit(_prepare_worker,next_uid)))
        return sample
    try:
        result=_build_pack(target,source["ordered_uids"],prepared,source_binding=source["binding"],source_files=old["source_files"],
            dependencies=dependencies,production=True,max_seconds=max(.001,deadline-time.monotonic()),free_reserve=MIN_FREE,
            guard=guard,resume_prefix=prefix,window_rows=WINDOW_ROWS)
        result.update(workers=1,pending_futures_ceiling=2,resume_receipt_path=str(receipt),
            new_prefix_image_SHA=prefix["prefix_images_sha256_new_baseline"],full_resume_elapsed_seconds=time.monotonic()-started)
        return result
    finally:
        for _,future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


SEED_SCHEMA = "futsalmot_prepared_letterbox_seed_cache_v2"
SEED_TARGET = RESEARCH/"cache/synthetic_bbox_960_v2"
MAX_SOURCE_RATE = 100_000_000
SOURCE_READ_CHUNK = 64*1024


class _ReadBudget:
    def __init__(self,rate,*,should_pause=None,check=None,clock=time.monotonic,sleep=time.sleep):
        if type(rate) is not int or not 1<=rate<=MAX_SOURCE_RATE:raise ValueError("G源读取限速必须1B/s..100000000B/s（100MB/s，十进制）")
        self.rate=rate;self.should_pause=should_pause or (lambda:False);self.check=check or (lambda:None)
        self.clock=clock;self.sleep=sleep;self.bytes_read=0;self.largest_raw_read=0;self.paused_seconds=0.
    def wait_ready(self):
        self.check()
        while self.should_pause():
            before=self.clock();self.sleep(1.);self.paused_seconds+=max(0.,self.clock()-before);self.check()
    def charge(self,n):
        self.bytes_read+=n;self.largest_raw_read=max(self.largest_raw_read,n)
        end=self.clock()+n/self.rate
        while self.clock()<end:
            self.check()
            remaining=end-self.clock()
            if remaining<=0:break
            self.sleep(min(1.,remaining))


class _ThrottledRaw(io.RawIOBase):
    def __init__(self,raw,budget):self.raw=raw;self.budget=budget
    @property
    def name(self):return self.raw.name
    def readable(self):return True
    def seekable(self):return self.raw.seekable()
    def writable(self):return False
    def fileno(self):return self.raw.fileno()
    def tell(self):return self.raw.tell()
    def seek(self,*args):return self.raw.seek(*args)
    def readinto(self,buffer):
        self.budget.wait_ready();view=memoryview(buffer);size=min(len(view),SOURCE_READ_CHUNK)
        data=self.raw.read(size)
        n=len(data);view[:n]=data;self.budget.charge(n);return n
    def read(self,size=-1):
        if size==0:return b""
        parts=[];remaining=size
        while remaining!=0:
            self.budget.wait_ready();n=SOURCE_READ_CHUNK if remaining<0 else min(SOURCE_READ_CHUNK,remaining)
            data=self.raw.read(n);self.budget.charge(len(data))
            if not data:break
            parts.append(data)
            if remaining>0:remaining-=len(data)
        return b"".join(parts)
    def readall(self):return self.read(-1)
    def close(self):
        if not self.closed:self.raw.close()
        super().close()


class SourceReadThrottle:
    """仅本缓存进程：G Dataset内容读取均经<=64KiB unbuffered RawIO分块计费。

    覆盖Path.read_bytes/read_text、read/readline/iterator/readinto。CPU单进程无预取，
    每块前可暂停，暂停/限速每秒检查deadline/STOP。绝不patch其它GPU进程或源码。
    """
    def __init__(self,rate_bytes_per_second=MAX_SOURCE_RATE,*,should_pause=None,check=None,
                 source_root=Path("G:/FutsalMOT_Dataset"),clock=time.monotonic,sleep=time.sleep):
        self.budget=_ReadBudget(rate_bytes_per_second,should_pause=should_pause,check=check,clock=clock,sleep=sleep)
        self.source_root=os.path.normcase(os.path.abspath(os.fspath(source_root)))
        self._original_builtin=None;self._original_io=None
    def _matches(self,file):
        if not isinstance(file,(str,bytes,os.PathLike)):return False
        path=os.path.normcase(os.path.abspath(os.fsdecode(file)))
        return path==self.source_root or path.startswith(self.source_root+os.sep)
    def __enter__(self):
        if self._original_builtin is not None:raise RuntimeError("限速context禁止重复进入")
        self._original_builtin=builtins.open;self._original_io=io.open
        def wrapped(file,mode="r",buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None):
            if not self._matches(file):return self._original_builtin(file,mode,buffering,encoding,errors,newline,closefd,opener)
            if any(c in mode for c in "wax+"):raise ValueError("缓存限速scope仅允许G数据源只读")
            raw=self._original_builtin(file,"rb",0,closefd=closefd,opener=opener)
            limited=_ThrottledRaw(raw,self.budget)
            binary=io.BufferedReader(limited,buffer_size=8192)
            if "b" in mode:return binary
            return io.TextIOWrapper(binary,encoding=encoding,errors=errors,newline=newline)
        self._wrapper=wrapped;builtins.open=wrapped;io.open=wrapped
        return self
    def __exit__(self,*args):
        if builtins.open is self._wrapper:builtins.open=self._original_builtin
        if io.open is self._wrapper:io.open=self._original_io
        self._original_builtin=None;self._original_io=None
    def wait_ready(self):self.budget.wait_ready()
    def measurements(self):
        return {"source_bytes_read":self.budget.bytes_read,"largest_raw_read_bytes":self.budget.largest_raw_read,
                "rate_ceiling_bytes_per_second":self.budget.rate,"paused_seconds":self.budget.paused_seconds,
                "scope":"this_process_G_Dataset_content_reads_only_no_other_process_patch"}


def _seed_plan(ordered_uids,catalogue):
    arrays={};locations={}
    for uid in ordered_uids:
        ref=catalogue[uid];seed=uid.split(":")[-3]
        if not seed.startswith("seed_") or len(seed)!=13 or any(c not in "0123456789abcdef" for c in seed[5:]):
            raise ValueError("UID缺原十六进制seed身份")
        if ref.get("domain")!="S" or ref.get("split")!="train":raise ValueError("per-seed缓存只接原S train")
        info=arrays.setdefault(seed,{"path":"arrays/"+seed+".npy","count":0})
        locations[uid]={"seed_id":seed,"seed_row":info["count"]};info["count"]+=1
    return arrays,locations


def seed_cache_requirements(counts):
    if not isinstance(counts,dict) or not counts or any(type(v) is not int or v<1 for v in counts.values()):raise ValueError("需每seed实际正图数")
    count=sum(counts.values());npy=count*ROW_BYTES+128*len(counts);upper=npy+MAX_SAMPLES+MAX_MANIFEST
    if upper>MAX_BYTES:raise ValueError("per-seed缓存保守上界超过128GiB")
    return {"count":count,"seed_arrays":len(counts),"images_npy_bytes":npy,"complete_upper_bytes":upper,
            "metadata_upper_bytes":MAX_SAMPLES+MAX_MANIFEST,"reserve_bytes":MIN_FREE,"required_free_bytes":upper+MIN_FREE}


class _SeedArrayStore:
    def __init__(self,root,arrays,*,mode="r",create=False):
        self.root=Path(root);self.arrays=arrays;self.mode=mode;self._active=None;self._seed=None;self.max_mapped_bytes=0
        if create:
            (self.root/"arrays").mkdir()
            for info in arrays.values():_create_images(self.root/info["path"],info["count"])
    def _select(self,seed):
        if seed not in self.arrays:raise ValueError("未知seed缓存数组")
        if self._seed!=seed:
            self.close();info=self.arrays[seed]
            self._active=_WindowedArray(self.root/info["path"],info["count"],mode=self.mode,window_rows=WINDOW_ROWS);self._seed=seed
        return self._active
    def write_exact(self,seed,row,pixels):
        array=self._select(seed);array.write_exact(row,pixels);self.max_mapped_bytes=max(self.max_mapped_bytes,array.max_mapped_bytes)
    def read_copy(self,seed,row):
        array=self._select(seed);out=array[row];self.max_mapped_bytes=max(self.max_mapped_bytes,array.max_mapped_bytes);return out
    def close(self):
        if self._active is not None:self.max_mapped_bytes=max(self.max_mapped_bytes,self._active.max_mapped_bytes);self._active.close()
        self._active=None;self._seed=None


def _build_seed_pack(target,ordered_uids,catalogue,prepared_factory,*,source_binding,source_files,dependencies,
                     production,guard=None,progress=None,free_reserve=MIN_FREE,max_seconds=MAX_SECONDS,
                     prefix=None,recovery_evidence=None):
    """v2单writer；已提交v1前缀只从D原缓存复制，原partial原bytes留存。"""
    target=_safe_path(target);partial=target.with_name(target.name+".partial")
    if target.exists() or partial.exists():raise FileExistsError("v2目标/partial已存在，禁止覆盖或自动重建")
    arrays,locations=_seed_plan(ordered_uids,catalogue)
    if production and (len(ordered_uids)!=COUNT or len(arrays)!=16):raise ValueError("正式v2必须完整16 train seed/24149图")
    requirements=seed_cache_requirements({k:v["count"] for k,v in arrays.items()})
    if shutil.disk_usage(target.parent).free<free_reserve+requirements["complete_upper_bytes"]:raise RuntimeError("保留旧v1后的D空间不足新v2完整上界+100GiB")
    first=0 if prefix is None else prefix["completed_rows"]
    if prefix is not None and [row["uid"] for row in prefix["rows"]]!=list(ordered_uids[:first]):raise ValueError("v1前缀不是当前有序UID池")
    manifest={"schema":SEED_SCHEMA,"state":"BUILDING","kind":"production" if production else "cpu_fixture",
        "count":len(ordered_uids),"source_binding":deepcopy(source_binding),"source_files":deepcopy(source_files),
        "implementation":_implementation(),"budget":requirements,"completed_rows":0,
        "writer":{"pid":os.getpid(),"version":"per_seed_windowed_v2","window_rows":WINDOW_ROWS,
                  "active_windows_max":2 if prefix is not None else 1,"legacy_copy_active_windows_max":2 if prefix is not None else 0,
                  "normal_write_active_windows_max":1,"consumer_active_windows_max":1,
                  "combined_mapped_view_ceiling_bytes":(2 if prefix is not None else 1)*WINDOW_ROWS*ROW_BYTES,"CPU_workers":1},
        "format":{"arrays":"arrays/<original_seed_id>.npy","samples":"samples.jsonl","layout":"NCHW_RGB_uint8_960_before_random_augmentation"},
        "recovery":deepcopy(recovery_evidence)}
    partial.mkdir();started=time.monotonic();rows=[];store=None;legacy=None;old_samples=None
    def check():
        if guard is not None:guard()
        if time.monotonic()-started>=max_seconds:raise TimeoutError("v2缓存冻结活动预算耗尽")
        if shutil.disk_usage(partial).free<free_reserve:raise RuntimeError("v2缓存D盘100GiB余量不足")
    try:
        _write_manifest(partial/"cache.json",manifest)
        store=_SeedArrayStore(partial,arrays,mode="r+",create=True)
        if prefix is not None:
            legacy_root=Path(prefix["partial_path"])
            if _stat(legacy_root/"images.npy")!=prefix["images_stat"] or _stat(legacy_root/"samples.jsonl")!=prefix["samples_stat"]:
                raise ValueError("旧v1前缀核验后变化，拒绝复制")
            legacy=_WindowedArray(legacy_root/"images.npy",len(ordered_uids),mode="r",window_rows=WINDOW_ROWS)
            old_samples=(legacy_root/"samples.jsonl").open("rb")
        with (partial/"samples.jsonl").open("xb") as out:
            for index,uid in enumerate(ordered_uids):
                check();location=locations[uid]
                if index<first:
                    prior=prefix["rows"][index];old_samples.seek(prior["offset"]);raw=old_samples.read(prior["bytes"])
                    if _sha(raw)!=prior["sha256"]:raise ValueError("旧v1标注前缀在复制时变化")
                    pixels=legacy[index]
                    if _sha(pixels.tobytes())!=prior["pixel_sha256"]:raise ValueError("旧v1像素前缀与新恢复基线不符")
                    metadata_sha=prior["sha256"];pixel_sha=prior["pixel_sha256"]
                else:
                    if legacy is not None:legacy.close();legacy=None;old_samples.close();old_samples=None
                    sample=prepared_factory(uid);image=sample.get("img")
                    if (not isinstance(image,torch.Tensor) or image.device.type!="cpu" or image.dtype!=torch.uint8
                        or tuple(image.shape)!=(3,960,960) or sample.get("uid")!=uid or sample.get("domain")!="S"
                        or sample.get("random_augmentation_implemented") is not False
                        or sample.get("source_reference",{}).get("split")!="train"
                        or sample.get("geometry",{}).get("letterbox_candidate_only") is not True):raise ValueError("v2只能存原增强前S train LetterBox")
                    binding=sample["image_source_binding"]
                    if _stat(binding["path"])!=list(binding["stat"]):raise ValueError("新样本原RGB身份变化")
                    metadata={k:deepcopy(v) for k,v in sample.items() if k!="img" and k not in _DYNAMIC}
                    encoded=_encode(metadata)
                    if not _equal(metadata,_decode(encoded)):raise RuntimeError("新metadata无损读回不一致")
                    raw=_json({"uid":uid,"row":index,"metadata":encoded});pixels=image.detach().contiguous().numpy()
                    metadata_sha=_sha(raw);pixel_sha=_sha(pixels.tobytes())
                if len(raw)>MAX_ROW or out.tell()+len(raw)>MAX_SAMPLES:raise ValueError("v2 JSONL超过冻结上界")
                store.write_exact(location["seed_id"],location["seed_row"],pixels)
                row={"uid":uid,"row":index,"offset":out.tell(),"bytes":len(raw),"sha256":metadata_sha,
                    "pixel_sha256":pixel_sha,"source_files":list(dependencies[uid]),**location}
                out.write(raw);rows.append(row)
                if (index+1)%WINDOW_ROWS==0:
                    out.flush();store.close();manifest.update(completed_rows=index+1,elapsed_seconds=time.monotonic()-started,
                        source_IO=None if progress is None else progress(),mapped_view_peak_bytes=store.max_mapped_bytes)
                    _write_manifest(partial/"cache.json",manifest)
            out.flush();os.fsync(out.fileno())
        store.close();check()
        if legacy is not None:legacy.close();legacy=None;old_samples.close();old_samples=None
        for binding in source_files.values():_check_binding(binding)
        if manifest["implementation"]!=_implementation():raise RuntimeError("v2构建期间预处理代码变化")
        array_files={seed:{**info,"stat":_stat(partial/info["path"])} for seed,info in arrays.items()}
        image_bytes=sum(info["stat"][2] for info in array_files.values())
        if image_bytes!=requirements["images_npy_bytes"]:raise ValueError("v2 per-seed NPY精确总字节不符")
        manifest.update(state="READY",rows=rows,array_files=array_files,samples_file={"stat":_stat(partial/"samples.jsonl")},
            elapsed_seconds=time.monotonic()-started,source_IO=None if progress is None else progress(),
            mapped_view_peak_bytes=store.max_mapped_bytes,
            acceptance={"legacy_prefix_rows_copied_D_to_D":first,"legacy_original_bytes_retained":prefix is not None,
                        "prefix_copy_exact_to_new_recovery_baseline":True,"new_rows_source_letterbox_write_read_exact":True,
                        "all_metadata_roundtrip_exact":True,"random_augmentation_cached":False})
        _write_manifest(partial/"cache.json",manifest)
        total=image_bytes+(partial/"samples.jsonl").stat().st_size+(partial/"cache.json").stat().st_size
        if total>MAX_BYTES:raise RuntimeError("v2完整缓存实际大小超过128GiB")
        if target.exists():raise FileExistsError("v2发布目标已存在，不覆盖")
        os.rename(partial,target)
        return {"state":"READY","path":str(target),"count":len(rows),"seed_arrays":len(arrays),"reused_prefix_rows":first,
            "images_npy_bytes":image_bytes,"total_bytes":total,"manifest_sha256":_sha((target/"cache.json").read_bytes()),
            "elapsed_seconds":manifest["elapsed_seconds"],"source_IO":manifest["source_IO"],"mapped_view_peak_bytes":store.max_mapped_bytes}
    except BaseException as exc:
        if store is not None:store.close()
        manifest.pop("rows",None);manifest.update(state="FAIL_STOP",completed_rows=len(rows),elapsed_seconds=time.monotonic()-started,
            error={"type":type(exc).__name__,"message":str(exc)},source_IO=None if progress is None else progress())
        _write_manifest(partial/"cache.json",manifest);raise
    finally:
        if store is not None:store.close()
        if legacy is not None:legacy.close()
        if old_samples is not None:old_samples.close()


def build_seed_cache(*,expected_old_manifest_sha256,old_manifest_evidence_path,old_writer_code_path,
                     index_path=SOURCE_INDEX,target=SEED_TARGET,legacy_target=DEFAULT_TARGET,explicitly_authorized=False,
                     max_seconds=None,approved_max_seconds=MAX_SECONDS,deadline_utc="2026-11-07T23:55:00Z",
                     rate_bytes_per_second=MAX_SOURCE_RATE,should_pause=None,stop_check=None):
    """新v2生产/迁移入口；root须冻结已获准活动预算与绝对截止，默认不扩原4h。

    保留原v1.partial，前缀D→D窗口复制，原G源read逐64KiB限速≤100MB/s（100000000B/s，十进制）。
    单CPU进程串行、无预取burst、低优先级；Real验证/慢速策略由should_pause注入。
    """
    if explicitly_authorized is not True:raise RuntimeError("v2实际缓存构建/迁移须本次明确许可")
    target=_safe_path(target);legacy_target=_safe_path(legacy_target);index_path=_safe_path(index_path)
    if target!=SEED_TARGET.absolute() or legacy_target!=DEFAULT_TARGET.absolute() or index_path!=SOURCE_INDEX.absolute():raise ValueError("v2目标/原v1/原canonical来源不符")
    if type(approved_max_seconds) is not int or approved_max_seconds<1:raise ValueError("必须冻结获准总活动时限")
    absolute=datetime.fromisoformat(deadline_utc.replace("Z","+00:00"));global_limit=datetime(2026,11,7,23,55,tzinfo=timezone.utc)
    if absolute.tzinfo is None or absolute>global_limit:raise ValueError("v2绝对截止不得超原全局截止")
    legacy=legacy_target.with_name(legacy_target.name+".partial");old_raw=(legacy/"cache.json").read_bytes();old=json.loads(old_raw)
    if _sha(old_raw)!=expected_old_manifest_sha256 or old.get("state")!="FAIL_STOP":raise ValueError("原v1 FAIL manifest不符")
    prior=float(old.get("active_elapsed_seconds_total",old["elapsed_seconds"]))
    remaining=math.floor(approved_max_seconds-prior)
    if max_seconds is None:max_seconds=remaining
    if type(max_seconds) is not int or not 1<=max_seconds<=remaining:raise ValueError("新缓存预算超过明确获准总时限扣已用时间")
    code_path=_safe_path(old_writer_code_path)
    if _sha(code_path.read_bytes())!=old["implementation"]["code_sha256"]["input_cache.py"]:raise ValueError("原冻结writer源码不符")
    import psutil
    if psutil.pid_exists(old["writer"]["pid"]):raise RuntimeError("原cache writer PID仍存在，须外层核身份退出")
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    started=time.monotonic();deadline=min(started+max_seconds,started+(absolute-datetime.now(timezone.utc)).total_seconds())
    def basic_guard():
        if stop_check is not None:stop_check()
        if time.monotonic()>=deadline:raise TimeoutError("v2冻结活动/绝对截止已到")
        _guard_memory()
        if shutil.disk_usage(target.parent).free<MIN_FREE:raise RuntimeError("v2保留旧缓存后D盘100GiB余量不足")
    basic_guard()
    with SourceReadThrottle(rate_bytes_per_second,should_pause=should_pause,check=basic_guard) as source_io:
        def guard():basic_guard();source_io.wait_ready()
        from .runner import production_closed
        from .catalogues import synthetic_bbox_catalogue
        production_closed();guard();source=synthetic_bbox_catalogue(index_path,split="train")
        if source["binding"]!=old["source_binding"]:raise ValueError("迁移前完整S train绑定变化")
        prefix=inspect_partial_cache(legacy_target,source["ordered_uids"],expected_manifest_sha256=expected_old_manifest_sha256,
            catalogue=source["catalogue"],require_production=True,check_source_files=True,verify_source_hashes=False,
            max_seconds=max(.001,deadline-time.monotonic()),guard=guard)
        for binding in old["source_files"].values():guard();_check_binding(binding,hash_contents=True)
        original=RequestBBoxDataset(source["catalogue"],view="train",synthetic_reader_options={"max_indexed_cameras":7,"max_indexed_frames":3000})
        def prepared(uid):
            guard()
            return original[{"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,"exposure_index":0,"augmentation_seed":0}]
        # 仅三张前缀原图重新核对；所有G内容读取同样经限速与Real优先暂停。
        checks=[];legacy_array=_WindowedArray(legacy/"images.npy",COUNT,mode="r",window_rows=WINDOW_ROWS)
        try:
            first=prefix["completed_rows"]
            for position in sorted({0,first//2,first-1}) if first else ():
                guard();uid=source["ordered_uids"][position];sample=prepared(uid)
                if not np.array_equal(legacy_array[position],sample["img"].numpy()):raise RuntimeError("前缀抽样原RGB重新LetterBox像素不一致")
                checks.append({"uid":uid,"global_row":position,"original_RGB_letterbox_exact":True})
        finally:legacy_array.close()
        evidence=_safe_path(old_manifest_evidence_path)
        if not evidence.is_relative_to(Path("G:/FutsalMOT_Training").absolute()):raise ValueError("旧FAIL evidence须G训练任务区域")
        evidence.parent.mkdir(parents=True,exist_ok=True)
        if evidence.exists():
            if evidence.read_bytes()!=old_raw:raise FileExistsError("旧FAIL evidence冲突，拒覆盖")
        else:
            with evidence.open("xb") as f:f.write(old_raw);f.flush();os.fsync(f.fileno())
        recovery={k:deepcopy(v) for k,v in prefix.items() if k not in ("manifest","rows")}
        recovery.update(old_manifest_evidence_path=str(evidence),old_implementation=old["implementation"],
            new_implementation=_implementation(),original_RGB_sample_checks=checks,original_v1_partial_retained=True,
            prior_active_seconds=prior,legacy_partial_write_stage_seconds=prior,
            legacy_elapsed_scope="legacy_partial_write_stage_only; caller_freezes_tighter_outer_runtime_upper_bound",
            approved_total_active_seconds=approved_max_seconds,max_this_run_seconds=max_seconds,
            caller_conservative_resume_budget_seconds=max_seconds,
            deadline_utc=deadline_utc,rate_ceiling_bytes_per_second=rate_bytes_per_second)
        receipt=evidence.with_name("per_seed_migration_receipt.json")
        with receipt.open("xb") as f:f.write(_json(recovery));f.flush();os.fsync(f.fileno())
        dependencies={}
        common={p for p in old["source_files"] if Path(p).name in ("dataset.json","RELOCATION.json","dataset_manifest.json","validation_receipt.json","G5_complete.json")}
        for uid,ref in source["catalogue"].items():
            seed=Path(ref["seed_dir"]);camera=seed/ref["camera_id"]
            dependencies[uid]=sorted(common|{str(seed/"seed_manifest.json"),str(camera/"camera.json"),str(camera/"annotations.jsonl")})
        result=_build_seed_pack(target,source["ordered_uids"],source["catalogue"],prepared,source_binding=source["binding"],
            source_files=old["source_files"],dependencies=dependencies,production=True,guard=guard,progress=source_io.measurements,
            max_seconds=max(.001,deadline-time.monotonic()),prefix=prefix,recovery_evidence=recovery)
        result.update(total_active_seconds_including_prior=prior+time.monotonic()-started,
            total_active_scope="legacy_partial_write_stage_plus_resume_only; true_outer_upper_bound_in_caller_resume_plan",
            migration_receipt_path=str(receipt))
        return result


class _CacheReader:
    def __init__(self,target,*,require_production=True,expected_manifest_sha256=None):
        self.target=_safe_path(target);self.require_production=require_production;self.expected_manifest_sha256=expected_manifest_sha256;self._images=None;self._manifest=None;self._index=None;self._manifest_stat=None
        self._open()
    def _open(self):
        if self.target.name.endswith(".partial"):raise ValueError("拒绝尚未发布的.partial缓存")
        path=self.target/"cache.json"
        if _source_stat(path)[2]>MAX_MANIFEST:raise ValueError("缓存manifest超界")
        raw=path.read_bytes();m=json.loads(raw)
        expected=self.expected_manifest_sha256
        if self.require_production and (not isinstance(expected,str) or len(expected)!=64):raise ValueError("正式消费者必须绑定精确manifest SHA256")
        if expected is not None and _sha(raw)!=expected:raise ValueError("缓存manifest SHA256与冻结run绑定不符")
        schema=m.get("schema")
        if schema not in (SCHEMA,SEED_SCHEMA) or m.get("state")!="READY":raise ValueError("缓存必须完整READY")
        expected_target=SEED_TARGET.absolute() if schema==SEED_SCHEMA else DEFAULT_TARGET.absolute()
        if self.require_production and (m.get("kind")!="production" or m.get("count")!=COUNT or self.target!=expected_target):
            raise ValueError("正式消费者拒绝fixture/非全量/非固定SSD目标")
        if m["implementation"]!=_implementation():raise ValueError("缓存预处理实现/依赖绑定变化")
        expected_entries={"cache.json","arrays","samples.jsonl"} if schema==SEED_SCHEMA else {"cache.json","images.npy","samples.jsonl"}
        if set(p.name for p in self.target.iterdir())!=expected_entries:raise ValueError("缓存根目录不符合既定文件结构")
        count=m.get("count");rows=m.get("rows")
        if type(count) is not int or count<1 or not isinstance(rows,list) or len(rows)!=count:raise ValueError("缓存行数不齐")
        index={};cursor=0
        for i,row in enumerate(rows):
            if row["uid"] in index or row["row"]!=i or row["offset"]!=cursor or not 0<row["bytes"]<=MAX_ROW:raise ValueError("缓存UID/行/offset不连续")
            cursor+=row["bytes"];index[row["uid"]]=row
        if cursor!=m["samples_file"]["stat"][2] or cursor>MAX_SAMPLES:raise ValueError("JSONL索引末尾/大小不齐")
        if _stat(self.target/"samples.jsonl")!=m["samples_file"]["stat"]:raise ValueError("缓存metadata文件身份变化")
        if schema==SEED_SCHEMA:
            arrays=m["array_files"]
            if self.require_production and len(arrays)!=16:raise ValueError("v2正式缓存必须16个原seed数组")
            if sum(item["count"] for item in arrays.values())!=count:raise ValueError("v2 per-seed图数汇总不齐")
            if {p.name for p in (self.target/"arrays").iterdir()}!={Path(item["path"]).name for item in arrays.values()}:raise ValueError("v2 arrays目录有缺失或额外文件")
            for seed,item in arrays.items():
                if item["path"]!="arrays/"+seed+".npy" or _stat(self.target/item["path"])!=item["stat"]:raise ValueError("v2数组路径/身份变化")
                _image_header(self.target/item["path"],item["count"])
            seen_locations=set()
            for row in rows:
                if row.get("seed_id") not in arrays or type(row.get("seed_row")) is not int or not 0<=row["seed_row"]<arrays[row["seed_id"]]["count"]:raise ValueError("v2每行seed/局部行非法")
                location=(row["seed_id"],row["seed_row"])
                if location in seen_locations:raise ValueError("v2每seed局部行重复")
                seen_locations.add(location)
            images=_SeedArrayStore(self.target,arrays,mode="r")
        else:
            if _stat(self.target/"images.npy")!=m["images_file"]["stat"]:raise ValueError("缓存文件身份/大小/mtime变化")
            images=_WindowedArray(self.target/"images.npy",count,mode="r",window_rows=WINDOW_ROWS)
        for row in m["source_files"].values():_check_binding(row,hash_contents=True)
        self._images=images;self._manifest=m;self._index=index;self._manifest_stat=_stat(path);self._manifest_sha=_sha(raw)
    def read_prepared(self,request,reference):
        if self._images is None:self._open()
        m=self._manifest
        if _stat(self.target/"cache.json")!=self._manifest_stat:raise ValueError("cache manifest运行期变化")
        uid=request.get("uid");row=self._index.get(uid)
        if row is None or request.get("domain")!="S" or request.get("stream","S")!="S" or request.get("data_domain","S")!="S":raise ValueError("缓存请求不是已冻结S UID/来源")
        if reference.get("domain")!="S" or reference.get("split")!="train":raise ValueError("缓存只接原S train引用")
        for path in row["source_files"]:_check_binding(m["source_files"][path])
        if _stat(self.target/"samples.jsonl")!=m["samples_file"]["stat"]:raise ValueError("缓存文件运行期变化")
        if m["schema"]==SEED_SCHEMA:
            info=m["array_files"][row["seed_id"]]
            if _stat(self.target/info["path"])!=info["stat"]:raise ValueError("per-seed缓存数组运行期变化")
        elif _stat(self.target/"images.npy")!=m["images_file"]["stat"]:raise ValueError("缓存文件运行期变化")
        with (self.target/"samples.jsonl").open("rb") as f:f.seek(row["offset"]);raw=f.read(row["bytes"])
        if _sha(raw)!=row["sha256"]:raise ValueError("缓存标注行SHA不符")
        saved=json.loads(raw)
        if saved["uid"]!=uid or saved["row"]!=row["row"]:raise ValueError("缓存标注行身份不符")
        metadata=_decode(saved["metadata"])
        if metadata["source_reference"]!=reference:raise ValueError("缓存与当前catalogue来源引用不符")
        binding=metadata["image_source_binding"]
        if _stat(binding["path"])!=list(binding["stat"]):raise ValueError("原RGB身份/size/mtime变化，拒用缓存")
        image=(self._images.read_copy(row["seed_id"],row["seed_row"]) if m["schema"]==SEED_SCHEMA
               else np.array(self._images[row["row"]],dtype=np.uint8,copy=True,order="C"))
        if _sha(image.tobytes())!=row["pixel_sha256"]:raise ValueError("缓存RGB像素SHA不符")
        metadata.update(img=torch.from_numpy(image),request=deepcopy(request),sampling_stream="S",data_domain="S")
        return metadata
    def close(self):
        _close_array(self._images);self._images=None
    def __getstate__(self):
        state=dict(self.__dict__);state.update(_images=None,_manifest=None,_index=None);return state
    def __del__(self):
        try:self.close()
        except Exception:pass


class CachedTrainingBBoxDataset(RequestBBoxDataset):
    """S读READY缓存，R维持原reader；始终调用原随机增强，不缓存增强输出。"""
    def __init__(self,catalogue,*,cache_root=SEED_TARGET,expected_manifest_sha256=None,view="train",**readeroptions):
        super().__init__(catalogue,view=view,**readeroptions)
        if view!="train":raise ValueError("缓存Dataset只用于train，不替代val/test管线")
        self.cache=_CacheReader(cache_root,require_production=True,expected_manifest_sha256=expected_manifest_sha256)
    def __getitem__(self,request):
        req,ref,stream,domain=self._request(request)
        original=self.cache.read_prepared(req,ref) if domain=="S" else super().__getitem__(req)
        return prepare_training_sample(original,view=self.view)


class CachedRequestPlanDataset(RequestPlanDataset):
    """保持make_loader的显式请求表/isinstance契约；仅供未来新冻结run接线。"""
    def __init__(self,catalogue,requests,*,cache_root=SEED_TARGET,expected_manifest_sha256=None,view="train",**readeroptions):
        if not isinstance(requests,(list,tuple)) or not requests:raise ValueError("需要明确非空请求表")
        self.dataset=CachedTrainingBBoxDataset(catalogue,cache_root=cache_root,expected_manifest_sha256=expected_manifest_sha256,view=view,**readeroptions)
        self.requests=deepcopy(list(requests))
        for request in self.requests:self.dataset._request(request)


def validate_ready_cache(input_cache,synthetic_binding):
    """未来新run冻结前只读校验完整READY/source绑定；不扫描缓存图像数组。"""
    if not isinstance(input_cache,dict) or set(input_cache)!={"path","manifest_sha256"}:raise ValueError("input_cache需精确path/manifest_sha256")
    reader=_CacheReader(input_cache["path"],require_production=True,expected_manifest_sha256=input_cache["manifest_sha256"])
    try:
        if reader._manifest["source_binding"]!=synthetic_binding:raise ValueError("缓存与当前完整S train来源绑定不符")
        return {"state":"READY","count":reader._manifest["count"],"path":str(reader.target),
                "manifest_sha256":reader._manifest_sha,"source_binding":deepcopy(synthetic_binding),
                "images_npy_bytes":(sum(item["stat"][2] for item in reader._manifest["array_files"].values())
                                    if reader._manifest["schema"]==SEED_SCHEMA else reader._manifest["images_file"]["stat"][2]),
                "samples_jsonl_bytes":reader._manifest["samples_file"]["stat"][2],
                "implementation":deepcopy(reader._manifest["implementation"])}
    finally:reader.close()
