"""固定本地官方YOLO11 detect起点；不提供正式训练准入。

复用Ultralytics8.3.237 AGPL-3.0。绕过会下载/自动安装的torch_safe_load。
只有固定manifest+SHA授权的不可变内存bytes可交torch.load(weights_only=False)。
这仍是可信官方pickle反序列化，不是不可信pickle沙箱。无网络/fuse/forward/CUDA。
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import torch
from .detectors import NativeDetectionNetwork, _native, _sources, SOURCE_BINDING

MANIFEST_SHA256="a7535fc4ba6a62ffa3dfd71356df4d2b9bcc89c8c17e0822fdc78f4aa85c12d4"
_DETECT={
 "n":(5613764,"0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
 "s":(19313732,"85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
 "m":(40684120,"d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"),
 "l":(51387343,"9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"),
 "x":(114636239,"7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"),
}
_AUTHORITY=object()

def _strict_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError("重复JSON key")
        result[key]=value
    return result

def _no_reparse(path):
    for part in (path,*path.parents):
        s=part.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s,"st_file_attributes",0)&0x400:
            raise ValueError("拒绝symlink/junction/reparse来源")
    return path.lstat()

def _identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def _read_bound(path,expected_bytes,expected_sha):
    before=_no_reparse(path)
    if not stat.S_ISREG(before.st_mode) or before.st_size!=expected_bytes:
        raise ValueError("来源须普通文件且大小与固定绑定一致")
    with path.open("rb") as f:
        if _identity(os.fstat(f.fileno()))!=_identity(before):raise ValueError("打开来源身份变化")
        payload=f.read(expected_bytes+1)
        after_fd=os.fstat(f.fileno())
    after=_no_reparse(path)
    if _identity(before)!=_identity(after_fd) or _identity(before)!=_identity(after):raise ValueError("读取来源变化")
    if len(payload)!=expected_bytes or hashlib.sha256(payload).hexdigest()!=expected_sha:
        raise ValueError("来源SHA/大小不匹配，禁止反序列化")
    return payload,_identity(after)

class _VerifiedStart:
    def __init__(self,authority,model,binding):
        if authority is not _AUTHORITY:raise ValueError("仅本加载器可签发验证结果")
        self._model=model;self._binding=deepcopy(binding)

def _verified_binding(proof,model):
    if type(proof) is not _VerifiedStart or proof._model is not model:
        raise ValueError("需要本加载器签发且绑定同一模型的验证结果")
    return deepcopy(proof._binding)

def load_local_yolo11_detect(weights_directory,*,model_size):
    """CPU载入固定detect；根目录+exact basename，不接受URL/任意pickle。

    发布权重可为FP16/全冻结；显式转FP32并恢复全层可训，唯独原DFL积分固定。
    不重建头。调用者负责独占只读来源；前后stat/SHA不提供跨进程文件锁。
    Python私有验证凭证用于防误用，不抵御可任意修改本进程代码的恶意调用者。
    caller须在首次导入Ultralytics前设置YOLO_OFFLINE=true；本入口也在任何
    路径访问/原生导入前强制核验，不修改环境，避免上游首次导入DNS在线探测。
    """
    if os.environ.get("YOLO_OFFLINE","").lower()!="true":
        raise RuntimeError("要求caller在首次原生导入前显式设置YOLO_OFFLINE=true")
    if model_size not in _DETECT:raise ValueError("只接受YOLO11 detect n/s/m/l/x")
    if not isinstance(weights_directory,(str,Path)) or "://" in str(weights_directory):raise ValueError("只接受本地目录")
    # 纯词法前置拒绝；不先absolute/stat，避免UNC解析或父目录越界。
    supplied=Path(weights_directory)
    if str(weights_directory).startswith((chr(92)*2,"//")) or supplied.drive.startswith(chr(92)*2):
        raise ValueError("拒绝UNC/网络权重目录")
    if ".." in supplied.parts:raise ValueError("拒绝父目录路径穿越")
    root=supplied.absolute()
    if not stat.S_ISDIR(_no_reparse(root).st_mode):raise ValueError("权重根必须本地普通目录")
    _sources();DetectionModel,Detect=_native()
    manifest_path=root/"manifest.json";ms=_no_reparse(manifest_path)
    if ms.st_size>1024*1024:raise ValueError("manifest超限")
    payload,manifest_identity=_read_bound(manifest_path,ms.st_size,MANIFEST_SHA256)
    manifest=json.loads(payload,object_pairs_hook=_strict_object,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError("非有限JSON")))
    if (manifest.get("schema_version")!="public_yolo11_weights_v1" or manifest.get("status")!="PASS"
        or manifest.get("official_repository")!="ultralytics/assets" or manifest.get("release_tag")!="v8.3.0"
        or len(manifest.get("files",[]))!=15):raise ValueError("固定manifest结构/发布绑定错误")
    name=f"yolo11{model_size}.pt";size,digest=_DETECT[model_size]
    rows=[r for r in manifest["files"] if r.get("file")==name]
    url=f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
    expected={"task":"detect","model_size":model_size,"bytes":size,"sha256":digest,"source_url":url,"pretraining":"COCO detection (80 classes)"}
    if len(rows)!=1 or any(rows[0].get(k)!=v for k,v in expected.items()):
        raise ValueError("manifest detect名称/尺寸/SHA/URL绑定不符")
    path=root/name
    blob,file_identity=_read_bound(path,size,digest)
    # 内存快照在SHA通过后不可被外部文件改写；不执行变化中的文件字节。
    checkpoint=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=False)
    if not isinstance(checkpoint,dict):raise ValueError("公开checkpoint必须dict")
    model=checkpoint.get("ema")
    if model is None:model=checkpoint.get("model")
    if type(model) is not DetectionModel:raise ValueError("只接受原生DetectionModel")
    head=model.model[-1]
    if type(head) is not Detect or head.nc!=80 or model.yaml.get("scale")!=model_size:
        raise ValueError("公开模型任务/COCO80/尺寸不一致")
    if not isinstance(model.names,dict) or set(model.names)!=set(range(80)) or model.names[0]!="person":
        raise ValueError("原COCO80类别绑定错误")
    from ultralytics.nn.modules.conv import Conv
    if not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()) or any(isinstance(m,Conv) and not hasattr(m,"bn") for m in model.modules()):
        raise ValueError("拒绝fused/缺原BN模型")
    for value in model.state_dict().values():
        if value.device.type!="cpu" or value.layout!=torch.strided or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError("来源状态必须有限普通CPU Tensor")
    model.float()
    for parameter in model.parameters():parameter.requires_grad_(True)
    head.dfl.conv.weight.requires_grad_(False)
    if not torch.equal(head.dfl.conv.weight.detach().reshape(-1),torch.arange(16,dtype=torch.float32,device="cpu")):
        raise ValueError("原DFL固定积分值不符")
    from ultralytics.utils import DEFAULT_CFG_DICT
    args=checkpoint.get("train_args") or {}
    if not isinstance(args,dict):raise ValueError("train_args必须dict")
    model.args={**DEFAULT_CFG_DICT,**args};model.task="detect";model.pt_path=str(path);model.eval()
    _,after_identity=_read_bound(path,size,digest)
    _,after_manifest=_read_bound(manifest_path,len(payload),MANIFEST_SHA256)
    if file_identity!=after_identity or manifest_identity!=after_manifest:raise ValueError("反序列化期间来源身份变化")
    binding={"kind":"fixed_local_official_yolo11_detect","public_pretraining_verified":True,
        "model_size":model_size,"task":"detect","file":name,"path":str(path),"sha256":digest,"bytes":size,
        "source_url":url,"manifest_path":str(manifest_path),"manifest_sha256":MANIFEST_SHA256,
        "file_identity":file_identity,"manifest_identity":manifest_identity,"code_binding":dict(SOURCE_BINDING),
        "publisher_digest_verified":False,"verification_scope":"fixed_local_acquisition_receipt_not_independent_publisher_digest"}
    return NativeDetectionNetwork(model,initialization_binding=_VerifiedStart(_AUTHORITY,model,binding))
