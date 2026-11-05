"""正式发布→canonical synthetic的同卷硬链接整理，仅显式begin/pilot/finish执行。

不搬/删/复制媒体，不改原发布/收据。必须先真实planner完整PLANNED_DRY_RUN，
caller审阅精确plan并提供当前无owned写入者的明确观察及现有证据绑定。库只核这些
文件/观察结构，不声称自行认证系统进程；caller负责外层独占。失败留部分目标和
FAIL_STOP，不自动接续或删除未知来源的部分产物。BUILDING永不作为可用数据入口。
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

from .layout import plan_synthetic_layout,_safe_path,_canonical_sha,_read_metadata,_stat_files,_expected_seed_files,SCHEMAS

MAX_METADATA_BYTES=512*1024*1024
MIN_FREE_BYTES=100*1024**3


def _require(value,message):
    if not value:raise ValueError(message)


def _now():return datetime.now(timezone.utc).isoformat()


def _file_stat(path):
    path=_safe_path(path);s=path.stat()
    _require(stat.S_ISREG(s.st_mode),"只允许普通源/目标文件")
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def _sha(path):
    before=_file_stat(path)
    with Path(path).open("rb") as f:
        _require((lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns))(os.fstat(f.fileno()))==before,"打开时文件变化")
        digest=hashlib.file_digest(f,"sha256").hexdigest()
        _require((lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns))(os.fstat(f.fileno()))==before,"读取时文件变化")
    _require(_file_stat(path)==before,"读取前后文件身份/大小/mtime变化")
    return digest,before


def _writer_evidence(value,root):
    _require(isinstance(value,dict) and value.get("assessment")=="NO_ACTIVE_OWNED_WRITERS", "缺明确无owned写入者观察，不能用True替代")
    _require(_safe_path(value.get("source_root",""))==root and value.get("active_writers")==[],"写入者观察未绑定源根或仍有活动写入者")
    observed=value.get("observed_at_utc")
    _require(isinstance(observed,str),"缺写入者观察时间")
    stamp=datetime.fromisoformat(observed.replace("Z","+00:00"))
    _require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0,"观察时间须为明确UTC")
    evidence=value.get("evidence")
    _require(isinstance(evidence,list) and bool(evidence),"无写入者必须有现有闭合证据path/SHA")
    checked=[]
    for item in evidence:
        _require(isinstance(item,dict) and set(item)=={"path","sha256"},"证据须明确path/sha256")
        path=_safe_path(item["path"])
        _require(_file_stat(path)[2]<=4*1024*1024,"闭合证据不是小文件")
        digest,identity=_sha(path)
        _require(digest==item["sha256"],"闭合证据SHA变化")
        checked.append({"path":str(path),"sha256":digest,"stat":list(identity)})
    return {"caller_observation":deepcopy(value),"checked_evidence":checked,
            "verification_scope":"caller_observed_process_quiescence_and_bound_files_not_library_process_certification"}


def begin_synthetic_layout(plan,*,writer_evidence,expected_plan_sha256=None):
    """唯一新整理入口；不接续任何既有target。返回只在本次进程内持有的session。"""
    return SyntheticLayoutApply(plan,writer_evidence=writer_evidence,expected_plan_sha256=expected_plan_sha256)


class SyntheticLayoutApply:
    def __init__(self,plan,*,writer_evidence,expected_plan_sha256=None):
        _require(isinstance(plan,dict) and plan.get("status")=="PLANNED_DRY_RUN" and plan.get("media_enumerated") is True,"必须先审阅完整PLANNED_DRY_RUN，不接metadata-only/布尔通过")
        self._plan=deepcopy(plan);self._plan_sha=_canonical_sha(self._plan)
        if expected_plan_sha256 is not None:_require(self._plan_sha==expected_plan_sha256,"已审阅精确plan SHA不符")
        self.root=_safe_path(plan["source_root"]);self.target=_safe_path(plan["target_root"])
        _require(self.target==self.root.parent.parent/"synthetic","只能整理到原数据根canonical synthetic")
        _require(not self.target.exists(),"目标已存在，拒覆盖或推测接续任何partial")
        fresh=plan_synthetic_layout(self.root,plan["source_binding"]["g5_receipt"],target_root=self.target,
                                    writers_closed=True,enumerate_files=True)
        _require(fresh.get("status")=="PLANNED_DRY_RUN" and _canonical_sha(fresh)==self._plan_sha,"实际完整planner重新核对与已审阅plan不符")
        _require(self.root.stat().st_dev==self.target.parent.stat().st_dev,"源目标非同卷，不允许复制回退")
        self._writer_evidence=deepcopy(writer_evidence)
        observations=[_writer_evidence(writer_evidence,self.root)]
        self._source_metadata={};binding=plan["source_binding"]
        for key,sha_key in (("dataset_manifest","manifest_sha256"),("validation_receipt","validation_sha256"),("g5_receipt","g5_sha256")):
            path=_safe_path(binding[key]);digest,identity=_sha(path)
            _require(identity[2]<=4*1024*1024 and digest==binding[sha_key],"源根小metadata绑定变化")
            self._source_metadata[str(path)]={"sha256":digest,"stat":identity}
        manifest,_=_read_metadata(self.root/"dataset_manifest.json")
        for name in SCHEMAS:
            path=self.root/"schemas"/name;digest,identity=_sha(path)
            _require(digest==manifest["schema_sha256"][name],"源schema绑定变化")
            self._source_metadata[str(path)]={"sha256":digest,"stat":identity}
        self._manifest=manifest
        self._rows=deepcopy(plan["candidate_dataset"]["seeds"])
        _require(len(self._rows)==plan["seed_count"] and len({r["directory"] for r in self._rows})==len(self._rows),"seed集合不闭合")
        self._entries={};groups={r["directory"]:[] for r in self._rows}
        logical=0
        for item in plan["files"]:
            relative=item["relative"];rel=Path(relative)
            _require(not rel.is_absolute() and ".." not in rel.parts and len(rel.parts)>=3,"计划含非法相对路径")
            group="/".join(rel.parts[:2])
            _require(group in groups and relative not in self._entries and item["operation"]=="hardlink","文件不属于唯一seed/hardlink计划")
            source=_safe_path(item["source"]);target=_safe_path(item["target"])
            _require(source==self.root/rel and target==self.target/rel,"旧新路径与精确relative不一致")
            identity=_file_stat(source)
            _require(identity[0]==self.root.stat().st_dev and identity[2:]==(item["bytes"],item["mtime_ns"]),"源文件stat或卷与已审阅plan变化")
            entry={"source_relative":relative,"target_relative":relative,"file_id":{"dev":identity[0],"ino":identity[1]},
                   "bytes":identity[2],"mtime_ns":identity[3]}
            self._entries[relative]=entry;groups[group].append(relative);logical+=identity[2]
        _require(len(self._entries)==plan["relocation"]["linked_file_count"] and logical==plan["relocation"]["logical_bytes"],"全表数量/逻辑字节与plan不齐")
        for row in self._rows:
            names=groups[row["directory"]]
            _require({x[len(row["directory"])+1:] for x in names}==_expected_seed_files(row),"seed文件结构与原计划不齐")
            _require(len(names)==row["counts"]["files"],"seed文件数不齐")
        self._groups=groups;self._created=[];self._verified=set();self._completed=[];self._owned={};self._failed=False;self._finished=False
        self._session=uuid.uuid4().hex
        free=self._space()
        self._index=deepcopy(plan["candidate_dataset"])
        self._index.update(status="BUILDING",relocation="RELOCATION.json",readiness="UNUSABLE_UNTIL_READY",
                           source_schemas={name:{"path":str(self.root/"schemas"/name),"sha256":manifest["schema_sha256"][name]} for name in SCHEMAS})
        self._receipt={"schema_version":"synthetic_relocation_v1","status":"BUILDING","phase":"BEGIN",
                       "session_id":self._session,"created_at_utc":_now(),"plan_sha256":self._plan_sha,
                       "source_root":str(self.root),"target_root":str(self.target),"source_binding":deepcopy(binding),
                       "writer_observations":observations,"source_metadata":deepcopy(self._source_metadata),
                       "files":list(self._entries.values()),"expected_linked_files":len(self._entries),
                       "logical_linked_media_and_annotation_bytes":logical,"media_copy_bytes":0,
                       "original_release_retained":True,"original_receipts_rewritten":False,
                       "max_new_metadata_peak_bytes":MAX_METADATA_BYTES,"minimum_free_bytes":MIN_FREE_BYTES,
                       "volume_free_before_bytes":free,"physical_directory_link_overhead_bytes":None,
                       "physical_overhead_note":"目录项/元数据实际开销未精确归因；逻辑547GB等不当作复制量",
                       "acceptance_level":"file_identity_size_mtime_exact_structure_plus_single_seed_readability_not_generation_revalidation"}
        self._metadata_budget()
        self.target.mkdir() # 原子拒已存在；只有实际begin才创建。
        try:
            self._write("dataset.json",self._index)
            self._persist()
        except BaseException as exc:
            self._fail(exc);raise

    def _space(self):
        free=shutil.disk_usage(self.target.parent).free
        _require(free>=MIN_FREE_BYTES,"卷剩余不足100GiB，停止而非复制/清理源")
        return free

    def _metadata_budget(self):
        a=len(json.dumps(self._receipt,ensure_ascii=False,allow_nan=False).encode("utf8"))+1
        b=len(json.dumps(self._index,ensure_ascii=False,allow_nan=False).encode("utf8"))+1
        _require(2*(a+b)<=MAX_METADATA_BYTES,"完整精确表及原子临时JSON峰值超过512MiB")

    def _write(self,name,value):
        path=_safe_path(self.target/name)
        raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",", ":"))+"\n").encode("utf8")
        _require(len(raw)<=MAX_METADATA_BYTES,"单份metadata超预算")
        if name in self._owned:
            _require(_file_stat(path)==self._owned[name][0] and _sha(path)[0]==self._owned[name][1],"本次metadata被外部改动，拒覆盖未知目标")
            temp=self.target/(".layout_apply_"+self._session+"_"+uuid.uuid4().hex+".tmp")
            with temp.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
            _require(_file_stat(path)==self._owned[name][0],"metadata原子发布前身份变化")
            os.replace(temp,path) # 仅原子替换本次拥有的JSON，绝不move/rename媒体。
        else:
            _require(not path.exists(),"新metadata拒覆盖未知目标")
            with path.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        digest,identity=_sha(path);self._owned[name]=(identity,digest)

    def _persist(self):
        self._receipt.update(created_links=len(self._created),verified_links=len(self._verified),
                             completed_seed_directories=list(self._completed),updated_at_utc=_now())
        self._metadata_budget();self._write("RELOCATION.json",self._receipt)

    def _live(self,writer_evidence=None):
        _require(not self._failed and not self._finished,"本次已FAIL_STOP/完成，不接续/重放")
        if writer_evidence is not None:self._writer_evidence=deepcopy(writer_evidence)
        proof=_writer_evidence(self._writer_evidence,self.root)
        self._receipt["writer_observations"].append(proof)
        for path,binding in self._source_metadata.items():
            _require(_file_stat(path)==tuple(binding["stat"]) and _sha(path)[0]==binding["sha256"],"源小metadata身份或SHA变化")
        for name,binding in self._owned.items():
            _require(_file_stat(self.target/name)==binding[0] and _sha(self.target/name)[0]==binding[1],"本次目标metadata身份/SHA变化")
        self._space()

    def _identity(self,relative):
        item=self._entries[relative]
        return (item["file_id"]["dev"],item["file_id"]["ino"],item["bytes"],item["mtime_ns"])

    def _same(self,relative):
        source=self.root/relative;target=self.target/relative;identity=self._identity(relative)
        _require(_file_stat(source)==identity and _file_stat(target)==identity and os.path.samefile(source,target),"源/目标fileID/size/mtime不是同一文件: "+relative)

    def _link_seed(self,row):
        self._space()
        for relative in self._groups[row["directory"]]:
            source=_safe_path(self.root/relative);target=_safe_path(self.target/relative)
            _require(_file_stat(source)==self._identity(relative),"link前源文件身份/size/mtime变化")
            _require(not target.exists(),"拒覆盖任何已有目标文件")
            target.parent.mkdir(parents=True,exist_ok=True)
            _safe_path(target.parent)
            os.link(source,target,follow_symlinks=False)
            self._created.append(relative)
            self._same(relative);self._verified.add(relative)
        self._verify_seed(row);self._completed.append(row["directory"])

    def _verify_seed(self,row):
        root=self.target/row["directory"];records,directories=_stat_files(root)
        expected=_expected_seed_files(row)
        expected_dirs={p.as_posix() for name in expected for p in Path(name).parents if p!=Path(".")}
        _require({r[0] for r in records}==expected and directories==expected_dirs,"target seed文件/机位目录结构不符")
        _require(len(records)==row["counts"]["files"] and sum(r[1] for r in records)==row["counts"]["bytes"],"target seed数量/字节不符")
        _require(_canonical_sha(records)==row["file_stat_fingerprint"],"target seed stat fingerprint与原发布不符")
        for relative in self._groups[row["directory"]]:self._same(relative)

    def _first_jsonl(self,path,maximum=2*1024*1024):
        before=_file_stat(path)
        with path.open("rb") as f:raw=f.readline(maximum+1)
        _require(raw.strip() and len(raw)<=maximum and _file_stat(path)==before,"首行JSONL为空/过大/读取期间变化")
        value=json.loads(raw);_require(isinstance(value,dict),"JSONL首行不是对象")
        return value

    def pilot(self,*,writer_evidence=None):
        try:
            self._live(writer_evidence)
            _require(not self._completed,"pilot只可执行一次，不能跳过/重复")
            row=next(r for r in self._rows if r["split"]=="train")
            self._link_seed(row);base=self.target/row["directory"]
            hashes={};read_bytes=0
            for name in ("seed_manifest.json","trajectory.jsonl","person_keypoints_coco.json"):
                relative=row["directory"]+"/"+name;self._same(relative)
                read_bytes+=self._entries[relative]["bytes"]
                _require(read_bytes<=MAX_METADATA_BYTES,"pilot三根标注metadata读取超512MiB")
                digest,_=_sha(self.root/relative);self._same(relative);hashes[name]=digest
            _require(hashes["seed_manifest.json"]==row["seed_manifest_sha256"],"pilot seed_manifest原SHA不符")
            # coco_export_sha256是生产coco_export_complete.json收据SHA，不是COCO内容SHA。
            seed,_=_read_metadata(base/"seed_manifest.json")
            _require(seed["seed_id"]==row["seed_id"] and seed["split"]==row["split"]
                     and seed["dataset_version"]==row["original_seed_dataset_version"]
                     and seed["trajectory_uid"]==row["trajectory_uid"],"pilot原身份/UID/split/捕获版本变化")
            camera_id="C5" if "C5" in row["camera_ids"] else row["camera_ids"][0]
            camera,_=_read_metadata(base/camera_id/"camera.json")
            _require(camera["camera_id"]==camera_id and [camera["width"],camera["height"]]==self._manifest["resolution"],"pilot camera身份/分辨率不符")
            annotation=self._first_jsonl(base/camera_id/"annotations.jsonl")
            trajectory=self._first_jsonl(base/"trajectory.jsonl")
            _require(annotation["seed_id"]==row["seed_id"] and annotation["camera_id"]==camera_id
                     and annotation["frame_index"]==1 and annotation["pose_schema_id"]=="futsal_coco17_v1"
                     and annotation["dataset_version"]==seed["dataset_version"],"pilot首行标注身份/COCO17/原版本不符")
            image=base/camera_id/"img1"/"000001.png"
            relative=image.relative_to(self.target).as_posix();self._same(relative)
            _require(camera["width"]*camera["height"]<=16*1024*1024 and self._entries[relative]["bytes"]<=64*1024*1024,"单张代表图超有界读取范围")
            from PIL import Image
            with Image.open(image) as decoded:
                _require(decoded.format=="PNG" and decoded.mode=="RGB" and decoded.size==(camera["width"],camera["height"]),"代表RGB PNG头/尺寸不符")
                decoded.load() # 只这一张可读性，不重做全像素/媒体hash或生成验收。
            self._same(relative);self._live()
            pilot={"status":"PASS","seed_directory":row["directory"],"verified_files":len(self._groups[row["directory"]]),
                   "root_file_source_sha256":hashes,"target_equivalence":"same_inode_no_second_hash",
                   "original_coco_export_receipt_sha256":row["coco_export_sha256"],
                   "original_trajectory_uid":seed["trajectory_uid"],"original_seed_dataset_version":seed["dataset_version"],
                   "first_annotation_frame":annotation["frame_index"],"trajectory_first_line_read":bool(trajectory),
                   "camera_id":camera_id,"representative_rgb_relative":relative,"rgb_images_decoded":1,"media_hashes":0,
                   "scope":"structure_identity_source_bindings_and_one_sample_readability_only"}
            self._receipt.update(phase="PILOT_VERIFIED",pilot=pilot)
            self._persist();return deepcopy(pilot)
        except BaseException as exc:self._fail(exc);raise

    def finish(self,*,writer_evidence=None):
        try:
            self._live(writer_evidence)
            _require(self._receipt.get("pilot",{}).get("status")=="PASS" and len(self._completed)==1,"必须本session实际pilot核验完成，不能布尔批准或跳过")
            for row in self._rows:
                if row["directory"] not in self._completed:self._link_seed(row)
            self._live()
            for row in self._rows:self._verify_seed(row)
            records,directories=_stat_files(self.target)
            expected=set(self._entries)|{"dataset.json","RELOCATION.json"}
            expected_dirs={p.as_posix() for name in self._entries for p in Path(name).parents if p!=Path(".")}
            _require({r[0] for r in records}==expected and directories==expected_dirs,"canonical目标存在缺项/未知额外文件或目录")
            _require(len(self._verified)==len(self._entries),"完整link核验未闭合")
            free=self._space()
            self._receipt.update(status="PASS",phase="ALL_LINKS_VERIFIED",volume_free_after_bytes=free,
                                 observed_volume_free_delta_bytes=self._receipt["volume_free_before_bytes"]-free,
                                 volume_delta_attribution="observed_whole_volume_delta_not_exclusive_physical_cost",
                                 destination_logical_bytes=sum(x["bytes"] for x in self._entries.values()))
            self._persist() # 精确全表/完整验收先持久，再原子发布READY。
            self._index.update(status="READY",readiness="READY",relocation_status="PASS")
            self._write("dataset.json",self._index)
            self._finished=True
            return {"status":"READY","linked_files":len(self._verified),"logical_bytes":self._plan["relocation"]["logical_bytes"],
                    "media_copy_bytes":0,"target_root":str(self.target),"relocation":str(self.target/"RELOCATION.json"),
                    "pilot":deepcopy(self._receipt["pilot"]),"original_release_retained":True}
        except BaseException as exc:self._fail(exc);raise

    def _fail(self,exc):
        if getattr(self,"_failed",False) or getattr(self,"_finished",False):return
        self._failed=True
        if not hasattr(self,"_receipt") or not self.target.exists():return
        self._receipt.update(status="FAIL_STOP",phase="STOPPED",error_type=type(exc).__name__,error=str(exc),
                             automatic_resume_allowed=False,source_retained=True)
        try:self._persist()
        except BaseException as record_error:exc.add_note("RELOCATION失败状态写入失败，保留全部partial: "+repr(record_error))
        if "dataset.json" in self._owned:
            try:
                self._index.update(status="FAIL_STOP",readiness="UNUSABLE_UNTIL_READY")
                self._write("dataset.json",self._index)
            except BaseException as record_error:exc.add_note("dataset状态闭合失败，原入口仍不得视可用: "+repr(record_error))
