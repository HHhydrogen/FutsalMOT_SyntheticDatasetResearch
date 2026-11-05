"""BBox只读CPU适配；派生标签不是原GT，也不代表YOLO loss已接入。"""
from __future__ import annotations

from array import array
import configparser
import copy
from collections import OrderedDict
import hashlib
import os
import stat
import csv
import json
import math
from pathlib import Path

from .data import inspect_dataset

# 只实现已核对的a04来源语义，未知协议失败而非猜测。
_POLICIES = {
    "soccernet_tracking": (
        "official_MOT_xywh_native_coordinates; target trackIDs=gameinfo role starts player or goalkeeper incl plural; no GT for inference",
        "gameinfo referee/staff/ball rows are non_target_ignore; unknown role not fabricated negative; original boxes preserved",
    ),
    "sportsmot_football": (
        "official_SportsMOT_on_field_athletes incl goalkeeper; MOT9 native coordinates and ID0 legal; amodal occluded extent inferred by authors; half_torso_outside may unlabel",
        "referee/coaches/spectators outside target; no invented ignore masks/boxes; incomplete annotation of other people is source-local limitation",
    ),
}


def _int(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label}必须为整数")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须为整数") from exc
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label}必须为不小于{minimum}的整数")
    return int(number)


def _relative(root: Path, relative: str) -> Path:
    """拒绝索引中的越界/链接逃逸；只解析路径，不读取图片。"""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("索引路径越出所属数据根")
    return target


def bbox_geometry(box, *, width: int, height: int, format: str) -> dict:
    """保留源坐标；MOT原生XYWH不擅自减1，合成半开XYXY不加1。"""
    width, height = _int(width, "width", 1), _int(height, "height", 1)
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError("BBox必须为四个有限数值")
    x1, y1, a, b = map(float, box)
    if format == "xywh":
        if a <= 0 or b <= 0:
            raise ValueError("原始XYWH宽高必须为正")
        x2, y2 = x1 + a, y1 + b
    elif format == "xyxy_half_open":
        x2, y2 = a, b
        if x2 <= x1 or y2 <= y1:
            raise ValueError("原始半开XYXY必须有正面积")
    else:
        raise ValueError("不支持的BBox坐标格式")
    cx1, cy1 = max(0., min(width, x1)), max(0., min(height, y1))
    cx2, cy2 = max(0., min(width, x2)), max(0., min(height, y2))
    valid = cx2 > cx1 and cy2 > cy1
    clipped = [cx1, cy1, cx2, cy2] if valid else None
    normalized = [(cx1+cx2)/(2*width), (cy1+cy2)/(2*height), (cx2-cx1)/width, (cy2-cy1)/height] if valid else None
    return {"source_bbox": list(box), "source_format": format, "source_xyxy": [x1,y1,x2,y2], "clipped_xyxy": clipped, "normalized_cxcywh": normalized, "bbox_supervision_valid": valid, "clipped": [x1,y1,x2,y2] != [cx1,cy1,cx2,cy2]}


def _record(track_id, box, width, height, format, **extra):
    return {"track_id": track_id, **bbox_geometry(box, width=width, height=height, format=format), **extra}


def _finish(sample, targets, ignores, absent):
    # 每行保持COCO80头person=0；没有随机重建单类头或single_cls行为。
    labels = [[0, *t["normalized_cxcywh"]] for t in targets if t["bbox_supervision_valid"]]
    return {**sample, "targets": targets, "ignore_targets": ignores, "absent_targets": absent,
            "yolo_labels": labels, "class_mapping": {"person": 0}, "model_head_classes": 80,
            "supervision": {"bbox_targets_available": True, "pose_loss_implemented": False,
                            "seg_loss_implemented": False, "ignore_loss_implemented": False,
                            "ignore_evaluator_implemented": False,
                            "requires_ignore_aware_loss": bool(ignores),
                            "native_yolo_training_ready": False},
            "read_scope": "metadata_and_requested_annotation_only_no_images",
            "release_validation_performed": False}


def _roles(path):
    info = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8-sig") as f:
        info.read_file(f)
    roles = {}
    for key, value in info["Sequence"].items():
        if key.startswith("trackletid_"):
            track_id = _int(key.removeprefix("trackletid_"), "gameinfo trackID", 1)
            roles[track_id] = value.split(";", 1)[0].strip().lower()
    if not roles:
        raise ValueError("SoccerNet缺少角色映射，不能把未知人物当背景")
    return roles


def read_real_bbox_sample(index_path, sample_uid: str, *, split: str = "train") -> dict:
    """仅允许train/val的冻结检测视图，先查UID/split再打开单clip GT。

    保留全部已标非目标框。不能把本函数的yolo_labels直接当作已支持ignore的
    原生YOLO输入；loss/评价适配仍缺失。不会读取test GT/图片。
    """
    if split not in ("train", "val"):
        raise ValueError("BBox准备适配只允许train/val，不开放test内容")
    index_path = Path(index_path)
    inspect_dataset(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source = index["source"]["source_id"]
    if source not in _POLICIES:
        raise ValueError("未实现的真实来源语义")
    target_policy, ignore_policy = _POLICIES[source]
    if (index["source"]["target_semantics"], index["source"]["ignore_policy"]) != (target_policy, ignore_policy):
        raise ValueError("来源目标/ignore协议变化，必须重新审阅适配")
    matches = []
    for sequence in index["sequences"]:
        if sequence["split"] != split:
            continue
        for frame in sequence["detection_frames"]:
            frame_uid = sequence["frame_uid_pattern"].format(frame_number=frame)
            uid = sequence["detection_sample_uid_pattern"].format(frame_uid=frame_uid)
            if uid == sample_uid:
                matches.append((sequence, frame, frame_uid))
    if len(matches) != 1:
        raise ValueError("UID不属于所选split的唯一冻结检测样本")
    seq, frame, frame_uid = matches[0]
    if (seq["source_id"], seq["target_policy"], seq["ignore_policy"]) != (source, target_policy, ignore_policy):
        raise ValueError("clip与来源语义不一致")
    root = _relative(index_path.parent, seq["sequence_path"])
    roles = _roles(_relative(root, seq["files"]["gameinfo"])) if source == "soccernet_tracking" else {}
    width, height = seq["width"], seq["height"]
    targets, ignores, seen = [], [], set()
    # 源GT可按track排序，因此不能在frame变大时提前退出；只保留请求帧。
    with _relative(root, seq["files"]["gt"]).open(encoding="utf-8-sig", newline="") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if len(row) != (10 if source == "soccernet_tracking" else 9):
                raise ValueError(f"GT第{line}行列数与来源不符")
            row_frame = _int(row[0], "frame", 1)
            if row_frame != frame:
                continue
            track_id = _int(row[1], "track ID", 1 if roles else 0)
            if track_id in seen:
                raise ValueError("同帧重复track ID")
            seen.add(track_id)
            box = [float(v) for v in row[2:6]]
            flag = _int(row[6], "GT flag")
            if flag not in (0, 1):
                raise ValueError("未知GT flag，不能猜测监督有效性")
            if source == "sportsmot_football" and _int(row[7], "SportsMOT class") != 1:
                raise ValueError("未核对的SportsMOT目标类别")
            role = roles.get(track_id) if roles else "official_on_field_athlete"
            target = (role is not None and role.startswith(("player", "goalkeeper"))) if roles else True
            item = _record(track_id, box, width, height, "xywh", role=role,
                           original_gt_row=list(row), source_line=line, gt_flag=flag)
            if target and flag:
                targets.append({**item, "class_id": 0})
            else:
                reason = "invalid_gt_flag" if not flag else ("unknown_role" if role is None else "non_target_role")
                ignores.append({**item, "ignore_reason": reason})
    return _finish({"sample_uid": sample_uid, "frame_uid": frame_uid, "clip_uid": seq["clip_uid"],
                    "match_uid": seq["match_uid"], "track_namespace": seq["track_namespace"],
                    "split": split, "domain": "R", "source_id": source, "frame_index": frame,
                    "width": width, "height": height,
                    "image_path": str(_relative(root, seq["image_file_pattern"].format(frame_number=frame))),
                    "target_policy": target_policy, "ignore_policy": ignore_policy,
                    "source_limitations": list(index["limitations"]) + (["unannotated_non_target_people_have_no_ignore_geometry"] if source == "sportsmot_football" else [])}, targets, ignores, [])


def read_synthetic_bbox_sample(seed_directory, camera_id: str, frame_index: int, *, split: str = "train") -> dict:
    """读原seed manifest及单帧JSONL；不把候选样本读取当全版发布验收。

    原生成数据没有frame/sample UID字段，derived_sample_uid明确标为本适配派生，
    含原dataset/version/seed/camera/frame，不冒称历史冻结UID。
    """
    root = Path(seed_directory)
    m = json.loads((root / "seed_manifest.json").read_text(encoding="utf-8"))
    if m["schema"] != "futsalmot_seed_manifest" or m["schema_version"] != "1.0.0":
        raise ValueError("未实现的合成seed schema")
    if split not in ("train", "dev") or m["split"] != split:
        raise ValueError("合成样本必须继承原seed split")
    frame_index = _int(frame_index, "frame_index", 1)
    if camera_id not in m["camera_ids"] or frame_index > m["frame_count"]:
        raise ValueError("机位或帧不在原seed清单")
    camera_root = _relative(root, camera_id)
    camera = json.loads((camera_root / "camera.json").read_text(encoding="utf-8"))
    if camera["schema"] != "futsalmot_camera_metadata" or camera["schema_version"] != "1.0.0" or camera["camera_id"] != camera_id:
        raise ValueError("相机metadata不匹配")
    row = None
    with (camera_root / "annotations.jsonl").open(encoding="utf-8") as f:
        for expected, line in enumerate(f, 1):
            value = json.loads(line)
            if value["frame_index"] != expected:
                raise ValueError("合成标注帧序列不连续")
            if expected == frame_index:
                row = value
                break
    if row is None:
        raise ValueError("请求帧尚无标注，不补齐或伪造")
    return _synthetic_bbox_row(m, camera, row, camera_root=camera_root,
                               camera_id=camera_id, frame_index=frame_index, split=split)


def _synthetic_bbox_row(m, camera, row, *, camera_root, camera_id, frame_index, split):
    """现有单帧入口和索引reader共享语义；只解析当前行，不缓存Pose。"""
    for key in ("dataset_id", "dataset_version", "seed_id"):
        if row[key] != m[key]:
            raise ValueError("帧与原seed身份/版本不一致")
    if (row["schema"], row["schema_version"], row["pose_schema_id"], row["camera_id"]) != ("futsalmot_frame_annotation", "1.0.0", "futsal_coco17_v1", camera_id):
        raise ValueError("未实现的合成帧schema或机位")
    width, height = camera["width"], camera["height"]
    if (row["camera_state"]["width"], row["camera_state"]["height"]) != (width, height):
        raise ValueError("逐帧与相机尺寸不一致")
    entities = {e["entity_id"]: e for e in m["entities"]}
    objects = row["objects"]
    if len(entities) != 10 or len(objects) != 10 or {o["entity_id"] for o in objects} != set(entities):
        raise ValueError("合成帧必须保留原十人身份")
    targets, absent = [], []
    for obj in objects:
        entity = entities[obj["entity_id"]]
        for key in ("track_id", "team", "role", "jersey_number", "instance_mask_id", "number_mask_id"):
            if obj[key] != entity[key]:
                raise ValueError("逐帧人物身份与seed映射不一致")
        if obj["class"] != "player" or obj["role"] not in ("player", "goalkeeper"):
            raise ValueError("未实现的合成目标类")
        pixels = _int(obj["visible_pixel_count"], "visible_pixel_count")
        identity = {"track_id": obj["track_id"], "entity_id": obj["entity_id"], "role": obj["role"],
                    "instance_mask_id": obj["instance_mask_id"], "visible_pixel_count": pixels}
        if obj["bbox_xyxy"] is None:
            if pixels or obj["bbox_source"] != "not_visible" or obj["visibility"] == "visible":
                raise ValueError("空框与可见像素语义不一致")
            absent.append({**identity, "bbox_supervision_valid": False, "reason": "no_visible_mask_pixels"})
            continue
        if pixels <= 0 or obj["bbox_source"] != "instance_mask" or obj["visibility"] != "visible":
            raise ValueError("有框人物必须来自可见实例mask")
        item = _record(obj["track_id"], obj["bbox_xyxy"], width, height, "xyxy_half_open", **{k:v for k,v in identity.items() if k != "track_id"})
        if item["clipped"] or not item["bbox_supervision_valid"]:
            raise ValueError("合成mask tight box应已在图内，拒绝悄悄修复原标注")
        targets.append({**item, "class_id": 0})
    derived_uid = f"detect:synthetic:{m['dataset_id']}:{m['dataset_version']}:{m['seed_id']}:{camera_id}:{frame_index:06d}"
    return _finish({"derived_sample_uid": derived_uid, "UID_origin": "adapter_derived_not_original_frozen_UID",
                    "split": split, "domain": "S", "dataset_id": m["dataset_id"], "dataset_version": m["dataset_version"],
                    "seed_id": m["seed_id"], "trajectory_uid": m["trajectory_uid"], "camera_id": camera_id,
                    "frame_index": frame_index, "source_step": row["source_step"], "width": width, "height": height,
                    "image_path": str(camera_root / "img1" / f"{frame_index:06d}.png"),
                    "source_limitations": ["sample_read_does_not_certify_full_release_or_mask_pixels"]}, targets, [], absent)


def _source_stat(path):
    """仅stat，不哈GT/图像；文件ID、大小、mtime为可读绑定，不是内容证明。"""
    path = Path(path)
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("标注来源路径禁止符号链接/重解析点")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("标注来源必须是普通文件")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class RealBBoxReader:
    """train/val有界clip缓存；不读取图片、test GT、Pose或增强结果。

    GT/gameinfo在第一次合法UID访问时绑定，此后即使LRU逐出也保留其小型stat
    身份，变化时失败而不刷新。max_cached_rows保守计入已缓存clip的全部非空GT
    行（含非检测采样帧）；逐行读取时先逐出LRU，使缓存加在建clip不超行上限。
    返回的深拷贝及调用方batch不在此缓存额度内。Windows spawn每worker各有一个
    冷缓存，因此总额度需要乘worker数；不代表已测得训练吞吐或内存字节峰值。
    """

    def __init__(self, index_path, *, max_cached_clips=2, max_cached_rows=100_000):
        self.max_cached_clips = _int(max_cached_clips, "max_cached_clips", 1)
        self.max_cached_rows = _int(max_cached_rows, "max_cached_rows", 1)
        self.index_path = Path(index_path).absolute()
        before = _source_stat(self.index_path)
        summary = inspect_dataset(self.index_path)
        raw = self.index_path.read_bytes()
        self._index = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if before != _source_stat(self.index_path) or digest != summary["sha256"]:
            raise ValueError("索引读取期间来源发生变化")
        self._index_stat, self._index_sha = before, digest
        self._source = self._index["source"]["source_id"]
        if self._source not in _POLICIES:
            raise ValueError("未实现的真实来源语义")
        self._target_policy, self._ignore_policy = _POLICIES[self._source]
        if (self._index["source"]["target_semantics"], self._index["source"]["ignore_policy"]) != (
                self._target_policy, self._ignore_policy):
            raise ValueError("来源目标/ignore协议变化，必须重新审阅适配")
        self._lookup, self._clips = {}, {}
        for seq in self._index["sequences"]:
            if seq["split"] not in ("train", "val"):
                continue  # 不解析或stat封存test的样本文件路径。
            if (seq["source_id"], seq["target_policy"], seq["ignore_policy"]) != (
                    self._source, self._target_policy, self._ignore_policy):
                raise ValueError("clip与来源语义不一致")
            self._clips[seq["clip_uid"]] = seq
            for frame in seq["detection_frames"]:
                frame_uid = seq["frame_uid_pattern"].format(frame_number=frame)
                sample_uid = seq["detection_sample_uid_pattern"].format(frame_uid=frame_uid)
                key = (seq["split"], sample_uid)
                if key in self._lookup:
                    raise ValueError("UID不属于所选split的唯一冻结检测样本")
                self._lookup[key] = (seq["clip_uid"], frame, frame_uid)
        self._bindings = {}
        self._cache = OrderedDict()
        self._cached_rows = self._hits = self._loads = 0

    def _check_index(self):
        if _source_stat(self.index_path) != self._index_stat:
            raise ValueError("索引来源身份/size/mtime变化，拒绝混版本")

    def _check_clip(self, clip_uid):
        binding = self._bindings[clip_uid]
        for name, (path, identity) in binding.items():
            if _source_stat(path) != identity:
                raise ValueError(f"{name}来源身份/size/mtime变化，拒绝混版本")

    def _bind_clip(self, seq):
        clip_uid = seq["clip_uid"]
        if clip_uid in self._bindings:
            self._check_clip(clip_uid)
            return
        raw_root = self.index_path.parent / seq["sequence_path"]
        root = _relative(self.index_path.parent, seq["sequence_path"])
        paths = {"gt": _relative(root, seq["files"]["gt"])}
        # 先检查原始逻辑路径，不能让resolve掩盖内部symlink/junction。
        _source_stat(raw_root / seq["files"]["gt"])
        if self._source == "soccernet_tracking":
            _source_stat(raw_root / seq["files"]["gameinfo"])
            paths["gameinfo"] = _relative(root, seq["files"]["gameinfo"])
        self._bindings[clip_uid] = {name: (str(path), _source_stat(path)) for name, path in paths.items()}

    def _evict(self):
        _, entry = self._cache.popitem(last=False)
        self._cached_rows -= entry["rows"]

    def _reserve_loading_rows(self, rows):
        if rows > self.max_cached_rows:
            raise ValueError("单clip GT非空行数超过max_cached_rows，不扩大缓存额度")
        while self._cached_rows + rows > self.max_cached_rows:
            self._evict()

    def _load_clip(self, seq):
        clip_uid = seq["clip_uid"]
        self._bind_clip(seq)
        while len(self._cache) >= self.max_cached_clips:
            self._evict()
        binding = self._bindings[clip_uid]
        roles = _roles(Path(binding["gameinfo"][0])) if "gameinfo" in binding else {}
        self._check_clip(clip_uid)
        requested = set(seq["detection_frames"])
        frames, seen, row_count = {}, {}, 0
        gt_path = Path(binding["gt"][0])
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            if _stat_tuple(os.fstat(stream.fileno())) != binding["gt"][1]:
                raise ValueError("GT打开前后来源发生变化")
            for line, row in enumerate(csv.reader(stream), 1):
                if not row:
                    continue
                row_count += 1
                self._reserve_loading_rows(row_count)
                if len(row) != (10 if self._source == "soccernet_tracking" else 9):
                    raise ValueError(f"GT第{line}行列数与来源不符")
                frame = _int(row[0], "frame", 1)
                if frame not in requested:
                    continue
                track_id = _int(row[1], "track ID", 1 if roles else 0)
                ids = seen.setdefault(frame, set())
                if track_id in ids:
                    raise ValueError("同帧重复track ID")
                ids.add(track_id)
                box = [float(v) for v in row[2:6]]
                flag = _int(row[6], "GT flag")
                if flag not in (0, 1):
                    raise ValueError("未知GT flag，不能猜测监督有效性")
                if self._source == "sportsmot_football" and _int(row[7], "SportsMOT class") != 1:
                    raise ValueError("未核对的SportsMOT目标类别")
                role = roles.get(track_id) if roles else "official_on_field_athlete"
                target = (role is not None and role.startswith(("player", "goalkeeper"))) if roles else True
                item = _record(track_id, box, seq["width"], seq["height"], "xywh",
                               role=role, original_gt_row=list(row), source_line=line, gt_flag=flag)
                targets, ignores = frames.setdefault(frame, ([], []))
                if target and flag:
                    targets.append({**item, "class_id": 0})
                else:
                    reason = "invalid_gt_flag" if not flag else ("unknown_role" if role is None else "non_target_role")
                    ignores.append({**item, "ignore_reason": reason})
            if _stat_tuple(os.fstat(stream.fileno())) != binding["gt"][1]:
                raise ValueError("GT读取期间来源发生变化")
        self._check_index()
        self._check_clip(clip_uid)
        entry = {"frames": frames, "rows": row_count}
        self._cache[clip_uid] = entry
        self._cached_rows += row_count
        self._loads += 1
        return entry

    def read(self, sample_uid: str, *, split: str = "train") -> dict:
        if split not in ("train", "val"):
            raise ValueError("BBox准备适配只允许train/val，不开放test内容")
        key = (split, sample_uid)
        if key not in self._lookup:
            raise ValueError("UID不属于所选split的唯一冻结检测样本")
        # 必须通过UID/split定位，之后才stat/open对应GT和gameinfo。
        self._check_index()
        clip_uid, frame, frame_uid = self._lookup[key]
        seq = self._clips[clip_uid]
        if clip_uid in self._cache:
            self._check_clip(clip_uid)
            entry = self._cache[clip_uid]
            self._cache.move_to_end(clip_uid)
            self._hits += 1
        else:
            entry = self._load_clip(seq)
        targets, ignores = entry["frames"].get(frame, ([], []))
        root = _relative(self.index_path.parent, seq["sequence_path"])
        result = copy.deepcopy(_finish({
            "sample_uid": sample_uid, "frame_uid": frame_uid, "clip_uid": clip_uid,
            "match_uid": seq["match_uid"], "track_namespace": seq["track_namespace"],
            "split": split, "domain": "R", "source_id": self._source, "frame_index": frame,
            "width": seq["width"], "height": seq["height"],
            "image_path": str(_relative(root, seq["image_file_pattern"].format(frame_number=frame))),
            "target_policy": self._target_policy, "ignore_policy": self._ignore_policy,
            "source_limitations": list(self._index["limitations"]) +
                (["unannotated_non_target_people_have_no_ignore_geometry"] if self._source == "sportsmot_football" else []),
        }, targets, ignores, []))
        self._check_index()
        self._check_clip(clip_uid)
        return result

    def cache_info(self):
        return {"cached_clips": tuple(self._cache), "cached_rows": self._cached_rows,
                "max_cached_clips": self.max_cached_clips, "max_cached_rows": self.max_cached_rows,
                "hits": self._hits, "clip_loads": self._loads}

    def __getstate__(self):
        self._check_index()
        return {"index_path": str(self.index_path), "max_cached_clips": self.max_cached_clips,
                "max_cached_rows": self.max_cached_rows, "index_stat": self._index_stat,
                "index_sha": self._index_sha, "bindings": copy.deepcopy(self._bindings)}

    def __setstate__(self, state):
        self.__init__(state["index_path"], max_cached_clips=state["max_cached_clips"],
                      max_cached_rows=state["max_cached_rows"])
        if self._index_stat != tuple(state["index_stat"]) or self._index_sha != state["index_sha"]:
            raise ValueError("spawn重建时索引来源变化，拒绝混版本")
        self._bindings = copy.deepcopy(state["bindings"])


def _stat_tuple(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class SyntheticBBoxReader:
    """单seed有界机位offset索引；只读JSONL/小metadata，不读媒体/COCO大JSON。

    首次按行流式校验帧序和已有BBox契约，只留每帧offset/length两个uint64；
    命中后seek请求行并解析一次JSON。LRU只淘汰offset，原stat来源绑定不忘记。
    stat为文件身份/size/mtime绑定而非内容证明，不全哈JSONL。此接口不认证全版
    发布，不改变原版本、UID或split；每个Windows worker独立冷索引/独立预算。
    """
    def __init__(self, seed_directory, *, split="train", max_indexed_cameras=2,
                 max_indexed_frames=1200):
        for name,value in (("max_indexed_cameras",max_indexed_cameras),("max_indexed_frames",max_indexed_frames)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}必须为正整数")
        self.root=Path(seed_directory).absolute()
        self.split=split
        self.max_indexed_cameras=max_indexed_cameras
        self.max_indexed_frames=max_indexed_frames
        self._metadata_bindings={}
        self._manifest=self._read_metadata(self.root/"seed_manifest.json")
        m=self._manifest
        if m["schema"] != "futsalmot_seed_manifest" or m["schema_version"] != "1.0.0":
            raise ValueError("未实现的合成seed schema")
        if split not in ("train","dev") or m["split"] != split:
            raise ValueError("合成样本必须继承原seed split")
        if type(m["frame_count"]) is not int or m["frame_count"] < 1:
            raise ValueError("原frame_count必须为正整数")
        ids=m["camera_ids"]
        if not isinstance(ids,list) or not ids or len(ids) != len(set(ids)):
            raise ValueError("原seed机位清单必须非空且唯一")
        self._cameras={}
        for camera_id in ids:
            raw=self.root/camera_id/"camera.json"
            _source_stat(raw) # resolve前拒绝链接/重解析点。
            camera=self._read_metadata(_relative(self.root,f"{camera_id}/camera.json"))
            if (camera["schema"],camera["schema_version"],camera["camera_id"]) != ("futsalmot_camera_metadata","1.0.0",camera_id):
                raise ValueError("相机metadata不匹配")
            self._cameras[camera_id]=camera
        self._annotation_bindings={}
        self._cache=OrderedDict()
        self._indexed_frames=self._index_builds=self._hits=self._seek_reads=0

    def _read_metadata(self,path):
        before=_source_stat(path)
        with Path(path).open("rb") as f:
            if _stat_tuple(os.fstat(f.fileno())) != before:
                raise ValueError("小metadata打开时来源变化")
            raw=f.read()
            if _stat_tuple(os.fstat(f.fileno())) != before:
                raise ValueError("小metadata读取时来源变化")
        value=json.loads(raw)
        if _source_stat(path) != before:
            raise ValueError("小metadata来源变化")
        self._metadata_bindings[str(Path(path))]=(before,hashlib.sha256(raw).hexdigest())
        return value

    def _check_metadata(self,camera_id=None):
        paths=[str(self.root/"seed_manifest.json")]
        if camera_id is not None:
            paths.append(str(_relative(self.root,f"{camera_id}/camera.json")))
        else:
            paths=list(self._metadata_bindings)
        for path in paths:
            if _source_stat(path) != self._metadata_bindings[path][0]:
                raise ValueError("合成metadata来源身份/size/mtime变化，拒绝混版本")

    def _annotation_path(self,camera_id):
        path=self.root/camera_id/"annotations.jsonl"
        _source_stat(path)
        return _relative(self.root,f"{camera_id}/annotations.jsonl")

    def _check_annotation(self,camera_id):
        path,identity=self._annotation_bindings[camera_id]
        if _source_stat(path) != identity:
            raise ValueError("合成JSONL来源身份/size/mtime变化，拒绝混版本")
        return Path(path),identity

    def _evict(self):
        _,offsets=self._cache.popitem(last=False)
        self._indexed_frames-=len(offsets)//2

    def _index_camera(self,camera_id):
        n=self._manifest["frame_count"]
        if n > self.max_indexed_frames:
            raise ValueError("单机位frame_count超offset索引预算，不能截断或无限缓存")
        if camera_id not in self._annotation_bindings:
            path=self._annotation_path(camera_id)
            self._annotation_bindings[camera_id]=(str(path),_source_stat(path))
        path,identity=self._check_annotation(camera_id)
        while self._cache and (len(self._cache) >= self.max_indexed_cameras or self._indexed_frames+n > self.max_indexed_frames):
            self._evict()
        offsets=array("Q")
        with path.open("rb") as f:
            if _stat_tuple(os.fstat(f.fileno())) != identity:
                raise ValueError("JSONL打开期间来源变化")
            expected=1
            while True:
                start=f.tell();raw=f.readline()
                if not raw:break
                if expected > n or not raw.strip():
                    raise ValueError("合成JSONL存在多行或空行，拒绝伪造帧")
                row=json.loads(raw)
                if type(row.get("frame_index")) is not int or row["frame_index"] != expected:
                    raise ValueError("合成标注帧序列不连续")
                _synthetic_bbox_row(self._manifest,self._cameras[camera_id],row,
                                    camera_root=path.parent,camera_id=camera_id,
                                    frame_index=expected,split=self.split)
                offsets.extend((start,len(raw)))
                expected+=1
            if expected != n+1:
                raise ValueError("合成JSONL缺行，与原frame_count不符")
            if _stat_tuple(os.fstat(f.fileno())) != identity:
                raise ValueError("JSONL索引构建期间来源变化")
        self._check_metadata(camera_id);self._check_annotation(camera_id)
        self._cache[camera_id]=offsets
        self._indexed_frames+=n;self._index_builds+=1
        return offsets

    def read(self,camera_id,frame_index,*,split=None):
        if split is not None and split != self.split:
            raise ValueError("合成样本必须继承原seed split")
        frame_index=_int(frame_index,"frame_index",1)
        if camera_id not in self._cameras or frame_index > self._manifest["frame_count"]:
            raise ValueError("机位或帧不在原seed清单")
        self._check_metadata(camera_id)
        if camera_id in self._cache:
            self._check_annotation(camera_id)
            offsets=self._cache[camera_id]
            self._cache.move_to_end(camera_id);self._hits+=1
        else:
            offsets=self._index_camera(camera_id)
        path,identity=self._check_annotation(camera_id)
        start,length=offsets[2*(frame_index-1):2*frame_index]
        with path.open("rb") as f:
            if _stat_tuple(os.fstat(f.fileno())) != identity:
                raise ValueError("JSONL目标行打开时来源变化")
            f.seek(start);raw=f.read(length)
            if len(raw) != length or _stat_tuple(os.fstat(f.fileno())) != identity:
                raise ValueError("JSONL目标行读取时来源变化")
        row=json.loads(raw)
        if type(row.get("frame_index")) is not int or row["frame_index"] != frame_index:
            raise ValueError("offset目标行与请求帧不一致")
        result=_synthetic_bbox_row(self._manifest,self._cameras[camera_id],row,
                                   camera_root=path.parent,camera_id=camera_id,
                                   frame_index=frame_index,split=self.split)
        self._check_metadata(camera_id);self._check_annotation(camera_id)
        self._seek_reads+=1
        return copy.deepcopy(result)

    def cache_info(self):
        return {"indexed_cameras":tuple(self._cache),"indexed_frames":self._indexed_frames,
                "offset_bytes":sum(x.buffer_info()[1]*x.itemsize for x in self._cache.values()),
                "max_indexed_cameras":self.max_indexed_cameras,"max_indexed_frames":self.max_indexed_frames,
                "index_builds":self._index_builds,"hits":self._hits,"seek_reads":self._seek_reads}

    def __getstate__(self):
        self._check_metadata()
        for camera_id in self._annotation_bindings:self._check_annotation(camera_id)
        return {"root":str(self.root),"split":self.split,
                "max_indexed_cameras":self.max_indexed_cameras,"max_indexed_frames":self.max_indexed_frames,
                "metadata_bindings":copy.deepcopy(self._metadata_bindings),
                "annotation_bindings":copy.deepcopy(self._annotation_bindings)}

    def __setstate__(self,state):
        self.__init__(state["root"],split=state["split"],
                      max_indexed_cameras=state["max_indexed_cameras"],max_indexed_frames=state["max_indexed_frames"])
        if self._metadata_bindings != state["metadata_bindings"]:
            raise ValueError("spawn冷索引重建时小metadata来源变化")
        self._annotation_bindings=copy.deepcopy(state["annotation_bindings"])
        for camera_id in self._annotation_bindings:self._check_annotation(camera_id)
