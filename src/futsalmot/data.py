"""只读索引和纯CPU监督契约；不实现YOLO标签或图像适配器。"""
from __future__ import annotations
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import uuid

DEFAULT_REAL = {
    "soccernet": "G:/FutsalMOT_Dataset/real/soccernet/dataset.json",
    "sportsmot": "G:/FutsalMOT_Dataset/real/sportsmot/dataset.json",
}

def load_config(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)  # JSON也是YAML1.2子集；附带配置无需第三方解析器。
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("非JSON兼容YAML需要环境中的PyYAML；未安装时不自动安装。") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("配置根必须为对象")
    return value

def inspect_dataset(path: str | Path) -> dict:
    """只读dataset.json；不打开任何原图、GT或样本路径。"""
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    if d.get("schema_version") != "canonical_real_dataset_v1":
        raise ValueError("仅支持已交付canonical_real_dataset_v1索引")
    matches = d["matches"]
    seqs = d["sequences"]
    split_by_match = {m["match_uid"]: m["split"] for m in matches}
    if len(split_by_match) != len(matches):
        raise ValueError("重复match UID")
    counts = {s: {"matches": 0, "clips": 0, "native_frames": 0, "detection_images": 0} for s in ("train", "val", "test")}
    for split in split_by_match.values():
        counts[split]["matches"] += 1
    clips = set()
    for seq in seqs:
        if seq["clip_uid"] in clips or not seq["complete"] or not seq["selected"]:
            raise ValueError("clip重复、不完整或未选中")
        clips.add(seq["clip_uid"])
        if seq["split"] != split_by_match[seq["match_uid"]]:
            raise ValueError("clip与父比赛split不一致")
        c = counts[seq["split"]]
        frames = seq["official_frame_count"]
        sampled = seq["detection_frames"]
        if len(set(sampled)) != len(sampled) or any(not 1 <= v <= frames for v in sampled):
            raise ValueError("检测帧重复或越界")
        c["clips"] += 1
        c["native_frames"] += frames
        c["detection_images"] += len(sampled)
    return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "status": d["status"], "panel_id": d["source_panel_id"], "counts": counts, "joint_use_allowed": d["joint_use_allowed"], "cross_source_disjoint_certified": d["cross_source_disjoint_certified"], "read_scope": "index_metadata_only_no_images_or_GT"}

def detection_train_uids(path: str | Path) -> list[str]:
    """只展开已冻结train索引的原sample UID，不打开样本内容。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for seq in d["sequences"]:
        if seq["split"] != "train":
            continue
        for frame in seq["detection_frames"]:
            uid = seq["frame_uid_pattern"].format(frame_number=frame)
            out.append(seq["detection_sample_uid_pattern"].format(frame_uid=uid))
    if not out or len(set(out)) != len(out):
        raise ValueError("train检测UID为空或重复")
    return out

def _seed(*values) -> int:
    raw = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")

def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value

class CycleSampler:
    """分域洗牌循环；每个曝光的增强seed与其他域取样次数无关。"""
    def __init__(self, uids, *, task: str, domain: str, seed: int):
        self.uids = tuple(uids)
        if not self.uids or len(set(self.uids)) != len(self.uids):
            raise ValueError("UID池必须非空且唯一")
        self.task, self.domain, self.seed = task, domain, seed
        self.pool_sha = hashlib.sha256(json.dumps(self.uids, ensure_ascii=True).encode()).hexdigest()
        self.rng = random.Random(_seed(task, domain, seed))
        self.order = list(range(len(self.uids)))
        self.rng.shuffle(self.order)
        self.position = self.epoch = self.draws = 0

    def take(self, count: int) -> list[dict]:
        if count < 0:
            raise ValueError("count不能为负")
        out = []
        for _ in range(count):
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self.rng.shuffle(self.order)
            out.append({"uid": self.uids[self.order[self.position]], "domain": self.domain, "epoch": self.epoch, "exposure_index": self.draws, "augmentation_seed": _seed(self.task, self.domain, self.seed, self.draws)})
            self.position += 1
            self.draws += 1
        return out

    def state_dict(self) -> dict:
        return {"version": 1, "task": self.task, "domain": self.domain, "seed": self.seed, "pool_sha256": self.pool_sha, "order": list(self.order), "position": self.position, "epoch": self.epoch, "draws": self.draws, "rng_state": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        # 所有检查和Random.setstate在独立候选上完成，非法状态不部分改变本对象。
        try:
            if not isinstance(state, dict) or type(state["version"]) is not int:
                raise ValueError("无效采样状态结构")
            for key, value in [("version", 1), ("task", self.task), ("domain", self.domain), ("seed", self.seed), ("pool_sha256", self.pool_sha)]:
                if state[key] != value:
                    raise ValueError("取样状态与当前身份/UID池不一致")
            n = len(self.uids)
            order = list(state["order"])
            if any(type(v) is not int for v in order) or sorted(order) != list(range(n)):
                raise ValueError("无效采样排列")
            for key in ("position", "epoch", "draws"):
                if type(state[key]) is not int or state[key] < 0:
                    raise ValueError("采样位置/epoch/曝光数须为非负整数")
            if state["position"] > n or state["draws"] != state["epoch"] * n + state["position"]:
                raise ValueError("曝光数与epoch/位置不一致")
            candidate_rng = random.Random(0)
            candidate_rng.setstate(_tuples(state["rng_state"]))
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("缺字段或损坏的采样状态") from exc
        self.rng = candidate_rng
        self.order, self.position = order, state["position"]
        self.epoch, self.draws = state["epoch"], state["draws"]


class PrefetchSampler:
    """有界事务采样：reserve只派发，FIFO commit才推进可检查点曝光。

    不执行worker/优化器。未来loop仅在更新成功后commit；取消/恢复会丢弃全部
    未提交请求，重新从已提交CycleSampler游标派发相同UID/增强seed序列。
    每实例只拥有一个stream。R_extra沿用旧CycleSampler的随机namespace，
    request.data_domain明确为R；它不是S或新增数据来源。worker只消费请求副本。
    """
    SCHEMA = "committed_prefetch_sampler_v1"

    def __init__(self, sampler: CycleSampler, *, max_pending_batches: int = 2):
        if not isinstance(sampler, CycleSampler):
            raise ValueError("需要已有CycleSampler流")
        if type(max_pending_batches) is not int or max_pending_batches < 1:
            raise ValueError("max_pending_batches必须为正整数")
        if sampler.domain not in ("R", "S", "R_extra"):
            raise ValueError("明确支持R/S/R_extra流，不能猜测数据域")
        self.max_pending_batches = max_pending_batches
        self._template = (sampler.uids, sampler.task, sampler.domain, sampler.seed)
        self._committed = self._clone(sampler.state_dict())
        self._dispatched = self._clone(self._committed.state_dict())
        self._pending = deque()
        self._instance = uuid.uuid4().hex
        self._generation = self._sequence = 0
        self._lock = threading.Lock()

    def _clone(self, state):
        uids, task, domain, seed = self._template
        candidate = CycleSampler(uids, task=task, domain=domain, seed=seed)
        candidate.load_state_dict(deepcopy(state))
        return candidate

    @property
    def pending_batches(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def committed_draws(self) -> int:
        with self._lock:
            return self._committed.draws

    def reserve(self, count: int) -> dict:
        if type(count) is not int or count < 1:
            raise ValueError("reserve count必须为正整数")
        with self._lock:
            if len(self._pending) >= self.max_pending_batches:
                raise RuntimeError("prefetch队列已满，拒绝无限派发")
            candidate = self._clone(self._dispatched.state_dict())
            rows = candidate.take(count)
            token = f"{self._instance}:{self._generation}:{self._sequence}"
            stream = candidate.domain
            request = {"token": token, "stream": stream,
                       "data_domain": "R" if stream == "R_extra" else stream,
                       "count": count, "samples": rows}
            internal = {"request": deepcopy(request), "end_state": candidate.state_dict()}
            self._pending.append(internal)
            self._dispatched = candidate
            self._sequence += 1
            return deepcopy(request)

    def commit(self, token: str) -> dict:
        """只提交队首token；返回原请求副本供未来loop记录真实曝光。"""
        with self._lock:
            if not isinstance(token, str) or not self._pending or token != self._pending[0]["request"]["token"]:
                raise ValueError("过期、重复、未知或非FIFO队首token")
            first = self._pending[0]
            candidate = self._clone(first["end_state"])
            result = deepcopy(first["request"])
            self._committed = candidate
            self._pending.popleft()
            return result

    def state_dict(self) -> dict:
        """不保存pending/派发位置/token；仅保存已提交的身份、游标与RNG。"""
        with self._lock:
            return {"schema_version": self.SCHEMA,
                    "sampler": deepcopy(self._committed.state_dict())}

    def cancel_pending(self) -> int:
        with self._lock:
            candidate = self._clone(self._committed.state_dict())
            cancelled = len(self._pending)
            self._dispatched = candidate
            self._pending.clear()
            self._generation += 1
            self._sequence = 0
            return cancelled

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            if not isinstance(state, dict) or set(state) != {"schema_version", "sampler"} or state["schema_version"] != self.SCHEMA:
                raise ValueError("无效事务采样检查点schema；不能恢复未提交prefetch")
            # 两个候选均成功后再赋值，坏状态/错误pool不改变当前队列或游标。
            committed = self._clone(state["sampler"])
            dispatched = self._clone(committed.state_dict())
            self._committed, self._dispatched = committed, dispatched
            self._pending.clear()
            self._generation += 1
            self._sequence = 0

def pose_supervision_masks(keypoints, *, pose_supervision_valid: bool) -> dict:
    """表达监督契约，不是已经接入YOLO criterion的实现。"""
    if len(keypoints) != 17 or any(len(k) != 3 for k in keypoints):
        raise ValueError("需要17个[x,y,v]")
    for x, y, v in keypoints:
        if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2) or any(isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) for z in (x, y)) or (v == 0 and (x != 0 or y != 0)):
            raise ValueError("非法COCO17可见性或v0坐标")
    return {"bbox_loss_valid": True, "pose_instance_valid": bool(pose_supervision_valid), "coordinate_valid": [bool(pose_supervision_valid and k[2] != 0) for k in keypoints], "kobj_loss_valid": [bool(pose_supervision_valid)] * 17, "kobj_target": [int(k[2] != 0) for k in keypoints]}

def instance_mask_contract(mask, instance_ids) -> dict:
    """保留不连通/孔洞的整数像素，不转桥接polygon、不resize。"""
    if not mask or not mask[0] or any(len(row) != len(mask[0]) for row in mask):
        raise ValueError("mask必须非空矩形")
    original_ids = tuple(instance_ids)
    ids = set(original_ids)
    if not ids or len(ids) != len(original_ids) or 0 in ids or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in ids):
        raise ValueError("实例ID必须为唯一正整数，背景为0")
    counts = Counter()
    for row in mask:
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0 or (v and v not in ids):
                raise ValueError("非法像素ID")
            counts[v] += 1
    return {"height": len(mask), "width": len(mask[0]), "pixel_counts": {v: counts[v] for v in sorted(ids)}, "background_pixels": counts[0], "representation": "original_integer_instances_no_polygon_conversion", "model_adapter_implemented": False}
