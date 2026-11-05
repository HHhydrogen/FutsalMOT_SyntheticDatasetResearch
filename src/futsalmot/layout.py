"""正式合成发布的只读迁移规划；不搬运、链接、删除或写出清单。

仅识别当前生产 single_seed_release.publish_release 的 v4 根协议。
默认只检查小型元数据；全量 stat 枚举必须显式开启且生产消费者已闭合。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
import stat

VERSION = "v4_coco17_20seed_long_active"
SCHEMAS = (
    "seed_manifest.schema.json", "frame_annotation.schema.json",
    "trajectory_record.schema.json", "camera_state.schema.json",
    "camera_metadata.schema.json", "pose_schema_futsal_coco17_v1.json",
    "dataset_manifest.schema.json", "release_validation.schema.json",
)
COUNT_KEYS = (
    "rgb_frames", "instance_masks", "number_masks", "person_records",
    "pose_points", "files", "bytes", "visible_number_records",
    "truncated_records", "out_of_frame_records",
)


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe_path(value):
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("路径必须绝对且不含上跳")
    # resolve之前逐级lstat，禁止沿任何现有祖先的symlink/junction进入。
    for q in reversed((p, *p.parents)):
        try:
            info = q.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("路径含符号链接或Windows重解析点: " + str(q))
    return p.resolve(strict=False)


def _read_metadata(path):
    path = _safe_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
        raise ValueError("小型JSON元数据缺失、类型错误或超过4MiB: " + str(path))
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON元数据必须为对象: " + str(path))
    return value, hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_sha(value, label):
    _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "缺合法SHA绑定: " + label)


def _direct_children(root):
    children = {}
    with os.scandir(root) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("根/分组目录含重解析点: " + entry.path)
            kind = "directory" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
            children[entry.name] = kind
    return children


def _nearest_existing(path):
    while not path.exists():
        path = path.parent
    return path


def _stat_files(root):
    """只取相对路径、大小及mtime；从不打开媒体，不跟随重解析点。"""
    root = _safe_path(root)
    records, directories, pending = [], set(), [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("子树含符号链接或Windows重解析点: " + entry.path)
                p = Path(entry.path)
                relative = p.relative_to(root).as_posix()
                if stat.S_ISDIR(info.st_mode):
                    directories.add(relative)
                    pending.append(p)
                elif stat.S_ISREG(info.st_mode):
                    records.append([relative, info.st_size, info.st_mtime_ns])
                else:
                    raise ValueError("子树含非普通文件: " + entry.path)
    return sorted(records), directories


def _expected_seed_files(row):
    expected = {"seed_manifest.json", "trajectory.jsonl", "person_keypoints_coco.json"}
    for camera in row["camera_ids"]:
        expected.update((camera + "/camera.json", camera + "/annotations.jsonl"))
        for folder in ("img1", "mask", "number_mask"):
            expected.update(f"{camera}/{folder}/{i:06d}.png"
                            for i in range(1, row["frames_per_camera"] + 1))
    return expected


def _seed_file_plan(root, target, row):
    """只允许已由调用方校验的小型row；用于发布闭合后的stat枚举。"""
    source = root / row["split"] / row["seed_id"]
    records, directories = _stat_files(source)
    expected = _expected_seed_files(row)
    _require({r[0] for r in records} == expected, "seed文件白名单不符: " + row["seed_id"])
    expected_dirs = set()
    for name in expected:
        expected_dirs.update(p.as_posix() for p in Path(name).parents if p != Path("."))
    _require(directories == expected_dirs, "seed目录白名单不符: " + row["seed_id"])
    _require(_canonical_sha(records) == row["file_stat_fingerprint"], "seed stat绑定变化")
    _require(len(records) == row["counts"]["files"], "seed实际文件数与发布绑定不符")
    _require(sum(r[1] for r in records) == row["counts"]["bytes"], "seed实际字节与发布绑定不符")
    return [
        {"source": str(source / rel), "target": str(target / row["split"] / row["seed_id"] / rel),
         "relative": f'{row["split"]}/{row["seed_id"]}/{rel}',
         "bytes": size, "mtime_ns": mtime, "operation": "hardlink"}
        for rel, size, mtime in records
    ]


def plan_synthetic_layout(release_root, g5_receipt, *, target_root=None,
                          writers_closed=False, enumerate_files=False):
    """返回内存计划及状态；任何状态下均不创建目录或写文件。

    writers_closed是调用方对渲染/导出/G写入/发布清理全部闭合的显式断言，
    G5自身不能证明此事。默认不枚举seed媒体。READY_FOR_ENUMERATION不代表
    可执行迁移；只有PLANNED_DRY_RUN包含逐文件计划，且仍无apply能力。
    正式root/receipt尚不存在为AWAITING，已出现但绑定不符为BLOCKED。
    """
    result = {"status": "BLOCKED", "writes_performed": False,
              "media_read_or_hashed": False, "media_enumerated": False,
              "files": [], "execution_supported": False}
    try:
        root = _safe_path(release_root)
        _require(root.name == VERSION and root.parent.name == "synthetic_research_v1",
                 "只接受正式synthetic_research_v1/v4发布根，不能使用_staging/test")
        target = _safe_path(target_root if target_root is not None else root.parent.parent / "synthetic")
        _require(target == root.parent.parent / "synthetic", "目标必须是同一数据根下已审核的synthetic")
        g5_path = _safe_path(g5_receipt)
        result.update(source_root=str(root), target_root=str(target))
        if not root.exists() or not g5_path.exists():
            return {**result, "status": "AWAITING", "reason": "正式发布根或明确G5_complete收据尚未出现"}
        if writers_closed is not True:
            return {**result, "status": "AWAITING", "reason": "尚未明确确认全部生产/发布清理消费者闭合"}
        _require(type(enumerate_files) is bool, "enumerate_files必须是明确bool")
        _require(root.is_dir(), "正式发布根不是目录")
        _require(not target.exists(), "canonical目标已存在；不覆盖或推测可续做")
        _require(root.stat().st_dev == _nearest_existing(target.parent).stat().st_dev,
                 "硬链接目标必须与发布根同卷")
        manifest, manifest_sha = _read_metadata(root / "dataset_manifest.json")
        receipt, receipt_sha = _read_metadata(root / "validation_receipt.json")
        g5, g5_sha = _read_metadata(g5_path)
        _require(g5["status"] == "published" and _safe_path(g5["release_path"]) == root,
                 "G5未published或绑定其他发布路径")
        _require(g5["manifest_sha256"] == manifest_sha and g5["validation_sha256"] == receipt_sha,
                 "G5与原根manifest/validation字节绑定不符")
        _require(manifest["schema"] == "futsalmot_dataset_manifest_v1"
                 and manifest["dataset_id"] == "futsalmot_synth_multitask"
                 and manifest["dataset_version"] == VERSION and manifest["diagnostic_only"] is False,
                 "非本次正式v4数据")
        _require(manifest["fps"] == 10 and manifest["resolution"] == [3840, 2160], "发布时空规格不符")
        without_fingerprint = {k: v for k, v in manifest.items() if k != "fingerprint"}
        _require(_canonical_sha(without_fingerprint) == manifest["fingerprint"]
                 == receipt["fingerprint"] == g5["fingerprint"], "根fingerprint绑定不符")
        _require(receipt["schema"] == "futsalmot_release_validation_v1"
                 and receipt["status"] == "PASS"
                 and receipt["dataset_manifest_sha256"] == manifest_sha
                 and receipt["all_owned_sessions_normal_exit"] is True, "缺正式根PASS/正常退出绑定")
        for key in ("frozen_sources_sha256", "plan_sha256"):
            _require_sha(manifest[key], key)
        _require(isinstance(manifest["release_attempt"], str) and bool(manifest["release_attempt"]),
                 "缺原发布attempt绑定")
        _require(manifest["fingerprint_scope"] ==
                 "canonical_manifest_without_fingerprint_including_source_validation_stat_bindings_no_rgb_byte_hash",
                 "未知fingerprint操作定义")
        rows = manifest["seeds"]
        _require(len(rows) == 20 and len({r["seed_id"] for r in rows}) == 20
                 and len({r["root_seed"] for r in rows}) == 20, "必须20个唯一manifest seed")
        _require(sum(r["split"] == "train" for r in rows) == 16
                 and sum(r["split"] == "dev" for r in rows) == 4, "必须原16train/4dev")
        _require(set(manifest["schema_sha256"]) == set(SCHEMAS), "发布schema集合不符")
        schema_bytes = 0
        for name in SCHEMAS:
            _, digest = _read_metadata(root / "schemas" / name)
            _require(digest == manifest["schema_sha256"][name], "发布schema绑定变化: " + name)
            schema_bytes += (root / "schemas" / name).stat().st_size
        seeds = []
        totals = dict.fromkeys(COUNT_KEYS, 0)
        for row in rows:
            seed_id, n, cameras = row["seed_id"], row["frames_per_camera"], row["camera_ids"]
            for key in ("source_frames_sha256", "recipe_sha256", "protocol_sha256",
                        "code_bundle_sha256", "module_dll_sha256", "pose_calibration_sha256",
                        "seed_manifest_sha256", "verification_sha256", "file_stat_fingerprint",
                        "session_exits_sha256", "coco_export_sha256"):
                _require_sha(row[key], key)
            for key, names in (("schemas", SCHEMAS[:6]), ("release_schemas", SCHEMAS[6:])):
                _require(set(row[key]) == set(names), "seed原schema绑定集合不符")
                for name in names:
                    _require_sha(row[key][name], name)
            _require(type(row["root_seed"]) is int and 30001 <= row["root_seed"] <= 30140
                     and seed_id == f'seed_{row["root_seed"]:08x}', "seed名称/原数字绑定不符")
            _require(row["split"] in ("train", "dev") and type(n) is int and 100 <= n <= 600,
                     "split/实际帧数不符")
            _require(len(cameras) in (6, 7) and len(set(cameras)) == len(cameras)
                     and {"C5", "C1", "C2", "C3", "C4"} <= set(cameras)
                     and set(cameras) <= {"C5", "C1", "C2", "C3", "C4", "P01", "P02", "P03", "P04"},
                     "机位必须继承原必选及实际局部集合")
            seed, digest = _read_metadata(root / row["split"] / seed_id / "seed_manifest.json")
            _require(digest == row["seed_manifest_sha256"], "seed_manifest原字节绑定不符")
            _require(seed["seed_id"] == seed_id and seed["root_seed"] == f'{row["root_seed"]:08x}'
                     and seed["split"] == row["split"] and seed["frame_count"] == n
                     and seed["camera_ids"] == cameras, "seed身份/split/帧数/机位与根不符")
            _require(isinstance(seed["trajectory_uid"], str) and bool(seed["trajectory_uid"])
                     and seed["dataset_id"] == manifest["dataset_id"]
                     and isinstance(seed["dataset_version"], str) and bool(seed["dataset_version"]),
                     "缺原UID/seed来源版本")
            selection = row["source_selection"]
            _require(selection["mode"] == "active_contiguous_v1" and selection["frame_count"] == n
                     and type(selection["start_step"]) is int
                     and type(selection["end_step_exclusive"]) is int
                     and type(selection["full_frame_count"]) is int
                     and 0 <= selection["start_step"] < selection["end_step_exclusive"]
                     <= selection["full_frame_count"]
                     and selection["end_step_exclusive"] - selection["start_step"] == n,
                     "原来源连续区间/实际帧数不符")
            for key in ("full_frames_sha256", "full_meta_sha256", "selection_sha256"):
                _require_sha(selection[key], key)
            _require(seed["source_selection"] == selection
                     and seed["trajectory_uid"] == row["source_frames_sha256"]
                     and seed["generation"]["protocol_sha256"] == row["protocol_sha256"]
                     and seed["generation"]["code_bundle_sha256"] == row["code_bundle_sha256"],
                     "原seed来源/协议绑定与发布row不符")
            # 复用v3捕获的seed可合法进入v4根；原seed版本必须保留，不能改写为v4。
            count = row["counts"]
            f = n * len(cameras)
            _require([count[k] for k in ("rgb_frames", "instance_masks", "number_masks",
                                         "person_records", "pose_points", "files")]
                     == [f, f, f, f * 10, f * 170, 3 * f + 2 * len(cameras) + 3],
                     "seed实际数量绑定不符")
            for key in COUNT_KEYS:
                _require(type(count[key]) is int and count[key] >= 0, "非法count: " + key)
                totals[key] += count[key]
            seeds.append({**copy.deepcopy(row), "directory": f'{row["split"]}/{seed_id}',
                          "trajectory_uid": seed["trajectory_uid"],
                          "original_seed_dataset_version": seed["dataset_version"]})
        _require(totals == manifest["totals"], "累计counts与原根totals不符")
        cameras = sum(len(r["camera_ids"]) for r in rows)
        full_files = totals["files"] + len(SCHEMAS) + 2
        _require(receipt["seed_count"] == 20 and receipt["camera_count"] == cameras
                 and receipt["expected_files"] == full_files == g5["files"], "根文件/seed/机位计数不符")
        for key in ("rgb_frames", "person_records", "pose_points"):
            _require(receipt[key] == totals[key], "根PASS数量不符: " + key)
        _require(g5["logical_bytes"] == totals["bytes"] + schema_bytes
                 + (root / "dataset_manifest.json").stat().st_size
                 + (root / "validation_receipt.json").stat().st_size, "G5原发布字节总量不符")
        bindings = {"release_root": str(root), "dataset_manifest": str(root / "dataset_manifest.json"),
                    "manifest_sha256": manifest_sha, "validation_receipt": str(root / "validation_receipt.json"),
                    "validation_sha256": receipt_sha, "g5_receipt": str(g5_path), "g5_sha256": g5_sha,
                    "fingerprint": manifest["fingerprint"], "dataset_version": VERSION}
        index = {"schema_version": "canonical_synthetic_dataset_v1",
                 "status": "PLANNED_NOT_MIGRATED", "source_release": bindings,
                 "seeds": seeds, "totals": copy.deepcopy(totals),
                 "limitations": copy.deepcopy(receipt.get("limitations", []))}
        result.update(status="READY_FOR_ENUMERATION", source_binding=bindings,
                      candidate_dataset=index, counts=totals, seed_count=20, camera_count=cameras,
                      reason="小型根/seed绑定已通过；尚未枚举，不是已可迁移或已整理完成")
        if not enumerate_files:
            return result
        result["media_enumerated"] = True
        _require(_direct_children(root) == {
            "dataset_manifest.json": "file", "validation_receipt.json": "file",
            "schemas": "directory", "train": "directory", "dev": "directory"},
            "正式根直接子项白名单不符")
        _require(_direct_children(root / "schemas") == {n: "file" for n in SCHEMAS},
                 "schema目录白名单不符")
        for split in ("train", "dev"):
            _require(_direct_children(root / split) ==
                     {r["seed_id"]: "directory" for r in rows if r["split"] == split},
                     "原split的seed目录集合不符")
        files = []
        for row in rows:
            files.extend(_seed_file_plan(root, target, row))
        _require(len(files) + len(SCHEMAS) + 2 == full_files, "正式根实际文件数变化")
        relocation = {"schema_version": "synthetic_relocation_plan_v1", "dry_run": True,
                      "source_binding": bindings, "files": files,
                      "planned_dataset_json": str(target / "dataset.json"),
                      "linked_file_count": len(files), "logical_bytes": totals["bytes"],
                      "media_copy_bytes": 0, "original_release_retained": True,
                      "original_receipts_rewritten": False}
        return {**result, "status": "PLANNED_DRY_RUN", "media_enumerated": True,
                "files": files, "relocation": relocation,
                "reason": "精确只读计划已生成，未创建任何目标或RELOCATION文件"}
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return {**result, "status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}",
                "files": []}
