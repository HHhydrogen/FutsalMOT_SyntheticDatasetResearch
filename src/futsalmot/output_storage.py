"""D盘只保留小控制入口，G盘原位保存训练证据；不加载模型或扫描媒体。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

SCHEMA = "futsalmot_direct_output_v1"
MANIFEST_SCHEMA = "futsalmot_control_manifest_v1"
MARKER_NAME = "RUN_LOCATION.json"
MANIFEST_NAME = "CONTROL_MANIFEST.json"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_MARKER_BYTES = 64 * 1024
MAX_ENTRIES = 4096
INITIAL_FILES = frozenset({
    "admission.json", "storage_revision.json", "R50_reference.json", "job_table.json",
    "code_bindings.json", MARKER_NAME, "launch.py", "requirements.lock",
})
FINAL_FILES = frozenset({
    "dispatch.json", "R50_protocol.json", "R100_protocol.json", "status.json",
    "budget_choice.json", "STOP.json",
})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _absolute(value: Path) -> Path:
    value = Path(value)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError("必须提供无 '..' 的绝对路径")
    return value.absolute()


def _no_links(path: Path, *, exists: bool = True) -> os.stat_result | None:
    """逐级lstat，不用resolve跟随junction；不存在的最终目标亦检查其祖先。"""
    lineage = list(reversed(path.parents)) + [path]
    final = None
    for index, part in enumerate(lineage):
        try:
            observed = part.lstat()
        except FileNotFoundError:
            if exists:
                raise ValueError(f"路径不存在: {part}") from None
            continue
        if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"禁止symlink/junction/reparse路径: {part}")
        if index < len(lineage) - 1 and not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"祖先不是目录: {part}")
        final = observed
    return final


def _directory(path: Path) -> None:
    info = _no_links(path)
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"必须为普通目录: {path}")


def _paths(control_group: Path, research_root: Path, output_base: Path) -> tuple[Path, Path, Path]:
    source, root, base = map(_absolute, (control_group, research_root, output_base))
    _directory(root)
    _directory(root / "runs")
    _directory(base)
    name = source.name
    invalid_name = (
        not name or name in {".", ".."} or name.endswith((".", " "))
        or any(char in name for char in '<>:"/\\|?*')
        or name.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)},
            *{f"LPT{i}" for i in range(1, 10)},
        }
    )
    if invalid_name or source.parent != root / "runs":
        raise ValueError("control_group必须是research_root/runs的精确直属子目录")
    if base.is_relative_to(root) or root.is_relative_to(base):
        raise ValueError("输出根必须与研究代码根分离，不能嵌套")
    _directory(source)
    target = base / name
    _no_links(target, exists=False)
    return source, base, target


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read(path: Path, *, limit: int = MAX_MARKER_BYTES) -> bytes:
    before = _no_links(path)
    if before is None or not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError(f"不是允许大小的普通文件: {path}")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != _identity(before):
            raise RuntimeError(f"读取前文件身份已变化: {path}")
        data = stream.read(limit + 1)
        finished = os.fstat(stream.fileno())
    after = _no_links(path)
    if (len(data) > limit or after is None or _identity(before) != _identity(finished)
            or _identity(before) != _identity(after) or len(data) != before.st_size):
        raise RuntimeError(f"读取期间文件已变化: {path}")
    return data


def _encode(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _json(data: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON重复字段: {key}")
            result[key] = value
        return result
    result = json.loads(data, object_pairs_hook=unique_pairs)
    if not isinstance(result, dict):
        raise ValueError("JSON必须为object")
    return result


def _write_new(path: Path, data: bytes) -> None:
    _no_links(path, exists=False)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if _read(path, limit=len(data)) != data:
        raise RuntimeError(f"写后读回不一致: {path}")


def _validate_marker(data: bytes, source: Path, target: Path) -> dict:
    value = _json(data)
    if (set(value) != {"schema", "version", "storage_mode", "control_group", "output_group", "created_utc"}
            or value.get("schema") != SCHEMA or type(value.get("version")) is not int
            or value["version"] != 1 or value.get("storage_mode") != "direct_g"
            or value.get("control_group") != str(source) or value.get("output_group") != str(target)
            or not isinstance(value.get("created_utc"), str) or _encode(value) != data):
        raise ValueError("RUN_LOCATION不是当前精确路径绑定")
    created = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
    if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
        raise ValueError("绑定时间必须为UTC")
    return value


def create_output_binding(control_group: Path, *, research_root: Path, output_base: Path) -> dict:
    """仅新建路径绑定；目标已存在时拒绝，不把绑定当作训练状态。"""
    source, _, target = _paths(control_group, research_root, output_base)
    if os.path.lexists(source / MARKER_NAME) or os.path.lexists(target):
        raise FileExistsError("绑定或G输出组已存在，禁止覆盖；已绑定组请调用resolve_output_group")
    marker = {"schema": SCHEMA, "version": 1, "storage_mode": "direct_g",
              "control_group": str(source), "output_group": str(target), "created_utc": _utc()}
    data = _encode(marker)
    # 创建失败时保留部分绑定供诊断，不静默删改已有证据或猜测恢复。
    target.mkdir(exist_ok=False)
    _write_new(target / MARKER_NAME, data)
    _write_new(source / MARKER_NAME, data)
    resolve_output_group(source, research_root=research_root, output_base=output_base)
    return marker


def resolve_output_group(control_group: Path, *, research_root: Path, output_base: Path) -> Path:
    """只接受按同名直属目标推导的双端同bytes收据，不跟随收据任意指针。"""
    source, _, target = _paths(control_group, research_root, output_base)
    _directory(target)
    left = _read(source / MARKER_NAME)
    _validate_marker(left, source, target)
    if _read(target / MARKER_NAME) != left:
        raise ValueError("D/G双端RUN_LOCATION不一致")
    return target


def _allowed(relative: str, *, final: bool) -> bool:
    if "/" not in relative:
        return relative in INITIAL_FILES or (final and relative in FINAL_FILES)
    item = Path(relative)
    return item.parts[0] == "code" and (item.suffix == ".py" or relative == "code/requirements.lock")


def _collect(source: Path, *, final: bool) -> dict[str, bytes]:
    payload = {}
    total = 0
    seen = 0
    def visit(directory: Path):
        nonlocal total, seen
        _directory(directory)
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            seen += 1
            if seen > MAX_ENTRIES:
                raise ValueError("控制树条目过多")
            info = _no_links(child)
            relative = child.relative_to(source).as_posix()
            if info is not None and stat.S_ISDIR(info.st_mode):
                if relative != "code" and not relative.startswith("code/"):
                    raise ValueError(f"控制树出现非白名单目录: {relative}")
                visit(child)
            elif info is not None and stat.S_ISREG(info.st_mode):
                if not _allowed(relative, final=True):
                    raise ValueError(f"控制树出现非白名单文件: {relative}")
                if info.st_size > MAX_FILE_BYTES:
                    raise ValueError(f"控制文件超过16MiB上限: {relative}")
                if not _allowed(relative, final=final):
                    continue
                data = _read(child, limit=MAX_FILE_BYTES)
                total += len(data)
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("控制快照总量超过32MiB上限")
                payload[relative] = data
            else:
                raise ValueError(f"控制树含特殊文件: {relative}")
    visit(source)
    return payload


def _rows(payload: dict[str, bytes]) -> list[dict]:
    return [{"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for relative, data in sorted(payload.items())]


def _target_entries(directory: Path) -> set[str]:
    """仅检查小control目录；不枚举G组内的模型、预测或日志。"""
    files = set()
    count = 0
    def visit(current: Path):
        nonlocal count
        _directory(current)
        for child in current.iterdir():
            count += 1
            if count > MAX_ENTRIES:
                raise ValueError("目标控制树条目过多")
            info = _no_links(child)
            relative = child.relative_to(directory).as_posix()
            if info is not None and stat.S_ISDIR(info.st_mode):
                if relative != "code" and not relative.startswith("code/"):
                    raise ValueError(f"目标控制树含未知目录: {relative}")
                visit(child)
            elif info is not None and stat.S_ISREG(info.st_mode):
                files.add(relative)
            else:
                raise ValueError(f"目标控制树含特殊文件: {relative}")
    visit(directory)
    return files


@contextmanager
def _capture_lock(target: Path):
    path = target / ".control_capture.lock"
    data = _encode({"pid": os.getpid(), "created_utc": _utc(), "token": uuid.uuid4().hex})
    _write_new(path, data)
    identity = _identity(path.lstat())
    try:
        yield
    finally:
        # 只退休本调用仍持有且原bytes的锁；未知/陈旧锁不猜测清理。
        if _identity(path.lstat()) != identity or _read(path) != data:
            raise RuntimeError("控制快照锁身份已变化，保留现场")
        path.unlink()


def _result(source: Path, target: Path, rows: list[dict], final: bool, manifest: bytes | None) -> dict:
    return {"control_group": str(source), "output_group": str(target),
            "control_directory": str(target / "control"), "files": rows,
            "file_count": len(rows), "total_bytes": sum(item["bytes"] for item in rows),
            "final": final, "scope": "control_snapshot_only",
            "manifest_path": str(target / "control" / MANIFEST_NAME) if manifest is not None else None,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest() if manifest is not None else None}


def capture_control(control_group: Path, *, research_root: Path, output_base: Path, final: bool = False) -> dict:
    """精确复制有界控制小文件，final封闭唯一清单；不搬/哈希训练大产物。"""
    if type(final) is not bool:
        raise TypeError("final必须为bool")
    source = _absolute(control_group)
    target = resolve_output_group(source, research_root=research_root, output_base=output_base)
    with _capture_lock(target):
        payload = _collect(source, final=final)
        rows = _rows(payload)
        if payload.get(MARKER_NAME) != _read(target / MARKER_NAME):
            raise ValueError("同步时源绑定已变化")
        destination = target / "control"
        _no_links(destination, exists=False)
        if not destination.exists():
            destination.mkdir(exist_ok=False)
        _directory(destination)
        existing = _target_entries(destination)
        manifest_path = destination / MANIFEST_NAME
        if MANIFEST_NAME in existing:
            if not final:
                raise ValueError("控制快照已封闭，后续只能以final=True精确复核")
            manifest_data = _read(manifest_path, limit=MAX_TOTAL_BYTES)
            manifest = _json(manifest_data)
            expected = {"schema": MANIFEST_SCHEMA, "scope": "control_snapshot_only", "state": "SEALED",
                        "storage_mode": "direct_g", "control_group": str(source), "output_group": str(target),
                        "files": rows, "file_count": len(rows),
                        "total_bytes": sum(len(data) for data in payload.values()), "media_files_copied": 0}
            if set(manifest) != set(expected) | {"created_utc"} or any(manifest.get(key) != value for key, value in expected.items()) or _encode(manifest) != manifest_data:
                raise ValueError("已封闭控制清单与当前源不一致，拒绝覆盖")
        else:
            manifest_data = None
        expected_paths = set(payload)
        if manifest_data is not None and existing != expected_paths | {MANIFEST_NAME}:
            raise ValueError("已封闭目标缺少或新增文件，拒绝修复覆盖")
        if existing - expected_paths - ({MANIFEST_NAME} if manifest_data is not None else set()):
            raise ValueError("目标控制树含不属于当前快照的文件，拒绝覆盖")
        # 先检查全部已有文件，避免发现后部冲突前先写入部分新文件。
        for relative, data in payload.items():
            output = destination / relative
            if relative in existing and _read(output, limit=MAX_FILE_BYTES) != data:
                raise ValueError(f"目标控制文件bytes冲突: {relative}")
        for relative, data in payload.items():
            if relative in existing:
                continue
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            _directory(output.parent)
            _write_new(output, data)
        if _collect(source, final=final) != payload:
            raise RuntimeError("同步期间D控制树变化，未封闭清单")
        if resolve_output_group(source, research_root=research_root, output_base=output_base) != target:
            raise RuntimeError("同步期间路径绑定变化")
        if final and manifest_data is None:
            manifest_data = _encode({"schema": MANIFEST_SCHEMA, "scope": "control_snapshot_only", "state": "SEALED",
                "storage_mode": "direct_g", "control_group": str(source), "output_group": str(target),
                "created_utc": _utc(), "files": rows, "file_count": len(rows),
                "total_bytes": sum(len(data) for data in payload.values()), "media_files_copied": 0})
            _write_new(manifest_path, manifest_data)
        return _result(source, target, rows, final, manifest_data)
