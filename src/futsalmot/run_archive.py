"""已闭合实验组的有界跨卷归档；先完整校验G副本，后精确释放D文件。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import uuid

import psutil

TERMINAL_STATES = frozenset({'COMPLETE', 'CALIBRATION_COMPLETE', 'FAIL_STOP', 'USER_STOP', 'LIMIT_STOP', 'BLOCKED'})
MANIFEST = 'ARCHIVE_MANIFEST.json'
RELOCATION = 'RELOCATION.json'
MAX_FILES = 200000
CHUNK_BYTES = 4 * 1024**2
RESERVE_BYTES = 100 * 1024**3


def _now():
    return datetime.now(timezone.utc).isoformat()


def _absolute(path):
    value = Path(path)
    if not value.is_absolute() or '..' in value.parts:
        raise ValueError('归档路径必须是无..的明确绝对路径')
    return value.absolute()


def _reparse(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _no_links(path):
    for part in reversed((path, *path.parents)):
        try:
            linked = _reparse(part)
        except FileNotFoundError:
            continue
        if linked:
            raise ValueError(f'拒绝symlink/junction/reparse路径：{part}')


def _identity(path):
    info = path.lstat()
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, maximum=256 * 1024**2):
    if path.stat().st_size > maximum:
        raise ValueError('归档JSON超出有界读取上限')
    return json.loads(path.read_bytes())


def _write(path, value, *, replace=False):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    target = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp') if replace else path
    with target.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def _tree(root):
    files, directories = {}, []
    pending = [root]
    while pending:
        current = pending.pop()
        _no_links(current)
        for entry in sorted(os.scandir(current), key=lambda item: item.name):
            path = Path(entry.path)
            if _reparse(path):
                raise ValueError(f'拒绝归档树内reparse：{path}')
            relative = path.relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                directories.append(relative)
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                files[relative] = _identity(path)
                if len(files) > MAX_FILES:
                    raise ValueError('归档组文件数超过有界上限')
            else:
                raise ValueError('只归档普通文件和真实目录')
    return files, sorted(directories)


def _process_rows():
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            yield process.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _matching_alive(pid, created):
    if not isinstance(pid, int) or not isinstance(created, (float, int)):
        return False
    try:
        return abs(psutil.Process(pid).create_time() - created) < .001
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as error:
        raise RuntimeError('无法验证登记PID是否已退出') from error


def _archive_owned_ids(research):
    # Windows uv的.venv启动器在真实Python退出前仍存活；只识别直属且参数完全相同的shim。
    owned = {os.getpid()}
    current = psutil.Process()
    try:
        parent = current.parent()
        if parent is not None:
            executable = os.path.normcase(os.path.abspath(parent.exe()))
            expected = os.path.normcase(str(research / '.venv' / 'Scripts' / 'python.exe'))
            arguments = current.cmdline()
            elapsed = current.create_time() - parent.create_time()
            if (executable == expected and len(arguments) > 1
                    and parent.cmdline()[1:] == arguments[1:] and 0 <= elapsed <= 60):
                owned.add(parent.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass  # 无法证明是自己的shim时仍交给普通写者guard拒绝。
    return owned


def _quiescent(source, research, allowed, *, owned_stub=False):
    for filename in (__file__, getattr(sys.modules.get('__main__'), '__file__', None)):
        if filename and Path(filename).absolute().is_relative_to(source):
            raise RuntimeError('归档调用者自身代码仍位于该组内，不能迁移')
    lock = research / '.research_gpu.lock'
    if lock.exists():
        binding = _json(lock, 4 * 1024**2)
        if os.path.normcase(os.path.abspath(str(binding.get('group', '')))) == os.path.normcase(str(source)):
            raise RuntimeError('该组GPU锁仍存在，不能归档')
    root_status = source / 'status.json'
    if not root_status.exists():
        if not owned_stub:
            raise ValueError('缺少组终态，不能归档')
    else:
        value = _json(root_status, 4 * 1024**2)
        if not (owned_stub and value.get('state') == 'ARCHIVED') and value.get('state') not in allowed:
            raise ValueError('组尚未达到允许的明确终态')
    inventory, _ = _tree(source)
    for relative in inventory:
        if PurePosixPath(relative).name != 'status.json':
            continue
        status_path = source / _relative(relative)
        _no_links(status_path)
        value = _json(status_path, 4 * 1024**2)
        if status_path == root_status and owned_stub and value.get('state') == 'ARCHIVED':
            continue
        if value.get('state') not in allowed:
            raise RuntimeError(f'子任务终态未闭合：{status_path}')
        if _matching_alive(value.get('pid'), value.get('create_time')):
            raise RuntimeError('终态登记的自有PID仍存活')
    needle = os.path.normcase(str(source)).replace('\\', '/')
    owned_processes = _archive_owned_ids(research)
    for row in _process_rows():
        if row.get('pid') in owned_processes:
            continue
        name = str(row.get('name') or '').lower()
        if not name.startswith(('python', 'uv', 'unrealeditor')):
            continue
        command = row.get('cmdline')
        if command is None:
            raise RuntimeError('无法读取潜在训练进程命令，不能确认无写者')
        normal = os.path.normcase(' '.join(command)).replace('\\', '/')
        if needle in normal:
            raise RuntimeError('仍有绑定该实验组的进程')


@contextmanager
def _writer(research, source, target):
    lock = research / '.run_archive.lock'
    value = {'schema': 'futsalmot_archive_writer_v1', 'pid': os.getpid(),
             'create_time': psutil.Process().create_time(), 'source': str(source), 'target': str(target)}
    _write(lock, value)
    identity = _identity(lock)
    try:
        yield
    finally:
        if lock.exists() and _identity(lock) == identity and _json(lock) == value:
            lock.unlink()


def _relative(text):
    value = PurePosixPath(text)
    if not text or value.is_absolute() or '..' in value.parts or '.' in value.parts or '\\' in text or ':' in text:
        raise ValueError('归档清单包含不安全相对路径')
    return Path(*value.parts)


def _verify_destination(target, receipt):
    actual, directories = _tree(target)
    actual.pop(MANIFEST, None)
    records = receipt['files']
    expected = {row['path'] for row in records}
    if len(expected) != len(records) or set(actual) != expected or directories != receipt['directories']:
        raise RuntimeError('G成品文件/目录集合与原清单不符')
    for row in records:
        path = target / _relative(row['path'])
        if row.get('size') != row['bytes'] or actual[row['path']]['bytes'] != row['bytes'] or _hash(path) != row['sha256']:
            raise RuntimeError(f'G副本size/SHA不符：{row["path"]}')


def _validate_receipt(receipt, source, target):
    if receipt.get('schema') != 'futsalmot_run_archive_v1' or receipt.get('status') != 'PASS':
        raise ValueError('既有目标不是可续接的PASS归档')
    if receipt.get('source') != str(source) or receipt.get('target') != str(target):
        raise ValueError('既有目标绑定另一来源')
    origin = receipt['source_directory_identity']
    current = _identity(source)
    if (origin['device'], origin['inode']) != (current['device'], current['inode']):
        raise ValueError('D来源目录身份已改变')
    for row in receipt['files']:
        _relative(row['path'])
    for directory in receipt['directories']:
        _relative(directory)


def _owned_stub(source, target, receipt_sha):
    path = source / RELOCATION
    if not path.exists():
        return None
    stub = _json(path, 4 * 1024**2)
    if (stub.get('schema') != 'futsalmot_run_relocation_v1' or stub.get('source') != str(source)
            or stub.get('target') != str(target) or stub.get('manifest_sha256') != receipt_sha):
        raise ValueError('D已有迁移入口与本次归档不一致')
    return stub


def _release(source, target, receipt, research, allowed):
    receipt_sha = _hash(target / MANIFEST)
    stub = _owned_stub(source, target, receipt_sha)
    _quiescent(source, research, allowed, owned_stub=stub is not None)
    actual, directories = _tree(source)
    records = {row['path']: row for row in receipt['files']}
    if stub is not None:
        actual.pop(RELOCATION, None)
        if (source / 'status.json').exists() and _json(source / 'status.json').get('state') == 'ARCHIVED':
            actual.pop('status.json', None)
    if not set(actual).issubset(records) or not set(directories).issubset(receipt['directories']):
        raise RuntimeError('D存在清单外新文件/目录，保留不释放')
    for relative, identity in actual.items():
        row = records[relative]
        if identity != row['source_identity'] or _hash(source / relative) != row['sha256']:
            raise RuntimeError(f'D来源内容/身份改变，保留：{relative}')
    if stub is None:
        # 没有owned stub时只允许完整来源；否则不能解释缺失文件。
        if set(actual) != set(records):
            raise RuntimeError('首次释放前D已有文件缺失')
        stub = {'schema': 'futsalmot_run_relocation_v1', 'status': 'COPY_VERIFIED_RELEASE_PENDING',
                'source': str(source), 'target': str(target), 'manifest': str(target / MANIFEST),
                'manifest_sha256': receipt_sha, 'original_terminal_state': receipt['original_terminal_state'],
                'files': len(records), 'bytes': receipt['total_bytes'], 'created_at_utc': _now()}
        _write(source / RELOCATION, stub)
    _quiescent(source, research, allowed, owned_stub=True)
    freed = sum(identity['bytes'] for identity in actual.values())
    # 只有清单内、刚核对过且身份未变的普通文件才可逐个释放；status最后处理。
    for relative in sorted(actual, key=lambda item: (item == 'status.json', item)):
        path = source / _relative(relative)
        _no_links(path)
        if _identity(path) != records[relative]['source_identity']:
            raise RuntimeError('释放期间文件身份变化；未释放文件保留')
        path.unlink()
    for relative in sorted(directories, key=lambda item: len(PurePosixPath(item).parts), reverse=True):
        path = source / _relative(relative)
        _no_links(path)
        path.rmdir()  # 非空会拒绝，绝不递归删除。
    status = {'schema': 'futsalmot_archived_status_v1', 'state': 'ARCHIVED',
              'original_terminal_state': receipt['original_terminal_state'],
              'target': str(target), 'manifest_sha256': receipt_sha, 'source_release_complete': True}
    _write(source / 'status.json', status, replace=(source / 'status.json').exists())
    stub = {**stub, 'status': 'PASS', 'state': 'ARCHIVED', 'destination': str(target),
            'manifest_path': str(target / MANIFEST), 'source_release_complete': True,
            'freed_bytes': freed, 'released_total_bytes': receipt['total_bytes'], 'completed_at_utc': _now()}
    _write(source / RELOCATION, stub, replace=True)
    remaining, empty_directories = _tree(source)
    if set(remaining) != {RELOCATION, 'status.json'} or empty_directories:
        raise RuntimeError('D释放未闭合；不冒称完成')
    return stub


def archive_group(source: Path, destination_root: Path, *, research_root: Path,
                  expected_terminal_states=TERMINAL_STATES):
    """归档一个闭合的直属实验组。G完整PASS后才逐文件释放D，不改原run/收据bytes。

    已发布PASS目标可幂等完成同来源D释放；未发布失败staging保留诊断证据。
    本工具不选择实验、不停止进程，也不迁移活动组。
    """
    source, destination_root, research = map(_absolute, (source, destination_root, research_root))
    allowed = frozenset(expected_terminal_states)
    if not allowed or not allowed <= TERMINAL_STATES:
        raise ValueError('调用者不能把RUNNING/未知状态作为可归档终态')
    for path in (research, source, destination_root):
        _no_links(path)
    if source.parent != research / 'runs' or not source.is_dir():
        raise ValueError('仅处理research_root/runs的精确直属组')
    if destination_root.is_relative_to(source) or source.is_relative_to(destination_root):
        raise ValueError('来源与归档目标不能嵌套')
    target = destination_root / source.name
    _no_links(target)
    destination_root.mkdir(parents=True, exist_ok=True)
    with _writer(research, source, target):
        if target.exists():
            receipt = _json(target / MANIFEST)
            _validate_receipt(receipt, source, target)
            _verify_destination(target, receipt)
            return _release(source, target, receipt, research, allowed)
        if (source / RELOCATION).exists():
            raise ValueError('D有迁移入口但G目标缺失，不能重新覆盖')
        _quiescent(source, research, allowed)
        inventory, directories = _tree(source)
        if MANIFEST in inventory or not inventory:
            raise ValueError('来源使用保留归档文件名或为空')
        total = sum(row['bytes'] for row in inventory.values())
        if shutil.disk_usage(destination_root).free < total + RESERVE_BYTES:
            raise RuntimeError('G归档新增容量不足，保留D，不减少证据')
        # 最多允许一个失败后的新staging；不覆盖、删除或伪装旧失败。
        previous = list(destination_root.glob('.' + source.name + '.archive-*'))
        if len(previous) >= 2:
            raise RuntimeError('同组已有两次未闭合staging；停止，需人工诊断')
        stage = destination_root / ('.' + source.name + '.archive-' + uuid.uuid4().hex)
        stage.mkdir()
        for relative in directories:
            (stage / _relative(relative)).mkdir()
        records = []
        for relative, identity in sorted(inventory.items()):
            origin, copied = source / _relative(relative), stage / _relative(relative)
            _no_links(origin)
            if _identity(origin) != identity:
                raise RuntimeError('拷贝前来源身份变化，保留D与staging')
            digest = hashlib.sha256()
            with origin.open('rb') as incoming, copied.open('xb') as outgoing:
                for chunk in iter(lambda: incoming.read(CHUNK_BYTES), b''):
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if _identity(origin) != identity or copied.stat().st_size != identity['bytes'] or _hash(copied) != digest.hexdigest():
                raise RuntimeError('拷贝期间来源改变或目标读回失败')
            records.append({'path': relative, 'size': identity['bytes'], 'bytes': identity['bytes'], 'sha256': digest.hexdigest(), 'source_identity': identity})
        after, after_dirs = _tree(source)
        if after != inventory or after_dirs != directories:
            raise RuntimeError('拷贝结束来源文件集合变化，保留D')
        for row in records:
            if _hash(source / _relative(row['path'])) != row['sha256']:
                raise RuntimeError('拷贝结束来源内容变化，保留D')
        _quiescent(source, research, allowed)
        receipt = {'schema': 'futsalmot_run_archive_v1', 'status': 'PASS', 'source': str(source),
                   'target': str(target), 'source_directory_identity': _identity(source),
                   'original_terminal_state': _json(source / 'status.json')['state'],
                   'files': records, 'directories': directories, 'total_bytes': total,
                   'source_bindings_bytes_preserved': True, 'copy_verified_at_utc': _now(),
                   'release_semantics': 'G完整size/SHA PASS后逐文件释放D；原路径绑定仅由RELOCATION补充'}
        _write(stage / MANIFEST, receipt)
        _verify_destination(stage, receipt)
        _no_links(target)
        if target.exists():
            raise FileExistsError('G最终目录已被占用，不覆盖')
        stage.rename(target)
        return _release(source, target, receipt, research, allowed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--destination-root', type=Path, required=True)
    parser.add_argument('--research-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive_group(args.source, args.destination_root, research_root=args.research_root), ensure_ascii=False))


if __name__ == '__main__':
    main()
