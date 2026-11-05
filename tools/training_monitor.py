"""训练原始日志的本地只读监控。只读小JSON/日志，不载入模型或媒体。"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import re
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
import webbrowser

import psutil

SERVICE = 'futsalmot-training-monitor'
VERSION = 9
REFRESH_SECONDS = 10
MAX_POINTS = 2000
MAX_SCAN_BYTES = 16 * 1024**2
MAX_LINE_BYTES = 2 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
FULL_HISTORY_POINTS = 50000
FULL_HISTORY_CHUNK = 1024**2
FULL_HISTORY_READERS = 2
FULL_HISTORY_BUDGET_SECONDS = 600
TERMINAL = {'COMPLETE', 'CALIBRATION_COMPLETE', 'FAIL_STOP', 'USER_STOP', 'LIMIT_STOP', 'BLOCKED'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def as_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def read_json(path, maximum=MAX_JSON_BYTES):
    if path.is_symlink() or path.is_junction():
        raise ValueError('拒绝重定向的监控文件')
    if path.stat().st_size > maximum:
        raise ValueError('JSON超过监控有界读取上限')
    with path.open('rb') as stream:
        return json.load(stream)


def _same_tree(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if '..' in path.parts or not path.is_relative_to(root):
        return False
    for parent in (path, *path.parents):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):
            return False
        if parent == root:
            break
    return True


@contextmanager
def open_shared_log(path):
    """Windows读日志时允许其它进程删除/重命名；每个短片后关闭句柄。"""
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(Path(path).absolute()), 0x80000000, 1 | 2 | 4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as stream:
        yield stream



def canonical_group_name(*, purpose, task, model, panel, strategy, seed, at=None, attempt=1):
    """未来单作业组名的纯格式化；不创建/改名目录或更改原run身份。"""
    if purpose not in ('main', 'cal50', 'cal100'):
        raise ValueError('purpose仅main/cal50/cal100')
    tasks = {'detect': 'bbox', 'bbox': 'bbox', 'segment': 'seg', 'seg': 'seg',
             'pose': 'pose', 'number': 'number', 'mot': 'mot'}
    task_slug = tasks.get(task) if isinstance(task, str) else None
    if task_slug is None:
        raise ValueError('未登记任务')
    def slug(value):
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError('model/panel须明确安全slug')
        result = value.lower()
        if re.fullmatch(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', result) is None:
            raise ValueError('model/panel禁止空白、路径或特殊字符')
        return result
    models, panels = slug(model), slug(panel)
    strategies = {'R': 'R', 'S': 'S', 'R->R': 'R2R', 'R→R': 'R2R', 'R2R': 'R2R',
                  'S->R': 'S2R', 'S→R': 'S2R', 'S2R': 'S2R', 'R+S': 'RS', 'RS': 'RS'}
    if not isinstance(strategy, str) or strategy not in strategies:
        raise ValueError('未登记策略')
    if type(seed) is not int or seed < 0 or type(attempt) is not int or not 1 <= attempt <= 99:
        raise ValueError('seed须非负整数、attempt须1..99，禁止bool')
    beijing = timezone(timedelta(hours=8))
    if at is None:
        at = datetime.now(beijing)
    if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError('at必须为明确时区的datetime')
    day = at.astimezone(beijing).strftime('%Y%m%d')
    return f'{purpose}_{task_slug}_{models}_{panels}_{strategies[strategy]}_seed{seed}_{day}_a{attempt:02d}'


def budget_summary(manifest):
    """从冻结协议显示计划预算；不改run，不以当前进度冒充终点预算。"""
    manifest = manifest if isinstance(manifest, dict) else {}
    p = manifest.get('protocol') or {}
    def integer(value): return value if type(value) is int and value >= 0 else None
    updates = integer(p.get('updates'))
    prefix = integer(p.get('prefix_updates')) or 0
    total = integer(p.get('trajectory_updates')) if prefix else updates
    if prefix and total is None and updates is not None: total = prefix + updates
    phases = p.get('phases')
    phase_updates = [integer(x.get('updates')) for x in phases if isinstance(x, dict)] if isinstance(phases, list) else []
    if any(x is None for x in phase_updates): phase_updates = []
    batch = integer(p.get('effective_batch_size')) or integer(p.get('effective_batch'))
    exposures = p.get('terminal_exposures')
    if not isinstance(exposures, dict) or any(integer(exposures.get(k)) is None for k in ('R', 'S')):
        exposures = None
        strategy = p.get('strategy')
        if updates is not None and batch:
            if prefix and strategy in ('S->R', 'R->R'):
                exposures = {'R': updates * batch + (prefix * batch if strategy == 'R->R' else 0), 'S': prefix * batch if strategy == 'S->R' else 0}
            elif strategy in ('R', 'R->R'): exposures = {'R': updates * batch, 'S': 0}
            elif strategy == 'S': exposures = {'R': 0, 'S': updates * batch}
            elif strategy == 'S->R' and len(phase_updates) == 2 and sum(phase_updates) == updates:
                exposures = {'R': phase_updates[1] * batch, 'S': phase_updates[0] * batch}
    n = integer(p.get('N_R')); full = integer(p.get('full_N_R')) or n
    pieces = [('总' + format(total, ',') + '步') if total is not None else '预算未登记']
    if prefix: pieces.append('继承' + str(prefix) + '+新增' + str(updates))
    elif len(phase_updates) > 1: pieces.append('阶段' + '+'.join(str(x) for x in phase_updates))
    if exposures is not None: pieces.append('R' + str(exposures['R']) + '/S' + str(exposures['S']))
    if n is not None: pieces.append('真实图' + str(n) + ('/' + str(full) if full and full != n else ''))
    return {'total_updates': total, 'new_updates': updates, 'prefix_updates': prefix, 'phase_updates': phase_updates,
            'planned_exposures': exposures, 'unique_real_images': n, 'full_real_images': full,
            'summary': ' · '.join(pieces)}


def display_label(manifest, run_name, group_name=''):
    """只读清单的统一人读名称；缺失/未知来源不由组名猜测。"""
    unknown = '未登记'
    manifest = manifest if isinstance(manifest, dict) else {}
    protocol = manifest.get('protocol')
    protocol = protocol if isinstance(protocol, dict) else {}
    names = [str(run_name), str(manifest.get('run_id') or ''), str(group_name)]
    schemas = [x for x in (protocol.get('schema_version'), manifest.get('schema_version')) if isinstance(x, str) and x]
    calibration = 'bbox_R_calibration_run_v1' in schemas or 'bbox_calibration_manifest_v1' in schemas
    formal = any(v in schemas for v in ('formal_bbox_five_strategy_v1', 'bbox_fixed_budget_mix_v1', 'bbox_short_pair_v1', 'bbox_budget1500_v1', 'bbox_low_label25_v1', 'bbox_label_curve_v1'))
    short_prefix = 'bbox_short_prefix_branch_v1' in schemas
    legacy_calibration = next((re.fullmatch(r'R(50|100)_seed[0-9]+', name) for name in names
                               if re.fullmatch(r'R(50|100)_seed[0-9]+', name)), None)
    legacy_formal = any(re.fullmatch(r'(?:R|S|R_to_R|S_to_R|R_plus_S)_seed[0-9]+', name) for name in names)
    canonical_purpose = next((m.group(1) for name in names
        if (m := re.match(r'^(main|cal50|cal100)_(?:bbox|seg|pose|number|mot)_', name))), None)
    epochs = protocol.get('epochs')
    if type(epochs) is not int or epochs not in (50, 100):
        epochs = int(legacy_calibration.group(1)) if legacy_calibration else None
    if short_prefix:
        purpose = '开发诊断'
    elif formal:
        purpose = '正式'
    elif calibration:
        purpose = '校准R' + str(epochs) if epochs else unknown
    elif not schemas and (legacy_formal or canonical_purpose == 'main'):
        purpose = '正式'
    elif not schemas and (legacy_calibration or canonical_purpose in ('cal50', 'cal100')):
        purpose = '校准R' + str(epochs or int(canonical_purpose[3:]))
    else:
        purpose = unknown
    tasks = {'detect': 'BBox', 'bbox': 'BBox', 'segment': 'Seg', 'seg': 'Seg',
             'pose': 'Pose', 'number': '号码', 'mot': 'MOT'}
    declared_task = protocol.get('task')
    task = tasks.get(declared_task, unknown) if isinstance(declared_task, str) else unknown
    if not declared_task:
        task = next((tasks[m.group(1)] for name in names
            if (m := re.match(r'^(?:(?:main|cal50|cal100)_)?(bbox|seg|pose|number|mot)_', name))), unknown)
    declared_model = protocol.get('model')
    model_match = re.fullmatch(r'yolo11([nsmlx])(?:[-_](seg|pose))?', declared_model.lower()) if isinstance(declared_model, str) else None
    if model_match is None and not declared_model:
        model_match = next((m for name in names if (m := re.search(r'(?:^|_)yolo11([nsmlx])(?:[-_](seg|pose))?(?:_|$)', name.lower()))), None)
        if model_match is None:
            model_match = next((m for name in names if (m := re.match(r'^bbox_([nsmlx])(?:_|$)', name.lower()))), None)
    model = ('YOLO11' + model_match.group(1) + ('-' + model_match.group(2) if model_match.lastindex and model_match.lastindex >= 2 and model_match.group(2) else '')) if model_match else unknown
    bindings = manifest.get('source_bindings')
    bindings = bindings if isinstance(bindings, dict) else {}
    real_panel = bindings.get('real_panel')
    real_panel = real_panel if isinstance(real_panel, dict) else {}
    index = real_panel.get('index')
    index = index if isinstance(index, dict) else {}
    path = index.get('path')
    panel = unknown
    if isinstance(path, str):
        parts = path.replace('\\', '/').rstrip('/').lower().split('/')
        if len(parts) >= 2 and parts[-1] == 'dataset.json':
            panel = {'soccernet': 'SoccerNet', 'sportsmot': 'SportsMOT'}.get(parts[-2], unknown)
    if panel == unknown:
        declared_source = real_panel.get('source') or index.get('source')
        panel = {'soccernet_tracking': 'SoccerNet', 'soccernet': 'SoccerNet',
                 'sportsmot_football': 'SportsMOT', 'sportsmot': 'SportsMOT'}.get(str(declared_source).lower(), unknown)
    declared_strategy = protocol.get('strategy')
    strategies = {'R': 'R', 'S': 'S', 'R->R': 'R→R', 'R→R': 'R→R', 'R2R': 'R→R',
                  'S->R': 'S→R', 'S→R': 'S→R', 'S2R': 'S→R', 'R+S': 'R+S', 'RS': 'R+S'}
    strategy = strategies.get(declared_strategy, unknown) if isinstance(declared_strategy, str) else unknown
    if not declared_strategy:
        old = next((m.group(1) for name in names if (m := re.match(r'^(R_to_R|S_to_R|R_plus_S|R|S)(?:50|100)?_seed[0-9]+$', name))), None)
        strategy = {'R_to_R': 'R→R', 'S_to_R': 'S→R', 'R_plus_S': 'R+S', 'R': 'R', 'S': 'S'}.get(old, unknown)
        if strategy == unknown:
            strategy = next((strategies[m.group(1)] for name in names
                if (m := re.search(r'_(R2R|S2R|RS|R|S)_seed[0-9]+(?:_|$)', name))), unknown)
    if short_prefix:
        strategy = {'A': 'S3000→R', 'B': 'R3000→R'}.get(protocol.get('arm'), unknown)
    seed = protocol.get('seed')
    if type(seed) is not int or seed < 0:
        seed = next((int(m.group(1)) for name in names if (m := re.search(r'(?:^|_)seed([0-9]+)(?:_|$)', name))), None)
    seed_label = f'seed{seed}' if seed is not None else unknown
    return '｜'.join((purpose, task, model, panel, strategy, seed_label))

def comparison_signature(manifest):
    protocol = manifest.get('protocol') or {}
    panel = (manifest.get('source_bindings') or {}).get('real_panel')
    task, metric = protocol.get('task'), protocol.get('main_metric')
    if not task or not metric or not isinstance(panel, dict) or not panel:
        return None
    raw = json.dumps({'task': task, 'main_metric': metric, 'real_panel': panel},
                     sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def tail_lines(path, max_bytes=32768, maximum_lines=40):
    try:
        with open_shared_log(path) as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            stream.seek(start)
            data = stream.read(max_bytes)
        if start:
            data = data.partition(b'\n')[2]
        return data.decode('utf-8', errors='replace').splitlines()[-maximum_lines:]
    except FileNotFoundError:
        return []


def scalar_point(row):
    update = as_number(row.get('update'))
    if update is None or update < 0:
        return None
    components = row.get('loss_components') or {}
    if not isinstance(components, dict):
        components = {}
    lr = row.get('lr', row.get('learning_rates'))
    if isinstance(lr, dict):
        rates = {str(key): as_number(value) for key, value in lr.items()}
        first = next((x for x in rates.values() if x is not None), None)
    elif isinstance(lr, (list, tuple)):
        rates = [as_number(x) for x in lr]
        first = next((x for x in rates if x is not None), None)
    else:
        first = as_number(lr)
        rates = [first] if first is not None else []
    return {'update': int(update), 'loss': as_number(row.get('loss')),
            'loss_box': as_number(components.get('box')),
            'loss_cls': as_number(components.get('cls')),
            'loss_dfl': as_number(components.get('dfl')),
            'gradient_norm': as_number(row.get('gradient_norm', row.get('gradient_norm_before_clip'))),
            'learning_rate': first, 'learning_rates': rates,
            'wall_seconds': as_number(row.get('wall_seconds')),
            'R_exposures': as_number(row.get('R_exposures')),
            'S_exposures': as_number(row.get('S_exposures')),
            'phase': row.get('phase') if isinstance(row.get('phase'), str) else None,
            'phase_update': as_number(row.get('phase_update', row.get('phase_updates')))}


class IncrementalLog:
    """最多保存2000个纯标量点；不保存UID/geometry/血缘大对象或文件句柄。"""
    def __init__(self, extractor=scalar_point, max_scan=MAX_SCAN_BYTES, max_points=MAX_POINTS):
        self.extractor = extractor
        self.max_scan = max_scan
        self.max_points = max_points
        self.path = None
        self.identity = None
        self.offset = 0
        self.pending = b''
        self.prefix = None
        self.points = deque(maxlen=max_points)
        self.records = 0
        self.history_truncated = False
        self.errors = deque(maxlen=3)
        self.discard_line = False
        self.bytes_last_read = 0

    def reset(self, path, identity):
        self.path, self.identity = str(path), identity
        self.offset, self.pending, self.prefix = 0, b'', None
        self.points.clear()
        self.records = 0
        self.history_truncated = False
        self.discard_line = False
        self.errors.clear()

    def update(self, path, *, allow_missing_initial=False):
        self.bytes_last_read = 0
        try:
            if path.is_symlink() or path.is_junction():
                raise ValueError('日志是重定向文件')
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            with open_shared_log(path) as stream:
                prefix = stream.read(min(128, info.st_size))
                changed = self.path != str(path) or self.identity != identity or info.st_size < self.offset
                # 前缀只在已经读过128B后固定，普通短日志增长不当作重写。
                if self.prefix is not None and len(self.prefix) == 128 and prefix != self.prefix:
                    changed = True
                if changed:
                    self.reset(path, identity)
                if self.path is None:
                    self.reset(path, identity)
                if len(prefix) == 128:
                    self.prefix = prefix
                start = self.offset
                skip_first = False
                if start == 0 and info.st_size > self.max_scan:
                    start = info.st_size - self.max_scan
                    self.history_truncated = True
                    skip_first = True
                stream.seek(start)
                data = stream.read(self.max_scan)
                self.bytes_last_read = len(data)
                self.offset = stream.tell()
            if skip_first:
                data = data.partition(b'\n')[2]
                self.pending = b''
            data = self.pending + data
            lines = data.split(b'\n')
            self.pending = lines.pop()
            for raw in lines:
                if self.discard_line:
                    self.discard_line = False
                    continue
                if not raw.strip():
                    continue
                if len(raw) > MAX_LINE_BYTES:
                    self.errors.append('一条日志超过2MiB，监控跳过，原文件未修改')
                    continue
                try:
                    row = json.loads(raw)
                    point = self.extractor(row)
                    if point is not None:
                        self.records += 1
                        self.points.append(point)
                except (ValueError, TypeError, AttributeError) as error:
                    self.errors.append('跳过完整但无法解析的日志行：' + str(error)[:100])
            if len(self.pending) > MAX_LINE_BYTES:
                self.pending = b''
                self.discard_line = True
                self.errors.append('未结束日志行超过2MiB，等待下一换行')
        except FileNotFoundError:
            # 0步公开起点评价前尚未创建steps是正常等待；已见过文件后丢失仍报告。
            if not (allow_missing_initial and self.path is None and self.offset == 0 and not self.points):
                self.errors.append('日志文件暂不存在：' + path.name + '；请核对当前实验状态')
        except (OSError, ValueError) as error:
            self.errors.append('日志读取失败：' + path.name + '：' + str(error)[:160])
        return list(self.points)


class FullHistoryReader:
    """显式请求后从头读纯标量；短片CPU/IO和总追赶时间均有限。"""
    def __init__(self, path_provider, *, max_points=FULL_HISTORY_POINTS,
                 chunk_bytes=FULL_HISTORY_CHUNK, budget_seconds=FULL_HISTORY_BUDGET_SECONDS,
                 slice_seconds=.05):
        self.path_provider = path_provider
        self.max_points, self.chunk_bytes = max_points, chunk_bytes
        self.budget_seconds, self.slice_seconds = budget_seconds, slice_seconds
        self.lock = threading.RLock()
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.thread = None
        self.last_requested = time.monotonic()
        self.points = deque(maxlen=max_points)
        self.queue = deque()
        self.pending = b''
        self.path = None
        self.identity = None
        self.prefix = None
        self.offset = self.total_bytes = self.parsed_records = 0
        self.generation = 0
        self.state, self.reason = 'loading', None
        self.catchup_started = time.monotonic()
        self.caught_up = False
        self.fatal = False
        self.invalid_records = 0
        self.discard_line = False
        self.bytes_last_read = 0

    def _reset(self, path, identity):
        self.path, self.identity = str(path), identity
        self.prefix = None
        self.points.clear(); self.queue.clear(); self.pending = b''
        self.offset = self.total_bytes = self.parsed_records = 0
        self.generation += 1
        self.state, self.reason = 'loading', None
        self.catchup_started = time.monotonic()
        self.caught_up = self.fatal = self.discard_line = False
        self.invalid_records = 0

    def request(self):
        with self.lock:
            self.last_requested = time.monotonic()
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name='training-full-history', daemon=True)
                self.thread.start()
        self.wake.set()
        return self.snapshot()

    def _loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                idle = time.monotonic() - self.last_requested > 60
            if idle:
                self.wake.wait(10); self.wake.clear()
                continue
            self.advance()
            with self.lock:
                loading = self.state == 'loading' and not self.fatal
                fatal = self.fatal
            if fatal:
                return
            if loading:
                self.stop_event.wait(.2)
            else:
                self.wake.wait(10); self.wake.clear()

    def advance(self):
        """测试可单步驱动；真实后台每片最多1MiB并每50ms让出解析。"""
        began = time.monotonic()
        self.bytes_last_read = 0
        try:
            path = Path(self.path_provider())
            if path.is_symlink() or path.is_junction():
                raise ValueError('全程日志为重定向文件')
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            with open_shared_log(path) as stream:
                prefix = stream.read(min(128, info.st_size))
                with self.lock:
                    changed = (self.path != str(path) or self.identity != identity or info.st_size < self.offset
                               or (self.prefix is not None and len(self.prefix) == 128 and prefix != self.prefix))
                    if changed:
                        self._reset(path, identity)
                    if len(prefix) == 128:
                        self.prefix = prefix
                    self.total_bytes = info.st_size
                    if self.caught_up and info.st_size > self.offset:
                        self.catchup_started = time.monotonic(); self.caught_up = False
                    offset, have_queue = self.offset, bool(self.queue)
                if not have_queue:
                    stream.seek(offset)
                    data = stream.read(self.chunk_bytes)
                    with self.lock:
                        self.bytes_last_read = len(data)
                        self.offset = stream.tell()
                        self.total_bytes = max(self.total_bytes, self.offset)
                        data = self.pending + data
                        lines = data.split(b'\n')
                        self.pending = lines.pop()
                        self.queue.extend(lines)
                        if len(self.pending) > MAX_LINE_BYTES:
                            self.pending = b''; self.discard_line = True; self.invalid_records += 1
            # 解析时不持有文件句柄，避免Windows归集删除/改名冲突。
            while not self.stop_event.is_set():
                with self.lock:
                    if not self.queue:
                        break
                    raw = self.queue.popleft()
                    discard = self.discard_line
                    if discard:
                        self.discard_line = False
                if not discard and raw.strip():
                    point = None
                    try:
                        if len(raw) > MAX_LINE_BYTES:
                            raise ValueError('单条完整日志超过2MiB')
                        point = scalar_point(json.loads(raw))
                    except (ValueError, TypeError, AttributeError):
                        with self.lock:
                            self.invalid_records += 1
                    if point is not None:
                        with self.lock:
                            self.points.append(point); self.parsed_records += 1
                if time.monotonic() - began >= self.slice_seconds:
                    break
            with self.lock:
                if self.offset >= self.total_bytes and not self.queue:
                    self.caught_up = True
                    self.reason = None
                    if self.parsed_records > self.max_points:
                        self.reason = f'日志超过{self.max_points}个标量点，仅保留最近窗口；原文件完整保留'
                    if self.invalid_records:
                        self.reason = (self.reason + '；' if self.reason else '') + f'{self.invalid_records}条完整记录无法解析或超限，缺口未补点'
                    self.state = 'limited' if self.reason else 'ready'
                elif time.monotonic() - self.catchup_started > self.budget_seconds:
                    self.state, self.fatal = 'limited', True
                    self.reason = f'全程读取追赶达到{self.budget_seconds}秒上限，当前仅为部分历史；未影响训练'
                else:
                    self.state = 'loading'
        except FileNotFoundError:
            with self.lock:
                if time.monotonic() - self.catchup_started > self.budget_seconds:
                    self.state, self.fatal = 'limited', True
                    self.reason = '等待步日志达到有界上限；当前没有可读全程历史，未影响训练'
                else:
                    self.state = 'loading'
                    self.reason = '步日志尚未产生或正在迁移，等待下一片刷新'
        except (OSError, ValueError, KeyError, TypeError) as error:
            with self.lock:
                self.state, self.fatal = 'error', True
                self.reason = '全程历史只读失败：' + str(error)[:180]

    def snapshot(self):
        with self.lock:
            points = list(self.points)
            return {'state': self.state, 'points': points, 'parsed_records': self.parsed_records,
                    'bytes_read': self.offset, 'total_bytes': self.total_bytes,
                    'first_update': points[0]['update'] if points else None,
                    'last_update': points[-1]['update'] if points else None,
                    'progress_fraction': min(1., self.offset / self.total_bytes) if self.total_bytes else (1. if self.caught_up else 0.),
                    'limited_reason': self.reason, 'history_generation': self.generation,
                    'pending_bytes': len(self.pending), 'max_points': self.max_points}

    def close(self):
        self.stop_event.set(); self.wake.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        return thread is None or not thread.is_alive()


def unavailable_history(state, reason):
    return {'state': state, 'points': [], 'parsed_records': 0, 'bytes_read': 0, 'total_bytes': None,
            'first_update': None, 'last_update': None, 'progress_fraction': None,
            'limited_reason': reason, 'history_generation': 0, 'pending_bytes': 0,
            'max_points': FULL_HISTORY_POINTS}


class FullHistoryPool:
    def __init__(self, path_provider, *, maximum=FULL_HISTORY_READERS):
        self.path_provider, self.maximum = path_provider, maximum
        self.readers = {}
        self.lock = threading.RLock()
        self.closed = False

    def request(self, identifier):
        with self.lock:
            if self.closed:
                return unavailable_history('error', '监控服务已关闭')
            reader = self.readers.get(identifier)
            if reader is None:
                if len(self.readers) >= self.maximum:
                    old_id, old = min(self.readers.items(), key=lambda item: item[1].last_requested)
                    if not old.close():
                        return unavailable_history('limited', '旧后台读取正在关闭，等待下次刷新')
                    del self.readers[old_id]
                reader = FullHistoryReader(lambda: self.path_provider(identifier))
                self.readers[identifier] = reader
            return reader.request()

    def close(self):
        with self.lock:
            self.closed = True
            for reader in self.readers.values():
                reader.close()


def _process_identity(value):
    pid, created = value.get('pid'), value.get('create_time')
    if not isinstance(pid, int) or as_number(created) is None:
        return {'pid': pid, 'alive': None, 'identity_verified': False}
    try:
        process = psutil.Process(pid)
        matches = abs(process.create_time() - created) < .001
        return {'pid': pid, 'alive': matches and process.is_running(), 'identity_verified': matches}
    except psutil.NoSuchProcess:
        return {'pid': pid, 'alive': False, 'identity_verified': False}
    except psutil.AccessDenied:
        return {'pid': pid, 'alive': None, 'identity_verified': False}



def confirmation_snapshot(root, archive_root):
    """只读固定suite的小控制/有界stdout；不打开GT、checkpoint或预测目录。"""
    root, archive_root = Path(root).absolute(), Path(archive_root).absolute()
    pointer = root / '.bbox_confirmation_monitor.json'
    if not pointer.exists():
        return {'state': 'NOT_REGISTERED', 'jobs': [], 'completed': 0, 'total': 48, 'errors': []}
    errors = []
    try:
        binding = read_json(pointer, 65536)
        control, output = Path(binding['control_group']), Path(binding['output_group'])
        if control.parent != root/'runs' or output != archive_root/'runs'/control.name or not control.name.startswith('bbox_confirmation_suite_'):
            raise ValueError('确认监视路径不是固定研究suite')
        if not _same_tree(control, root) or not _same_tree(output, archive_root):
            raise ValueError('确认监视拒绝重定向路径')
        suite_path = control/'suite_request.json'
        if hashlib.sha256(suite_path.read_bytes()).hexdigest() != binding['suite_sha256']:
            raise ValueError('确认suite摘要变化')
        suite = read_json(suite_path)
        if suite.get('schema_version') != 'bbox_confirmation_suite_v1' or len(suite.get('jobs', [])) != 48:
            raise ValueError('不是固定48项评价')
        spec = read_json(Path(suite['spec_reference']['path']))
        models = {m['model_id']: m for m in spec['models']}
        rows, seconds = [], []
        for item in suite['jobs']:
            key = item['job_id']
            if Path(key).name != key or key in ('.', '..'):
                raise ValueError('无效job token')
            job, out = control/'jobs'/key, output/'jobs'/key
            if not _same_tree(job, root) or not _same_tree(out, archive_root):
                raise ValueError('确认job路径重定向')
            model = models[item['model_id']]
            row = {**item, 'seed': model['seed'], 'real_fraction': model['real_fraction'], 'strategy': model['strategy'], 'state': 'PENDING', 'images': 0, 'total_images': 750, 'metrics': None, 'elapsed_seconds': None, 'deadline_utc': None}
            result = read_json(out/'result.json') if (out/'result.json').exists() else None
            life = read_json(job/'lifetime_receipt.json') if (job/'lifetime_receipt.json').exists() else None
            if result:
                row['images'] = result.get('summary', {}).get('images', 0)
                row['state'] = result['state']
                closed = life and life.get('state') == 'VERIFIED_EXIT' and life.get('owned_still_alive') == [] and life.get('GPU_lock_retired') is True and (out/'control/CONTROL_MANIFEST.json').exists()
                if row['state'] == 'COMPLETE':
                    row['state'] = 'COMPLETE' if closed else 'FINALIZING'
                    row['metrics'] = result.get('summary', {}).get('main_metrics')
                row['error'] = result.get('error', {}).get('message')
            elif (job/'runtime_limit.json').exists():
                row['state'] = 'INITIALIZING'
                for line in tail_lines(out/'stdout.log', max_bytes=65536, maximum_lines=40):
                    try: event = json.loads(line)
                    except ValueError: continue
                    if event.get('event') == 'eval_progress':
                        row.update(state='INFERENCE', images=event['images'])
                    elif event.get('event') == 'evaluation_complete':
                        row.update(state='FINALIZING', images=event['images'])
                runtime = read_json(job/'runtime_limit.json')
                row['elapsed_seconds'] = max(0, time.time()-runtime['dispatch_started_unix'])
                row['deadline_utc'] = datetime.fromtimestamp(runtime['deadline_unix'], timezone.utc).isoformat()
                if row['elapsed_seconds'] > 900:
                    row['state'] = 'WAITING_EXIT'
                dispatch_path = job/'dispatch_wrapper.json'
                if dispatch_path.exists():
                    row['process'] = _process_identity(read_json(dispatch_path))
            if life:
                row['elapsed_seconds'] = life.get('full_process_seconds', life.get('observed_seconds'))
                if row['state'] == 'COMPLETE' and as_number(row['elapsed_seconds']) is not None:
                    seconds.append(row['elapsed_seconds'])
            rows.append(row)
        completed = sum(r['state'] == 'COMPLETE' for r in rows)
        active = next((r for r in rows if r['state'] not in ('PENDING', 'COMPLETE')), None)
        dispatch = read_json(control/'suite_dispatch.json') if (control/'suite_dispatch.json').exists() else None
        terminal = read_json(control/'suite_result.json') if (control/'suite_result.json').exists() else None
        state = terminal['state'] if terminal else 'RUNNING' if dispatch else 'READY'
        if (control/'STOP.json').exists() or (output/'STOP.json').exists():
            state = 'STOP_REQUESTED' if not terminal else state
        parent = _process_identity(dispatch) if dispatch else None
        if dispatch and not terminal and parent['alive'] is False:
            state = 'ATTENTION_PARENT_EXITED'
        total_images = sum(min(750, max(0, r['images'])) for r in rows)
        return {'state': state, 'jobs': rows, 'completed': completed, 'total': 48, 'completed_models': sum(all(r['state']=='COMPLETE' for r in rows if r['model_id']==m) for m in models), 'images': total_images, 'total_images': 36000, 'active': active, 'eta_seconds': (48-completed)*statistics.mean(seconds) if seconds else None, 'measured_completed_job_seconds': seconds, 'suite_process': parent, 'control_group': str(control), 'output_group': str(output), 'error': terminal.get('error') if terminal else None, 'errors': errors, 'progress_note': '当前图数为原生stdout最近进度，最多约200图更新一次；完成须真实退出并封存。', 'generated_at_utc': utc_now()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'state': 'UNAVAILABLE', 'jobs': [], 'completed': 0, 'total': 48, 'errors': [str(exc)[:240]]}


class Monitor:
    def __init__(self, research_root, archive_root, *, refresh_seconds=REFRESH_SECONDS):
        self.root = Path(research_root).absolute()
        self.archive_root = Path(archive_root).absolute()
        self.refresh_seconds = refresh_seconds
        self.steps = {}
        self.checkpoints = {}
        self.lock = threading.RLock()
        self.cached = {}
        self.system_cached = None
        self.system_at = 0.0
        self.relocation_proofs = {}
        self.histories = FullHistoryPool(self._history_path)
        psutil.cpu_percent(interval=None)

    def _relocation_target(self, group, rel):
        """发布后的G小manifest可证时立刻转读G，避免占用D释放窗口。"""
        if rel.get('schema') != 'futsalmot_run_relocation_v1' or rel.get('status') not in ('PASS', 'COPY_VERIFIED_RELEASE_PENDING'):
            raise ValueError('迁移状态尚无可验证的G发布')
        target = Path(rel['target']).absolute()
        expected_parent = (self.archive_root / 'runs').absolute()
        norm = lambda value: os.path.normcase(str(Path(value).absolute()))
        if (norm(rel.get('source', '')) != norm(group) or target.parent != expected_parent
                or target.name != group.name or not _same_tree(target, expected_parent)):
            raise ValueError('迁移source/target越过授权组路径')
        expected_sha = rel.get('manifest_sha256')
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(x not in '0123456789abcdef' for x in expected_sha):
            raise ValueError('迁移缺少明确小manifest SHA')
        manifest_path = target / 'ARCHIVE_MANIFEST.json'
        info = manifest_path.stat()
        if info.st_size > MAX_JSON_BYTES or manifest_path.is_symlink() or manifest_path.is_junction():
            raise ValueError('归集小manifest超限或重定向')
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, expected_sha)
        key = str(manifest_path)
        if self.relocation_proofs.get(key) != identity:
            raw = manifest_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                raise ValueError('归集小manifest SHA不符')
            receipt = json.loads(raw)
            if (receipt.get('schema') != 'futsalmot_run_archive_v1' or receipt.get('status') != 'PASS'
                    or norm(receipt.get('source', '')) != norm(group) or norm(receipt.get('target', '')) != norm(target)):
                raise ValueError('归集小manifest未绑定本source/target/PASS')
            self.relocation_proofs[key] = identity
        complete = rel.get('status') == 'PASS' and rel.get('source_release_complete') is True
        return target, not complete, complete

    def _direct_location(self, control_group, output_group=None):
        """双端同bytes marker只证明直接写G布局，不冒充归档或实验完成。"""
        control = Path(control_group).absolute()
        expected_control = (self.root / 'runs').absolute()
        if control.parent != expected_control or not _same_tree(control, expected_control):
            raise ValueError('direct_g控制组不在研究runs直属目录')
        marker = control / 'RUN_LOCATION.json'
        if not _same_tree(marker, control):
            raise ValueError('direct_g控制marker为重定向路径')
        with open_shared_log(marker) as stream:
            control_raw = stream.read(65537)
        if len(control_raw) > 65536:
            raise ValueError('direct_g marker超过64KiB')
        location = json.loads(control_raw)
        if (location.get('schema') != 'futsalmot_direct_output_v1' or location.get('version') != 1
                or location.get('storage_mode') != 'direct_g'):
            raise ValueError('未知直接输出marker合同')
        norm = lambda value: os.path.normcase(str(Path(value).absolute()))
        output = Path(location['output_group']).absolute()
        expected_output = (self.archive_root / 'runs' / control.name).absolute()
        if (norm(location.get('control_group', '')) != norm(control) or norm(output) != norm(expected_output)
                or not _same_tree(output, self.archive_root / 'runs')):
            raise ValueError('direct_g双端路径未绑定授权同名直属组')
        if output_group is not None and norm(output_group) != norm(output):
            raise ValueError('direct_g输出marker与发现的G组不一致')
        peer = output / 'RUN_LOCATION.json'
        if not _same_tree(peer, output):
            raise ValueError('direct_g输出marker为重定向路径')
        with open_shared_log(peer) as stream:
            output_raw = stream.read(65537)
        if output_raw != control_raw:
            raise ValueError('direct_g双端marker bytes不一致')
        return output, control

    def inventory(self):
        groups, errors = {}, []
        source = self.root / 'runs'
        try:
            children = sorted(source.iterdir()) if source.is_dir() else []
        except OSError as error:
            children = []
            errors.append('实验入口读取失败：' + str(error))
        for group in children[:200]:
            if not group.is_dir() or group.name.startswith('.') or not _same_tree(group, source):
                continue
            actual, archived, archive_pending, release_complete = group, False, False, False
            relocation = group / 'RELOCATION.json'
            direct = group / 'RUN_LOCATION.json'
            control, storage_mode = group, 'local_d'
            if direct.exists():
                try:
                    if relocation.exists():
                        raise ValueError('direct_g与归档迁移marker不能混用')
                    actual, control = self._direct_location(group)
                    storage_mode = 'direct_g'
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    errors.append(group.name + '：直接输出绑定暂不可用：' + str(error)[:120])
                    groups[group.name] = None
                    continue
            elif relocation.exists():
                try:
                    rel = read_json(relocation)
                    target, archive_pending, release_complete = self._relocation_target(group, rel)
                    actual, archived = target, True
                    control, storage_mode = target, 'archived_g'
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(group.name + '：迁移入口暂不可读：' + str(error)[:100])
            groups[group.name] = (actual, archived, archive_pending, release_complete, control, storage_mode)
        try:
            archive_children = sorted((self.archive_root / 'runs').iterdir()) if (self.archive_root / 'runs').is_dir() else []
        except OSError:
            archive_children = []
        for group in archive_children[:200]:
            if group.name in groups or group.name.startswith('.') or not group.is_dir() or not _same_tree(group, self.archive_root / 'runs'):
                continue
            if (group / 'RUN_LOCATION.json').is_file():
                try:
                    location = read_json(group / 'RUN_LOCATION.json', maximum=65536)
                    output, control = self._direct_location(Path(location['control_group']), group)
                    groups[group.name] = (output, False, False, False, control, 'direct_g')
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    errors.append(group.name + '：G直接输出绑定暂不可用：' + str(error)[:120])
            elif (group / 'ARCHIVE_MANIFEST.json').is_file():
                groups[group.name] = (group, True, False, None, group, 'archived_g')
        entries = []
        for group_name, layout in groups.items():
            if layout is None:
                continue
            group, archived, archive_pending, release_complete, control, storage_mode = layout
            try:
                group_status = read_json(control / 'status.json') if (control / 'status.json').exists() else {}
                candidates = sorted(group.iterdir())
            except (OSError, ValueError) as error:
                errors.append(group_name + '：组入口暂不可读：' + str(error)[:100])
                continue
            for directory in candidates[:50]:
                if not directory.is_dir() or not (directory / 'run.json').is_file() or not _same_tree(directory, group):
                    continue
                try:
                    manifest = read_json(directory / 'run.json')
                    state = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
                    protocol = manifest.get('protocol', {})
                    model, strategy, seed = protocol.get('model'), protocol.get('strategy'), protocol.get('seed')
                    entry = {'id': hashlib.sha256((group_name + '/' + directory.name).encode()).hexdigest()[:20],
                             'label': display_label(manifest, directory.name, group_name),
                             'budget': budget_summary(manifest),
                             'menu_label': display_label(manifest, directory.name, group_name) + '｜' + budget_summary(manifest)['summary'],
                             'state': state.get('state', 'UNKNOWN'), 'group_id': group_name,
                             'model': model, 'strategy': strategy, 'seed': seed,
                             'active': state.get('state') == 'RUNNING' and not archived,
                             'archived': archived, 'archive_pending': archive_pending, 'source_release_complete': release_complete,
                             'storage_mode': storage_mode, 'control_group_path': str(control), 'output_group_path': str(group),
                             '_path': directory, '_group': group, '_control_group': control,
                             '_manifest': manifest, '_status': state, '_group_status': group_status}
                    entries.append(entry)
                except (OSError, ValueError, AttributeError) as error:
                    errors.append(directory.name + '：运行清单暂不可读：' + str(error)[:120])
        return entries, errors

    def _default(self, entries):
        try:
            binding = read_json(self.root / '.research_gpu.lock')
            locked = Path(binding['group']).name
            if _process_identity(binding)['alive']:
                matched = [entry for entry in entries if entry['active'] and entry['group_id'] == locked]
                if matched:
                    return matched[-1]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        active = [entry for entry in entries if entry['active']]
        return (active or entries)[-1] if entries else None

    def _validation(self, directory, protocol=None):
        rows, errors = [], []
        schedule = (protocol or {}).get('evaluation_schedule') or []
        planned = {item.get('tag'): item for item in schedule if isinstance(item, dict) and isinstance(item.get('tag'), str)}
        evaluation = directory / 'evaluation'
        try:
            children = sorted(evaluation.iterdir()) if evaluation.is_dir() else []
        except OSError as error:
            return [], ['验证记录暂不可读：' + str(error)]
        for folder in children[:500]:
            summary_path = folder / 'summary.json'
            if not folder.name.startswith('step') or not summary_path.is_file() or not _same_tree(summary_path, evaluation):
                continue
            try:
                summary = read_json(summary_path)
                metrics = summary.get('main_metrics') or {}
                tag = summary.get('tag', folder.name)
                item = planned.get(tag, {})
                global_update = summary.get('global_update', item.get('global_update', item.get('update', summary.get('update'))))
                rows.append({'update': global_update, 'tag': tag,
                             'phase': summary.get('phase', item.get('phase')),
                             'phase_update': summary.get('phase_update', summary.get('update')),
                             'AP50_95': as_number(metrics.get('AP50_95')), 'AP50': as_number(metrics.get('AP50')),
                             'AP75': as_number(metrics.get('AP75')), 'wall_seconds': as_number(summary.get('wall_seconds')),
                             'images': summary.get('images')})
            except (OSError, ValueError, AttributeError) as error:
                errors.append(folder.name + '：验证摘要暂不可读：' + str(error)[:100])
        return sorted(rows, key=lambda x: ((as_number(x['update']) or 0), x['tag'])), errors

    @staticmethod
    def _phase(state, update, logs):
        if state == 'COMPLETE':
            return {'kind': 'complete', 'label': '本次训练与末步验证已完成', 'tag': None, 'images': None, 'total': None}
        if state in TERMINAL:
            return {'kind': 'stopped', 'label': state, 'tag': None, 'images': None, 'total': None}
        event = None
        for line in logs:
            try:
                value = json.loads(line)
                if value.get('event') in ('eval_progress', 'evaluation_complete', 'update'):
                    event = value
            except (ValueError, AttributeError):
                pass
        if event and event.get('event') == 'eval_progress':
            tag = str(event.get('tag', ''))
            digits = ''.join(x for x in tag.split('_')[0] if x.isdigit())
            at = int(digits) if digits else update
            if update <= at:
                return {'kind': 'validation', 'label': '正在验证（验证期间训练步数保持不变）',
                        'tag': tag, 'images': event.get('images'), 'total': event.get('total')}
        if state == 'RUNNING' and update == 0:
            return {'kind': 'initializing', 'label': '初始化/公开起点评价，尚无参数更新',
                    'tag': None, 'images': None, 'total': None}
        return {'kind': 'training' if state == 'RUNNING' else 'unknown',
                'label': '正在训练' if state == 'RUNNING' else '等待完整运行状态',
                'tag': None, 'images': None, 'total': None}

    def _run(self, entry):
        directory, group, state = entry['_path'], entry['_group'], entry['_status']
        protocol = dict(entry['_manifest'].get('protocol') or {})
        protocol.setdefault('effective_batch', protocol.get('effective_batch_size'))
        wanted = ('task', 'main_metric', 'model', 'strategy', 'seed', 'epochs', 'updates', 'N_R', 'N_S', 'N_val', 'input_size',
                  'microbatch_size', 'effective_batch', 'effective_batch_size', 'numeric_mode', 'num_workers', 'checkpoint_every', 'evaluation_every')
        identifier = entry['id']
        reader = self.steps.setdefault(identifier, IncrementalLog())
        state_update = as_number(state.get('update')) or 0
        initial_wait = state.get('state') == 'RUNNING' and state_update == 0
        points = reader.update(directory / 'steps.jsonl', allow_missing_initial=initial_wait)
        latest = points[-1] if points else {}
        update = max(state_update, latest.get('update', 0))
        total = as_number(protocol.get('updates'))
        validation, errors = self._validation(directory, protocol)
        stdout = tail_lines(group / 'controller.stdout.log')
        stderr = tail_lines(group / 'controller.stderr.log', maximum_lines=12)
        phase = self._phase(state.get('state'), update, stdout)
        cp = self.checkpoints.setdefault(identifier, IncrementalLog(extractor=lambda row: {'update': row.get('update')} if row.get('kind') == 'checkpoint' else None))
        cp.update(directory / 'checkpoints.jsonl', allow_missing_initial=initial_wait)
        eta = None
        timings = [row['wall_seconds'] for row in points[-200:] if row.get('wall_seconds') and row['wall_seconds'] > 0]
        validation_times = [row['wall_seconds'] for row in validation if row['wall_seconds'] and row['wall_seconds'] > 0]
        if state.get('state') == 'COMPLETE':
            eta = 0.0
        elif total and timings and validation_times and state.get('state') == 'RUNNING':
            remaining = max(0, total - update)
            interval = as_number(protocol.get('evaluation_every')) or 1000
            schedule = protocol.get('evaluation_schedule')
            if isinstance(schedule, list) and schedule and all(isinstance(item, dict) and isinstance(item.get('tag'), str) for item in schedule):
                expected = {item['tag'] for item in schedule}
                done = {row['tag'] for row in validation}
            else:
                expected = {0, int(total)} | set(range(int(interval), int(total) + 1, int(interval)))
                done = {int(row['update']) for row in validation if as_number(row.get('update')) is not None}
            pending_count = len(expected - done)
            if phase['kind'] == 'validation' and as_number(phase.get('total')) and as_number(phase.get('images')) is not None:
                pending_count -= min(1.0, max(0.0, phase['images'] / phase['total']))
            eta = remaining * statistics.median(timings) + max(0, pending_count) * statistics.mean(validation_times)
        last_mtime = None
        for path in (directory / 'status.json', directory / 'steps.jsonl', group / 'controller.stdout.log', group / 'controller.stderr.log', entry['_control_group'] / 'status.json'):
            try:
                last_mtime = max(last_mtime or 0, path.stat().st_mtime)
            except OSError:
                pass
        process = _process_identity(state)
        if state.get('state') == 'RUNNING' and process['alive'] is False:
            errors.append('状态记录RUNNING，但对应训练进程已退出；等待控制器最终状态')
            phase = {'kind': 'unknown', 'label': '训练进程已退出，等待最终状态', 'tag': None, 'images': None, 'total': None}
        run = {key: entry[key] for key in ('id', 'label', 'menu_label', 'budget', 'state', 'group_id', 'archived', 'archive_pending', 'source_release_complete',
                                                  'storage_mode', 'control_group_path', 'output_group_path')}
        run.update(comparison_signature=comparison_signature(entry['_manifest']), path=str(directory), group_path=str(group), group_state=entry['_group_status'].get('state'),
                   protocol={key: protocol.get(key) for key in wanted},
                   progress={'update': update, 'total_updates': total, 'fraction': min(1., update / total) if total else None,
                             'R_exposures': latest.get('R_exposures'), 'S_exposures': latest.get('S_exposures'),
                             'phase': latest.get('phase', state.get('phase')),
                             'phase_update': latest.get('phase_update', state.get('phase_update'))},
                   latest={key: latest.get(key) for key in ('loss', 'loss_box', 'loss_cls', 'loss_dfl', 'gradient_norm', 'learning_rates', 'wall_seconds')},
                   phase=phase, curves={'loss': points, 'validation': validation}, checkpoint_count=cp.records,
                   last_activity_utc=datetime.fromtimestamp(last_mtime, timezone.utc).isoformat() if last_mtime else None,
                   eta_seconds=eta, eta_note=('估计包含剩余训练和既定验证；原始证据直接写入G盘，无结束后整组搬运。速度变化会调整估计。'
                                              if entry.get('storage_mode') == 'direct_g' else '估计包含剩余训练和既定验证，不含结束后的G盘迁移；速度变化会调整估计。'),
                   log_tail=stdout[-28:] + (['[stderr]'] + stderr if stderr else []),
                   history_truncated=reader.history_truncated, process=process,
                   errors=list(reader.errors) + list(cp.errors) + errors)
        return run

    def system(self):
        if self.system_cached is not None and time.monotonic() - self.system_at < self.refresh_seconds:
            return self.system_cached
        result = {'cpu_percent': None, 'ram_free_bytes': None, 'ram_total_bytes': None, 'disks': {}, 'gpu': None, 'gpu_error': None}
        try:
            memory = psutil.virtual_memory()
            result.update(cpu_percent=psutil.cpu_percent(interval=None), ram_free_bytes=memory.available, ram_total_bytes=memory.total)
        except (OSError, psutil.Error):
            pass
        for label, path in (('D', self.root), ('G', self.archive_root)):
            try:
                usage = shutil.disk_usage(path if path.exists() else Path(path.anchor))
                result['disks'][label] = {'free_bytes': usage.free, 'total_bytes': usage.total}
            except OSError:
                result['disks'][label] = None
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            query = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=2, check=True, creationflags=flags)
            values = query.stdout.strip().splitlines()[0].split(',')
            numbers = [float(value.strip()) for value in values[1:]]
            result['gpu'] = {'name': values[0].strip(), 'utilization_percent': numbers[0],
                             'memory_used_bytes': int(numbers[1] * 1024**2), 'memory_total_bytes': int(numbers[2] * 1024**2),
                             'memory_free_bytes': int(numbers[3] * 1024**2)}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
            result['gpu_error'] = 'GPU只读查询不可用：' + str(error)[:160]
        self.system_cached, self.system_at = result, time.monotonic()
        return result

    def _snapshot(self, identifier=None, cache_mode='tail'):
        cache_key = (identifier or '', cache_mode)
        with self.lock:
            cached = self.cached.get(cache_key)
            if cached and time.monotonic() - cached[0] < self.refresh_seconds:
                return cached[1]
            entries, errors = self.inventory()
            selected = next((entry for entry in entries if entry['id'] == identifier), None) if identifier else self._default(entries)
            if identifier and selected is None:
                errors.append('所选实验当前不存在或迁移尚未闭合，等待刷新')
            result = {'generated_at_utc': utc_now(), 'refresh_seconds': self.refresh_seconds,
                      'runs': [{key: value for key, value in entry.items() if not key.startswith('_')} for entry in entries],
                      'selected_run_id': selected['id'] if selected else None, 'run': None,
                      'confirmation': confirmation_snapshot(self.root, self.archive_root),
                      'system': self.system(), 'errors': errors}
            if selected:
                try:
                    result['run'] = self._run(selected)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result['errors'].append('读取实验快照失败：' + str(error)[:160])
            self.cached[cache_key] = (time.monotonic(), result)
            # API只允许已发现ID；缓存键不能靠任意query无限增长。
            if len(self.cached) > 100:
                self.cached = {cache_key: self.cached[cache_key]}
            return result


    def _history_path(self, identifier):
        with self.lock:
            entries, _ = self.inventory()
            entry = next((row for row in entries if row['id'] == identifier), None)
            if entry is None:
                raise ValueError('实验不存在或迁移入口尚未闭合')
            return entry['_path'] / 'steps.jsonl'

    def snapshot(self, identifier=None, history=None):
        if history not in (None, 'full'):
            raise ValueError('未知history模式')
        result = self._snapshot(identifier, history or 'tail')
        if history != 'full':
            return result
        if result['run'] is None:
            return {**result, 'history': unavailable_history('error', '所选实验不可用，未启动后台历史读取')}
        return {**result, 'history': self.histories.request(result['selected_run_id'])}

    def close(self):
        self.histories.close()


class MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 4

    def server_close(self):
        try:
            if hasattr(self, 'monitor'):
                self.monitor.close()
        finally:
            super().server_close()

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    server_version = 'TrainingMonitor/1'
    sys_version = ''

    def log_message(self, *_):
        pass

    def _reply(self, status, value, content_type='application/json; charset=utf-8'):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self._reply(405, {'error': '本监控仅支持只读GET'})

    def do_GET(self):
        port = self.server.server_address[1]
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host', '').lower() not in allowed:
            self._reply(403, {'error': '仅本机监控地址可访问'})
            return
        origin = self.headers.get('Origin')
        if origin and origin.rstrip('/').lower() not in {f'http://{host}' for host in allowed}:
            self._reply(403, {'error': '跨站读取拒绝'})
            return
        try:
            parsed = urlsplit(self.path)
            if parsed.path == '/api/health':
                self._reply(200, self.server.health)
            elif parsed.path == '/api/confirmation':
                self._reply(200, confirmation_snapshot(self.server.monitor.root, self.server.monitor.archive_root))
            elif parsed.path == '/api/status':
                try:
                    params = parse_qs(parsed.query, max_num_fields=4, keep_blank_values=True, strict_parsing=True)
                except ValueError:
                    self._reply(400, {'error': '无效查询参数'})
                    return
                if set(params) - {'run', 'history'} or any(len(values) != 1 for values in params.values()):
                    self._reply(400, {'error': '未知或重复查询参数'})
                    return
                identifier = params.get('run', [None])[0]
                history = params.get('history', [None])[0]
                if ('run' in params and (not identifier or len(identifier) != 20 or any(x not in '0123456789abcdef' for x in identifier))):
                    self._reply(400, {'error': '无效实验ID'})
                    return
                if history not in (None, 'full'):
                    self._reply(400, {'error': '无效history模式'})
                    return
                self._reply(200, self.server.monitor.snapshot(identifier, history=history))
            elif parsed.path == '/training_monitor_charts.js':
                script = self.server.js_path
                if script.is_symlink() or script.is_junction() or script.stat().st_size > 256 * 1024:
                    raise ValueError('图表脚本超限或路径重定向')
                self._reply(200, script.read_bytes(), 'text/javascript; charset=utf-8')
            elif parsed.path == '/':
                page = self.server.html_path
                if page.stat().st_size > 256 * 1024:
                    raise ValueError('监控页面超过256KiB')
                self._reply(200, page.read_bytes(), 'text/html; charset=utf-8')
            else:
                self._reply(404, {'error': '不存在该监控端点'})
        except (OSError, ValueError) as error:
            self._reply(503, {'error': '监控读取暂不可用：' + str(error)[:160]})


def create_server(research_root, archive_root, *, port=8765, html_path=None, duration_hours=12):
    if not 0 < duration_hours <= 168:
        raise ValueError('监控时长必须在0–168小时内')
    root = Path(research_root).absolute()
    server = MonitorHTTPServer(('127.0.0.1', port), Handler)
    server.monitor = Monitor(root, archive_root)
    server.html_path = Path(html_path) if html_path else Path(__file__).with_suffix('.html')
    server.js_path = server.html_path.with_name('training_monitor_charts.js')
    started = datetime.now(timezone.utc)
    server.health = {'service': SERVICE, 'version': VERSION, 'pid': os.getpid(),
                     'create_time': psutil.Process().create_time(), 'root': str(root),
                     'url': f'http://127.0.0.1:{server.server_address[1]}', 'started_at_utc': started.isoformat(),
                     'deadline_utc': (started + timedelta(hours=duration_hours)).isoformat(), 'read_only': True}
    server.timeout = .5
    return server


def owner_alive(value):
    if value.get('service') != SERVICE or not value.get('command'):
        raise RuntimeError('既有监控登记不是本工具的可信记录')
    state = _process_identity(value)
    if state['alive'] is None:
        raise RuntimeError('无法核对既有监控PID身份')
    if not state['alive']:
        return False
    try:
        actual = psutil.Process(value['pid']).cmdline()
        if actual != value['command']:
            raise RuntimeError('登记PID仍在但命令不同，拒绝覆盖监控登记')
        return True
    except psutil.AccessDenied as error:
        raise RuntimeError('无法核对既有监控进程命令') from error
    except psutil.NoSuchProcess:
        return False


def existing_monitor(root):
    """只复用PID/命令/根目录与本地健康接口一致的已有服务，不碰训练进程。"""
    path = Path(root).absolute() / '.training_monitor.json'
    if not path.exists():
        return None
    value = read_json(path, maximum=65536)
    if value.get('root') != str(Path(root).absolute()):
        raise RuntimeError('监控登记绑定另一研究目录')
    if not owner_alive(value):
        return None
    parsed = urlsplit(str(value.get('url', '')))
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.username or parsed.password
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.port is None):
        raise RuntimeError('拒绝复用非本地监控URL')
    with urlopen(value['url'].rstrip('/') + '/api/health', timeout=2) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise RuntimeError('已有监控健康响应超限')
    health = json.loads(raw)
    keys = ('service', 'root', 'pid', 'create_time', 'url', 'read_only')
    if any(health.get(key) != value.get(key) for key in keys) or health.get('service') != SERVICE:
        raise RuntimeError('已有监控身份与登记不符')
    return value['url']


@contextmanager
def registration(server):
    root = Path(server.health['root'])
    path = root / '.training_monitor.json'
    if path.exists():
        old = read_json(path, maximum=65536)
        if old.get('root') != str(root):
            raise RuntimeError('监控登记绑定另一研究目录')
        if owner_alive(old):
            raise RuntimeError('已有可信监控服务：' + str(old.get('url')))
        identity = (path.stat().st_dev, path.stat().st_ino)
        data = path.read_bytes()
        if (path.stat().st_dev, path.stat().st_ino) != identity or path.read_bytes() != data:
            raise RuntimeError('监控登记在核对期间变化')
        path.unlink()
    value = {**server.health, 'command': psutil.Process().cmdline()}
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    identity = (path.stat().st_dev, path.stat().st_ino)
    try:
        yield value
    finally:
        if path.exists() and (path.stat().st_dev, path.stat().st_ino) == identity and path.read_bytes() == raw:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--archive-root', type=Path, default=Path('G:/FutsalMOT_Training'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--duration-hours', type=float, default=12)
    parser.add_argument('--deadline-utc', help='界面升级沿原绝对截止，不延长服务')
    parser.add_argument('--stop-file', type=Path, help='只停止本只读服务')
    parser.add_argument('--open', action='store_true', help='打开本地监控；已有可信服务则直接复用')
    args = parser.parse_args()
    if not args.root.is_dir():
        raise ValueError('研究目录不存在')
    existing = existing_monitor(args.root)
    if existing:
        if args.open:
            webbrowser.open(existing)
        print(json.dumps({'state': 'EXISTING_MONITOR', 'url': existing}, ensure_ascii=False))
        return
    server = create_server(args.root, args.archive_root, port=args.port, duration_hours=args.duration_hours)
    if args.deadline_utc:
        deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
        if deadline.tzinfo is None or deadline.timestamp() <= time.time():
            raise ValueError('服务原截止已到或无时区')
        server.health['deadline_utc'] = min(deadline, datetime.fromisoformat(server.health['deadline_utc'])).isoformat()
    try:
        with registration(server):
            if args.open:
                webbrowser.open(server.health['url'])
            end = datetime.fromisoformat(server.health['deadline_utc']).timestamp()
            while time.time() < end and not (args.stop_file and args.stop_file.exists()):
                server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
