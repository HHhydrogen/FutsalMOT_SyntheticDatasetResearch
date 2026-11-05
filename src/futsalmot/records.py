"""仅在显式调用时写实际记录；没有自动建runs目录。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import random
import json
from pathlib import Path
import tempfile
import stat
import threading

CHECKPOINT_KEYS = frozenset({"schema_version", "raw_model", "ema", "optimizer", "rng", "sampler", "phase", "phase_updates", "committed_updates", "source_bindings", "numeric_mode", "scaler"})

def validate_checkpoint_contract(state: dict) -> None:
    if not isinstance(state, dict):
        raise ValueError("检查点根必须为状态字典")
    missing = CHECKPOINT_KEYS - state.keys()
    if missing:
        raise ValueError("检查点缺字段: " + ",".join(sorted(missing)))
    if not isinstance(state["schema_version"], str) or not state["schema_version"]:
        raise ValueError("schema_version必须为非空字符串")
    if not isinstance(state["phase"], str) or not state["phase"]:
        raise ValueError("phase必须明确")
    if state["numeric_mode"] not in ("fp32", "amp_bf16", "amp_fp16"):
        raise ValueError("未冻结/不支持的数值模式")
    if state["numeric_mode"] in ("fp32", "amp_bf16") and state["scaler"] is not None:
        raise ValueError("FP32/BF16检查点scaler应为null；只有AMP FP16可保存scaler")
    if state["scaler"] is not None and not isinstance(state["scaler"], dict):
        raise ValueError("scaler必须为state_dict或null")
    for name in ("committed_updates", "phase_updates"):
        if isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
            raise ValueError("更新数必须为非负整数")
    if state["phase_updates"] > state["committed_updates"]:
        raise ValueError("阶段更新不能超过累计提交更新")
    for name in ("raw_model", "ema", "optimizer", "rng", "sampler", "source_bindings"):
        if not isinstance(state[name], dict):
            raise ValueError(f"{name}必须为state_dict/状态字典，不保存模型对象")
    if not state["source_bindings"]:
        raise ValueError("缺数据/模型/代码来源绑定")


CHECKPOINT_SCHEMA = "futsalmot_checkpoint_v1"


def _torch():
    # inspect/dry-run及stdlib契约检查不会隐式导入Torch。
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("完整检查点功能需要已安装的Torch；不自动安装") from exc
    return torch


def _binding_digest(bindings: dict) -> str:
    try:
        raw = json.dumps(bindings, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("来源绑定必须为可规范化JSON字典") from exc
    return hashlib.sha256(raw).hexdigest()


def _safe_state_tree(value, torch, active=None):
    """仅原生state_dict数据/张量；不pickle模型或任意自定义对象，不改dtype。"""
    if value is None or type(value) in (bool, int, float, str, bytes) or torch.is_tensor(value):
        return
    if isinstance(value, (dict, list, tuple)):
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError("检查点状态不能有循环容器")
        active.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                if type(key) not in (str, int, float, bool, bytes, tuple):
                    raise ValueError("检查点字典键类型不支持")
                _safe_state_tree(key, torch, active)
                _safe_state_tree(child, torch, active)
        else:
            for child in value:
                _safe_state_tree(child, torch, active)
        active.remove(id(value))
        return
    raise ValueError("检查点只接受原生状态字典、标量和Tensor；NumPy RNG请使用capture_rng_state")


@dataclass(frozen=True)
class CheckpointReceipt:
    """由本工具保存时返回，须由可信run记录保管；SHA不是来源真实性签名。"""
    path: str
    sha256: str
    bytes: int
    source_bindings_sha256: str
    schema_version: str = CHECKPOINT_SCHEMA
    producer: str = "futsalmot.records/v1"


def save_checkpoint(path: str | Path, state: dict) -> CheckpointReceipt:
    """同步完整保存到唯一新路径。调用者须在提交边界固定状态且不得并发修改。

    不裁字段、不half化；传入的是raw/EMA/optimizer等state_dict，不是网络对象。
    临时文件完整写入、fsync、关闭后，使用同目录硬链接原子无覆盖发布。
    不支持硬链接的文件系统明确失败，不退化为可覆盖的replace。
    """
    validate_checkpoint_contract(state)
    if state["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("保存仅支持明确的futsalmot_checkpoint_v1 schema")
    bindings_sha = _binding_digest(state["source_bindings"])
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("拒绝覆盖既有检查点")
    if not destination.parent.is_dir():
        raise FileNotFoundError("检查点父目录须由实际run显式创建")
    torch = _torch()
    _safe_state_tree(state, torch)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".futsalmot-checkpoint-", suffix=".tmp", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save({"producer": "futsalmot.records/v1", "state": state}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # 文件已关闭；发布前完整流式摘要，不一次性载入整份权重。
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = temporary.stat().st_size
        os.link(temporary, destination)  # 若其他writer先占用，原子失败，绝不覆盖。
        return CheckpointReceipt(str(destination.resolve()), digest, size, bindings_sha)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(receipt: CheckpointReceipt, *, expected_bindings: dict) -> dict:
    """仅消费本工具保存并从可信run记录取得的路径+摘要收据。

    固定map_location=cpu、weights_only=True；拒绝公开.pt/任意pickle及摘要/绑定
    不符的文件。收据不提供密码学认证，不能信任随陌生权重附带的伪造收据。
    返回完整状态；不替调用者加载网络/优化器、不自动恢复RNG、不挑best。
    """
    if not isinstance(receipt, CheckpointReceipt) or receipt.producer != "futsalmot.records/v1" or receipt.schema_version != CHECKPOINT_SCHEMA:
        raise ValueError("需要本工具可信run的完整检查点收据")
    if not expected_bindings or _binding_digest(expected_bindings) != receipt.source_bindings_sha256:
        raise ValueError("调用者来源绑定与检查点不符")
    path = Path(receipt.path)
    if not path.is_absolute() or str(path.resolve()) != receipt.path or path.is_symlink():
        raise ValueError("检查点路径与收据不符")
    # 同一已打开文件描述符先hash再读取，避免hash后重新按路径开另一文件。
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != receipt.bytes or hashlib.file_digest(stream, "sha256").hexdigest() != receipt.sha256:
            raise ValueError("检查点字节数或SHA与可信收据不符")
        stream.seek(0)
        payload = _torch().load(stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("producer") != receipt.producer or set(payload) != {"producer", "state"}:
        raise ValueError("不是本工具的完整检查点容器")
    state = payload["state"]
    validate_checkpoint_contract(state)
    if state["schema_version"] != receipt.schema_version or _binding_digest(state["source_bindings"]) != receipt.source_bindings_sha256:
        raise ValueError("检查点内部schema/绑定与收据不符")
    return state


def capture_rng_state(*, include_cuda: bool = False) -> dict:
    """捕获Python/NumPy全局/Torch CPU RNG；默认不访问任何CUDA API。

    独立random.Random/NumPy Generator/采样器须由其消费者另存状态。本函数不
    初始化GPU；只有调用者已授权且CUDA已初始化时，才可显式include_cuda。
    """
    import numpy as np
    torch = _torch()
    cuda = None
    if include_cuda:
        if not torch.cuda.is_initialized():
            raise RuntimeError("未初始化CUDA；拒绝为采集RNG初始化或探测设备")
        cuda = torch.cuda.get_rng_state_all()
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"schema_version": "futsalmot_rng_v1", "python": random.getstate(),
            "numpy": {"bit_generator": name, "keys": keys.tolist(), "position": int(position),
                      "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
            "torch_cpu": torch.get_rng_state().clone(), "cuda": cuda}


def restore_rng_state(state: dict, *, allow_cuda: bool = False) -> None:
    """先校验CPU随机状态，再恢复全局流；CUDA分支必须显式允许且已初始化。

    只恢复RNG，不调用网络/优化器或采样器。CUDA恢复尚需未来GPU准入验证。
    """
    import numpy as np
    torch = _torch()
    required = {"schema_version", "python", "numpy", "torch_cpu", "cuda"}
    if not isinstance(state, dict) or set(state) != required or state["schema_version"] != "futsalmot_rng_v1":
        raise ValueError("不完整或未知RNG schema")
    cuda = state["cuda"]
    if cuda is not None:
        if not allow_cuda:
            raise ValueError("检查点含CUDA RNG，调用者未允许恢复")
        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA未初始化，不能恢复CUDA RNG")
        if not isinstance(cuda, list) or not cuda:
            raise ValueError("CUDA RNG状态必须为非空列表")
    n = state["numpy"]
    if not isinstance(n, dict) or set(n) != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("NumPy RNG缺必要字段")
    numpy_state = (n["bit_generator"], np.array(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"])
    # 用独立CPU发生器预检，避免缺字段/坏状态导致全局RNG只恢复一半。
    random.Random(0).setstate(state["python"])
    np.random.RandomState(0).set_state(numpy_state)
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)

def _strict_json(raw: bytes):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError("JSON存在重复key")
            result[key]=value
        return result
    def nonfinite(value):
        raise ValueError("JSON不允许NaN/Infinity")
    try:
        value=json.loads(raw.decode("utf-8"),object_pairs_hook=unique,parse_constant=nonfinite)
        json.dumps(value,allow_nan=False) # 也拒绝1e999被标准loads解析成Infinity。
        return value
    except (UnicodeError,json.JSONDecodeError,TypeError) as exc:
        raise ValueError("损坏或非UTF8合法JSON") from exc


def _record_payload(record,*,indent=None):
    if not isinstance(record,dict):
        raise ValueError("运行记录/manifest须为JSON对象")
    try:
        raw=(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=indent)+"\n").encode("utf-8")
    except (TypeError,ValueError,UnicodeError) as exc:
        raise ValueError("记录必须为有限、可编码的合法JSON") from exc
    _strict_json(raw) # int/string混合键被JSON转换成同名时也拒绝。
    return raw


def _canonical_manifest(value):
    normalized=_strict_json(_record_payload(value))
    return json.dumps(normalized,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(",", ":"))


def _guard_record_path(path,*,directory=False,allow_missing=False):
    path=Path(path)
    target=None
    for part in (path,*path.parents):
        try:
            info=part.lstat()
        except FileNotFoundError:
            if allow_missing:continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400:
            raise ValueError("记录路径拒绝符号链接/junction/重解析点")
        if part == path:
            target=info
            good=stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
            if not good:raise ValueError("记录路径必须为真实目录/普通文件")
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("记录父路径必须为真实目录")
    return target


def _record_stat(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)


def _read_record_file(path):
    before=_record_stat(_guard_record_path(path))
    with Path(path).open("rb") as stream:
        if _record_stat(os.fstat(stream.fileno())) != before:
            raise ValueError("记录打开时来源发生变化")
        raw=stream.read()
        if _record_stat(os.fstat(stream.fileno())) != before:
            raise ValueError("记录读取时来源发生变化")
    if _record_stat(_guard_record_path(path)) != before:
        raise ValueError("记录来源发生变化")
    return raw,before


class RunRecorder:
    """显式持久记录；对象内锁，不提供跨进程互斥或优化器事务原子性。

    resume只校验既有日志与调用者已验证检查点的完整持久边界，不恢复模型。
    不截断/修补/删除不齐记录；I/O失败后对象FAIL_STOP，保留所有partial证据。
    """
    STEP_KEYS=frozenset({"update","phase","domain","uids","loss","lr","gradient_norm","R_exposures","S_exposures"})

    def __init__(self,directory: str|Path,manifest:dict):
        payload=_record_payload(manifest,indent=2) # 先验证，不为坏manifest建目录。
        self.directory=Path(directory).absolute()
        try:
            existing=self.directory.lstat()
        except FileNotFoundError:
            existing=None
        if existing is not None:
            _guard_record_path(self.directory,directory=stat.S_ISDIR(existing.st_mode))
            raise FileExistsError("拒绝覆盖既有run")
        _guard_record_path(self.directory,directory=True,allow_missing=True)
        self.directory.mkdir(parents=True)
        self._initialize_state()
        try:
            with (self.directory/"run.json").open("xb") as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            self._manifest_raw,self._manifest_stat=_read_record_file(self.directory/"run.json")
            if self._manifest_raw != payload:raise ValueError("manifest写入后来源不一致")
        except BaseException:
            self._failed=True
            raise

    def _initialize_state(self):
        info=_guard_record_path(self.directory,directory=True)
        self._directory_id=(info.st_dev,info.st_ino)
        self._lock=threading.RLock()
        self._last_step=0
        self._failed=False
        self._log_stats={"steps.jsonl":None,"events.jsonl":None}

    @classmethod
    def resume(cls,directory: str|Path,*,expected_manifest:dict,checkpoint_committed_updates:int):
        if type(checkpoint_committed_updates) is not int or checkpoint_committed_updates < 0:
            raise ValueError("可信检查点committed_updates必须为非负整数")
        expected=_canonical_manifest(expected_manifest)
        result=cls.__new__(cls)
        result.directory=Path(directory).absolute()
        _guard_record_path(result.directory,directory=True) # 不mkdir，不新建缺失run。
        result._initialize_state()
        raw,identity=_read_record_file(result.directory/"run.json")
        actual=_strict_json(raw)
        if not isinstance(actual,dict) or _canonical_manifest(actual) != expected:
            raise ValueError("BLOCKED: 不可变run.json与期望manifest绑定不一致")
        last=0
        for name in ("steps.jsonl","events.jsonl"):
            path=result.directory/name
            info=_guard_record_path(path,allow_missing=True)
            if info is None:continue
            before=_record_stat(info)
            with path.open("rb") as stream:
                if _record_stat(os.fstat(stream.fileno())) != before:
                    raise ValueError("BLOCKED: 日志打开时变化")
                for line_number,line in enumerate(stream,1):
                    if not line.endswith(b"\n"):
                        raise ValueError(f"BLOCKED: {name}第{line_number}行截断/缺持久换行")
                    record=_strict_json(line)
                    if not isinstance(record,dict):raise ValueError("BLOCKED: 日志行必须为对象")
                    if name == "steps.jsonl":
                        cls._validate_step(record,last)
                        last=record["update"]
                if _record_stat(os.fstat(stream.fileno())) != before:
                    raise ValueError("BLOCKED: 日志读取期间变化")
            if _record_stat(_guard_record_path(path)) != before:
                raise ValueError("BLOCKED: 日志读取后变化")
            result._log_stats[name]=before
        if last != checkpoint_committed_updates:
            raise ValueError("BLOCKED: 日志末步与可信检查点提交数不齐，禁止猜测重放或修补")
        # 所有校验完成后对象才可供追加；小manifest须仍为同一原字节。
        current,current_stat=_read_record_file(result.directory/"run.json")
        if current != raw or current_stat != identity:
            raise ValueError("BLOCKED: 恢复期间manifest变化")
        result._manifest_raw,result._manifest_stat=raw,identity
        result._last_step=last
        return result

    @classmethod
    def _validate_step(cls,record,last):
        if not isinstance(record,dict) or cls.STEP_KEYS-record.keys():
            raise ValueError("逐步记录缺必要字段")
        if type(record["update"]) is not int or record["update"] != last+1:
            raise ValueError("更新记录必须为逐步连续的正整数")

    def _assert_live(self):
        if self._failed:raise RuntimeError("记录对象已FAIL_STOP；保留旧证据，禁止继续追加")

    def event(self,event:dict) -> None:
        with self._lock:
            self._assert_live()
            payload=_record_payload(event)
            self._append_payload("events.jsonl",payload)

    def step(self,record:dict) -> None:
        with self._lock:
            self._assert_live()
            self._validate_step(record,self._last_step)
            payload=_record_payload(record)
            self._append_payload("steps.jsonl",payload)
            self._last_step=record["update"]

    def _append(self,name:str,record:dict) -> None:
        """保留旧私有入口；调用者不应绕过step的连续性契约。"""
        if name not in self._log_stats:raise ValueError("未知日志文件")
        if name == "steps.jsonl":return self.step(record)
        return self.event(record)

    def _append_payload(self,name,payload):
        try:
            info=_guard_record_path(self.directory,directory=True)
            if (info.st_dev,info.st_ino) != self._directory_id:
                raise ValueError("run目录身份变化")
            raw,identity=_read_record_file(self.directory/"run.json")
            if raw != self._manifest_raw or identity != self._manifest_stat:
                raise ValueError("不可变manifest来源/字节变化")
            path=self.directory/name
            previous=self._log_stats[name]
            current=_guard_record_path(path,allow_missing=True)
            if (None if current is None else _record_stat(current)) != previous:
                raise ValueError("日志来源/持久边界变化，拒绝继续追加")
            flags=os.O_WRONLY|os.O_APPEND|getattr(os,"O_BINARY",0)
            if previous is None:flags|=os.O_CREAT|os.O_EXCL
            descriptor=os.open(path,flags,0o600)
            try:
                opened_stream=os.fdopen(descriptor,"ab")
            except BaseException:
                os.close(descriptor) # fdopen未接管句柄时也必须释放；文件证据保留。
                raise
            with opened_stream as stream:
                opened=os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or (previous is not None and _record_stat(opened) != previous):
                    raise ValueError("日志打开后身份变化")
                _guard_record_path(path) # 写前再拒绝父路径或目标reparse。
                written=stream.write(payload)
                if written != len(payload):raise OSError("日志发生短写，保留partial证据")
                stream.flush();os.fsync(stream.fileno())
                after=_record_stat(os.fstat(stream.fileno()))
                if after[2] != (0 if previous is None else previous[2])+len(payload):
                    raise ValueError("日志长度与本对象追加边界不一致")
            if _record_stat(_guard_record_path(path)) != after:
                raise ValueError("日志追加后来源变化")
            self._log_stats[name]=after
        except BaseException:
            self._failed=True
            raise


@contextmanager
def temporary_diagnostic(manifest: dict | None = None):
    """非主线诊断显式使用；正常或异常退出都删除临时记录。

    正式训练和预登记预算校准使用持久RunRecorder，不能用此上下文删主线证据。
    本上下文不创建项目runs目录，也不把诊断日志搬入Saved。
    """
    with tempfile.TemporaryDirectory(prefix="futsalmot-diagnostic-") as temporary:
        recorder = RunRecorder(Path(temporary) / "diagnostic", {**(manifest or {}), "record_scope": "temporary_diagnostic", "not_formal_training": True})
        yield recorder
