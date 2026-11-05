"""显式16曝光reservation的候选串行微批桥接；不是Trainer或提交事务。

不采样、播种、commit或运行模型。即使生成器全部消费，调用方仍须在真正成功的
optimizer更新后显式提交原token。token仅检查格式/请求一致性，本函数没有sampler
实例，不能认证token仍处于pending；取消/重派前调用方须停止并丢弃旧迭代器。
部分读取不回滚外部模型/文件系统状态，不冒充完整训练事务原子性。
"""
from __future__ import annotations

from copy import deepcopy
import re

EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZES = (1, 2, 4, 8, 16)
_LINEAGE_FIELDS = ("request", "source_reference", "image_source_binding", "sampling_stream", "data_domain")


def _reservation_requests(reservation, microbatch_size):
    if type(microbatch_size) is not int or microbatch_size not in MICROBATCH_SIZES:
        raise ValueError("microbatch_size只允许1/2/4/8/16")
    if not isinstance(reservation, dict) or set(reservation) != {"token", "count", "stream", "data_domain", "samples"}:
        raise ValueError("需要PrefetchSampler完整reservation envelope")
    token = reservation["token"]
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}:(0|[1-9][0-9]*):(0|[1-9][0-9]*)", token) is None:
        raise ValueError("reservation token格式非法，不代表pending真实性认证")
    if type(reservation["count"]) is not int or reservation["count"] != EFFECTIVE_BATCH_SIZE:
        raise ValueError("有效batch reservation count必须恰好16")
    stream = reservation["stream"]
    if stream not in ("R", "S", "R_extra"):
        raise ValueError("reservation stream非法")
    domain = "R" if stream == "R_extra" else stream
    if reservation["data_domain"] != domain:
        raise ValueError("reservation stream/data_domain不一致")
    samples = reservation["samples"]
    if not isinstance(samples, (list, tuple)) or len(samples) != EFFECTIVE_BATCH_SIZE:
        raise ValueError("reservation samples数量必须恰好16")
    requests = []
    previous_exposure = previous_epoch = None
    for sample in samples:
        if not isinstance(sample, dict) or not {"uid", "domain", "epoch", "exposure_index", "augmentation_seed"} <= sample.keys():
            raise ValueError("reservation请求缺UID/domain/epoch/exposure/augmentation")
        if not isinstance(sample["uid"], str) or not sample["uid"]:
            raise ValueError("请求UID必须为非空字符串")
        if sample["domain"] not in (stream, domain):
            raise ValueError("请求domain与reservation stream不一致")
        if sample.get("stream", stream) != stream or sample.get("data_domain", domain) != domain:
            raise ValueError("请求显式stream/data_domain与reservation不一致")
        for field in ("epoch", "exposure_index", "augmentation_seed"):
            if type(sample[field]) is not int or sample[field] < 0:
                raise ValueError("请求epoch/exposure/augmentation必须为非负整数")
        if previous_exposure is not None and sample["exposure_index"] != previous_exposure + 1:
            raise ValueError("reservation samples曝光必须连续且保持原顺序")
        if previous_epoch is not None and sample["epoch"] - previous_epoch not in (0, 1):
            raise ValueError("reservation samples epoch必须保持原循环顺序")
        if "view" in sample and sample["view"] not in ("train", "val"):
            raise ValueError("请求view只允许train/val，不开放test")
        if "split" in sample and sample["split"] not in (("train", "val") if domain == "R" else ("train", "dev")):
            raise ValueError("请求source split非法，不开放test")
        request = deepcopy(sample)
        request.setdefault("stream", stream)
        request.setdefault("data_domain", domain)
        requests.append(request)
        previous_exposure, previous_epoch = sample["exposure_index"], sample["epoch"]
    return token, stream, domain, requests


def _collate(samples):
    # 模块import与reservation元数据校验不导入Torch或读取任何数据。
    from .batches import collate_bbox_samples
    return collate_bbox_samples(samples)


def iter_reserved_microbatches(reservation, dataset, *, microbatch_size):
    """返回只按当前微批顺序读取的生成器；函数调用时仅快照/验证16条小请求。

    dataset须实现__getitem__(显式请求dict)，不接受整数自动采样。只补足envelope
    给出的stream/data_domain，原UID/exposure/augmentation不变，不改reservation。
    每组核对Dataset请求、UID/域及collate五字段sample_lineage，拒绝None或错序。
    本函数只持有本组样本和输出；调用方自行保留旧输出的内存不属于本函数控制。
    """
    token, stream, domain, requests = _reservation_requests(reservation, microbatch_size)
    if not callable(getattr(dataset, "__getitem__", None)):
        raise ValueError("dataset必须接受显式请求的__getitem__")

    def iterate():
        for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size):
            group = requests[offset:offset + microbatch_size]
            samples, expected_lineage = [], []
            for request in group:
                sample = dataset[deepcopy(request)]
                if not isinstance(sample, dict) or any(field not in sample for field in _LINEAGE_FIELDS):
                    raise ValueError("Dataset缺完整五字段sample lineage")
                if (sample.get("uid") != request["uid"] or sample.get("domain") != domain
                        or sample["request"] != request or sample["sampling_stream"] != stream
                        or sample["data_domain"] != domain):
                    raise ValueError("Dataset请求/UID/stream/data_domain错误或错序")
                if not all(isinstance(sample[key], dict) and bool(sample[key])
                           for key in ("request", "source_reference", "image_source_binding")):
                    raise ValueError("Dataset来源/请求绑定缺失，不接受纯几何样本")
                samples.append(sample)
                expected_lineage.append({key: deepcopy(sample[key]) for key in _LINEAGE_FIELDS})
                del sample
            collated = _collate(samples)
            if not isinstance(collated, dict):
                raise ValueError("collate必须返回批次dict")
            lineage = collated.get("sample_lineage")
            if (not isinstance(lineage, list) or len(lineage) != len(group)
                    or any(item is None or not isinstance(item, dict) or set(item) != set(_LINEAGE_FIELDS)
                           for item in lineage)
                    or lineage != expected_lineage):
                raise ValueError("collate sample_lineage缺失、错误或顺序不一致")
            if (collated.get("uids") != [request["uid"] for request in group]
                    or collated.get("domains") != [domain] * len(group)):
                raise ValueError("collate UID/domain顺序不一致")
            collated["reservation_token"] = token
            collated["effective_batch_size"] = EFFECTIVE_BATCH_SIZE
            collated["microbatch_offset"] = offset
            collated["commit_performed"] = False
            del samples, expected_lineage
            yield collated
            # 继续读取下一组前，主动释放本函数对旧输出的引用。
            del collated

    return iterate()
