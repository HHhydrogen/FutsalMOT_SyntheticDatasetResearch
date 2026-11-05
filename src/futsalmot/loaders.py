"""有限reservation窗口→真实CPU DataLoader候选桥接，不是训练/提交授权。

只准num_workers=0、pin_memory=False；workers2/4、persistent/pin/CUDA未验证并拒绝。
只持有1–2个窗口的小请求metadata，不缓存全部图像；不reserve/commit/cancel。
caller须保证token仍有效。取消/恢复后必须停止并丢弃旧迭代器，本接口不能认证
pending真实性或回滚外部消费者。窗口单次迭代，不隐式重开epoch/重放失败窗口。
"""
from __future__ import annotations
from copy import deepcopy

import torch
from torch.utils.data import DataLoader,Dataset

from .execution import _reservation_requests,_LINEAGE_FIELDS,EFFECTIVE_BATCH_SIZE
from .batches import collate_bbox_samples


def _generator(generator):
    if not isinstance(generator,torch.Generator) or generator.device.type != "cpu":
        raise ValueError("必须显式提供caller独占CPU torch.Generator，不使用全局RNG")


def loader_generator_state(generator):
    """只捕获独立loader Generator，不是sampler/窗口消费游标或完整恢复状态。"""
    _generator(generator)
    return {"schema_version":"futsalmot_loader_generator_v1","state":generator.get_state().clone()}


def restore_loader_generator_state(generator,state):
    """caller在新窗口创建/迭代前显式恢复专属Generator；不播种/改全局RNG。"""
    _generator(generator)
    if not isinstance(state,dict) or set(state) != {"schema_version","state"} or state["schema_version"] != "futsalmot_loader_generator_v1":
        raise ValueError("未知loader Generator状态schema")
    value=state["state"]
    if not isinstance(value,torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8:
        raise ValueError("loader RNG需要CPU ByteTensor")
    candidate=torch.Generator(device="cpu");candidate.set_state(value) # 独立候选验证，不使用全局流。
    generator.set_state(candidate.get_state())


class _FiniteWindowBatches:
    """模块级可pickle批次请求sampler；只生成已冻结窗口内的明确dict请求。"""
    def __init__(self,windows,microbatch_size):
        self.windows=deepcopy(windows)
        self.microbatch_size=microbatch_size

    def __len__(self):return len(self.windows)*(EFFECTIVE_BATCH_SIZE//self.microbatch_size)

    def __iter__(self):
        for index,window in enumerate(self.windows):
            for start in range(0,EFFECTIVE_BATCH_SIZE,self.microbatch_size):
                yield [{"window_index":index,"sample_offset":position,
                        "reservation_token":window["token"],"request":deepcopy(window["requests"][position])}
                       for position in range(start,start+self.microbatch_size)]


class _WindowRequestDataset(Dataset):
    """worker只消费主进程生成的明确请求；模块级定义，默认整数索引拒绝。"""
    def __init__(self,dataset,windows,microbatch_size):
        self.dataset=dataset
        self.windows=deepcopy(windows)
        self.microbatch_size=microbatch_size

    def __len__(self):return len(self.windows)*EFFECTIVE_BATCH_SIZE

    def __getitem__(self,index):
        if not isinstance(index,dict) or set(index) != {"window_index","sample_offset","reservation_token","request"}:
            raise ValueError("DataLoader只接受本有限窗口明确请求，不支持整数自动采样")
        i,position=index["window_index"],index["sample_offset"]
        if type(i) is not int or not 0 <= i < len(self.windows) or type(position) is not int or not 0 <= position < EFFECTIVE_BATCH_SIZE:
            raise ValueError("请求越出caller明确窗口")
        window=self.windows[i];expected=window["requests"][position]
        if index["reservation_token"] != window["token"] or index["request"] != expected:
            raise ValueError("跨窗口token/UID/曝光请求混用或被篡改")
        sample=self.dataset[deepcopy(expected)]
        if not isinstance(sample,dict) or any(key not in sample for key in _LINEAGE_FIELDS):
            raise ValueError("Dataset缺完整五字段lineage，不接受纯letterbox样本")
        if (sample.get("uid") != expected["uid"] or sample.get("domain") != window["domain"]
            or sample["request"] != expected or sample["sampling_stream"] != window["stream"]
            or sample["data_domain"] != window["domain"]):
            raise ValueError("Dataset输出UID/domain/request与明确窗口不一致")
        return {"sample":sample,"expected_request":deepcopy(expected),"window_index":i,
                "sample_offset":position,"reservation_token":window["token"],
                "microbatch_size":self.microbatch_size,"stream":window["stream"],"domain":window["domain"]}


def collate_reserved_window(items):
    """模块级worker callback；与execution输出同型token/offset/有效batch及lineage。"""
    if not items or not all(isinstance(item,dict) for item in items):raise ValueError("需要明确窗口样本列表")
    first=items[0];m=first["microbatch_size"];start=first["sample_offset"]
    if len(items) != m or start % m or start+m > EFFECTIVE_BATCH_SIZE:
        raise ValueError("微批大小/offset越过16图reservation边界")
    samples=[];expected_lineage=[]
    for offset,item in enumerate(items,start):
        if (any(item.get(key) != first.get(key) for key in ("window_index","reservation_token","microbatch_size","stream","domain"))
            or item.get("sample_offset") != offset):
            raise ValueError("collate跨窗口混token或微批错序")
        sample=item["sample"]
        if sample["request"] != item["expected_request"]:
            raise ValueError("collate前请求血缘已变化")
        samples.append(sample)
        expected_lineage.append({key:deepcopy(sample[key]) for key in _LINEAGE_FIELDS})
    output=collate_bbox_samples(samples)
    if (output.get("sample_lineage") != expected_lineage or any(x is None for x in output["sample_lineage"])
        or output["uids"] != [item["expected_request"]["uid"] for item in items]
        or output["domains"] != [first["domain"]]*m):
        raise ValueError("真实collate血缘/UID/domain缺失或顺序变化")
    output.update(reservation_token=first["reservation_token"],effective_batch_size=EFFECTIVE_BATCH_SIZE,
                  microbatch_offset=start,reservation_window_index=first["window_index"],commit_performed=False)
    return output


class ReservationDataLoader:
    """1–2个reservation组成的真实DataLoader单次有限窗口，尚非正式DataLoader协议。

    显式Generator只供Torch迭代器内部base_seed消费，不为数据/增强编造seed。
    constructor不读取图像；worker0仅next时读取当前微批。caller若保留旧输出，
    其内存不属于本接口控制。所有reader/source绑定仍由原RequestBBoxDataset负责。
    """
    def __init__(self,reservations,dataset,*,microbatch_size,generator,num_workers=0,
                 pin_memory=False,persistent_workers=False):
        if type(num_workers) is not int or num_workers != 0 or pin_memory is not False or persistent_workers is not False:
            raise ValueError("BLOCKED: 只验证num_workers=0/pin_memory=False；进程/pin/persistent未准入")
        _generator(generator)
        if not isinstance(reservations,(list,tuple)) or not 1 <= len(reservations) <= 2:
            raise ValueError("caller必须显式提供1–2个reservation有限窗口")
        if not callable(getattr(dataset,"__getitem__",None)):raise ValueError("需要明确请求Dataset")
        windows=[];seen_tokens=set();last_by_stream={};manager_stream={}
        for reservation in reservations:
            token,stream,domain,requests=_reservation_requests(reservation,microbatch_size)
            if token in seen_tokens:raise ValueError("窗口重复reservation token")
            seen_tokens.add(token)
            nonce,generation,sequence=token.split(":");identity=(nonce,int(generation));sequence=int(sequence)
            if identity in manager_stream and manager_stream[identity] != stream:
                raise ValueError("同一manager世代被错误标成不同stream")
            manager_stream[identity]=stream
            if stream in last_by_stream:
                prior=last_by_stream[stream]
                if identity != prior["identity"] or sequence != prior["sequence"]+1 or requests[0]["exposure_index"] != prior["last_exposure"]+1:
                    raise ValueError("同stream窗口token世代/顺序/曝光不连续")
            last_by_stream[stream]={"identity":identity,"sequence":sequence,"last_exposure":requests[-1]["exposure_index"]}
            windows.append({"token":token,"stream":stream,"domain":domain,"requests":requests})
        self._generator=generator
        self._started=False
        self._window_count=len(windows)
        self._loader=DataLoader(_WindowRequestDataset(dataset,windows,microbatch_size),
                                batch_sampler=_FiniteWindowBatches(windows,microbatch_size),
                                collate_fn=collate_reserved_window,num_workers=0,pin_memory=False,
                                persistent_workers=False,prefetch_factor=None,generator=generator)

    def __len__(self):return len(self._loader)

    def __iter__(self):
        if self._started:raise RuntimeError("有限窗口已使用，禁止隐式重开/重放；caller须明确新建窗口")
        self._started=True
        return iter(self._loader)

    def generator_state_dict(self):return loader_generator_state(self._generator)

    def implementation_info(self):
        return {"backend":"torch.utils.data.DataLoader","num_workers":0,"pin_memory":False,
                "persistent_workers":False,"finite_reservations":self._window_count,
                "candidate_only":True,"commit_performed":False,"multiprocess_verified":False}
