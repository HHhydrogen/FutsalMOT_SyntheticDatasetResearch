"""fresh CPU恢复夹具：微小模块/手填SGD状态，禁止forward/backward/step。"""
from copy import deepcopy
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

_ENV=os.environ.get("CUDA_VISIBLE_DEVICES");os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch
from futsalmot.restore import restore_cpu_checkpoint
from futsalmot.checkpoint_state import snapshot_checkpoint
from futsalmot.optimizers import make_sgd_optimizer
from futsalmot.training import initialize_ema_stage
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.records import capture_rng_state,restore_rng_state,save_checkpoint,load_checkpoint

_THREAD=None
_BLOCK=None

def setUpModule():
    global _THREAD,_BLOCK
    _THREAD=torch.get_num_threads();torch.set_num_threads(2)
    _BLOCK=mock.patch("torch.cuda._lazy_init",side_effect=AssertionError("CPU fixture禁止CUDA初始化"));_BLOCK.start()

def tearDownModule():
    if _BLOCK is not None:_BLOCK.stop()
    if _THREAD is not None:torch.set_num_threads(_THREAD)
    if _ENV is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    else:os.environ["CUDA_VISIBLE_DEVICES"]=_ENV


class Tiny(torch.nn.Module):
    def __init__(self,width=2,dtype=torch.float32):
        super().__init__()
        self.linear=torch.nn.Linear(width,2,dtype=dtype,device="cpu")
        self.norm=torch.nn.BatchNorm1d(2,dtype=dtype,device="cpu")
        self.frozen=torch.nn.Parameter(torch.tensor([.5],dtype=dtype),requires_grad=False)
        self.register_buffer("counter",torch.tensor(7,dtype=torch.int64))
    def forward(self,*args,**kwargs):raise AssertionError("fixture不允许forward")


def sgd(model,decay=.0005):
    # 故意使用阶段0 LR/momentum，合法恢复应载入保存的实际值。
    return make_sgd_optimizer(model,lr=0.,momentum=.8,weight_decay=decay)


def fixture():
    raw=Tiny()
    optimizer=make_sgd_optimizer(raw,lr=.001,momentum=.937,weight_decay=.0005)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter]["momentum_buffer"]=torch.full_like(parameter,.125)
    q=PrefetchSampler(CycleSampler(["a","b","c"],task="detect",domain="R",seed=17))
    first=q.reserve(16);q.commit(first["token"])
    pending=q.reserve(16)
    ema=initialize_ema_stage(raw.state_dict());ema["updates"]=1
    bundle=snapshot_checkpoint(raw_state=raw.state_dict(),ema_state=ema,optimizer_state=optimizer.state_dict(),
                               phase="final",phase_updates=1,committed_updates=1,
                               source_bindings={"data":"fixture_pool","model":"fixture_public_start","code":"fixture_code","protocol":"fixture_protocol"},
                               numeric_mode="amp_bf16",scaler_state=None,samplers={"R":q},
                               cumulative_exposures={"R":16,"S":0},extra_state={"fixture_aux":{"marker":torch.tensor([3.])}})
    return raw,optimizer,q,pending,bundle


def meta(b):return {k:deepcopy(b[k]) for k in ("phase","phase_updates","committed_updates","numeric_mode","cumulative_exposures","effective_batch")}


def restore(b,q,**overrides):
    options=dict(expected_bindings=b["source_bindings"],expected_metadata=meta(b),expected_samplers={"R":q},
                 raw_factory=Tiny,ema_factory=Tiny,sgd_factory=sgd)
    options.update(overrides)
    return restore_cpu_checkpoint(b,**options)


def next_rng():return random.random(),np.random.random(3),torch.rand(3,device="cpu")


class RestoreTests(unittest.TestCase):
    def setUp(self):self.before=capture_rng_state()
    def tearDown(self):restore_rng_state(self.before)
    def assert_rng_equal(self,a,b):
        self.assertEqual(a["python"],b["python"]);self.assertEqual(a["numpy"],b["numpy"])
        self.assertTrue(torch.equal(a["torch_cpu"],b["torch_cpu"]))

    def test_real_save_load_fresh_restore_full_state_and_next_rng_sampler_match(self):
        original,opt,q,pending,b=fixture()
        original_weights={k:v.clone() for k,v in original.state_dict().items()}
        expected_next=next_rng();next_rng()
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"fixture.pt";receipt=save_checkpoint(path,b)
            loaded=load_checkpoint(receipt,expected_bindings=b["source_bindings"])
            result=restore(loaded,q)
            actual_next=next_rng()
            self.assertEqual(expected_next[0],actual_next[0])
            self.assertTrue(np.array_equal(expected_next[1],actual_next[1]))
            self.assertTrue(torch.equal(expected_next[2],actual_next[2]))
            raw,ema=result["raw_model"],result["ema_model"]
            self.assertIsNot(raw,original);self.assertIsNot(raw,ema)
            self.assertTrue(raw.training);self.assertFalse(ema.training)
            self.assertFalse(raw.frozen.requires_grad);self.assertFalse(any(p.requires_grad for p in ema.parameters()))
            for key,value in raw.state_dict().items():
                self.assertTrue(torch.equal(value,b["raw_model"][key]))
                self.assertTrue(torch.equal(original.state_dict()[key],original_weights[key]))
            saved=b["optimizer"]
            self.assertEqual(result["optimizer"].state_dict()["param_groups"],saved["param_groups"])
            self.assertEqual([g["lr"] for g in result["optimizer"].param_groups],[.001]*3)
            for parameter,state in result["optimizer"].state.items():
                self.assertTrue(torch.equal(state["momentum_buffer"],torch.full_like(parameter,.125)))
            restored=result["samplers"]["R"]
            self.assertEqual(restored.pending_batches,0);self.assertEqual(restored.committed_draws,16)
            replay=restored.reserve(16);self.assertEqual(replay["samples"],pending["samples"])
            with self.assertRaises(ValueError):restored.commit(pending["token"])
            self.assertEqual(q.pending_batches,1);self.assertEqual(q.committed_draws,16)
            first=next(iter(result["optimizer"].state.values()))["momentum_buffer"];first.zero_()
            self.assertTrue(all(torch.equal(s["momentum_buffer"],torch.full_like(p,.125)) for p,s in opt.state.items()))
            self.assertEqual(result["ema_updates"],1);self.assertFalse(result["trainer_restored"])
            result["auxiliary_state"]["fixture_aux"]["marker"].zero_()
            self.assertEqual(b["fixture_aux"]["marker"].item(),3.)
        self.assertFalse(path.exists())

    def test_bad_keys_shape_dtype_group_topology_momentum_and_UIDpool_fail(self):
        _,_,q,_,b=fixture()
        def missing(state):state["raw_model"].pop("counter")
        def group_names(state):state["optimizer"]["param_groups"][0]["parameter_names"].reverse()
        def momentum_shape(state):
            key=next(iter(state["optimizer"]["state"]))
            state["optimizer"]["state"][key]["momentum_buffer"]=torch.zeros(99,dtype=torch.float32)
        def momentum_dtype(state):
            key=next(iter(state["optimizer"]["state"]))
            state["optimizer"]["state"][key]["momentum_buffer"]=torch.zeros(2,dtype=torch.float64)
        for change in (missing,group_names,momentum_shape,momentum_dtype):
            bad=deepcopy(b);change(bad)
            with self.assertRaises(ValueError):restore(bad,q)
        for factory in (lambda:Tiny(width=3),lambda:Tiny(dtype=torch.float64)):
            with self.assertRaises(ValueError):restore(b,q,raw_factory=factory)
        with self.assertRaises(ValueError):restore(b,q,sgd_factory=lambda raw:sgd(raw,decay=.0004))
        wrong=PrefetchSampler(CycleSampler(["other"],task="detect",domain="R",seed=17))
        with self.assertRaises(ValueError):restore(b,q,expected_samplers={"R":wrong})

    def test_shared_factory_shared_storage_and_nonCPU_rejected_before_load(self):
        _,_,q,_,b=fixture()
        shared=Tiny();before=shared.linear.weight.detach().clone()
        with self.assertRaisesRegex(ValueError,"共享Module"):restore(b,q,raw_factory=lambda:shared,ema_factory=lambda:shared)
        self.assertTrue(torch.equal(shared.linear.weight,before))
        a,c=Tiny(),Tiny();c.linear.weight=torch.nn.Parameter(a.linear.weight.detach())
        with self.assertRaisesRegex(ValueError,"共享存储"):restore(b,q,raw_factory=lambda:a,ema_factory=lambda:c)
        with self.assertRaisesRegex(ValueError,"CPU"):restore(b,q,raw_factory=lambda:Tiny().to(device="meta"))

    def test_factory_failure_restores_python_numpy_torch_rng_and_existing_objects(self):
        original,opt,q,pending,b=fixture()
        before=capture_rng_state();weights={k:v.clone() for k,v in original.state_dict().items()};state=q.state_dict()
        def failed_factory():
            random.random();np.random.random(7);torch.rand(7,device="cpu")
            raise RuntimeError("fixture factory failure")
        with self.assertRaisesRegex(RuntimeError,"factory failure"):restore(b,q,ema_factory=failed_factory)
        self.assert_rng_equal(before,capture_rng_state())
        self.assertEqual(q.state_dict(),state);self.assertEqual(q.pending_batches,1)
        for key,value in original.state_dict().items():self.assertTrue(torch.equal(value,weights[key]))
        # 失败发生在raw工厂随机初始化之后，也必须恢复调用前全局RNG。
        with self.assertRaises(ValueError):restore(b,q,raw_factory=lambda:Tiny(width=3))
        self.assert_rng_equal(before,capture_rng_state())

    def test_CUDA_and_FP16_bundle_block_without_initializing_device_or_factory(self):
        _,_,q,_,b=fixture();before=capture_rng_state()
        cuda=deepcopy(b);cuda["rng"]["cuda"]=[torch.tensor([1],dtype=torch.uint8)]
        with mock.patch("torch.cuda.is_available",side_effect=AssertionError("不得探测CUDA")),mock.patch("torch.cuda.is_initialized",side_effect=AssertionError("CPU路径不得查询CUDA")):
            with self.assertRaisesRegex(ValueError,"CUDA"):restore(cuda,q)
            fp16=deepcopy(b);fp16.update(numeric_mode="amp_fp16",scaler={"scale":65536.})
            factory=mock.Mock(side_effect=AssertionError("BLOCKED时不构造模型"))
            with self.assertRaisesRegex(ValueError,"BLOCKED.*FP16"):restore(fp16,q,raw_factory=factory)
            factory.assert_not_called()
        self.assert_rng_equal(before,capture_rng_state())

if __name__=="__main__":unittest.main()
