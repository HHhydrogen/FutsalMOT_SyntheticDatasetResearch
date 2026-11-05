"""随机原生YOLO11n/64 CPU真实梯度/SGD与完整状态续接夹具，不是主线成绩。"""
import os
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-runtime-")
_ENV={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR","YOLO_OFFLINE")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1";os.environ["YOLO_OFFLINE"]="true"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo");os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import torch
from futsalmot.runtime_backend import NativeBBoxBackend
from futsalmot.detectors import build_random_yolo11,NativeDetectionNetwork
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.optimizers import apply_sgd_schedule
from futsalmot.training import schedule_at,criterion_backward_loss,update_ema_state
from futsalmot.records import capture_rng_state,restore_rng_state,save_checkpoint,load_checkpoint

_RNG=None
_THREADS=None
_GUARDS=[]


def setUpModule():
    global _RNG,_THREADS
    _RNG=capture_rng_state();_THREADS=torch.get_num_threads();torch.set_num_threads(2)
    for obj,name in ((torch.cuda,"_lazy_init"),(torch.cuda,"init"),(torch.cuda,"is_available")):
        g=patch.object(obj,name,side_effect=AssertionError("CPU runtime fixture禁止GPU"));g.start();_GUARDS.append(g)


def tearDownModule():
    for guard in reversed(_GUARDS):guard.stop()
    if _RNG is not None:restore_rng_state(_RNG)
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    _TEMP.cleanup()
    for k,v in _ENV.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v


def backend(network=None,**kwargs):
    return NativeBBoxBackend(network or build_random_yolo11(model_size="n"),stage_updates=4,
                             microbatch_size=2,input_size=64,**kwargs)


def batches():
    gen=torch.Generator().manual_seed(71)
    out=[]
    for j in range(8):
        uids=[f"fixture:{2*j+i}" for i in range(2)]
        lineage=[{"request":{"uid":uid,"domain":"R","epoch":0,"exposure_index":2*j+i,"augmentation_seed":5+i},
                  "data_domain":"R","sampling_stream":"R","source_reference":{"fixture":True},
                  "image_source_binding":{"fixture":True}} for i,uid in enumerate(uids)]
        out.append({"img":torch.rand((2,3,64,64),generator=gen),"numeric_input":"float32_RGB_NCHW_divided_by_255_once",
                    "cls":torch.zeros((2,1)),"bboxes":torch.tensor([[.5,.5,.4,.5],[.4,.4,.3,.3]]),
                    "batch_idx":torch.tensor([0,1]),"ignore_bboxes":torch.empty((0,4)),"ignore_batch_idx":torch.empty((0,),dtype=torch.int64),
                    "uids":uids,"domains":["R","R"],"sample_lineage":lineage})
    return out


def same(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


class RuntimeBackendTests(unittest.TestCase):
    def test_actual_update_equals_manual_same_microbatch_reference_and_ema(self):
        left=backend()
        network=NativeDetectionNetwork(deepcopy(left.model),initialization_binding=deepcopy(left.network.initialization_binding))
        right=backend(network)
        # Compare all state including BN; reference uses original functional EMA primitive.
        data=batches()
        result=left.step_update(data,update=1)
        apply_sgd_schedule(right.optimizer,schedule_at(1,4,final_stage=True))
        right.optimizer.zero_grad(set_to_none=True)
        for batch in data:
            returned=right.criterion(right.network.training_heads(batch["img"]),batch)
            criterion_backward_loss(returned)["loss"].backward()
        torch.nn.utils.clip_grad_norm_(right._parameters,10.,error_if_nonfinite=True)
        right.optimizer.step()
        expected_ema=update_ema_state(right.ema_state,right.model.state_dict(),update=1)
        self.assertTrue(same(left.model.state_dict(),right.model.state_dict()))
        self.assertTrue(same(left.optimizer.state_dict(),right.optimizer.state_dict()))
        for key,value in left.ema_state["state_dict"].items():
            if value.is_floating_point():self.assertTrue(torch.allclose(value,expected_ema["state_dict"][key],rtol=1e-6,atol=1e-8),key)
            else:self.assertTrue(torch.equal(value,expected_ema["state_dict"][key]))
        self.assertEqual(result["images"],16);self.assertEqual(len(result["microbatches"]),8)
        self.assertEqual(len(result["sample_lineage"]),16)
        self.assertGreater(result["gradient_norm_before_clip"],0)
        self.assertLessEqual(result["gradient_norm_after_clip"],10.0001)
        self.assertTrue(result["requires_caller_sampler_commit"])
        self.assertEqual(left.ema_state["updates"],1)
        self.assertFalse(left.model.model[-1].dfl.conv.weight.requires_grad)

    def test_full_checkpoint_readback_fresh_restore_then_next_update_exact(self):
        original=backend()
        sampler=PrefetchSampler(CycleSampler([f"u{i}" for i in range(17)],task="detect",domain="R",seed=17))
        window=sampler.reserve(16)
        original.step_update(batches(),update=1);sampler.commit(window["token"])
        bindings={"fixture":"runtime_native_cpu"}
        bundle=original.snapshot(samplers={"R":sampler},source_bindings=bindings,committed_updates=1,
                                 cumulative_exposures={"R":16,"S":0},extra_state={"loader_generator":{"fixture":True}})
        with tempfile.TemporaryDirectory(dir=_TEMP.name) as t:
            receipt=save_checkpoint(Path(t)/"full.pt",bundle)
            loaded=load_checkpoint(receipt,expected_bindings=bindings)
        fresh=backend()
        next_sampler=PrefetchSampler(CycleSampler([f"u{i}" for i in range(17)],task="detect",domain="R",seed=17))
        out=fresh.restore_bundle(loaded,samplers={"R":next_sampler},expected_bindings=bindings)
        self.assertEqual(out["phase_updates"],1);self.assertTrue(out["RNG_restored_last"])
        self.assertTrue(same(original.optimizer.state_dict(),fresh.optimizer.state_dict()))
        self.assertTrue(same(sampler.state_dict(),next_sampler.state_dict()))
        rng=torch.get_rng_state().clone();expected=torch.rand(4);fresh2=backend()
        fresh2.restore_bundle(loaded,samplers={"R":PrefetchSampler(CycleSampler([f"u{i}" for i in range(17)],task="detect",domain="R",seed=17))},expected_bindings=bindings)
        self.assertTrue(torch.equal(expected,torch.rand(4)))
        a=original.step_update(batches(),update=2);b=fresh.step_update(batches(),update=2)
        self.assertTrue(same(original.model.state_dict(),fresh.model.state_dict()))
        self.assertTrue(same(original.ema_state,fresh.ema_state));self.assertTrue(same(original.optimizer.state_dict(),fresh.optimizer.state_dict()))
        self.assertEqual(a["loss"],b["loss"])

    def test_ema_eval_is_independent_and_preserves_raw_BN_and_parameters(self):
        value=backend();value.step_update(batches(),update=1)
        before=deepcopy(value.model.state_dict())
        ema=value.ema_network_for_eval()
        output=ema.evaluation_output(torch.zeros((2,3,64,64)))
        self.assertEqual(tuple(output[0].shape),(2,84,84))
        self.assertTrue(same(before,value.model.state_dict()))
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))
        self.assertFalse({id(p) for p in ema.model.parameters()} & {id(p) for p in value.model.parameters()})

    def test_incomplete_nonfinite_or_extra_micro_fails_without_silent_retry(self):
        for mode in ("short","extra","nan"):
            with self.subTest(mode=mode):
                value=backend();data=batches()
                if mode=="short":data=data[:-1]
                if mode=="extra":data=data+data[:1]
                if mode=="nan":data[0]["img"][0,0,0,0]=float("nan")
                with self.assertRaises(ValueError):value.step_update(data,update=1)
                self.assertTrue(value.failed);self.assertFalse(value.failure["optimizer_step_returned_not_committed"])
                self.assertEqual(value.optimizer_steps_attempted,0)
                with self.assertRaises(ValueError):value.step_update(batches(),update=1)
                with self.assertRaises(ValueError):value.snapshot(samplers={},source_bindings={},committed_updates=0,cumulative_exposures={"R":0,"S":0})

    def test_mode_and_restore_binding_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):backend(numeric_mode="amp_fp16")
        with self.assertRaises(ValueError):backend(numeric_mode="amp_bf16")
        with self.assertRaises(RuntimeError):backend(device="cuda",allow_initialized_cuda=False)
        value=backend()
        with self.assertRaises(ValueError):value.step_update(batches(),update=2)
        # A different stage budget cannot be substituted while restoring a trusted state.
        sample=PrefetchSampler(CycleSampler(["u"],task="detect",domain="R",seed=17))
        bundle=value.snapshot(samplers={"R":sample},source_bindings={"fixture":1},committed_updates=0,cumulative_exposures={"R":0,"S":0})
        wrong=NativeBBoxBackend(build_random_yolo11(model_size="n"),stage_updates=5,microbatch_size=2,input_size=64)
        with self.assertRaises(ValueError):wrong.restore_bundle(bundle,samplers={"R":sample},expected_bindings={"fixture":1})
        self.assertTrue(wrong.failed)

if __name__=="__main__":unittest.main()
