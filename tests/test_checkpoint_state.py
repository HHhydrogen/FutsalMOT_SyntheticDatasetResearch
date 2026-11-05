"""一致CPU快照夹具；小张量/手填优化器状态，不建模型、不step或使用GPU。"""
from collections import OrderedDict
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

_ENV=os.environ.get("CUDA_VISIBLE_DEVICES");os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch
from futsalmot.checkpoint_state import snapshot_checkpoint,validate_checkpoint_bundle,prepare_restore_payload
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.records import capture_rng_state,save_checkpoint,load_checkpoint
from futsalmot.training import initialize_ema_stage

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


def streams(counts):
    out={}
    for domain,count in counts.items():
        q=PrefetchSampler(CycleSampler(["u","v","w"],task="detect",domain=domain,seed=17))
        if count:
            req=q.reserve(count);q.commit(req["token"])
        out[domain]=q
    return out


def arguments(counts=None,phase_updates=1,phase="final"):
    counts={"R":16,"S":16,"R_extra":16} if counts is None else counts
    raw=OrderedDict(weight=torch.tensor([1.234567,-2.],dtype=torch.float32,requires_grad=True),
                    counter=torch.tensor(7,dtype=torch.int64))
    raw._metadata={"":{"version":3}}
    ema=initialize_ema_stage(raw);ema["updates"]=phase_updates
    optimizer={"state":{0:{"momentum_buffer":torch.tensor([.123456789012345],dtype=torch.float32)}},
               "param_groups":[{"params":[0],"lr":.001,"momentum":.937}],
               "extra":("keep_all",[torch.tensor([3],dtype=torch.int64)])}
    r=counts.get("R",0)+counts.get("R_extra",0);s=counts.get("S",0)
    return dict(raw_state=raw,ema_state=ema,optimizer_state=optimizer,phase=phase,
                phase_updates=phase_updates,committed_updates=(r+s)//16,
                source_bindings={"data":"fixture_v3","model":"public_fixture_sha","code":"fixture_sha","protocol":"fixture_protocol"},
                numeric_mode="amp_bf16",scaler_state=None,samplers=streams(counts),
                cumulative_exposures={"R":r,"S":s},extra_state={"caller_extra":{"nested":[torch.tensor([.25])]}})


def metadata(bundle):
    return {key:deepcopy(bundle[key]) for key in ("phase","phase_updates","committed_updates","numeric_mode","cumulative_exposures","effective_batch")}


def validate(bundle,args,**changes):
    options=dict(expected_bindings=args["source_bindings"],expected_metadata=metadata(bundle),expected_samplers=args["samplers"])
    options.update(changes)
    validate_checkpoint_bundle(bundle,**options)


class CheckpointStateTests(unittest.TestCase):
    def test_CPU_snapshot_clones_every_state_preserves_dtype_metadata_and_pending_exclusion(self):
        a=arguments();pending=a["samplers"]["R"].reserve(16)
        bundle=snapshot_checkpoint(**a)
        self.assertEqual(bundle["cumulative_exposures"],{"R":32,"S":16})
        self.assertEqual(bundle["sampler"]["R"]["sampler"]["draws"],16)
        self.assertNotIn("pending",bundle["sampler"]["R"])
        self.assertNotIn("token",repr(bundle["sampler"]))
        self.assertEqual(a["samplers"]["R"].pending_batches,1)
        self.assertEqual(bundle["raw_model"]._metadata,{"":{"version":3}})
        before=bundle["raw_model"]["weight"].clone()
        with torch.no_grad():a["raw_state"]["weight"].fill_(99)
        a["ema_state"]["state_dict"]["weight"].fill_(98)
        a["optimizer_state"]["state"][0]["momentum_buffer"].fill_(97)
        a["optimizer_state"]["param_groups"][0]["lr"]=9
        a["raw_state"]._metadata[""]["version"]=999
        a["extra_state"]["caller_extra"]["nested"][0].fill_(96)
        self.assertTrue(torch.equal(bundle["raw_model"]["weight"],before))
        self.assertFalse(bundle["raw_model"]["weight"].requires_grad)
        self.assertEqual(bundle["optimizer"]["state"][0]["momentum_buffer"].dtype,torch.float32)
        self.assertEqual(bundle["optimizer"]["state"][0]["momentum_buffer"].item(),torch.tensor(.123456789012345,dtype=torch.float32).item())
        self.assertEqual(bundle["optimizer"]["extra"][1][0].dtype,torch.int64)
        self.assertEqual(bundle["optimizer"]["extra"][1][0].item(),3)
        self.assertEqual(bundle["optimizer"]["param_groups"][0]["lr"],.001)
        self.assertEqual(bundle["raw_model"]._metadata[""]["version"],3)
        self.assertEqual(bundle["caller_extra"]["nested"][0].item(),.25)
        self.assertEqual(pending["samples"][0]["exposure_index"],16)

    def test_save_load_candidate_payload_RNG_sampler_continuation_no_live_state_mutation(self):
        a=arguments();pending=a["samplers"]["R"].reserve(16);bundle=snapshot_checkpoint(**a)
        before_rng=capture_rng_state();sampler_before=a["samplers"]["R"].state_dict()
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"fixture.pt";receipt=save_checkpoint(path,bundle)
            loaded=load_checkpoint(receipt,expected_bindings=a["source_bindings"])
            payload=prepare_restore_payload(loaded,expected_bindings=a["source_bindings"],
                                            expected_metadata=metadata(bundle),expected_samplers=a["samplers"])
            self.assertEqual(payload["rng"]["python"],bundle["rng"]["python"])
            self.assertEqual(payload["rng"]["numpy"],bundle["rng"]["numpy"])
            self.assertTrue(torch.equal(payload["rng"]["torch_cpu"],bundle["rng"]["torch_cpu"]))
            self.assertEqual(payload["sampler"],bundle["sampler"])
            self.assertEqual(payload["raw_model"]._metadata,bundle["raw_model"]._metadata)
            candidate=CycleSampler(["u","v","w"],task="detect",domain="R",seed=17)
            candidate.load_state_dict(payload["sampler"]["R"]["sampler"])
            self.assertEqual(candidate.take(16),pending["samples"])
            payload["raw_model"]["weight"].fill_(-1)
            self.assertFalse(torch.equal(payload["raw_model"]["weight"],loaded["raw_model"]["weight"]))
        self.assertFalse(path.exists())
        self.assertEqual(a["samplers"]["R"].state_dict(),sampler_before)
        self.assertEqual(a["samplers"]["R"].pending_batches,1)
        after_rng=capture_rng_state()
        self.assertEqual(before_rng["python"],after_rng["python"])
        self.assertEqual(before_rng["numpy"],after_rng["numpy"])
        self.assertTrue(torch.equal(before_rng["torch_cpu"],after_rng["torch_cpu"]))

    def test_stage_zero_accepts_carried_global_history_and_resets_EMA(self):
        a=arguments({"S":32,"R":0},phase_updates=0)
        b=snapshot_checkpoint(**a)
        self.assertEqual((b["phase_updates"],b["committed_updates"],b["ema"]["updates"]),(0,2,0))
        self.assertTrue(torch.equal(b["raw_model"]["weight"],b["ema"]["state_dict"]["weight"]))
        zero=snapshot_checkpoint(**arguments({"R":0},phase_updates=0,phase="first"))
        self.assertEqual(zero["cumulative_exposures"],{"R":0,"S":0})
        a["ema_state"]["state_dict"]["weight"].add_(1)
        with self.assertRaisesRegex(ValueError,"阶段0"):snapshot_checkpoint(**a)

    def test_dtype_EMA_counters_exposure_and_sampler_boundary_mismatch_rejected(self):
        changes=[lambda a:a.update(phase_updates=4),
                 lambda a:a["ema_state"].update(updates=0),
                 lambda a:a.update(committed_updates=4),
                 lambda a:a["cumulative_exposures"].update(R=31),
                 lambda a:a.update(raw_state={"weight":torch.tensor([1.],dtype=torch.float16)}),
                 lambda a:a["ema_state"]["state_dict"].pop("counter"),
                 lambda a:a.update(numeric_mode="amp_bf16",scaler_state={"scale":2.}),
                 lambda a:a.update(numeric_mode="amp_fp16",scaler_state=None),
                 lambda a:a.update(samplers={"unknown":next(iter(a["samplers"].values()))}),
                 lambda a:a.update(samplers={"R":next(iter(a["samplers"].values())).state_dict()}),
                 lambda a:a.update(extra_state={"rng":{}})]
        for change in changes:
            a=arguments();change(a)
            with self.assertRaises(ValueError):snapshot_checkpoint(**a)
        with self.assertRaisesRegex(ValueError,"16图"):snapshot_checkpoint(**arguments({"R":8,"S":8}))
        with mock.patch("futsalmot.checkpoint_state.capture_rng_state",return_value={}):
            with self.assertRaisesRegex(ValueError,"RNG"):snapshot_checkpoint(**arguments())

    def test_restore_requires_actual_run_UID_pool_task_seed_and_trusted_metadata(self):
        a=arguments();b=snapshot_checkpoint(**a)
        wrong=dict(a["samplers"])
        wrong["R"]=PrefetchSampler(CycleSampler(["other_UID"],task="detect",domain="R",seed=17))
        with self.assertRaisesRegex(ValueError,"UID池"):validate(b,a,expected_samplers=wrong)
        with self.assertRaises(ValueError):validate(b,a,expected_bindings={"data":"wrong"})
        meta=metadata(b);meta["phase_updates"]=0
        with self.assertRaisesRegex(ValueError,"metadata"):validate(b,a,expected_metadata=meta)
        bad=deepcopy(b);bad["sampler"]["R"]["pending"]={"token":"must_not_restore"}
        with self.assertRaisesRegex(ValueError,"pending"):validate(bad,a)
        bad=deepcopy(b);bad["rng"].pop("torch_cpu")
        with self.assertRaisesRegex(ValueError,"RNG"):validate(bad,a)
        bad=deepcopy(b);bad["sampler"]["R"]["sampler"]["draws"]+=16
        with self.assertRaises(ValueError):validate(bad,a)

    def test_optimizer_must_be_finite_FP32_without_silent_cast(self):
        for label,value in [("FP16",torch.tensor([1.],dtype=torch.float16)),
                            ("FP64",torch.tensor([1.],dtype=torch.float64)),
                            ("NaN_tensor",torch.tensor([float("nan")],dtype=torch.float32))]:
            with self.subTest(label=label):
                a=arguments();a["optimizer_state"]["state"][0]["momentum_buffer"]=value
                with self.assertRaisesRegex(ValueError,"optimizer"):snapshot_checkpoint(**a)
        for value in (float("nan"),float("inf")):
            with self.subTest(hyperparameter=value):
                a=arguments();a["optimizer_state"]["param_groups"][0]["lr"]=value
                with self.assertRaisesRegex(ValueError,"optimizer"):snapshot_checkpoint(**a)
        # 读回准入也必须拒绝；不能只检查builder路径。
        a=arguments();b=snapshot_checkpoint(**a)
        b["optimizer"]["state"][0]["momentum_buffer"]=torch.tensor([1.],dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,"optimizer"):validate(b,a)

    def test_explicit_CUDA_context_requires_complete_CUDA_RNG_no_GPU_call(self):
        a=arguments()
        for permission,initialized,include in [(True,True,False),(True,False,False),(False,True,False)]:
            with self.subTest(flags=(permission,initialized,include)):
                with self.assertRaisesRegex(ValueError,"CUDA"):
                    snapshot_checkpoint(**a,allow_cuda=permission,cuda_initialized=initialized,include_cuda_rng=include)
        b=snapshot_checkpoint(**a)
        with self.assertRaisesRegex(ValueError,"CUDA RNG"):
            validate(b,a,allow_cuda=True,cuda_initialized=True)

    def test_default_snapshot_does_not_access_CUDA_and_requires_explicit_permission(self):
        class NoCuda:
            def __getattr__(self,name):raise AssertionError("默认CPU分支触及CUDA: "+name)
        a=arguments()
        with mock.patch.object(torch,"cuda",NoCuda()):
            snapshot_checkpoint(**a)
            with self.assertRaisesRegex(ValueError,"CUDA"):snapshot_checkpoint(**a,include_cuda_rng=True)
        fp16=arguments();fp16.update(numeric_mode="amp_fp16",scaler_state={"scale":65536.,"growth_tracker":0})
        self.assertEqual(snapshot_checkpoint(**fp16)["scaler"]["scale"],65536.)

if __name__=="__main__":unittest.main()
