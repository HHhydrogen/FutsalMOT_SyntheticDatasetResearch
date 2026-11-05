"""完整检查点CPU检查；临时权重不留存，不执行优化器step或GPU操作。"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from futsalmot.records import (
    CHECKPOINT_SCHEMA, capture_rng_state, load_checkpoint, restore_rng_state,
    save_checkpoint, validate_checkpoint_contract,
)


def state_fixture():
    return dict(schema_version=CHECKPOINT_SCHEMA, raw_model={}, ema={}, optimizer={},
                rng={}, sampler={"position":7,"epoch":2,"rng":[1,2,3]}, phase="final",
                phase_updates=50, committed_updates=150,
                source_bindings={"dataset":"fixture:sha","model":"public_start_sha","code":"fixture_code"},
                numeric_mode="amp_bf16",scaler=None, cumulative_exposures={"R":800,"S":1600})


class CheckpointContractTests(unittest.TestCase):
    def test_reject_bad_fields_before_any_torch_import(self):
        base=state_fixture()
        bad=[]
        missing=dict(base);missing.pop("rng");bad.append(missing)
        for key,value in [("schema_version",None),("phase",""),("phase_updates",True),
                          ("phase_updates",1.5),("committed_updates",-1),("raw_model",object()),
                          ("source_bindings",{}),("numeric_mode","auto"),("scaler",{})]:
            bad.append({**base,key:value})
        with tempfile.TemporaryDirectory() as t, mock.patch("futsalmot.records._torch",side_effect=AssertionError("must validate before import")):
            for candidate in bad:
                with self.assertRaises(ValueError):save_checkpoint(Path(t)/"bad.pt",candidate)
            self.assertEqual(list(Path(t).iterdir()),[])
        validate_checkpoint_contract({**base,"numeric_mode":"amp_fp16","scaler":{}})

    def test_partial_write_and_publish_failure_leave_no_temp_files(self):
        class BrokenTorch:
            @staticmethod
            def is_tensor(value):return False
            @staticmethod
            def save(state,stream):
                stream.write(b"incomplete-checkpoint")
                raise OSError("fixture interrupted write")
        class TinyTorch(BrokenTorch):
            @staticmethod
            def save(state,stream):stream.write(b"complete-fixture-only")
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"checkpoint.pt"
            with mock.patch("futsalmot.records._torch",return_value=BrokenTorch):
                with self.assertRaisesRegex(OSError,"interrupted"):save_checkpoint(path,state_fixture())
            self.assertEqual(list(Path(t).iterdir()),[])
            with mock.patch("futsalmot.records._torch",return_value=TinyTorch), mock.patch("futsalmot.records.os.link",side_effect=OSError("fixture publication failure")):
                with self.assertRaisesRegex(OSError,"publication"):save_checkpoint(path,state_fixture())
            self.assertEqual(list(Path(t).iterdir()),[])

    def test_no_overwrite_and_existing_parent_required(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"existing.pt";path.write_bytes(b"original_evidence")
            with mock.patch("futsalmot.records._torch",side_effect=AssertionError("no import")):
                with self.assertRaises(FileExistsError):save_checkpoint(path,state_fixture())
                with self.assertRaises(FileNotFoundError):save_checkpoint(Path(t)/"absent"/"new.pt",state_fixture())
            self.assertEqual(path.read_bytes(),b"original_evidence")
            self.assertEqual([p.name for p in Path(t).iterdir()],["existing.pt"])


HAVE_TORCH_NUMPY = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAVE_TORCH_NUMPY,"Torch/NumPy尚未安装：真实序列化与RNG用例待环境完成")
class TorchCheckpointTests(unittest.TestCase):
    def test_complete_cpu_state_and_precision_roundtrip_without_step(self):
        import torch
        p=torch.tensor([1.23456789012345,-2.],dtype=torch.float64,requires_grad=True,device="cpu")
        optimizer=torch.optim.SGD([p],lr=.001,momentum=.937)
        # 直接设置已有状态夹具，不调用backward/step、不训练。
        optimizer.state[p]["momentum_buffer"]=torch.tensor([.125,.25],dtype=torch.float64)
        state=state_fixture()
        state.update(raw_model={"weight":p.detach().clone(),"counter":torch.tensor(150)},
                     ema={"weight":torch.tensor([1.1,2.2],dtype=torch.float32),"updates":50},
                     optimizer=optimizer.state_dict(),rng=capture_rng_state(),
                     auxiliary_state={"loss_scale":.123456789012345,"meaning":"extra_fields_preserved"})
        original=p.detach().clone()
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"update_000150.pt"
            receipt=save_checkpoint(path,state)
            loaded=load_checkpoint(receipt,expected_bindings=state["source_bindings"])
            self.assertEqual(set(loaded),set(state))
            for key in ["sampler","source_bindings","cumulative_exposures","auxiliary_state","phase","phase_updates","committed_updates","numeric_mode","scaler"]:
                self.assertEqual(loaded[key],state[key])
            self.assertEqual(loaded["raw_model"]["weight"].dtype,torch.float64)
            self.assertEqual(loaded["ema"]["weight"].dtype,torch.float32)
            self.assertTrue(torch.equal(loaded["raw_model"]["weight"],original))
            self.assertTrue(torch.equal(loaded["optimizer"]["state"][0]["momentum_buffer"],optimizer.state[p]["momentum_buffer"]))
            self.assertEqual(loaded["optimizer"]["param_groups"],state["optimizer"]["param_groups"])
            self.assertEqual(loaded["raw_model"]["weight"].device.type,"cpu")
            self.assertEqual(loaded["rng"]["python"],state["rng"]["python"])
            self.assertEqual(loaded["rng"]["numpy"],state["rng"]["numpy"])
            self.assertTrue(torch.equal(loaded["rng"]["torch_cpu"],state["rng"]["torch_cpu"]))
            self.assertEqual([x.name for x in Path(t).iterdir()],[path.name])
        self.assertFalse(path.exists())
        self.assertTrue(torch.equal(p.detach(),original))

    def test_cpu_rng_continuation_and_default_never_accesses_cuda(self):
        import numpy as np
        import torch
        class NoCuda:
            def __getattr__(self,name):raise AssertionError("CPU路径访问CUDA: "+name)
        before=capture_rng_state()
        try:
            random.seed(17);np.random.seed(17);torch.random.default_generator.manual_seed(17)
            with mock.patch.object(torch,"cuda",NoCuda()):
                checkpoint=capture_rng_state()
                expected=(random.random(),np.random.random(4),torch.rand(4,device="cpu"))
                random.random();np.random.random(9);torch.rand(9,device="cpu")
                restore_rng_state(checkpoint)
                actual=(random.random(),np.random.random(4),torch.rand(4,device="cpu"))
            self.assertEqual(expected[0],actual[0])
            self.assertTrue(np.array_equal(expected[1],actual[1]))
            self.assertTrue(torch.equal(expected[2],actual[2]))
            with self.assertRaises(ValueError):restore_rng_state({**checkpoint,"cuda":[torch.tensor([1],dtype=torch.uint8)]})
        finally:
            restore_rng_state(before)

    def test_binding_digest_path_and_corruption_rejected_before_load(self):
        import torch
        state=state_fixture();state["raw_model"]={"weight":torch.tensor([1.],device="cpu")}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"checkpoint.pt"
            receipt=save_checkpoint(path,state)
            with mock.patch("futsalmot.records._torch",side_effect=AssertionError("must reject before unpickle")):
                with self.assertRaises(ValueError):load_checkpoint(receipt,expected_bindings={"other":"source"})
                with self.assertRaises(ValueError):load_checkpoint(replace(receipt,path="relative.pt"),expected_bindings=state["source_bindings"])
                content=path.read_bytes();path.write_bytes(content[:-1]+bytes([content[-1]^1]))
                with self.assertRaisesRegex(ValueError,"SHA"):load_checkpoint(receipt,expected_bindings=state["source_bindings"])
            with self.assertRaises(FileExistsError):save_checkpoint(path,state)

if __name__ == "__main__":unittest.main()
