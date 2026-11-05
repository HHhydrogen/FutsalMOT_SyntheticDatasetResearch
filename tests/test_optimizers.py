"""微小nn.Module/SGD构造夹具；不执行forward/backward/optimizer.step。"""
import builtins
import importlib.util
import unittest
from unittest.mock import patch

import torch
from torch import nn

from futsalmot.optimizers import make_sgd_optimizer, apply_sgd_schedule
from futsalmot.training import schedule_at


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.ln = nn.LayerNorm(2)
        self.scalar = nn.Parameter(torch.tensor([3.]))
        self.dfl = nn.Linear(2, 1, bias=False)
        self.dfl.requires_grad_(False)


def make(model=None):
    return make_sgd_optimizer(Small() if model is None else model,
                              lr=.01, momentum=.937, weight_decay=.0005)


class OptimizerPreparationTests(unittest.TestCase):
    def setUp(self):
        for name in ("_lazy_init", "init", "is_available"):
            guard = patch.object(torch.cuda, name, side_effect=AssertionError("不探测/初始化CUDA"))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(torch.optim.SGD, "step", side_effect=AssertionError("优化器更新必须为0"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_unique_groups_frozen_exclusion_and_no_parameter_mutation(self):
        model = Small()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        optimizer = make(model)
        groups = {g["name"]: g for g in optimizer.param_groups}
        self.assertEqual(set(groups), {"bias", "weight", "norm"})
        self.assertEqual(set(groups["bias"]["parameter_names"]), {"conv.bias", "bn.bias", "ln.bias"})
        self.assertEqual(set(groups["norm"]["parameter_names"]), {"bn.weight", "ln.weight"})
        self.assertEqual(set(groups["weight"]["parameter_names"]), {"scalar", "conv.weight"})
        for name, group in groups.items():
            self.assertEqual(group["weight_decay"], .0005 if name == "weight" else 0)
            self.assertEqual(group["lr"], .01)
            self.assertEqual(group["momentum"], .937)
            self.assertTrue(group["nesterov"])
        identities = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(flags, {name: p.requires_grad for name, p in model.named_parameters()})
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_schedule_syncs_all_groups_without_step_or_state_change(self):
        optimizer = make()
        for step in (0, 1, 50, 1000):
            schedule = schedule_at(step, 1000, final_stage=True)
            result = apply_sgd_schedule(optimizer, schedule)
            self.assertEqual(result["step"], step)
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], schedule["lr"])
                self.assertEqual(group["momentum"], schedule["momentum"])
            self.assertEqual(len(optimizer.state), 0)

    def test_fresh_phase_optimizer_does_not_inherit_momentum(self):
        model = Small()
        old = make(model)
        p = old.param_groups[1]["params"][0]
        old.state[p]["momentum_buffer"] = torch.ones_like(p)  # 自造state，绝未step。
        new = make_sgd_optimizer(model, lr=.001, momentum=.937, weight_decay=.0005)
        self.assertEqual(len(old.state), 1)
        self.assertEqual(len(new.state), 0)
        self.assertEqual(new.param_groups[0]["lr"], .001)

    def test_shared_alias_and_undefined_contrastive_exception_are_refused(self):
        model = Small()
        model.alias = model.conv
        with self.assertRaisesRegex(ValueError, "共享|重复"):
            make(model)
        model = Small()
        model.register_parameter("another", model.scalar)
        with self.assertRaisesRegex(ValueError, "共享|重复"):
            make(model)
        model = Small()
        model.logit_scale = nn.Parameter(torch.tensor([1.]))
        with self.assertRaisesRegex(ValueError, "logit_scale"):
            make(model)

    def test_illegal_parameters_scalars_and_tampered_group_bindings(self):
        for model in (Small().half(), Small().double(), nn.Linear(1, 1, device="meta"), nn.LazyLinear(2)):
            with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):
                make(model)
        empty = Small().requires_grad_(False)
        with self.assertRaises(ValueError):
            make(empty)
        bad_scalars = [
            {"lr": float("nan")}, {"lr": -.1}, {"lr": True},
            {"momentum": 0}, {"momentum": 1}, {"weight_decay": -.1},
        ]
        for changed in bad_scalars:
            values = {"lr": .01, "momentum": .937, "weight_decay": .0005, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                make_sgd_optimizer(Small(), **values)
        optimizer = make()
        before = [g["lr"] for g in optimizer.param_groups]
        optimizer.param_groups[2]["weight_decay"] = .4
        with self.assertRaises(ValueError):
            apply_sgd_schedule(optimizer, schedule_at(1, 1000, final_stage=False))
        self.assertEqual(before, [g["lr"] for g in optimizer.param_groups])
        for schedule in ({"step": True, "updates": 100, "lr": .01, "momentum": .9},
                         {"step": 1, "updates": 100, "lr": .01, "momentum": float("inf")}, {}):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                apply_sgd_schedule(make(), schedule)
        foreign = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=.01, momentum=.9, nesterov=True)
        with self.assertRaisesRegex(ValueError, "绑定"):
            apply_sgd_schedule(foreign, schedule_at(1, 100, final_stage=True))

    def test_fake_cuda_property_rejected_before_any_initialization(self):
        class CudaTaggedParameter(nn.Parameter):
            @property
            def device(self):
                return torch.device("cuda:0")
        model = nn.Module()
        model.weight = CudaTaggedParameter(torch.ones(1))
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "初始化CUDA"):
                make(model)

    def test_module_import_is_lazy(self):
        import futsalmot.optimizers as module
        spec = importlib.util.spec_from_file_location("optimizer_lazy_fixture", module.__file__)
        loaded = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("import helper模块不应导入Torch")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            spec.loader.exec_module(loaded)


if __name__ == "__main__":
    unittest.main()
