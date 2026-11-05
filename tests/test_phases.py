"""微小CPU模块/手填旧状态的阶段边界检查；实际optimizer更新为0。"""
from copy import deepcopy
import importlib.util
import builtins
import unittest
from unittest.mock import patch

import torch
from torch import nn

from futsalmot.data import CycleSampler, PrefetchSampler
from futsalmot.optimizers import make_sgd_optimizer
from futsalmot.training import initialize_ema_stage, domain_at
from futsalmot.phases import start_final_phase, _boundary_counts


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(2,2)
        self.register_buffer("running",torch.tensor([4.]))
        self.register_buffer("count",torch.tensor(7,dtype=torch.int64))


def fixture(strategy="S->R",u_s=2,u_r=3):
    model=Tiny()
    optimizer=make_sgd_optimizer(model,lr=.01,momentum=.937,weight_decay=.0005)
    parameter=optimizer.param_groups[1]["params"][0]
    optimizer.state[parameter]["momentum_buffer"]=torch.ones_like(parameter)
    ema=initialize_ema_stage(model.state_dict())
    ema["updates"]=u_s  # 仅手填状态夹具，不执行任何optimizer.step。
    ema["state_dict"]["running"].fill_(-9)
    counts={"R":0,"S":0,"R_extra":0}
    for step in range(1,u_s+1):
        counts[domain_at(strategy,step,u_s,u_r)[0]]+=16
    samplers={}
    for stream in ("R","S","R_extra"):
        q=PrefetchSampler(CycleSampler(["s0","s1"] if stream=="S" else ["r0","r1"],
                                      task="detect",domain=stream,seed=17))
        if counts[stream]:
            q.commit(q.reserve(counts[stream])["token"])
        samplers[stream]=q
    kwargs=dict(strategy=strategy,u_s=u_s,u_r=u_r,committed_updates=u_s,
                current_phase="first",phase_updates=u_s,samplers=samplers,
                cumulative_exposures={"R":counts["R"]+counts["R_extra"],"S":counts["S"]},
                current_optimizer=optimizer,current_ema=ema)
    return model,kwargs,parameter


class PhaseBoundaryTests(unittest.TestCase):
    def setUp(self):
        for name in ("_lazy_init","init","is_available"):
            guard=patch.object(torch.cuda,name,side_effect=AssertionError("禁止CUDA初始化/探测"))
            guard.start();self.addCleanup(guard.stop)
        guard=patch.object(torch.optim.SGD,"step",side_effect=AssertionError("优化器更新须为0"))
        guard.start();self.addCleanup(guard.stop)

    def test_all_long_strategies_reset_only_new_optimizer_and_EMA(self):
        for strategy in ("S->R","R->R","R+S"):
            with self.subTest(strategy=strategy):
                model,kwargs,parameter=fixture(strategy)
                raw={k:v.detach().clone() for k,v in model.state_dict().items()}
                old_ema={k:v.clone() for k,v in kwargs["current_ema"]["state_dict"].items()}
                snapshots={k:q.state_dict() for k,q in kwargs["samplers"].items()}
                exposures=deepcopy(kwargs["cumulative_exposures"])
                old_momentum=kwargs["current_optimizer"].state[parameter]["momentum_buffer"].clone()
                result=start_final_phase(model,**kwargs)
                self.assertIsNot(result["optimizer"],kwargs["current_optimizer"])
                self.assertEqual(len(result["optimizer"].state),0)
                self.assertEqual(result["ema"]["updates"],0)
                for group in result["optimizer"].param_groups:
                    self.assertEqual(group["lr"],0)
                    self.assertEqual(group["momentum"],.8)
                    self.assertEqual(group["weight_decay"],.0005 if group["name"]=="weight" else 0)
                for key,value in model.state_dict().items():
                    self.assertTrue(torch.equal(value,raw[key]))
                    self.assertTrue(torch.equal(result["ema"]["state_dict"][key],raw[key]))
                    self.assertTrue(torch.equal(kwargs["current_ema"]["state_dict"][key],old_ema[key]))
                    self.assertNotEqual(result["ema"]["state_dict"][key].data_ptr(),value.data_ptr())
                self.assertEqual(result["ema"]["state_dict"]["count"].item(),7)
                self.assertTrue(torch.equal(kwargs["current_optimizer"].state[parameter]["momentum_buffer"],old_momentum))
                self.assertTrue(all(g["lr"]==.01 for g in kwargs["current_optimizer"].param_groups))
                self.assertEqual(snapshots,{k:q.state_dict() for k,q in kwargs["samplers"].items()})
                self.assertEqual(kwargs["cumulative_exposures"],exposures)
                self.assertEqual(result["metadata"]["cumulative_exposures"],exposures)
                self.assertFalse(result["metadata"]["samplers_reset"])
                self.assertFalse(result["metadata"]["lifecycle_committed"])
                with self.assertRaises(ValueError):
                    start_final_phase(model,**{**kwargs,"current_phase":"final","phase_updates":0})

    def test_exact_interleaved_first_prefix_formula_without_large_step_loop(self):
        for u_s in range(2,8):
            for u_r in range(2,9):
                expected={"R":0,"S":0,"R_extra":0}
                for j in range(1,u_s+1):
                    expected[domain_at("R+S",j,u_s,u_r)[0]]+=16
                self.assertEqual(_boundary_counts("R+S",u_s,u_r),expected)
        large=_boundary_counts("R+S",20000,6563)
        self.assertEqual(large["R"],16*(20000*6563//26563))
        self.assertEqual(large["R"]+large["S"],320000)

    def test_single_phase_early_late_repeat_and_bad_budget_are_refused(self):
        model,kwargs,_=fixture()
        changes=[{"strategy":"R"},{"strategy":"S"},{"committed_updates":1},{"committed_updates":3},
                 {"current_phase":"final"},{"phase_updates":1},{"phase_updates":True},{"u_s":0},{"u_r":1}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                start_final_phase(model,**{**kwargs,**change})

    def test_actual_live_pending_and_committed_counts_must_agree(self):
        model,kwargs,_=fixture()
        q=kwargs["samplers"]["R"]
        q.reserve(16)
        with self.assertRaisesRegex(ValueError,"pending"):
            start_final_phase(model,**kwargs)
        self.assertEqual(q.pending_batches,1)
        q.cancel_pending()
        q.commit(q.reserve(16)["token"])
        with self.assertRaisesRegex(ValueError,"曝光"):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        for changed in ({"R":1,"S":31},{"R":0,"S":16},{"R":False,"S":32}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                start_final_phase(model,**{**kwargs,"cumulative_exposures":changed})
        with self.assertRaisesRegex(ValueError,"PrefetchSampler"):
            start_final_phase(model,**{**kwargs,"samplers":{k:q.state_dict() for k,q in kwargs["samplers"].items()}})

    def test_identity_pool_seed_old_EMA_and_optimizer_bindings_refused(self):
        model,kwargs,_=fixture("R->R")
        replacement=PrefetchSampler(CycleSampler(["different"],task="detect",domain="R_extra",seed=17))
        replacement.commit(replacement.reserve(32)["token"])
        kwargs["samplers"]["R_extra"]=replacement
        with self.assertRaisesRegex(ValueError,"UID池"):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        kwargs["samplers"]["R"]=PrefetchSampler(CycleSampler(["r0","r1"],task="detect",domain="R",seed=29))
        with self.assertRaisesRegex(ValueError,"seed"):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        kwargs["current_ema"]["updates"]=1
        with self.assertRaisesRegex(ValueError,"EMA"):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        kwargs["current_ema"]["state_dict"].pop("count")
        with self.assertRaisesRegex(ValueError,"EMA"):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        foreign=make_sgd_optimizer(Tiny(),lr=.01,momentum=.937,weight_decay=.0005)
        with self.assertRaisesRegex(ValueError,"绑定当前raw"):
            start_final_phase(model,**{**kwargs,"current_optimizer":foreign})
        kwargs["current_optimizer"].param_groups[1]["weight_decay"]=.01
        with self.assertRaises(ValueError):
            start_final_phase(model,**kwargs)
        model,kwargs,_=fixture()
        with self.assertRaisesRegex(ValueError,"CPU"):
            start_final_phase(Tiny().to("meta"),**kwargs)

    def test_module_import_remains_lazy(self):
        import futsalmot.phases as module
        spec=importlib.util.spec_from_file_location("futsalmot._phase_lazy_fixture",module.__file__)
        loaded=importlib.util.module_from_spec(spec)
        original=builtins.__import__
        def guarded(name,*args,**kw):
            if name=="torch" or name.startswith("torch."):
                raise AssertionError("模块import不应导入Torch")
            return original(name,*args,**kw)
        with patch("builtins.__import__",guarded):
            spec.loader.exec_module(loaded)


if __name__=="__main__":unittest.main()
