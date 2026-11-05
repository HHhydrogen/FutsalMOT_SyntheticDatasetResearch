"""PREP-012有限CPU fixture生命周期；临时run全部删除，真实参数更新0。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from futsalmot.lifecycle import run_cpu_fixture,CPUFixtureBackend,FixtureBackendFailure,SCOPE
import futsalmot.lifecycle as lifecycle
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.execution import _reservation_requests
from futsalmot.dataset import RequestBBoxDataset
from futsalmot.records import CheckpointReceipt,load_checkpoint,RunRecorder
from futsalmot.training import train

_OWN="test_dataset" not in sys.modules
fixture_module=importlib.import_module("test_dataset")
_THREAD=None
_BLOCKS=[]

def setUpModule():
    global _THREAD
    _THREAD=torch.get_num_threads();torch.set_num_threads(2)
    for obj,name in [(torch.cuda,"_lazy_init"),(torch.cuda,"is_available"),(torch.cuda,"init"),(torch.accelerator,"is_available"),(torch.optim.SGD,"step")]:
        guard=patch.object(obj,name,side_effect=AssertionError("fixture不得GPU或optimizer.step"));guard.start();_BLOCKS.append(guard)

def tearDownModule():
    for guard in reversed(_BLOCKS):guard.stop()
    if _THREAD is not None:torch.set_num_threads(_THREAD)
    if _OWN:fixture_module.tearDownModule()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[1.]]));self.bias=torch.nn.Parameter(torch.tensor([0.]))
        self.register_buffer("count",torch.tensor(3,dtype=torch.int64))
    def forward(self,*a,**k):raise AssertionError("fixture不得forward")


def measured():
    return dict(loss=0.,loss_components={"box":0.,"cls":0.,"dfl":0.},
                loss_definition=SCOPE+": explicit fixture values; no measured model loss",
                backward_loss=0.,backward_components={"box":0.,"cls":0.,"dfl":0.},
                gradient_norm_before_clip=0.,gradient_norm_after_clip=0.,elapsed_seconds=0.)


def streams(strategy,r="r",s="s"):
    names=("R",) if strategy=="R" else ("S",) if strategy=="S" else ("R","R_extra") if strategy=="R->R" else ("R","S")
    return {name:PrefetchSampler(CycleSampler([s if name=="S" else r],task="detect",domain=name,seed=17)) for name in names}


class MetadataLoader:
    def __init__(self,root):self.root=Path(root);self.windows=[]
    def __call__(self,reservation,microbatch_size,generator):
        token,stream,domain,requests=_reservation_requests(reservation,microbatch_size)
        self.windows.append((token,stream))
        for offset in range(0,16,microbatch_size):
            group=requests[offset:offset+microbatch_size]
            reference={"domain":domain,"split":"train"}
            if domain=="R":reference["index_path"]=str(self.root/"metadata_fixture.json")
            else:reference.update(seed_dir=str(self.root/"seed_00000001"),camera_id="C5",frame_index=1)
            lineage=[{"request":deepcopy(req),"source_reference":deepcopy(reference),
                      "image_source_binding":{"path":str(self.root/"metadata_fixture.png"),"stat":(1,2,3,4)},
                      "sampling_stream":stream,"data_domain":domain} for req in group]
            yield {"reservation_token":token,"effective_batch_size":16,"microbatch_offset":offset,
                   "uids":[r["uid"] for r in group],"domains":[domain]*len(group),"sample_lineage":lineage}


def kwargs(root,strategy="R",**extra):
    result=dict(directory=Path(root)/"run",raw_model=Tiny(),samplers=streams(strategy),strategy=strategy,training_seed=17,
                n_s=1,n_r=1,epochs_s=32,epochs_r=32,microbatch_size=16,
                generator=torch.Generator(device="cpu").manual_seed(17),backend=CPUFixtureBackend(measured()),
                source_bindings={"origin":"temporary_cpu_fixture_only"},max_updates=2048,evaluation_basis="phase",
                loader_factory=MetadataLoader(root))
    result.update(extra);return result


def events(path):return [json.loads(line) for line in (Path(path)/"events.jsonl").read_text(encoding="utf8").splitlines()]


class LifecycleTests(unittest.TestCase):
    def test_short_real_loader_long_strategy_full_records_reset_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);catalogue,_,_=fixture_module.fixture(root/"data")
            q=streams("R+S",fixture_module.R_UID,fixture_module.S_UID);observations=[];evaluations=[]
            def measurement(context,traces):
                self.assertTrue(all(g["lr"]==context["schedule"]["lr"] for g in context["groups"]))
                self.assertTrue(all(g["momentum"]==context["schedule"]["momentum"] for g in context["groups"]))
                observations.append((context["update"],len(traces)));return measured()
            args=kwargs(root,"R+S",samplers=q,microbatch_size=4,loader_factory=None,dataset=RequestBBoxDataset(catalogue),
                        backend=CPUFixtureBackend(measurement),evaluation_callback=lambda value:evaluations.append((value["phase"],value["phase_updates"],value["model_role"])))
            raw={k:v.clone() for k,v in args["raw_model"].state_dict().items()}
            result=run_cpu_fixture(**args)
            self.assertEqual(result["status"],"CPU_FIXTURE_COMPLETE");self.assertEqual(result["actual_parameter_updates"],0)
            self.assertEqual(observations,[(1,4),(2,4),(3,4),(4,4)])
            self.assertEqual(evaluations,[("first",0,"stage0_reference"),("first",2,"terminal_EMA"),("final",2,"terminal_EMA")])
            self.assertEqual(len(result["checkpoints"]),4)
            rows=[json.loads(x) for x in (args["directory"]/"steps.jsonl").read_text().splitlines()]
            self.assertEqual([r["phase_update"] for r in rows],[1,2,1,2])
            self.assertTrue(all(r["execution_scope"]==SCOPE and r["actual_parameter_updates"]==0 for r in rows))
            self.assertEqual([sum(q.committed_draws for key,q in args["samplers"].items() if key==d) for d in ("R","S")],[32,32])
            manifest=json.loads((args["directory"]/"run.json").read_text())
            for receipt in result["checkpoints"]:
                bundle=load_checkpoint(CheckpointReceipt(**receipt),expected_bindings=manifest["source_bindings"])
                self.assertEqual(bundle["execution_scope"],SCOPE)
                self.assertEqual(bundle["actual_parameter_updates"],0)
                if bundle["phase"]=="final" and bundle["phase_updates"]==0:
                    self.assertEqual(bundle["committed_updates"],2);self.assertEqual(bundle["ema"]["updates"],0)
                    self.assertEqual(bundle["optimizer"]["state"],{})
            for key,value in args["raw_model"].state_dict().items():self.assertTrue(torch.equal(value,raw[key]))
            self.assertEqual(train({})["status"],"BLOCKED")
            self.assertTrue((args["directory"]/"stdout.log").exists());self.assertTrue((args["directory"]/"stderr.log").exists())

    def test_metadata_1050_boundaries_50_final_dedup_and_phase1000_eval(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);evaluations=[]
            args=kwargs(root,"R->R",epochs_s=800,epochs_r=16000,
                        evaluation_callback=lambda v:evaluations.append((v["committed_updates"],v["phase_updates"],v["reasons"])))
            result=run_cpu_fixture(**args)
            self.assertEqual(result["fixture_committed_updates"],1050)
            self.assertEqual(result["cumulative_exposures"],{"R":16800,"S":0})
            self.assertEqual(len(result["checkpoints"]),23) # first0/50 + final0/50...1000。
            self.assertEqual(evaluations,[(0,0,["PT_reference_fixture_only"]),(50,50,["stage_final"]),(1050,1000,["interval1000","stage_final"])])
            points=[(v["phase"],v["phase_updates"],v["committed_updates"]) for v in events(args["directory"]) if v["event"]=="checkpoint_saved"]
            self.assertEqual(len(points),len(set(points)))
            self.assertIn(("first",50,50),points);self.assertIn(("final",0,50),points)

    def test_metadata_evaluation_role_distinguishes_periodic_from_terminal_without_retiming(self):
        # 原表达式metadata复现已证实1000且非末步曾误标terminal_EMA；只改标签。
        self.assertEqual(lifecycle._evaluation_role(0,["PT_reference_fixture_only"]),"stage0_reference")
        self.assertEqual(lifecycle._evaluation_role(1000,["interval1000"]),"EMA_at_update")
        self.assertEqual(lifecycle._evaluation_role(1001,["stage_final"]),"terminal_EMA")
        self.assertEqual(lifecycle._evaluation_role(1000,["interval1000","stage_final"]),"terminal_EMA")

    def test_backend_failure_and_manual_cancel_stop_without_commit_or_second_dispatch(self):
        for cancel in (False,True):
            with tempfile.TemporaryDirectory() as t:
                backend=CPUFixtureBackend(measured(),**({"cancel_update":1} if cancel else {"fail_update":1}))
                args=kwargs(t,backend=backend)
                with self.assertRaises(KeyboardInterrupt if cancel else FixtureBackendFailure):run_cpu_fixture(**args)
                self.assertEqual(args["samplers"]["R"].committed_draws,0)
                self.assertEqual(args["samplers"]["R"].pending_batches,1)
                saved=events(args["directory"])
                self.assertEqual(len([x for x in saved if x["event"]=="update_started"]),1)
                stop=saved[-1];self.assertEqual(stop["event"],"fixture_run_stopped")
                self.assertFalse(stop["automatic_resume_allowed"]);self.assertFalse(stop["retry_performed"])
                self.assertFalse((args["directory"]/"steps.jsonl").exists())

    def test_commit_advanced_then_raised_is_observed_as_inconsistent_without_replay(self):
        with tempfile.TemporaryDirectory() as t:
            args=kwargs(t);q=args["samplers"]["R"];original=q.commit
            def interrupted(token):
                original(token)
                raise OSError("fixture commit advanced before return")
            with patch.object(q,"commit",interrupted):
                with self.assertRaisesRegex(OSError,"advanced before return"):run_cpu_fixture(**args)
            self.assertEqual(q.committed_draws,16)
            stop=events(args["directory"])[-1]
            self.assertTrue(stop["inconsistent_boundary"])
            self.assertEqual(stop["observed_sampler_exposures"],{"R":16,"S":0})
            self.assertEqual(stop["observed_sampler_committed_updates"],1)
            self.assertFalse(stop["automatic_resume_allowed"])
            self.assertEqual(len(args["loader_factory"].windows),1)
            self.assertFalse((args["directory"]/"steps.jsonl").exists())

    def test_STOP_marks_sampler_observation_failure_unknown_without_local_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            args=kwargs(t);q=args["samplers"]["R"];commit=q.commit;state=q.state_dict;failed=False
            def interrupted(token):
                nonlocal failed
                commit(token);failed=True
                raise OSError("fixture commit exception")
            def observe():
                if failed:raise RuntimeError("fixture observation unavailable")
                return state()
            with patch.object(q,"commit",interrupted),patch.object(q,"state_dict",observe):
                with self.assertRaisesRegex(OSError,"commit exception"):run_cpu_fixture(**args)
            stop=events(args["directory"])[-1]
            self.assertTrue(stop["observed_unknown"]);self.assertTrue(stop["inconsistent_boundary"])
            self.assertIsNone(stop["observed_sampler_exposures"])
            self.assertIsNone(stop["observed_sampler_committed_updates"])
            self.assertEqual(stop["observed_sampler_draws"],{"R":None})
            self.assertFalse(stop["automatic_resume_allowed"])
            self.assertIn("R",stop["sampler_observation_errors"])

    def test_record_IO_partial_after_commit_is_persistently_inconsistent_not_replayed(self):
        with tempfile.TemporaryDirectory() as t:
            args=kwargs(t)
            def broken(recorder,record):
                recorder._failed=True
                with (recorder.directory/"steps.jsonl").open("xb") as f:f.write(b'{"update":');f.flush()
                raise OSError("fixture partial record I/O")
            with patch.object(RunRecorder,"step",broken):
                with self.assertRaisesRegex(OSError,"partial record"):run_cpu_fixture(**args)
            self.assertEqual(args["samplers"]["R"].committed_draws,16)
            stop=json.loads((args["directory"]/"STOP.json").read_text())
            self.assertEqual((stop["committed_updates"],stop["recorded_updates"]),(1,0))
            self.assertTrue(stop["inconsistent_boundary"]);self.assertFalse(stop["automatic_resume_allowed"])
            self.assertEqual((args["directory"]/"steps.jsonl").read_bytes(),b'{"update":')
            manifest=json.loads((args["directory"]/"run.json").read_text())
            with self.assertRaises(ValueError):RunRecorder.resume(args["directory"],expected_manifest=manifest,checkpoint_committed_updates=0)
            self.assertEqual(len(args["loader_factory"].windows),1)

    def test_explicit_basis_and_fixture_only_gates_before_run_creation(self):
        with tempfile.TemporaryDirectory() as t:
            args=kwargs(t,evaluation_basis=None)
            with self.assertRaisesRegex(ValueError,"basis"):run_cpu_fixture(**args)
            self.assertFalse(args["directory"].exists())
            args=kwargs(t,backend=lambda:True)
            with self.assertRaisesRegex(ValueError,"backend"):run_cpu_fixture(**args)
            self.assertFalse(args["directory"].exists())
            args=kwargs(t,microbatch_size=3)
            with self.assertRaisesRegex(ValueError,"microbatch"):run_cpu_fixture(**args)
            self.assertFalse(args["directory"].exists())
            args=kwargs(t,generator=None)
            with self.assertRaises(ValueError):run_cpu_fixture(**args)
            self.assertFalse(args["directory"].exists())
            args=kwargs(t,evaluation_basis="global")
            result=run_cpu_fixture(**args)
            self.assertEqual(result["evaluation_basis"],"global")

if __name__=="__main__":unittest.main()
