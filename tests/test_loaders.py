"""真实DataLoader0worker有限窗口CPU夹具；临时图/GT自动清理，无模型/step。"""
from copy import deepcopy
import importlib
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from futsalmot.loaders import ReservationDataLoader,loader_generator_state,restore_loader_generator_state,collate_reserved_window
from futsalmot.execution import iter_reserved_microbatches
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.dataset import RequestBBoxDataset
import futsalmot.dataset as dataset_module

_OWN_FIXTURE="test_dataset" not in sys.modules
fixture_module=importlib.import_module("test_dataset")
_THREADS=None
_CV_THREADS=None
_BLOCKS=[]

def setUpModule():
    global _THREADS,_CV_THREADS
    _THREADS=torch.get_num_threads();torch.set_num_threads(2)
    _CV_THREADS=fixture_module.cv2.getNumThreads();fixture_module.cv2.setNumThreads(2)
    for obj,name in [(torch.cuda,"_lazy_init"),(torch.cuda,"init"),(torch.cuda,"is_available"),(torch.accelerator,"is_available")]:
        guard=patch.object(obj,name,side_effect=AssertionError("0worker CPU不得初始化/探测GPU"));guard.start();_BLOCKS.append(guard)

def tearDownModule():
    for guard in reversed(_BLOCKS):guard.stop()
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    if _CV_THREADS is not None:fixture_module.cv2.setNumThreads(_CV_THREADS)
    if _OWN_FIXTURE:fixture_module.tearDownModule()


def sampler(stream="R",uids=None):return PrefetchSampler(CycleSampler(uids or ["a","b"],task="detect",domain=stream,seed=17))

def generator():
    # seed由本夹具caller明确提供，loader接口本身不设seed。
    return torch.Generator(device="cpu").manual_seed(123)


class TinyRequestDataset:
    def __init__(self,missing_lineage=False):self.calls=[];self.missing_lineage=missing_lineage
    def __getitem__(self,request):
        self.calls.append(deepcopy(request))
        domain=request["data_domain"];root=Path(__file__).absolute().parent
        reference={"domain":domain,"split":"train"}
        if domain == "R":reference["index_path"]=str(root/"never_opened.json")
        else:reference.update(seed_dir=str(root/"never_opened_seed"),camera_id="C5",frame_index=1)
        value={"img":torch.zeros((3,960,960),dtype=torch.uint8),"uid":request["uid"],"domain":domain,
               "uid_field":"sample_uid" if domain=="R" else "derived_sample_uid",
               "geometry":{"input_shape":(960,960)},"targets":[],"ignore_targets":[],"absent_targets":[],
               "cls":torch.empty((0,1)),"bboxes":torch.empty((0,4)),"ignore_bboxes":torch.empty((0,4)),
               "request":deepcopy(request),"source_reference":reference,
               "image_source_binding":{"path":str(root/"never_opened.png"),"stat":(1,2,100,3)},
               "sampling_stream":request["stream"],"data_domain":domain}
        if self.missing_lineage:
            for key in ("request","source_reference","image_source_binding","sampling_stream","data_domain"):value.pop(key)
        return value


class ReservationLoaderTests(unittest.TestCase):
    def test_real_two_windows_32_files_dataset_collate_trace_no_advance_and_cancel_replay(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,_,_=fixture_module.fixture(Path(t));dataset=RequestBBoxDataset(catalogue)
            q=sampler(uids=[fixture_module.R_UID]);first,second=q.reserve(16),q.reserve(16)
            source=deepcopy([first,second]);g=generator();global_before=torch.get_rng_state().clone()
            with patch("futsalmot.dataset.decode_rgb_image",wraps=dataset_module.decode_rgb_image) as decode:
                window=ReservationDataLoader([first,second],dataset,microbatch_size=4,generator=g)
                self.assertEqual(decode.call_count,0)
                first["samples"][0]["uid"]="caller_mutated_envelope"
                iterator=iter(window)
                self.assertEqual(decode.call_count,0)
                output=next(iterator)
                self.assertEqual(decode.call_count,4)
                output["sample_lineage"][0]["request"]["uid"]="output_mutated_only"
                offsets=[output["microbatch_offset"]];tokens=[output["reservation_token"]]
                self.assertEqual(q.committed_draws,0)
                for output in iterator:
                    offsets.append(output["microbatch_offset"]);tokens.append(output["reservation_token"])
                    self.assertEqual(output["effective_batch_size"],16);self.assertFalse(output["commit_performed"])
                    self.assertTrue(all(x is not None for x in output["sample_lineage"]))
                    self.assertTrue(all(x["request"]["uid"]==fixture_module.R_UID for x in output["sample_lineage"]))
                    self.assertEqual(output["sample_lineage"][0]["source_reference"]["split"],"train")
                self.assertEqual(decode.call_count,32)
                self.assertEqual(offsets,[0,4,8,12]*2)
                self.assertEqual(tokens,[source[0]["token"]]*4+[source[1]["token"]]*4)
            self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
            self.assertEqual(q.committed_draws,0);self.assertEqual(q.pending_batches,2)
            q.commit(source[0]["token"]);self.assertEqual(q.committed_draws,16)
            self.assertEqual(q.cancel_pending(),1)
            replay=q.reserve(16);self.assertEqual(replay["samples"],source[1]["samples"])
            with self.assertRaises(ValueError):q.commit(source[1]["token"])
            with self.assertRaises(RuntimeError):iter(window)
            self.assertEqual(window.implementation_info()["backend"],"torch.utils.data.DataLoader")

    def test_outputs_match_execution_and_cross_stream_order_preserved(self):
        q=sampler("R_extra");s=sampler("S");a,b=q.reserve(16),s.reserve(16)
        observed=[]
        for out in ReservationDataLoader([a,b],TinyRequestDataset(),microbatch_size=8,generator=generator()):
            observed.append((out["reservation_token"],out["microbatch_offset"],deepcopy(out["sample_lineage"])))
        expected=[]
        for reservation in (a,b):
            for out in iter_reserved_microbatches(reservation,TinyRequestDataset(),microbatch_size=8):
                expected.append((out["reservation_token"],out["microbatch_offset"],deepcopy(out["sample_lineage"])))
        self.assertEqual(observed,expected)
        self.assertEqual(observed[0][2][0]["sampling_stream"],"R_extra")
        self.assertEqual(observed[0][2][0]["data_domain"],"R")
        self.assertEqual(q.committed_draws,s.committed_draws)
        self.assertEqual(q.committed_draws,0)

    def test_generator_state_explicit_resume_and_global_torch_rng_unchanged(self):
        q=sampler();reservation=q.reserve(16);g=generator();initial=loader_generator_state(g)
        global_before=torch.get_rng_state().clone()
        first=ReservationDataLoader([reservation],TinyRequestDataset(),microbatch_size=4,generator=g)
        for out in first:pass
        final=first.generator_state_dict();self.assertFalse(torch.equal(initial["state"],final["state"]))
        g2=torch.Generator(device="cpu");restore_loader_generator_state(g2,initial)
        second=ReservationDataLoader([reservation],TinyRequestDataset(),microbatch_size=4,generator=g2)
        for out in second:pass
        self.assertTrue(torch.equal(final["state"],second.generator_state_dict()["state"]))
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        final["state"].zero_();self.assertFalse(torch.equal(final["state"],g.get_state()))
        before=g2.get_state().clone()
        with self.assertRaises((RuntimeError,ValueError)):restore_loader_generator_state(g2,{"schema_version":"futsalmot_loader_generator_v1","state":torch.empty(0,dtype=torch.uint8)})
        self.assertTrue(torch.equal(before,g2.get_state()))

    def test_invalid_windows_modes_tokens_and_none_lineage_rejected(self):
        q=sampler();a,b=q.reserve(16),q.reserve(16)
        dataset=TinyRequestDataset()
        cases=[[],[a,a],[b,a],[a,b,a]]
        bad=deepcopy(a);bad["samples"].reverse();cases.append([bad])
        for windows in cases:
            with self.assertRaises(ValueError):ReservationDataLoader(windows,dataset,microbatch_size=4,generator=generator())
        for options in ({"num_workers":2},{"num_workers":4},{"pin_memory":True},{"persistent_workers":True}):
            with self.assertRaisesRegex(ValueError,"BLOCKED"):ReservationDataLoader([a],dataset,microbatch_size=4,generator=generator(),**options)
        with self.assertRaises(ValueError):ReservationDataLoader([a],dataset,microbatch_size=4,generator=None)
        self.assertEqual(dataset.calls,[])
        bad_loader=ReservationDataLoader([a],TinyRequestDataset(missing_lineage=True),microbatch_size=4,generator=generator())
        with self.assertRaisesRegex(ValueError,"lineage"):next(iter(bad_loader))

    def test_pickle_callbacks_are_module_level_and_mixed_token_or_wrong_request_refused(self):
        q=sampler();a,b=q.reserve(16),q.reserve(16);base=TinyRequestDataset()
        window=ReservationDataLoader([a,b],base,microbatch_size=2,generator=generator())
        dataset=pickle.loads(pickle.dumps(window._loader.dataset))
        batches=pickle.loads(pickle.dumps(window._loader.batch_sampler))
        callback=pickle.loads(pickle.dumps(collate_reserved_window))
        indices=next(iter(batches));items=[dataset[x] for x in indices]
        out=callback(items);self.assertEqual(out["microbatch_offset"],0)
        bad=deepcopy(indices[0]);bad["request"]["uid"]="not_provided_UID"
        calls=len(dataset.dataset.calls)
        with self.assertRaises(ValueError):dataset[bad]
        self.assertEqual(len(dataset.dataset.calls),calls)
        items[1]["reservation_token"]=b["token"]
        with self.assertRaisesRegex(ValueError,"混token"):callback(items)
        with self.assertRaises(ValueError):dataset[0]
        self.assertEqual(base.calls,[])

if __name__=="__main__":unittest.main()
