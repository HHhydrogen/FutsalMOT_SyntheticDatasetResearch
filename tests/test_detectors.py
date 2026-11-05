"""真实随机YOLO11n/64CPU前向与给定eval概率机械桥，零backward/step/公开pt。"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-native-detector-")
_ENV={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo")
os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import torch
from futsalmot.detectors import build_random_yolo11,NativeDetectionNetwork,decode_native_eval_rows,postprocess_native_eval
from futsalmot.records import capture_rng_state,restore_rng_state

_NETWORK=None
_RNG=None
_THREADS=None
_GUARDS=[]

def setUpModule():
    global _NETWORK,_RNG,_THREADS
    _RNG=capture_rng_state();_THREADS=torch.get_num_threads();torch.set_num_threads(2)
    for obj,name in [(torch.cuda,"_lazy_init"),(torch.cuda,"init"),(torch.cuda,"is_available"),(torch.optim.SGD,"step"),(torch,"load")]:
        guard=patch.object(obj,name,side_effect=AssertionError("fixture禁止GPU/step/加载任何pt"));guard.start();_GUARDS.append(guard)
    _NETWORK=build_random_yolo11(model_size="n")

def tearDownModule():
    global _NETWORK
    _NETWORK=None
    if _RNG is not None:restore_rng_state(_RNG)
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    for guard in reversed(_GUARDS):guard.stop()
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


def crafted():
    p=torch.zeros((1,84,3),dtype=torch.float32)
    p[:,:4,:]=torch.tensor([[[10.,20.,30.],[10.,20.,30.],[4.,6.,8.],[6.,8.,10.]]])
    p[0,4,0]=.8;p[0,4+5,0]=.9 # person也高，但80类winner是5。
    p[0,4,1]=.95;p[0,4+5,1]=.2
    return p,[torch.zeros((1,144,1,1)) for _ in range(3)]


def geometry(uid):
    return {"original_shape":(960,960),"input_shape":(960,960),"resized_shape":(960,960),"scale":(1.,1.),
            "padding_ltrb":(0,0,0,0),"source_uid":uid,"channel_order":"RGB","letterbox_candidate_only":True}


class NativeDetectorTests(unittest.TestCase):
    def test_native_train_heads_eval_shapes_DFL_freeze_and_no_parameter_updates(self):
        network=_NETWORK
        self.assertEqual(network.initialization_binding["kind"],"random_installed_yaml_fixture_not_public_pretrained")
        self.assertFalse(network.initialization_binding["public_pretraining_verified"])
        head=network.model.model[-1]
        self.assertEqual((head.nc,head.reg_max),(80,16));self.assertFalse(head.dfl.conv.weight.requires_grad)
        self.assertTrue(torch.equal(head.dfl.conv.weight.reshape(-1),torch.arange(16,dtype=torch.float32)))
        self.assertGreater(sum(isinstance(m,torch.nn.BatchNorm2d) for m in network.model.modules()),0)
        parameters={name:p.detach().clone() for name,p in network.model.named_parameters()}
        image=torch.zeros((2,3,64,64),dtype=torch.float32)
        with torch.no_grad():heads=network.training_heads(image)
        self.assertEqual([tuple(x.shape) for x in heads],[(2,144,8,8),(2,144,4,4),(2,144,2,2)])
        output=network.evaluation_output(image)
        self.assertEqual(tuple(output[0].shape),(2,84,84))
        with self.assertRaisesRegex(ValueError,"960"):
            postprocess_native_eval(output,uids=["a","b"],geometries=[geometry("a"),geometry("b")])
        rows=decode_native_eval_rows(output)
        self.assertEqual([tuple(x.shape) for x in rows],[(84,6),(84,6)])
        self.assertTrue(all(x.dtype==torch.float32 and x.device.type=="cpu" and x.is_contiguous() for x in rows))
        for name,p in network.model.named_parameters():
            self.assertTrue(torch.equal(p,parameters[name]));self.assertIsNone(p.grad)

    def test_decoding_xywh_all80_argmax_no_extra_sigmoid_filter_or_input_mutation(self):
        output=crafted();before=output[0].clone();rows=decode_native_eval_rows(output)[0]
        self.assertEqual(rows[0,:4].tolist(),[8.,7.,12.,13.])
        self.assertEqual(rows[:,5].tolist(),[5.,0.,0.])
        self.assertEqual(rows[0,4].item(),torch.tensor(.9).item())
        self.assertEqual(rows[1,4].item(),torch.tensor(.95).item())
        self.assertEqual(rows[2,4].item(),0.) # 零分也保留，decoder不阈值过滤。
        rows.zero_();self.assertTrue(torch.equal(output[0],before))

    def test_invalid_training_logits_probabilities_or_head_structure_rejected(self):
        output=crafted()
        with self.assertRaises(ValueError):decode_native_eval_rows(output[1])
        for index,value in [((0,4,0),1.1),((0,2,0),0.),((0,0,0),float("nan"))]:
            bad=(output[0].clone(),output[1]);bad[0][index]=value
            with self.assertRaises(ValueError):decode_native_eval_rows(bad)
        with self.assertRaises(ValueError):decode_native_eval_rows((torch.zeros((1,83,3)),output[1]))
        with self.assertRaises(ValueError):decode_native_eval_rows((output[0],output[1][:2]))

    def test_dense960_decode_delegates_existing_candidate_person_postprocess(self):
        features=[torch.zeros((1,144,size,size)) for size in (120,60,30)]
        count=120**2+60**2+30**2
        p=torch.zeros((1,84,count));p[:,0:2]=480.;p[:,2:4]=20.
        p[0,4,0]=.8;p[0,4+7,0]=.9;p[0,4,1]=.95
        result=postprocess_native_eval((p,features),uids=["fixture"],geometries=[geometry("fixture")])[0]
        self.assertEqual(result["input_prediction_count"],count)
        self.assertEqual(result["final_prediction_count"],1)
        self.assertEqual(result["raw_predictions"][0]["class_id"],7)
        self.assertIn("non_person",result["raw_predictions"][0]["filter_reasons"])
        self.assertEqual(result["predictions"][0]["input_prediction_index"],1)
        self.assertTrue(result["candidate_only"]);self.assertFalse(result["official_benchmark"])

    def test_explicit_CPU_inputs_and_CUDA_permission_and_DFL_invariant(self):
        with self.assertRaises(ValueError):_NETWORK.training_heads(torch.zeros((1,3,64,64),dtype=torch.uint8))
        with self.assertRaises(ValueError):_NETWORK.training_heads(torch.full((1,3,64,64),2.))
        with self.assertRaises(RuntimeError):NativeDetectionNetwork(_NETWORK.model,device="cuda",allow_initialized_cuda=False)
        with patch.object(torch.cuda,"is_initialized",return_value=False):
            with self.assertRaises(RuntimeError):NativeDetectionNetwork(_NETWORK.model,device="cuda",allow_initialized_cuda=True)
        with self.assertRaisesRegex(ValueError,"公开"):
            NativeDetectionNetwork(_NETWORK.model,initialization_binding={"kind":"caller_claim","public_pretraining_verified":True})
        dfl=_NETWORK.model.model[-1].dfl.conv.weight
        dfl.requires_grad_(True)
        try:
            with self.assertRaisesRegex(ValueError,"冻结"):_NETWORK.training_heads(torch.zeros((1,3,64,64)))
        finally:dfl.requires_grad_(False)

if __name__=="__main__":unittest.main()
