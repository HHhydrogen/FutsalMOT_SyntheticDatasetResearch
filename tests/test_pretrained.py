"""临时可信锚夹具+mock反序列化；不读15份公开pt、不forward/step/CUDA。"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
import hashlib
import json
import types
import stat

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-pretrained-")
_ENV={key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR","YOLO_OFFLINE")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_OFFLINE"]="true"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo")
os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import torch
from ultralytics.nn.modules.conv import Conv
from futsalmot import pretrained as p
from futsalmot.detectors import NativeDetectionNetwork
_RNG=None
_THREADS=None

def setUpModule():
    global _RNG,_THREADS
    _RNG=torch.get_rng_state().clone();_THREADS=torch.get_num_threads();torch.set_num_threads(2)

def tearDownModule():
    if _RNG is not None:torch.set_rng_state(_RNG)
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value

class FixtureHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc=80;self.reg_max=16;self.end2end=False;self.export=False;self.xyxy=False
        self.dfl=torch.nn.Module();self.dfl.conv=torch.nn.Conv2d(16,1,1,bias=False)
        with torch.no_grad():self.dfl.conv.weight.copy_(torch.arange(16).reshape(1,16,1,1))
        self.dfl.conv.weight.requires_grad_(False)

class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Sequential(Conv(3,4),FixtureHead())
        self.yaml={"scale":"n"};self.names={i:"person" if i==0 else str(i) for i in range(80)}
    def forward(self,*args,**kwargs):raise AssertionError("禁止fixture forward")

class PretrainedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="pretrained-case-",dir=_TEMP.name)
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.blob=b"fixture-pickle-not-executed";self.digest=hashlib.sha256(self.blob).hexdigest()
        self.path=self.root/"yolo11n.pt";self.path.write_bytes(self.blob)
        row={"file":"yolo11n.pt","task":"detect","model_size":"n","bytes":len(self.blob),"sha256":self.digest,
             "source_url":"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
             "pretraining":"COCO detection (80 classes)"}
        self.manifest={"schema_version":"public_yolo11_weights_v1","status":"PASS","official_repository":"ultralytics/assets",
                       "release_tag":"v8.3.0","files":[row]+[{"file":str(i)} for i in range(14)]}
        self.write_manifest()
        self.patches=[]
        for obj,name,value in [(p,"MANIFEST_SHA256",self.manifest_sha),(p,"_DETECT",{"n":(len(self.blob),self.digest)}),
            (p,"_native",lambda:(FixtureModel,FixtureHead))]:
            g=patch.object(obj,name,value);g.start();self.patches.append(g)
        g=patch("futsalmot.detectors._native",return_value=(FixtureModel,FixtureHead));g.start();self.patches.append(g)
        for obj,name in [(torch.cuda,"_lazy_init"),(torch.cuda,"init"),(torch.cuda,"is_available"),(torch.optim.SGD,"step")]:
            g=patch.object(obj,name,side_effect=AssertionError("禁止CUDA/step"));g.start();self.patches.append(g)
        self.addCleanup(lambda:[g.stop() for g in reversed(self.patches)])
    def write_manifest(self):
        payload=json.dumps(self.manifest).encode()
        (self.root/"manifest.json").write_bytes(payload);self.manifest_sha=hashlib.sha256(payload).hexdigest()
    def deserialize(self,stream,**kwargs):
        self.assertEqual(kwargs,{"map_location":"cpu","weights_only":False})
        self.assertEqual(stream.read(),self.blob)
        model=FixtureModel().half()
        for value in model.parameters():value.requires_grad_(False)
        return {"model":model,"ema":None,"train_args":{"task":"detect"}}
    def test_verified_cpu_conversion_full_trainability_BN_binding_and_no_caller_boolean(self):
        with patch.object(torch,"load",side_effect=self.deserialize) as load,patch.object(torch,"arange",wraps=torch.arange) as arange:
            network=p.load_local_yolo11_detect(self.root,model_size="n")
        arange.assert_any_call(16,dtype=torch.float32,device="cpu")
        self.assertEqual(load.call_count,1)
        self.assertTrue(network.initialization_binding["public_pretraining_verified"])
        self.assertEqual(network.initialization_binding["sha256"],self.digest)
        self.assertEqual(network.initialization_binding["manifest_sha256"],self.manifest_sha)
        self.assertFalse(network.initialization_binding["publisher_digest_verified"])
        self.assertEqual(network.model.model[-1].nc,80)
        for name,value in network.model.named_parameters():
            self.assertEqual(value.device.type,"cpu");self.assertEqual(value.dtype,torch.float32)
            self.assertEqual(value.requires_grad,not name.endswith("dfl.conv.weight"))
        self.assertTrue(hasattr(network.model.model[0],"bn"));self.assertFalse(network.model.training)
        with self.assertRaisesRegex(ValueError,"verified"):
            NativeDetectionNetwork(network.model,initialization_binding={"public_pretraining_verified":True})
        with self.assertRaises(ValueError):p._verified_binding(object(),network.model)
    def test_corrupt_weight_manifest_and_unknown_inputs_never_deserialize(self):
        with patch.object(torch,"load",side_effect=AssertionError("不得反序列化")):
            for size in ("n-seg","n-pose","../n","unknown"):
                with self.assertRaises(ValueError):p.load_local_yolo11_detect(self.root,model_size=size)
            with self.assertRaises(ValueError):p.load_local_yolo11_detect("https://example.invalid",model_size="n")
            with self.assertRaises(ValueError):p.load_local_yolo11_detect("//server/share",model_size="n")
            self.path.write_bytes(b"X"*len(self.blob))
            with self.assertRaisesRegex(ValueError,"SHA"):p.load_local_yolo11_detect(self.root,model_size="n")
            self.path.write_bytes(self.blob)
            with (self.root/"manifest.json").open("ab") as f:f.write(b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):p.load_local_yolo11_detect(self.root,model_size="n")
    def test_explicit_offline_required_before_native_import_or_path_access(self):
        with patch.object(p,"_sources",side_effect=AssertionError("不得导入原生模块")), \
             patch.object(p,"_native",side_effect=AssertionError("不得导入原生模块")), \
             patch.object(p,"_no_reparse",side_effect=AssertionError("不得读路径")), \
             patch.object(Path,"absolute",side_effect=AssertionError("不得解析路径")):
            for value in (None,"false","1",""):
                with self.subTest(value=value),patch.dict(os.environ):
                    if value is None:os.environ.pop("YOLO_OFFLINE",None)
                    else:os.environ["YOLO_OFFLINE"]=value
                    with self.assertRaisesRegex(RuntimeError,"YOLO_OFFLINE"):
                        p.load_local_yolo11_detect(self.root,model_size="n")

    def test_nonlocal_and_parent_traversal_rejected_before_any_filesystem_access(self):
        rejected=("//server/share",r"\\server\share",r"\\?\UNC\server\share",
                  str(self.root/".."/"elsewhere"),"../weights",r"folder\..\weights")
        with patch.object(p,"_no_reparse",side_effect=AssertionError("前置拒绝不得读路径")), \
             patch.object(Path,"absolute",side_effect=AssertionError("前置拒绝不得解析绝对路径")), \
             patch.object(torch,"load",side_effect=AssertionError("不得反序列化")):
            for value in rejected:
                with self.subTest(value=value),self.assertRaises(ValueError):
                    p.load_local_yolo11_detect(value,model_size="n")

    def test_pinned_manifest_record_URL_task_and_name_are_required(self):
        for key,value in (("task","pose"),("source_url","https://evil.invalid/yolo11n.pt"),("model_size","s"),("file","../yolo11n.pt")):
            with self.subTest(key=key):
                original=deepcopy(self.manifest)
                self.manifest["files"][0][key]=value;self.write_manifest()
                with patch.object(p,"MANIFEST_SHA256",self.manifest_sha),patch.object(torch,"load",side_effect=AssertionError("不得反序列化")):
                    with self.assertRaises(ValueError):p.load_local_yolo11_detect(self.root,model_size="n")
                self.manifest=original
    def test_source_mutation_after_verified_bytes_stops_return(self):
        def mutate(stream,**kwargs):
            result=self.deserialize(stream,**kwargs);self.path.write_bytes(b"Z"*len(self.blob));return result
        with patch.object(torch,"load",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"SHA"):p.load_local_yolo11_detect(self.root,model_size="n")
    def test_non_model_wrong_scale_fused_and_nonfinite_are_rejected(self):
        bad=FixtureModel();bad.yaml["scale"]="s"
        fused=FixtureModel();del fused.model[0].bn
        badfinite=FixtureModel()
        with torch.no_grad():badfinite.model[0].conv.weight.fill_(float("nan"))
        for model in (torch.nn.Linear(1,1),bad,fused,badfinite):
            with self.subTest(model=type(model).__name__),patch.object(torch,"load",return_value={"model":model}):
                with self.assertRaises(ValueError):p.load_local_yolo11_detect(self.root,model_size="n")
    def test_reparse_source_rejected_before_deserialize(self):
        original=Path.lstat
        def lstat(path,*args,**kwargs):
            s=original(path,*args,**kwargs)
            if path==self.path:return types.SimpleNamespace(st_mode=stat.S_IFLNK,st_file_attributes=0x400)
            return s
        with patch.object(Path,"lstat",lstat),patch.object(torch,"load",side_effect=AssertionError("不得反序列化")):
            with self.assertRaisesRegex(ValueError,"reparse"):p.load_local_yolo11_detect(self.root,model_size="n")

if __name__=="__main__":unittest.main()
