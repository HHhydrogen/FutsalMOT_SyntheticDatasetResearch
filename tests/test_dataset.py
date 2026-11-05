"""候选请求Dataset端到端CPU夹具，只读自造临时PNG/JPEG/GT/JSONL。"""
from copy import deepcopy
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-dataset-config-")
_ENV={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","YOLO_CONFIG_DIR","MPLCONFIGDIR")}
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["YOLO_CONFIG_DIR"]=str(Path(_TEMP.name)/"yolo")
os.environ["MPLCONFIGDIR"]=str(Path(_TEMP.name)/"mpl")
import cv2
import numpy as np
from PIL import Image
import torch
from futsalmot.dataset import RequestBBoxDataset,decode_rgb_image
from futsalmot.data import CycleSampler,PrefetchSampler
from futsalmot.adapters import RealBBoxReader
from test_adapters import real_fixture,synthetic_fixture

_THREAD=None
_CV_THREAD=None
_BLOCK=None

def setUpModule():
    global _THREAD,_CV_THREAD,_BLOCK
    _THREAD=torch.get_num_threads();torch.set_num_threads(2)
    _CV_THREAD=cv2.getNumThreads();cv2.setNumThreads(2)
    _BLOCK=mock.patch("torch.cuda._lazy_init",side_effect=AssertionError("CPU fixture禁止CUDA初始化"));_BLOCK.start()

def tearDownModule():
    if _BLOCK is not None:_BLOCK.stop()
    if _THREAD is not None:torch.set_num_threads(_THREAD)
    if _CV_THREAD is not None:cv2.setNumThreads(_CV_THREAD)
    _TEMP.cleanup()
    for key,value in _ENV.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


R_UID="detect:source:clip:000001"
S_UID="detect:synthetic:fixture_synth:v3_source:seed_00007546:C5:000001"


def request(uid=R_UID,domain="R"):
    return dict(uid=uid,domain=domain,exposure_index=7,augmentation_seed=12345,epoch=1)


def fixture(root):
    real=root/"真实来源_汉字";synth=root/"合成来源_汉字"
    index,_,_=real_fixture(real)
    synthetic_fixture(synth)
    rpath=real/"sequences/clip/img1/000001.jpg";rpath.parent.mkdir(parents=True)
    spath=synth/"C5/img1/000001.png";spath.parent.mkdir(parents=True)
    rgb=np.full((50,100,3),[230,10,3],dtype=np.uint8)
    exif=Image.Exif();exif[274]=6 # header方向存在，原标注像素不能跟着旋转。
    Image.fromarray(rgb).save(rpath,quality=100,subsampling=0,exif=exif)
    Image.fromarray(rgb).save(spath)
    catalogue={R_UID:dict(domain="R",index_path=str(index),split="train"),
               S_UID:dict(domain="S",seed_dir=str(synth),camera_id="C5",frame_index=1,split="train")}
    return catalogue,rpath,spath


class RequestDatasetTests(unittest.TestCase):
    def test_R_S_RGB_unicode_orientation_and_request_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,rpath,spath=fixture(Path(t));d=RequestBBoxDataset(catalogue)
            r=d[request()];s=d[request(S_UID,"S")]
            self.assertEqual(r["uid"],R_UID);self.assertEqual(s["uid"],S_UID)
            self.assertEqual(s["source_reference"]["split"],"train")
            self.assertEqual(r["img"].dtype,torch.uint8)
            self.assertTrue(r["img"].is_contiguous())
            self.assertEqual(r["geometry"]["original_shape"],(50,100))
            self.assertTrue(np.allclose(r["img"][:,300,300].numpy(),[230,10,3],atol=2))
            self.assertEqual(s["img"][:,300,300].tolist(),[230,10,3])
            self.assertTrue(r["candidate_only"])
            self.assertFalse(r["random_augmentation_implemented"])
            self.assertTrue(r["caller_release_validation_required"])
            self.assertEqual(r["request"],request())
            r["source_reference"]["split"]="test";r["request"]["augmentation_seed"]=-1
            self.assertEqual(d[request()]["request"]["augmentation_seed"],12345)
            self.assertEqual(d[request()]["source_reference"]["split"],"train")

    def test_prefetch_R_extra_request_does_not_commit_or_reseed(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,_,_=fixture(Path(t));d=RequestBBoxDataset(catalogue)
            q=PrefetchSampler(CycleSampler([R_UID],task="detect",domain="R_extra",seed=17))
            reserved=q.reserve(1);before=q.state_dict()
            result=d[reserved["samples"][0]]
            self.assertEqual(result["sampling_stream"],"R_extra")
            self.assertEqual(result["data_domain"],"R")
            self.assertEqual(q.state_dict(),before)
            self.assertEqual(q.committed_draws,0);self.assertEqual(q.pending_batches,1)
            self.assertEqual(result["request"],reserved["samples"][0])

    def test_bad_request_catalogue_and_reader_identity_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,_,_=fixture(Path(t));d=RequestBBoxDataset(catalogue)
            cases=[0,"uid",{},request("unknown"),request(R_UID,"S"),
                   {**request(),"exposure_index":True},{**request(),"augmentation_seed":-1},
                   {**request(),"stream":"S"},{**request(),"split":"val"}]
            for req in cases:
                with self.assertRaises(ValueError):d[req]
            changed=deepcopy(catalogue);changed[R_UID]["split"]="test"
            with self.assertRaises(ValueError):RequestBBoxDataset(changed)
            original=RealBBoxReader.read
            def corrupt(self,*args,**kwargs):
                value=original(self,*args,**kwargs);value["sample_uid"]="wrong_UID";return value
            with mock.patch.object(RealBBoxReader,"read",corrupt):
                with self.assertRaisesRegex(ValueError,"读回UID"):d[request()]
            wrong_s=deepcopy(catalogue);wrong_s["wrong_derived_UID"]=wrong_s.pop(S_UID)
            with self.assertRaisesRegex(ValueError,"读回UID"):RequestBBoxDataset(wrong_s)[request("wrong_derived_UID","S")]

    def test_header_limits_corrupt_and_shape_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as t:
            _,_,path=fixture(Path(t))
            with mock.patch("futsalmot.dataset.cv2.imdecode",side_effect=AssertionError("header应先拒绝")):
                with self.assertRaisesRegex(ValueError,"header尺寸"):decode_rgb_image(path,expected_shape=(1,1))
                with self.assertRaisesRegex(ValueError,"像素数"):decode_rgb_image(path,expected_shape=(50,100),max_pixels=4999)
                with self.assertRaisesRegex(ValueError,"编码字节"):decode_rgb_image(path,expected_shape=(50,100),max_encoded_bytes=4)
            path.write_bytes(b"not_an_image")
            with self.assertRaises(ValueError):decode_rgb_image(path,expected_shape=(50,100))
            Image.new("RGBA",(100,50)).save(path)
            with self.assertRaisesRegex(ValueError,"RGB"):decode_rgb_image(path,expected_shape=(50,100))
            Image.new("RGB",(100,50)).save(path)
            with mock.patch("futsalmot.dataset.cv2.imdecode",return_value=None):
                with self.assertRaisesRegex(ValueError,"解码损坏"):decode_rgb_image(path,expected_shape=(50,100))

    def test_image_source_change_during_and_after_decode_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,_,path=fixture(Path(t));d=RequestBBoxDataset(catalogue)
            d[request(S_UID,"S")]
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"来源"):d[request(S_UID,"S")]
        with tempfile.TemporaryDirectory() as t:
            _,_,path=fixture(Path(t))
            original=cv2.imdecode
            def mutate(*args,**kwargs):
                rgb=original(*args,**kwargs);path.write_bytes(path.read_bytes()+b"changed");return rgb
            with mock.patch("futsalmot.dataset.cv2.imdecode",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"来源变化"):decode_rgb_image(path,expected_shape=(50,100))

    def test_pickle_cold_readers_preserve_bindings_without_RGB_cache(self):
        with tempfile.TemporaryDirectory() as t:
            catalogue,_,spath=fixture(Path(t));d=RequestBBoxDataset(catalogue)
            original=d[request()];d[request(S_UID,"S")]
            restored=pickle.loads(pickle.dumps(d)) # 本测试刚生成的可信对象。
            for reader in restored._readers.values():
                info=reader.cache_info()
                self.assertEqual(info.get("cached_rows",info.get("indexed_frames")),0)
            actual=restored[request()]
            self.assertTrue(torch.equal(actual["img"],original["img"]))
            self.assertEqual(actual["request"],original["request"])
            encoded=pickle.dumps(d);spath.write_bytes(spath.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"来源变化"):pickle.loads(encoded)

    def test_explicit_view_prevents_train_validation_split_leakage(self):
        from test_adapters import write_json
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);catalogue,_,_=fixture(root)
            with self.assertRaisesRegex(ValueError,"split"):RequestBBoxDataset(catalogue,view="val")
            with self.assertRaisesRegex(ValueError,"view"):RequestBBoxDataset(catalogue,view="test")
            mixed=deepcopy(catalogue);mixed[S_UID]["split"]="dev"
            with self.assertRaisesRegex(ValueError,"split"):RequestBBoxDataset(mixed,view="train")
            d=RequestBBoxDataset(catalogue,view="train")
            with self.assertRaisesRegex(ValueError,"view"):d[{**request(),"view":"val"}]
            # 自造val/dev来源，证明不能靠把view标签改名绕过原source split。
            real=root/"val_real";synth=root/"dev_synth"
            index,_,_=real_fixture(real,split="val");m,_=synthetic_fixture(synth)
            m["split"]="dev";write_json(synth/"seed_manifest.json",m)
            rpath=real/"sequences/clip/img1/000001.jpg";rpath.parent.mkdir(parents=True)
            spath=synth/"C5/img1/000001.png";spath.parent.mkdir(parents=True)
            Image.new("RGB",(100,50),(20,30,40)).save(rpath)
            Image.new("RGB",(100,50),(20,30,40)).save(spath)
            refs={R_UID:dict(domain="R",index_path=str(index),split="val"),
                  S_UID:dict(domain="S",seed_dir=str(synth),camera_id="C5",frame_index=1,split="dev")}
            val=RequestBBoxDataset(refs,view="val")
            self.assertEqual(val[request()]["source_reference"]["split"],"val")
            self.assertEqual(val[request(S_UID,"S")]["source_reference"]["split"],"dev")

    def test_reparse_and_nonregular_paths_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as t:
            _,_,path=fixture(Path(t))
            with self.assertRaises(ValueError):decode_rgb_image(path.parent,expected_shape=(50,100))
            # Windows无需创建管理员symlink；模拟lstat给出重解析属性，必须在open前失败。
            from types import SimpleNamespace
            original=Path.lstat
            def lstat(target,*args,**kwargs):
                value=original(target,*args,**kwargs)
                if target == path:return SimpleNamespace(st_mode=value.st_mode,st_file_attributes=0x400)
                return value
            with mock.patch.object(Path,"lstat",lstat),mock.patch.object(Path,"open",side_effect=AssertionError("不可打开重解析路径")):
                with self.assertRaisesRegex(ValueError,"重解析"):decode_rgb_image(path,expected_shape=(50,100))

if __name__=="__main__":unittest.main()
