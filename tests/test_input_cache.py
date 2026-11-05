"""无损SSD输入缓存的小CPU夹具；不读真实库/不建实际pack/不使用CUDA。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

_TEMP=tempfile.TemporaryDirectory(prefix="futsalmot-cache-test-config-")
_ENV={k:os.environ.get(k) for k in ("YOLO_OFFLINE","YOLO_CONFIG_DIR","MPLCONFIGDIR","CUDA_VISIBLE_DEVICES")}
os.environ.update(YOLO_OFFLINE="true",YOLO_CONFIG_DIR=str(Path(_TEMP.name)/"yolo"),MPLCONFIGDIR=str(Path(_TEMP.name)/"mpl"),CUDA_VISIBLE_DEVICES="-1")
import numpy as np
from PIL import Image
import torch
import cv2
import futsalmot.input_cache as cache
from futsalmot.dataset import RequestBBoxDataset
from futsalmot.runner_data import RequestPlanDataset
from futsalmot.train_data import prepare_training_sample
from test_adapters import synthetic_fixture
_THREADS=None
_CV=None
_GUARD=None


def setUpModule():
    global _THREADS,_CV,_GUARD
    _THREADS=torch.get_num_threads();torch.set_num_threads(2)
    _CV=cv2.getNumThreads();cv2.setNumThreads(2)
    _GUARD=patch("torch.cuda._lazy_init",side_effect=AssertionError("缓存CPU夹具不得初始化CUDA"));_GUARD.start()


def tearDownModule():
    _GUARD.stop();torch.set_num_threads(_THREADS);cv2.setNumThreads(_CV);_TEMP.cleanup()
    for k,v in _ENV.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v


def fixture(root):
    source=root/"source";manifest,row=synthetic_fixture(source)
    manifest["camera_ids"]=["C5","C1"]
    (source/"seed_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    camera=json.loads((source/"C5/camera.json").read_bytes());camera["camera_id"]="C1"
    (source/"C1").mkdir();(source/"C1/camera.json").write_text(json.dumps(camera),encoding="utf-8")
    another=deepcopy(row);another["camera_id"]="C1"
    (source/"C1/annotations.jsonl").write_text(json.dumps(another)+"\n",encoding="utf-8")
    catalogue={}
    for i,cam in enumerate(("C5","C1")):
        image=source/cam/"img1/000001.png";image.parent.mkdir()
        rgb=np.empty((50,100,3),dtype=np.uint8);rgb[...,0]=np.arange(100,dtype=np.uint8)[None,:]+i*100;rgb[...,1]=29;rgb[...,2]=np.arange(50,dtype=np.uint8)[:,None]
        Image.fromarray(rgb).save(image)
        uid=f"detect:synthetic:fixture_synth:v3_source:seed_00007546:{cam}:000001"
        catalogue[uid]={"domain":"S","seed_dir":str(source),"camera_id":cam,"frame_index":1,"split":"train"}
    original=RequestBBoxDataset(catalogue,view="train")
    paths=[source/"seed_manifest.json",*[source/cam/name for cam in ("C5","C1") for name in ("camera.json","annotations.jsonl")]]
    bound={str(p):cache._file_binding(p) for p in paths}
    dependencies={uid:list(bound) for uid in catalogue}
    def request(uid,seed=37,exposure=2):return {"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,"exposure_index":exposure,"augmentation_seed":seed}
    def prepared(uid):return original[request(uid)]
    target=root/"pack"
    receipt=cache._build_pack(target,list(catalogue),prepared,source_binding={"fixture":True},source_files=bound,
        dependencies=dependencies,production=False,free_reserve=0,max_seconds=30)
    return catalogue,original,request,target,receipt,source


class InputCacheTests(unittest.TestCase):
    def test_writer_never_opens_whole_array_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(cache.np.lib.format,"open_memmap",side_effect=AssertionError("whole array mapping forbidden")):
                catalogue,original,request,target,receipt,source=fixture(Path(temporary))
            self.assertEqual(receipt["state"],"READY")

    def test_exact_production_capacity_without_allocating_array(self):
        p=cache.cache_requirements()
        self.assertEqual(p["images_npy_bytes"],66767155328)
        self.assertEqual(p["image_payload_bytes"],66767155200)
        self.assertEqual(p["npy_header_bytes"],128)
        self.assertEqual(p["complete_upper_bytes"],66767155328+2*1024**3+64*1024**2)
        self.assertLess(p["complete_upper_bytes"],128*1024**3)
        self.assertEqual(p["required_free_bytes"],p["complete_upper_bytes"]+100*1024**3)
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError):cache.cache_requirements(bad)

    def test_metadata_tensor_dtype_shape_tuple_roundtrip(self):
        value={"boxes":torch.tensor([[.1,.2,.3,.4]],dtype=torch.float32),"empty":torch.empty((0,4),dtype=torch.float32),
               "ints":torch.tensor([123],dtype=torch.int64),"shape":(2160,3840),"nested":[None,True,{"x":3.5}]}
        self.assertTrue(cache._equal(value,cache._decode(json.loads(cache._json(cache._encode(value))))))
        for bad in (float("nan"),object(),{"__cache_type__":"spoof"}):
            with self.assertRaises(ValueError):cache._encode(bad)

    def test_real_reader_cache_pixels_labels_geometry_and_augmentation_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary))
            self.assertEqual(set(p.name for p in target.iterdir()),{"cache.json","images.npy","samples.jsonl"})
            self.assertEqual(receipt["count"],2);self.assertFalse(target.with_name("pack.partial").exists())
            reader=cache._CacheReader(target,require_production=False,expected_manifest_sha256=receipt["manifest_sha256"])
            try:
                for uid,ref in reversed(list(catalogue.items())):
                    for seed,exposure in ((37,2),(883,17)):
                        req=request(uid,seed,exposure);actual=reader.read_prepared(req,ref);direct=original[req]
                        self.assertTrue(cache._equal(actual,direct))
                        a=prepare_training_sample(actual,view="train");b=prepare_training_sample(direct,view="train")
                        self.assertTrue(cache._equal(a,b))
                self.assertEqual(reader._images.mode,"r")
            finally:reader.close()

    def test_pixel_corruption_even_with_restored_mtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));path=target/"images.npy";st=path.stat()
            with path.open("r+b") as f:f.seek(128);value=f.read(1);f.seek(128);f.write(bytes([value[0]^1]))
            os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
            reader=cache._CacheReader(target,require_production=False)
            try:
                uid=next(iter(catalogue))
                with self.assertRaisesRegex(ValueError,"像素SHA"):reader.read_prepared(request(uid),catalogue[uid])
            finally:reader.close()

    def test_source_RGB_stat_change_rejects_cached_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));reader=cache._CacheReader(target,require_production=False)
            try:
                uid=next(iter(catalogue));path=source/"C5/img1/000001.png";path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"原RGB"):reader.read_prepared(request(uid),catalogue[uid])
            finally:reader.close()

    def test_GT_change_rejects_cache_even_when_RGB_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));reader=cache._CacheReader(target,require_production=False)
            try:
                uid=next(iter(catalogue));path=source/"C5/annotations.jsonl";path.write_bytes(path.read_bytes()+b" ")
                with self.assertRaisesRegex(ValueError,"来源身份"):reader.read_prepared(request(uid),catalogue[uid])
            finally:reader.close()

    def test_published_cache_is_immutable_and_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));uid=next(iter(catalogue))
            with self.assertRaises(FileExistsError):cache._build_pack(target,[uid],lambda _:None,source_binding={},source_files={},dependencies={},production=False,free_reserve=0)
            failed=Path(temporary)/"failed"
            with self.assertRaisesRegex(RuntimeError,"fixture failure"):
                cache._build_pack(failed,[uid],lambda _:(_ for _ in ()).throw(RuntimeError("fixture failure")),source_binding={},source_files={},dependencies={},production=False,free_reserve=0)
            partial=failed.with_name("failed.partial")
            self.assertFalse(failed.exists());self.assertEqual(json.loads((partial/"cache.json").read_bytes())["state"],"FAIL_STOP")
            with self.assertRaisesRegex(ValueError,"partial"):cache._CacheReader(partial,require_production=False)

    def test_manifest_binding_production_fixture_and_bad_reference_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError,"SHA256"):cache._CacheReader(target,require_production=False,expected_manifest_sha256="0"*64)
            with self.assertRaisesRegex(ValueError,"fixture"):cache._CacheReader(target,expected_manifest_sha256=receipt["manifest_sha256"])
            reader=cache._CacheReader(target,require_production=False)
            try:
                uid=next(iter(catalogue));bad={**catalogue[uid],"frame_index":2}
                with self.assertRaisesRegex(ValueError,"来源引用"):reader.read_prepared(request(uid),bad)
                with self.assertRaises(ValueError):reader.read_prepared({**request(uid),"domain":"R"},catalogue[uid])
            finally:reader.close()

    def test_readonly_reader_pickles_without_live_memmap_and_revalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));reader=cache._CacheReader(target,require_production=False)
            restored=pickle.loads(pickle.dumps(reader))
            try:
                self.assertIsNone(restored._images);uid=next(iter(catalogue))
                self.assertTrue(cache._equal(restored.read_prepared(request(uid),catalogue[uid]),original[request(uid)]))
            finally:reader.close();restored.close()

    def test_compatible_loader_type_and_no_unapproved_actual_build(self):
        self.assertTrue(issubclass(cache.CachedRequestPlanDataset,RequestPlanDataset))
        with self.assertRaisesRegex(RuntimeError,"明确许可"):cache.build_input_cache()
        for bad in (0,14401,True,1.5):
            with self.assertRaises(ValueError):cache.build_input_cache(explicitly_authorized=True,max_seconds=bad)
        for bad in (0,3,True):
            with self.assertRaises(ValueError):cache.build_input_cache(explicitly_authorized=True,workers=bad)

    def test_two_spawn_CPU_workers_return_exact_unaugmented_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary))
            with cache.ProcessPoolExecutor(max_workers=2,mp_context=cache.multiprocessing.get_context("spawn"),
                    initializer=cache._worker_init,initargs=(catalogue,)) as pool:
                futures=[(uid,pool.submit(cache._prepare_worker,uid)) for uid in catalogue]
                self.assertLessEqual(len(futures),4)
                for uid,future in futures:
                    actual=future.result(timeout=30)
                    fixed={"uid":uid,"domain":"S","stream":"S","data_domain":"S","epoch":0,"exposure_index":0,"augmentation_seed":0}
                    self.assertTrue(cache._equal(actual,original[fixed]))

    def test_windowed_mapping_RSS_and_view_lifetime_are_bounded(self):
        import psutil
        process=psutil.Process()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"scratch.npy";cache._create_images(path,24)
            image=np.full((3,960,960),113,np.uint8);baseline=process.memory_info().rss
            array=cache._WindowedArray(path,24,mode="r+",window_rows=4);peak=baseline
            for i in range(24):
                array.write_exact(i,image);peak=max(peak,process.memory_info().rss)
            self.assertLessEqual(array.max_mapped_bytes,4*cache.ROW_BYTES)
            self.assertEqual(array.windows_opened,6)
            array.close();self.assertIsNone(array._view)
            self.assertLess(peak-baseline,4*cache.ROW_BYTES+48*1024**2)
            self.assertLess(process.memory_info().rss-baseline,32*1024**2)

    def test_partial_resume_reuses_complete_prefix_without_original_RGB_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));uids=list(catalogue)
            m=json.loads((target/"cache.json").read_bytes());partial=target.with_name("pack.partial");target.rename(partial)
            prefix_line=(partial/"samples.jsonl").read_bytes().splitlines(keepends=True)[0]
            (partial/"samples.jsonl").write_bytes(prefix_line)
            m.update(state="FAIL_STOP",completed_rows=1,elapsed_seconds=12.,error={"type":"RuntimeError","message":"fixture guard"})
            for key in ("rows","images_file","samples_file"):m.pop(key,None)
            cache._write_manifest(partial/"cache.json",m);old_bytes=(partial/"cache.json").read_bytes();old_sha=cache._sha(old_bytes)
            second=original[request(uids[1])]
            with patch.object(cache.RequestBBoxDataset,"__getitem__",side_effect=AssertionError("prefix must not decode original RGB")):
                inspected=cache.inspect_partial_cache(target,uids,expected_manifest_sha256=old_sha,catalogue=catalogue,
                    require_production=False,check_source_files=True,verify_source_hashes=True,window_rows=1)
            self.assertEqual(inspected["completed_rows"],1);self.assertFalse(inspected["old_pixel_sha_durable"])
            self.assertFalse(inspected["original_4K_read"]);self.assertEqual(len(inspected["rows"]),1)
            self.assertTrue(inspected["source_files_revalidated"])
            image=source/"C5/img1/000001.png";identity=image.stat()
            os.utime(image,ns=(identity.st_atime_ns,identity.st_mtime_ns+1000000))
            try:
                with self.assertRaisesRegex(ValueError,"原RGB stat"):
                    cache.inspect_partial_cache(target,uids,expected_manifest_sha256=old_sha,catalogue=catalogue,
                        require_production=False,check_source_files=True,verify_source_hashes=False)
            finally:os.utime(image,ns=(identity.st_atime_ns,identity.st_mtime_ns))
            calls=[]
            def prepare(uid):
                calls.append(uid)
                if uid!=uids[1]:raise AssertionError("already completed UID must not be regenerated")
                return deepcopy(second)
            inspected["recovery"]={"fixture":True,"old_manifest_sha256":old_sha}
            dependencies={uid:inspected["rows"][0]["source_files"] for uid in uids}
            final=cache._build_pack(target,uids,prepare,source_binding=m["source_binding"],source_files=m["source_files"],
                dependencies=dependencies,production=False,max_seconds=30,free_reserve=0,resume_prefix=inspected,window_rows=1)
            self.assertEqual(calls,[uids[1]]);self.assertEqual(final["reused_prefix_rows"],1)
            self.assertTrue((target/"samples.jsonl").read_bytes().startswith(prefix_line))
            self.assertEqual(set(p.name for p in target.iterdir()),{"cache.json","images.npy","samples.jsonl"})
            reader=cache._CacheReader(target,require_production=False)
            try:
                for uid,ref in catalogue.items():
                    self.assertTrue(cache._equal(reader.read_prepared(request(uid),ref),original[request(uid)]))
            finally:reader.close()

    def test_unknown_JSONL_tail_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalogue,original,request,target,receipt,source=fixture(Path(temporary));m=json.loads((target/"cache.json").read_bytes())
            partial=target.with_name("pack.partial");target.rename(partial)
            raw=(partial/"samples.jsonl").read_bytes().splitlines(keepends=True)[0]+b"unknown_tail"
            (partial/"samples.jsonl").write_bytes(raw);m.update(state="FAIL_STOP",completed_rows=1,elapsed_seconds=1.)
            m.pop("rows");cache._write_manifest(partial/"cache.json",m)
            before=(partial/"cache.json").read_bytes()
            with self.assertRaisesRegex(ValueError,"尾部"):
                cache.inspect_partial_cache(target,list(catalogue),expected_manifest_sha256=cache._sha(before),catalogue=catalogue,require_production=False)
            self.assertEqual((partial/"samples.jsonl").read_bytes(),raw)
            self.assertEqual((partial/"cache.json").read_bytes(),before)

    def test_resume_requires_explicit_authorization_and_one_worker(self):
        args={"expected_old_manifest_sha256":"0"*64,"old_manifest_evidence_path":"G:/not_written.json","old_writer_code_path":"D:/not_read.py"}
        with self.assertRaisesRegex(RuntimeError,"明确授权"):cache.resume_input_cache(**args)
        with self.assertRaisesRegex(ValueError,"一个CPU worker"):cache.resume_input_cache(**args,explicitly_authorized=True,workers=2)

    def test_source_throttle_covers_binary_read_readinto_and_text_iteration(self):
        self.assertEqual(cache.MAX_SOURCE_RATE,100_000_000)
        class Clock:
            def __init__(self):self.now=0.;self.sleeps=[]
            def time(self):return self.now
            def sleep(self,n):self.sleeps.append(n);self.now+=n
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);binary=root/"source.bin";data=bytes(range(256))*1300;binary.write_bytes(data)
            text=root/"source.jsonl";lines=["足球GT第%d行\n"%i for i in range(40)];text.write_text("".join(lines),encoding="utf-8")
            clock=Clock();original=cache.builtins.open
            with cache.SourceReadThrottle(source_root=root,clock=clock.time,sleep=clock.sleep) as limiter:
                self.assertEqual(binary.read_bytes(),data)
                buffer=bytearray(len(data)+10)
                with binary.open("rb") as f:n=f.readinto(buffer)
                self.assertEqual(n,len(data));self.assertEqual(bytes(buffer[:n]),data)
                with text.open("r",encoding="utf-8") as f:self.assertEqual(list(f),lines)
                result=limiter.measurements()
            self.assertIs(cache.builtins.open,original)
            expected=2*len(data)+len(text.read_bytes())
            self.assertEqual(result["source_bytes_read"],expected)
            self.assertLessEqual(result["largest_raw_read_bytes"],65536)
            self.assertGreaterEqual(clock.now+1e-10,expected/cache.MAX_SOURCE_RATE)
            self.assertTrue(all(0<n<=1 for n in clock.sleeps))

    def test_throttle_guard_crosses_chunk_deadline_without_negative_sleep(self):
        now=[0.];sleeps=[]
        def check():now[0]+=.001
        def sleep(seconds):
            if seconds<0:raise ValueError("sleep length must be non-negative")
            sleeps.append(seconds);now[0]+=seconds
        budget=cache._ReadBudget(100_000_000,check=check,clock=lambda:now[0],sleep=sleep)
        budget.charge(3279)
        self.assertEqual(budget.bytes_read,3279)
        self.assertEqual(sleeps,[])
        self.assertEqual(now[0],.001)

    def test_throttle_pause_and_STOP_checks_are_bounded(self):
        now=[0.];pauses=[True,True,False];guards=[]
        def sleep(n):self.assertLessEqual(n,1.);now[0]+=n
        budget=cache._ReadBudget(cache.MAX_SOURCE_RATE,should_pause=lambda:pauses.pop(0) if pauses else False,
            check=lambda:guards.append(now[0]),clock=lambda:now[0],sleep=sleep)
        budget.wait_ready();self.assertEqual(now[0],2.);self.assertGreaterEqual(len(guards),3)
        for bad in (0,cache.MAX_SOURCE_RATE+1,150_000_000,True):
            with self.assertRaises(ValueError):cache.SourceReadThrottle(bad)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"data";path.write_bytes(b"safe")
            original=cache.builtins.open
            with self.assertRaisesRegex(RuntimeError,"STOP"):
                with cache.SourceReadThrottle(source_root=Path(temporary),check=lambda:(_ for _ in ()).throw(RuntimeError("STOP"))):path.read_bytes()
            self.assertIs(cache.builtins.open,original)

    def test_per_seed_capacity_and_single_active_window(self):
        counts={f"seed_{i:08x}":1500 for i in range(16)};counts["seed_00000000"]=1649
        result=cache.seed_cache_requirements(counts)
        self.assertEqual(result["images_npy_bytes"],66767155200+16*128)
        self.assertEqual(result["count"],24149)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);arrays={k:{"path":"arrays/"+k+".npy","count":2} for k in ("seed_00000001","seed_00000002")}
            store=cache._SeedArrayStore(root,arrays,mode="r+",create=True);image=np.full((3,960,960),71,np.uint8)
            store.write_exact("seed_00000001",0,image);old=store._active
            store.write_exact("seed_00000002",0,image)
            self.assertIsNone(old._view);self.assertNotEqual(old,store._active)
            self.assertLessEqual(store.max_mapped_bytes,32*cache.ROW_BYTES)
            self.assertTrue(np.array_equal(store.read_copy("seed_00000001",0),image));store.close()

    def test_v2_pack_consumer_preserves_pixels_metadata_and_augmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue,original,request,v1,receipt,source=fixture(root)
            m=json.loads((v1/"cache.json").read_bytes());dependencies={row["uid"]:row["source_files"] for row in m["rows"]}
            v2=root/"v2"
            result=cache._build_seed_pack(v2,list(catalogue),catalogue,lambda uid:original[request(uid)],
                source_binding=m["source_binding"],source_files=m["source_files"],dependencies=dependencies,production=False,free_reserve=0,max_seconds=30)
            self.assertEqual(set(p.name for p in v2.iterdir()),{"arrays","cache.json","samples.jsonl"})
            self.assertEqual(result["seed_arrays"],1)
            published=json.loads((v2/"cache.json").read_bytes())
            self.assertEqual(published["writer"]["active_windows_max"],1)
            self.assertEqual(published["writer"]["normal_write_active_windows_max"],1)
            reader=cache._CacheReader(v2,require_production=False)
            try:
                for uid,ref in catalogue.items():
                    a=reader.read_prepared(request(uid),ref);b=original[request(uid)]
                    self.assertTrue(cache._equal(a,b))
                    self.assertTrue(cache._equal(prepare_training_sample(a,view="train"),prepare_training_sample(b,view="train")))
            finally:reader.close()

    def test_v1_prefix_migrates_D_to_D_without_regenerating_completed_RGB(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue,original,request,v1,receipt,source=fixture(root);uids=list(catalogue)
            m=json.loads((v1/"cache.json").read_bytes());old_rows=deepcopy(m["rows"]);partial=v1.with_name("pack.partial");v1.rename(partial)
            prefix_line=(partial/"samples.jsonl").read_bytes().splitlines(keepends=True)[0];(partial/"samples.jsonl").write_bytes(prefix_line)
            m.update(state="FAIL_STOP",completed_rows=1,elapsed_seconds=1.);m.pop("rows");cache._write_manifest(partial/"cache.json",m)
            old_manifest=(partial/"cache.json").read_bytes()
            prefix=cache.inspect_partial_cache(v1,uids,expected_manifest_sha256=cache._sha(old_manifest),catalogue=catalogue,require_production=False)
            calls=[]
            def prepared(uid):
                calls.append(uid)
                if uid==uids[0]:raise AssertionError("已完成RGB不得重读")
                return original[request(uid)]
            dependencies={row["uid"]:row["source_files"] for row in old_rows};v2=root/"v2"
            result=cache._build_seed_pack(v2,uids,catalogue,prepared,source_binding=m["source_binding"],source_files=m["source_files"],
                dependencies=dependencies,production=False,free_reserve=0,max_seconds=30,prefix=prefix,recovery_evidence={"fixture":True})
            self.assertEqual(calls,[uids[1]]);self.assertEqual(result["reused_prefix_rows"],1)
            self.assertEqual((partial/"cache.json").read_bytes(),old_manifest)
            self.assertEqual((partial/"samples.jsonl").read_bytes(),prefix_line)
            self.assertTrue((v2/"samples.jsonl").read_bytes().startswith(prefix_line))
            published=json.loads((v2/"cache.json").read_bytes())
            self.assertEqual(published["writer"]["active_windows_max"],2)
            self.assertEqual(published["writer"]["legacy_copy_active_windows_max"],2)
            self.assertEqual(published["writer"]["consumer_active_windows_max"],1)
            self.assertEqual(published["writer"]["combined_mapped_view_ceiling_bytes"],64*cache.ROW_BYTES)
            reader=cache._CacheReader(v2,require_production=False)
            try:
                for uid,ref in catalogue.items():self.assertTrue(cache._equal(reader.read_prepared(request(uid),ref),original[request(uid)]))
            finally:reader.close()

    def test_system_memory_guard_limits_without_resource_mutation(self):
        for physical,commit in ((4*cache.GIB-1,9*cache.GIB),(5*cache.GIB,8*cache.GIB-1)):
            with patch.object(cache,"_system_memory",return_value={"physical_available_bytes":physical,"commit_available_bytes":commit}):
                with self.assertRaises(RuntimeError):cache._guard_memory()
        with patch.object(cache,"_system_memory",return_value={"physical_available_bytes":4*cache.GIB,"commit_available_bytes":8*cache.GIB}):
            self.assertEqual(cache._guard_memory()["commit_available_bytes"],8*cache.GIB)


if __name__=="__main__":unittest.main()
