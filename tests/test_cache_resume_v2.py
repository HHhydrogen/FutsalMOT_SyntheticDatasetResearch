"""v2显式续建的有界CPU夹具；不读真实G库/不启动CUDA。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_ENV_KEYS=('YOLO_CONFIG_DIR','MPLCONFIGDIR','YOLO_OFFLINE','CUDA_VISIBLE_DEVICES')
_OLD={k:os.environ.get(k) for k in _ENV_KEYS}
_CONFIG=tempfile.TemporaryDirectory(prefix='futsalmot-resume-v2-config-')
os.environ.update(YOLO_CONFIG_DIR=str(Path(_CONFIG.name)/'yolo'),MPLCONFIGDIR=str(Path(_CONFIG.name)/'mpl'),YOLO_OFFLINE='true',CUDA_VISIBLE_DEVICES='-1')
import numpy as np
import torch
from futsalmot import input_cache as c
from futsalmot import cache_resume_v2 as m


def tearDownModule():
    _CONFIG.cleanup()
    for k,v in _OLD.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v


def fixture(root):
    seed=root/'source/seed_00007546';cam=seed/'C1';cam.mkdir(parents=True)
    for p in (seed/'seed_manifest.json',cam/'camera.json',cam/'annotations.jsonl'):p.write_text('{}\n',encoding='utf-8')
    image=cam/'rgb.fake';image.write_bytes(b'fixed-rgb-stat')
    files={str(p):c._file_binding(p) for p in (seed/'seed_manifest.json',cam/'camera.json',cam/'annotations.jsonl')}
    catalogue={};samples={}
    for i in range(3):
        uid=f'detect:synthetic:fixture:v1:seed_00007546:C1:{i+1:06}'
        ref={'domain':'S','split':'train','seed_dir':str(seed),'camera_id':'C1','frame_index':i+1};catalogue[uid]=ref
        samples[uid]={'uid':uid,'domain':'S','source_reference':deepcopy(ref),'image_source_binding':{'path':str(image),'stat':c._stat(image)},
            'random_augmentation_implemented':False,'geometry':{'letterbox_candidate_only':True},'img':torch.full((3,960,960),i+10,dtype=torch.uint8),
            'boxes':torch.tensor([[.1,.2,.3,.4]],dtype=torch.float32)}
    ordered=list(catalogue);calls=0
    def fail_after_one(uid):
        if uid!=ordered[0]:raise TimeoutError('original fixture limit')
        return samples[uid]
    target=root/'pack'
    try:c._build_seed_pack(target,ordered,catalogue,fail_after_one,source_binding={'fixture':True},source_files=files,
        dependencies={u:list(files) for u in ordered},production=False,free_reserve=0,max_seconds=30)
    except TimeoutError:pass
    partial=target.with_name('pack.partial');raw=(partial/'cache.json').read_bytes()
    return {'target':target,'partial':partial,'ordered':ordered,'catalogue':catalogue,'samples':samples,'raw':raw,'sha':c._sha(raw),'evidence':root/'evidence/original_FAIL.json'}


def run(f,**kw):
    args=dict(expected_manifest_sha256=f['sha'],evidence_path=f['evidence'],free_reserve=0,max_seconds=30)
    args.update(kw)
    return m._resume_pack(f['target'],f['ordered'],f['catalogue'],lambda u:f['samples'][u],**args)


class ResumeV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='futsalmot-resume-v2-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.f=fixture(self.root)
        self.cuda=patch('torch.cuda._lazy_init',side_effect=AssertionError('no CUDA'));self.cuda.start();self.addCleanup(self.cuda.stop)
    def test_resume_preserves_prefix_bytes_and_ready_reader_exact(self):
        f=self.f;before=(f['partial']/'samples.jsonl').read_bytes();result=run(f)
        self.assertEqual(result['state'],'READY');self.assertEqual(result['reused_prefix_rows'],1)
        self.assertTrue((f['target']/'samples.jsonl').read_bytes().startswith(before));self.assertEqual(f['evidence'].read_bytes(),f['raw'])
        receipt=json.loads(f['evidence'].with_name('prefix_resume_receipt.json').read_text());self.assertFalse(receipt['old_pixel_sha_durable'])
        reader=c._CacheReader(f['target'],require_production=False,expected_manifest_sha256=result['manifest_sha256'])
        try:
            for uid in f['ordered']:
                actual=reader.read_prepared({'uid':uid,'domain':'S','stream':'S','data_domain':'S'},f['catalogue'][uid])
                self.assertTrue(torch.equal(actual['img'],f['samples'][uid]['img']));self.assertTrue(c._equal(actual['boxes'],f['samples'][uid]['boxes']))
        finally:reader.close()
    def test_wrong_manifest_sha_rejected_without_mutation(self):
        with self.assertRaisesRegex(ValueError,'SHA'):run(self.f,expected_manifest_sha256='0'*64)
        self.assertEqual((self.f['partial']/'cache.json').read_bytes(),self.f['raw']);self.assertFalse(self.f['evidence'].exists())
    def test_count_mismatch_rejected(self):
        f=self.f;d=json.loads(f['raw']);d['completed_rows']=2;(f['partial']/'cache.json').write_bytes(c._json(d));f['sha']=c._sha((f['partial']/'cache.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'提交数不齐'):run(f)
    def test_incomplete_tail_rejected_no_truncation(self):
        p=self.f['partial']/'samples.jsonl'
        with p.open('ab') as out:out.write(b'{')
        raw=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'未知/不完整'):run(self.f)
        self.assertEqual(raw,p.read_bytes())
    def test_extra_complete_tail_rejected(self):
        p=self.f['partial']/'samples.jsonl'
        with p.open('ab') as out:out.write(b'{}\n')
        raw=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'未知/不完整'):run(self.f)
        self.assertEqual(raw,p.read_bytes())
    def test_changed_source_metadata_rejected(self):
        p=Path(next(iter(json.loads(self.f['raw'])['source_files'])));p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'来源'):run(self.f)
    def test_wrong_uid_order_rejected(self):
        f=self.f;f['ordered'][0],f['ordered'][1]=f['ordered'][1],f['ordered'][0]
        with self.assertRaisesRegex(ValueError,'UID/row'):run(f)
    def test_unknown_array_rejected(self):
        (self.f['partial']/'arrays/extra.npy').write_bytes(b'unknown')
        with self.assertRaisesRegex(ValueError,'未知额外'):run(self.f)
    def test_failure_after_one_new_commit_keeps_exact_boundary(self):
        f=self.f;f['samples'][f['ordered'][2]]['domain']='R'
        with self.assertRaisesRegex(ValueError,'续建仅接受'):run(f)
        d=json.loads((f['partial']/'cache.json').read_bytes());self.assertEqual(d['state'],'FAIL_STOP');self.assertEqual(d['completed_rows'],2)
        self.assertEqual(len((f['partial']/'samples.jsonl').read_bytes().splitlines()),2);self.assertEqual(f['evidence'].read_bytes(),f['raw'])
    def test_stop_before_mutation_keeps_original_failure(self):
        def stop():raise InterruptedError('user stop')
        with self.assertRaises(InterruptedError):run(self.f,guard=stop)
        self.assertEqual((self.f['partial']/'cache.json').read_bytes(),self.f['raw'])
    def test_deadline_before_mutation(self):
        with self.assertRaises(TimeoutError):run(self.f,max_seconds=0)
        self.assertEqual((self.f['partial']/'cache.json').read_bytes(),self.f['raw'])
    def test_disk_reserve_guard(self):
        usage=type('Usage',(),{'free':0})()
        with patch.object(m.shutil,'disk_usage',return_value=usage):
            with self.assertRaisesRegex(RuntimeError,'余量'):run(self.f,free_reserve=1)
    def test_prior_failure_evidence_must_be_unique(self):
        self.f['evidence'].parent.mkdir();self.f['evidence'].write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):run(self.f)
        self.assertEqual(self.f['evidence'].read_bytes(),b'preserve')
    def test_no_whole_array_map_and_bounded_peak(self):
        with patch.object(c.np.lib.format,'open_memmap',side_effect=AssertionError('whole map')):result=run(self.f)
        self.assertLessEqual(result['mapped_view_peak_bytes'],c.WINDOW_ROWS*c.ROW_BYTES)
    def test_missing_explicit_authorization_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'新明确授权'):m.resume_seed_cache(expected_manifest_sha256='0'*64,original_manifest_evidence_path='unused',
            old_writer_code_path='unused',previous_writer_identity={},max_seconds=1,approved_max_seconds=1,deadline_utc='2026-11-03T00:00:00Z')
    def test_new_budget_must_be_finite_explicit(self):
        for seconds in (True,0,-1,1.1):
            with self.assertRaises(ValueError):m.resume_seed_cache(expected_manifest_sha256='0'*64,original_manifest_evidence_path='unused',
                old_writer_code_path='unused',previous_writer_identity={},max_seconds=seconds,approved_max_seconds=1,deadline_utc='2026-11-03T00:00:00Z',explicitly_authorized=True)
    def test_deadline_cannot_extend_global(self):
        with self.assertRaisesRegex(ValueError,'全局'):m.resume_seed_cache(expected_manifest_sha256='0'*64,original_manifest_evidence_path='unused',
            old_writer_code_path='unused',previous_writer_identity={},max_seconds=1,approved_max_seconds=1,deadline_utc='2026-11-08T00:00:00Z',explicitly_authorized=True)
    def test_memory_guard_stays_four_eight_GiB(self):
        with patch.object(c,'_system_memory',return_value={'physical_available_bytes':4*c.GIB-1,'commit_available_bytes':20*c.GIB}):
            with self.assertRaises(RuntimeError):c._guard_memory()
        with patch.object(c,'_system_memory',return_value={'physical_available_bytes':20*c.GIB,'commit_available_bytes':8*c.GIB-1}):
            with self.assertRaises(RuntimeError):c._guard_memory()

    def test_published_ready_survives_display_callback_failure(self):
        def callback(value):
            if value['state']=='READY':raise RuntimeError('display only')
        result=run(self.f,progress_callback=callback)
        self.assertEqual(result['state'],'READY');self.assertEqual(result['post_publish_progress_error']['message'],'display only')
        self.assertTrue(self.f['target'].exists());self.assertFalse(self.f['partial'].exists())
        self.assertEqual(json.loads((self.f['target']/'cache.json').read_text())['state'],'READY')
    def test_pixel_flush_precedes_durable_jsonl(self):
        events=[];real_write=c._SeedArrayStore.write_exact;real_close=c._SeedArrayStore.close;real_fsync=os.fsync
        def write(store,*args):events.append('write');return real_write(store,*args)
        def close(store):events.append('close');return real_close(store)
        def sync(fd):events.append('fsync');return real_fsync(fd)
        with patch.object(c._SeedArrayStore,'write_exact',write),patch.object(c._SeedArrayStore,'close',close),patch.object(m.os,'fsync',sync):run(self.f)
        for i,action in enumerate(events):
            if action=='write':
                next_sync=events.index('fsync',i+1)
                self.assertIn('close',events[i+1:next_sync])
    def test_combined_known_cache_capacity_guard(self):
        f=self.f;legacy=self.root/'legacy.partial';legacy.mkdir();(legacy/'images.npy').write_bytes(b'x'*20)
        with patch.object(c,'DEFAULT_TARGET',self.root/'legacy'),patch.object(c,'MAX_BYTES',10):
            with self.assertRaisesRegex(RuntimeError,r'v1\+v2'):m._combined_actual_bytes(f['target'])
    def test_manifest_is_bounded_before_read(self):
        with patch.object(c,'MAX_MANIFEST',1):
            with self.assertRaisesRegex(ValueError,'64MiB'):run(self.f)
    def test_prefix_receipt_carries_new_not_historical_pixel_baseline(self):
        run(self.f);receipt=json.loads(self.f['evidence'].with_name('prefix_resume_receipt.json').read_text())
        self.assertFalse(receipt['old_pixel_sha_durable']);self.assertEqual(len(receipt['prefix_rows_new_baseline']),1)
        self.assertEqual(receipt['prefix_rows_new_baseline'][0]['pixel_sha256'],c._sha(self.f['samples'][self.f['ordered'][0]]['img'].numpy().tobytes()))

    def test_no_manifest_io_after_publish(self):
        target_manifest=self.f['target']/'cache.json';original=Path.read_bytes
        def read(path):
            if path==target_manifest:raise OSError('published manifest read forbidden')
            return original(path)
        with patch.object(Path,'read_bytes',read):result=run(self.f)
        self.assertEqual(result['state'],'READY');self.assertEqual(len(result['manifest_sha256']),64)
        self.assertTrue(self.f['target'].exists());self.assertFalse(self.f['partial'].exists())
    def test_progress_io_failure_after_publish_is_display_only(self):
        def progress():
            if self.f['target'].exists():raise OSError('published progress unavailable')
            return {'bytes':123}
        result=run(self.f,progress=progress,progress_callback=lambda _:None)
        self.assertEqual(result['state'],'READY');self.assertEqual(result['source_IO'],{'bytes':123})
        self.assertEqual(result['post_publish_progress_error']['type'],'OSError')
        self.assertTrue(self.f['target'].exists());self.assertFalse(self.f['partial'].exists())
    def test_total_bytes_uses_final_manifest_actual_stat(self):
        result=run(self.f);target=self.f['target']
        actual=sum(p.stat().st_size for p in (target/'arrays').iterdir())+(target/'cache.json').stat().st_size+(target/'samples.jsonl').stat().st_size
        self.assertEqual(result['total_bytes'],actual)
