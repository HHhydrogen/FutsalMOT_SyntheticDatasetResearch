"""已获用户新有限预算授权的v2原位续建；不修改旧冻结writer或活动GPU包。"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from . import input_cache as cache


def _once(path, raw):
    path=cache._safe_path(path)
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())


def _dependencies(manifest, ref):
    common={p for p in manifest['source_files'] if Path(p).name in
            ('dataset.json','RELOCATION.json','dataset_manifest.json','validation_receipt.json','G5_complete.json')}
    seed=Path(ref['seed_dir']);camera=seed/ref['camera_id']
    result=sorted(common|{str(seed/'seed_manifest.json'),str(camera/'camera.json'),str(camera/'annotations.jsonl')})
    if any(p not in manifest['source_files'] for p in result):raise ValueError('前缀GT/metadata来源绑定不完整')
    return result


def _inspect(partial, ordered_uids, catalogue, *, expected_manifest_sha256, guard, require_production):
    """只读D前缀；旧FAIL未保存逐行pixel SHA，本次新建基线而非原真源重验。"""
    path=cache._safe_path(partial/'cache.json')
    if path.stat().st_size>cache.MAX_MANIFEST:raise ValueError('旧FAIL manifest超过64MiB')
    raw=path.read_bytes()
    if cache._sha(raw)!=expected_manifest_sha256:raise ValueError('v2旧FAIL manifest SHA不符')
    manifest=json.loads(raw);count=manifest.get('count');first=manifest.get('completed_rows')
    if manifest.get('schema')!=cache.SEED_SCHEMA or manifest.get('state')!='FAIL_STOP':raise ValueError('必须明确v2 FAIL_STOP')
    if type(first) is not int or type(count) is not int or not 0<=first<=count or count!=len(ordered_uids) or len(set(ordered_uids))!=count:
        raise ValueError('已提交数/完整UID池不符')
    if require_production and (manifest.get('kind')!='production' or count!=cache.COUNT):raise ValueError('非完整生产缓存')
    if manifest['implementation']!=cache._implementation():raise ValueError('原缓存预处理实现与依赖改变')
    if set(p.name for p in partial.iterdir())!={'cache.json','arrays','samples.jsonl'}:raise ValueError('v2 partial存在未知文件')
    arrays,locations=cache._seed_plan(ordered_uids,catalogue)
    if require_production and len(arrays)!=16:raise ValueError('正式缓存必须16个seed')
    if {p.name for p in (partial/'arrays').iterdir()}!={Path(a['path']).name for a in arrays.values()}:raise ValueError('数组缺失/未知额外文件')
    identities={str(cache._safe_path(partial/a['path'])):cache._stat(cache._safe_path(partial/a['path'])) for a in arrays.values()}
    cache._safe_path(partial/'arrays');cache._safe_path(partial/'samples.jsonl')
    identities[str(partial/'samples.jsonl')]=cache._stat(partial/'samples.jsonl')
    identities[str(partial/'cache.json')]=cache._stat(partial/'cache.json')
    if identities[str(partial/'samples.jsonl')][2]>cache.MAX_SAMPLES:raise ValueError('原JSONL超界')
    for a in arrays.values():guard();cache._image_header(partial/a['path'],a['count'])
    for binding in manifest['source_files'].values():guard();cache._check_binding(binding,hash_contents=True)
    rows=[];labels=hashlib.sha256();pixels=hashlib.sha256();offset=0
    store=cache._SeedArrayStore(partial,arrays,mode='r')
    try:
        with (partial/'samples.jsonl').open('rb') as f:
            while True:
                guard();line=f.readline(cache.MAX_ROW+1)
                if not line:break
                i=len(rows)
                if i>=first or len(line)>cache.MAX_ROW or not line.endswith(b'\n'):raise ValueError('JSONL有未知/不完整尾部，拒绝截断猜提交')
                saved=json.loads(line);uid=ordered_uids[i]
                if saved.get('uid')!=uid or saved.get('row')!=i:raise ValueError('前缀UID/row顺序改变')
                metadata=cache._decode(saved['metadata'])
                if metadata.get('uid')!=uid or metadata.get('domain')!='S' or metadata.get('source_reference')!=catalogue[uid]:raise ValueError('前缀metadata来源不符')
                if not cache._equal(metadata,cache._decode(cache._encode(metadata))):raise ValueError('前缀metadata roundtrip失败')
                image=metadata['image_source_binding']
                if cache._stat(image['path'])!=list(image['stat']):raise ValueError('前缀原RGB stat变化')
                location=locations[uid];data=store.read_copy(location['seed_id'],location['seed_row']).tobytes()
                labels.update(line);pixels.update(data)
                rows.append({'uid':uid,'row':i,'offset':offset,'bytes':len(line),'sha256':cache._sha(line),
                             'pixel_sha256':cache._sha(data),'source_files':_dependencies(manifest,catalogue[uid]),**location})
                offset+=len(line)
    finally:store.close()
    if len(rows)!=first or offset!=identities[str(partial/'samples.jsonl')][2]:raise ValueError('原JSONL完整行数与提交数不齐')
    for p,identity in identities.items():
        if cache._stat(p)!=identity:raise ValueError('前缀检查期间文件变化')
    return {'manifest':manifest,'raw_manifest':raw,'rows':rows,'arrays':arrays,'locations':locations,'identities':identities,
            'prefix_samples_sha256':labels.hexdigest(),'prefix_pixels_sha256_new_baseline':pixels.hexdigest(),
            'old_pixel_sha_durable':False,'mapped_view_peak_bytes':store.max_mapped_bytes}


def _resume_pack(target, ordered_uids, catalogue, prepared_factory, *, expected_manifest_sha256,
                 evidence_path, guard=None, progress=None, progress_callback=None, max_seconds=3600,
                 free_reserve=cache.MIN_FREE, require_production=False, original_sample_checks=False):
    """只供显式生产wrapper或有界CPU夹具；原位append，不复制NPY/不截断未知尾。"""
    target=cache._safe_path(target);partial=cache._safe_path(target.with_name(target.name+'.partial'));evidence_path=cache._safe_path(evidence_path)
    if target.exists():raise FileExistsError('正式缓存已存在，不覆盖')
    if not partial.is_dir():raise FileNotFoundError('没有待恢复v2.partial')
    if evidence_path.exists() or evidence_path.with_name('prefix_resume_receipt.json').exists():raise FileExistsError('恢复证据已存在，不猜重放')
    started=time.monotonic()
    def check():
        if guard is not None:guard()
        if time.monotonic()-started>=max_seconds:raise TimeoutError('新冻结缓存续建预算已到')
        if shutil.disk_usage(partial).free<free_reserve:raise RuntimeError('续建D余量不足100GiB')
    check()
    inspected=_inspect(partial,ordered_uids,catalogue,expected_manifest_sha256=expected_manifest_sha256,guard=check,require_production=require_production)
    manifest=deepcopy(inspected['manifest']);rows=inspected['rows'];first=len(rows);arrays=inspected['arrays'];locations=inspected['locations']
    checks=[]
    if original_sample_checks and first:
        store=cache._SeedArrayStore(partial,arrays,mode='r')
        try:
            for i in sorted({0,first//2,first-1}):
                check();uid=ordered_uids[i];sample=prepared_factory(uid);loc=locations[uid]
                if not np.array_equal(store.read_copy(loc['seed_id'],loc['seed_row']),sample['img'].numpy()):raise RuntimeError('前缀3张原RGB抽样不一致')
                checks.append({'uid':uid,'row':i,'original_RGB_letterbox_exact':True})
        finally:store.close()
    # 原FAIL原bytes先写不可变G证据，再允许原位修改manifest和未提交尾。
    evidence_path.parent.mkdir(parents=True,exist_ok=True);_once(evidence_path,inspected['raw_manifest'])
    receipt={'schema':'futsalmot_explicit_v2_prefix_resume_v1','original_manifest_sha256':expected_manifest_sha256,
             'original_FAIL_manifest':str(evidence_path),'completed_rows':first,'file_identities_before_resume':inspected['identities'],
             'prefix_samples_sha256':inspected['prefix_samples_sha256'],'prefix_pixels_sha256_new_baseline':inspected['prefix_pixels_sha256_new_baseline'],
             'old_pixel_sha_durable':False,'prefix_rows_new_baseline':deepcopy(rows),'pixel_evidence_scope':'inherited frozen writer write/read exact; current D prefix new baseline',
             'original_RGB_sample_checks':checks,'prefix_sample_bytes_preserved':True,'uncommitted_tail_policy':'write only remaining original ordered UID rows; no truncation/deletion',
             'new_resume_module_sha256':cache._sha(Path(__file__).read_bytes()),'row_count':len(ordered_uids),'max_seconds':max_seconds}
    _once(evidence_path.with_name('prefix_resume_receipt.json'),cache._json(receipt))
    for p,identity in inspected['identities'].items():
        if cache._stat(p)!=identity:raise ValueError('验证后原文件身份变化')
    manifest.update(state='BUILDING',completed_rows=first,error=None,
                    resume_v2=receipt,writer={'pid':os.getpid(),'version':'per_seed_windowed_v2_explicit_resume','window_rows':cache.WINDOW_ROWS,'CPU_workers':1,
                                             'normal_write_active_windows_max':1,'combined_mapped_view_ceiling_bytes':cache.WINDOW_ROWS*cache.ROW_BYTES})
    store=cache._SeedArrayStore(partial,arrays,mode='r+')
    def report(state):
        if progress_callback is not None:progress_callback({'state':state,'completed_rows':len(rows),'source_IO':None if progress is None else progress(),
            'mapped_view_peak_bytes':store.max_mapped_bytes,'reused_prefix_rows':first})
    try:
        cache._write_manifest(partial/'cache.json',manifest);report('BUILDING')
        with (partial/'samples.jsonl').open('ab') as out:
            for i in range(first,len(ordered_uids)):
                check();uid=ordered_uids[i];sample=prepared_factory(uid);image=sample.get('img')
                if (not isinstance(image,torch.Tensor) or image.device.type!='cpu' or image.dtype!=torch.uint8 or tuple(image.shape)!=(3,960,960)
                    or sample.get('uid')!=uid or sample.get('domain')!='S' or sample.get('random_augmentation_implemented') is not False
                    or sample.get('source_reference')!=catalogue[uid] or sample.get('source_reference',{}).get('split')!='train'
                    or sample.get('geometry',{}).get('letterbox_candidate_only') is not True):raise ValueError('续建仅接受原增强前S train LetterBox')
                binding=sample['image_source_binding']
                if cache._stat(binding['path'])!=list(binding['stat']):raise ValueError('新行原RGB stat变化')
                metadata={k:deepcopy(v) for k,v in sample.items() if k!='img' and k not in cache._DYNAMIC};encoded=cache._encode(metadata)
                if not cache._equal(metadata,cache._decode(encoded)):raise RuntimeError('新metadata roundtrip失败')
                raw=cache._json({'uid':uid,'row':i,'metadata':encoded});pixels=image.detach().contiguous().numpy();location=locations[uid]
                if len(raw)>cache.MAX_ROW or out.tell()+len(raw)>cache.MAX_SAMPLES:raise ValueError('新JSONL超冻结上界')
                store.write_exact(location['seed_id'],location['seed_row'],pixels)
                row={'uid':uid,'row':i,'offset':out.tell(),'bytes':len(raw),'sha256':cache._sha(raw),'pixel_sha256':cache._sha(pixels.tobytes()),
                     'source_files':_dependencies(manifest,catalogue[uid]),**location}
                out.write(raw);out.flush();rows.append(row)
                if (i+1)%cache.WINDOW_ROWS==0:
                    store.close();out.flush();os.fsync(out.fileno());manifest.update(completed_rows=len(rows),elapsed_seconds=time.monotonic()-started,
                        source_IO=None if progress is None else progress(),mapped_view_peak_bytes=store.max_mapped_bytes)
                    cache._write_manifest(partial/'cache.json',manifest);report('BUILDING')
            store.close();out.flush();os.fsync(out.fileno())
        store.close();check()
        for binding in manifest['source_files'].values():check();cache._check_binding(binding)
        if manifest['implementation']!=cache._implementation():raise RuntimeError('续建期间预处理实现变化')
        array_files={seed:{**a,'stat':cache._stat(partial/a['path'])} for seed,a in arrays.items()}
        requirements=cache.seed_cache_requirements({k:a['count'] for k,a in arrays.items()})
        image_bytes=sum(a['stat'][2] for a in array_files.values())
        if image_bytes!=requirements['images_npy_bytes']:raise ValueError('NPY总量不齐')
        manifest.update(state='READY',completed_rows=len(rows),rows=rows,array_files=array_files,samples_file={'stat':cache._stat(partial/'samples.jsonl')},
            elapsed_seconds=time.monotonic()-started,source_IO=None if progress is None else progress(),mapped_view_peak_bytes=store.max_mapped_bytes,
            acceptance={'legacy_prefix_rows_copied_D_to_D':(manifest.get('recovery') or {}).get('completed_rows',0),'v2_prefix_rows_reused_in_place':first,
                        'original_v2_FAIL_manifest_retained':True,'prefix_current_D_baseline_verified':True,'new_rows_source_letterbox_write_read_exact':True,
                        'all_metadata_roundtrip_exact':True,'random_augmentation_cached':False})
        # D端前缀原bytes再次有界读回；不重复原4K图像读取。
        prefix_sha=hashlib.sha256();remaining=inspected['identities'][str(partial/'samples.jsonl')][2]
        with (partial/'samples.jsonl').open('rb') as f:
            while remaining:
                check();block=f.read(min(1024**2,remaining));prefix_sha.update(block);remaining-=len(block)
                if not block:raise RuntimeError('前缀JSONL意外变短')
        if prefix_sha.hexdigest()!=inspected['prefix_samples_sha256']:raise RuntimeError('原JSONL前缀bytes变化')
        cache._write_manifest(partial/'cache.json',manifest)
        total=image_bytes+(partial/'samples.jsonl').stat().st_size+(partial/'cache.json').stat().st_size
        if total>cache.MAX_BYTES:raise RuntimeError('续建缓存超过128GiB')
        # 所有可能失败的结果IO/计时/进度计算先完成，再执行不可回滚的同卷发布。
        result={'state':'READY','path':str(target),'count':len(rows),'seed_arrays':len(arrays),'reused_prefix_rows':first,
                'manifest_sha256':cache._sha((partial/'cache.json').read_bytes()),'total_bytes':total,'elapsed_seconds':time.monotonic()-started,
                'source_IO':None if progress is None else progress(),'mapped_view_peak_bytes':store.max_mapped_bytes,
                'resume_receipt':str(evidence_path.with_name('prefix_resume_receipt.json')),'post_publish_progress_error':None}
        if target.exists():raise FileExistsError('READY发布目标已存在')
        os.rename(partial,target)
        # 发布后仅调用可选显示回调，任何回调错误均不会回写已不存在的partial。
        try:report('READY')
        except BaseException as exc:result['post_publish_progress_error']={'type':type(exc).__name__,'message':str(exc)}
        return result
    except BaseException as exc:
        store.close();manifest.pop('rows',None);manifest.update(state='FAIL_STOP',completed_rows=len(rows),elapsed_seconds=time.monotonic()-started,
            error={'type':type(exc).__name__,'message':str(exc)},source_IO=None if progress is None else progress())
        cache._write_manifest(partial/'cache.json',manifest);report('FAIL_STOP');raise
    finally:store.close()



def _combined_actual_bytes(target):
    """仅stat已知v1三文件与v2固定文件，不遍历数据集/不跟reparse。"""
    total=0
    legacy=cache.DEFAULT_TARGET.with_name(cache.DEFAULT_TARGET.name+'.partial')
    for name in ('cache.json','images.npy','samples.jsonl'):
        p=cache._safe_path(legacy/name)
        if p.exists():total+=p.stat().st_size
    actual=target if target.exists() else target.with_name(target.name+'.partial')
    for name in ('cache.json','samples.jsonl'):
        p=cache._safe_path(actual/name)
        if p.exists():total+=p.stat().st_size
    arrays=cache._safe_path(actual/'arrays')
    # 只有既定16个seed数组所在的一层，拒绝任何额外目录或非NPY文件。
    for p in arrays.iterdir():
        p=cache._safe_path(p)
        if not p.is_file() or p.suffix!='.npy':raise ValueError('缓存arrays有未知对象')
        total+=p.stat().st_size
    if total>cache.MAX_BYTES:raise RuntimeError('v1+v2实际缓存超过128GiB')
    return total


def resume_seed_cache(*, expected_manifest_sha256, original_manifest_evidence_path, old_writer_code_path,
                      previous_writer_identity, max_seconds, approved_max_seconds, deadline_utc,
                      index_path=cache.SOURCE_INDEX,target=cache.SEED_TARGET,explicitly_authorized=False,
                      rate_bytes_per_second=cache.MAX_SOURCE_RATE,should_pause=None,stop_check=None,progress_callback=None):
    """明确新许可的新有限续建；不继承旧4h截止、不自动再次重试。"""
    if explicitly_authorized is not True:raise RuntimeError('v2续建必须用户新明确授权')
    if type(max_seconds) is not int or type(approved_max_seconds) is not int or not 1<=max_seconds<=approved_max_seconds:raise ValueError('冻结新有限预算不合法')
    absolute=datetime.fromisoformat(deadline_utc.replace('Z','+00:00'))
    if absolute.tzinfo is None or absolute>datetime(2026,11,7,23,55,tzinfo=timezone.utc):raise ValueError('新截止不能超过原研究全局截止')
    target=cache._safe_path(target);index_path=cache._safe_path(index_path);evidence=cache._safe_path(original_manifest_evidence_path)
    if target!=cache.SEED_TARGET.absolute() or index_path!=cache.SOURCE_INDEX.absolute():raise ValueError('仅原canonical缓存目标和来源')
    if not evidence.is_relative_to(Path('G:/FutsalMOT_Training').absolute()):raise ValueError('原FAIL证据必须存G训练区域')
    partial=cache._safe_path(target.with_name(target.name+'.partial'));old_path=cache._safe_path(partial/'cache.json')
    if old_path.stat().st_size>cache.MAX_MANIFEST:raise ValueError('旧FAIL manifest超过64MiB')
    old_raw=old_path.read_bytes();old=json.loads(old_raw)
    if cache._sha(old_raw)!=expected_manifest_sha256:raise ValueError('旧FAIL SHA不符')
    if cache._sha(cache._safe_path(old_writer_code_path).read_bytes())!=old['implementation']['code_sha256']['input_cache.py']:raise ValueError('旧冻结writer实现不符')
    import psutil
    identity=previous_writer_identity
    if not isinstance(identity,dict) or type(identity.get('pid')) is not int or not isinstance(identity.get('create_time'),(int,float)) or not isinstance(identity.get('cmdline'),list):raise ValueError('必须精确旧writer身份')
    if identity['pid']!=old['writer']['pid']:raise ValueError('旧writer PID不匹配')
    try:
        prior=psutil.Process(identity['pid'])
        if abs(prior.create_time()-identity['create_time'])<.001:raise RuntimeError('旧writer仍存活，拒绝并发续建')
    except psutil.NoSuchProcess:pass
    lock=target.parent/('.'+target.name+'.resume.lock')
    lockraw=cache._json({'pid':os.getpid(),'create_time':psutil.Process().create_time(),'cmdline':psutil.Process().cmdline(),'target':str(target)})
    _once(lock,lockraw)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
        import cv2
        cv2.setNumThreads(1)
        started=time.monotonic();deadline=min(started+max_seconds,started+(absolute-datetime.now(timezone.utc)).total_seconds())
        last_capacity_check=[float('-inf')]
        def basic_guard():
            if stop_check is not None:stop_check()
            if time.monotonic()>=deadline:raise TimeoutError('新冻结缓存续建截止已到')
            cache._guard_memory()
            now=time.monotonic()
            if now-last_capacity_check[0]>=10:
                _combined_actual_bytes(target);last_capacity_check[0]=now
            if shutil.disk_usage(target.parent).free<cache.MIN_FREE:raise RuntimeError('续建D余量不足100GiB')
        basic_guard()
        with cache.SourceReadThrottle(rate_bytes_per_second,should_pause=should_pause,check=basic_guard) as source_io:
            def guard():basic_guard();source_io.wait_ready()
            from .runner import production_closed
            from .catalogues import synthetic_bbox_catalogue
            production_closed();guard();source=synthetic_bbox_catalogue(index_path,split='train')
            if source['binding']!=old['source_binding']:raise ValueError('完整S train来源绑定变化')
            original=cache.RequestBBoxDataset(source['catalogue'],view='train',synthetic_reader_options={'max_indexed_cameras':7,'max_indexed_frames':3000})
            def prepared(uid):
                guard();return original[{'uid':uid,'domain':'S','stream':'S','data_domain':'S','epoch':0,'exposure_index':0,'augmentation_seed':0}]
            return _resume_pack(target,source['ordered_uids'],source['catalogue'],prepared,
                expected_manifest_sha256=expected_manifest_sha256,evidence_path=evidence,guard=guard,progress=source_io.measurements,
                progress_callback=progress_callback,max_seconds=max(.001,deadline-time.monotonic()),require_production=True,original_sample_checks=True)
    finally:
        if lock.exists() and lock.read_bytes()==lockraw:
            retired=evidence.with_name('retired_writer_lock.json');_once(retired,lockraw);lock.unlink()
