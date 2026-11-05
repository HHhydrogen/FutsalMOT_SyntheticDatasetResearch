"""CPU调度helper的小临时文件/线程检查；不读真实库，不用GPU。"""
import builtins
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

spec=importlib.util.spec_from_file_location('cache_acceleration',Path(__file__).resolve().parents[1]/'tools/cache_acceleration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AccelerationTests(unittest.TestCase):
    def test_periodic_guard_first_and_interval(self):
        now=[0.];calls=[];guard=m.PeriodicGuard(lambda:calls.append(now[0]),clock=lambda:now[0])
        for _ in range(100):guard()
        self.assertEqual(calls,[0.]);now[0]=.999;guard();self.assertEqual(len(calls),1)
        now[0]=1.;guard();self.assertEqual(calls,[0.,1.])
    def test_periodic_guard_concurrent_heavy_check_once(self):
        calls=[];lock=threading.Lock()
        def check():
            with lock:calls.append(1)
            time.sleep(.01);return 'checked'
        guard=m.PeriodicGuard(check,clock=lambda:0.)
        with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(lambda _:guard(),range(32)))
        self.assertEqual(calls,[1]);self.assertEqual(set(values),{'checked'})
    def test_guard_exception_is_not_swallowed_or_retried(self):
        calls=[]
        def fail():calls.append(1);raise OSError('stop')
        guard=m.PeriodicGuard(fail)
        for _ in range(2):
            with self.assertRaisesRegex(OSError,'stop'):guard()
        self.assertEqual(calls,[1])
    def test_invalid_guards(self):
        for interval in (-1,True,float('inf')):
            with self.assertRaises(ValueError):m.PeriodicGuard(lambda:None,interval)
    def test_binary_read_seek_into_iteration_filno_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'source.bin';raw=b'abc\ndef\n';p.write_bytes(raw)
            before=builtins.open;before_io=io.open
            with m.SerializedSourceReads(root) as source:
                with p.open('rb') as f:
                    self.assertTrue(isinstance(f.fileno(),int));self.assertEqual(f.read(3),b'abc');self.assertEqual(f.seek(0),0)
                    buf=bytearray(4);self.assertEqual(f.readinto(buf),4);self.assertEqual(buf,b'abc\n')
                    self.assertEqual(list(f),[b'def\n'])
                measurement=source.measurements()
            self.assertIs(builtins.open,before);self.assertIs(io.open,before_io)
            self.assertEqual(measurement['source_bytes_read'],11);self.assertEqual(measurement['peak_open_sources'],1)
            self.assertEqual(measurement['open_sources'],0);self.assertFalse(measurement['rate_limit_enabled'])
    def test_text_counts_characters_not_fictional_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'text.txt';p.write_text('球员\n',encoding='utf-8')
            with m.SerializedSourceReads(root) as source:
                with p.open('r',encoding='utf-8') as f:self.assertEqual(f.read(),'球员\n')
                stats=source.measurements()
            self.assertEqual(stats['source_bytes_read'],0);self.assertEqual(stats['text_characters_read'],3)
    def test_cross_worker_source_open_to_close_is_serial(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'s.bin';p.write_bytes(b'x'*1024);active=0;peak=0;lock=threading.Lock()
            def worker(_):
                nonlocal active,peak
                with p.open('rb') as f:
                    with lock:active+=1;peak=max(peak,active)
                    value=f.read();time.sleep(.005)
                    with lock:active-=1
                return value
            with m.SerializedSourceReads(root) as source:
                with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(worker,range(8)))
                stats=source.measurements()
            self.assertEqual(peak,1);self.assertEqual(stats['peak_open_sources'],1);self.assertEqual(stats['source_bytes_read'],8192)
            self.assertTrue(all(v==b'x'*1024 for v in values))
    def test_other_paths_and_writes_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source_root=root/'sources';source_root.mkdir();other=root/'other.txt';p=source_root/'new.txt'
            with m.SerializedSourceReads(source_root) as source:
                other.write_text('outside');p.write_text('write passthrough')
                self.assertEqual(other.read_text(),'outside');stats=source.measurements()
            self.assertEqual(stats['peak_open_sources'],0);self.assertEqual(stats['source_bytes_read'],0)
    def test_context_restores_open_on_read_error(self):
        before=builtins.open;before_io=io.open
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                with m.SerializedSourceReads(td):open(Path(td)/'missing','rb')
        self.assertIs(builtins.open,before);self.assertIs(io.open,before_io)
    def test_explicit_unbuffered_option_kept(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_bytes(b'a')
            with m.SerializedSourceReads(td):
                with open(p,'rb',buffering=0) as f:self.assertEqual(f.read(),b'a')
    def test_serial_source_guard_exception_releases_and_restores(self):
        before=builtins.open
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_bytes(b'a')
            def fail():raise InterruptedError('cancel')
            with self.assertRaises(InterruptedError):
                with m.SerializedSourceReads(td,check=fail):open(p,'rb')
        self.assertIs(builtins.open,before)
    def test_ordered_prefetch_consumes_exact_order_and_bounded_futures(self):
        ordered=[str(i) for i in range(9)];calls=[];lock=threading.Lock()
        def prepare(uid):
            with lock:calls.append(uid)
            time.sleep(.001*(int(uid)%3));return bytearray([int(uid)])
        with m.OrderedPrefetch(ordered,0,prepare,workers=4,max_pending=4) as prefetch:
            results=[prefetch(u) for u in ordered];stats=prefetch.measurements()
        self.assertEqual(results,[bytearray([i]) for i in range(9)]);self.assertEqual(sorted(calls),sorted(ordered));self.assertLessEqual(stats['peak_pending'],4)
    def test_prefix_samples_are_synchronous_and_do_not_repeat_futures(self):
        calls=[];writer=threading.get_ident()
        def prepare(uid):calls.append((uid,threading.get_ident()));return uid
        ordered=[str(i) for i in range(10)]
        with m.OrderedPrefetch(ordered,6,prepare,workers=2,max_pending=2) as prefetch:
            self.assertEqual(prefetch('0'),'0');self.assertEqual(prefetch('3'),'3');self.assertEqual(prefetch('5'),'5')
            values=[prefetch(u) for u in ordered[6:]]
        self.assertEqual(values,ordered[6:]);self.assertEqual([tid for uid,tid in calls if uid in ('0','3','5')],[writer]*3)
        for uid in ordered[6:]:self.assertEqual(sum(u==uid for u,_ in calls),1)
    def test_non_next_pending_does_not_duplicate_prepare(self):
        calls=[]
        def prepare(uid):calls.append(uid);return uid
        with m.OrderedPrefetch(['a','b','c'],0,prepare,max_pending=3) as prefetch:
            self.assertEqual(prefetch('b'),'b');self.assertEqual([prefetch(u) for u in ('a','b','c')],['a','b','c'])
        self.assertEqual(calls.count('b'),1)
    def test_worker_exception_propagates_and_pool_is_closed(self):
        def prepare(uid):
            if uid=='b':raise ValueError('bad sample')
            time.sleep(.002);return uid
        prefetch=m.OrderedPrefetch(['a','b','c'],0,prepare,workers=2,max_pending=2)
        self.assertEqual(prefetch('a'),'a')
        with self.assertRaisesRegex(ValueError,'bad sample'):prefetch('b')
        self.assertTrue(prefetch.measurements()['closed']);self.assertFalse(any(t.is_alive() for t in prefetch._pool._threads))
    def test_sync_prefix_exception_closes_pending_workers(self):
        def prepare(uid):
            if uid=='old':raise ValueError('prefix error')
            time.sleep(.003);return uid
        prefetch=m.OrderedPrefetch(['old','a','b'],1,prepare,workers=2,max_pending=2)
        with self.assertRaises(ValueError):prefetch('old')
        self.assertFalse(any(t.is_alive() for t in prefetch._pool._threads))
    def test_close_cancels_pending_and_waits_active_workers(self):
        started=threading.Event();release=threading.Event();calls=[]
        def prepare(uid):calls.append(uid);started.set();release.wait(1);return uid
        prefetch=m.OrderedPrefetch(['a','b','c','d'],0,prepare,workers=1,max_pending=4)
        self.assertTrue(started.wait(1));timer=threading.Timer(.02,release.set);timer.start();prefetch.close();timer.join()
        self.assertEqual(calls,['a']);self.assertFalse(any(t.is_alive() for t in prefetch._pool._threads));prefetch.close()
    def test_invalid_prefetch_rejected(self):
        for args in ((['a','a'],0),(['a'],-1),(['a'],True)):
            with self.assertRaises(ValueError):m.OrderedPrefetch(*args,prepare=lambda u:u)
