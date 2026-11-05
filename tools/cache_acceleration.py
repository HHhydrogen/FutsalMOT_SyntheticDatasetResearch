"""缓存续建的CPU调度工具；仅本进程源文件串行I/O与有界有序预取。"""
from __future__ import annotations
import builtins
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
from pathlib import Path
import threading
import time


class PeriodicGuard:
    """多worker共享的至多每interval秒一次重检查；第一次立即检查，失败保持可见。"""
    def __init__(self,check,interval=1.0,clock=time.monotonic):
        if not callable(check) or not callable(clock) or isinstance(interval,bool) or not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval<0:
            raise ValueError('guard需callable与有限非负interval')
        self.check=check;self.interval=float(interval);self.clock=clock;self._lock=threading.Lock()
        self._last=float('-inf');self._result=None;self._error=None
    def __call__(self):
        with self._lock:
            if self._error is not None:raise self._error
            now=self.clock()
            if now-self._last>=self.interval:
                try:self._result=self.check()
                except BaseException as exc:self._error=exc;raise
                self._last=self.clock()
            return self._result


class _LockedSource:
    def __init__(self,raw,owner):self._raw=raw;self._owner=owner;self._closed=False
    def __getattr__(self,name):return getattr(self._raw,name)
    def __enter__(self):return self
    def __exit__(self,*args):self.close();return False
    def _read(self,method,*args,**kwargs):
        if self._owner.check is not None:self._owner.check()
        started=time.monotonic()
        try:
            result=getattr(self._raw,method)(*args,**kwargs)
        finally:
            elapsed=time.monotonic()-started
            with self._owner._stats_lock:self._owner._read_wall+=elapsed
        if method=='readinto' or method=='readinto1':
            byte_count=0 if result is None else result;characters=0
        elif isinstance(result,(bytes,bytearray,memoryview)):
            byte_count=len(result);characters=0
        elif isinstance(result,str):byte_count=0;characters=len(result)
        elif isinstance(result,list):
            byte_count=sum(len(x) for x in result if isinstance(x,(bytes,bytearray,memoryview)))
            characters=sum(len(x) for x in result if isinstance(x,str))
        else:byte_count=characters=0
        with self._owner._stats_lock:
            self._owner._source_bytes+=byte_count;self._owner._text_characters+=characters
        return result
    def read(self,*args,**kwargs):return self._read('read',*args,**kwargs)
    def read1(self,*args,**kwargs):return self._read('read1',*args,**kwargs)
    def readinto(self,*args,**kwargs):return self._read('readinto',*args,**kwargs)
    def readinto1(self,*args,**kwargs):return self._read('readinto1',*args,**kwargs)
    def readline(self,*args,**kwargs):return self._read('readline',*args,**kwargs)
    def readlines(self,*args,**kwargs):return self._read('readlines',*args,**kwargs)
    def __iter__(self):return self
    def __next__(self):
        line=self.readline()
        if line in (b'',''):raise StopIteration
        return line
    def close(self):
        if self._closed:return
        self._closed=True
        try:self._raw.close()
        finally:
            with self._owner._stats_lock:self._owner._open_count-=1
            self._owner._source_lock.release()
    def __del__(self):
        # 常规使用必须with/close；仅同线程遗忘close时尽力回收，不操作别的进程。
        try:self.close()
        except Exception:pass


class SerializedSourceReads:
    """一个源文件从open到close持RLock，不sleep/限速，不修改源或其他进程。

    只计binary API实际返回字节；文字API单独计字符，不虚称编码字节。
    同线程嵌套源文件允许RLock重入，peak_open_sources如实显示嵌套句柄数。
    """
    def __init__(self,source_root,check=None,buffer_bytes=1024**2):
        if type(buffer_bytes) is not int or buffer_bytes<1 or check is not None and not callable(check):raise ValueError('buffer/check非法')
        self.source_root=os.path.normcase(os.path.abspath(os.fspath(source_root))).rstrip('\\/')
        self.check=check;self.buffer_bytes=buffer_bytes;self._source_lock=threading.RLock();self._stats_lock=threading.Lock()
        self._source_bytes=0;self._text_characters=0;self._read_wall=0.;self._open_count=0;self._peak_open=0;self._entered=False
    def _matches(self,file,mode):
        if isinstance(file,int) or not isinstance(mode,str) or any(c in mode for c in 'wax+') or 'r' not in mode:return False
        try:path=os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(file))))
        except (TypeError,ValueError,OSError):return False
        return path==self.source_root or path.startswith(self.source_root+os.sep)
    def _open(self,original,file,mode='r',buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None):
        if not self._matches(file,mode):return original(file,mode,buffering,encoding,errors,newline,closefd,opener)
        if self.check is not None:self.check()
        self._source_lock.acquire()
        try:
            if self.check is not None:self.check()
            # 保留调用者显式buffering=0或其他值；仅默认buffer改为获准大块缓冲。
            actual_buffer=self.buffer_bytes if buffering==-1 else buffering
            raw=original(file,mode,actual_buffer,encoding,errors,newline,closefd,opener)
            with self._stats_lock:
                self._open_count+=1;self._peak_open=max(self._peak_open,self._open_count)
            return _LockedSource(raw,self)
        except BaseException:self._source_lock.release();raise
    def __enter__(self):
        if self._entered:raise RuntimeError('同一source context不可重入')
        self._entered=True;self._builtins_open=builtins.open;self._io_open=io.open
        def builtin(*args,**kwargs):return self._open(self._builtins_open,*args,**kwargs)
        def iofunc(*args,**kwargs):return self._open(self._io_open,*args,**kwargs)
        self._builtin_wrapper=builtin;self._io_wrapper=iofunc;builtins.open=builtin;io.open=iofunc
        return self
    def __exit__(self,*args):
        # 仅恢复本context的patch，不覆盖无关第三方后续patch。
        if builtins.open is self._builtin_wrapper:builtins.open=self._builtins_open
        if io.open is self._io_wrapper:io.open=self._io_open
        self._entered=False
        return False
    def measurements(self):
        with self._stats_lock:
            return {'source_bytes_read':self._source_bytes,'read_wall_seconds':self._read_wall,'text_characters_read':self._text_characters,
                    'peak_open_sources':self._peak_open,'open_sources':self._open_count,'buffer_bytes':self.buffer_bytes,
                    'rate_limit_enabled':False,'scope':'this_process_builtins_io_read_only_source_root_serial_open_to_close_no_sleep'}


class OrderedPrefetch:
    """单writer有序消费，最多max_pending个future，prepare由调用者维护线程局部Dataset。"""
    def __init__(self,ordered_uids,start,prepare,workers=4,max_pending=4,check=None):
        self.ordered=tuple(ordered_uids)
        if (type(start) is not int or not 0<=start<=len(self.ordered) or type(workers) is not int or workers<1
            or type(max_pending) is not int or max_pending<1 or not callable(prepare) or check is not None and not callable(check)
            or any(not isinstance(uid,str) or not uid for uid in self.ordered) or len(set(self.ordered))!=len(self.ordered)):
            raise ValueError('有序预取参数非法')
        self.prepare=prepare;self.check=check;self.max_pending=max_pending;self._next=start;self._submit=start;self._pending={};self._closed=False
        self._pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='cache-prepare');self._peak_pending=0;self._consumer=threading.RLock()
        self._fill()
    def _work(self,uid):
        if self.check is not None:self.check()
        result=self.prepare(uid)
        if self.check is not None:self.check()
        return result
    def _fill(self):
        while not self._closed and self._submit<len(self.ordered) and len(self._pending)<self.max_pending:
            i=self._submit;self._pending[i]=self._pool.submit(self._work,self.ordered[i]);self._submit+=1
            self._peak_pending=max(self._peak_pending,len(self._pending))
    @property
    def max_pending(self):return self._max_pending
    @max_pending.setter
    def max_pending(self,value):self._max_pending=value
    def __call__(self,uid):
        with self._consumer:
            if self._closed:raise RuntimeError('预取已关闭')
            try:
                if self.check is not None:self.check()
                if self._next<len(self.ordered) and uid==self.ordered[self._next]:
                    future=self._pending.pop(self._next);result=future.result();self._next+=1;self._fill();return result
                # 旧前缀三抽样不属于下一新行，直接同步prepare，不重排队列。
                for i,future in self._pending.items():
                    if self.ordered[i]==uid:return future.result()
                return self._work(uid)
            except BaseException:
                self.close();raise
    def close(self):
        with self._consumer:
            if self._closed:return
            self._closed=True
            for future in self._pending.values():future.cancel()
            self._pool.shutdown(wait=True,cancel_futures=True);self._pending.clear()
    def measurements(self):
        return {'pending':len(self._pending),'peak_pending':self._peak_pending,'max_pending':self.max_pending,'next_row':self._next,'closed':self._closed}
    def __enter__(self):return self
    def __exit__(self,*args):self.close();return False
