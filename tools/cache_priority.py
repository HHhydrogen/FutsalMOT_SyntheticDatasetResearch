"""缓存进程的只读Real优先策略；不修改/暂停训练进程。"""
from __future__ import annotations
import json,statistics,time
from pathlib import Path
from tools.training_monitor import open_shared_log


def _json(path,limit=2*1024**2):
    with open_shared_log(path) as f: data=f.read(limit+1)
    if len(data)>limit:raise ValueError('control JSON over bound')
    return json.loads(data)


def recent_step_metrics(path,maximum=64,max_bytes=2*1024**2):
    with open_shared_log(path) as f:
        f.seek(0,2);size=f.tell();offset=max(0,size-max_bytes);f.seek(offset);raw=f.read(max_bytes)
    lines=raw.splitlines()
    if offset:lines=lines[1:]
    if not raw.endswith(b'\n'):lines=lines[:-1]
    rows=[]
    for line in lines[-maximum:]:
        value=json.loads(line)
        if type(value.get('update')) is int and type(value.get('wall_seconds')) in (int,float) and value['wall_seconds']>0:
            rows.append({'update':value['update'],'wall_seconds':value['wall_seconds']})
    return rows


class RealTrainingPriority:
    def __init__(self,output_group,*,baseline_seconds,slowdown_ratio=1.10,poll_seconds=10,cooldown_seconds=30,clock=time.monotonic):
        if not baseline_seconds>0 or not 1<slowdown_ratio<=1.25:raise ValueError('invalid measured baseline/threshold')
        self.group=Path(output_group);self.baseline=float(baseline_seconds);self.ratio=float(slowdown_ratio)
        self.poll=float(poll_seconds);self.cooldown=float(cooldown_seconds);self.clock=clock
        self.next_check=0.;self.hold_until=0.;self.paused=True;self.reason='initial_check';self.last={}

    def __call__(self):
        now=self.clock()
        if now<self.next_check:return self.paused or now<self.hold_until
        self.next_check=now+self.poll
        try:
            group=_json(self.group/'status.json');state=group.get('state')
            if state=='FORMAL_BLOCK_COMPLETE':
                self.paused=False;self.reason='real_block_complete';self.last={'state':state};return False
            if state!='RUNNING':
                self.paused=True;self.reason='real_state_not_running_or_complete';return True
            job=group.get('active_job')
            if not isinstance(job,str) or Path(job).name!=job:raise ValueError('unsafe active job')
            directory=self.group/job;manifest=_json(directory/'run.json');protocol=manifest['protocol']
            if protocol.get('strategy') not in ('R','R->R'):raise ValueError('unregistered GPU job')
            status=_json(directory/'status.json');rows=recent_step_metrics(directory/'steps.jsonl')
            update=max(status.get('update',0),rows[-1]['update'] if rows else 0)
            pending=[x['tag'] for x in protocol.get('evaluation_schedule',[]) if x.get('global_update')==update
                     and not (directory/'evaluation'/x['tag']/'summary.json').is_file()]
            self.last={'job':job,'update':update,'baseline_seconds':self.baseline,'sample_count':len(rows),'pending_validation_tags':pending}
            if pending or len(rows)<8:
                self.paused=True;self.reason='real_validation_or_startup';return True
            median=statistics.median(x['wall_seconds'] for x in rows);self.last['median_seconds']=median
            if median>self.baseline*self.ratio:
                self.hold_until=now+self.cooldown;self.paused=True;self.reason='real_slowdown_backoff';return True
            self.paused=now<self.hold_until;self.reason='cooldown' if self.paused else 'real_training_healthy'
            return self.paused
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.paused=True;self.reason='control_unavailable_pause';self.last={'error':repr(exc)};return True

    def snapshot(self):return {'paused':self.paused,'reason':self.reason,**self.last}
