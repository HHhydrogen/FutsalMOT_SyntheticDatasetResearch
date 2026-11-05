"""持久记录续接CPU夹具；只在临时目录写自造记录，无模型/真实数据。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from futsalmot.records import RunRecorder,temporary_diagnostic


def row(update=1):
    return dict(update=update,phase="final",domain="R_extra",data_domain="R",uids=["fixture"],
                loss=1.,lr=.001,gradient_norm=.5,R_exposures=update,S_exposures=0,
                extra_evidence={"components":[1.,2.,3.],"source":"fixture"})


def manifest():
    return {"source_bindings":{"data":"fixture_data_sha","code":"fixture_code_sha"},"shape":(960,960)}


class RecordResumeTests(unittest.TestCase):
    def test_durable_append_resume_exact_boundary_and_extra_fields_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run"
            with mock.patch("futsalmot.records.os.fsync",wraps=os.fsync) as sync:
                r=RunRecorder(path,manifest());r.step(row());r.event({"event":"fixture_no_training"})
                self.assertGreaterEqual(sync.call_count,3)
            original_manifest=(path/"run.json").read_bytes()
            expected=manifest();expected["shape"]=[960,960] # 规范JSON允许tuple写盘后的list表示。
            resumed=RunRecorder.resume(path,expected_manifest=expected,checkpoint_committed_updates=1)
            resumed.step(row(2));resumed.event({"event":"resumed_fixture"})
            steps=[json.loads(s) for s in (path/"steps.jsonl").read_text(encoding="utf8").splitlines()]
            self.assertEqual(steps,[row(),row(2)])
            self.assertEqual((path/"run.json").read_bytes(),original_manifest)
            self.assertEqual(resumed._last_step,2)
            self.assertEqual(len((path/"events.jsonl").read_text().splitlines()),2)
            with self.assertRaises(FileExistsError):RunRecorder(path,manifest())
            occupied=Path(t)/"occupied_file";occupied.write_bytes(b"existing_evidence")
            with self.assertRaises(FileExistsError):RunRecorder(occupied,manifest())
            self.assertEqual(occupied.read_bytes(),b"existing_evidence")

    def test_phase_zero_without_steps_allowed_missing_run_never_created(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";RunRecorder(path,manifest())
            zero=RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)
            self.assertFalse((path/"steps.jsonl").exists());self.assertEqual(zero._last_step,0)
            zero.step(row())
            absent=Path(t)/"absent"
            with self.assertRaises(FileNotFoundError):RunRecorder.resume(absent,expected_manifest=manifest(),checkpoint_committed_updates=0)
            self.assertFalse(absent.exists())

    def test_wrong_manifest_checkpoint_and_missing_records_block_without_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest());r.step(row())
            before={p.name:p.read_bytes() for p in path.iterdir()}
            wrong=manifest();wrong["source_bindings"]["data"]="different_sha"
            with self.assertRaisesRegex(ValueError,"绑定"):RunRecorder.resume(path,expected_manifest=wrong,checkpoint_committed_updates=1)
            for checkpoint in (0,2,True,1.5,-1):
                with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=checkpoint)
            self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})
            # 缺少正更新日志不能凭检查点计数重建。
            empty=Path(t)/"empty";RunRecorder(empty,manifest())
            with self.assertRaisesRegex(ValueError,"不齐"):RunRecorder.resume(empty,expected_manifest=manifest(),checkpoint_committed_updates=1)
            self.assertFalse((empty/"steps.jsonl").exists())

    def test_truncation_nonfinite_duplicate_keys_gaps_and_missing_fields_rejected(self):
        good=json.dumps(row()).encode()
        missing=row();missing.pop("uids")
        cases=[good,b'{"update":',b"\n",json.dumps(row(2)).encode()+b"\n",
               json.dumps(missing).encode()+b"\n",
               good.replace(b'"update": 1',b'"update": 1,"update":1')+b"\n",
               good.replace(b'"loss": 1.0',b'"loss": NaN')+b"\n",
               good.replace(b'"loss": 1.0',b'"loss": Infinity')+b"\n",
               good.replace(b'"loss": 1.0',b'"loss": 1e999')+b"\n",
               good.replace(b'"source": "fixture"',b'"source":"a","source":"b"')+b"\n"]
        for raw in cases:
            with tempfile.TemporaryDirectory() as t:
                path=Path(t)/"run";RunRecorder(path,manifest());steps=path/"steps.jsonl";steps.write_bytes(raw)
                with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=1)
                self.assertEqual(steps.read_bytes(),raw)
        for raw in (b'{"x":NaN}',b'{"x":1,"x":2}',b'{"x":{"y":1,"y":2}}'):
            with tempfile.TemporaryDirectory() as t:
                path=Path(t)/"run";RunRecorder(path,manifest());(path/"run.json").write_bytes(raw)
                with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)
                self.assertEqual((path/"run.json").read_bytes(),raw)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";RunRecorder(path,manifest());(path/"events.jsonl").write_bytes(b'{"event":"truncated"}')
            with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)

    def test_validation_before_write_does_not_advance_or_poison_healthy_recorder(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest())
            for update in (True,1.0,0,2):
                with self.assertRaises(ValueError):r.step({**row(),"update":update})
            with self.assertRaises(ValueError):r.step({**row(),"loss":float("nan")})
            with self.assertRaises(ValueError):r.event({"bad":{1,2}})
            self.assertEqual(r._last_step,0);self.assertFalse((path/"steps.jsonl").exists())
            self.assertFalse((path/"events.jsonl").exists())
            r.step(row());self.assertEqual(r._last_step,1)
            invalid=Path(t)/"bad_manifest"
            with self.assertRaises(ValueError):RunRecorder(invalid,{"nan":float("inf")})
            self.assertFalse(invalid.exists())

    def test_partial_write_failstop_keeps_evidence_and_cursor_unadvanced(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest())
            original=os.fdopen
            @contextmanager
            def broken(fd,*args,**kwargs):
                with original(fd,*args,**kwargs) as real:
                    class Partial:
                        def write(self,payload):
                            real.write(payload[:8]);real.flush();raise OSError("fixture partial write")
                        def __getattr__(self,key):return getattr(real,key)
                    yield Partial()
            with mock.patch("futsalmot.records.os.fdopen",broken):
                with self.assertRaisesRegex(OSError,"partial write"):r.step(row())
            self.assertEqual(r._last_step,0)
            evidence=(path/"steps.jsonl").read_bytes();self.assertEqual(len(evidence),8)
            for call in (lambda:r.step(row()),lambda:r.event({"event":"must_not_append"})):
                with self.assertRaisesRegex(RuntimeError,"FAIL_STOP"):call()
            self.assertEqual((path/"steps.jsonl").read_bytes(),evidence)
            with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)

    def test_fdopen_failure_closes_descriptor_but_preserves_failstop_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest())
            descriptors=[];original=os.open
            def opened(*args,**kwargs):
                fd=original(*args,**kwargs);descriptors.append(fd);return fd
            with mock.patch("futsalmot.records.os.open",opened),mock.patch("futsalmot.records.os.fdopen",side_effect=OSError("fixture fdopen failure")):
                with self.assertRaisesRegex(OSError,"fdopen failure"):r.event({"event":"fixture"})
            try:
                with self.assertRaises(OSError):os.fstat(descriptors[0])
            finally:
                try:os.close(descriptors[0])
                except OSError:pass
            self.assertEqual((path/"events.jsonl").read_bytes(),b"")
            with self.assertRaisesRegex(RuntimeError,"FAIL_STOP"):r.event({"event":"must_not_retry"})
            self.assertEqual(r._last_step,0)

    def test_reparse_nonregular_changed_manifest_and_temporary_cleanup(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest())
            (path/"steps.jsonl").mkdir()
            with self.assertRaises(ValueError):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";RunRecorder(path,manifest());target=path/"run.json"
            original=Path.lstat
            def reparse(p,*args,**kwargs):
                info=original(p,*args,**kwargs)
                if p == target:return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400)
                return info
            with mock.patch.object(Path,"lstat",reparse),mock.patch.object(Path,"open",side_effect=AssertionError("reparse不得打开")):
                with self.assertRaisesRegex(ValueError,"重解析"):RunRecorder.resume(path,expected_manifest=manifest(),checkpoint_committed_updates=0)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"run";r=RunRecorder(path,manifest());target=path/"run.json"
            changed=target.read_bytes()+b" ";target.write_bytes(changed)
            with self.assertRaisesRegex(ValueError,"manifest"):r.event({"event":"stop"})
            with self.assertRaisesRegex(RuntimeError,"FAIL_STOP"):r.step(row())
            self.assertEqual(target.read_bytes(),changed)
        with temporary_diagnostic() as diagnostic:
            temp=diagnostic.directory;diagnostic.event({"event":"fixture"})
        self.assertFalse(temp.exists())

if __name__=="__main__":unittest.main()
