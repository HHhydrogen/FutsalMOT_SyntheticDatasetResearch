"""闭合实验归档的真实文件校验/安全释放夹具；不读真实数据、不加载GPU。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from futsalmot import run_archive as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.research = self.root / 'research'
        self.source = self.research / 'runs' / 'R_group'
        self.source.mkdir(parents=True)
        self.destination = self.root / 'G_training' / 'runs'
        self.destination.mkdir(parents=True)
        (self.source / 'status.json').write_text('{"state":"COMPLETE"}', encoding='utf-8')
        (self.source / 'run.json').write_bytes(b'{"original_bound_path":"D:/old/run"}\n')
        (self.source / 'checkpoints').mkdir()
        (self.source / 'checkpoints' / '000500.pt').write_bytes(bytes(range(256)) * 13)
        (self.source / 'empty').mkdir()
        self.original = {p.relative_to(self.source).as_posix(): p.read_bytes()
                         for p in self.source.rglob('*') if p.is_file()}
        self.addCleanup(patch.stopall)
        patch.object(archive, '_process_rows', return_value=[]).start()
        patch.object(archive, 'RESERVE_BYTES', 0).start()

    def invoke(self, **kwargs):
        return archive.archive_group(self.source, self.destination, research_root=self.research, **kwargs)

    @property
    def target(self):
        return self.destination / self.source.name

    def assert_source_original(self):
        for relative, content in self.original.items():
            self.assertEqual((self.source / relative).read_bytes(), content)

    def test_real_copy_hash_and_release_preserves_original_bindings(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['source_release_complete'])
        self.assertEqual(result['state'], 'ARCHIVED')
        self.assertEqual(result['destination'], str(self.target))
        self.assertEqual(result['manifest_path'], str(self.target / archive.MANIFEST))
        self.assertEqual(result['freed_bytes'], sum(map(len, self.original.values())))
        receipt = json.loads((self.target / archive.MANIFEST).read_bytes())
        self.assertEqual(receipt['original_terminal_state'], 'COMPLETE')
        self.assertEqual(receipt['total_bytes'], sum(map(len, self.original.values())))
        self.assertEqual(receipt['directories'], ['checkpoints', 'empty'])
        self.assertEqual({p.name for p in self.source.iterdir()}, {'RELOCATION.json', 'status.json'})
        for row in receipt['files']:
            content = (self.target / row['path']).read_bytes()
            self.assertEqual(content, self.original[row['path']])
            self.assertEqual(row['sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(row['size'], row['bytes'])
        self.assertEqual(json.loads((self.source / 'status.json').read_bytes())['state'], 'ARCHIVED')
        self.assertFalse((self.research / '.run_archive.lock').exists())

    def test_repeat_does_not_rewrite_archived_originals_or_manifest(self):
        self.invoke()
        before = {p.relative_to(self.target).as_posix(): p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        repeated = self.invoke()
        self.assertEqual(repeated['freed_bytes'], 0)
        after = {p.relative_to(self.target).as_posix(): p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(list(self.destination.glob('.*archive-*')), [])

    def test_exact_source_and_no_nested_or_lexical_escape(self):
        for source, dest in ((self.source.parent, self.destination),
                             (self.source / 'checkpoints', self.destination),
                             (self.source, self.source / 'nested'),
                             (self.source / '..' / 'R_group', self.destination)):
            with self.subTest(source=source, dest=dest), self.assertRaises(ValueError):
                archive.archive_group(source, dest, research_root=self.research)
        self.assert_source_original()

    def test_nonterminal_and_unsafe_allowed_override_rejected(self):
        (self.source / 'status.json').write_text('{"state":"RUNNING"}', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.invoke()
        with self.assertRaises(ValueError):
            self.invoke(expected_terminal_states={'RUNNING'})
        self.assertFalse(self.target.exists())

    def test_nonterminal_child_rejected(self):
        (self.source / 'checkpoints' / 'status.json').write_text('{"state":"RUNNING"}', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()

    def test_registered_pid_creation_identity_must_have_exited(self):
        (self.source / 'status.json').write_text('{"state":"FAIL_STOP","pid":123,"create_time":4.0}', encoding='utf-8')
        with patch.object(archive, '_matching_alive', return_value=True), self.assertRaises(RuntimeError):
            self.invoke()
        self.assertFalse(self.target.exists())

    def test_pid_reuse_with_different_creation_is_not_original_owner(self):
        import psutil
        value = {'state': 'FAIL_STOP', 'pid': os.getpid(), 'create_time': psutil.Process().create_time() - 100}
        (self.source / 'status.json').write_text(json.dumps(value), encoding='utf-8')
        self.assertFalse(archive._matching_alive(value['pid'], value['create_time']))
        self.assertEqual(self.invoke()['original_terminal_state'], 'FAIL_STOP')

    def test_gpu_lock_and_bound_command_prevent_archive(self):
        lock = self.research / '.research_gpu.lock'
        lock.write_text(json.dumps({'group': str(self.source), 'pid': 321}), encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.invoke()
        lock.unlink()
        with patch.object(archive, '_process_rows', return_value=[{'pid': 99999, 'name': 'python.exe', 'cmdline': ['python', str(self.source / 'launch.py')]}]), self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()

    def test_reparse_tree_entry_refused_without_following(self):
        real = archive._reparse
        def injected(path):
            return path.name == 'checkpoints' or real(path)
        with patch.object(archive, '_reparse', side_effect=injected), self.assertRaises(ValueError):
            self.invoke()
        self.assert_source_original()

    def test_broken_reparse_ancestor_is_rejected_without_exists_follow(self):
        ancestor = self.destination.parent
        real = archive._reparse
        def injected(path):
            return True if path == ancestor else real(path)
        with patch.object(archive, '_reparse', side_effect=injected), self.assertRaises(ValueError):
            self.invoke()
        self.assert_source_original()

    def test_bound_G_manifest_tampering_or_extra_file_blocks_release_resume(self):
        original_unlink = Path.unlink
        def stop_first(path, *args, **kwargs):
            if path == self.source / 'checkpoints' / '000500.pt':
                raise OSError('fixture stop before release')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', stop_first), self.assertRaises(OSError):
            self.invoke()
        (self.target / 'unexpected').write_bytes(b'extra')
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()
        self.assertEqual((self.target / 'unexpected').read_bytes(), b'extra')

    def test_source_added_during_copy_preserved_and_not_published(self):
        real_hash = archive._hash
        injected = False
        def changing(path):
            nonlocal injected
            value = real_hash(path)
            if not injected and '.archive-' in str(path):
                injected = True
                (self.source / 'new_writer_file').write_bytes(b'new')
            return value
        with patch.object(archive, '_hash', side_effect=changing), self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()
        self.assertEqual((self.source / 'new_writer_file').read_bytes(), b'new')
        self.assertFalse(self.target.exists())
        self.assertEqual(len(list(self.destination.glob('.*archive-*'))), 1)

    def test_copy_readback_corruption_preserves_D(self):
        real_hash = archive._hash
        def corrupted(path):
            return '0' * 64 if '.archive-' in str(path) else real_hash(path)
        with patch.object(archive, '_hash', side_effect=corrupted), self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()
        self.assertFalse(self.target.exists())

    def test_published_PASS_can_resume_interrupted_precise_release(self):
        original_unlink = Path.unlink
        interrupted = False
        def stop_one(path, *args, **kwargs):
            nonlocal interrupted
            if path == self.source / 'run.json' and not interrupted:
                interrupted = True
                raise OSError('fixture release interrupted')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', stop_one), self.assertRaises(OSError):
            self.invoke()
        self.assertTrue(self.target.exists())
        self.assertFalse((self.source / 'checkpoints' / '000500.pt').exists())
        self.assertEqual((self.source / 'run.json').read_bytes(), self.original['run.json'])
        before = (self.target / archive.MANIFEST).read_bytes()
        result = self.invoke()
        self.assertTrue(result['source_release_complete'])
        self.assertEqual((self.target / archive.MANIFEST).read_bytes(), before)

    def test_unknown_destination_conflict_never_overwritten(self):
        self.target.mkdir()
        existing = self.target / 'other'
        existing.write_bytes(b'untouched')
        with self.assertRaises(FileNotFoundError):
            self.invoke()
        self.assertEqual(existing.read_bytes(), b'untouched')
        self.assert_source_original()

    def test_insufficient_G_admission_keeps_D(self):
        with patch.object(archive, 'RESERVE_BYTES', 10**20), self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_source_original()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.destination.glob('.*archive-*')), [])

    def test_exact_direct_venv_archiver_shim_is_not_a_group_writer(self):
        current, shim = MagicMock(), MagicMock()
        current.pid, shim.pid = os.getpid(), 99997
        current.parent.return_value = shim
        current.create_time.return_value = 1000.1
        shim.create_time.return_value = 1000.0
        shim.exe.return_value = str(self.research / '.venv' / 'Scripts' / 'python.exe')
        current.cmdline.return_value = ['native', '-m', 'futsalmot.run_archive', str(self.source)]
        shim.cmdline.return_value = ['shim', '-m', 'futsalmot.run_archive', str(self.source)]
        rows = [{'pid': shim.pid, 'name': 'python.exe', 'cmdline': shim.cmdline()}]
        with patch.object(archive.psutil, 'Process', return_value=current), patch.object(archive, '_process_rows', return_value=rows):
            archive._quiescent(self.source, self.research, archive.TERMINAL_STATES)
        shim.cmdline.return_value = ['shim', str(self.source / 'launch.py')]
        rows[0]['cmdline'] = shim.cmdline()
        with patch.object(archive.psutil, 'Process', return_value=current), patch.object(archive, '_process_rows', return_value=rows), self.assertRaises(RuntimeError):
            archive._quiescent(self.source, self.research, archive.TERMINAL_STATES)

    def test_caller_cannot_archive_its_own_frozen_main(self):
        import __main__
        with patch.object(__main__, '__file__', str(self.source / 'launch.py')), self.assertRaises(RuntimeError):
            archive._quiescent(self.source, self.research, archive.TERMINAL_STATES)

    def test_stale_archive_writer_lock_is_not_blindly_removed(self):
        lock = self.research / '.run_archive.lock'
        lock.write_bytes(b'{"other":"writer"}')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(lock.read_bytes(), b'{"other":"writer"}')
        self.assert_source_original()


if __name__ == '__main__':
    unittest.main()
