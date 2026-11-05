"""只读训练监控的有界CPU/HTTP夹具；不接触真实run、GPU、媒体或G盘。"""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SPEC = importlib.util.spec_from_file_location('futsalmot_training_monitor_test_module', Path(__file__).parents[1] / 'tools/training_monitor.py')
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def append_row(path, value, newline=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + ('\n' if newline else ''))


def fixture(root, *, update=2000, state='RUNNING'):
    research, archive = root / 'research', root / 'archive'
    group = research / 'runs/group_a'
    run = group / 'R50_seed17'
    cfg = {'model': 'yolo11s', 'strategy': 'R', 'seed': 17, 'epochs': 50, 'updates': 6563, 'N_R': 2100,
           'N_val': 750, 'input_size': 960, 'microbatch_size': 8, 'effective_batch': 16, 'numeric_mode': 'fp32',
           'num_workers': 2, 'checkpoint_every': 50, 'evaluation_every': 1000}
    write_json(group / 'status.json', {'state': 'RUNNING'})
    write_json(run / 'run.json', {'protocol': cfg, 'source_bindings': {'not_loaded': 'metadata only'}})
    write_json(run / 'status.json', {'state': state, 'update': update})
    append_row(run / 'steps.jsonl', {'update': update, 'loss': 2.2, 'loss_components': {'box': .9, 'cls': .4, 'dfl': .9},
                                   'gradient_norm_before_clip': 6.5, 'lr': .001, 'wall_seconds': .5,
                                   'R_exposures': update * 16, 'S_exposures': 0, 'sample_lineage': ['discard me']})
    append_row(run / 'checkpoints.jsonl', {'kind': 'checkpoint', 'update': 0})
    write_json(run / 'evaluation/step000000_PT/summary.json', {'update': 0, 'tag': 'step000000_PT',
               'images': 750, 'wall_seconds': 330., 'main_metrics': {'AP50_95': .38, 'AP50': .79, 'AP75': .31}})
    write_json(run / 'evaluation/step001000/summary.json', {'update': 1000, 'tag': 'step001000',
               'images': 750, 'wall_seconds': 310., 'main_metrics': {'AP50_95': .52, 'AP50': .94}})
    append_row(group / 'controller.stdout.log', {'event': 'update', 'update': update})
    append_row(group / 'controller.stdout.log', {'event': 'eval_progress', 'tag': f'step{update:06d}', 'images': 400, 'total': 750})
    (group / 'controller.stderr.log').write_text('', encoding='utf-8')
    archive.mkdir()
    return research, archive, group, run


def relocation_stub(group, target, *, complete=True):
    receipt = {'schema': 'futsalmot_run_archive_v1', 'status': 'PASS', 'source': str(group), 'target': str(target), 'files': []}
    write_json(target / 'ARCHIVE_MANIFEST.json', receipt)
    digest = hashlib.sha256((target / 'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()
    write_json(group / 'RELOCATION.json', {'schema': 'futsalmot_run_relocation_v1',
               'status': 'PASS' if complete else 'COPY_VERIFIED_RELEASE_PENDING',
               'source_release_complete': complete, 'source': str(group), 'target': str(target), 'manifest_sha256': digest})



class NamingTests(unittest.TestCase):
    def test_formal_label_uses_explicit_source_and_preserves_strategy(self):
        manifest = {'schema_version': 'formal_bbox_five_strategy_v1', 'protocol': {'task': 'detect', 'model': 'yolo11s', 'strategy': 'R->R', 'seed': 17},
                    'source_bindings': {'real_panel': {'index': {'path': r'G:\FutsalMOT_Dataset\real\soccernet\dataset.json'}}}}
        self.assertEqual(monitor.display_label(manifest, 'R_to_R_seed17'), '正式｜BBox｜YOLO11s｜SoccerNet｜R→R｜seed17')
        manifest['source_bindings']['real_panel']['index']['path'] = 'G:/FutsalMOT_Dataset/real/sportsmot/dataset.json'
        manifest['protocol'].update(task='segment', model='yolo11x-seg', strategy='R+S')
        self.assertEqual(monitor.display_label(manifest, 'R_plus_S_seed17'), '正式｜Seg｜YOLO11x-seg｜SportsMOT｜R+S｜seed17')

    def test_calibration_and_registered_legacy_name_fallback(self):
        for epochs in (50, 100):
            m = {'protocol': {'schema_version': 'bbox_R_calibration_run_v1', 'epochs': epochs, 'task': 'detect', 'model': 'yolo11s', 'strategy': 'R', 'seed': 17}}
            self.assertEqual(monitor.display_label(m, f'R{epochs}_seed17'), f'校准R{epochs}｜BBox｜YOLO11s｜未登记｜R｜seed17')
        self.assertEqual(monitor.display_label({}, 'R50_seed17', 'bbox_s_seed17_calibration_20261101_a02'),
                         '校准R50｜BBox｜YOLO11s｜未登记｜R｜seed17')
        self.assertEqual(monitor.display_label({}, 'S_to_R_seed29', 'bbox_s_soccernet_seed29_core_20261102_a01'),
                         '正式｜BBox｜YOLO11s｜未登记｜S→R｜seed29')

    def test_unknown_source_is_never_inferred_from_name_or_substring(self):
        m = {'protocol': {'schema_version': 'formal_bbox_five_strategy_v1', 'task': 'pose', 'model': 'yolo11n-pose', 'strategy': 'S', 'seed': 43},
             'source_bindings': {'real_panel': {'index': {'path': 'G:/anything/soccernet_other/dataset.json'}}}}
        self.assertEqual(monitor.display_label(m, 'S_seed43', 'soccernet'), '正式｜Pose｜YOLO11n-pose｜未登记｜S｜seed43')
        self.assertEqual(monitor.display_label({}, 'unknown', 'soccernet'), '未登记｜未登记｜未登记｜未登记｜未登记｜未登记')
        m['source_bindings']['real_panel'] = {'source': 'soccernet_tracking'}
        self.assertIn('｜SoccerNet｜', monitor.display_label(m, 'S_seed43'))

    def test_canonical_group_uses_beijing_date_and_safe_registered_parts(self):
        from datetime import datetime, timezone
        at = datetime(2026, 11, 1, 18, tzinfo=timezone.utc)
        self.assertEqual(monitor.canonical_group_name(purpose='main', task='detect', model='YOLO11s', panel='SoccerNet', strategy='R→R', seed=17, at=at),
                         'main_bbox_yolo11s_soccernet_R2R_seed17_20261102_a01')
        self.assertEqual(monitor.canonical_group_name(purpose='cal100', task='pose', model='yolo11x-pose', panel='sportsmot', strategy='S->R', seed=43, at=at, attempt=2),
                         'cal100_pose_yolo11x-pose_sportsmot_S2R_seed43_20261102_a02')
        kwargs = dict(purpose='main', task='detect', model='yolo11s', panel='soccernet', strategy='R', seed=17, at=at)
        for changed in ({'model':'../model'}, {'panel':'soccer net'}, {'panel':r'G:\source'}, {'seed':True}, {'attempt':True}, {'attempt':100}, {'at':datetime(2026,11,1)}, {'purpose':'unknown'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                monitor.canonical_group_name(**{**kwargs, **changed})

    def test_discovery_new_label_keeps_original_stable_id_and_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            original = (run/'run.json').read_bytes()
            instance = monitor.Monitor(research, archive, refresh_seconds=0)
            entries, errors = instance.inventory()
            self.assertFalse(errors)
            self.assertEqual(entries[0]['id'], hashlib.sha256((group.name+'/'+run.name).encode()).hexdigest()[:20])
            self.assertEqual(entries[0]['label'], '校准R50｜未登记｜YOLO11s｜未登记｜R｜seed17')
            self.assertEqual((run/'run.json').read_bytes(), original)

class IncrementalTests(unittest.TestCase):
    def test_partial_line_append_and_only_scalar_memory(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            append_row(path, {'update': 1, 'loss': 3., 'geometry': ['never retain']})
            with path.open('ab') as stream:
                stream.write(b'{"update":2,"loss":')
            reader = monitor.IncrementalLog()
            self.assertEqual([x['update'] for x in reader.update(path)], [1])
            with path.open('ab') as stream:
                stream.write(b'2.2}\n')
            points = reader.update(path)
            self.assertEqual([x['update'] for x in points], [1, 2])
            self.assertEqual(reader.bytes_last_read, 5)
            self.assertNotIn('geometry', points[0])
            reader.update(path)
            self.assertEqual(reader.bytes_last_read, 0)
            self.assertEqual(reader.records, 2)

    def test_initial_tail_and_points_are_bounded(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            for index in range(100):
                append_row(path, {'update': index, 'loss': index / 10, 'ignored': 'x' * 100})
            reader = monitor.IncrementalLog(max_scan=1024, max_points=5)
            points = reader.update(path)
            self.assertTrue(reader.history_truncated)
            self.assertLessEqual(reader.bytes_last_read, 1024)
            self.assertEqual(len(points), 5)
            self.assertEqual(points[-1]['update'], 99)
            append_row(path, {'update': 100, 'loss': 1})
            self.assertEqual(reader.update(path)[-1]['update'], 100)
            self.assertLess(reader.bytes_last_read, 100)

    def test_truncate_and_file_identity_rotation_reset(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            append_row(path, {'update': 1, 'loss': 2})
            append_row(path, {'update': 2, 'loss': 2})
            reader = monitor.IncrementalLog(); reader.update(path)
            path.write_text('{"update":7}\n', encoding='utf-8')
            self.assertEqual([x['update'] for x in reader.update(path)], [7])
            other = Path(t) / 'new.jsonl'; append_row(other, {'update': 9})
            other.replace(path)
            self.assertEqual([x['update'] for x in reader.update(path)], [9])
            self.assertEqual(reader.records, 1)

    def test_malformed_nan_and_nonstep_records_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            path.write_text('bad json\n{"update":3,"loss":NaN}\n{"other":1}\n', encoding='utf-8')
            reader = monitor.IncrementalLog(); points = reader.update(path)
            self.assertEqual(reader.records, 1)
            self.assertIsNone(points[-1]['loss'])
            self.assertTrue(reader.errors)

    def test_oversized_line_is_skipped_and_handle_not_held(self):
        with tempfile.TemporaryDirectory() as t, patch.object(monitor, 'MAX_LINE_BYTES', 64):
            path = Path(t) / 'steps.jsonl'
            path.write_text('x' * 80 + '\n{"update":2}\n', encoding='utf-8')
            reader = monitor.IncrementalLog(); self.assertEqual(reader.update(path)[-1]['update'], 2)
            self.assertTrue(reader.errors)
            path.rename(Path(t) / 'moved.jsonl')
            reader.update(path, allow_missing_initial=True)
            self.assertIn('steps.jsonl', reader.errors[-1])
            missing = monitor.IncrementalLog()
            self.assertEqual(missing.update(path, allow_missing_initial=True), [])
            self.assertFalse(missing.errors)
            denied = monitor.IncrementalLog()
            with patch.object(monitor, 'open_shared_log', side_effect=PermissionError(13, 'fixture read denied')):
                denied.update(Path(t) / 'moved.jsonl', allow_missing_initial=True)
            self.assertIn('moved.jsonl', denied.errors[-1]); self.assertIn('13', denied.errors[-1])


class FullHistoryTests(unittest.TestCase):
    @staticmethod
    def drain(reader, limit=1000):
        for _ in range(limit):
            reader.advance()
            if reader.snapshot()['state'] in ('ready', 'limited', 'error'):
                return reader.snapshot()
        raise AssertionError('finite fixture did not finish')

    def test_full_history_starts_at_first_real_update_and_retains_scalar_fields(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            for step in range(1, 31):
                append_row(path, {'update': step, 'loss': 3 / step, 'lr': .001, 'gradient_norm_before_clip': 5.2,
                                  'wall_seconds': .48, 'sample_lineage': ['x' * 300]})
            reader = monitor.FullHistoryReader(lambda: path, chunk_bytes=256)
            value = self.drain(reader)
            self.assertEqual(value['state'], 'ready')
            self.assertEqual([x['update'] for x in value['points']], list(range(1, 31)))
            self.assertEqual(value['parsed_records'], 30)
            self.assertEqual(value['bytes_read'], path.stat().st_size)
            self.assertEqual(value['points'][0]['gradient_norm'], 5.2)
            self.assertEqual(value['points'][0]['learning_rate'], .001)
            self.assertEqual(value['points'][0]['wall_seconds'], .48)
            self.assertNotIn('sample_lineage', value['points'][0])
            self.assertEqual(value['history_generation'], 1)

    def test_full_history_append_and_halfline_do_not_rescan(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'; append_row(path, {'update': 1})
            reader = monitor.FullHistoryReader(lambda: path, chunk_bytes=128)
            first = self.drain(reader)
            with path.open('ab') as stream:stream.write(b'{"update":2')
            partial = self.drain(reader)
            self.assertEqual(partial['parsed_records'], 1)
            self.assertGreater(partial['pending_bytes'], 0)
            before = partial['bytes_read']
            with path.open('ab') as stream:stream.write(b'}\n')
            reader.advance(); value = reader.snapshot()
            self.assertEqual([x['update'] for x in value['points']], [1, 2])
            self.assertEqual(reader.bytes_last_read, 2)
            self.assertEqual(value['bytes_read'], before + 2)
            self.assertEqual(value['history_generation'], first['history_generation'])
            reader.advance(); self.assertEqual(reader.bytes_last_read, 0)

    def test_full_history_rotation_truncate_and_migration_reset_generation(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); path = root / 'D.jsonl'; append_row(path, {'update': 1}); append_row(path, {'update': 2})
            location = [path]
            reader = monitor.FullHistoryReader(lambda: location[0])
            self.drain(reader); one = reader.snapshot()['history_generation']
            other = root / 'other.jsonl'; append_row(other, {'update': 9}); other.replace(path)
            self.assertEqual(self.drain(reader)['first_update'], 9)
            self.assertGreater(reader.snapshot()['history_generation'], one)
            previous = reader.snapshot()['history_generation']
            target = root / 'G.jsonl'; append_row(target, {'update': 10}); location[0] = target
            self.assertEqual(self.drain(reader)['first_update'], 10)
            self.assertGreater(reader.snapshot()['history_generation'], previous)
            target.write_text('{"update":4}\n', encoding='utf-8')
            self.assertEqual(self.drain(reader)['last_update'], 4)

    def test_full_history_cap_and_parse_gaps_are_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'
            for step in range(1, 8):append_row(path, {'update': step})
            with path.open('ab') as stream:stream.write(b'bad row\n')
            reader = monitor.FullHistoryReader(lambda: path, max_points=3)
            value = self.drain(reader)
            self.assertEqual(value['state'], 'limited')
            self.assertEqual(value['parsed_records'], 7)
            self.assertEqual([x['update'] for x in value['points']], [5, 6, 7])
            self.assertIn('3', value['limited_reason']); self.assertIn('缺口', value['limited_reason'])

    def test_full_history_budget_and_missing_first_step_are_bounded(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'not_yet.jsonl'
            reader = monitor.FullHistoryReader(lambda: path, budget_seconds=1)
            reader.advance(); self.assertEqual(reader.snapshot()['state'], 'loading')
            append_row(path, {'update': 1})
            self.assertEqual(self.drain(reader)['last_update'], 1)
            for step in range(2, 20):append_row(path, {'update': step})
            ticks = iter(range(100))
            with patch.object(monitor.time, 'monotonic', side_effect=lambda: next(ticks)):
                limited = monitor.FullHistoryReader(lambda: path, chunk_bytes=4, budget_seconds=.5)
                limited.advance()
            self.assertEqual(limited.snapshot()['state'], 'limited')
            self.assertTrue(limited.fatal)

    def test_shared_reader_allows_delete_without_holding_training_file(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'; path.write_bytes(b'123456')
            with monitor.open_shared_log(path) as stream:
                path.unlink()
                self.assertEqual(stream.read(), b'123456')
            self.assertFalse(path.exists())

    def test_two_reader_pool_is_bounded_and_close_stops_workers(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'steps.jsonl'; append_row(path, {'update': 1})
            pool = monitor.FullHistoryPool(lambda identifier: path, maximum=2)
            try:
                pool.request('a'); pool.request('b'); pool.request('c')
                self.assertLessEqual(len(pool.readers), 2)
            finally:pool.close()
            self.assertTrue(all(x.thread is None or not x.thread.is_alive() for x in pool.readers.values()))

    def test_opt_in_full_cache_isolated_and_unknown_ID_never_starts_reader(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            app = monitor.Monitor(research, archive, refresh_seconds=0)
            try:
                with patch.object(app, 'system', return_value={}):
                    tail = app.snapshot(); self.assertNotIn('history', tail); self.assertFalse(app.histories.readers)
                    missing = app.snapshot('0' * 20, history='full')
                    self.assertEqual(missing['history']['state'], 'error'); self.assertFalse(app.histories.readers)
                    self.assertIsNone(missing['history']['progress_fraction']); self.assertIn('history_generation', missing['history'])
                    full = app.snapshot(history='full')
                    self.assertIn('history', full); self.assertEqual(len(app.histories.readers), 1)
                    self.assertNotIn('history', app.snapshot())
                    self.assertIn(('', 'tail'), app.cached); self.assertIn(('', 'full'), app.cached)
            finally:app.close()

    def test_comparison_signature_requires_same_task_metric_and_panel(self):
        first = {'protocol': {'task': 'detect', 'main_metric': 'COCO AP', 'model': 's'},
                 'source_bindings': {'real_panel': {'train': 'abc', 'val': 'def'}}}
        other = json.loads(json.dumps(first)); other['protocol']['model'] = 'n'
        self.assertEqual(monitor.comparison_signature(first), monitor.comparison_signature(other))
        other['source_bindings']['real_panel']['val'] = 'different'
        self.assertNotEqual(monitor.comparison_signature(first), monitor.comparison_signature(other))
        self.assertIsNone(monitor.comparison_signature({'protocol': {}}))


class SnapshotTests(unittest.TestCase):
    def test_live_validation_phase_metrics_and_eta(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            app = monitor.Monitor(research, archive, refresh_seconds=0)
            with patch.object(app, 'system', return_value={}):
                result = app.snapshot()
            value = result['run']
            self.assertEqual(value['phase']['kind'], 'validation')
            self.assertEqual(value['phase']['images'], 400)
            self.assertEqual(value['progress']['update'], 2000)
            self.assertEqual(value['latest']['loss_box'], .9)
            self.assertEqual(value['latest']['learning_rates'], [.001])
            self.assertEqual(value['latest']['gradient_norm'], 6.5)
            self.assertEqual(value['curves']['validation'][-1]['AP50_95'], .52)
            self.assertEqual(value['checkpoint_count'], 1)
            self.assertGreater(value['eta_seconds'], (6563 - 2000) * .5)
            self.assertEqual(value['progress']['R_exposures'], 32000)
            write_json(run / 'status.json', {'state': 'RUNNING', 'update': 0})
            (run / 'steps.jsonl').unlink(); (group / 'controller.stdout.log').write_text('', encoding='utf-8')
            initial = monitor.Monitor(research, archive, refresh_seconds=0)
            with patch.object(initial, 'system', return_value={}):
                waiting = initial.snapshot()['run']
                self.assertEqual(waiting['phase']['kind'], 'initializing')
                self.assertIn('尚无参数更新', waiting['phase']['label'])
                self.assertIsNone(waiting['phase']['images']); self.assertFalse(waiting['errors'])
                append_row(group / 'controller.stdout.log', {'event': 'eval_progress', 'tag': 'step000000_PT', 'images': 200, 'total': 750})
                evaluating = initial.snapshot()['run']
            self.assertEqual(evaluating['phase']['kind'], 'validation')
            self.assertEqual(evaluating['phase']['images'], 200); self.assertFalse(evaluating['errors'])

    def test_cached_poll_and_incremental_growth(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            app = monitor.Monitor(research, archive)
            with patch.object(app, 'system', return_value={}):
                one = app.snapshot()
                append_row(run / 'steps.jsonl', {'update': 2001, 'loss': 2.1})
                self.assertIs(app.snapshot(), one)
                app.cached.clear()
                two = app.snapshot()
            self.assertEqual(two['run']['progress']['update'], 2001)
            self.assertEqual(two['run']['phase']['kind'], 'training')

    def test_relocation_keeps_opaque_run_id_and_follows_G_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t), state='COMPLETE')
            app = monitor.Monitor(research, archive, refresh_seconds=0)
            with patch.object(app, 'system', return_value={}):
                before = app.snapshot()
                target = archive / 'runs' / group.name; target.parent.mkdir()
                shutil.move(str(group), str(target)); group.mkdir()
                relocation_stub(group, target)
                write_json(group / 'status.json', {'state': 'ARCHIVED'})
                after = app.snapshot(before['selected_run_id'])
            self.assertEqual(after['selected_run_id'], before['selected_run_id'])
            self.assertTrue(after['run']['archived'])
            self.assertTrue(after['run']['path'].startswith(str(archive)))
            self.assertEqual(after['run']['eta_seconds'], 0)

    def test_verified_G_release_pending_does_not_open_D_run_files(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t), state='COMPLETE')
            target = archive / 'runs' / group.name; target.parent.mkdir()
            shutil.copytree(group, target); relocation_stub(group, target, complete=False)
            app = monitor.Monitor(research, archive)
            real_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.is_relative_to(run):
                    raise AssertionError('Release window must not open D run files')
                return real_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded), patch.object(app, 'system', return_value={}):
                result = app.snapshot()
            self.assertTrue(result['run']['archive_pending'])
            self.assertFalse(result['run']['source_release_complete'])
            self.assertTrue(result['run']['path'].startswith(str(archive)))
            self.assertEqual(result['run']['state'], 'COMPLETE')

    def test_unknown_or_changed_G_manifest_cannot_redirect_reader(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            target = archive / 'runs' / group.name; target.parent.mkdir()
            shutil.copytree(group, target); relocation_stub(group, target, complete=False)
            (target / 'ARCHIVE_MANIFEST.json').write_text('{}', encoding='utf-8')
            app = monitor.Monitor(research, archive)
            entries, errors = app.inventory()
            self.assertEqual(entries[0]['_path'], run)
            self.assertFalse(entries[0]['archive_pending']); self.assertTrue(errors)

    def test_relocation_outside_allowed_root_is_not_followed(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            outside = Path(t) / 'outside'; outside.mkdir()
            write_json(group / 'RELOCATION.json', {'status': 'PASS', 'source_release_complete': True, 'target': str(outside)})
            app = monitor.Monitor(research, archive)
            entries, errors = app.inventory()
            self.assertEqual(entries[0]['_path'], run)
            self.assertTrue(errors)

    def test_no_weights_or_raw_prediction_access_and_unknown_run_clear(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            (run / 'checkpoints').mkdir(); (run / 'checkpoints/model.pt').write_bytes(b'forbidden')
            raw = run / 'evaluation/step000000_PT/raw'; raw.mkdir(); (raw / '1.npz').write_bytes(b'forbidden')
            real_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.suffix in ('.pt', '.npz'):
                    raise AssertionError('Monitor opened a model or raw prediction')
                return real_open(path, *args, **kwargs)
            app = monitor.Monitor(research, archive, refresh_seconds=0)
            with patch.object(Path, 'open', guarded), patch.object(app, 'system', return_value={}):
                self.assertIsNotNone(app.snapshot()['run'])
                missing = app.snapshot('0' * 20)
            self.assertIsNone(missing['run']); self.assertTrue(missing['errors'])

    def test_missing_or_invalid_metadata_is_not_a_fake_zero_result(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            (run / 'run.json').write_text('{', encoding='utf-8')
            app = monitor.Monitor(research, archive)
            with patch.object(app, 'system', return_value={}):
                result = app.snapshot()
            self.assertIsNone(result['run']); self.assertTrue(result['errors'])

    def test_exited_pid_is_not_labelled_healthy_training(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, group, run = fixture(Path(t))
            app = monitor.Monitor(research, archive)
            with patch.object(app, 'system', return_value={}), patch.object(monitor, '_process_identity', return_value={'pid': 42, 'alive': False, 'identity_verified': False}):
                result = app.snapshot()
            self.assertEqual(result['run']['state'], 'RUNNING')
            self.assertEqual(result['run']['phase']['kind'], 'unknown')
            self.assertTrue(result['run']['errors'])

    def test_gpu_query_cached_and_failure_exposes_null(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root / 'archive').mkdir()
            app = monitor.Monitor(root, root / 'archive')
            with patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout='NVIDIA Test, 70, 8192, 16384, 8192\n')) as call:
                result = app.system(); again = app.system()
                self.assertIs(result, again); call.assert_called_once()
                self.assertEqual(result['gpu']['memory_total_bytes'], 16384 * 1024**2)
            app.system_cached = None
            with patch.object(monitor.subprocess, 'run', side_effect=FileNotFoundError('not available')):
                result = app.system()
                self.assertIsNone(result['gpu']); self.assertTrue(result['gpu_error'])


def direct_g_fixture(root):
    research, archive, control, run = fixture(root)
    output = archive / 'runs' / control.name
    output.mkdir(parents=True)
    shutil.move(str(run), str(output / run.name))
    for name in ('controller.stdout.log', 'controller.stderr.log'):
        shutil.move(str(control / name), str(output / name))
    # 控制状态应来自D，即使G根存在另一个状态文件，也不能冒用。
    write_json(output / 'status.json', {'state': 'FAIL_STOP'})
    location = {'schema': 'futsalmot_direct_output_v1', 'version': 1, 'storage_mode': 'direct_g',
                'control_group': str(control), 'output_group': str(output), 'created_utc': '2026-11-01T00:00:00Z'}
    raw = (json.dumps(location, sort_keys=True, separators=(',', ':')) + '\n').encode()
    (control / 'RUN_LOCATION.json').write_bytes(raw)
    (output / 'RUN_LOCATION.json').write_bytes(raw)
    return research, archive, control, output, output / run.name


class DirectGTests(unittest.TestCase):
    def test_direct_G_is_live_output_not_archived_and_D_controls_group_state(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, control, output, run = direct_g_fixture(Path(t))
            app = monitor.Monitor(research, archive)
            with patch.object(app, 'system', return_value={}):
                value = app.snapshot()['run']
            self.assertEqual(value['storage_mode'], 'direct_g')
            self.assertIn('无结束后整组搬运', value['eta_note'])
            self.assertNotIn('不含结束后的G盘迁移', value['eta_note'])
            self.assertFalse(value['archived']); self.assertFalse(value['archive_pending'])
            self.assertEqual(value['group_state'], 'RUNNING')
            self.assertEqual(value['state'], 'RUNNING')
            self.assertEqual(value['control_group_path'], str(control))
            self.assertEqual(value['output_group_path'], str(output))
            self.assertEqual(value['phase']['images'], 400)
            self.assertEqual(value['progress']['update'], 2000)
            self.assertEqual(app._history_path(value['id']), run / 'steps.jsonl')
            self.assertFalse((output / 'ARCHIVE_MANIFEST.json').exists())

    def test_both_ends_discover_once_and_keep_original_opaque_ID(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, control, output, run = direct_g_fixture(Path(t))
            app = monitor.Monitor(research, archive)
            rows, errors = app.inventory()
            expected = hashlib.sha256((control.name + '/' + run.name).encode()).hexdigest()[:20]
            self.assertEqual(len(rows), 1); self.assertFalse(errors)
            self.assertEqual(rows[0]['id'], expected)
            self.assertEqual(rows[0]['_path'], run)

    def test_G_marker_discovers_valid_output_when_D_listing_is_temporarily_empty(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, control, output, run = direct_g_fixture(Path(t))
            original = Path.iterdir
            def listing(path):
                return iter([]) if path == research / 'runs' else original(path)
            app = monitor.Monitor(research, archive)
            with patch.object(Path, 'iterdir', listing):
                rows, errors = app.inventory()
            self.assertEqual(len(rows), 1); self.assertFalse(errors)
            self.assertEqual(rows[0]['_control_group'], control)
            self.assertEqual(rows[0]['storage_mode'], 'direct_g')

    def test_single_endpoint_marker_or_equivalent_JSON_with_different_bytes_rejected(self):
        for scenario in ('missing', 'different_bytes'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as t:
                research, archive, control, output, run = direct_g_fixture(Path(t))
                marker = output / 'RUN_LOCATION.json'
                if scenario == 'missing':marker.unlink()
                else:marker.write_bytes(marker.read_bytes() + b' ')
                rows, errors = monitor.Monitor(research, archive).inventory()
                self.assertEqual(rows, []); self.assertTrue(errors)

    def test_outside_or_wrong_direct_control_output_binding_rejected(self):
        for field in ('control_group', 'output_group'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as t:
                research, archive, control, output, run = direct_g_fixture(Path(t))
                location = json.loads((control / 'RUN_LOCATION.json').read_bytes())
                location[field] = str(Path(t) / 'outside' / control.name)
                raw = json.dumps(location).encode()
                (control / 'RUN_LOCATION.json').write_bytes(raw); (output / 'RUN_LOCATION.json').write_bytes(raw)
                rows, errors = monitor.Monitor(research, archive).inventory()
                self.assertEqual(rows, []); self.assertTrue(errors)

    def test_reparse_marker_is_not_followed(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, control, output, run = direct_g_fixture(Path(t))
            real = Path.is_junction
            def redirected(path):return path == output / 'RUN_LOCATION.json' or real(path)
            with patch.object(Path, 'is_junction', redirected):
                rows, errors = monitor.Monitor(research, archive).inventory()
            self.assertEqual(rows, []); self.assertTrue(errors)

    def test_D_gpu_lock_auto_follows_G_run(self):
        with tempfile.TemporaryDirectory() as t:
            research, archive, control, output, run = direct_g_fixture(Path(t))
            other = research / 'runs/group_b/R50_seed17'
            write_json(other / 'run.json', json.loads((run / 'run.json').read_bytes()))
            write_json(other / 'status.json', {'state': 'RUNNING', 'update': 0})
            write_json(research / '.research_gpu.lock', {'group': str(control), 'pid': monitor.os.getpid(),
                       'create_time': monitor.psutil.Process().create_time()})
            app = monitor.Monitor(research, archive)
            with patch.object(app, 'system', return_value={}):value = app.snapshot()['run']
            self.assertEqual(value['group_id'], control.name)
            self.assertEqual(value['path'], str(run))


@contextmanager
def live_server(root):
    research, archive, group, run = fixture(root)
    page = root / 'page.html'; page.write_text('<!doctype html><title>fixture</title>', encoding='utf-8')
    page.with_name('training_monitor_charts.js').write_text('window.monitorFixtureChart = true;', encoding='utf-8')
    server = monitor.create_server(research, archive, port=0, html_path=page, duration_hours=.01)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    server.monitor.system = lambda: {}
    try:
        with monitor.registration(server):
            thread.start()
            yield server, research
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


class HTTPTests(unittest.TestCase):
    def test_real_local_health_status_and_page(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                url = server.health['url']
                with urlopen(url + '/api/health', timeout=2) as response:
                    health = json.load(response)
                self.assertEqual(health['service'], monitor.SERVICE); self.assertTrue(health['read_only'])
                with urlopen(url + '/api/status', timeout=2) as response:
                    self.assertEqual(json.load(response)['run']['progress']['update'], 2000)
                with urlopen(url, timeout=2) as response:
                    self.assertIn(b'<title>fixture</title>', response.read())
            self.assertFalse((research / '.training_monitor.json').exists())

    def test_controlled_chart_JS_and_full_history_HTTP(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                url = server.health['url']
                with urlopen(url + '/training_monitor_charts.js', timeout=2) as response:
                    self.assertIn('javascript', response.headers['Content-Type'])
                    self.assertIn(b'monitorFixtureChart', response.read())
                with urlopen(url + '/api/status?history=full', timeout=2) as response:
                    value = json.load(response)
                self.assertIn(value['history']['state'], ('loading', 'ready'))
                self.assertEqual(server.health['version'], 9)

    def test_unknown_duplicate_blank_or_malformed_params_are_bad_requests(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                url = server.health['url']
                for query in ('unknown=1', 'history=tail', 'history=full&history=full', 'run=', 'history=', 'x',
                              'run=' + '0' * 20 + '&run=' + '1' * 20):
                    with self.subTest(query=query), self.assertRaises(HTTPError) as error:
                        urlopen(url + '/api/status?' + query, timeout=2)
                    self.assertEqual(error.exception.code, 400)
                self.assertFalse(server.monitor.histories.readers)

    def test_existing_service_is_reused_and_open_does_not_bind_again(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                self.assertEqual(monitor.existing_monitor(research), server.health['url'])
                argv = ['training_monitor.py', '--root', str(research), '--open']
                with patch.object(sys, 'argv', argv), patch.object(monitor.webbrowser, 'open') as browser, patch.object(monitor, 'create_server', side_effect=AssertionError('duplicate bind')):
                    monitor.main()
                browser.assert_called_once_with(server.health['url'])

    def test_same_port_cannot_spawn_second_server(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                with self.assertRaises(OSError):
                    monitor.create_server(research, Path(t) / 'archive', port=server.server_address[1])

    def test_post_path_traversal_host_origin_and_run_id_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with live_server(Path(t)) as (server, research):
                url = server.health['url']
                cases = [(Request(url + '/api/status', data=b'{}'), 405),
                         (Request(url + '/../AGENTS.md'), 404),
                         (Request(url + '/api/status', headers={'Host': 'evil.invalid:9'}), 403),
                         (Request(url + '/api/status', headers={'Origin': 'http://evil.invalid'}), 403),
                         (Request(url + '/api/status?run=../../file'), 400)]
                for request, code in cases:
                    with self.subTest(code=code, url=request.full_url):
                        with self.assertRaises(HTTPError) as error:
                            urlopen(request, timeout=2)
                        self.assertEqual(error.exception.code, code)

    def test_registration_does_not_delete_changed_or_unknown_helper(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); archive = root / 'archive'; archive.mkdir()
            server = monitor.create_server(root, archive, port=0)
            try:
                with monitor.registration(server):
                    path = root / '.training_monitor.json'
                    path.write_text('{"changed":true}', encoding='utf-8')
                self.assertTrue(path.exists())
                with self.assertRaises(RuntimeError):
                    with monitor.registration(server):
                        self.fail('unknown helper accepted')
            finally:
                server.server_close()

    def test_live_owner_command_mismatch_is_not_stale_cleanup(self):
        value = {'service': monitor.SERVICE, 'pid': 1, 'create_time': 1., 'command': ['old']}
        with patch.object(monitor, '_process_identity', return_value={'alive': True}), patch.object(monitor.psutil, 'Process', return_value=SimpleNamespace(cmdline=lambda: ['different'])):
            with self.assertRaises(RuntimeError):
                monitor.owner_alive(value)

    def test_duration_and_unsafe_redirect_boundaries(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with self.assertRaises(ValueError):monitor.create_server(root, root, port=0, duration_hours=0)
            self.assertFalse(monitor._same_tree(root.parent, root))
            with patch.object(Path, 'is_junction', return_value=True):
                self.assertFalse(monitor._same_tree(root / 'file', root))


class FormalPhaseMonitorTests(unittest.TestCase):
    def test_phase_local_validation_uses_frozen_global_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            schedule = [
                {'tag': 'step030187_first_final', 'global_update': 30187, 'phase': 'first'},
                {'tag': 'step030187_final_initial', 'global_update': 30187, 'phase': 'final'},
                {'tag': 'step031187_final_interval', 'global_update': 31187, 'phase': 'final'},
            ]
            for item, local in zip(schedule, (30187, 0, 1000)):
                write_json(run / 'evaluation' / item['tag'] / 'summary.json',
                           {'tag': item['tag'], 'update': local, 'images': 750,
                            'main_metrics': {'AP50_95': .5}, 'wall_seconds': 320})
            instance = monitor.Monitor(root, root / 'archive')
            try:
                rows, errors = instance._validation(run, {'evaluation_schedule': schedule})
                self.assertFalse(errors)
                self.assertEqual([row['update'] for row in rows], [30187, 30187, 31187])
                self.assertEqual({row['phase_update'] for row in rows[:2]}, {0, 30187})
                self.assertEqual(len({row['tag'] for row in rows}), 3)
            finally:
                instance.close()

    def test_formal_eta_counts_both_boundary_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            research, archive, group, run = fixture(Path(temporary), update=2000)
            manifest = monitor.read_json(run / 'run.json')
            manifest['protocol']['updates'] = 3000
            manifest['protocol']['evaluation_schedule'] = [
                {'tag': 'step000000_PT', 'global_update': 0},
                {'tag': 'step001000', 'global_update': 1000},
                {'tag': 'step002000_first_final', 'global_update': 2000},
                {'tag': 'step002000_final_initial', 'global_update': 2000},
                {'tag': 'step003000_final_final', 'global_update': 3000},
            ]
            write_json(run / 'run.json', manifest)
            append_row(group / 'controller.stdout.log', {'event': 'update', 'update': 2000})
            instance = monitor.Monitor(research, archive)
            try:
                value = instance.snapshot()['run']
                self.assertAlmostEqual(value['eta_seconds'], 1000 * .5 + 3 * 320)
            finally:
                instance.close()

    def test_scalar_keeps_phase_without_uid_lineage(self):
        row = monitor.scalar_point({'update': 30188, 'phase': 'final', 'phase_update': 1, 'uids': ['private']})
        self.assertEqual((row['update'], row['phase'], row['phase_update']), (30188, 'final', 1))
        self.assertNotIn('uids', row)


if __name__ == '__main__':
    unittest.main()
