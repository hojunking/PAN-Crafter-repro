"""CPU/local-only evidence and fake-Sheet tests; never contact a live sheet."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from reporting_extra import sensor_backfill as b


def write(path, value, yaml_file=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value) if yaml_file else json.dumps(value))


def fixture(root, trainer='g20', server='s1'):
    run = trainer.upper() + '_GF2_TEST_' + server
    wd = root / 'work_dir' / run
    field = dict(run_id=run, server_id=server, role='T', sensor='GF2', num_bands=4,
        campaign_id=b.CAMPAIGNS[trainer], teacher_seed=91002, student_seed=None,
        sheet_tab=b.TABS[server], input_layout='P0', width=112, depth=[1, 2, 3],
        reference_id='R1', teacher_alias='T1', profile='C100')
    cfg = dict(trainer=trainer, work_dir='work_dir/' + run, seed=91002, num_bands=4, max_pixel=1023.,
        model_args=dict(hidden_size=112, depth=[1, 2, 3]), **{trainer: field})
    source = dict(files={'example.py': 'a' * 64})
    source['content_sha256'] = b.object_sha(source['files'])
    identity = dict(config_sha256=b.object_sha(cfg), data_sha256='c' * 64, source_identity=source)
    context = dict(identity, campaign_id=b.CAMPAIGNS[trainer], run_id=run)
    write(wd / 'meta/config.resolved.yaml', cfg, True)
    write(wd / 'meta/training_start_manifest.json', dict(context, started_at_utc='2026-09-20T19:00:00+00:00'))
    write(wd / 'meta/training_status.json', dict(training_complete=True, actual_updates=50000,
        training_seconds=3600., updated_at_utc='2026-09-20T20:30:00+00:00'))
    write(wd / 'official/postrun_status.json', dict(identity, actual_updates=50000,
        official_complete=True, sheet_uploaded=True, completed_at_utc='2026-09-20T20:31:00+00:00'))
    write(wd / 'official/profile.json', dict(identity, num_bands=4, sensor='GF2', params_m=4.31,
        flops_g=124.321234, infer_ms=8.1234567, mem_mb=321.123))
    records = []
    for step in b.GRID:
        candidate = dict(identity, update=step, model_sha256=__import__('hashlib').sha256(b'model').hexdigest())
        records.append(dict(update=step, checkpoint_identity=candidate, val_ergas=.51,
            rr={key: .5 for key in b.RR.values()}, fr={key: .96 for key in b.FR.values()}))
    write(wd / 'official/raw_grid.json', dict(context, complete=True, records=records))
    record = records[-1] if trainer == 'g20' else records[0]
    name, selection_id = ('exact50k', 'EXACT50K') if trainer == 'g20' else ('raw_max', 'RAW_MAX')
    selected = dict(context, official_complete=True, normal_same_step_A_U=True, sensor='GF2',
        selection_id=selection_id, n_evaluated=50, step=record['update'],
        checkpoint_sha256=record['checkpoint_identity']['model_sha256'],
        **{k: record[k] for k in ('rr', 'fr', 'val_ergas', 'checkpoint_identity')})
    write(wd / ('official/' + name + '.json'), selected)
    write(wd / f'candidates/{record["update"]}/identity.json', record['checkpoint_identity'])
    (wd / f'candidates/{record["update"]}/model.safetensors').write_bytes(b'model')
    write(wd / 'official/upload_receipt.json', dict(readback_verified=True, run_id=run,
        server=server, sensor='GF2', campaign_id=b.CAMPAIGNS[trainer], worksheet=b.TABS[server], gid=123, row=4))
    return wd


def table_for(evidence):
    p = evidence['prefix']
    values = {'Run': evidence['run_id'], 'Notes': 'Original notes: ACCEPTED_WITH_PARITY_EXCEPTION; SHA=abc.',
        p + ' sensor': 'GF2', p + ' campaign': evidence['campaign_id'], p + ' run id': evidence['run_id'],
        p + ' server': evidence['server'], p + ' source SHA256': evidence['source_sha256'], **evidence['metrics'],
        'Params(M)': 4.31, 'FLOPs(G)': 124.321234, 'Date': '', 'Unknown user column': 'preserve me'}
    return [[], [], list(values), list(values.values())]


class FakeSheet:
    id = 123
    title = 'GF2-s1'

    def __init__(self, table, controls=None):
        self.table = copy.deepcopy(table)
        self.col_count = max(map(len, table))
        self.row_count = len(table)
        self.spreadsheet = self
        self.controls = controls or dict(properties=dict(sheetId=self.id))
        self.calls = []
        self.corrupt_after_write = False

    def get_all_values(self, **kwargs):
        self.calls.append(('read', kwargs))
        return copy.deepcopy(self.table)

    def row_values(self, row, **kwargs):
        return copy.deepcopy(self.table[row - 1])

    def fetch_sheet_metadata(self, **kwargs):
        return dict(sheets=[copy.deepcopy(self.controls)])

    def add_cols(self, number):
        self.calls.append(('add_cols', number)); self.col_count += number

    def batch_update(self, edits, **kwargs):
        self.calls.append(('update', copy.deepcopy(edits)))
        for item in edits:
            row, col = list(b._range_cells(item['range']))[0]
            while len(self.table[row - 1]) < col:
                self.table[row - 1].append('')
            self.table[row - 1][col - 1] = item['values'][0][0]
        if self.corrupt_after_write:
            col = self.table[2].index('ERGAS↓')
            self.table[3][col] = 9.99

    def batch_format(self, formats):
        self.calls.append(('format', copy.deepcopy(formats)))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wd = fixture(self.root)

    def mutate(self, name, function):
        path = self.wd / name
        value = json.loads(path.read_text()); function(value); write(path, value)

    def test_complete_actual_evidence(self):
        evidence = b.load_evidence(self.wd, 's1')
        self.assertEqual(evidence['metadata']['Date'], '2026-09-21')
        self.assertEqual(evidence['metadata']['Selection'], 'Exact50K')
        self.assertEqual(evidence['metadata']['Wall(h)'], 1.5)
        self.assertNotIn('Train(h)', evidence['metadata'])
        self.assertEqual(evidence['metrics']['Train(h)'], 1.)
        self.assertIn('meta/training_start_manifest.json', evidence['source_files'])

    def test_qg_raw_max_uses_earliest_tie_not_scc(self):
        wd = fixture(self.root, 'qg40', 's3')
        p = wd / 'official/raw_grid.json'; grid = b.read_json(p)
        grid['records'][1]['rr']['scc'] = .99
        write(p, grid)
        self.assertEqual(b.load_evidence(wd, 's3')['metadata']['Selection'], 'RAW_MAX')

    def test_wrong_server_rejected(self):
        with self.assertRaisesRegex(ValueError, 'server'):
            b.load_evidence(self.wd, 's2')

    def test_incomplete_training_rejected(self):
        self.mutate('meta/training_status.json', lambda d: d.update(actual_updates=49999))
        with self.assertRaisesRegex(ValueError, 'Exact50K'):
            b.load_evidence(self.wd, 's1')

    def test_unverified_original_upload_rejected(self):
        self.mutate('official/upload_receipt.json', lambda d: d.update(readback_verified=False))
        with self.assertRaisesRegex(ValueError, 'verified'):
            b.load_evidence(self.wd, 's1')

    def test_tampered_start_config_rejected(self):
        self.mutate('meta/training_start_manifest.json', lambda d: d.update(config_sha256='f' * 64))
        with self.assertRaisesRegex(ValueError, 'config checksum'):
            b.load_evidence(self.wd, 's1')

    def test_tampered_source_rejected(self):
        self.mutate('official/exact50k.json', lambda d: d['source_identity'].update(content_sha256='f' * 64))
        with self.assertRaisesRegex(ValueError, 'source identity'):
            b.load_evidence(self.wd, 's1')

    def test_tampered_selected_metrics_rejected(self):
        self.mutate('official/exact50k.json', lambda d: d['rr'].update(ergas=123))
        with self.assertRaisesRegex(ValueError, 'Main selection'):
            b.load_evidence(self.wd, 's1')

    def test_tampered_checkpoint_bytes_rejected(self):
        (self.wd / 'candidates/50000/model.safetensors').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
            b.load_evidence(self.wd, 's1')

    def test_missing_grid_candidate_rejected(self):
        self.mutate('official/raw_grid.json', lambda d: d['records'].pop())
        with self.assertRaisesRegex(ValueError, 'fixed-grid'):
            b.load_evidence(self.wd, 's1')

    def test_recovery_time_not_reported_as_finish(self):
        self.mutate('meta/training_status.json', lambda d: d.update(recovered_from_exact50k=True))
        evidence = b.load_evidence(self.wd, 's1')
        self.assertEqual(evidence['metadata']['G20 completed UTC'], '')
        self.assertEqual(evidence['metadata']['Wall(h)'], '')

    def test_helper_cannot_return_result_value(self):
        with patch('reporting_extra.sensor_sheet.metadata_values', return_value={'Train(h)': 1., 'HQNR↑': .99}):
            with self.assertRaisesRegex(ValueError, 'original result'):
                b.load_evidence(self.wd, 's1')

    def test_ready_requires_completed_uploaded_local_runs(self):
        fixture(self.root, 'qg40', 's3')
        self.assertEqual(b.ready_runs(self.root, 's1'), [self.wd])
        self.mutate('official/postrun_status.json', lambda d: d.update(sheet_uploaded=False))
        self.assertEqual(b.ready_runs(self.root, 's1'), [])


class SheetTests(EvidenceTests):
    def setUp(self):
        super().setUp()
        self.evidence = b.load_evidence(self.wd, 's1')
        self.table = table_for(self.evidence)
        self.sheet = FakeSheet(self.table)

    def test_dry_run_does_not_write_sheet_or_files(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = b.sync_run(self.wd, 's1', self.sheet)
        self.assertFalse(result['apply'])
        self.assertEqual(self.sheet.table, self.table)
        self.assertEqual([k for k, _ in self.sheet.calls], ['read'])
        self.assertEqual(sorted(str(p) for p in self.root.rglob('*')), before)

    def test_apply_preserves_numbers_notes_original_receipt_and_other_cells(self):
        receipt = (self.wd / 'official/upload_receipt.json').read_bytes()
        result = b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertTrue(result['readback_verified'])
        labels = b.labels_for(self.sheet.table[2]); row = self.sheet.table[3]
        self.assertIn('Original notes: ACCEPTED_WITH_PARITY_EXCEPTION; SHA=abc.', row[labels['Notes'] - 1])
        for label, value in self.evidence['metrics'].items():
            self.assertEqual(row[labels[label] - 1], value)
        self.assertEqual(row[labels['Unknown user column'] - 1], 'preserve me')
        self.assertEqual((self.wd / 'official/upload_receipt.json').read_bytes(), receipt)
        self.assertTrue(Path(result['snapshot']).is_file())
        self.assertTrue(list((self.root / 'work_dir/_sensor_sheet').rglob('*.receipt.json')))
        edits = next(v for k, v in self.sheet.calls if k == 'update')
        metric_cells = {b.a1(4, labels[k]) for k in self.evidence['metrics']}
        self.assertFalse(metric_cells & {edit['range'] for edit in edits})

    def test_second_pass_idempotent(self):
        b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.sheet.calls.clear()
        result = b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertTrue(result['unchanged'])
        self.assertEqual([k for k, _ in self.sheet.calls], ['read'])

    def test_original_uploader_notes_refresh_is_repaired(self):
        b.sync_run(self.wd, 's1', self.sheet, apply=True)
        labels = b.labels_for(self.sheet.table[2])
        self.sheet.table[3][labels['Notes'] - 1] = 'Refreshed original evidence'
        result = b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertFalse(result.get('unchanged', False))
        self.assertIn('Refreshed original evidence', self.sheet.table[3][labels['Notes'] - 1])

    def test_wrong_sheet_rejected_before_mutation(self):
        self.sheet.title = 'GF2-s2'
        with self.assertRaisesRegex(ValueError, 'Worksheet'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertEqual(self.sheet.calls, [])

    def test_no_new_or_duplicate_rows(self):
        for table in (self.table[:3], self.table + [self.table[3]]):
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                b.plan_update(table, self.evidence)

    def test_source_mismatch_rejected(self):
        col = self.table[2].index('G20 source SHA256')
        self.table[3][col] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'conflict'):
            b.plan_update(self.table, self.evidence)

    def test_rounded_metric_not_treated_as_exact(self):
        self.evidence['metrics']['ERGAS↓'] = .500000123
        with self.assertRaisesRegex(ValueError, 'full-precision'):
            b.plan_update(self.table, self.evidence)

    def test_formula_validation_and_merge_guard(self):
        col = self.table[2].index('Notes')
        for cell in ({'userEnteredValue': {'formulaValue': '=A1'}}, {'dataValidation': {'condition': {}}}, {'chipRuns': [{}]}):
            self.sheet.controls = dict(properties=dict(sheetId=123), data=[dict(startRow=3, startColumn=col,
                rowData=[dict(values=[cell])])])
            with self.assertRaisesRegex(ValueError, 'controlled cell'):
                b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.sheet.controls = dict(properties=dict(sheetId=123), merges=[dict(sheetId=123, startRowIndex=3,
            endRowIndex=4, startColumnIndex=0, endColumnIndex=2)])
        with self.assertRaisesRegex(ValueError, 'merged'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertFalse(any(k == 'update' for k, _ in self.sheet.calls))

    def test_sheetwide_protection_guards_new_columns(self):
        self.sheet.controls = dict(properties=dict(sheetId=123), protectedRanges=[dict(range=dict(sheetId=123))])
        with self.assertRaisesRegex(ValueError, 'protected'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)

    def test_metric_format_formula_is_guarded_too(self):
        col = self.table[2].index('ERGAS↓')
        self.sheet.controls = dict(properties=dict(sheetId=123), data=[dict(startRow=3, startColumn=col,
            rowData=[dict(values=[dict(userEnteredValue=dict(formulaValue='=0.5'))])])])
        with self.assertRaisesRegex(ValueError, 'formula'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)

    def test_user_owned_extra_column_is_not_formatted(self):
        plan = b.plan_update(self.table, self.evidence)
        user_cell = b.a1(4, self.table[2].index('Unknown user column') + 1)
        self.assertTrue(all(':' not in item['range'] for item in plan['formats']))
        self.assertNotIn(user_cell, {item['range'] for item in plan['formats']})

    def test_notes_race_refused_before_first_write(self):
        original = self.sheet.fetch_sheet_metadata
        def concurrent_edit(**kwargs):
            result = original(**kwargs)
            self.sheet.table[3][self.table[2].index('Notes')] = 'A new manual note'
            return result
        self.sheet.fetch_sheet_metadata = concurrent_edit
        with self.assertRaisesRegex(ValueError, 'changed during reporting'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertFalse(any(k in ('add_cols', 'update', 'format') for k, _ in self.sheet.calls))
        self.assertFalse((self.root / 'work_dir/_sensor_sheet').exists())

    def test_header_race_refused_before_first_write(self):
        original = self.sheet.fetch_sheet_metadata
        def concurrent_edit(**kwargs):
            result = original(**kwargs)
            self.sheet.table[2][0] = 'Manual renamed Run'
            return result
        self.sheet.fetch_sheet_metadata = concurrent_edit
        with self.assertRaisesRegex(ValueError, 'changed during reporting'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertFalse(any(k in ('add_cols', 'update', 'format') for k, _ in self.sheet.calls))

    def test_original_metric_readback_change_fails_and_keeps_snapshot(self):
        self.sheet.corrupt_after_write = True
        with self.assertRaisesRegex(ValueError, 'Original row changed'):
            b.sync_run(self.wd, 's1', self.sheet, apply=True)
        store = self.root / 'work_dir/_sensor_sheet'
        self.assertTrue(list(store.rglob('*.before.json')))
        self.assertFalse(list(store.rglob('*.receipt.json')))

    def test_original_writer_lock_blocks_apply(self):
        import fcntl
        with (self.root / 'work_dir/.g20_sheet_write.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                b.sync_run(self.wd, 's1', self.sheet, apply=True)
        self.assertEqual(self.sheet.calls, [])


class IndependenceTests(unittest.TestCase):
    def test_import_does_not_load_numerical_packages(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; import reporting_extra.sensor_backfill; '
            'assert not any(k.split(".")[0] in {"torch", "g20", "qg40", "fh12", "numpy"} for k in sys.modules)'],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
