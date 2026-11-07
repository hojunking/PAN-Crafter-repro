from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ta2.evaluation import file_sha
from panda_wv3_hyper import evaluation, plan, publication as P
from panda_wv3_hyper.transport import prepare, reconcile_rows, _verify_atomic_append
from panda_wv3_hyper.repair_publication_incident import build_request


def simple(step, h, e, val=None):
    result = dict(step=step, fr={'metrics': {'HQNR': h}}, rr={'metrics': {'ERGAS': e}})
    if val is not None:
        result['validation'] = dict(RRE50=val, RRE90=val + .1, Val_ERGAS=e)
    return result


def fixture_record(root, cfg, step, h=.960, e=2.03, val=None):
    directory = Path(root) / 'runs' / cfg['run_id']; directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f'synthetic_{step}.pt'
    checkpoint.write_bytes(f'SYNTHETIC_ONLY_{step}'.encode())
    record = {'schema': 'WV3_HYPER_EVAL_V1', 'run_id': cfg['run_id'],
        'config_sha256': cfg['config_sha256'], 'step': step, 'stage': cfg['stage'],
        'same_state_verified': True, 'reference_warp': False, 'inference_tta': False,
        'model_sha256': 'a' * 64, 'eval_protocol_id': 'b' * 64,
        'recorded_at': '2026-11-08T01:00:00+00:00', 'jqm_variant': 'SYNTHETIC',
        'elapsed_seconds': 2, 'inference_parameters': 2890794,
        'checkpoint': {'path': str(checkpoint), 'file_sha256': file_sha(checkpoint),
                       'model_sha256': 'a' * 64, 'step': step}}
    if cfg['stage'] == 'stage2':
        record['donor_selection'] = {'donor_run_id': cfg['donor_run_id'],
            'selected': {'step': cfg['donor_pin']['selected_step'],
                         'file_sha256': cfg['donor_pin']['checkpoint_sha256']}}
    if val is not None:
        record['validation'] = dict(RRE50=val, RRE90=val + .1, Val_ERGAS=e)
    rre = {'registration': {'valid': True, 'shift': [.1, .2]},
           'rre_px': float(np.linalg.norm([.1, .2]))}
    for split, values in [('rr', {'CC': .99, 'ERGAS': e, 'PSNR': 38., 'Q8': .92,
                                 'RMSE': 26., 'SAM': 2.7, 'SCC': .98, 'SSIM': .97}),
                          ('fr', {'HQNR': h, 'D_s': .03, 'D_lambda': .01, 'JQM': .97})]:
        record[split] = {'per_scene': [{'scene_id': f'WV3:{split}:{i}', 'metrics': values,
                                       'rre_rgb': deepcopy(rre)} for i in range(20)],
                        'metrics': {k: float(np.mean([v] * 20)) for k, v in values.items()}}
    record['test_rre_rgb'] = {s: evaluation.rre_summary(record[s]['per_scene']) for s in ('rr', 'fr')}
    target = directory / f'evaluation_{step:07d}.json'
    target.write_text(json.dumps(record))
    return dict(record, evidence_path=str(target), evidence_sha256=file_sha(target))


def snapshot(value, width=72):
    dest = P.destination()
    return {'metadata': {'spreadsheetId': dest['spreadsheet_id'], 'sheets': [{'properties': {
        'sheetId': dest['sheet_id'], 'title': dest['sheet_title'],
        'gridProperties': {'columnCount': width, 'rowCount': 3000}}}]},
        'header': list(P.HEADERS), 'rows': deepcopy(value['rows']), 'notes': deepcopy(value['notes']),
        'cells': [], 'grid_row_count': 3000}


class FakeSheets:
    BASE = 'https://fake.invalid/'

    def __init__(self, snap):
        self.snap = deepcopy(snap); self.calls = []

    def snapshot(self, dest):
        return deepcopy(self.snap)

    def _request(self, method, url, **kwargs):
        self.calls.append((method, url, deepcopy(kwargs)))
        if url.endswith(':append'):
            raise AssertionError('Heuristic values.append must never be used')
        if url.endswith(':batchUpdate'):
            for req in kwargs['json']['requests']:
                if 'appendCells' in req:
                    for payload in req['appendCells']['rows']:
                        cells = payload['values']
                        self.snap['rows'].append([next(iter(c['userEnteredValue'].values())) for c in cells])
                        self.snap['notes'].append(cells[0]['note'])
                    continue
                self.assert_owned(req)
                update = req['updateCells']; index = update['range']['startRowIndex'] - 5
                cells = update['rows'][0]['values']
                self.snap['rows'][index] = [next(iter(c['userEnteredValue'].values())) for c in cells]
                self.snap['notes'][index] = cells[0]['note']
            return {}
        raise AssertionError(url)

    @staticmethod
    def assert_owned(req):
        if set(req) != {'updateCells'}:
            raise AssertionError('Unexpected destructive/appendCells operation')
        bounds = req['updateCells']['range']
        if bounds['startColumnIndex'] != 0 or bounds['endColumnIndex'] != 64:
            raise AssertionError('Write escaped A:BL')


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.no_cost = patch.object(P, 'shared_cost', return_value=None)
        self.no_cost.start(); self.addCleanup(self.no_cost.stop)

    def test_fixed_endpoint_survives_failed_targets_and_merges_when_representative(self):
        records = [simple(1000, .961, 2.03), simple(29000, .94, 2.2)]
        selected = P.choose(records, 'stage2')
        self.assertEqual([(e['step'], s) for e, s, _ in selected],
            [(1000, 'HQNR_QUALIFIED_MIN_ERGAS|PASS_ALL_STEPS'), (29000, 'FIXED29K')])
        merged = P.choose([simple(29000, .9585, 2.040)], 'stage2')
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][1], 'HQNR_QUALIFIED_MIN_ERGAS|PASS_ALL_STEPS|FIXED29K')
        stage1 = P.choose([simple(1000, .9, 2.1, .1), simple(50000, .95, 2.05, .2)], 'stage1')
        self.assertEqual([(r['step'], s) for r, s, _ in stage1], [(1000, 'BEST_VAL_RRE'), (50000, 'FIXED50K')])

    def test_full_precision_targets_duplicate_records_and_fallback(self):
        below = simple(29000, .958499999, 2.039)
        selected = P.choose([below, deepcopy(below)], 'stage2')
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1], 'BEST_HQNR_FALLBACK|FIXED29K')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            P.choose([below, simple(29000, .96, 2.03)], 'stage2')

    def test_all_registered_cases_have_status_rows_and_coefficients(self):
        with tempfile.TemporaryDirectory() as temp:
            for server, count in [('s1', 30), ('s3', 33)]:
                value = P.project(temp, server)
                self.assertEqual(len(value['rows']), count)
                self.assertEqual(len({r[P.INDEX['Result_ID']] for r in value['rows']}), count)
                self.assertTrue(all(r[P.INDEX['Bucket']] == 'ablations' for r in value['rows']))
                self.assertTrue(all(r[P.INDEX['Experiment']].endswith('(RGB)') for r in value['rows']))
                self.assertIn('alpha=' if server == 's1' else 'lambda_spa=', value['rows'][0][P.INDEX['Approach']])

    def test_status_becomes_representative_without_blank_and_extra_columns_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = plan.resolve('s1', 'H2_A0', 1)
            before = P.project(temp, 's1'); snap = snapshot(before)
            foreign = deepcopy(snap['rows'][0]); foreign[P.INDEX['Campaign_ID']] = 'FOREIGN'
            foreign[P.INDEX['Result_ID']] = 'FOREIGN-ID'
            snap['rows'].insert(0, foreign); snap['notes'].insert(0, 'Do not change')
            fixture_record(temp, cfg, 1000)
            after = P.project(temp, 's1'); requests, additions = prepare(snap, after, 's1')
            self.assertEqual(additions, []); self.assertEqual(len(requests), 1)
            self.assertGreater(requests[0]['updateCells']['range']['startRowIndex'], 5)
            self.assertEqual(requests[0]['updateCells']['range']['endColumnIndex'], 64)
            self.assertTrue(requests[0]['updateCells']['rows'][0]['values'][0]['userEnteredValue'])
            self.assertEqual(snap['metadata']['sheets'][0]['properties']['gridProperties']['columnCount'], 72)

    def test_rep_swap_with_existing_pass_reuses_slots_and_never_clears_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = plan.resolve('s1', 'H2_A0', 1)
            fixture_record(temp, cfg, 1000, .960, 2.039)
            fixture_record(temp, cfg, 2000, .961, 2.038)
            before = P.project(temp, 's1'); snap = snapshot(before)
            fixture_record(temp, cfg, 3000, .962, 2.037)
            after = P.project(temp, 's1'); requests, additions = prepare(snap, after, 's1')
            self.assertEqual(len(additions), 1)
            self.assertTrue(all(req.get('updateCells', {}).get('rows', [{'values': [True]}])[0]['values'][0]
                                for req in requests))
            self.assertEqual(len([r for r in after['rows'] if r[P.INDEX['Run_ID']] == cfg['run_id']]), 3)

    def test_project_rounded_metrics_keep_raw_proof_and_donor_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = plan.resolve('s1', 'H2_A0', 1)
            fixture_record(temp, cfg, 29000, .958499, 2.041234)
            value = P.project(temp, 's1')
            row = next(r for r in value['rows'] if r[P.INDEX['Run_ID']] == cfg['run_id'])
            self.assertEqual(row[P.INDEX['HQNR↑']], .9585)
            note = json.loads(row[P.INDEX['Original description / notes']])
            self.assertFalse(note['publication']['joint_pass'])
            self.assertNotEqual(note['reporting']['raw_metrics']['HQNR↑'], .9585)
            self.assertEqual(prepare(snapshot(value), value, 's1'), ([], []))
            tampered = snapshot(value)
            i = next(i for i, r in enumerate(tampered['rows']) if r[P.INDEX['Run_ID']] == cfg['run_id'])
            note['donor_selection']['donor_run_id'] = 'wrong'
            tampered['rows'][i][P.INDEX['Original description / notes']] = json.dumps(note)
            with self.assertRaisesRegex(ValueError, 'donor_selection'):
                prepare(tampered, value, 's1')

    def test_reused_pruned_weights_only_via_verified_loader(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as old:
            cfg = plan.resolve('s1', 'H2_REF', 1)
            source = fixture_record(old, cfg, 29000)
            Path(source['checkpoint']['path']).unlink()
            source['reuse'] = {'source_run_id': 'ORIGINAL', 'weight_status': 'METRICS_ONLY_WEIGHT_PRUNED',
                               'receipt_sha256': 'c' * 64, 'new_inference_performed': False}
            directory = Path(temp) / 'runs' / cfg['run_id']; directory.mkdir(parents=True)
            (directory / 'reuse_receipt.json').write_text('{}')
            with patch('panda_wv3_hyper.reuse.load_reused_records', return_value=[source]) as loader:
                value = P.project(temp, 's1'); loader.assert_called_once_with(cfg, directory)
            row = next(r for r in value['rows'] if r[P.INDEX['Run_ID']] == cfg['run_id'])
            self.assertEqual(row[P.INDEX['Status']], 'REUSED_COMPLETE')
            note = json.loads(row[P.INDEX['Original description / notes']])
            self.assertEqual(note['reuse']['weight_status'], 'METRICS_ONLY_WEIGHT_PRUNED')
            with patch('panda_wv3_hyper.reuse.load_reused_records', side_effect=ValueError('invalid receipt')):
                with self.assertRaisesRegex(ValueError, 'invalid receipt'):
                    P.project(temp, 's1')

    def test_fresh_missing_checkpoint_and_incomplete_eval_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = plan.resolve('s1', 'H2_A0', 1)
            source = fixture_record(temp, cfg, 1000)
            Path(source['checkpoint']['path']).unlink()
            with self.assertRaises(FileNotFoundError):
                P.project(temp, 's1')
        with tempfile.TemporaryDirectory() as temp:
            source = fixture_record(temp, cfg, 1000)
            path = Path(source['evidence_path']); record = json.loads(path.read_text())
            record['rr']['per_scene'].pop(); path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                P.project(temp, 's1')

    def test_mocked_append_readback_and_idempotent_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            value = P.project(temp, 's3'); empty = snapshot(dict(value, rows=[], notes=[]))
            adapter = FakeSheets(empty)
            first = reconcile_rows(adapter, value, Path(temp) / 'pub', 's3', apply=True)
            self.assertTrue(first['readback_verified']); self.assertEqual(first['row_count'], 33)
            self.assertEqual(sum('appendCells' in r for c in adapter.calls
                                 for r in c[2]['json']['requests']), 1)
            count = len(adapter.calls)
            second = reconcile_rows(adapter, value, Path(temp) / 'pub', 's3', apply=True)
            self.assertTrue(second['readback_verified']); self.assertEqual(len(adapter.calls), count)
            self.assertEqual(adapter.snap['metadata']['sheets'][0]['properties']['gridProperties']['columnCount'], 72)

    def test_gapped_foreign_subtables_never_choose_earlier_table_end(self):
        with tempfile.TemporaryDirectory() as temp:
            value = P.project(temp, 's1'); snap = snapshot(dict(value, rows=[], notes=[]))
            # A short A:O table is followed by another table and a distant tail.
            # The failed Google values.append path chose the earlier A:O end.
            foreign = [''] * 64
            foreign[0] = 'foreign short subtable'; foreign[P.INDEX['Campaign_ID']] = 'OLD'
            tail = deepcopy(foreign); tail[0] = 'foreign distant tail'
            snap['rows'] = [deepcopy(foreign), deepcopy(foreign)] + [[''] * 64 for _ in range(90)] + [tail]
            snap['notes'] = [''] * len(snap['rows'])
            before = deepcopy(snap); adapter = FakeSheets(snap)
            receipt = reconcile_rows(adapter, value, Path(temp) / 'pub', 's1', apply=True)
            self.assertEqual(adapter.snap['rows'][:len(before['rows'])], before['rows'])
            self.assertEqual(min(receipt['appended_rows']), 6 + len(before['rows']))
            self.assertTrue(all(not c[1].endswith(':append') for c in adapter.calls))
            request = adapter.calls[0][2]['json']['requests'][0]
            additions = request['appendCells']['rows']
            bad = deepcopy(adapter.snap)
            bad['rows'][0] = bad['rows'][-1]; bad['notes'][0] = bad['notes'][-1]
            with self.assertRaisesRegex(Exception, 'overlaps data'):
                _verify_atomic_append(before, bad, additions)

    def test_native_controls_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            value = P.project(temp, 's1'); snap = snapshot(value)
            snap['cells'] = [[{'userEnteredValue': {'formulaValue': '=1+1'}}]]
            with self.assertRaisesRegex(Exception, 'Formula'):
                prepare(snap, value, 's1')

    def test_incident_repair_only_restores_exact_known_writes_and_original_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            value = P.project(temp, 's1'); current = snapshot(value)
            current['cells'] = [[{'userEnteredValue': {'stringValue': 'placeholder'},
                                   'note': 'original note'}] for _ in current['rows']]
            original = deepcopy(current)
            original['rows'][0] = [''] * 64
            original['rows'][0][0] = 'Original foreign experiment'
            original['rows'][0][P.INDEX['Campaign_ID']] = 'FOREIGN'
            original['cells'][0] = [
                {'userEnteredValue': {'stringValue': 'Original foreign experiment'},
                 'note': 'original note', 'userEnteredFormat': {'numberFormat': {'pattern': '0.00'}}}]
            request, summary = build_request(original, current, value['rows'], first=6, last=6)
            self.assertEqual(summary['changed_rows'], [6])
            self.assertEqual(request['requests'][0]['updateCells']['range']['endColumnIndex'], 64)
            restored = request['requests'][0]['updateCells']['rows'][0]['values'][0]
            self.assertEqual(restored['userEnteredValue']['stringValue'], 'Original foreign experiment')
            self.assertEqual(restored['note'], 'original note')
            self.assertNotIn('userEnteredFormat', restored)
            altered = deepcopy(current); altered['rows'][0][0] = 'Someone else edited this'
            with self.assertRaisesRegex(ValueError, 'not an exact known'):
                build_request(original, altered, value['rows'], first=6, last=6)
            altered = deepcopy(current); altered['cells'][0][0]['note'] = 'new external note'
            with self.assertRaisesRegex(ValueError, 'notes changed'):
                build_request(original, altered, value['rows'], first=6, last=6)


if __name__ == '__main__':
    unittest.main()
