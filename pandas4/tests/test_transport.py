from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote

from pandas4 import publication as P
from pandas4.tests.test_reporting import config, persist
from sheet_publish_v2.transport import _typed, TransportError
from pandas4.transport import reconcile_rows


class FakeGoogle:
    """An explicitly cleared tail affects appendCells, NOT values.append."""
    BASE = 'https://fake/'

    def __init__(self, value):
        d = value['destination']
        self.data = {'metadata': {'spreadsheetId': d['spreadsheet_id'], 'sheets': [{
            'properties': {'sheetId': d['sheet_id'], 'title': d['sheet_title'],
                           'gridProperties': {'rowCount': 10000, 'columnCount': 64}}}]},
            'header': list(P.HEADERS), 'rows': [], 'notes': [], 'cells': [], 'grid_row_count': 10000}
        self.calls = []
        self.cleared_extent = 9477
        self.before_append = None
        self.fail_after_append = False
        self.fail_before_notes = False

    def snapshot(self, dest):
        return deepcopy(self.data)

    def add(self, row, note=''):
        self.data['rows'].append(list(row))
        self.data['notes'].append(note)
        self.data['cells'].append([{'userEnteredValue': _typed(v), 'note': note if i == 0 else ''}
                                   for i, v in enumerate(row)])

    def _request(self, method, url, **kwargs):
        self.calls.append((method, url, deepcopy(kwargs)))
        if url.endswith(':append'):
            assert unquote(url).endswith("/values/'WV3-ablations'!A5:BL:append")
            assert kwargs['params'] == {'valueInputOption': 'RAW', 'insertDataOption': 'OVERWRITE'}
            if self.before_append:
                hook, self.before_append = self.before_append, None
                hook()
            start = 6 + len(self.data['rows'])
            for row in kwargs['json']['values']:
                self.add(row)
            if self.fail_after_append:
                self.fail_after_append = False
                raise TimeoutError('Committed append, lost response')
            return {'updates': {'updatedRange': "'WV3-ablations'!A%d:BL%d" %
                                (start, len(self.data['rows']) + 5)}}
        if self.fail_before_notes:
            self.fail_before_notes = False
            raise TimeoutError('Notes request not committed')
        for req in kwargs['json']['requests']:
            assert list(req) == ['updateCells'], 'No appendCells, insert, sort or delete'
            update = req['updateCells']
            assert update['fields'] == 'userEnteredValue,note'
            i = update['range']['startRowIndex'] - 5
            cells = update['rows'][0]['values']
            self.data['rows'][i] = [next(iter(c['userEnteredValue'].values())) for c in cells]
            self.data['notes'][i] = cells[0]['note']
            self.data['cells'][i] = deepcopy(cells)
        return {}


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = config()
        persist(self.root, self.cfg)
        self.value = P.project_campaign(self.root, 's4')
        self.api = FakeGoogle(self.value)

    def publish(self, value=None, server='s4'):
        return reconcile_rows(self.api, value or self.value, self.root, server, apply=True)

    def test_explicit_empty_tail_does_not_displace_new_row(self):
        historic = ['history'] + [''] * 63
        for _ in range(25):
            self.api.add(historic)
        before = deepcopy(self.api.data['rows'])
        result = self.publish()
        self.assertEqual(result['sheet_rows'], [31])
        self.assertEqual(self.api.data['rows'][:25], before)
        self.assertEqual(self.api.data['notes'][-1], self.value['notes'][0])
        self.assertEqual(len(list(self.root.glob('cloud_backup_*.json'))), 1)
        self.assertEqual(self.api.cleared_extent, 9477)

    def test_lost_append_response_is_recovered_without_duplicate(self):
        self.api.fail_after_append = True
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertEqual(self.api.data['notes'], [''])
        self.assertTrue(self.publish()['readback_verified'])
        self.assertEqual(len(self.api.data['rows']), 1)
        self.assertEqual(sum(url.endswith(':append') for _, url, _ in self.api.calls), 1)

    def test_failed_notes_phase_recovers(self):
        self.api.fail_before_notes = True
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertTrue(self.publish()['readback_verified'])
        self.assertEqual(len(self.api.data['rows']), 1)

    def test_concurrent_s1_and_s3_append_after_s4_snapshot_keeps_all(self):
        from panda40 import publication as old_publication
        from panda40.tests.test_reporting import config as old_config, persist as old_persist
        from panda40.transport import reconcile_rows as old_reconcile
        others = []
        for server in ('s1', 's3'):
            oldroot = self.root / server
            old_persist(oldroot, old_config(server))
            other = old_publication.project_campaign(oldroot, server)
            others.append((server, oldroot, other))
        def concurrent():
            for server, oldroot, other in others:
                old_reconcile(self.api, other, oldroot, server, apply=True)
        self.api.before_append = concurrent
        self.assertEqual(self.publish()['sheet_rows'], [8])
        for i, (_, _, other) in enumerate(others):
            self.assertEqual(self.api.data['rows'][i], other['rows'][0])
            self.assertEqual(self.api.data['notes'][i], other['notes'][0])
        self.assertEqual(self.api.data['rows'][2], self.value['rows'][0])
        # Later fallback replacement must not clear or modify either foreign row.
        persist(self.root, self.cfg, 4000, .961, 2.04)
        self.publish(P.project_campaign(self.root, 's4'))
        for i, (_, _, other) in enumerate(others):
            self.assertEqual(self.api.data['rows'][i], other['rows'][0])
            self.assertEqual(self.api.data['notes'][i], other['notes'][0])

    def test_repeated_readback_has_no_writes(self):
        self.publish()
        self.api.calls.clear()
        self.assertTrue(self.publish()['readback_verified'])
        self.assertEqual(self.api.calls, [])

    def test_fallback_to_pass_only_clears_owned_row(self):
        self.api.add(['history'] + [''] * 63, 'preserve')
        self.publish()
        persist(self.root, self.cfg, 4000, .961, 2.04)
        result = self.publish(P.project_campaign(self.root, 's4'))
        self.assertEqual(result['sheet_rows'], [8])
        self.assertEqual(self.api.data['rows'][0][0], 'history')
        self.assertEqual(self.api.data['notes'][0], 'preserve')
        self.assertEqual(self.api.data['rows'][1], [''] * 64)

    def test_empty_gap_and_tail_data_are_not_overwritten(self):
        self.api.add([''] * 64)
        self.api.add(['foreign tail'] + [''] * 63)
        self.assertEqual(self.publish()['sheet_rows'], [8])
        self.assertEqual(self.api.data['rows'][1][0], 'foreign tail')

    def test_raw_string_zero_and_false_roundtrip(self):
        value = deepcopy(self.value)
        value['rows'][0][0] = '=NOT_A_FORMULA()'
        value['rows'][0][1:3] = [0, False]
        # Server ownership must remain valid for the roundtrip.
        value['rows'][0][P.INDEX['Server']] = 's4'
        self.publish(value)
        self.assertEqual(self.api.data['rows'][0], value['rows'][0])

    def test_unanchored_or_native_tail_blocks_before_any_write(self):
        for cell in ({'note': 'manual'}, {'dataValidation': {'strict': True}},
                     {'chipRuns': [{'startIndex': 0}]},
                     {'userEnteredValue': {'formulaValue': '=""'}}):
            with self.subTest(cell=cell):
                api = FakeGoogle(self.value)
                api.add([''] * 64)
                api.data['cells'][0][0] = cell
                with self.assertRaises(TransportError):
                    reconcile_rows(api, self.value, self.root, 's4', apply=True)
                self.assertFalse(api.calls)
        self.api.add(['', 'manual data'] + [''] * 62)
        with self.assertRaisesRegex(ValueError, 'Unanchored'):
            self.publish()
        self.assertFalse(self.api.calls)

    def test_append_target_protection_blocks(self):
        self.api.data['metadata']['sheets'][0]['protectedRanges'] = [{'range': {
            'sheetId': self.value['destination']['sheet_id'], 'startRowIndex': 5, 'endRowIndex': 6}}]
        with self.assertRaisesRegex(TransportError, 'Protected'):
            self.publish()
        self.assertFalse(self.api.calls)

    def test_duplicates_unknown_ownership_and_cost_conflicts_block(self):
        self.api.add(self.value['rows'][0], self.value['notes'][0])
        value = deepcopy(self.value)
        value['owned_run_keys'] = []
        with self.assertRaisesRegex(ValueError, 'absent from local'):
            self.publish(value)
        self.api.add(self.value['rows'][0], self.value['notes'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.publish()
        value = deepcopy(self.value)
        value['conflicts'] = ['MISSING_COST']
        with self.assertRaisesRegex(ValueError, 'cost conflicts'):
            self.publish(value)
        self.assertFalse(self.api.calls)

    def test_dry_run_cannot_write(self):
        report = reconcile_rows(self.api, self.value, self.root, 's4')
        self.assertEqual(report['new_rows'], 1)
        self.assertFalse(self.api.calls)



if __name__ == '__main__':
    unittest.main()
