import re
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from qg40.upload import (apply_upsert, metric_formats, plan_upsert, selection_values,
                        upload_run, validate_target_metadata)
from qg40.sheet_helpers import controlled_reason, fetch_controls
from qg40.common import object_sha


def parse_cell(cell):
    letters, row = re.fullmatch(r'([A-Z]+)([0-9]+)', cell).groups()
    col = 0
    for char in letters:
        col = col * 26 + ord(char) - 64
    return int(row), col


class Worksheet:
    title, id, col_count, row_count = 'QB-s1', 77, 12, 50
    def __init__(self):
        self.table = [[], [], ['Run', 'QG40 sensor', 'QG40 campaign', 'QG40 run id', 'QG40 upload status', 'Q4↑'], ['benchmark', '', '', '', '', '.9']]
        self.spreadsheet = self
        self.writes, self.formats, self.corrupt = [], [], False
    def get_all_values(self):
        return [row[:] for row in self.table]
    def row_values(self, number, **kwargs):
        row = self.table[number - 1][:] if number <= len(self.table) else []
        if self.corrupt and kwargs and number > 4:
            row[5] = 0
        return row
    def fetch_sheet_metadata(self, **kwargs):
        return {'sheets': [{'properties': {'sheetId': self.id}, 'data': []}]}
    def add_cols(self, n):
        self.col_count += n
    def add_rows(self, n):
        self.row_count += n
    def batch_update(self, edits, **kwargs):
        self.writes.extend(edits)
        for edit in edits:
            row, col = parse_cell(edit['range'])
            while len(self.table) < row:
                self.table.append([])
            while len(self.table[row - 1]) < col:
                self.table[row - 1].append('')
            self.table[row - 1][col - 1] = edit['values'][0][0]
    def batch_format(self, formats):
        self.formats.extend(formats)


def values():
    return {'Run': 'QGBASE_QB_test', 'QG40 sensor': 'QB', 'QG40 campaign': 'campaign',
            'QG40 run id': 'QGBASE_QB_test', 'QG40 upload status': 'READBACK_PENDING',
            'Q4↑': .987654321, 'Target step': 2020, 'Target JQM↑': '', 'QG40 Student seed': 82001}


class UploadTests(unittest.TestCase):
    def test_upload_receipt_keeps_effective_merged_payload_hash(self):
        case = SimpleNamespace(server_id='s1', sensor='QB')
        verified = {'payload_sha256': 'effective-merged-payload-sha', 'readback_verified': True}
        with patch('qg40.upload.row_values', return_value=values()), \
                patch('qg40.upload.case_for', return_value=case), \
                patch('qg40.upload.Path.read_text', return_value='{}'), \
                patch('qg40.upload.locked', return_value=nullcontext()), \
                patch('qg40.upload.apply_upsert', return_value=verified), \
                patch('qg40.upload.atomic_json') as save:
            receipt = upload_run('fixture', root='/unused-readonly', activated=True, worksheet=Worksheet())
        self.assertEqual(receipt['payload_sha256'], 'effective-merged-payload-sha')
        self.assertEqual(save.call_args.args[1]['payload_sha256'], 'effective-merged-payload-sha')

    def test_existing_manual_notes_are_preserved_idempotently_and_hashed(self):
        ws = Worksheet()
        incoming = dict(values(), Notes='Official note; strict FAILED; authorized exception.')
        original = dict(incoming)
        apply_upsert(ws, incoming)
        notes_index = ws.table[2].index('Notes')
        ws.table[4][notes_index] += '\nHuman note: do not discard.'
        expected_notes = ws.table[4][notes_index]
        plan = plan_upsert(ws.table, ws.table[2], incoming)
        self.assertEqual(plan['values']['Notes'], expected_notes)
        receipt = apply_upsert(ws, incoming)
        self.assertEqual(ws.table[4][notes_index], expected_notes)
        self.assertEqual(receipt['payload_sha256'], object_sha(plan['values']))
        self.assertNotEqual(receipt['payload_sha256'], object_sha(incoming))
        self.assertEqual(apply_upsert(ws, incoming)['payload_sha256'], receipt['payload_sha256'])
        self.assertEqual(ws.table[4][notes_index], expected_notes)
        self.assertEqual(incoming, original)

    def test_distinct_manual_notes_merge_before_readback(self):
        ws = Worksheet()
        incoming = dict(values(), Notes='New automatic note.')
        apply_upsert(ws, incoming)
        notes_index = ws.table[2].index('Notes')
        ws.table[4][notes_index] = 'Independent human note.'
        apply_upsert(ws, incoming)
        self.assertEqual(ws.table[4][notes_index],
                         'New automatic note.\n[Previous Sheet note] Independent human note.')

    def test_unowned_protected_cells_are_neither_written_nor_formatted(self):
        ws = Worksheet()
        ws.table[2].extend(['Date', 'RAW_MAX HQNR↑', 'Human field'])
        ws.fetch_sheet_metadata = lambda **kw: {'sheets': [{
            'properties': {'sheetId': ws.id},
            'protectedRanges': [{'range': {'sheetId': ws.id, 'startRowIndex': 4,
                'endRowIndex': 5, 'startColumnIndex': 6, 'endColumnIndex': 9}}]}]}
        apply_upsert(ws, values())
        unowned = {'G5', 'H5', 'I5'}
        self.assertFalse(unowned.intersection(edit['range'] for edit in ws.writes))
        self.assertFalse(unowned.intersection(fmt['range'] for fmt in ws.formats))
        self.assertTrue(all(':' not in fmt['range'] for fmt in ws.formats))

    def test_owned_notes_control_rejects_before_any_mutation(self):
        ws = Worksheet()
        ws.table[2].append('Notes')
        ws.fetch_sheet_metadata = lambda **kw: {'sheets': [{
            'properties': {'sheetId': ws.id},
            'protectedRanges': [{'range': {'sheetId': ws.id, 'startRowIndex': 4,
                'endRowIndex': 5, 'startColumnIndex': 6, 'endColumnIndex': 7}}]}]}
        before = ws.get_all_values()
        with self.assertRaisesRegex(ValueError, 'protected range'):
            apply_upsert(ws, dict(values(), Notes='New note'))
        self.assertEqual(ws.writes, [])
        self.assertEqual(ws.formats, [])
        self.assertEqual(ws.table, before)

    def test_no_live_activation_by_default(self):
        with self.assertRaises(PermissionError):
            upload_run('anything')

    def test_append_readback_precision_and_idempotent_upsert(self):
        ws = Worksheet()
        historic = ws.table[3][:]
        receipt = apply_upsert(ws, values())
        self.assertEqual(receipt['row'], 5)
        self.assertEqual(ws.table[3], historic)
        self.assertEqual(ws.table[4][5], .987654321)
        self.assertEqual(ws.table[4][4], 'READBACK_VERIFIED')
        self.assertEqual(apply_upsert(ws, values())['row'], 5)
        self.assertEqual(len(ws.table), 5)
        numeric = {f['range']: f['format']['numberFormat']['pattern']
                   for f in ws.formats if 'numberFormat' in f['format']}
        self.assertEqual(numeric['F5'], '0.0000')
        self.assertEqual(numeric['G5'], '0')

    def test_readback_failure_does_not_publish_verified(self):
        ws = Worksheet()
        ws.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback mismatch'):
            apply_upsert(ws, values())
        self.assertEqual(ws.table[4][4], 'READBACK_PENDING')

    def test_formula_cell_is_not_overwritten(self):
        ws = Worksheet()
        ws.fetch_sheet_metadata = lambda **kw: {'sheets': [{'properties': {'sheetId': ws.id},
            'data': [{'startRow': 4, 'startColumn': 5, 'rowData': [{'values': [
                {'userEnteredValue': {'formulaValue': '=1+1'}}]}]}]}]}
        with self.assertRaisesRegex(ValueError, 'formula'):
            apply_upsert(ws, values())
        self.assertEqual(ws.writes, [])

    def test_local_sheet_helpers_preserve_structural_guards(self):
        base = {'properties': {'sheetId': 77}}
        for name, value in [('dataValidation', {'condition': {'type': 'ONE_OF_LIST'}}),
                            ('chipRuns', [{'startIndex': 0}]),
                            ('dataSourceFormula', {'formula': '=DATA()'}),
                            ('dataSourceTable', {'dataSourceId': 'source'})]:
            sheet = dict(base, data=[{'startRow': 4, 'startColumn': 5,
                                     'rowData': [{'values': [{name: value}]}]}])
            self.assertEqual(controlled_reason(dict(sheet=sheet, named_ranges={}), 5, 6), name)
        sheet = dict(base, protectedRanges=[{'range': {'sheetId': 77},
            'unprotectedRanges': [{'startRowIndex': 4, 'endRowIndex': 5,
                                   'startColumnIndex': 5, 'endColumnIndex': 6}]}])
        controls = dict(sheet=sheet, named_ranges={})
        self.assertIsNone(controlled_reason(controls, 5, 6))
        self.assertEqual(controlled_reason(controls, 5, 7), 'protected range')
        sheet = dict(base, protectedRanges=[{'namedRangeId': 'unknown'}])
        with self.assertRaises(ValueError):
            controlled_reason(dict(sheet=sheet, named_ranges={}), 5, 6)
        sheet = dict(base, tables=[{'range': {'startRowIndex': 4, 'startColumnIndex': 5},
                                    'columnProperties': [{'columnIndex': 0, 'columnType': 'TEXT'}]}])
        self.assertEqual(controlled_reason(dict(sheet=sheet, named_ranges={}), 5, 6), 'typed table column')
        sheet = dict(base, merges=[{'startRowIndex': 4, 'startColumnIndex': 5}])
        self.assertEqual(controlled_reason(dict(sheet=sheet, named_ranges={}), 5, 6), 'merged cell')
        ws = Worksheet()
        ws.fetch_sheet_metadata = lambda **kw: {'sheets': []}
        with self.assertRaises(ValueError):
            fetch_controls(ws, [3, 5])

    def test_qgbase_alias_and_duplicate_guard(self):
        ws = Worksheet()
        ws.table[3][0] = values()['Run']
        with self.assertRaisesRegex(ValueError, 'crosswalk'):
            plan_upsert(ws.table, ws.table[2], values())
        crosswalk = {values()['Run']: dict(row=4, sensor='QB', campaign='campaign', run_id=values()['Run'])}
        self.assertEqual(plan_upsert(ws.table, ws.table[2], values(), alias_crosswalk=crosswalk)['row'], 4)
        with self.assertRaisesRegex(ValueError, 'Duplicate Sheet header'):
            plan_upsert(ws.table, ['Run', 'Run'], values())

    def test_no_eligible_empty_and_only_metric_display_format(self):
        row = selection_values('Target', dict(selection_id='TARGET', target_status='no_eligible', official_complete=True))
        self.assertEqual(row['Target Q4↑'], '')
        self.assertEqual(row['Target JQM↑'], '')
        self.assertEqual(row['Target checkpoint SHA256'], '')
        self.assertFalse(any('Q8' in key for key in row))
        formats = metric_formats({'Q4↑': 1, 'Target RMSE↓': 2, 'QG40 q_ref': 3, 'Target step': 4}, 7)
        self.assertEqual([f['range'] for f in formats], ['A7', 'B7', 'D7'])
        self.assertEqual(formats[-1]['format']['numberFormat']['pattern'], '0')

    def test_missing_or_wv3_threshold_is_rejected(self):
        selected = dict(n_eligible=0, joint_pass=False, strong_joint_pass=False, target_status='no_eligible')
        target = dict(selected, threshold_comparison='>', hqnr_threshold=.964, ergas_goal=.552,
                      ergas_strong_goal=.522, n_evaluated=50, selector_order=['ergas', '-scc', '-psnr', 'step'])
        validate_target_metadata(target, selected, 'GF2')
        target['hqnr_threshold'] = .9585
        with self.assertRaisesRegex(ValueError, 'strict sensor threshold'):
            validate_target_metadata(target, selected, 'GF2')
        target.pop('hqnr_threshold')
        with self.assertRaises(ValueError):
            validate_target_metadata(target, selected, 'GF2')


if __name__ == '__main__':
    unittest.main()
