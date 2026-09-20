import re
import unittest
from unittest.mock import Mock

from g20.upload import (apply_upsert, metric_formats, plan_upsert, selection_values,
                        upload_run, validate_target_metadata, open_campaign_worksheet)
from qg40.sheet_helpers import controlled_reason, fetch_controls


def parse_cell(cell):
    letters, row = re.fullmatch(r'([A-Z]+)([0-9]+)', cell).groups()
    col = 0
    for char in letters:
        col = col * 26 + ord(char) - 64
    return int(row), col


class Worksheet:
    title, id, col_count, row_count = 'GF2-s1', 77, 12, 50
    def __init__(self):
        self.table = [[], [], ['Run', 'G20 sensor', 'G20 campaign', 'G20 run id', 'G20 upload status', 'Q4↑'], ['benchmark', '', '', '', '', '.9']]
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
    return {'Run': 'G20_GF2_test', 'G20 sensor': 'GF2', 'G20 campaign': 'campaign',
            'G20 run id': 'G20_GF2_test', 'G20 upload status': 'READBACK_PENDING',
            'Q4↑': .987654321, 'Target step': 2020, 'Target JQM↑': '', 'G20 Student seed': 82001}


class UploadTests(unittest.TestCase):
    def test_only_explicit_gf2_s4_creation_and_no_other_sensor_rename(self):
        import gspread
        book = Mock()
        book.worksheet.side_effect = gspread.WorksheetNotFound('missing')
        fresh = Worksheet()
        fresh.title, fresh.table = 'GF2-s4', []
        book.add_worksheet.return_value = fresh
        with self.assertRaises(PermissionError):
            open_campaign_worksheet(book, 's4')
        book.worksheet.assert_not_called()
        with self.assertRaises(ValueError):
            open_campaign_worksheet(book, 's1', activated=True)
        book.add_worksheet.assert_not_called()
        self.assertIs(open_campaign_worksheet(book, 's4', activated=True), fresh)
        book.add_worksheet.assert_called_once_with(title='GF2-s4', rows=1000, cols=26)
        self.assertEqual(fresh.row_values(3), ['Run'])

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
        self.assertTrue(all(f['format']['numberFormat']['pattern'] == '0.0000' for f in ws.formats))
        self.assertNotIn('G5', [f['range'] for f in ws.formats])  # selected step stays integer

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
        crosswalk = {values()['Run']: dict(row=4, sensor='GF2', campaign='campaign', run_id=values()['Run'])}
        self.assertEqual(plan_upsert(ws.table, ws.table[2], values(), alias_crosswalk=crosswalk)['row'], 4)
        with self.assertRaisesRegex(ValueError, 'Duplicate Sheet header'):
            plan_upsert(ws.table, ['Run', 'Run'], values())

    def test_no_eligible_empty_and_only_metric_display_format(self):
        row = selection_values('Target', dict(selection_id='TARGET', target_status='no_eligible', official_complete=True, test_aware=True, primary=False))
        self.assertEqual(row['Target Q4↑'], '')
        self.assertEqual(row['Target JQM↑'], '')
        self.assertEqual(row['Target checkpoint SHA256'], '')
        self.assertFalse(any('Q8' in key for key in row))
        formats = metric_formats({'Q4↑': 1, 'Target RMSE↓': 2, 'G20 q_ref': 3, 'Target step': 4}, 7)
        self.assertEqual([f['range'] for f in formats], ['A7', 'B7'])

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
