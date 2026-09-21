import re
import unittest
from unittest.mock import patch

from l100.common import object_sha
from l100.plan import CAMPAIGN_ID
from l100.upload import apply_upsert, plan_upsert, upload_run


class Worksheet:
    title, id, col_count, row_count = 'GF2-s4', 77, 12, 50
    def __init__(self):
        self.table = [[], [], ['Run', 'L100 sensor', 'L100 campaign', 'L100 run id', 'L100 upload status', 'Q4↑'],
                      ['benchmark', '', '', '', '', '.9']]
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
            letters, row = re.fullmatch(r'([A-Z]+)([0-9]+)', edit['range']).groups()
            row, col = int(row), 0
            for char in letters:
                col = col * 26 + ord(char) - 64
            while len(self.table) < row:
                self.table.append([])
            while len(self.table[row - 1]) < col:
                self.table[row - 1].append('')
            self.table[row - 1][col - 1] = edit['values'][0][0]
    def batch_format(self, formats):
        self.formats.extend(formats)


def values():
    return {'Run': 'L100I1_GF2_fixture_v2', 'L100 sensor': 'GF2', 'L100 campaign': CAMPAIGN_ID,
        'L100 run id': 'L100I1_GF2_fixture_v2', 'L100 upload status': 'READBACK_PENDING',
        'Q4↑': .987654321, 'EXACT_FINAL step': 100000, 'MID50_OF100 step': 50000,
        'EXACT_FINAL JQM↑': .9123456789, 'Notes': 'Official LOCAL-T note.'}


class UploadTests(unittest.TestCase):
    def test_raw_precision_display_only_and_historic_row_preserved(self):
        ws = Worksheet()
        old = ws.table[3][:]
        receipt = apply_upsert(ws, values())
        self.assertEqual(ws.table[3], old)
        self.assertEqual(ws.table[4][5], .987654321)
        self.assertEqual(receipt['row'], 5)
        self.assertTrue(receipt['readback_verified'])
        self.assertTrue(all(':' not in entry['range'] for entry in ws.formats))
        self.assertTrue(any(entry['format'].get('numberFormat', {}).get('pattern') == '0.0000'
                            for entry in ws.formats))
        self.assertNotIn('Exact50K step', ws.table[2])

    def test_manual_notes_preserved_idempotently_and_receipt_hash_effective_payload(self):
        ws = Worksheet()
        incoming = values()
        apply_upsert(ws, incoming)
        pos = ws.table[2].index('Notes')
        ws.table[4][pos] += '\nHuman observation: retain.'
        expected = ws.table[4][pos]
        plan = plan_upsert(ws.table, ws.table[2], incoming)
        receipt = apply_upsert(ws, incoming)
        self.assertEqual(ws.table[4][pos], expected)
        self.assertEqual(receipt['payload_sha256'], object_sha(plan['values']))
        self.assertEqual(apply_upsert(ws, incoming)['payload_sha256'], receipt['payload_sha256'])
        self.assertEqual(incoming, values())

    def test_alias_duplicate_and_campaign_collision_rejected(self):
        for row in ([values()['Run'], '', '', ''],
                    [values()['Run'], 'GF2', 'another_campaign', values()['Run']]):
            ws = Worksheet()
            ws.table.append(row)
            with self.assertRaises(ValueError):
                plan_upsert(ws.table, ws.table[2], values())
        ws = Worksheet()
        apply_upsert(ws, values())
        ws.table.append(ws.table[4][:])
        with self.assertRaises(ValueError):
            plan_upsert(ws.table, ws.table[2], values())

    def test_owned_formula_rejects_before_any_write(self):
        ws = Worksheet()
        ws.fetch_sheet_metadata = lambda **kw: {'sheets': [{'properties': {'sheetId': ws.id},
            'data': [{'startRow': 4, 'startColumn': 5,
                      'rowData': [{'values': [{'userEnteredValue': {'formulaValue': '=1'}}]}]}]}]}
        with self.assertRaisesRegex(ValueError, 'formula'):
            apply_upsert(ws, values())
        self.assertFalse(ws.writes)

    def test_optimistic_notes_change_rejected_before_write(self):
        ws = Worksheet()
        original = ws.get_all_values
        calls = []
        def changed():
            calls.append(1)
            table = original()
            if len(calls) == 2:
                table[3].append('new human text')
            return table
        ws.get_all_values = changed
        with self.assertRaisesRegex(ValueError, 'changed during'):
            apply_upsert(ws, values())
        self.assertFalse(ws.writes)

    def test_readback_failure_not_marked_successful(self):
        ws = Worksheet()
        ws.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback'):
            apply_upsert(ws, values())
        self.assertEqual(ws.table[4][4], 'READBACK_PENDING')

    def test_upload_requires_explicit_activation_before_read(self):
        with patch('l100.upload.row_values') as rows:
            with self.assertRaises(PermissionError):
                upload_run('fixture')
            rows.assert_not_called()


if __name__ == '__main__':
    unittest.main()
