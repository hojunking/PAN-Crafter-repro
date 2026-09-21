"""Local-only structural repair tests, including protected controls and races."""
import copy
from pathlib import Path
import tempfile
import unittest

from reporting_extra.sensor_layout import COMMON_HEADERS
from reporting_extra.sensor_repair_layout import repair_layout


def literal(value):
    return {'userEnteredValue': {'stringValue' if isinstance(value, str) else 'numberValue': value}}


class FakeSheet:
    id = 42
    title = 'GF2-s4'

    def __init__(self):
        self.spreadsheet = self
        self.rows = [[{} for _ in range(40)] for _ in range(7)]
        headers = ['', 'Run', '캠페인', 'G20 source SHA256', 'G20 run id', *COMMON_HEADERS[3:]]
        self.rows[0][0] = literal('GF2 s4 historical results')
        self.rows[1][3] = literal('RR')
        self.rows[1][11] = literal('FR·paper mat20')
        self.rows[1][16] = literal('Cost')
        for col, value in enumerate(headers):
            self.rows[2][col] = literal(value)
            self.rows[3][col] = literal('row4-' + str(col))
        self.rows[3][headers.index('Date')] = dict(literal('2026-08-09'), userEnteredFormat={'numberFormat': {'type': 'TEXT'}})
        self.rows[3][headers.index('Notes')] = literal('User note\nACCEPTED_WITH_PARITY_EXCEPTION exact SHA unchanged')
        self.rows[3][headers.index('HQNR↑')] = literal(.956123456789)
        self.rows[4][35] = literal('Unnamed column user data')
        self.rows[4][35]['note'] = 'Keep attached note'
        self.properties = dict(sheetId=self.id, title=self.title, gridProperties=dict(rowCount=7, columnCount=40))
        self.merges = [dict(sheetId=self.id, startRowIndex=1, endRowIndex=2,
                            startColumnIndex=a, endColumnIndex=b) for a, b in ((3, 11), (11, 16), (16, 18))]
        self.extra = {}
        self.calls = []
        self.reads = 0
        self.race = False
        self.corrupt = False

    @property
    def row_count(self):
        return self.properties['gridProperties']['rowCount']

    @property
    def col_count(self):
        return self.properties['gridProperties']['columnCount']

    def fetch_sheet_metadata(self, **kwargs):
        self.reads += 1
        if self.race and self.reads == 2:
            self.rows[3][1] = literal('User changed Run while planning')
        return {'sheets': [dict(properties=copy.deepcopy(self.properties), merges=copy.deepcopy(self.merges),
            data=[dict(startRow=0, startColumn=0, rowData=[{'values': copy.deepcopy(row)} for row in self.rows])], **copy.deepcopy(self.extra))]}

    def get_worksheet_by_id(self, sid):
        assert sid == self.id
        return self

    def batch_update(self, body):
        self.calls.append(copy.deepcopy(body))
        for item in body['requests']:
            if 'unmergeCells' in item:
                self.merges.remove(item['unmergeCells']['range'])
            elif 'appendDimension' in item:
                length = item['appendDimension']['length']
                for row in self.rows:
                    row.extend({} for _ in range(length))
                self.properties['gridProperties']['columnCount'] += length
            elif 'insertDimension' in item:
                col = item['insertDimension']['range']['startIndex']
                for row in self.rows:
                    row.insert(col, {})
                self.properties['gridProperties']['columnCount'] += 1
            elif 'moveDimension' in item:
                request = item['moveDimension']
                start, dest = request['source']['startIndex'], request['destinationIndex']
                for row in self.rows:
                    value = row.pop(start)
                    row.insert(dest if dest < start else dest - 1, value)
            elif 'updateCells' in item:
                request = item['updateCells']
                start = request['start']
                for ri, source in enumerate(request['rows'], start['rowIndex']):
                    for ci, cell in enumerate(source['values'], start['columnIndex']):
                        self.rows[ri][ci]['userEnteredValue'] = copy.deepcopy(cell['userEnteredValue'])
            elif 'repeatCell' in item:
                request = item['repeatCell']
                region = request['range']
                if request['fields'] == 'userEnteredValue':
                    for ri in range(region['startRowIndex'], region['endRowIndex']):
                        for ci in range(region['startColumnIndex'], region['endColumnIndex']):
                            self.rows[ri][ci].pop('userEnteredValue', None)
            elif 'updateSheetProperties' in item:
                self.properties['gridProperties'].update(item['updateSheetProperties']['properties']['gridProperties'])
            elif 'updateDimensionProperties' not in item:
                raise AssertionError('Unknown or destructive request')
        if self.corrupt:
            self.rows[3][13] = literal(.99)


class LayoutRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = FakeSheet()

    def test_dry_run_is_read_only_and_native_plan_preserves_custom_columns(self):
        before = copy.deepcopy(self.ws.rows)
        result = repair_layout(self.ws, self.root, 's4')
        self.assertEqual(result['final_headers'][:29], list(COMMON_HEADERS))
        self.assertEqual(result['final_headers'][29:31], ['G20 source SHA256', 'G20 run id'])
        self.assertEqual(self.ws.rows, before)
        self.assertFalse(self.ws.calls)
        self.assertFalse(list(self.root.iterdir()))

    def test_apply_preserves_metrics_notes_date_and_unlabeled_cells(self):
        old_date = copy.deepcopy(self.ws.rows[3][23])
        result = repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertTrue(result['readback_verified'])
        self.assertEqual(self.ws.rows[3][21], old_date)
        self.assertEqual(self.ws.rows[3][13]['userEnteredValue']['numberValue'], .956123456789)
        self.assertIn('ACCEPTED_WITH_PARITY_EXCEPTION', self.ws.rows[3][22]['userEnteredValue']['stringValue'])
        self.assertEqual(self.ws.rows[4][35]['note'], 'Keep attached note')
        self.assertTrue(Path(result['snapshot']).is_file())
        self.assertFalse(self.ws.merges)
        self.assertEqual(len(self.ws.calls), 1)
        second = repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertTrue(second['unchanged'])
        self.assertEqual(len(self.ws.calls), 1)

    def test_wrong_server_never_reads_or_writes(self):
        with self.assertRaisesRegex(ValueError, 'selected server'):
            repair_layout(self.ws, self.root, 's1', apply=True)
        self.assertEqual(self.ws.reads, 0)

    def test_controls_anywhere_in_tab_block_before_write(self):
        for control in ('protectedRanges', 'tables'):
            self.ws.extra = {control: [{'range': {'sheetId': self.ws.id}}]}
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, 'protected ranges or native tables'):
                repair_layout(self.ws, self.root, 's4', apply=True)
        self.ws.extra = {}
        for key, value in [('userEnteredValue', {'formulaValue': '=INDIRECT("B4")'}),
                           ('dataValidation', {'strict': True}), ('chipRuns', [{'startIndex': 0}]),
                           ('dataSourceFormula', {'dataSourceId': 'source'})]:
            self.ws.rows[6][39] = {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)

    def test_unknown_merge_and_unknown_row2_content_refuse(self):
        self.ws.merges.append(dict(sheetId=42, startRowIndex=0, endRowIndex=1, startColumnIndex=0, endColumnIndex=5))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.ws.merges.pop()
        self.ws.rows[1][35] = literal('User comment, not an owned group')
        with self.assertRaisesRegex(ValueError, 'Unowned'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)

    def test_concurrent_user_edit_is_detected_before_mutation(self):
        self.ws.race = True
        with self.assertRaisesRegex(ValueError, 'changed after layout planning'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)

    def test_readback_corruption_keeps_before_snapshot_not_success_receipt(self):
        self.ws.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback changed historical'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertTrue(list(self.root.rglob('*.before.json')))
        self.assertFalse(list(self.root.rglob('*.receipt.json')))

    def test_duplicate_headers_fail_before_mutation(self):
        self.ws.rows[2][35] = literal('Run')
        with self.assertRaisesRegex(ValueError, 'unique'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)

    def test_unknown_text_at_known_merge_anchor_is_not_owned(self):
        self.ws.rows[1][3] = literal('User-authored research claim')
        with self.assertRaisesRegex(ValueError, 'Unowned'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)

    def test_rich_header_is_rejected_before_formatting(self):
        self.ws.rows[2][1]['textFormatRuns'] = [{'startIndex': 0, 'format': {'bold': True}}]
        with self.assertRaisesRegex(ValueError, 'Rich-text header'):
            repair_layout(self.ws, self.root, 's4', apply=True)
        self.assertFalse(self.ws.calls)


if __name__ == '__main__':
    unittest.main()
