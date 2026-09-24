"""Mock Sheet acceptance: no network/credential access or historical edits."""
import copy
import re
import tempfile
import unittest
from unittest import mock

from maina_hqnr.common import object_sha
from maina_hqnr.test_results import complete_summary
from maina_hqnr.upload import (COLLECTION_TAB, HEADER, SECTION_TITLE, TABS,
    apply_collection_upsert, apply_upsert, collection_formula, flatten_summary, flush_outbox)


class Sheet:
    def __init__(self, title, rows=None):
        self.title = title; self.id = 42; self.row_count = 100; self.col_count = len(HEADER)
        self.rows = copy.deepcopy(rows or []); self.formats = []; self.inputs = []

    def row_values(self, number, **kwargs):
        return copy.deepcopy(self.rows[number-1]) if number <= len(self.rows) else []

    def get_all_values(self, **kwargs):
        return copy.deepcopy(self.rows)

    def add_cols(self, count): self.col_count += count
    def add_rows(self, count): self.row_count += count

    def update(self, range_name, values, value_input_option):
        self.inputs.append(value_input_option)
        match = re.fullmatch(r'([A-Z]+)(\d+)', range_name); column = 0
        for letter in match[1]: column = column*26+ord(letter)-64
        row = int(match[2])
        while len(self.rows) < row-1+len(values): self.rows.append([])
        for offset, values_row in enumerate(values):
            target = self.rows[row-1+offset]
            while len(target) < column-1+len(values_row): target.append('')
            target[column-1:column-1+len(values_row)] = copy.deepcopy(values_row)

    def insert_rows(self, values, row, value_input_option):
        self.inputs.append(value_input_option); self.rows[row-1:row-1] = copy.deepcopy(values)

    def batch_format(self, formats): self.formats.extend(copy.deepcopy(formats))


class UploadAcceptance(unittest.TestCase):
    def test_exact_full_precision_values_with_four_decimal_display(self):
        payload = flatten_summary(complete_summary())[0]
        sheet = Sheet(TABS['s4']); receipt = apply_upsert(sheet, payload)
        self.assertTrue(receipt['readback_verified'])
        self.assertEqual(sheet.rows[1][HEADER.index('hqnr')], payload['hqnr'])
        self.assertTrue(all(mode == 'RAW' for mode in sheet.inputs))
        metric_columns = {HEADER.index(k)+1 for k in ('hqnr', 'rmse', 'cc', 'jqm')}
        self.assertGreaterEqual(sum(fmt['format']['numberFormat']['pattern'] == '0.0000' for fmt in sheet.formats), len(metric_columns))
        apply_upsert(sheet, payload); self.assertEqual(len(sheet.rows), 2)

    def test_readback_detects_metric_rounding(self):
        class RoundedSheet(Sheet):
            def row_values(self, number, **kwargs):
                row = super().row_values(number, **kwargs)
                if number > 1 and len(row) > HEADER.index('hqnr'):
                    row[HEADER.index('hqnr')] = round(row[HEADER.index('hqnr')], 4)
                return row
        payload = flatten_summary(complete_summary())[0]
        payload['hqnr'] = .9612164300901325
        with self.assertRaises(ValueError):
            apply_upsert(RoundedSheet(TABS['s4']), payload)

    def test_historical_tabs_and_headers_are_never_overwritten(self):
        payload = flatten_summary(complete_summary())[0]
        for sheet in (Sheet('WV3-s4', [['historic']]), Sheet(TABS['s4'], [['different header']])):
            before = copy.deepcopy(sheet.rows)
            with self.assertRaises(ValueError): apply_upsert(sheet, payload)
            self.assertEqual(sheet.rows, before)

    def test_collection_projection_preserves_history_and_sorted_formula(self):
        history = [['G23 original values'], ['User representative table'], ['main', .99]]
        sheet = ProjectionSheet(COLLECTION_TAB, history)
        base = complete_summary('s4', 1)
        variant = complete_summary('s4', 1, 'AL05')
        sheet.payloads = [flatten_summary(variant, base)[0], flatten_summary(base)[0]]
        apply_collection_upsert(sheet, flatten_summary(variant, base)[0])
        apply_collection_upsert(sheet, flatten_summary(base)[0])
        self.assertEqual(sheet.rows[:len(history)], history)
        self.assertIn('order by Col'+str(len(HEADER)+1), collection_formula())
        self.assertIn(TABS['s4'], collection_formula()); self.assertIn(TABS['s5'], collection_formula())
        self.assertIn("='HQNR_MAX50'", collection_formula())
        self.assertEqual(sum(mode == 'USER_ENTERED' for mode in sheet.inputs), 1)
        before = copy.deepcopy(sheet.rows)
        apply_collection_upsert(sheet, flatten_summary(base)[1])
        self.assertEqual(sheet.rows, before)

    def test_nonowner_never_mutates_shared_collection(self):
        payload = flatten_summary(complete_summary('s5', 0))[0]
        sheet = ProjectionSheet(COLLECTION_TAB, [['historical']])
        with self.assertRaises(ValueError): apply_collection_upsert(sheet, payload)
        self.assertEqual(sheet.rows, [['historical']]); self.assertEqual(sheet.inputs, [])
        owner = flatten_summary(complete_summary('s4', 1))[0]
        sheet.payloads = [owner, payload]
        apply_collection_upsert(sheet, owner)
        before = copy.deepcopy((sheet.rows, sheet.inputs, sheet.formats))
        self.assertTrue(apply_collection_upsert(sheet, payload)['readback_verified'])
        self.assertEqual((sheet.rows, sheet.inputs, sheet.formats), before)

    def test_delayed_projection_keeps_upload_pending(self):
        sheet = ProjectionSheet(COLLECTION_TAB, [['historical']])
        with self.assertRaisesRegex(ValueError, 'READBACK_PENDING'):
            apply_collection_upsert(sheet, flatten_summary(complete_summary())[0])
        self.assertEqual(sheet.rows[0], ['historical'])

    def test_failure_row_has_no_fabricated_metrics_or_profile(self):
        summary = complete_summary(); summary.update(complete=False, status='PENDING_EVAL_NOT_COMPARABLE', selections={})
        row = flatten_summary(summary)[0]
        self.assertEqual(row['hqnr'], ''); self.assertEqual(row['rmse'], '')
        self.assertEqual(row['Inference(s)'], ''); self.assertEqual(row['FLOPs(G)'], '')
        self.assertNotIn('\n', row['Methods']); self.assertNotIn(str(summary['case']['seed']), row['Methods'])

    def test_explicit_activation_required_before_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch('maina_hqnr.upload._worksheets', side_effect=AssertionError('no API')):
                with self.assertRaises(PermissionError): flush_outbox(root, 's4')

    def test_methods_real_step_and_metadata_test_aware(self):
        payload = flatten_summary(complete_summary())[0]
        self.assertTrue(payload['test_aware']); self.assertFalse(payload['independent_test'])
        self.assertEqual(payload['selected_step'], 1010); self.assertIn('HQNR@1010', payload['Methods'])
        self.assertEqual(payload['teacher_alias'], 'F1')


class ProjectionSheet(Sheet):
    """Google recalculation is mocked; formula bytes/ownership are tested separately."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.payloads = []

    def get_all_values(self, **kwargs):
        rows = super().get_all_values(**kwargs)
        for index, row in enumerate(rows):
            if row and row[0] == collection_formula():
                projected = [[dict(payload, readback_status='READBACK_VERIFIED')[key] for key in HEADER]
                             for payload in self.payloads]
                return rows[:index]+projected
        return rows

    def row_values(self, number, **kwargs):
        if kwargs.get('value_render_option') == 'FORMULA':
            return super().row_values(number, **kwargs)
        rows = self.get_all_values(**kwargs)
        return rows[number-1] if number <= len(rows) else []

    def insert_rows(self, *args, **kwargs):
        raise AssertionError('Shared collection must never insert/update data rows')


if __name__ == '__main__':
    unittest.main()
