"""Offline Sheet integrity tests; no credentials, network or model inference."""
import copy
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fh12.common import atomic_json, read_json
from reporting_extra import upload as U


class FakeSheet:
    def __init__(self, headers, row, gid):
        self.id = gid
        self.col_count = len(headers)
        self.cells = {2: ['original groups'] * len(headers), 3: headers.copy(), 4: row.copy()}
        self.writes = []
        self.formats = []
        self.corrupt = False
        self.title = 'fixture sheet'
        self.controls = {'properties': {'sheetId': gid}}
        self.spreadsheet = types.SimpleNamespace(fetch_sheet_metadata=lambda **kwargs: {'sheets': [self.controls]})

    def row_values(self, row, **kwargs):
        result = self.cells.get(row, []).copy()
        if self.corrupt and self.writes and row == 4 and kwargs:
            labels = U.label_map(self.cells[3])
            result[labels['RMSE↓'] - 1] = 'unexpected corruption'
        return result

    def get_all_values(self, **kwargs):
        return [self.cells.get(row, []).copy() for row in range(1, max(self.cells) + 1)]

    def add_cols(self, count):
        self.col_count += count

    def batch_update(self, edits, **kwargs):
        assert kwargs.get('value_input_option') == 'RAW'
        self.writes.extend(copy.deepcopy(edits))
        for edit in edits:
            match = re.fullmatch(r'([A-Z]+)(\d+)', edit['range'])
            col = 0
            for letter in match[1]:
                col = col * 26 + ord(letter) - 64
            row = self.cells.setdefault(int(match[2]), [])
            row.extend([''] * max(0, col - len(row)))
            row[col - 1] = edit['values'][0][0]

    def batch_format(self, formats):
        self.formats.extend(copy.deepcopy(formats))


def report_fixture(campaign='FH20R1', server='s1'):
    result = dict(schema=U.SCHEMA, run_id='fixture-run', campaign_id=f'WV3_{campaign}_FIXTURE_v1',
                  server_id=server, config_sha256='config-sha', data_sha256='data-sha',
                  jqm_variant='SRF-substitute (NNLS-normalized); not SIPSA-equivalent', inference_devices=['cpu'],
                  evaluator_identity={'source': 'fixture-source'}, selections={})
    for index, (name, (_, selection_id)) in enumerate(U.SELECTIONS.items()):
        step = 50000 if name == 'exact50k' else 1010 * (index + 1)
        result['selections'][name] = dict(status='complete', selection_id=selection_id, step=step,
                                         checkpoint_sha256=f'checkpoint-{step}', official_report_sha256=f'official-{name}',
                                         metrics=dict(rmse=19.12345678901234 + index, cc=.98234567890123 + index * .001,
                                                      jqm=.93234567890123 + index * .001))
    return result


def sheet_fixture(report):
    campaign = U.campaign_prefix(report)
    headers = ['Human field', 'Run', '캠페인', 'RMSE↓', 'CC↑', 'JQM↑', 'Notes', 'NOA', 'HQNR(V64)↑',
               'ERGAS↓', 'HQNR↑', 'FH20R1 Student seed', 'FH20R1 LR schedule', 'FH12 tau_R',
               'RAW_MAX ERGAS↓', 'RR_VAL_SELECTED val ERGAS', campaign + ' campaign', campaign + ' run id']
    values = {'Human field': '=42', 'Run': report['run_id'], '캠페인': campaign,
              'Notes': 'must preserve human notes', 'NOA': 'preserve NOA', 'HQNR(V64)↑': .954987,
              'ERGAS↓': 2.000000123456789, 'HQNR↑': .9599991234567,
              'FH20R1 Student seed': 73501, 'FH20R1 LR schedule': 'BASE', 'FH12 tau_R': .02000000001,
              'RAW_MAX ERGAS↓': 2.000000123456789, 'RR_VAL_SELECTED val ERGAS': 2.0300098765,
              campaign + ' campaign': report['campaign_id'], campaign + ' run id': report['run_id']}
    for name, (prefix, _) in U.SELECTIONS.items():
        item = report['selections'][name]
        for suffix, key in ((' step', 'step'), (' checkpoint SHA256', 'checkpoint_sha256')):
            label = prefix + suffix
            headers.append(label)
            values[label] = item[key] if item[key] is not None else ''
    return FakeSheet(headers, [values.get(label, '') for label in headers], U.GIDS[report['server_id']])


class UploadTests(unittest.TestCase):
    def test_exact_precision_and_original_cells_unchanged(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        before = dict(zip(sheet.cells[3], sheet.cells[4]))
        receipt = U.write_supplement(sheet, report, 'report-sha')
        after = dict(zip(sheet.cells[3], sheet.cells[4]))
        for label, value in before.items():
            if label not in U.EXTRA_LABELS:
                self.assertEqual(after[label], value, label)
        self.assertEqual(after['RMSE↓'], 19.12345678901234)
        self.assertEqual(after['RAW_MAX JQM↑'], report['selections']['raw_max']['metrics']['jqm'])
        self.assertEqual(after['Exact50K CC↑'], report['selections']['exact50k']['metrics']['cc'])
        self.assertEqual(after['Extra metrics status'], 'VERIFIED')
        self.assertIn('NNLS', after['Extra metrics JQM variant'])
        self.assertEqual(after['Extra metrics inference devices'], 'cpu')
        self.assertTrue(receipt['readback_verified'])
        self.assertEqual(receipt['report_sha256'], 'report-sha')

    def test_repeated_upload_is_idempotent(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        U.write_supplement(sheet, report, 'report-sha')
        before = copy.deepcopy(sheet.cells)
        columns = sheet.col_count
        U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.cells, before)
        self.assertEqual(sheet.col_count, columns)

    def test_all_servers_both_campaigns_dynamic_layout(self):
        for server in U.GIDS:
            for campaign in ('FH12', 'FH20R1'):
                with self.subTest(server=server, campaign=campaign):
                    report = report_fixture(campaign, server)
                    sheet = sheet_fixture(report)
                    # A shifted layout is intentional (including s5).
                    sheet.cells[3].insert(2, 'Another human column')
                    sheet.cells[4].insert(2, 'preserve inserted cell')
                    receipt = U.write_supplement(sheet, report, 'report-sha')
                    self.assertEqual(receipt['server_id'], server)
                    self.assertEqual(sheet.cells[4][2], 'preserve inserted cell')

    def test_no_eligible_target_is_blank_not_zero(self):
        report = report_fixture()
        item = report['selections']['target_selection']
        item.update(status='no_eligible', step=None, checkpoint_sha256=None, metrics=None)
        sheet = sheet_fixture(report)
        U.write_supplement(sheet, report, 'report-sha')
        cells = dict(zip(sheet.cells[3], sheet.cells[4]))
        for label in U.EXTRA_LABELS:
            self.assertEqual(cells['Target ' + label], '')

    def test_format_only_numeric_quality_metrics(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        U.write_supplement(sheet, report, 'report-sha')
        labels = U.label_map(sheet.cells[3])
        formatted = {item['range'] for item in sheet.formats}
        for label in ('RMSE↓', 'JQM↑', 'ERGAS↓', 'RAW_MAX ERGAS↓', 'Exact50K CC↑', 'RR_VAL_SELECTED val ERGAS'):
            self.assertIn(f'{U.column(labels[label])}4:{U.column(labels[label])}4', formatted)
        for label in ('Human field', 'NOA', 'HQNR(V64)↑', 'FH20R1 Student seed', 'FH20R1 LR schedule',
                      'FH12 tau_R', 'RAW_MAX step', 'Exact50K checkpoint SHA256'):
            self.assertNotIn(f'{U.column(labels[label])}4:{U.column(labels[label])}4', formatted)
        self.assertTrue(all(item['format'] == {'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}}
                            for item in sheet.formats))

    def test_far_right_unheaded_human_data_is_preserved(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        previous_extent = len(sheet.cells[3]) + 5
        sheet.cells[5] = [''] * (previous_extent - 1) + ['far right human note']
        U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.cells[5][-1], 'far right human note')
        self.assertEqual(U.label_map(sheet.cells[3])['RAW_MAX RMSE↓'], previous_extent + 1)

    def test_missing_base_row_does_not_create_or_write(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        sheet.cells[4] = ['old history']
        with self.assertRaises(U.BaseRowMissing):
            U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.writes, [])

    def test_run_only_or_wrong_campaign_row_is_not_absent(self):
        for foreign_campaign in ('', 'WV3_FH20R1_OTHER_CAMPAIGN_v1'):
            report = report_fixture(); sheet = sheet_fixture(report)
            labels = U.label_map(sheet.cells[3])
            sheet.cells[4][labels['FH20R1 campaign'] - 1] = foreign_campaign
            with self.assertRaisesRegex(ValueError, 'Existing Run row lacks'):
                U.write_supplement(sheet, report, 'report-sha')
            self.assertEqual(sheet.writes, [])

    def test_missing_compound_key_headers_is_pending_not_absent(self):
        report = report_fixture(); sheet = sheet_fixture(report)
        sheet.cells[3][U.label_map(sheet.cells[3])['FH20R1 campaign'] - 1] = 'unrelated'
        with self.assertRaises(U.BaseUploadPending) as failure:
            U.write_supplement(sheet, report, 'report-sha')
        self.assertNotIsInstance(failure.exception, U.BaseRowMissing)
        self.assertEqual(sheet.writes, [])

    def test_duplicate_row_refuses_writes(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        sheet.cells[5] = sheet.cells[4].copy()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.writes, [])

    def test_wrong_gid_refuses_writes(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        sheet.id = U.GIDS['s2']
        with self.assertRaisesRegex(ValueError, 'GID'):
            U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.writes, [])

    def test_formula_validation_chips_and_datasource_refuse_writes(self):
        for payload in ({'userEnteredValue': {'formulaValue': '=1+1'}},
                        {'dataValidation': {'condition': {'type': 'ONE_OF_LIST'}}},
                        {'chipRuns': [{'startIndex': 0, 'chip': {}}]},
                        {'dataSourceFormula': {'dataSourceId': 'protected-source'}},
                        {'dataSourceTable': {'dataSourceId': 'protected-source'}}):
            report = report_fixture(); sheet = sheet_fixture(report)
            col = U.label_map(sheet.cells[3])['RMSE↓']
            sheet.controls['data'] = [{'startRow': 3, 'startColumn': col - 1,
                                       'rowData': [{'values': [payload]}]}]
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                U.write_supplement(sheet, report, 'report-sha')
            self.assertEqual(sheet.writes, [])

    def test_protected_merged_and_typed_cells_refuse_writes(self):
        for kind in ('protectedRanges', 'merges', 'tables'):
            report = report_fixture(); sheet = sheet_fixture(report)
            col = U.label_map(sheet.cells[3])['JQM↑']
            grid = dict(sheetId=sheet.id, startRowIndex=3, endRowIndex=4,
                        startColumnIndex=col - 1, endColumnIndex=col)
            if kind == 'protectedRanges':
                value = [{'range': grid}]
            elif kind == 'merges':
                value = [grid]
            else:
                value = [{'range': grid, 'columnProperties': [{'columnIndex': 0, 'columnType': 'DOUBLE'}]}]
            sheet.controls[kind] = value
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                U.write_supplement(sheet, report, 'report-sha')
            self.assertEqual(sheet.writes, [])

    def test_protected_core_metric_skips_format_and_preserves_value(self):
        report = report_fixture(); sheet = sheet_fixture(report)
        col = U.label_map(sheet.cells[3])['HQNR↑']
        sheet.controls['protectedRanges'] = [{'range': dict(sheetId=sheet.id, startRowIndex=3, endRowIndex=4,
                                                          startColumnIndex=col - 1, endColumnIndex=col)}]
        U.write_supplement(sheet, report, 'report-sha')
        self.assertNotIn(f'{U.column(col)}4:{U.column(col)}4', {v['range'] for v in sheet.formats})
        self.assertEqual(sheet.cells[4][col - 1], .9599991234567)

    def test_structural_read_failure_is_not_write_permission(self):
        report = report_fixture(); sheet = sheet_fixture(report)
        sheet.spreadsheet = types.SimpleNamespace(fetch_sheet_metadata=lambda **kwargs: {})
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            U.write_supplement(sheet, report, 'report-sha')
        self.assertEqual(sheet.writes, [])

    def test_wrong_step_or_sha_refuses_writes(self):
        for key in ('RAW_MAX step', 'Target checkpoint SHA256', 'Exact50K checkpoint SHA256'):
            report = report_fixture()
            sheet = sheet_fixture(report)
            sheet.cells[4][U.label_map(sheet.cells[3])[key] - 1] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'checkpoint differ'):
                U.write_supplement(sheet, report, 'report-sha')
            self.assertEqual(sheet.writes, [])

    def test_wrong_selector_or_nonfinite_refuses_writes(self):
        for mutation in ('selector', 'nonfinite', 'missing', 'exact'):
            report = report_fixture()
            sheet = sheet_fixture(report)
            if mutation == 'selector':
                report['selections']['raw_max']['selection_id'] = 'TARGET'
            elif mutation == 'nonfinite':
                report['selections']['raw_max']['metrics']['rmse'] = float('nan')
            elif mutation == 'missing':
                del report['selections']['raw_max']['official_report_sha256']
            else:
                report['selections']['exact50k']['step'] = 49490
            with self.assertRaises(ValueError):
                U.write_supplement(sheet, report, 'report-sha')
            self.assertEqual(sheet.writes, [])

    def test_corrupt_readback_never_certifies_success(self):
        report = report_fixture()
        sheet = sheet_fixture(report)
        sheet.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback mismatch'):
            U.write_supplement(sheet, report, 'report-sha')
        labels = U.label_map(sheet.cells[3])
        self.assertEqual(sheet.cells[4][labels['Extra metrics status'] - 1], 'READBACK_PENDING')

    def test_api_wrapper_saves_receipt_separately(self):
        report = report_fixture(server='s3')
        sheet = sheet_fixture(report)
        gu = types.SimpleNamespace(CRED='fixture.json', SHEET='fixture-book', ORIGIN_ROW=2)
        calls = []
        book = types.SimpleNamespace(worksheet=lambda name: calls.append(name) or sheet)
        api = types.SimpleNamespace(service_account=lambda **kwargs: types.SimpleNamespace(open=lambda name: book))
        with tempfile.TemporaryDirectory(prefix='extra-upload-test-') as tmp:
            root = Path(tmp)
            official = root / 'work_dir' / report['run_id'] / 'official/postrun_status.json'
            atomic_json(official, {'unchanged': True})
            with patch.object(U, '_validated_report', return_value=(report, 'report-sha')), \
                    patch.object(U, 'legacy_constants', return_value=gu), \
                    patch('tools.fh12_runner.detect_server', return_value='s3'), \
                    patch.dict('sys.modules', {'gspread': api}):
                receipt = U.upload_run(report['run_id'], root=root)
            path = root / 'work_dir' / report['run_id'] / 'supplemental_metrics/upload_receipt.json'
            self.assertEqual(read_json(path), receipt)
            self.assertEqual(read_json(official), {'unchanged': True})
        self.assertEqual(calls, ['WV3-s3(5090)'])

    def test_api_wrapper_rejects_wrong_local_server(self):
        report = report_fixture(server='s4')
        with tempfile.TemporaryDirectory(prefix='extra-upload-test-') as tmp:
            with patch.object(U, '_validated_report', return_value=(report, 'report-sha')), \
                    patch('tools.fh12_runner.detect_server', return_value='s1'):
                with self.assertRaisesRegex(ValueError, 'another server'):
                    U.upload_run(report['run_id'], root=Path(tmp))

    def test_proven_missing_base_row_recovers_once_for_both_campaigns(self):
        for campaign in ('FH12', 'FH20R1'):
            report = report_fixture(campaign=campaign)
            receipt = {'readback_verified': True}
            with patch.object(U, 'upload_run', side_effect=[U.BaseRowMissing('missing'), receipt]) as supplement, \
                    patch.object(U, '_validated_report', return_value=(report, 'report-sha')), \
                    patch('fh12.upload.upload_run') as fh12_base, \
                    patch('fh20r1.upload.upload_run') as fh20_base:
                self.assertEqual(U.upload_with_base_recovery(report['run_id'], root=Path('/fixture')), receipt)
                expected = fh12_base if campaign == 'FH12' else fh20_base
                unused = fh20_base if campaign == 'FH12' else fh12_base
                expected.assert_called_once_with(report['run_id'], root=Path('/fixture'))
                unused.assert_not_called()
                self.assertEqual(supplement.call_count, 2)

    def test_pending_or_conflict_never_triggers_base_write(self):
        for failure in (U.BaseUploadPending('missing provenance headers'), ValueError('Run conflict')):
            with patch.object(U, 'upload_run', side_effect=failure), \
                    patch('fh12.upload.upload_run') as fh12_base, \
                    patch('fh20r1.upload.upload_run') as fh20_base:
                with self.assertRaises(type(failure)):
                    U.upload_with_base_recovery('fixture-run', root=Path('/fixture'))
                fh12_base.assert_not_called(); fh20_base.assert_not_called()

    def test_failed_recovery_is_not_repeated(self):
        report = report_fixture()
        with patch.object(U, 'upload_run', side_effect=U.BaseRowMissing('missing')) as supplement, \
                patch.object(U, '_validated_report', return_value=(report, 'report-sha')), \
                patch('fh20r1.upload.upload_run') as base:
            with self.assertRaises(U.BaseRowMissing):
                U.upload_with_base_recovery(report['run_id'], root=Path('/fixture'))
            base.assert_called_once()
            self.assertEqual(supplement.call_count, 2)


if __name__ == '__main__':
    unittest.main()
