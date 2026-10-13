"""Offline fixed64, immutable local mirror and real transport-boundary tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import reporting as R


def arguments(**changes):
    data = dict(server='s1', dataset='WV3', run_id='EGR2_WV3_s1_EF00_H50_R1_S261013101',
        case_id='EF00', selector='OBSERVED', eval_scope='RR_FR_FULL', eval_protocol_id='test-v2',
        checkpoint_sha=None, model_state_sha='a'*64,
        metrics={'HQNR': .958000123456789, 'ERGAS': 2.0399999999999,
                 'RMSE': None, 'CC': .99876543212345, 'JQM': .97234567891234},
        fields={'Attempt': 1, 'Selected_step': 1000, 'Repeat': 1, 'Seed': 261013101,
                'Status': 'RUNNING'},
        extra={'same_state_verified': True, 'lambda_abs': 1., 'lambda_eps': .001,
               'weights_saved': False, 'joint_pass': False},
        provenance={'config_sha256': 'b'*64, 'code_commit': 'c'*40,
                    'dataset_manifest_sha256': 'd'*64, 'evaluator_revision': 'official-v1'},
        timestamp='2026-10-13T14:30:00+09:00')
    data.update(changes)
    return data


def record(**changes):
    return R.build_record(**arguments(**changes))


class FakeTransport:
    def __init__(self):
        self.snapshot = {**R.destination('s1', 'WV3'), 'column_count': 64,
            'headers': list(R.HEADERS), 'a2_note': json.dumps(
                {'schema_version': R.SCHEMA_VERSION, 'reset_epoch': R.RESET_EPOCH})}
        self.rows = {}
        self.append_calls = self.verified_calls = 0
        self.timeout_after_append = self.corrupt_readback = False
        self.format_calls = []

    def schema_snapshot(self):
        return copy.deepcopy(self.snapshot)

    def find_result_ids(self, identity):
        return [i for i, row in self.rows.items() if row[R.INDEX['Result_ID']] == identity]

    def append_row(self, row):
        number = max(self.rows, default=5)+1
        self.rows[number] = list(row)
        self.append_calls += 1
        if self.timeout_after_append:
            self.timeout_after_append = False
            raise TimeoutError('response lost after committed append')
        return number

    def read_row(self, number):
        row = list(self.rows[number])
        if self.corrupt_readback:
            row[R.INDEX['HQNR↑']] = .1
        return row

    def set_verified(self, number):
        self.rows[number][R.INDEX['Readback']] = 'VERIFIED'
        self.verified_calls += 1

    def format_row(self, number, observation):
        self.format_calls.append(number)


def batch_transport():
    """The production GSpreadTransport with only its HTTP worksheet mocked."""
    worksheet, spreadsheet = mock.Mock(), mock.Mock()
    worksheet.id, worksheet.title, worksheet.row_count = 261001010, 'WV3-ablations', 3000
    spreadsheet.id = R.SPREADSHEET_ID
    transport = R.GSpreadTransport(spreadsheet, worksheet, server='s1', dataset='WV3')
    transport._actual_row_count = 3000
    transport.schema_snapshot = mock.Mock(return_value=FakeTransport().snapshot)
    transport.rows = {}
    transport.timeout_after_append = False

    def get(area, **kwargs):
        assert area == 'BH6:BH'
        return [[transport.rows[i][R.INDEX['Result_ID']]] if i in transport.rows else []
                for i in range(6, max(transport.rows, default=5)+1)]

    def append_rows(rows, **kwargs):
        assert kwargs['value_input_option'] == 'RAW'
        assert kwargs['insert_data_option'] == 'OVERWRITE'
        first = max(transport.rows, default=5)+1
        for offset, row in enumerate(rows):
            transport.rows[first+offset] = list(row)
        if transport.timeout_after_append:
            transport.timeout_after_append = False
            raise TimeoutError('batch committed but response lost')
        return {'updates': {'updatedRange': "'WV3-ablations'!A"+str(first)+':BL'+str(first+len(rows)-1)}}

    def batch_get(areas, **kwargs):
        assert kwargs['value_render_option'] == 'UNFORMATTED_VALUE'
        return [[list(transport.rows[int(area.split(':')[0][1:])])] for area in areas]

    def batch_update(updates, **kwargs):
        assert kwargs['value_input_option'] == 'RAW'
        for item in updates:
            transport.rows[int(item['range'][2:])][R.INDEX['Readback']] = item['values'][0][0]

    worksheet.get.side_effect = get
    worksheet.append_rows.side_effect = append_rows
    worksheet.batch_get.side_effect = batch_get
    worksheet.batch_update.side_effect = batch_update
    return transport


class ReportingTests(unittest.TestCase):
    def setUp(self):
        # Never contend with another test process or a real live publisher.
        self.lock_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.lock_directory.cleanup)
        isolated = mock.patch.object(R.tempfile, 'gettempdir', return_value=self.lock_directory.name)
        isolated.start()
        self.addCleanup(isolated.stop)

    def test_fixed64_precision_blanks_method_and_logical_state(self):
        item = record()
        R.validate_record(item)
        row = item['row']
        self.assertEqual(len(row), 64)
        self.assertEqual(row[R.INDEX['HQNR↑']], .958000123456789)
        self.assertEqual(row[R.INDEX['RMSE↓']], '')
        self.assertEqual(row[R.INDEX['Checkpoint_SHA']], '')
        self.assertIs(row[R.INDEX['Test_aware']], True)
        self.assertEqual(row[R.INDEX['CC↑']], .99876543212345)
        self.assertEqual(row[R.INDEX['JQM↑']], .97234567891234)
        self.assertTrue(item['record_id'].startswith('EGR2-'))
        note = json.loads(row[R.INDEX['Original description / notes']])
        self.assertEqual(note['model_state_sha256'], 'a'*64)
        for key, value in R.METHOD_CONTRACT.items():
            self.assertEqual(note[key], value)

    def test_all_owned_destinations_new_workbook_only(self):
        for server, ds in [('s1', 'WV3'), ('s2', 'QB'), ('s3', 'GF2')]:
            dest = R.destination(server, ds)
            self.assertEqual(dest['sheet_title'], ds+'-ablations')
            self.assertEqual(dest['spreadsheet_id'], '1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE')
        for server, ds in [('s1', 'GF2'), ('s4', 'GF2'), ('s2', 'WV3')]:
            with self.assertRaises(R.ReportingError):
                R.destination(server, ds)

    def test_schema_no_auto_repair_or_display_metadata_substitute(self):
        good = FakeTransport().snapshot
        for key, value in {'spreadsheet_id': 'old', 'sheet_id': 1, 'sheet_title': 'analysis',
                           'column_count': 152, 'headers': list(reversed(R.HEADERS)),
                           'a2_note': '{"reset_epoch":"old"}'}.items():
            with self.subTest(key=key), self.assertRaises(R.ReportingError):
                R.validate_remote_schema({**good, key: value}, server='s1', dataset='WV3')
        with self.assertRaises(R.ReportingError):
            R.validate_remote_schema({**good, 'a2_note': '', 'a2_value': good['a2_note']}, server='s1', dataset='WV3')
        R.validate_remote_schema({**good, 'a2_note': 'schema_version=PAN_FIXED64_V1\nreset_epoch=2026-10-13'}, server='s1', dataset='WV3')

    def test_strict_new_method_and_namespace(self):
        for extra in [{'method_revision': 'BGlobalRR-v1'}, {'lambda_eps_FR': .001},
                      {'requires_source_scene_map': True}, {'b_global': [0, 1]},
                      {'identity': {}}, {'identity': None}]:
            with self.subTest(extra=extra), self.assertRaises((R.ReportingError, ValueError)):
                record(extra=extra)
        for run in ['F00-R1-H50', 'PANDA_OLD', '']:
            with self.assertRaises(R.ReportingError):
                record(run_id=run)
        for key in R.METHOD_CONTRACT:
            with self.assertRaises(R.ReportingError):
                record(extra={'evaluation_identity': {**R.METHOD_CONTRACT, key: 'old'}})

    def test_numeric_observations_require_state_and_same_state_pair(self):
        for changes in [{'model_state_sha': None}, {'model_state_sha': 1},
                        {'extra': {}}, {'extra': {'same_state_verified': True, 'fr_model_state_sha256': 'b'*64}},
                        {'model_state_sha': None, 'metrics': {}, 'extra': {'numeric_observations': {'gradient': 1.}}}]:
            with self.subTest(changes=changes), self.assertRaises(R.ReportingError):
                record(**changes)
        item = record(metrics={}, model_state_sha=None, eval_scope='STATUS', extra={'reason': 'not yet evaluated'})
        R.validate_record(item)
        self.assertEqual(item['row'][R.INDEX['HQNR↑']], '')

    def test_qualified_saved_sha_only_and_unsaved_never_fabricated(self):
        artifact = {'weights_saved': True, 'file_sha256': 'b'*64, 'model_state_sha256': 'a'*64}
        item = record(checkpoint_sha='b'*64, extra={'artifact': artifact, 'joint_pass': True, 'same_state_verified': True})
        R.validate_record(item)
        self.assertEqual(item['row'][R.INDEX['Checkpoint_SHA']], 'b'*64)
        for extra in [{'joint_pass': True}, {'artifact': artifact, 'joint_pass': False},
                      {'artifact': {**artifact, 'file_sha256': 'c'*64}, 'joint_pass': True},
                      {'artifact': {**artifact, 'model_state_sha256': 'c'*64}, 'joint_pass': True},
                      {'artifact': artifact, 'joint_pass': True, 'weights_saved': False}]:
            with self.assertRaises(R.ReportingError):
                record(checkpoint_sha='b'*64, extra={**extra, 'same_state_verified': True})
        with self.assertRaises(R.ReportingError):
            record(extra={'artifact': artifact, 'joint_pass': True, 'same_state_verified': True})

    def test_multistate_aggregate_has_no_fictitious_single_model_identity(self):
        extra = {'aggregate_no_single_model': True, 'numeric_observations': {
            'HQNR_mean': .95, 'source_records': [
                {'record_id': 'EGR2-'+'b'*64, 'model_state_sha256': 'a'*64},
                {'record_id': 'EGR2-'+'c'*64, 'model_state_sha256': 'd'*64}]}}
        kwargs = {'metrics': {}, 'model_state_sha': None,
                  'eval_scope': 'CAMPAIGN_AGGREGATE', 'extra': extra}
        item = record(**kwargs)
        R.validate_record(item)
        self.assertEqual(item['row'][R.INDEX['HQNR↑']], '')
        self.assertEqual(item['row'][R.INDEX['Checkpoint_SHA']], '')
        self.assertIsNone(item['identity']['model_state_sha256'])
        self.assertEqual(item['source']['extra'], extra)
        for changes in [{'eval_scope': 'DIAGNOSTIC'}, {'model_state_sha': 'a'*64},
                        {'metrics': {'HQNR': .95}}, {'fields': {'HQNR↑': .95}},
                        {'extra': {**extra, 'aggregate_no_single_model': False}},
                        {'checkpoint_sha': 'b'*64}]:
            with self.subTest(changes=changes), self.assertRaises(R.ReportingError):
                record(**{**kwargs, **changes})

    def test_unknown_invalid_fields_and_finite_raw_types(self):
        for fields in [{'unknown': 1}, {'Campaign_ID': 'OLD'}, {'lambda_E': .1},
                       {'Test_aware': 'true'}, {'Seed': '261013101'}]:
            with self.assertRaises(R.ReportingError):
                record(fields=fields)
        for metrics in [{'new_metric': 1.}, {'HQNR': float('nan')}, {'HQNR': '0.958'}, {'HQNR': True}]:
            with self.assertRaises(R.ReportingError):
                record(metrics=metrics)
        with self.assertRaises(R.ReportingError):
            record(timestamp='2026-10-13T14:30:00')

    def test_ids_disambiguate_state_scope_selector_not_repeated_payload(self):
        first = record()['record_id']
        self.assertEqual(first, record()['record_id'])
        for changes in [{'model_state_sha': 'b'*64}, {'selector': 'FINAL'}, {'eval_scope': 'RR_SCENE:001'}]:
            self.assertNotEqual(first, record(**changes)['record_id'])

    def test_tampered_self_consistent_digests_cannot_bypass_source_envelope(self):
        item = record()
        item['row'][R.INDEX['HQNR↑']] = .1
        item['payload_sha256'] = R._sha(item['row'])
        with self.assertRaises(R.ReportingError):
            R.validate_record(item)
        item = record()
        item['identity']['campaign'] = R.PARENT_CAMPAIGN_ID
        item['record_id'] = 'EGR2-'+R._sha(item['identity'])
        item['row'][R.INDEX['Result_ID']] = item['record_id']
        item['payload_sha256'] = R._sha(item['row'])
        with self.assertRaises(R.ReportingError):
            R.validate_record(item)

    def test_bounded_inline_numeric_parts_are_complete_deterministic_and_replayable(self):
        observations = {'numeric.path.'+str(i).zfill(4): [i*.123456789]*25 for i in range(350)}
        kwargs = arguments(metrics={}, eval_scope='DIAGNOSTIC', extra={'numeric_observations': observations})
        parts = R.build_records_split(**kwargs)
        self.assertGreater(len(parts), 1)
        self.assertEqual(parts, R.build_records_split(**kwargs))
        got = {}
        for item in parts:
            R.validate_record(item)
            self.assertLessEqual(len(item['row'][R.INDEX['Original description / notes']]), 45000)
            got.update(item['source']['extra']['numeric_observations'])
            self.assertEqual(item['identity']['model_state_sha256'], 'a'*64)
        self.assertEqual(got, observations)
        self.assertEqual(len({item['record_id'] for item in parts}), len(parts))
        self.assertEqual(R.build_records_split(**arguments()), [record()])
        with self.assertRaises(R.ReportingError):
            R.build_records_split(**arguments(metrics={}, extra={'metadata': 'x'*50000}))
        with self.assertRaises(R.ReportingError):
            R.build_records_split(**arguments(metrics={}, extra={'numeric_observations': {'one': 'x'*50000}}))

    def test_local_first_idempotency_conflict_crash_recovery_and_journal_containment(self):
        item = record()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            box = R.LocalOutbox(root)
            box.enqueue(item, journal='runs/EF00/evaluations.jsonl')
            box.enqueue(item)
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()), 1)
            with self.assertRaises(R.RecordConflict):
                box.enqueue(record(metrics={'HQNR': .1}))
            (box.outbox/(item['record_id']+'.json')).unlink()
            self.assertEqual(box.recover(), 1)
            self.assertEqual(box.recover(), 0)
            self.assertEqual(box.pending(), [item])
            (Path(root)/'escape').symlink_to(other, target_is_directory=True)
            for journal in ['../escape.jsonl', '/tmp/escape.jsonl', 'escape/records.jsonl']:
                with self.assertRaises(R.ReportingError):
                    box.enqueue(item, journal=journal)

    def test_full_readback_local_mirror_and_idempotence(self):
        item, transport = record(), FakeTransport()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            box.enqueue(item)
            result = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((result['verified'], result['pending']), (1, 0))
            receipt = json.loads((box.receipts/(item['record_id']+'.json')).read_text())
            self.assertEqual(receipt['row'], transport.rows[6])
            R.validate_receipt(receipt, item)
            self.assertTrue((box.outbox/(item['record_id']+'.json')).exists())
            self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['verified'], 0)
            R.publish_record(transport, item)
            self.assertEqual(transport.append_calls, 1)

    def test_ambiguous_append_retry_never_duplicates(self):
        item, transport = record(), FakeTransport()
        transport.timeout_after_append = True
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root); box.enqueue(item)
            self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['failed'], 1)
            self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['deferred'], 1)
            with mock.patch.object(R.time, 'time', return_value=10**12):
                self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['verified'], 1)
            self.assertEqual(transport.append_calls, 1)

    def test_wrong_readback_and_remote_conflict_do_not_get_verified(self):
        item, transport = record(), FakeTransport()
        transport.corrupt_readback = True
        with self.assertRaises(R.ReportingError):
            R.publish_record(transport, item)
        self.assertEqual(transport.verified_calls, 0)
        transport.corrupt_readback = False
        transport.rows[6][R.INDEX['CC↑']] = .2
        with self.assertRaises(R.RecordConflict):
            R.publish_record(transport, item)
        self.assertEqual(transport.append_calls, 1)

    def test_fake_verified_receipt_cannot_suppress_pending(self):
        item = record()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root); box.enqueue(item)
            R.atomic_json(box.receipts/(item['record_id']+'.json'),
                          {'status': 'READBACK_VERIFIED', 'payload_sha256': item['payload_sha256']})
            with self.assertRaises(R.ReportingError):
                box.pending()

    def test_unsupported_publish_limits_and_independent_enqueue_lock(self):
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            for limit in [0, -1, True, 1.5]:
                with self.assertRaises(R.ReportingError):
                    box.publish_once(FakeTransport(), server='s1', dataset='WV3', limit=limit)
            path = Path(tempfile.gettempdir())/'panda_edgeres48_261001010.publisher.lock'
            with R._lock(path, nonblocking=True):
                box.enqueue(record())
                with self.assertRaises(R.ReportingError):
                    box.publish_once(FakeTransport(), server='s1', dataset='WV3')
            self.assertEqual(len(box.pending()), 1)

    def test_bulk_real_transport_two_readbacks_single_raw_append(self):
        transport = batch_transport()
        records = [record(selector='OBS_'+str(i)) for i in range(32)]
        receipts = transport.publish_records(records)
        self.assertEqual(len(receipts), 32)
        self.assertEqual(transport.worksheet.append_rows.call_count, 1)
        self.assertEqual(transport.worksheet.batch_get.call_count, 2)
        for receipt, item in zip(receipts, records):
            R.validate_receipt(receipt, item)
        transport.publish_records(records)
        self.assertEqual(transport.worksheet.append_rows.call_count, 1)

    def test_bulk_retry_after_ambiguous_append(self):
        transport = batch_transport(); transport.timeout_after_append = True
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            for i in range(4):
                box.enqueue(record(selector='OBS_'+str(i)))
            self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['failed'], 4)
            with mock.patch.object(R.time, 'time', return_value=10**12):
                report = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((report['verified'], report['pending']), (4, 0))
            self.assertEqual(transport.worksheet.append_rows.call_count, 1)

    def test_bulk_bound_and_only_missing_append(self):
        transport = batch_transport()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            for i in range(40):
                box.enqueue(record(selector='OBS_'+str(i)))
            report = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((report['verified'], report['pending']), (32, 8))
            self.assertEqual(box.publish_once(transport, server='s1', dataset='WV3')['verified'], 8)
            self.assertEqual(transport.worksheet.append_rows.call_count, 2)
        a, b, c = [record(selector=name) for name in ['A', 'B', 'C']]
        transport = batch_transport(); transport.publish_records([a])
        transport.publish_records([a, b])
        self.assertEqual(len(transport.worksheet.append_rows.call_args.args[0]), 1)
        transport.rows[6][R.INDEX['HQNR↑']] = .1
        with self.assertRaises(R.RecordConflict):
            transport.publish_records([a, c])
        self.assertEqual(transport.worksheet.append_rows.call_count, 2)

    def test_bulk_forged_receipts_not_committed_as_success(self):
        transport = FakeTransport()
        transport.publish_records = lambda records: [
            {'record_id': item['record_id'], 'payload_sha256': item['payload_sha256'],
             'status': 'READBACK_VERIFIED'} for item in records]
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root); box.enqueue(record())
            result = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((result['verified'], result['failed'], result['pending']), (0, 1, 1))

    def test_rows_only_resize_and_display_format_do_not_round_raw_values(self):
        transport = batch_transport(); transport._actual_row_count = 5
        item = record()
        transport.publish_records([item])
        transport.worksheet.resize.assert_called_once_with(rows=1005)
        self.assertEqual(transport.rows[6][R.INDEX['HQNR↑']], .958000123456789)
        formatting = transport.spreadsheet.batch_update.call_args.args[0]['requests']
        patterns = [req['repeatCell']['cell']['userEnteredFormat'].get('numberFormat', {}).get('pattern')
                    for req in formatting]
        self.assertIn('0.0000', patterns)


if __name__ == '__main__':
    unittest.main()
