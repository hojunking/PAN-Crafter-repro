"""Offline reporting contract tests. No credentials or Google connection needed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from teacher48 import reporting as R


def record(**overrides):
    data = dict(server="s1", dataset="WV3", run_id="F00-R1-H50", case_id="F00",
                selector="OBSERVED", eval_scope="RR_FR_FULL", eval_protocol_id="test-v1",
                checkpoint_sha="a" * 64, metrics={"HQNR": .958000123456789, "ERGAS": 2.0399999999999,
                                                      "RMSE": None, "CC": .99876543212345, "JQM": .97234567891234},
                fields={"Attempt": 1, "Selected_step": 1000, "Repeat": 1, "Seed": 261013101,
                        "Status": "RUNNING"},
                extra={"same_state_verified": True, "lambda_abs": 1.0, "lambda_eps_RR": .001,
                       "lambda_eps_FR": .001, "joint_pass": True},
                provenance={"config_sha256": "b" * 64, "code_commit": "c" * 40,
                            "dataset_manifest_sha256": "d" * 64, "evaluator_revision": "official-v1"},
                timestamp="2026-10-13T14:30:00+09:00")
    data.update(overrides)
    return R.build_record(**data)


class FakeTransport:
    def __init__(self):
        self.snapshot = {**R.destination("s1", "WV3"), "column_count": 64,
                         "headers": list(R.HEADERS),
                         "a2_note": json.dumps({"schema_version": R.SCHEMA_VERSION, "reset_epoch": R.RESET_EPOCH})}
        self.rows = {}
        self.append_calls = 0
        self.verified_calls = 0
        self.timeout_after_append = False
        self.corrupt_readback = False
        self.format_calls = []

    def schema_snapshot(self):
        return copy.deepcopy(self.snapshot)

    def find_result_ids(self, identity):
        return [index for index, row in self.rows.items() if row[R.INDEX["Result_ID"]] == identity]

    def append_row(self, row):
        index = max(self.rows.keys(), default=5) + 1
        self.rows[index] = list(row)
        self.append_calls += 1
        if self.timeout_after_append:
            self.timeout_after_append = False
            raise TimeoutError("response lost after successful append")
        return index

    def read_row(self, index):
        row = list(self.rows[index])
        if self.corrupt_readback:
            row[R.INDEX["HQNR↑"]] = 0.1
        return row

    def set_verified(self, index):
        self.rows[index][R.INDEX["Readback"]] = "VERIFIED"
        self.verified_calls += 1

    def format_row(self, row, observation):
        self.format_calls.append(row)


def batch_transport():
    """Exercise real batch code with an in-memory worksheet HTTP boundary."""
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
            assert item['range'].startswith('AZ')
            transport.rows[int(item['range'][2:])][R.INDEX['Readback']] = item['values'][0][0]

    worksheet.get.side_effect = get
    worksheet.append_rows.side_effect = append_rows
    worksheet.batch_get.side_effect = batch_get
    worksheet.batch_update.side_effect = batch_update
    return transport


class Fixed64Tests(unittest.TestCase):
    def test_fixed64_precision_types_and_new_method_fields(self):
        item = record()
        R.validate_record(item)
        row = item["row"]
        self.assertEqual(len(row), 64)
        self.assertEqual(row[R.INDEX["HQNR↑"]], .958000123456789)
        self.assertEqual(row[R.INDEX["RMSE↓"]], "")
        self.assertIs(row[R.INDEX["Test_aware"]], True)
        self.assertEqual(row[R.INDEX["JQM↑"]], .97234567891234)
        self.assertEqual(row[R.INDEX["CC↑"]], .99876543212345)
        for name in R.OBSOLETE_PARAMETERS:
            self.assertEqual(row[R.INDEX[name]], "")
        notes = json.loads(row[R.INDEX["Original description / notes"]])
        self.assertEqual(notes["extra_parameters"]["lambda_eps_FR"], .001)
        self.assertEqual(notes["schema_version"], "PAN_FIXED64_V1")

    def test_unmeasured_failed_and_blocked_records_are_not_filtered(self):
        item = record(checkpoint_sha=None, metrics={}, selector="BLOCKED_DATA_PROTOCOL",
                      eval_scope="STATUS", fields={"Status": "BLOCKED_DATA_PROTOCOL"},
                      extra={"reason": "native training FR missing"})
        self.assertEqual(item["row"][R.INDEX["HQNR↑"]], "")
        self.assertEqual(item["row"][R.INDEX["ERGAS↓"]], "")
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            box.enqueue(item)
            self.assertEqual(box.publish_once(FakeTransport(), server="s1", dataset="WV3")["verified"], 1)

    def test_only_owned_dataset_ablations(self):
        for server, dataset, title in (("s1", "WV3", "WV3-ablations"),
                                       ("s2", "QB", "QB-ablations"), ("s3", "GF2", "GF2-ablations")):
            self.assertEqual(R.destination(server, dataset)["sheet_title"], title)
        for server, dataset in (("s1", "GF2"), ("s4", "GF2"), ("s2", "WV3"), ("s3", "WV2")):
            with self.assertRaises(R.ReportingError):
                R.destination(server, dataset)

    def test_reject_wrong_workbook_tab_columns_header_and_note(self):
        good = FakeTransport().snapshot
        mutations = {"spreadsheet_id": "1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0",
                     "sheet_title": "ablations", "sheet_id": 1697831403,
                     "column_count": 152, "headers": list(reversed(R.HEADERS)),
                     "a2_note": '{"schema_version":"PAN_FIXED64_V1","reset_epoch":"2026-09-13"}'}
        for key, bad in mutations.items():
            with self.subTest(key=key), self.assertRaises(R.ReportingError):
                R.validate_remote_schema({**good, key: bad}, server="s1", dataset="WV3")
        with self.assertRaises(R.ReportingError):
            R.validate_remote_schema({**good, "a2_note": "", "a2_value": good["a2_note"]}, server="s1", dataset="WV3")
        R.validate_remote_schema({**good, "a2_note": "schema_version=PAN_FIXED64_V1\nreset_epoch=2026-10-13"}, server="s1", dataset="WV3")

    def test_unknown_fields_old_campaign_legacy_params_and_wrong_types_fail(self):
        for fields in ({"unknown": 1}, {"Campaign_ID": "TA2_OLD"}, {"lambda_E": .1},
                       {"Test_aware": "true"}, {"Seed": "261013101"}):
            with self.subTest(fields=fields), self.assertRaises(R.ReportingError):
                record(fields=fields)
        with self.assertRaises(R.ReportingError):
            record(metrics={"new_metric": 1.0})
        with self.assertRaises(R.ReportingError):
            record(metrics={"HQNR": float("nan")})
        with self.assertRaises(R.ReportingError):
            record(timestamp="2026-10-13T14:30:00")

    def test_checkpoint_consistency_guard(self):
        for changes in ({"checkpoint_sha": None}, {"extra": {}},
                        {"extra": {"same_state_verified": True, "fr_model_state_sha256": "e" * 64}},
                        {"fields": {"Checkpoint_SHA": "b" * 64}}):
            with self.subTest(changes=changes), self.assertRaises(R.ReportingError):
                record(**changes)

    def test_selectors_and_scene_scopes_have_distinct_ids(self):
        observed = record()["record_id"]
        self.assertNotEqual(observed, record(selector="EXACT_FINAL")["record_id"])
        self.assertNotEqual(observed, record(eval_scope="RR_SCENE:001")["record_id"])
        self.assertNotEqual(observed, record(checkpoint_sha="b" * 64)["record_id"])
        self.assertEqual(observed, record()["record_id"])

    def test_local_first_duplicate_conflict_and_recover(self):
        item = record()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            box.enqueue(item, journal="runs/F00/evaluations.jsonl")
            box.enqueue(item)
            self.assertEqual(len((Path(root) / "records.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len(box.pending()), 1)
            changed = record(metrics={"HQNR": .959, "ERGAS": 2.03})
            with self.assertRaises(R.RecordConflict):
                box.enqueue(changed)
            (box.outbox / (item["record_id"] + ".json")).unlink()
            self.assertEqual(box.recover(), 1)
            self.assertEqual(box.pending(), [item])
            self.assertEqual(box.recover(), 0)
            with self.assertRaises(R.ReportingError):
                box.enqueue(item, journal="../escape.jsonl")

    def test_publish_full_readback_receipt_mirror_and_idempotence(self):
        item, transport = record(), FakeTransport()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            box.enqueue(item)
            report = box.publish_once(transport, server="s1", dataset="WV3")
            self.assertEqual((report["verified"], report["pending"]), (1, 0))
            receipt = json.loads((box.receipts / (item["record_id"] + ".json")).read_text())
            self.assertEqual(receipt["row"], transport.rows[6])
            self.assertEqual(receipt["row"][R.INDEX["Readback"]], "VERIFIED")
            self.assertEqual(receipt["status"], "READBACK_VERIFIED")
            self.assertTrue((box.outbox / (item["record_id"] + ".json")).exists())
            self.assertEqual(box.publish_once(transport, server="s1", dataset="WV3")["verified"], 0)
            R.publish_record(transport, item)
            self.assertEqual(transport.append_calls, 1)

    def test_ambiguous_timeout_is_recovered_without_second_append(self):
        item, transport = record(), FakeTransport()
        transport.timeout_after_append = True
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            box.enqueue(item)
            report = box.publish_once(transport, server="s1", dataset="WV3")
            self.assertEqual(report["failed"], 1)
            self.assertEqual(transport.verified_calls, 0)
            self.assertEqual(box.publish_once(transport, server="s1", dataset="WV3")["deferred"], 1)
            with mock.patch.object(R.time, "time", return_value=10**12):
                report = box.publish_once(transport, server="s1", dataset="WV3")
            self.assertEqual(report["verified"], 1)
            self.assertEqual(transport.append_calls, 1)
            self.assertEqual(transport.verified_calls, 1)

    def test_conflicting_remote_row_is_never_overwritten(self):
        item, transport = record(), FakeTransport()
        transport.append_row(item["row"])
        transport.rows[6][R.INDEX["ERGAS↓"]] = 1.99
        with self.assertRaises(R.RecordConflict):
            R.publish_record(transport, item)
        self.assertEqual(transport.verified_calls, 0)
        self.assertEqual(transport.rows[6][R.INDEX["ERGAS↓"]], 1.99)

    def test_corrupt_readback_not_marked_verified(self):
        transport = FakeTransport()
        transport.corrupt_readback = True
        with self.assertRaises(R.ReportingError):
            R.publish_record(transport, record())
        self.assertEqual(transport.verified_calls, 0)

    def test_bad_schema_never_append_or_mutate(self):
        transport = FakeTransport()
        transport.snapshot["headers"][0] = "=FORMULA()"
        with self.assertRaises(R.ReportingError):
            R.publish_record(transport, record())
        self.assertEqual(transport.append_calls, 0)

    def test_raw_boolean_and_number_readback_not_formatted_strings(self):
        row = record()["row"]
        self.assertTrue(R._row_equal(row, row))
        for header, bad in (("Test_aware", "TRUE"), ("HQNR↑", "0.958000123456789"), ("Test_aware", 1)):
            changed = list(row)
            changed[R.INDEX[header]] = bad
            self.assertFalse(R._row_equal(changed, row))

    def test_publisher_requests_are_serialized_without_training_lock(self):
        lockpath = Path(tempfile.gettempdir()) / "panda_teacher48_261001010.publisher.lock"
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            with R._lock(lockpath, nonblocking=True):
                # Enqueue remains available while the publisher lock is held.
                box.enqueue(record())
                with self.assertRaises(R.ReportingError):
                    box.publish_once(FakeTransport(), server="s1", dataset="WV3")
            self.assertEqual(len(box.pending()), 1)

    def test_real_transport_reads_a2_note_and_resizes_only_rows_using_fresh_count(self):
        # gspread's local count can drift upward on OVERWRITE append; exercise
        # the actual adapter without any auth client or network request.
        worksheet = mock.Mock()
        worksheet.id, worksheet.title, worksheet.row_count = 261001010, 'WV3-ablations', 8888
        worksheet.get.side_effect = lambda area, **kw: ([list(R.HEADERS)] if area == 'A5:BL5'
            else [['existing-' + str(i)] for i in range(2995)] if area == 'BH6:BH' else [])
        worksheet.append_rows.return_value = {'updates': {'updatedRange': "'WV3-ablations'!A3001:BL3001"}}
        spreadsheet = mock.Mock()
        spreadsheet.id = R.SPREADSHEET_ID
        spreadsheet.fetch_sheet_metadata.return_value = {
            'spreadsheetId': R.SPREADSHEET_ID, 'sheets': [{
                'properties': {'sheetId': 261001010, 'title': 'WV3-ablations',
                               'gridProperties': {'rowCount': 3000, 'columnCount': 64}},
                'data': [{'startRow': 1, 'rowData': [{'values': [{'note': json.dumps({
                    'schema_version': R.SCHEMA_VERSION, 'reset_epoch': R.RESET_EPOCH})}]}]}]}]}
        transport = R.GSpreadTransport(spreadsheet, worksheet, server='s1', dataset='WV3')
        R.validate_remote_schema(transport.schema_snapshot(), server='s1', dataset='WV3')
        self.assertEqual(transport.append_row(record()['row']), 3001)
        worksheet.resize.assert_called_once_with(rows=4000)
        kwargs = worksheet.append_rows.call_args.kwargs
        self.assertEqual(kwargs['value_input_option'], 'RAW')
        self.assertEqual(kwargs['table_range'], 'A5:BL4000')
        self.assertEqual(kwargs['insert_data_option'], 'OVERWRITE')
        query = spreadsheet.fetch_sheet_metadata.call_args.kwargs['params']
        self.assertEqual(query['ranges'], "'WV3-ablations'!A2")
        self.assertIn('note', query['fields'])
        requests = spreadsheet.batch_update.call_args.args[0]['requests']
        self.assertEqual(requests[0]['setBasicFilter']['filter']['range']['endColumnIndex'], 64)

    def test_real_batch_publishes_32_with_one_raw_append_and_two_readbacks(self):
        transport = batch_transport()
        records = [record(eval_scope='RR_SCENE:'+str(i)) for i in range(32)]
        receipts = transport.publish_records(records)
        self.assertEqual(len(receipts), 32)
        self.assertEqual(transport.worksheet.append_rows.call_count, 1)
        self.assertEqual(transport.worksheet.batch_get.call_count, 2)
        self.assertEqual(transport.worksheet.batch_update.call_count, 1)
        for receipt in receipts:
            self.assertEqual(receipt['row'], transport.rows[receipt['row_number']])
            self.assertEqual(receipt['row'][R.INDEX['Readback']], 'VERIFIED')
        # Retrying already committed batch performs no append or marker writes.
        transport.publish_records(records)
        self.assertEqual(transport.worksheet.append_rows.call_count, 1)
        self.assertEqual(transport.worksheet.batch_update.call_count, 1)

    def test_real_batch_timeout_recovers_all_ids_without_duplicate_append(self):
        transport = batch_transport()
        transport.timeout_after_append = True
        records = [record(eval_scope='FR_SCENE:'+str(i)) for i in range(8)]
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            for item in records:
                box.enqueue(item)
            first = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((first['failed'], first['pending']), (8, 8))
            with mock.patch.object(R.time, 'time', return_value=10**12):
                second = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((second['verified'], second['pending']), (8, 0))
            self.assertEqual(transport.worksheet.append_rows.call_count, 1)
            self.assertEqual(len(transport.rows), 8)

    def test_real_batch_only_appends_missing_and_rejects_existing_conflict(self):
        transport = batch_transport()
        records = [record(eval_scope='RR_SCENE:'+str(i)) for i in range(4)]
        transport.publish_records(records[:2])
        transport.publish_records(records)
        self.assertEqual(len(transport.rows), 4)
        self.assertEqual(len(transport.worksheet.append_rows.call_args.args[0]), 2)
        transport.rows[6][R.INDEX['HQNR↑']] = .8
        with self.assertRaises(R.RecordConflict):
            transport.publish_records(records)
        self.assertEqual(transport.worksheet.append_rows.call_count, 2)

    def test_batch_default_pass_is_bounded_and_preserves_remainder(self):
        transport = batch_transport()
        with tempfile.TemporaryDirectory() as root:
            box = R.LocalOutbox(root)
            for i in range(40):
                box.enqueue(record(eval_scope='RR_SCENE:'+str(i)))
            first = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((first['verified'], first['pending']), (32, 8))
            second = box.publish_once(transport, server='s1', dataset='WV3')
            self.assertEqual((second['verified'], second['pending']), (8, 0))


if __name__ == "__main__":
    unittest.main()
