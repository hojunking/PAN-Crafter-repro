"""Reporting contracts with an in-memory Sheets fake; no credentials/network."""
import copy
import math
import re
import tempfile
import unittest
from pathlib import Path

from panpv.common import atomic_json, canonical_sha, file_sha, load_json
from panpv.reporting import (DedicatedSheet, HEADERS, IDX, SHEET, SHEET_ID, SPREADSHEET_ID,
                             RR_KEYS, FR_KEYS, build_row, publish_pending)


def col_index(letters):
    value = 0
    for ch in letters: value = value * 26 + ord(ch) - ord("A") + 1
    return value - 1


class FakeBook:
    id = SPREADSHEET_ID

    def __init__(self):
        self.title, self.sheet_id = SHEET, SHEET_ID
        self.rows = {5: HEADERS.copy()}
        self.calls = []
        self.native = {}
        self.row_count = 1000
        self.corrupt_readback = False

    def area(self, area):
        title, cells = area.split("!")
        assert title == "'" + SHEET + "'", "Another sheet was addressed"
        match = re.fullmatch(r"([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?", cells)
        a, first, b, last = match.groups()
        return int(first), int(last or first), col_index(a), col_index(b or a)

    def values_get(self, area, params=None):
        first, last, c, d = self.area(area)
        rows = [self.rows.get(i, [""]*64)[c:d+1] for i in range(first, last+1)]
        if self.corrupt_readback and self.calls and first == last:
            rows[0][0] = "BROKEN"
        return {"values": rows}

    def fetch_sheet_metadata(self, params=None):
        properties = dict(sheetId=self.sheet_id, title=self.title,
                          gridProperties=dict(rowCount=self.row_count, columnCount=64))
        result = dict(properties=properties)
        if params.get("includeGridData"):
            first, _, _, _ = self.area(params["ranges"][0])
            cells = copy.deepcopy(self.native.get(first, [{} for _ in range(64)]))
            for i, value in enumerate(self.rows.get(first, [""] * 64)):
                if value != "": cells[i]["userEnteredValue"] = {"stringValue": str(value)}
            result["data"] = [dict(startRow=first-1, startColumn=0, rowData=[dict(values=cells)])]
        return {"spreadsheetId": self.id, "sheets": [result]}

    def values_batch_update(self, body):
        assert body["valueInputOption"] == "RAW"
        self.calls.append(copy.deepcopy(body))
        for part in body["data"]:
            first, last, c, d = self.area(part["range"])
            assert first == last and first >= 6 and first <= 1000
            assert not c <= IDX["Review"] <= d, "Manual Review was written"
            values = self.rows.setdefault(first, [""] * 64)
            assert len(part["values"][0]) == d-c+1
            values[c:d+1] = part["values"][0]


class FakeClient:
    def __init__(self, book): self.book, self.opened = book, 0
    def open_by_key(self, key):
        assert key == SPREADSHEET_ID
        self.opened += 1
        return self.book


def row():
    value = [""] * 64
    for key, x in dict(Scene_ID="scene", Method="BROVEY8", Status="EVAL_COMPLETE",
                       Server="s1", Result_ID="d"*64).items(): value[IDX[key]] = x
    value[IDX["FR_HQNR↑"]] = 0.945
    return value


def fixture(root):
    protocol = dict(protocol_sha256="p"*64,
        rr_protocol={"scope": "RR_WINDOWS"}, fr_protocol={"scope": "FR_TILES"},
        audit=[dict(rr_mask_sha256="a"*64, fr_mask_sha256="b"*64)], implementation={"file": "f"*64})
    atomic_json(root / "protocol.json", {"scenes": {"scene": protocol}})
    registry = dict(campaign_id="campaign", source_root=str(root), work_root=str(root),
        execution_server="s1", target_sensor="WV2", seed=261002,
        scenes=[dict(scene_id="scene", sha256="i"*64, patch_index_sha256="j"*64)],
        donor=dict(sha256="d"*64),
        protocol=dict(path=str(root / "protocol.json"), sha256=file_sha(root / "protocol.json")))
    atomic_json(root / "registry.json", registry)
    report_path = root / "evaluations/scene_BROVEY8/report.json"
    identity = dict(method="BROVEY8", scene_id="scene", source_sha256="i"*64, final=True,
                    declared_checkpoint_sha256=None, protocol_sha256="p"*64,
                    brovey=dict(weights=[.125]*8))
    report = dict(schema="PANPV_PRIVATE_TILE_EVALUATION_v1", status="COMPLETE", full_scene=False,
        official_benchmark=False, identity=identity, report_path=str(report_path), eval_seconds=12.5,
        rr=dict(protocol=protocol["rr_protocol"], samples=1,
                metrics={v: 0.3+i for i,v in enumerate(RR_KEYS.values())}),
        fr=dict(protocol=protocol["fr_protocol"], tiles=1,
                metrics=dict({v: 0.2+i for i,v in enumerate(FR_KEYS.values())}, jqm=None)))
    report["payload_sha256"] = canonical_sha(report)
    atomic_json(report_path, report)
    receipt = dict(state="COMPLETE", registry_sha256=file_sha(root / "registry.json"),
        stage=dict(id="scene_BROVEY8", kind="baseline", scene_id="scene", method="BROVEY8"),
        result=dict(selector="FIXED", completed_step=0, checkpoint_sha256=None, result=report),
        artifacts=[dict(path=str(report_path.relative_to(root)), sha256=file_sha(report_path))],
        recorded_utc="2026-10-04T00:00:00+00:00")
    receipt_path = root / "stage_receipts/scene_BROVEY8.json"
    atomic_json(receipt_path, receipt)
    atomic_json(root / "outbox/scene_BROVEY8.json", dict(receipt=str(receipt_path)))
    return registry, receipt


class ReportingTests(unittest.TestCase):
    def test_append_exact_readback_and_review_never_written(self):
        book = FakeBook()
        book.rows[6] = [""] * 64
        book.rows[6][IDX["Review"]] = "manual reserved row"
        result = DedicatedSheet(book).publish(row())
        self.assertEqual(result["row"], 7)
        self.assertEqual(book.rows[6][IDX["Review"]], "manual reserved row")
        self.assertEqual(book.rows[7][IDX["Readback"]], "VERIFIED")
        self.assertEqual(len(book.calls), 2)

    def test_idempotent_dedupe_preserves_manual_review(self):
        book = FakeBook()
        writer = DedicatedSheet(book)
        writer.publish(row())
        book.rows[6][IDX["Review"]] = "accepted by human"
        count = len(book.calls)
        writer.publish(row())
        self.assertEqual(len(book.calls), count)
        self.assertEqual(book.rows[6][IDX["Review"]], "accepted by human")

    def test_result_key_metric_mismatch_never_overwritten(self):
        book = FakeBook()
        writer = DedicatedSheet(book)
        writer.publish(row())
        changed = row(); changed[IDX["FR_HQNR↑"]] = .999
        with self.assertRaisesRegex(ValueError, "different metrics"):
            writer.publish(changed)
        self.assertEqual(book.rows[6][IDX["FR_HQNR↑"]], .945)

    def test_numeric_transport_one_ulp_tolerance_but_not_metric_changes(self):
        original = row()
        actual = original.copy()
        actual[IDX["FR_HQNR↑"]] = math.nextafter(.945, math.inf)
        self.assertTrue(DedicatedSheet._matches(actual, original))
        actual[IDX["FR_HQNR↑"]] = .945 + 1e-5
        self.assertFalse(DedicatedSheet._matches(actual, original))
        actual[IDX["FR_HQNR↑"]] = float("nan")
        self.assertFalse(DedicatedSheet._matches(actual, original))
        actual = original.copy(); original[IDX["Input_adaptation"]] = False
        actual[IDX["Input_adaptation"]] = 0
        self.assertFalse(DedicatedSheet._matches(actual, original))

    def test_wrong_tabs_and_headers_fail_without_writes(self):
        for attr, value in (("title", "WV3-main"), ("sheet_id", 7), ("id", "other-book")):
            book = FakeBook(); setattr(book, attr, value)
            with self.assertRaises(ValueError): DedicatedSheet(book)
            self.assertEqual(book.calls, [])
        book = FakeBook(); book.rows[5][0] = "changed"
        with self.assertRaises(ValueError): DedicatedSheet(book)
        self.assertEqual(book.calls, [])

    def test_blank_formula_and_native_dropdown_are_preserved(self):
        book = FakeBook()
        cells = [{} for _ in range(64)]
        cells[0] = dict(userEnteredValue={"formulaValue": '=IF(1,"","")'})
        book.native[6] = cells
        with self.assertRaisesRegex(ValueError, "genuinely blank"):
            DedicatedSheet(book).publish(row())
        self.assertEqual(book.calls, [])
        cells[0] = {}; cells[1] = dict(dataValidation=dict(condition=dict(type="ONE_OF_LIST", values=[{"userEnteredValue":"ZS_WV3"}])))
        with self.assertRaisesRegex(ValueError, "validation"):
            DedicatedSheet(book).publish(row())
        self.assertEqual(book.calls, [])

    def test_failed_readback_cannot_mark_verified(self):
        book = FakeBook(); writer = DedicatedSheet(book); book.corrupt_readback = True
        with self.assertRaisesRegex(ValueError, "readback"):
            writer.publish(row())
        self.assertEqual(book.rows[6][IDX["Readback"]], "")

    def test_row_limit_never_grows_sheet(self):
        book = FakeBook(); book.row_count = 6; book.rows[6] = ["reserved"] * 64
        with self.assertRaisesRegex(ValueError, "row limit"):
            DedicatedSheet(book).publish(row())
        self.assertEqual(book.calls, [])

    def test_completed_metrics_are_exactly_mapped_and_missing_stays_blank(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); registry, receipt = fixture(root)
            data = build_row(root, registry, receipt)
            self.assertEqual(data[IDX["FR_HQNR↑"]], .2)
            self.assertEqual(data[IDX["RR_ERGAS↓"]], .3)
            self.assertEqual(data[IDX["FR_JQM↑"]], "")
            self.assertEqual(data[IDX["Infer_s"]], "")
            self.assertEqual(data[IDX["Log_URL"]], "")
            self.assertIn("NOT_FULL_SCENE", data[IDX["FR_scope"]])
            self.assertEqual(data[IDX["Input_adaptation"]], False)
            book = FakeBook(); client = FakeClient(book)
            result = publish_pending(root, registry, client=client)
            self.assertEqual(result["outcomes"][0]["status"], "SHEET_READBACK_VERIFIED")
            publish_pending(root, registry, client=client)
            self.assertEqual(client.opened, 1)

    def test_nonfinite_metrics_and_checkpoint_mismatch_are_queued(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); registry, receipt = fixture(root)
            receipt["result"]["result"]["rr"]["metrics"]["ergas"] = float("nan")
            with self.assertRaises(ValueError): build_row(root, registry, receipt)
            registry, receipt = fixture(root)
            receipt["result"]["result"]["identity"]["declared_checkpoint_sha256"] = "x" * 64
            report = receipt["result"]["result"]
            report["payload_sha256"] = canonical_sha({k:v for k,v in report.items() if k != "payload_sha256"})
            with self.assertRaisesRegex(ValueError, "different method/scene/checkpoint"):
                build_row(root, registry, receipt)

    def test_actual_inference_cost_and_parameter_fields(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); registry, receipt = fixture(root)
            report = receipt["result"]["result"]
            report.update(infer_seconds=1.25, params_m=0, hardware="Observed CPU")
            report["payload_sha256"] = canonical_sha({k:v for k,v in report.items() if k != "payload_sha256"})
            atomic_json(report["report_path"], report)
            receipt["artifacts"][0]["sha256"] = file_sha(report["report_path"])
            result = build_row(root, registry, receipt)
            self.assertEqual(result[IDX["Infer_s"]], 1.25)
            self.assertEqual(result[IDX["Params_M"]], 0)
            self.assertEqual(result[IDX["Hardware"]], "Observed CPU")
            self.assertEqual(result[IDX["EndToEnd_s"]], "")
            self.assertNotIn("Infer_s blank", result[IDX["Notes"]])

    def test_training_blocks_stay_local_without_credentials(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); registry, receipt = fixture(root)
            receipt["stage"]["kind"] = "train"
            atomic_json(root / "stage_receipts/scene_BROVEY8.json", receipt)
            client = FakeClient(FakeBook())
            result = publish_pending(root, registry, client=client)
            self.assertEqual(result["outcomes"][0]["status"], "LOCAL_TRAINING_BLOCK_ONLY")
            self.assertEqual(client.opened, 0)

    def test_reporting_lock_stays_inside_only_writable_campaign_root(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name); root = base / "campaign"; repo = base / "readonly_repo"
            root.mkdir(); repo.mkdir(); repo.chmod(0o555)
            try:
                registry, receipt = fixture(root)
                registry["source_root"] = str(repo)
                atomic_json(root / "registry.json", registry)
                receipt["registry_sha256"] = file_sha(root / "registry.json")
                atomic_json(root / "stage_receipts/scene_BROVEY8.json", receipt)
                result = publish_pending(root, registry, client=FakeClient(FakeBook()))
                self.assertEqual(result["outcomes"][0]["status"], "SHEET_READBACK_VERIFIED")
                self.assertTrue((root / ".reporting.lock").is_file())
                self.assertFalse((repo / "work_dir").exists())
            finally:
                repo.chmod(0o755)


if __name__ == "__main__": unittest.main()
