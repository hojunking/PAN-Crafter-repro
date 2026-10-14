import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import transport as t
from .selection import HEADERS, INDEX, SPREADSHEET_ID, CAMPAIGN_ID, destination


KEY = [CAMPAIGN_ID, "WV3", "s1", "EGR2_test", 1]


def row(rid="VIEW-fixture", selector="BEST_HQNR_FALLBACK", step=1000):
    values = [""] * 64
    fields = dict(zip(("Campaign_ID", "Dataset", "Server", "Run_ID", "Attempt"), KEY))
    fields.update({"Experiment": "EF01 · R1", "Result_ID": rid,
                   "Selector": selector, "Selected_step": step, "Bucket": "ablations",
                   "Original description / notes": '{"state":"' + "a" * 64 + '"}',
                   "Eval_scope": "EVAL_SUMMARY", "Test_aware": True})
    for field in HEADERS[3:15]:
        fields[field] = .912345678912345
    for name, value in fields.items():
        values[INDEX[name]] = value
    return values


def projection(rows=None):
    rows = [row()] if rows is None else rows
    notes = ["literal note =NOT_A_FORMULA" for _ in rows]
    return {"destination": destination("s1"), "rows": rows, "notes": notes,
            "view_hash": t.digest({"rows": rows, "notes": notes}), "source_digest": "d" * 64,
            "raw_ids": ["EGR2-old"], "owned_run_keys": [KEY], "conflicts": [],
            "raw_count": 2, "summary_count": 1, "runs": [], "selected": [], "excluded": []}


def snapshot(rows=None, notes=None):
    rows = [row("EGR2-old", "OBSERVED_1000")] if rows is None else rows
    notes = [""] * len(rows) if notes is None else notes
    cells = [[{"userEnteredValue": t._typed(v), **({"note": notes[i]} if j == 0 else {})}
              for j, v in enumerate(r)] for i, r in enumerate(rows)]
    return {"metadata": {"spreadsheetId": SPREADSHEET_ID, "sheets": [
        {"properties": {"sheetId": 261001010, "title": "WV3-ablations",
                        "gridProperties": {"rowCount": 1000, "columnCount": 64}},
         "protectedRanges": [{"range": {"sheetId": 261001010, "startRowIndex": 0, "endRowIndex": 5}}]},
        {"properties": {"sheetId": 261001012, "title": "GF2-ablations",
                        "gridProperties": {"rowCount": 1000, "columnCount": 64}}}]},
        "header": list(HEADERS), "grid_row_count": 1000,
        "rows": rows, "notes": notes, "cells": cells}


class Fake:
    def __init__(self, snap=None):
        self.data = snap or snapshot()
        self.writes = []
        self.reads = 0
        self.race = False
        self.corrupt = False
        self.uncertain = False

    def snapshot(self, dest):
        self.reads += 1
        if self.race and self.reads == 2:
            self.data["rows"][0][0] = "competing writer"
        return copy.deepcopy(self.data)

    def replace(self, dest, rows, notes, **kwargs):
        self.writes.append((dest, copy.deepcopy(rows), kwargs))
        self.data = snapshot(copy.deepcopy(rows), copy.deepcopy(notes))
        if self.corrupt:
            self.data["rows"][0][3] = str(rows[0][3])
        if self.uncertain:
            self.uncertain = False
            raise t.TransportError("simulated timeout after commit")


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_dry_run_never_mutates(self):
        adapter = Fake()
        report = t.reconcile(adapter, projection(), self.tmp.name)
        self.assertEqual(report["action"], "DRY_RUN")
        self.assertEqual(adapter.writes, [])

    def test_atomic_replace_backup_and_readback(self):
        adapter = Fake()
        report = t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertTrue(report["readback_verified"])
        self.assertEqual(len(adapter.writes), 1)
        backup = json.loads(Path(report["backup_path"]).read_text())
        self.assertEqual(t.digest(backup), report["backup_sha256"])
        self.assertEqual(backup["snapshot"]["rows"][0][59], "EGR2-old")

    def test_second_poll_skips_write(self):
        adapter = Fake()
        t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        second = t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertEqual(second["action"], "UNCHANGED_VERIFIED")
        self.assertEqual(len(adapter.writes), 1)

    def test_unknown_row_after_gap_fails_closed(self):
        adapter = Fake(snapshot([row("EGR2-old"), [""] * 64, row("EGR2-foreign")]))
        with self.assertRaisesRegex(t.TransportError, "absent"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertFalse(adapter.writes)

    def test_other_run_not_removed(self):
        other = row(); other[INDEX["Run_ID"]] = "EGR2_unknown"
        adapter = Fake(snapshot([other]))
        with self.assertRaisesRegex(t.TransportError, "foreign"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)

    def test_existing_different_view_hash_format_owned(self):
        adapter = Fake(snapshot([row("VIEW-v1-previous-hash")] ))
        report = t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertTrue(report["readback_verified"])

    def test_header_digest_must_match(self):
        adapter = Fake(); adapter.data["header"][0] = "Renamed"
        with self.assertRaisesRegex(t.TransportError, "header"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)

    def test_native_controls_all_fail_closed(self):
        for control, value in (("dataValidation", {"strict": True}),
                               ("chipRuns", [{"startIndex": 0}]),
                               ("textFormatRuns", [{"startIndex": 0}]),
                               ("userEnteredValue", {"formulaValue": "=1+1"})):
            with self.subTest(control=control):
                adapter = Fake(); adapter.data["cells"][0][20][control] = value
                with self.assertRaises(t.TransportError):
                    t.reconcile(adapter, projection(), self.tmp.name, apply=True)
                self.assertFalse(adapter.writes)

    def test_protected_data_rejected_header_preserved(self):
        adapter = Fake()
        adapter.data["metadata"]["sheets"][0]["protectedRanges"].append(
            {"range": {"sheetId": 261001010, "startRowIndex": 5, "endRowIndex": 6}})
        with self.assertRaisesRegex(t.TransportError, "Protected"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)

    def test_competing_writer_detected_before_mutation(self):
        adapter = Fake(); adapter.race = True
        with self.assertRaisesRegex(t.TransportError, "changed during"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertFalse(adapter.writes)

    def test_stringified_numeric_readback_is_not_success(self):
        adapter = Fake(); adapter.corrupt = True
        with self.assertRaisesRegex(t.TransportError, "Readback mismatch"):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertFalse((Path(self.tmp.name) / "last_verified.json").exists())
        self.assertTrue((Path(self.tmp.name) / "pending_write.json").exists())

    def test_uncertain_write_recovered_by_read_not_append(self):
        adapter = Fake(); adapter.uncertain = True
        with self.assertRaises(t.TransportError):
            t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        report = t.reconcile(adapter, projection(), self.tmp.name, apply=True)
        self.assertEqual(report["action"], "UNCHANGED_VERIFIED")
        self.assertEqual(len(adapter.writes), 1)

    def test_source_conflicts_no_network(self):
        adapter = Fake(); project = projection(); project["conflicts"] = ["bad source"]
        with self.assertRaises(t.TransportError):
            t.reconcile(adapter, project, self.tmp.name, apply=True)
        self.assertEqual(adapter.reads, 0)

    def test_readback_checks_every_metric_and_provenance(self):
        expected = [row()]
        for col in list(range(3, 15)) + [0, 28, 29, 32, 40, 41, 59, 60]:
            actual = copy.deepcopy(expected)
            actual[0][col] = "corrupted"
            self.assertFalse(t._matrix_equal(actual, expected))

    def test_typed_strings_are_raw_not_formulas(self):
        self.assertEqual(t._typed("=IMPORTXML(...)"), {"stringValue": "=IMPORTXML(...)"})
        self.assertEqual(t._typed(True), {"boolValue": True})
        with self.assertRaises(t.TransportError):
            t._typed(float("nan"))

    def test_compact_single_batch_never_shrinks_or_other_tabs(self):
        adapter = object.__new__(t.GoogleSheets)
        calls = []
        adapter._request = lambda *a, **kw: calls.append((a, kw)) or {}
        adapter.replace(destination("s1"), [row()], ["note"], previous_count=10000,
                        grid_row_count=10005, occupied_tail_rows=[1, 2, 9999])
        self.assertEqual(len(calls), 1)
        requests = calls[0][1]["json"]["requests"]
        self.assertEqual([next(iter(r)) for r in requests], ["updateCells"] * 3)
        self.assertEqual(requests[1]["updateCells"]["rows"][0]["values"][0],
                         {"userEnteredValue": {"stringValue": ""}, "note": ""})
        self.assertEqual(len(requests[1]["updateCells"]["rows"]), 2)
        self.assertEqual(requests[1]["updateCells"]["fields"], "userEnteredValue,note")
        self.assertEqual(requests[1]["updateCells"]["range"]["startRowIndex"], 6)
        self.assertEqual(requests[1]["updateCells"]["range"]["endRowIndex"], 8)
        self.assertEqual(requests[2]["updateCells"]["range"]["startRowIndex"], 10004)
        self.assertEqual(requests[2]["updateCells"]["range"]["endRowIndex"], 10005)
        for request in requests:
            body = next(iter(request.values()))
            self.assertEqual(body["range"]["sheetId"], 261001010)
            self.assertGreaterEqual(body["range"]["startRowIndex"], 5)
        self.assertLess(len(json.dumps(requests)), 20000)

    def test_clear_only_view_uses_explicit_empty_string_oneof(self):
        adapter = object.__new__(t.GoogleSheets)
        calls = []
        adapter._request = lambda *a, **kw: calls.append(kw["json"]) or {}
        adapter.replace(destination("s1"), [], [], previous_count=120, grid_row_count=1000)
        clear = calls[0]["requests"]
        self.assertEqual(len(clear), 1)
        self.assertEqual(clear[0]["updateCells"]["rows"][0]["values"][0]["userEnteredValue"], {"stringValue": ""})
        self.assertEqual(len(clear[0]["updateCells"]["rows"]), 120)
        self.assertEqual(clear[0]["updateCells"]["range"]["startRowIndex"], 5)
        self.assertEqual(clear[0]["updateCells"]["range"]["endRowIndex"], 125)

    def test_explicit_rowdata_works_when_backend_ignores_repeat_and_implicit_clears(self):
        adapter = object.__new__(t.GoogleSheets); adapter.page_rows = 400
        meta = snapshot([])["metadata"]
        grid = [[{"userEnteredValue": {"stringValue": h}} for h in HEADERS]]
        grid += [[{"userEnteredValue": t._typed(v)} for v in row("EGR2-old")]] * 3
        def request(method, url, **kw):
            if method == "GET":
                if "ranges" not in kw["params"]:
                    return meta
                return {"sheets": [{"data": [{"startRow": 4,
                        "rowData": [{"values": values} for values in grid]}]}]}
            for req in kw["json"]["requests"]:
                if "updateCells" in req:
                    body = req["updateCells"]
                    # Observed live behavior: uncovered rows are NOT cleared.
                    for i, cells in enumerate(body.get("rows", [])):
                        grid[body["range"]["startRowIndex"] - 4 + i] = cells["values"]
                elif "repeatCell" in req:
                    # All repeatCell variants were no-ops on the live backend.
                    pass
            return {}
        adapter._request = request
        self.assertEqual(len(adapter.snapshot(destination("s1"))["rows"]), 3)
        request("POST", "ignored", json={"requests": [
            {"repeatCell": {"range": {"startRowIndex": 6, "endRowIndex": 8}, "cell": {}}},
            {"repeatCell": {"range": {"startRowIndex": 6, "endRowIndex": 8},
                            "cell": {"userEnteredValue": {"stringValue": ""}, "note": ""}}},
            {"updateCells": {"range": {"startRowIndex": 6, "endRowIndex": 8}, "rows": []}}]})
        self.assertEqual(len(adapter.snapshot(destination("s1"))["rows"]), 3)
        adapter.replace(destination("s1"), [row()], ["A-note"], previous_count=3, grid_row_count=1000)
        after = adapter.snapshot(destination("s1"))
        self.assertEqual(len(after["rows"]), 1)
        self.assertEqual(after["rows"], [row()])
        self.assertEqual(after["notes"], ["A-note"])

    def test_clear_row_must_be_inside_backed_up_tail(self):
        adapter = object.__new__(t.GoogleSheets)
        calls = []
        adapter._request = lambda *a, **kw: calls.append(kw) or {}
        for bad in ([0], [12], [-1]):
            with self.assertRaises(t.TransportError):
                adapter.replace(destination("s1"), [row()], ["note"], previous_count=12,
                                grid_row_count=1000, occupied_tail_rows=bad)
        self.assertFalse(calls)

    def test_row_extension_only_when_required(self):
        adapter = object.__new__(t.GoogleSheets)
        calls = []
        adapter._request = lambda *a, **kw: calls.append(kw) or {}
        adapter.replace(destination("s1"), [row(), row("VIEW-two")], ["a", "b"], previous_count=0, grid_row_count=6)
        self.assertEqual(calls[0]["json"]["requests"][0], {
            "appendDimension": {"sheetId": 261001010, "dimension": "ROWS", "length": 1}})

    def test_grid_scan_batches_bounded_ranges_and_reads_after_gaps(self):
        adapter = object.__new__(t.GoogleSheets); adapter.page_rows = 400
        calls = []
        def request(method, url, **kw):
            calls.append(kw)
            if "ranges" not in kw["params"]:
                meta = snapshot([])["metadata"]
                meta["sheets"][0]["properties"]["gridProperties"]["rowCount"] = 10000
                return meta
            cells = [{"userEnteredValue": t._typed(v)} for v in row()]
            return {"sheets": [{"data": [
                {"startRow": 4, "rowData": [{"values": [{"userEnteredValue": {"stringValue": h}} for h in HEADERS]}]},
                {"startRow": 9999, "rowData": [{"values": cells}]}]}]}
        adapter._request = request
        snap = adapter.snapshot(destination("s1"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[1]["params"]["ranges"]), 25)
        self.assertEqual(len(snap["rows"]), 9995)
        self.assertEqual(snap["rows"][-1][59], "VIEW-fixture")

    def test_only_three_ablations_destinations(self):
        project = projection()
        project["destination"] = {"spreadsheet_id": SPREADSHEET_ID, "sheet_title": "paper", "sheet_id": 1923997092}
        with self.assertRaisesRegex(t.TransportError, "allowlist"):
            t.reconcile(Fake(), project, self.tmp.name, apply=True)

    def test_cli_offline_default_never_constructs_google_client(self):
        from . import __main__ as cli
        with patch("sheet_publish_v2.selection.project_campaign", return_value=projection()), \
                patch.object(cli, "legacy_publishers", return_value=[]), \
                patch.object(cli, "GoogleSheets") as client:
            self.assertEqual(cli.main(["--server", "s1", "--root", self.tmp.name]), 0)
            client.assert_not_called()

    def test_cli_active_legacy_blocks_apply_without_google_write(self):
        from . import __main__ as cli
        with patch("sheet_publish_v2.selection.project_campaign", return_value=projection()), \
                patch.object(cli, "legacy_publishers", return_value=[{"pid": 123, "argv": ["test"]}]), \
                patch.object(cli, "GoogleSheets") as client:
            self.assertEqual(cli.main(["--server", "s1", "--root", self.tmp.name, "--apply"]), 1)
            client.assert_not_called()

    def test_two_poll_watch_never_publishes_raw_events_and_skips_same_hash(self):
        from . import __main__ as cli
        adapter = Fake()
        with patch("sheet_publish_v2.selection.project_campaign", return_value=projection()), \
                patch.object(cli, "legacy_publishers", return_value=[]), \
                patch.object(cli, "GoogleSheets", return_value=adapter), \
                patch.object(cli.time, "sleep"):
            result = cli.main(["--server", "s1", "--root", self.tmp.name,
                               "--apply", "--watch", "--max-polls", "2"])
        self.assertEqual(result, 0)
        self.assertEqual(len(adapter.writes), 1)
        self.assertEqual(adapter.reads, 3)
        self.assertTrue(all(r[59].startswith("VIEW-") for r in adapter.data["rows"]))
        proof = json.loads((Path(self.tmp.name) / "sheet_publish_v2/last_poll.json").read_text())
        self.assertEqual(proof["action"], "UNCHANGED_LOCAL_HASH")
        self.assertFalse(proof["fresh_readback"])


if __name__ == "__main__":
    unittest.main()
