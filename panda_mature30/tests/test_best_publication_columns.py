"""Expanded native tabs must not block A:BL or weaken ownership checks."""
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from panda_mature30 import best_publication as B, reporting as R
from panda_mature30.best_publication import transport as T
from sheet_publish_v2.transport import TransportError
from .test_reporting import config, persist


def snapshot(value, *, width=83, rows=None, notes=None):
    dest = value["destination"]
    return {"metadata": {"spreadsheetId": dest["spreadsheet_id"], "sheets": [{
        "properties": {"sheetId": dest["sheet_id"], "title": dest["sheet_title"],
                       "gridProperties": {"columnCount": width, "rowCount": 10000}}}]},
        "header": list(R.HEADERS), "grid_row_count": 10000,
        "rows": rows or [], "notes": notes or [], "cells": []}


class ExpandedColumnTests(unittest.TestCase):
    def test_extra_columns_allowed_without_mutating_native_metadata(self):
        for server in ("s1", "s3"):
            with tempfile.TemporaryDirectory() as root:
                persist(root, config(server))
                value = B.project(root, server)
                for width in (64, 65, 83, 200):
                    with self.subTest(server=server, width=width):
                        snap = snapshot(value, width=width)
                        before = deepcopy(snap)
                        requests, additions = T.prepare(snap, value, server)
                        self.assertEqual(snap, before)
                        self.assertEqual(len(additions), 1)
                        self.assertEqual(len(additions[0]["values"]), 64)

    def test_truncated_shifted_and_wrong_workbook_headers_still_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            value = B.project(root, "s1")
            for kind in ("width", "swap", "insert", "workbook", "wide_row", "wide_cells"):
                snap = snapshot(value)
                if kind == "width":
                    snap["metadata"]["sheets"][0]["properties"]["gridProperties"]["columnCount"] = 63
                if kind == "swap": snap["header"][3:5] = snap["header"][3:5][::-1]
                if kind == "insert": snap["header"].insert(3, "New column")
                if kind == "workbook": snap["metadata"]["spreadsheetId"] = "OTHER"
                if kind == "wide_row": snap["rows"] = [[""] * 83]
                if kind == "wide_cells": snap["cells"] = [[{}] * 83]
                with self.subTest(kind=kind), self.assertRaises(TransportError):
                    T.prepare(snap, value, "s1")

    def test_protection_and_cell_safety_not_disabled_by_extra_columns(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            value = B.project(root, "s1")
            for kind in ("protectedRanges", "merges", "tables", "formula", "validation", "chip"):
                snap = snapshot(value)
                tab = snap["metadata"]["sheets"][0]
                bounds = {"sheetId": value["destination"]["sheet_id"], "startRowIndex": 5,
                          "endRowIndex": 6, "startColumnIndex": 63, "endColumnIndex": 65}
                if kind in ("protectedRanges", "tables"): tab[kind] = [{"range": bounds}]
                if kind == "merges": tab[kind] = [bounds]
                if kind == "formula": snap["cells"] = [[{"userEnteredValue": {"formulaValue": "=1"}}]]
                if kind == "validation": snap["cells"] = [[{"dataValidation": {"strict": True}}]]
                if kind == "chip": snap["cells"] = [[{"chipRuns": [{"startIndex": 0}]}]]
                with self.subTest(kind=kind), self.assertRaises(TransportError):
                    T.prepare(snap, value, "s1")
            # Protection beyond BL is unrelated and must be preserved, not removed.
            snap = snapshot(value)
            snap["metadata"]["sheets"][0]["protectedRanges"] = [{"range": {
                "sheetId": value["destination"]["sheet_id"], "startRowIndex": 5,
                "endRowIndex": 100, "startColumnIndex": 64, "endColumnIndex": 83}}]
            before = deepcopy(snap)
            T.prepare(snap, value, "s1")
            self.assertEqual(snap, before)

    def test_83_column_backfill_and_repoll_preserve_foreign_and_right_cells(self):
        class Adapter:
            BASE = "https://example.invalid/"

            def __init__(self, data):
                self.data = deepcopy(data)
                self.right_cells = {i: ["manual", "=1+2"] for i in range(12)}
                self.writes = []

            def snapshot(self, dest):
                return deepcopy(self.data)

            def _request(self, method, url, **kwargs):
                self.writes.append(url)
                if url.endswith(":append"):
                    assert unquote(url).endswith("'WV3-ablations'!A5:A10000:append")
                    assert kwargs["params"]["insertDataOption"] == "OVERWRITE"
                    rows = kwargs["json"]["values"]
                    assert all(len(r) == 64 for r in rows)
                    self.data["rows"].extend(rows)
                    self.data["notes"].extend([""] * len(rows))
                    return {"updates": {"updatedRange": "'WV3-ablations'!A8:BL8",
                                        "updatedRows": 1, "updatedColumns": 64, "updatedCells": 64}}
                for req in kwargs["json"]["requests"]:
                    assert list(req) == ["updateCells"]
                    update = req["updateCells"]
                    assert update["range"]["startColumnIndex"] == 0
                    assert update["range"]["endColumnIndex"] == 64
                    assert update["fields"] == "userEnteredValue,note"
                    index = update["range"]["startRowIndex"] - 5
                    cells = update["rows"][0]["values"]
                    self.data["rows"][index] = [next(iter(c.get("userEnteredValue", {}).values()), "") for c in cells]
                    self.data["notes"][index] = cells[0].get("note", "")
                return {}

        with tempfile.TemporaryDirectory() as root:
            persist(root, step=5000, h=.956)
            before = B.project(root, "s1")
            foreign = [""] * 64
            foreign[0], foreign[R.INDEX["Server"]] = "foreign run", "s3"
            adapter = Adapter(snapshot(before, rows=[foreign, *before["rows"]],
                                       notes=["foreign note", *before["notes"]]))
            right = deepcopy(adapter.right_cells)
            persist(root, step=7000, h=.957)
            persist(root, config(policy="DOWN"), 17000, h=.9604)
            value = B.project(root, "s1")
            state = Path(root) / "transport"
            state.mkdir()
            receipt = T.reconcile_rows(adapter, value, state, "s1", apply=True)
            self.assertTrue(receipt["readback_verified"])
            self.assertEqual(receipt["row_count"], 2)
            self.assertEqual(adapter.data["rows"][0], foreign)
            self.assertEqual(adapter.data["notes"][0], "foreign note")
            self.assertEqual(adapter.right_cells, right)
            self.assertEqual(adapter.data["metadata"]["sheets"][0]["properties"]["gridProperties"]["columnCount"], 83)
            calls = len(adapter.writes)
            T.reconcile_rows(adapter, value, state, "s1", apply=True)
            self.assertEqual(len(adapter.writes), calls)
            self.assertEqual(len(list(state.glob("cloud_backup_*.json"))), 1)

    def test_append_response_rejects_shift_partial_and_occupied_row(self):
        dest = {"sheet_title": "WV3-ablations"}
        response = {"updates": {"updatedRange": "'WV3-ablations'!A135:BL137",
                                "updatedRows": 3, "updatedColumns": 64, "updatedCells": 192}}
        T._verify_append_response(response, dest, 3, 129)
        for key, value in (("updatedRange", "'WV3-ablations'!T135:CE137"),
                           ("updatedRange", "'WV3-ablations'!A134:BL136"),
                           ("updatedRange", "'other'!A135:BL137"),
                           ("updatedColumns", 63), ("updatedRows", 2), ("updatedCells", 191)):
            bad = deepcopy(response); bad["updates"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(TransportError):
                T._verify_append_response(bad, dest, 3, 129)

    def test_unanchored_foreign_row_blocks_append_without_writing(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            value = B.project(root, "s1")
            foreign = [""] * 64; foreign[19] = "unanchored manual content"
            snap = snapshot(value, rows=[foreign])
            with self.assertRaisesRegex(ValueError, "Unanchored"):
                T.prepare(snap, value, "s1")

    def test_explicit_restart_reloads_best_writer_but_default_is_idempotent(self):
        module = importlib.import_module("panda_mature30.best_publication.restart")
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "gspread").mkdir()
            (repo / "gspread/server.txt").write_text("s3")
            root = repo / "work_dir/panda_mature30" / R.CAMPAIGN_ID / "s3/WV3"
            root.mkdir(parents=True)
            persist(root, config("s3"))
            value = B.project(root, "s3")
            adapter = SimpleNamespace(snapshot=lambda dest: snapshot(value))
            argv = ["python", "-m", module.MODULE, "--root", str(root), "--server", "s3", "--apply", "--watch"]
            with patch.object(module.legacy, "ROOT", repo), \
                 patch.object(module, "project", return_value=value), \
                 patch("sheet_publish_v2.transport.GoogleSheets", return_value=adapter), \
                 patch.object(module, "find_writers", return_value=[(123, argv)]), \
                 patch.object(module, "stop_writer") as stop, \
                 patch.object(module.subprocess, "Popen", return_value=SimpleNamespace(pid=456)) as popen:
                module.restart("s3")
                stop.assert_not_called(); popen.assert_not_called()
                module.restart("s3", replace_running=True)
                stop.assert_called_once_with(123, argv, root, "s3")
                popen.assert_called_once()
                stop.reset_mock(); popen.reset_mock()
                bad = snapshot(value); bad["header"][3] = "Unexpected column"
                adapter.snapshot = lambda dest: bad
                with self.assertRaises(TransportError):
                    module.restart("s3", replace_running=True)
                stop.assert_not_called(); popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
