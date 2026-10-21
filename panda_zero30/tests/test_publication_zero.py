"""Offline same-state best-HQNR, isolated cost, and anchored readback tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from panda_band30.tests.test_reporting import evaluation as original_evaluation
from panda_mature30.tests.test_reporting import reused_receipt
from panda_zero30 import best_publication as B, cost as C, plan, publication as P, reporting as R
from panda_zero30.best_publication import transport as T
from panda_zero30.best_publication.restart import matches_writer
from panda_zero30.spec import geometry_manifest
from sheet_publish_v2.transport import TransportError
from experiment_cost.test_store import measurement


def persist(root, case="ZERO", *, server="s1", replica="R1", step=1000, h=.95, e=2.1):
    cfg = plan.resolve_case(server, case, replica)
    ev = original_evaluation(cfg, step, h, e)
    ev["mature30_deployment"] = {"schema": "PANDA_ZERO30_DEPLOYMENT_V1",
        "policy_id": case, "completed_update": step, "geometry_key": "s000",
        "geometry": geometry_manifest("s000")}
    # A nonzero diagnostic u must never become an applied Student PAN shift.
    for split in ("rr", "fr"):
        for scene in ev[split]["per_scene"]:
            scene["correction_dy_dx"] = [.3, -.4]
    directory = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory / "config.json", cfg)
    R.atomic_json(directory / f"evaluation_{step:07d}.json", ev)
    return R.emit_evaluation(root, cfg, ev, timestamp="2026-10-21T00:00:00+00:00")


def snapshot(value, rows=None, notes=None):
    dest = value["destination"]
    return {"metadata": {"spreadsheetId": dest["spreadsheet_id"], "sheets": [{
        "properties": {"sheetId": dest["sheet_id"], "title": dest["sheet_title"],
                       "gridProperties": {"columnCount": 83, "rowCount": 10000}}}]},
        "header": list(R.HEADERS), "grid_row_count": 10000,
        "rows": rows or [], "notes": notes or [], "cells": []}


class PublicationTests(unittest.TestCase):
    def test_all_policies_servers_and_metrics_single_best_not_target_or_final(self):
        for server in ("s1", "s3"):
            with tempfile.TemporaryDirectory() as root:
                for case in ("ZERO", "PAIR0", "ROTCONS"):
                    persist(root, case, server=server, step=1000, h=.965, e=2.2)
                    persist(root, case, server=server, step=2000, h=.960, e=2.04)
                    persist(root, case, server=server, step=50000, h=.961, e=2.03)
                view = B.project(root, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(len(view["rows"]), 3)
                self.assertEqual(view["summary_count"], 9)
                for row in view["rows"]:
                    self.assertEqual(row[R.INDEX["Selected_step"]], 1000)
                    self.assertEqual(row[R.INDEX["Selector"]], "BEST_HQNR")
                    self.assertEqual(row[R.INDEX["HQNR↑"]], .965)
                    self.assertEqual(row[R.INDEX["ERGAS↓"]], 2.2)
                    for metric in ("RMSE↓", "CC↑", "JQM↑", "Q4/Q8↑"):
                        self.assertIsInstance(row[R.INDEX[metric]], (int, float))
                    B.verify_owned(row, view)

    def test_reused_donor_original_receipt_and_metrics_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            receipt, ev = reused_receipt(Path(root) / "old")
            before = deepcopy((receipt, ev))
            cfg = plan.resolve_case("s1", plan.DONORS[0])
            R.emit_evaluation(root, cfg, ev, selector_receipt=receipt)
            view = B.project(root, "s1")
            self.assertFalse(view["conflicts"], view["conflicts"])
            self.assertEqual((receipt, ev), before)
            self.assertEqual(view["rows"][0][R.INDEX["Selector"]], "BEST_VAL_RRE")
            self.assertEqual(view["rows"][0][R.INDEX["Updates"]], 0)

    def test_zero_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            path = next(Path(root).glob("runs/*/attempt_01/evaluation_*.json"))
            ev = json.loads(path.read_text())
            for field, value in (("pan_applied_dy_dx", [.1, 0.]),
                                 ("ms_applied_dy_dx", [0., .1]),
                                 ("output_applied_dy_dx", [.1, 0.])):
                changed = deepcopy(ev)
                changed["fr"]["per_scene"][0][field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "ZERO"))

    def test_running_best_same_id_all_metrics_and_cost_from_selected_step(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=5000, h=.956, e=2.2)
            old = B.project(root, "s1")
            persist(root, step=6000, h=.955, e=2.0)
            record = persist(root, step=7000, h=.957, e=2.3)
            persist(root, step=50000, h=.954, e=2.03)
            def attach(view, root, **kwargs):
                self.assertEqual([s["step"] for s in view["selected"]], [7000])
                view["rows"][0][R.INDEX["Infer(ms)"]] = 12.3
            with patch.object(C, "attach_cost", side_effect=attach):
                new = B.project(root, "s1")
            self.assertEqual(old["rows"][0][R.INDEX["Result_ID"]],
                             new["rows"][0][R.INDEX["Result_ID"]])
            self.assertEqual(new["rows"][0][3:15], record["row"][3:15])
            self.assertEqual(new["rows"][0][R.INDEX["Infer(ms)"]], 12.3)

    def test_ties_ergas_then_earlier_step(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.95, e=2.2)
            persist(root, step=2000, h=.95, e=2.1)
            persist(root, step=5000, h=.95, e=2.1)
            self.assertEqual(B.project(root, "s1")["selected"][0]["step"], 2000)

    def test_cost_signature_shared_single_inference_but_not_server_or_donor(self):
        cfg = plan.resolve_case("s1", "ZERO")
        sig = C.signature(cfg, 1000)
        for case in ("ZERO", "PAIR0", "ROTCONS"):
            self.assertEqual(sig, C.signature(plan.resolve_case("s1", case, "R2"), 50000))
        self.assertNotEqual(sig, C.signature(plan.resolve_case("s3", "ZERO"), 1000))
        self.assertNotEqual(sig, C.signature(plan.resolve_case("s1", plan.DONORS[0]), 50000))
        value = measurement("s1")
        value["gpu_isolation_assertion"].update(
            mode="STANDALONE_ISOLATED_NO_TRAINER_GPU", training_state_retained_in_memory=False)
        value["deployment_signature_sha256"] = R.digest(sig)
        for split in ("rr", "fr"):
            value["observations"][split]["full_pipeline"] = {"samples_ms": [10.] * 10}
        with tempfile.TemporaryDirectory() as root:
            saved = C.save_profile(cfg, 1000, root, value)
            self.assertEqual(saved, C.find_profile(plan.resolve_case("s1", "ROTCONS"), 50000, root))
            bad = deepcopy(value)
            bad["optimizer_updates"] = 1
            with self.assertRaises(ValueError):
                C.validate_measurement(bad, sig)

    def test_anchored_append_readback_updates_and_preserves_foreign_rows(self):
        class Adapter:
            BASE = "https://example.invalid/"
            def __init__(self, data):
                self.data, self.writes = deepcopy(data), []
            def snapshot(self, dest):
                return deepcopy(self.data)
            def _request(self, method, url, **kwargs):
                self.writes.append(url)
                if url.endswith(":append"):
                    assert unquote(url).endswith("'WV3-ablations'!A5:A10000:append")
                    rows = kwargs["json"]["values"]
                    start = 6 + len(self.data["rows"])
                    self.data["rows"].extend(rows)
                    self.data["notes"].extend([""] * len(rows))
                    return {"updates": {"updatedRange": f"'WV3-ablations'!A{start}:BL{start+len(rows)-1}",
                        "updatedRows": len(rows), "updatedColumns": 64, "updatedCells": 64 * len(rows)}}
                for request in kwargs["json"]["requests"]:
                    update = request["updateCells"]
                    assert update["range"]["startColumnIndex"] == 0
                    assert update["range"]["endColumnIndex"] == 64
                    index = update["range"]["startRowIndex"] - 5
                    cells = update["rows"][0]["values"]
                    self.data["rows"][index] = [next(iter(c.get("userEnteredValue", {}).values()), "") for c in cells]
                    self.data["notes"][index] = cells[0].get("note", "")
                return {}
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            first = B.project(root, "s1")
            foreign = [""] * 64
            foreign[0], foreign[R.INDEX["Server"]] = "previous campaign", "s3"
            adapter = Adapter(snapshot(first, rows=[foreign], notes=["foreign note"]))
            state = Path(root) / "transport"
            state.mkdir()
            got = T.reconcile_rows(adapter, first, state, "s1", apply=True)
            self.assertTrue(got["readback_verified"])
            persist(root, step=2000, h=.96)
            persist(root, "PAIR0", step=1000)
            view = B.project(root, "s1")
            got = T.reconcile_rows(adapter, view, state, "s1", apply=True)
            self.assertEqual(got["row_count"], 2)
            self.assertEqual(adapter.data["rows"][0], foreign)
            self.assertEqual(adapter.data["notes"][0], "foreign note")
            before = len(adapter.writes)
            T.reconcile_rows(adapter, view, state, "s1", apply=True)
            self.assertEqual(len(adapter.writes), before)
            self.assertEqual(adapter.data["metadata"]["sheets"][0]["properties"]["gridProperties"]["columnCount"], 83)

    def test_shifted_append_response_wrong_headers_unanchored_rows_rejected(self):
        dest = {"sheet_title": "WV3-ablations"}
        bad = {"updates": {"updatedRange": "'WV3-ablations'!T135:CE135",
            "updatedRows": 1, "updatedColumns": 64, "updatedCells": 64}}
        with self.assertRaises(TransportError):
            T._verify_append_response(bad, dest, 1, 129)
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            view = B.project(root, "s1")
            snap = snapshot(view)
            snap["header"][3] = "wrong"
            with self.assertRaises(TransportError):
                T.prepare(snap, view, "s1")
            foreign = [""] * 64
            foreign[19] = "unanchored"
            with self.assertRaisesRegex(ValueError, "Unanchored"):
                T.prepare(snapshot(view, rows=[foreign]), view, "s1")

    def test_publisher_restart_never_matches_old_campaign_or_training(self):
        root = Path("/tmp/campaign")
        args = ["python", "-m", "panda_zero30.best_publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch"]
        self.assertTrue(matches_writer(args, root, "s1"))
        self.assertFalse(matches_writer(args, root, "s3"))
        for module in ("panda_zero30", "panda_mature30.best_publication"):
            wrong = list(args); wrong[2] = module
            self.assertFalse(matches_writer(wrong, root, "s1"))

    def test_host_publisher_and_cost_do_not_import_training_dependencies(self):
        code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'h5py', 'google'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
import panda_zero30.publication, panda_zero30.best_publication, panda_zero30.transport, panda_zero30.cost
"""
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr)


if __name__ == "__main__":
    unittest.main()
