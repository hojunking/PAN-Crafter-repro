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
from panda_scale30 import best_publication as B, cost as C, plan, publication as P, reporting as R
from panda_scale30.best_publication import transport as T
from panda_scale30.best_publication.restart import matches_writer
from panda_scale30.spec import geometry_manifest
from panda_scale30.contract import CASES
from sheet_publish_v2.transport import TransportError
from experiment_cost.test_store import measurement


def persist(root, case="S020BASE", *, server="s1", replica="R1", step=1000, h=.95, e=2.1):
    cfg = plan.resolve_case(server, case, replica)
    ev = original_evaluation(cfg, step, h, e)
    key = cfg["policy_id"].lower()
    ev["mature30_deployment"] = {"schema": "PANDA_SCALE30_DEPLOYMENT_V1",
        "policy_id": cfg["policy_id"], "completed_update": step, "geometry_key": key,
        "geometry": geometry_manifest(key)}
    for split in ("rr", "fr"):
        for scene in ev[split]["per_scene"]:
            scene["correction_dy_dx"] = [.3, -.4]
            scene["pan_applied_dy_dx"] = [.3 * cfg["shift_scale"], -.4 * cfg["shift_scale"]]
    directory = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory / "config.json", cfg)
    R.atomic_json(directory / f"evaluation_{step:07d}.json", ev)
    return R.emit_evaluation(root, cfg, ev, timestamp="2026-10-22T00:00:00+00:00")


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
                runs = [run for group in plan.groups(server) for run in group["runs"]]
                for case, replica in runs:
                    persist(root, case, server=server, replica=replica, step=1000, h=.965, e=2.2)
                    persist(root, case, server=server, replica=replica, step=2000, h=.960, e=2.04)
                    persist(root, case, server=server, replica=replica, step=50000, h=.961, e=2.03)
                view = B.project(root, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(len(view["rows"]), len(runs))
                self.assertEqual(view["summary_count"], 3 * len(runs))
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

    def test_actual_positive_correction_and_unchanged_references_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            path = next(Path(root).glob("runs/*/attempt_01/evaluation_*.json"))
            ev = json.loads(path.read_text())
            for field, value in (("pan_applied_dy_dx", [0., 0.]),
                                 ("pan_applied_dy_dx", [.3, -.4]),
                                 ("ms_applied_dy_dx", [0., .1]),
                                 ("output_applied_dy_dx", [.1, 0.])):
                changed = deepcopy(ev)
                changed["fr"]["per_scene"][0][field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S020BASE"))
            for field, value in (("reference_warp", True), ("shift_scale", .1)):
                changed = deepcopy(ev)
                changed[field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S020BASE"))

    def test_registered_knobs_single_view_and_geometry_cannot_be_relabelled(self):
        cfg = plan.resolve_case("s1", "S020BASE")
        for key, value in (("rotation_consistency_weight", .05), ("student_views", 2),
                           ("pair_transform", "EXACT_ROT90_CCW"), ("shift_scale", 0.),
                           ("lr_S", 9e-5), ("lr_U", 9e-5), ("policy_id", "ZERO"),
                           ("inference_tta", True), ("tuning_parameters", {})):
            changed = deepcopy(cfg)
            changed[key] = value
            with self.assertRaises(ValueError, msg=key):
                R._config(changed)
        changed = deepcopy(cfg)
        changed["training_spec"]["lr"] = 9e-5
        with self.assertRaises(ValueError):
            R._config(changed)
        for key, value in (("alpha", .125), ("gamma", 1.), ("beta", .01)):
            changed = deepcopy(cfg)
            changed["stage2_spec"][key] = value
            with self.assertRaises(ValueError, msg=key):
                R._config(changed)

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

    def test_cost_signature_binds_actual_scale_but_not_training_knobs(self):
        cfg = plan.resolve_case("s1", "S020BASE")
        sig = C.signature(cfg, 1000)
        for case, replica in [run for group in plan.groups("s1") for run in group["runs"]]:
            current = plan.resolve_case("s1", case, replica)
            if CASES[case]["scale"] == .2:
                self.assertEqual(sig, C.signature(current, 50000))
            else:
                self.assertNotEqual(sig, C.signature(current, 50000))
        self.assertNotEqual(sig, C.signature(plan.resolve_case("s3", "S020BASE"), 1000))
        self.assertNotEqual(sig, C.signature(plan.resolve_case("s1", plan.DONORS[0]), 50000))
        value = measurement("s1")
        value["gpu_isolation_assertion"].update(
            mode="STANDALONE_ISOLATED_NO_TRAINER_GPU", training_state_retained_in_memory=False)
        value["deployment_signature_sha256"] = R.digest(sig)
        value["deployment"] = {"phase": "stage2", "geometry_key": "s020", "pan_shift_scale": .2,
            "pan_execution": sig["pan_execution"], "training_teacher_included": False, "graph": "A+S"}
        for split in ("rr", "fr"):
            value["observations"][split]["full_pipeline"] = {"samples_ms": [10.] * 10}
        with tempfile.TemporaryDirectory() as root:
            saved = C.save_profile(cfg, 1000, root, value)
            self.assertEqual(saved, C.find_profile(cfg, 50000, root))
            bad = deepcopy(value)
            bad["optimizer_updates"] = 1
            with self.assertRaises(ValueError):
                C.validate_measurement(bad, sig)
            bad = deepcopy(value)
            bad["deployment"]["pan_shift_scale"] = 0.
            with self.assertRaises(ValueError):
                C.validate_measurement(bad, sig)

    def test_profile_registry_covers_all_deployed_scales_once_per_server(self):
        from panda_scale30.cost.profiling import registered_profiles
        for server in ("s1", "s3"):
            profiles = registered_profiles(server)
            policies = [cfg["policy_id"] for cfg, _ in profiles if cfg["phase"] == "stage2"]
            cases = [case for group in plan.groups(server) for case, _ in group["runs"]]
            self.assertEqual(set(policies), {CASES[case]["policy_id"] for case in cases})
            self.assertEqual(len(policies), len(set(policies)))
            self.assertEqual(sum(cfg["phase"] == "donor" for cfg, _ in profiles), 1)

    def test_profiler_builds_actual_positive_warp_and_single_student_without_teacher(self):
        import torch
        from pa.warp import warp_pan
        from panda_stage2_condq.tests.test_method import batch
        from panda_scale30.cost.profiling import build_deployment, registered_profiles
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cfg, step in registered_profiles("s1"):
                if cfg["phase"] != "stage2":
                    continue
                graph = build_deployment(cfg, None, step)
                self.assertEqual(set(graph.inference._modules), {"aligner", "student"})
                with torch.no_grad():
                    graph.aligner.fc2.bias.copy_(torch.tensor([1., -.5]))
                rr = batch()
                with patch("panda_scale30.method.warp_pan", wraps=warp_pan) as warp, \
                     patch.object(graph.teacher.backbone, "forward",
                                  wraps=graph.teacher.backbone.forward) as student:
                    out = graph.native_prediction(rr)
                self.assertEqual(warp.call_count, 1)
                self.assertEqual(student.call_count, 1)
                torch.testing.assert_close(out["pan_correction"], cfg["shift_scale"] * out["u"],
                                           rtol=0, atol=0)
                torch.testing.assert_close(out["pan_aligned"], warp_pan(rr.pan, out["pan_correction"]),
                                           rtol=0, atol=0)
                self.assertGreater(int(torch.count_nonzero(out["pan_correction"])), 0)
        finally:
            torch.set_num_threads(old_threads)

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
            persist(root, "S010BASE", step=1000)
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
        args = ["python", "-m", "panda_scale30.best_publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch"]
        self.assertTrue(matches_writer(args, root, "s1"))
        self.assertFalse(matches_writer(args, root, "s3"))
        for module in ("panda_scale30", "panda_mature30.best_publication"):
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
import panda_scale30.publication, panda_scale30.best_publication, panda_scale30.transport, panda_scale30.cost
"""
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr)


if __name__ == "__main__":
    unittest.main()
