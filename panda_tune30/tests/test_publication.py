"""Offline qualified minimum-ERGAS, isolated cost, and anchored readback tests."""
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
from panda_tune30 import best_publication as B, cost as C, plan, publication as P, reporting as R
from panda_tune30.best_publication import transport as T
from panda_tune30.best_publication.restart import matches_writer
from panda_tune30.spec import geometry_manifest
from panda_tune30.contract import CASES
from sheet_publish_v2.transport import TransportError


def persist(root, case="S100DIFFUBILC30LR080", *, server="s1", replica="R1", step=1000, h=.989, e=2.03):
    cfg = plan.resolve_case(server, case, replica)
    ev = original_evaluation(cfg, step, h, e)
    key = cfg["policy_id"].lower()
    ev["mature30_deployment"] = {"schema": "PANDA_NEXT30_DEPLOYMENT_V1",
        "policy_id": cfg["policy_id"], "completed_update": step, "geometry_key": key,
        "geometry": geometry_manifest(key), "student_sampling": deepcopy(cfg["student_sampling"]),
        "student_input": deepcopy(cfg["student_input"]), "student_warp": deepcopy(cfg["student_warp"])}
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
    def test_all_cases_servers_publish_all_passes_and_one_representative(self):
        for server in ("s1", "s3"):
            with tempfile.TemporaryDirectory() as root:
                runs = [run for group in plan.groups(server) for run in group["runs"]]
                for case, replica in runs:
                    persist(root, case, server=server, replica=replica, step=1000, h=.995, e=2.039)
                    persist(root, case, server=server, replica=replica, step=2000, h=.9585, e=2.030)
                    persist(root, case, server=server, replica=replica,
                            step=plan.resolve_case(server, case, replica)["planned_horizon"], h=.9584, e=2.0)
                view = B.project(root, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(len(view["rows"]), 2 * len(runs))
                self.assertEqual(view["summary_count"], 3 * len(runs))
                representatives = []
                for row in view["rows"]:
                    if row[R.INDEX["Selector"]] == B.SELECTOR:
                        representatives.append(row)
                        self.assertEqual(row[R.INDEX["Selected_step"]], 2000 +
                            json.loads(row[R.INDEX["Original description / notes"]])["extra_parameters"]["configuration"].get("inherited_updates", 0))
                        self.assertEqual(row[R.INDEX["HQNR↑"]], .9585)
                        self.assertEqual(row[R.INDEX["ERGAS↓"]], 2.030)
                    else:
                        self.assertEqual(row[R.INDEX["Selector"]], B.PASS_SELECTOR)
                        self.assertEqual(row[R.INDEX["Selected_step"]], 1000 +
                            json.loads(row[R.INDEX["Original description / notes"]])["extra_parameters"]["configuration"].get("inherited_updates", 0))
                    for metric in ("RMSE↓", "CC↑", "JQM↑", "Q4/Q8↑"):
                        self.assertIsInstance(row[R.INDEX[metric]], (int, float))
                    B.verify_owned(row, view)
                self.assertEqual(len(representatives), len(runs))
                self.assertTrue(all(r["pass_count"] == r["kept_count"] == 2 for r in view["runs"]))


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
            note = json.loads(view["rows"][0][R.INDEX["Original description / notes"]])
            self.assertEqual(note["publication"]["locally_completed_updates"], 0)


    def test_actual_positive_correction_and_unchanged_references_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            path = next(Path(root).glob("runs/*/attempt_01/evaluation_*.json"))
            ev = json.loads(path.read_text())
            for field, value in (("pan_applied_dy_dx", [0., 0.]),
                                 ("pan_applied_dy_dx", [.15, -.2]),
                                 ("ms_applied_dy_dx", [0., .1]),
                                 ("output_applied_dy_dx", [.1, 0.])):
                changed = deepcopy(ev)
                changed["fr"]["per_scene"][0][field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S100DIFFUBILC30LR080"))
            for field, value in (("reference_warp", True), ("shift_scale", .1)):
                changed = deepcopy(ev)
                changed[field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S100DIFFUBILC30LR080"))


    def test_registered_knobs_single_view_and_geometry_cannot_be_relabelled(self):
        cfg = plan.resolve_case("s1", "S100DIFFUBILC30LR080")
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
            persist(root, step=5000, h=.995, e=2.2)
            old = B.project(root, "s1")
            persist(root, step=6000, h=.9584, e=2.0)
            record = persist(root, step=7000, h=.9585, e=2.03)
            persist(root, step=30000, h=.999, e=2.3)
            def attach(view, root, **kwargs):
                self.assertEqual([s["step"] for s in view["selected"]], [7000])
                view["rows"][0][R.INDEX["Infer(ms)"]] = 12.3
            with patch.object(C, "attach_cost", side_effect=attach):
                new = B.project(root, "s1")
            self.assertEqual(old["rows"][0][R.INDEX["Result_ID"]],
                             new["rows"][0][R.INDEX["Result_ID"]])
            self.assertEqual(new["rows"][0][3:15], record["row"][3:15])
            self.assertEqual(new["rows"][0][R.INDEX["Infer(ms)"]], 12.3)


    def test_ergas_ties_choose_higher_hqnr_then_earlier_step(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.9585, e=2.03)
            persist(root, step=2000, h=.989, e=2.03)
            persist(root, step=5000, h=.989, e=2.03)
            self.assertEqual(B.project(root, "s1")["selected"][0]["step"], 2000)


    def test_exact_inclusive_thresholds_and_below_threshold_fallback(self):
        for h, e, count, passed in ((.9585, 2.040, 1, 1), (.9585 - 1e-9, 2.040, 1, 0),
                                    (.9585, 2.040 + 1e-9, 1, 0), (.95, 2.03, 1, 0)):
            with self.subTest(h=h, e=e), tempfile.TemporaryDirectory() as root:
                persist(root, h=h, e=e)
                before = {str(p): p.read_bytes() for p in Path(root).rglob("*.json")}
                view = B.project(root, "s1")
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(len(view["rows"]), count)
                self.assertEqual(view["runs"][0]["pass_count"], passed)
                self.assertEqual(view["runs"][0]["kept_count"], count)
                self.assertEqual(view["runs"][0]["hqnr_min"], .9585)
                for path, content in before.items():
                    self.assertEqual(Path(path).read_bytes(), content)
                if count:
                    note = json.loads(view["rows"][0][R.INDEX["Original description / notes"]])
                    self.assertFalse(note["publication"]["target_thresholds_are_annotations_only"])
                    self.assertEqual(note["publication"]["publication_joint_pass"], bool(passed))
                    self.assertEqual(note["publication"]["fallback_used"], h < B.HQNR_MIN)
                    self.assertEqual(view["rows"][0][R.INDEX["Selector"]],
                                     B.FALLBACK_SELECTOR if h < B.HQNR_MIN else B.SELECTOR)
                    B.verify_owned(view["rows"][0], view)
                    self.assertTrue(note["publication"]["is_case_representative"])
                    self.assertFalse(note["publication"]["representative_requires_ergas_max"])
                    self.assertEqual(note["publication"]["ergas_max"], 2.040)
                    # Historical source config remains truthful and immutable.
                    self.assertEqual(note["extra_parameters"]["configuration"]["hqnr_min"], .9585)


    def test_representative_above_ergas_limit_is_kept_and_not_labelled_as_pass(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.999, e=2.1)
            persist(root, step=2000, h=.9585, e=2.05)
            persist(root, step=3000, h=.990, e=2.08)
            view = B.project(root, "s1")
            self.assertFalse(view["conflicts"], view["conflicts"])
            self.assertEqual([c["step"] for c in view["selected"]], [2000])
            self.assertEqual(view["runs"][0]["pass_count"], 0)
            row = view["rows"][0]
            self.assertIn("ERGAS 미달", row[R.INDEX["Selection@Step"]])
            B.verify_owned(row, view)
            bad = deepcopy(row)
            note = json.loads(bad[R.INDEX["Original description / notes"]])
            note["publication"]["selection"] = B.PASS_SELECTOR
            bad[R.INDEX["Selector"]] = B.PASS_SELECTOR
            bad[R.INDEX["Original description / notes"]] = R.canonical_json(note)
            c = view["selected"][0]
            bad[R.INDEX["Result_ID"]] = P.display_id(B.POLICY, c["key"], c["protocol"], c["step"])
            with self.assertRaisesRegex(ValueError, "thresholds"):
                B.verify_owned(bad, view)


    def test_new_representative_retains_previous_pass_without_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.989, e=2.1)
            first = B.project(root, "s1")
            representative_id = first["rows"][0][R.INDEX["Result_ID"]]
            persist(root, step=2000, h=.990, e=2.04)
            second = B.project(root, "s1")
            self.assertEqual(len(second["rows"]), 1)
            self.assertEqual(second["rows"][0][R.INDEX["Result_ID"]], representative_id)
            persist(root, step=3000, h=.9585, e=2.03)
            persist(root, step=4000, h=.999, e=2.2)
            third = B.project(root, "s1")
            self.assertEqual([c["step"] for c in third["selected"]], [3000, 2000])
            ids = [r[R.INDEX["Result_ID"]] for r in third["rows"]]
            self.assertEqual(ids[0], representative_id)
            self.assertEqual(len(set(ids)), 2)
            self.assertEqual([r[R.INDEX["Selector"]] for r in third["rows"]], [B.SELECTOR, B.PASS_SELECTOR])
            for row in third["rows"]:
                B.verify_owned(row, third)


    def test_hqnr_fallback_ties_and_transition_to_qualified_min_ergas(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.957, e=2.0)
            persist(root, step=2000, h=.958, e=2.3)
            persist(root, step=3000, h=.958, e=2.2)
            persist(root, step=4000, h=.958, e=2.2)
            old = B.project(root, "s1")
            self.assertEqual([c["step"] for c in old["selected"]], [3000])
            self.assertEqual(old["rows"][0][R.INDEX["Selector"]], B.FALLBACK_SELECTOR)
            self.assertIn("HQNR 미달", old["rows"][0][R.INDEX["Selection@Step"]])
            B.verify_owned(old["rows"][0], old)
            persist(root, step=5000, h=.9585, e=2.8)
            new = B.project(root, "s1")
            self.assertEqual([c["step"] for c in new["selected"]], [5000])
            self.assertEqual(new["rows"][0][R.INDEX["Selector"]], B.SELECTOR)
            self.assertEqual(new["rows"][0][R.INDEX["Result_ID"]], old["rows"][0][R.INDEX["Result_ID"]])
            # A previously valid fallback remains verifiable for in-place update.
            requests = B.row_requests(snapshot(new, rows=old["rows"], notes=old["notes"]), new, "s1")
            self.assertEqual(len(requests), 1)
            self.assertIn("updateCells", requests[0])


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
            persist(root, step=2000, h=.9585, e=2.02)
            persist(root, "S100DIFFUBILC25", step=1000)
            view = B.project(root, "s1")
            got = T.reconcile_rows(adapter, view, state, "s1", apply=True)
            self.assertEqual(got["row_count"], 3)
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
        args = ["python", "-m", "panda_tune30.best_publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch"]
        self.assertTrue(matches_writer(args, root, "s1"))
        self.assertFalse(matches_writer(args, root, "s3"))
        for module in ("panda_tune30", "panda_schedule30.best_publication", "panda_scale30.best_publication", "panda_mature30.best_publication"):
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
import panda_tune30.publication, panda_tune30.best_publication, panda_tune30.best_publication.transport, panda_tune30.cost
"""
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr)


    def test_fresh_lineage_rejects_continuation_binding(self):
        cfg = plan.resolve_case("s1", "S100DIFFUBILC30LR080", "R1")
        P._verify_source_binding(cfg, Path("/tmp"), set())
        for key, value in (("inherited_updates", 15000), ("initialization", "FORK_PRESERVE_COMPLETE_STATE"),
                           ("source_binding", {"calibration_preserved": True}), ("source_case", "S100DIFFC30")):
            changed = deepcopy(cfg); changed[key] = value
            with self.assertRaises(ValueError):
                P._verify_source_binding(changed, Path("/tmp"), set())

    def test_registered_width_is_bound_to_model_manifest_and_readable_sheet(self):
        from panda_tune30.contract import input_manifest, sampling_manifest
        for width, case in ((112, "S100DIFFUBILC30LR080"), (128, "S100DIFFUBILC30W128")):
            cfg = plan.resolve_case("s3", case, "R1")
            with tempfile.TemporaryDirectory() as root:
                record = persist(root, case, server="s3")
                self.assertIn(f"W{width}D123", record["row"][R.INDEX["Model / inputs"]])
                approach = record["row"][R.INDEX["Approach"]]
                for token in (f"W{width}D123", "Hard alpha=", "gamma=", "lr=", "L4 warp", "full-u (scale=1)"):
                    self.assertIn(token, approach)
                self.assertEqual(cfg["student_input"], input_manifest("CORRECTED_DIFF", width=width))
                self.assertEqual(cfg["student_input"]["additional_parameters"], 9*width)
                self.assertEqual(cfg["student_sampling"], sampling_manifest("UBIL", "CORRECTED_DIFF", width=width))
                self.assertFalse(B.project(root, "s3")["conflicts"])
            for key in ("student_architecture", "stage2_spec"):
                changed = deepcopy(cfg)
                changed[key]["width" if key == "student_architecture" else "student_width"] = 128 if width == 112 else 112
                with self.assertRaises(ValueError):
                    R._config(changed)
            changed = deepcopy(cfg)
            changed["student_input"]["additional_parameters"] += 1
            with self.assertRaises(ValueError):
                R._config(changed)

    def test_cost_signatures_and_profile_enumeration_distinguish_width(self):
        from panda_tune30.cost.profiling import registered_profiles
        controls = [plan.resolve_case("s3", case, "R1") for case in
                    ("S100DIFFUBILC30LR080", "S100DIFFUBILC30W128")]
        signatures = [C.signature(cfg, 1000) for cfg in controls]
        self.assertNotEqual(R.digest(signatures[0]), R.digest(signatures[1]))
        self.assertEqual([s["reconstructor"]["width"] for s in signatures], [112, 128])
        for sig in signatures:
            self.assertEqual(sig["inference_passes"], 1)
            self.assertEqual(sig["phase"], "stage2")
            self.assertIn("panda_tune30/reconstruction.py", sig["source_sha256"])
        widths = [cfg["student_architecture"]["width"] for cfg, step in registered_profiles("s3")
                  if cfg["phase"] == "stage2"]
        self.assertEqual(sorted(widths), [112, 128])

    def test_actual_profile_graph_uses_registered_width_and_excludes_teacher(self):
        import torch
        from panda_tune30.cost.profiling import build_deployment
        previous = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            for width, case in ((112, "S100DIFFUBILC30LR080"), (128, "S100DIFFUBILC30W128")):
                cfg = plan.resolve_case("s3", case, "R1")
                graph = build_deployment(cfg, None, 1000)
                student_count = sum(p.numel() for p in graph.inference.student.parameters())
                self.assertEqual(student_count, cfg["student_sampling"]["student_parameters"])
                self.assertEqual(student_count, {112: 2785336, 128: 3633800}[width])
                self.assertIs(graph.teacher, graph.inference.student)
                self.assertEqual(sum(p.numel() for p in graph.parameters()),
                                 student_count + sum(p.numel() for p in graph.aligner.parameters()))
        finally:
            torch.set_num_threads(previous)

    def test_profiling_preserves_30h_clock_and_closeout(self):
        from panda_tune30.contract import BUDGET
        from panda_tune30.cost.__main__ import profiling_deadline
        total, reserve = BUDGET["total_seconds"], BUDGET["closeout_seconds"]
        self.assertEqual(total, 30*3600)
        self.assertEqual(profiling_deadline({"started_unix": 1000.}, 1100., 1200), 2300.)
        near_end = 1000 + total - reserve - 10
        self.assertEqual(profiling_deadline({"started_unix": 1000.}, near_end, 1200), near_end+10)
        for now in (near_end+10, near_end+11):
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                profiling_deadline({"started_unix": 1000.}, now, 1200)
        for state in ({}, {"started_unix": float("nan")}, {"started_unix": 1200.}):
            with self.assertRaises(ValueError):
                profiling_deadline(state, 1100., 1200)

    def test_profiling_rollback_preserves_persisted_and_monotonic_elapsed(self):
        from panda_tune30.contract import BUDGET
        from panda_tune30.cost.__main__ import profiling_deadline, persist_profiling_elapsed
        limit = BUDGET["total_seconds"] - BUDGET["closeout_seconds"]
        state = {"started_unix": 1000., "elapsed_seconds": limit-10,
                 "status": "PREPARING_DEPLOYMENT_COST", "runs": []}
        self.assertEqual(profiling_deadline(state, 1100., 1200), 1110.)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"campaign_state.json"
            elapsed = persist_profiling_elapsed(path, state, elapsed_at_open=limit-10,
                monotonic_at_open=100., now=1090., monotonic_now=105.)
            self.assertEqual(elapsed, limit-5)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["started_unix"], 1000.)
            self.assertEqual(saved["status"], "PREPARING_DEPLOYMENT_COST")
            self.assertEqual(profiling_deadline(saved, 1090., 1200), 1095.)
            persist_profiling_elapsed(path, state, elapsed_at_open=limit-10,
                monotonic_at_open=100., now=1080., monotonic_now=110.)
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                profiling_deadline(json.loads(path.read_text()), 1080., 1200)
        for bad in (-1., float("nan"), float("inf"), True, "1"):
            with self.assertRaisesRegex(ValueError, "Invalid durable elapsed"):
                profiling_deadline({"started_unix": 1000., "elapsed_seconds": bad}, 1100., 1200)

    def test_budget_terminal_status_is_read_back_before_watcher_exit(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            state = {"status": "FINISHED_WITH_BUDGET_SKIPS", "runs": []}
            R.atomic_json(Path(root)/"campaign_state.json", state)
            value = B.project(root, "s1")
            self.assertEqual(P._terminal_state(root), state)
            self.assertTrue(P.terminal_upload_complete(root, value, {"readback_verified": True}, state))
            self.assertFalse(P.terminal_upload_complete(root, value, {"readback_verified": False}, state))
            R.atomic_json(Path(root)/"campaign_state.json", {"status": "RUNNING"})
            self.assertFalse(P.terminal_upload_complete(root, value, {"readback_verified": True}, state))


if __name__ == "__main__":
    unittest.main()
