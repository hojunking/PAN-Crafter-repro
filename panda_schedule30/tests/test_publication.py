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
from panda_schedule30 import best_publication as B, cost as C, plan, publication as P, reporting as R
from panda_schedule30.best_publication import transport as T
from panda_schedule30.best_publication.restart import matches_writer
from panda_schedule30.spec import geometry_manifest
from panda_schedule30.contract import CASES
from sheet_publish_v2.transport import TransportError
from experiment_cost.test_store import measurement


def bind_source(root, cfg):
    """Small host-only source evidence fixture; no real training checkpoint."""
    cfg = deepcopy(cfg)
    source = Path(root) / cfg["source_checkpoint_relative_path"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic source file used only by offline publication tests")
    run_id = (f"PS30_WV3_{cfg['server'].upper()}_{cfg['source_case']}_"
              f"NOSOFT_{cfg['replica']}_S{cfg['seed']}")
    campaign = "PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1"
    sha = P._file_sha(source)
    metadata = {"config": {"run_id": run_id, "server": cfg["server"], "seed": cfg["seed"],
                "case": cfg["source_case"], "campaign_id": campaign},
                "artifact": {"file_sha256": sha, "completed_update": 25000,
                             "model_state_sha256": "a" * 64},
                "code_pin": {"fixture": "code"}, "evaluator_pin": {"fixture": "eval"}}
    R.atomic_json(source.with_suffix(".json"), metadata)
    cfg["source_binding"] = {
        "schema": "PANDA_SCHEDULE30_EXPLICIT_SOURCE25K_TRANSFER_V1",
        "source_run_id": run_id, "source_campaign_id": campaign,
        "source_server": cfg["server"], "source_seed": cfg["seed"],
        "source_completed_update": 25000, "target_absolute_horizon": cfg["planned_horizon"],
        "new_updates": cfg["planned_horizon"] - 25000,
        "source_checkpoint": str(source.resolve()), "source_sidecar": str(source.with_suffix(".json").resolve()),
        "source_file_sha256": sha, "source_model_state_sha256": "a" * 64,
        "student_state_sha256": "b" * 64,
        "source_code_pin_sha256": R.digest(metadata["code_pin"]),
        "source_evaluator_pin_sha256": R.digest(metadata["evaluator_pin"]),
        "optimizer_preserved": True, "optimizer_step_preserved": 25000,
        "rng_preserved": True, "data_stream_preserved": True,
        "source_optimizer_and_rng_validated_by_unchanged_loader": True,
    }
    return cfg


def persist(root, case="S020C30", *, server="s1", replica="R1", step=1000, h=.989, e=2.03):
    cfg = plan.resolve_case(server, case, replica)
    if cfg.get("inherited_updates"):
        cfg = bind_source(root, cfg)
        if step <= cfg["inherited_updates"]:
            step += cfg["inherited_updates"]
    ev = original_evaluation(cfg, step, h, e)
    key = cfg["policy_id"].lower()
    ev["mature30_deployment"] = {"schema": "PANDA_SCHEDULE30_DEPLOYMENT_V1",
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
    def test_per_case_horizon_and_continuation_update_accounting(self):
        cases = (("S020C30", "R1", 30000, 30000),
                 ("S020C35", "R1", 35000, 35000),
                 ("S020BASE", "R2", 50000, 50000),
                 ("S020F25C35", "R1", 35000, 10000))
        for case, replica, horizon, local in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                persist(root, case, replica=replica, step=horizon, h=.9585, e=2.04)
                view = B.project(root, "s1")
                self.assertFalse(view["conflicts"], view["conflicts"])
                row = view["rows"][0]
                self.assertEqual(row[R.INDEX["Selected_step"]], horizon)
                self.assertEqual(row[R.INDEX["Updates"]], local)
                self.assertEqual(row[R.INDEX["Lifetime_updates"]], horizon)
                note = json.loads(row[R.INDEX["Original description / notes"]])
                pub = note["publication"]
                self.assertTrue(pub["final_horizon"])
                self.assertEqual(pub["planned_horizon"], horizon)
                self.assertEqual(pub["locally_completed_updates"], local)
                self.assertEqual(pub["inherited_updates"], horizon-local)
                self.assertNotIn("final50k", pub)

    def test_continuation_source_is_required_and_changed_file_blocks_publication(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = bind_source(root, plan.resolve_case("s1", "S020F25C35"))
            P._verify_source_binding(cfg, root, set())
            changed = deepcopy(cfg)
            changed.pop("source_binding")
            with self.assertRaisesRegex(ValueError, "SOURCE_BINDING"):
                P._verify_source_binding(changed, root, set())
            for field, bad in (("source_server", "s3"), ("optimizer_preserved", False),
                               ("source_completed_update", 24000), ("rng_preserved", False)):
                changed = deepcopy(cfg)
                changed["source_binding"][field] = bad
                with self.assertRaisesRegex(ValueError, "SOURCE_BINDING"):
                    P._verify_source_binding(changed, root, set())
            persist(root, "S020F25C35", step=35000)
            source = Path(cfg["source_binding"]["source_checkpoint"])
            source.write_bytes(b"changed source bytes")
            view = B.project(root, "s1")
            self.assertTrue(any(c["reason"] == "CONTINUATION_SOURCE_FILE_CHANGED"
                                for c in view["conflicts"]))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                B.row_requests(snapshot(view), view, "s1")

    def test_historical_scale_campaign_rows_are_never_adopted_or_updated(self):
        from panda_scale30 import best_publication as old_best
        from panda_scale30.tests.test_publication_scale import persist as old_persist
        with tempfile.TemporaryDirectory() as root:
            old_root, new_root = Path(root)/"old", Path(root)/"new"
            old_root.mkdir(); new_root.mkdir()
            old_persist(old_root)
            old = old_best.project(old_root, "s1")
            persist(new_root)
            new = B.project(new_root, "s1")
            requests = B.row_requests(snapshot(new, old["rows"], old["notes"]), new, "s1")
            self.assertEqual(len(requests), 1)
            self.assertIn("appendCells", requests[0])
            self.assertNotEqual(old["rows"][0][R.INDEX["Result_ID"]],
                                new["rows"][0][R.INDEX["Result_ID"]])

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
                                 ("pan_applied_dy_dx", [.3, -.4]),
                                 ("ms_applied_dy_dx", [0., .1]),
                                 ("output_applied_dy_dx", [.1, 0.])):
                changed = deepcopy(ev)
                changed["fr"]["per_scene"][0][field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S020C30"))
            for field, value in (("reference_warp", True), ("shift_scale", .1)):
                changed = deepcopy(ev)
                changed[field] = value
                with self.assertRaises(ValueError):
                    P._verify_corrections(changed, plan.resolve_case("s1", "S020C30"))


    def test_registered_knobs_single_view_and_geometry_cannot_be_relabelled(self):
        cfg = plan.resolve_case("s1", "S020C30")
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
            persist(root, "S010C30", step=1000)
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
        args = ["python", "-m", "panda_schedule30.best_publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch"]
        self.assertTrue(matches_writer(args, root, "s1"))
        self.assertFalse(matches_writer(args, root, "s3"))
        for module in ("panda_schedule30", "panda_scale30.best_publication", "panda_mature30.best_publication"):
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
import panda_schedule30.publication, panda_schedule30.best_publication, panda_schedule30.best_publication.transport, panda_schedule30.cost
"""
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr)


if __name__ == "__main__":
    unittest.main()
