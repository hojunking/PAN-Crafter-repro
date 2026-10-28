"""Offline integration checks for the 2026-10-28 sensor queue handoff.

Synthetic publication fixtures are never experiment results. No cloud access,
process signals, Git writes, or GPU work is performed by these tests.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


# Captured before implementing the separate successor package. In addition to
# WV3 plans, these inventories protect training, evaluation and publication.
ORIGINAL_SOURCE_PIN_DIGESTS = {
    "panda_s1loss9": "25eef0736dfa8c680864f9bd2f9a9f8b8bcd5486005a19851ba4595e42bf036d",
    "panda_tune30": "f879f4218fc2b7c96116c02f11b646afc655017e85dbb200c02e482b9cad3aab",
    "panda_hard36": "2e2e2ad19416d6d1532df6f50aea393f6be686c3f6810e9af34cc9f6a752a1b5",
}


def registered(server):
    from panda_sensor_refine import plan
    return [plan.resolve_case(server, case, replica)
            for group in plan.groups(server) for case, replica in group["runs"]]


def persist(root, cfg, *, step, hqnr, ergas):
    from panda_band30.tests.test_reporting import evaluation
    from panda_sensor_refine import reporting as R
    from panda_sensor_refine.spec import geometry_manifest
    ev = evaluation(cfg, step, hqnr, ergas)
    key = cfg["policy_id"].lower()
    ev["mature30_deployment"] = {
        "schema": "PANDA_NEXT30_DEPLOYMENT_V1", "policy_id": cfg["policy_id"],
        "completed_update": step, "geometry_key": key,
        "geometry": geometry_manifest(key),
        "student_sampling": deepcopy(cfg["student_sampling"]),
        "student_input": deepcopy(cfg["student_input"]),
        "student_warp": deepcopy(cfg["student_warp"]),
    }
    for split in ("rr", "fr"):
        for scene in ev[split]["per_scene"]:
            scene["correction_dy_dx"] = [.3, -.4]
            scene["pan_applied_dy_dx"] = [.3, -.4]
    directory = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory / "config.json", cfg)
    R.atomic_json(directory / f"evaluation_{step:07d}.json", ev)
    return R.emit_evaluation(root, cfg, ev, timestamp="2026-10-28T00:00:00+00:00")


class SensorRefineIntegration(unittest.TestCase):
    def test_original_wv3_and_legacy_source_pins_unchanged(self):
        for package, expected in ORIGINAL_SOURCE_PIN_DIGESTS.items():
            with self.subTest(package=package):
                pin = importlib.import_module(package + ".runtime").code_pin()
                actual = hashlib.sha256(json.dumps(
                    pin, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                self.assertEqual(actual, expected,
                                 "Original source changed during successor integration")

    def test_dataset_specific_plan_keeps_full_aligner_and_only_r4_extension(self):
        from panda_sensor_refine import plan
        expected = {
            "s1": ("QB", "LR120", 1.2e-4, "C40", 40000, .9315, 3.5),
            "s3": ("GF2", "LR150", 1.5e-4, "C60", 60000, .9645, .522),
        }
        self.assertEqual(plan.CAMPAIGN_ID, "PANDA_S13_SENSOR_REFINE_R34_20261028_v1")
        for server, (sensor, lr_case, rate, long_case, horizon, h, e) in expected.items():
            with self.subTest(server=server):
                cfgs = registered(server)
                self.assertEqual(len(cfgs), 10)
                self.assertEqual(len({c["run_id"] for c in cfgs}), 10)
                self.assertEqual([c["case"] for c in cfgs[:4]],
                                 [sensor + "_" + c for c in ("BASE", lr_case, long_case, "H050")])
                self.assertEqual(cfgs[0]["replica"], "R2")
                self.assertEqual(cfgs[1]["lr_S"], rate)
                self.assertEqual(cfgs[2]["planned_horizon"], horizon)
                self.assertEqual((cfgs[3]["stage2_spec"]["alpha"],
                                  cfgs[3]["stage2_spec"]["gamma"]), (.125, .25))
                self.assertEqual(sum(c["phase"] == "donor" for c in cfgs), 3)
                self.assertEqual({c["epsilon_radius_rr"] for c in cfgs}, {3., 4.})
                self.assertEqual(sum(c["epsilon_radius_rr"] == 4. for c in cfgs), 2)
                for cfg in cfgs:
                    self.assertTrue(cfg["run_id"].startswith("PSRF_"))
                    self.assertEqual((cfg["hqnr_min"], cfg["ergas_max"]), (h, e))
                    self.assertEqual(cfg["shift_scale"], 1.)
                    self.assertEqual(cfg["epsilon_distribution"], "UNIFORM_AREA_DISK")
                    self.assertEqual(cfg["baseline_resolved"]["alignment_protocol"]["epsilon"]["radius"],
                                     cfg["epsilon_radius_rr"])
                    self.assertEqual(cfg["initialization"], "FRESH")
                    self.assertFalse(cfg["inference_tta"])
                    if cfg["phase"] == "stage2":
                        self.assertEqual(cfg["stage2_spec"]["beta"], 0.)
                        self.assertEqual(cfg["student_input"]["channels"], 6)

    def test_visible_sheet_labels_expose_radius_coefficients_and_student_knobs(self):
        from panda_sensor_refine import reporting as R
        for server in ("s1", "s3"):
            for cfg in registered(server):
                with self.subTest(case=cfg["case"]):
                    text = R.approach_description(cfg)
                    pair = cfg["stage1_loss_parameters"]
                    for token in (f"lambda_abs={pair['lambda_abs']:g}",
                                  f"lambda_epsilon={pair['lambda_epsilon']:g}",
                                  f"epsilon_radius_rr={cfg['epsilon_radius_rr']:g}", "L4"):
                        self.assertIn(token, text)
                    if cfg["phase"] == "stage2":
                        self.assertIn("beta=0", text)
                        self.assertIn(f"lr={cfg['lr_S']:g}", text)
                        self.assertIn(f"{cfg['planned_horizon']//1000}K", text)
                        self.assertIn("scale=1", text)

    def test_real_projection_publishes_all_passes_and_correct_representative(self):
        from panda_sensor_refine import best_publication as B, reporting as R
        for server in ("s1", "s3"):
            h, e = R.thresholds(server)
            for cfg in (c for c in registered(server) if c["phase"] == "stage2"):
                with self.subTest(case=cfg["case"]), tempfile.TemporaryDirectory() as root:
                    # Best ERGAS below the HQNR threshold must not replace a
                    # qualifying representative or become a joint-pass row.
                    persist(root, cfg, step=1000, hqnr=h + .001, ergas=e)
                    persist(root, cfg, step=2000, hqnr=h, ergas=e - .001)
                    persist(root, cfg, step=3000, hqnr=h - .001, ergas=e - .01)
                    view = B.project(root, server)
                    self.assertFalse(view["conflicts"], view["conflicts"])
                    self.assertEqual(len(view["rows"]), 2)
                    self.assertEqual({r[R.INDEX["Selected_step"]] for r in view["rows"]}, {1000, 2000})
                    representative = next(r for r in view["rows"] if r[R.INDEX["Selector"]] == B.SELECTOR)
                    self.assertEqual(representative[R.INDEX["Selected_step"]], 2000)
                    for row in view["rows"]:
                        B.verify_owned(row, view)
                        note = json.loads(row[R.INDEX["Original description / notes"]])
                        self.assertTrue(note["publication"]["publication_joint_pass"])
                        self.assertEqual((note["publication"]["hqnr_min"],
                                          note["publication"]["ergas_max"]), (h, e))

    def test_fallback_and_transport_dry_run_preserve_other_campaign_rows(self):
        from panda_sensor_refine import best_publication as B, reporting as R
        from panda_sensor_refine.best_publication import transport as T
        class OfflineAdapter:
            def __init__(self, snapshot):
                self.data = deepcopy(snapshot)
            def snapshot(self, destination):
                return deepcopy(self.data)
            def _request(self, *args, **kwargs):
                raise AssertionError("A dry-run must never perform a cloud write")
        for server in ("s1", "s3"):
            h, e = R.thresholds(server)
            cfg = next(c for c in registered(server) if c["phase"] == "stage2")
            with tempfile.TemporaryDirectory() as root:
                persist(root, cfg, step=1000, hqnr=h - .002, ergas=e - .1)
                persist(root, cfg, step=2000, hqnr=h - .001, ergas=e + .1)
                view = B.project(root, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(view["rows"][0][R.INDEX["Selector"]], B.FALLBACK_SELECTOR)
                self.assertEqual(view["rows"][0][R.INDEX["Selected_step"]], 2000)
                foreign = [""] * 64
                foreign[0] = "Preserve existing WV3/SENSOR5/LOSS9 result"
                foreign[R.INDEX["Campaign_ID"]] = "FOREIGN_CAMPAIGN"
                dest = view["destination"]
                snapshot = {"metadata": {"spreadsheetId": dest["spreadsheet_id"], "sheets": [{
                    "properties": {"sheetId": dest["sheet_id"], "title": dest["sheet_title"],
                                   "gridProperties": {"columnCount": 83, "rowCount": 10000}}}]},
                    "header": list(R.HEADERS), "grid_row_count": 10000,
                    "rows": [foreign], "notes": ["untouched"], "cells": []}
                adapter = OfflineAdapter(snapshot)
                result = T.reconcile_rows(adapter, view, root, server, apply=False)
                self.assertEqual(result["action"], "DRY_RUN")
                self.assertEqual(result["new_rows"], 1)
                self.assertFalse(result["readback_verified"])
                self.assertEqual(adapter.data, snapshot)

    def test_real_control_loop_holds_failed_dependencies_and_runs_independent_pairs(self):
        from panda_sensor_refine.runtime import Campaign
        class OfflineCampaign(Campaign):
            # Exercise the actual registered control loop without executing
            # optimization, source-data reads, donor loading or evaluation.
            def admission(self, catalog):
                self.migration = {"external_donor": {
                    "status": "FAILED_SUPPORT", "failure": {"synthetic_offline_proof": True}}}
            def register_external_donor(self):
                raise AssertionError("A failed external donor must not be loaded")
            def validate_preflight(self, cfg):
                if cfg["case"] == "S1_A200_E030_R4":
                    raise ValueError("SUPPORT_VIOLATION: disposable offline fixture")
            def run_case(self, cfg):
                self.state["runs"].append({"run_id": cfg["run_id"], "case": cfg["case"],
                    "status": "COMPLETED", "step": cfg["planned_horizon"]})
            def register_donor(self, case):
                self.state.setdefault("synthetic_registered_donors", []).append(case)
        with tempfile.TemporaryDirectory() as root:
            campaign = OfflineCampaign("s3", output=root, device="cpu")
            # Signals are also isolated: the test exercises state transitions
            # while leaving the hosting agent's actual handlers untouched.
            with patch("panda_sensor_refine.runtime.signal.signal"):
                result = campaign.run(catalog=Path(root) / "unused.json")
            self.assertEqual(result["status"], "FINISHED_WITH_EXPLICIT_FAILURES")
            self.assertEqual(len(result["held_runs"]), 5)
            self.assertEqual(len(result["isolated_support_failures"]), 1)
            failure = next(iter(result["isolated_support_failures"].values()))
            self.assertEqual(failure["step"], 0)
            self.assertEqual(failure["evidence_kind"], "FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES")
            self.assertNotIn(failure["run_id"], {r["run_id"] for r in result["runs"]})
            self.assertEqual({r["case"] for r in result["runs"]}, {
                "S1_A200_E010_R3", "GF2_A200_E010_R3",
                "S1_A100_E030_R3", "GF2_A100_E030_R3"})
            self.assertEqual(result["synthetic_registered_donors"],
                             ["S1_A200_E010_R3", "S1_A100_E030_R3"])


if __name__ == "__main__":
    unittest.main()
