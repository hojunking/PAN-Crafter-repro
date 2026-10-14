"""Data-free policy/provenance/admission tests for the two 48h lanes."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from . import plan

ROOT = Path(__file__).resolve().parents[1]


def baseline(server="s1"):
    return json.loads((ROOT / "config" / "directshift48" / (server + ".baseline.json")).read_text())


def config(server="s1", **options):
    return plan.resolve_case(server, baseline=baseline(server), **options)


def point(h=.959, e=2.03, step=25000, name="unit", passed=False):
    return {"HQNR": h, "ERGAS": e, "completed_update": step, "run_id": name,
            "joint_pass": passed, "full_method_eligible": True,
            "metrics_complete": True, "metrics_finite": True, "same_state_verified": True}


def evaluation(server="s1", h=None, e=None):
    lane = plan.LANES[server]
    c = config(server)
    ident = {**plan.METHOD_CONTRACT, "run_id": c["run_id"], "attempt": 1,
             "seed": c["seed"], "completed_update": 25000,
             "model_state_sha256": "a" * 64, "inference_policy_digest": "b" * 64,
             "evaluator_revision": "test-evaluator", "alignment_protocol_digest": c["alignment_protocol_digest"]}
    rr = {"ERGAS": lane["ergas_max"] if e is None else e, "SCC": .97,
          "SAM": 2., "PSNR": 38., "SSIM": .96, "RMSE": 12., "CC": .98,
          "Q8" if server == "s1" else "Q4": .95}
    fr = {"HQNR": lane["hqnr_min"] if h is None else h, "D_s": .02,
          "D_lambda": .01, "JQM": .8}
    return {"full_method_eligible": True, "server": server, "dataset": lane["dataset"],
            **ident, **{key: dict(ident) for key in ("state_before", "rr_state", "fr_state", "state_after")},
            "rr": {"metrics": rr, "per_scene": [{"scene_id": "rr" + str(i), "metrics": dict(rr)} for i in range(20)]},
            "fr": {"metrics": fr, "per_scene": [{"scene_id": "fr" + str(i), "metrics": dict(fr)} for i in range(20)]}}


def qualify(record, server="s1"):
    return plan.qualify_evaluation(record, server=server,
        expected_rr_ids=["rr" + str(i) for i in range(20)],
        expected_fr_ids=["fr" + str(i) for i in range(20)])


class RegistryTests(unittest.TestCase):
    def test_01_document_hashes(self):
        for path, expected in [(ROOT / plan.PLAN_RELATIVE_PATH, plan.PLAN_SHA256),
                               (ROOT / "research_log" / plan.METHOD_FILE_NAME, plan.METHOD_SHA256)]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_02_only_requested_lanes_and_independent_seeds(self):
        self.assertEqual(set(plan.LANES), {"s1", "s3"})
        self.assertEqual(plan.LANES["s1"]["dataset"], "WV3")
        self.assertEqual(plan.LANES["s3"]["dataset"], "GF2")
        seeds = [seed for lane in plan.LANES.values() for seed in lane["seeds"]]
        self.assertEqual(len(set(seeds)), 6)
        with self.assertRaises(ValueError): plan.catalog("s2")

    def test_03_catalogs_exact_and_same_sensor_order(self):
        self.assertEqual(plan.LOSS_ORDER["s1"], plan.LOSS_ORDER["s3"])
        self.assertEqual(set(plan.LOSS_ORDER["s1"]), set(plan.LOSS_CASES))
        for server in plan.LANES:
            b = baseline(server)
            plan.validate_baseline(b)
            actual = json.loads((ROOT / "config" / "directshift48" / (server + ".catalog.json")).read_text())
            self.assertEqual(actual, json.loads(json.dumps(plan.catalog(server))))
            self.assertEqual(b["teacher_architecture"]["width"], 112)
            self.assertEqual(b["teacher_architecture"]["depth"], [1, 2, 3])
            self.assertEqual(b["probes"]["response_radii"], [.25, .5, 1, 2, 3])

    def test_04_case_grid_all_positive_fresh_and_R3(self):
        for server in plan.LANES:
            for case in plan.LOSS_CASES:
                for replica in ("R1", "R2", "R3"):
                    c = config(server, loss_case=case, replica=replica)
                    self.assertGreater(c["lambda_abs"], 0)
                    self.assertGreater(c["lambda_epsilon"], 0)
                    self.assertEqual(c["lambda_rec"], 1)
                    self.assertEqual(c["epsilon_radius_rr"], 3)
                    self.assertEqual(c["epsilon_distribution"], "UNIFORM_AREA_DISK")
                    self.assertEqual(c["initialization"], "FRESH")
                    self.assertIsNone(c["donor_checkpoint"])
                    self.assertTrue(c["run_id"].startswith("DS3_"))
                    self.assertFalse(c["stage2"])
                    self.assertFalse(c["requires_fr_train"])

    def test_05_protocol_pure_no_torch_import(self):
        code = "import sys; from directshift48 import plan; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)

    def test_06_missing_baseline_fields_rejected(self):
        for key in plan.BASELINE_REQUIRED:
            b = baseline(); del b[key]
            with self.subTest(key=key), self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_07_invalid_baseline_numeric_rejected(self):
        for key, value in [("lr_A0", 0), ("lr_U0", float("nan")), ("optimizer_eps", True),
                           ("weight_decay0", -1), ("warmup_updates0", 50000),
                           ("microbatch_RR", 11), ("amp", True), ("tf32", True),
                           ("precision", "fp16"), ("absolute_roi_margin", 7)]:
            b = baseline(); b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_08_legacy_fields_forbidden(self):
        for key in plan.LEGACY_FORBIDDEN:
            b = baseline(); b[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError): plan.validate_baseline(b)
            c = config(); c[key] = None
            with self.subTest(config_key=key), self.assertRaises(ValueError): plan.validate_method_config(c)

    def test_09_method_config_corruption_rejected(self):
        changes = [("epsilon_radius_rr", 2), ("epsilon_distribution", "UNIFORM_RADIUS"),
                   ("lambda_abs", True), ("lambda_epsilon", 0), ("lambda_rec", True),
                   ("requires_fr_train", True), ("stage2", True), ("confirmation", True),
                   ("donor_checkpoint", "old.pt"), ("attempt", 0), ("run_id", "EGR2_old"),
                   ("hqnr_min", .960), ("ergas_max", 2.045), ("alignment_protocol_digest", "0" * 64)]
        for key, value in changes:
            c = config(); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): plan.validate_method_config(c)

    def test_10_support_and_protocol_identity_pinned(self):
        b = baseline()
        self.assertEqual(b["alignment_protocol"], plan.alignment_protocol_manifest(b))
        self.assertEqual(b["support_policy"]["aligner_view_margin"], 8)
        self.assertEqual(b["support_policy"]["local_convolution_receptive_field"], 25)
        self.assertEqual(b["support_policy"]["absolute_stencil_support_margin"], 7)
        b["alignment_protocol"]["units"] = "RR_to_FR_times4"
        with self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_11_h50_only_complete_eval_grid(self):
        steps = plan.evaluation_steps("H50")
        self.assertEqual(len(steps), 27)
        self.assertEqual(steps, tuple(sorted(set(steps))))
        self.assertEqual(steps[0], 1000)
        self.assertEqual(steps[-1], 50000)
        self.assertIn(25000, steps)
        for bad in (True, "H100", "H150", 25000):
            with self.assertRaises(ValueError): plan.evaluation_steps(bad)

    def test_12_optimizer_multipliers_use_baseline(self):
        for server in plan.LANES:
            for optimizer, (ma, mu, mw) in plan.OPTIMIZER_CASES.items():
                c = config(server, optimizer_case=optimizer)
                b = c["baseline_resolved"]
                self.assertEqual(c["lr_A"], b["lr_A0"] * ma)
                self.assertEqual(c["lr_U"], b["lr_U0"] * mu)
                self.assertEqual(c["weight_decay"], b["weight_decay0"] * mw)

    def test_13_rng_streams_independent_but_reused_across_r1_cases(self):
        a, b, c = config(), config(loss_case="DF07"), config(replica="R2")
        self.assertEqual(a["stream_seeds"], b["stream_seeds"])
        self.assertNotEqual(a["stream_seeds"], c["stream_seeds"])
        values = list(a["stream_seeds"].values())
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(a["stream_seeds"]["diagnostic_probe"], c["stream_seeds"]["diagnostic_probe"])


class EvaluationAndSelectionTests(unittest.TestCase):
    def test_14_original_targets_inclusive_and_sheet_separate(self):
        for server in plan.LANES:
            self.assertTrue(qualify(evaluation(server), server)["joint_pass"])
        self.assertFalse(qualify(evaluation(h=.960, e=2.045))["joint_pass"])
        self.assertTrue(qualify(evaluation(h=.959, e=2.04))["joint_pass"])
        self.assertEqual(plan.SHEET_TARGETS["s1"], {"hqnr_min": .960, "ergas_max": 2.045})

    def test_15_twenty_scenes_mandatory(self):
        for branch in ("rr", "fr"):
            r = evaluation(); r[branch]["per_scene"].pop()
            self.assertFalse(qualify(r)["joint_pass"])
            r = evaluation(); r[branch]["per_scene"][0]["scene_id"] = r[branch]["per_scene"][1]["scene_id"]
            self.assertFalse(qualify(r)["joint_pass"])

    def test_16_all_twelve_metrics_required_and_finite(self):
        for branch in ("rr", "fr"):
            for key in evaluation()[branch]["metrics"]:
                r = evaluation(); del r[branch]["metrics"][key]
                self.assertFalse(qualify(r)["joint_pass"])
                r = evaluation(); r[branch]["per_scene"][0]["metrics"][key] = float("nan")
                self.assertFalse(qualify(r)["joint_pass"])

    def test_17_same_state_and_protocol_all_four_views(self):
        for branch in ("state_before", "rr_state", "fr_state", "state_after"):
            for key in ("model_state_sha256", "alignment_protocol_digest", "method_revision"):
                r = evaluation(); r[branch][key] = "invalid"
                self.assertFalse(qualify(r)["joint_pass"])

    def test_18_aggregate_must_equal_exact_scene_mean(self):
        r = evaluation(); r["fr"]["metrics"]["HQNR"] += .001
        self.assertFalse(qualify(r)["joint_pass"])
        r = evaluation(); r["rr"]["metrics"]["ERGAS"] -= .001
        self.assertFalse(qualify(r)["joint_pass"])

    def test_19_no_silent_wrong_sensor_or_old_namespace(self):
        r = evaluation(); r["dataset"] = "GF2"
        self.assertFalse(qualify(r)["joint_pass"])
        r = evaluation(); r["method_revision"] = "EdgeGuidedResidual-S1-v1"
        self.assertFalse(qualify(r)["joint_pass"])

    def test_20_hqnr_first_even_if_candidate_misses_artifact_target(self):
        high_hqnr = point(.962, 2.2, name="highest-hqnr")
        qualified = point(.960, 2.03, name="qualified", passed=True)
        self.assertIs(plan.select_candidate([qualified, high_hqnr]), high_hqnr)
        self.assertIs(plan.representative([qualified, high_hqnr]), qualified)
        self.assertIsNone(plan.representative([high_hqnr]))

    def test_21_ties_ergas_then_step_then_runid(self):
        rows = [point(.96, 2.1, 24000, "a"), point(.96, 2.0, 26000, "a"),
                point(.96, 2.0, 24000, "z"), point(.96, 2.0, 24000, "b")]
        self.assertEqual(plan.select_candidate(rows)["run_id"], "b")

    def test_22_invalid_evaluations_never_candidates(self):
        for key in ("metrics_complete", "metrics_finite", "same_state_verified", "full_method_eligible"):
            p = point(); p[key] = False
            self.assertIsNone(plan.select_candidate([p]))


class BudgetTests(unittest.TestCase):
    def test_23_estimate_raw_time(self):
        self.assertEqual(plan.estimate_remaining_seconds(remaining_updates=50000,
            update_p90_seconds=.1, remaining_evaluations=27, rr_fr_eval_seconds=10,
            artifact_seconds=30, probe_seconds=20), 5320)

    def test_24_both_confirmations_and_closeout_reserved(self):
        r = plan.admit_new_run(elapsed_seconds=3600, estimated_to_25k_seconds=5000,
            estimated_full_seconds=10000, selected_recipe_seconds=20000)
        self.assertTrue(r["admitted"])
        self.assertEqual(r["estimated_run_seconds"], 12000)
        self.assertEqual(r["confirmation_reserve_seconds"], 48000)
        self.assertEqual(r["estimated_finish_with_reserves_seconds"], 70800)

    def test_25_confirmation_count_adjusts_without_double_safety(self):
        for remaining in (0, 1, 2):
            r = plan.admit_new_run(elapsed_seconds=0, estimated_to_25k_seconds=1,
                estimated_full_seconds=10, remaining_confirmations=remaining)
            self.assertEqual(r["confirmation_reserve_seconds"], 12 * remaining)

    def test_26_new_run_after_46h_or_phase_boundary_rejected(self):
        for elapsed, phase, reason in [(46 * 3600, 46, "NOT_STARTED_NO_NEW_AFTER_46H"),
                                      (22 * 3600, 22, "NOT_STARTED_PHASE_BOUNDARY")]:
            r = plan.admit_new_run(elapsed_seconds=elapsed, estimated_to_25k_seconds=1,
                estimated_full_seconds=2, remaining_confirmations=0, phase_end_hours=phase)
            self.assertFalse(r["admitted"])
            self.assertEqual(r["reason"], reason)

    def test_27_no_oversubscription_due_to_omitted_replicas(self):
        r = plan.admit_new_run(elapsed_seconds=30 * 3600, estimated_to_25k_seconds=10000,
            estimated_full_seconds=20000, remaining_confirmations=2)
        self.assertFalse(r["admitted"])
        self.assertTrue(r["switch_to_confirmation"])

    def test_28_unmeasured_or_negative_timing_fails_closed(self):
        with self.assertRaises(ValueError):
            plan.estimate_remaining_seconds(remaining_updates=50000, update_p90_seconds=0,
                remaining_evaluations=27, rr_fr_eval_seconds=1, artifact_seconds=1)
        with self.assertRaises(ValueError):
            plan.admit_new_run(elapsed_seconds=0, estimated_to_25k_seconds=10,
                estimated_full_seconds=20, closing_reserve_seconds=60)

    def test_29_baseline_and_confirmations_pruning_protected(self):
        for run in ({"loss_case": "DF00", "optimizer_case": None, "replica": "R1"},
                    {"loss_case": "DF01", "replica": "R2", "confirmation": True}):
            self.assertFalse(plan.should_prune(run, [], stagnation_window=3, tolerance=0)["prune"])


if __name__ == "__main__":
    unittest.main()
