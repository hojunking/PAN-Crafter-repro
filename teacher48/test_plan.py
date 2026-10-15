"""CPU-only campaign policy tests; no datasets, launches, or sheet writes."""
import copy
import math
import unittest

from . import plan


def baseline():
    return {
        **plan.METHOD_CONTRACT,
        "teacher_architecture": {"bands": 8, "width": 112, "depth": [1, 2, 3]},
        "aligner_architecture": {"name": "SceneAwarePANAligner", "view_margin": 4},
        "teacher_input_layout": "P0", "teacher_base": "BICUBIC_MS",
        "optimizer": "AdamW", "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8,
        "lr_U0": 1e-4, "lr_A0": 1e-5, "weight_decay0": 0.01,
        "warmup_updates0": 1000, "min_lr": 0., "scheduler_family": "warmup_cosine",
        "effective_batch_RR": 8, "effective_batch_FR": 2,
        "microbatch_RR": 4, "microbatch_FR": 1,
        "accumulation_RR": 2, "accumulation_FR": 2,
        "RR_train_HW": [64, 64], "FR_train_HW": [128, 128],
        "precision": "fp32", "augmentation": {"dihedral": True},
        "evaluator_revision": "synthetic-test-only-v1",
        "gradient_kernel_roi_normalization": {
            "sensor": "WV3", "rgb_indices": [4, 2, 1],
            "operator": "forward_difference", "padding": "reflect",
            "roi_margin": 4, "radiometry": "test_0_1",
        },
    }


def evaluation(server="s1", h=None, e=None):
    lane = plan.LANES[server]
    h = lane["hqnr_min"] if h is None else h
    e = lane["ergas_max"] if e is None else e
    identity = {**plan.METHOD_CONTRACT, "run_id": "unit-run", "attempt": 1, "seed": lane["seeds"][0],
                "completed_update": 25_000, "model_state_sha256": "a" * 64,
                "inference_policy_digest": "native-once-v1", "evaluator_revision": "official-v1",
                "prior_rule_digest": "unit-conversion-v1"}
    rr = {"ERGAS": e, "SCC": .97, "SAM": 2.1, "PSNR": 38., "SSIM": .96,
          "RMSE": 12., "CC": .98, "Q8" if lane["dataset"] == "WV3" else "Q4": .95}
    fr = {"HQNR": h, "D_s": .02, "D_lambda": .01, "JQM": .8}
    return {
        "full_method_eligible": True, "server": server, "dataset": lane["dataset"],
        **identity,
        **{key: copy.deepcopy(identity) for key in ("state_before", "rr_state", "fr_state", "state_after")},
        "rr": {"metrics": rr, "per_scene": [{"scene_id": name, "metrics": dict(rr)} for name in ("rr1", "rr2")]},
        "fr": {"metrics": fr, "per_scene": [{"scene_id": name, "metrics": dict(fr)} for name in ("fr1", "fr2")]},
    }


def qualify(record, server="s1"):
    return plan.qualify_evaluation(record, server=server,
                                  expected_rr_ids=["rr1", "rr2"], expected_fr_ids=["fr1", "fr2"])


def numeric(h=.95, e=2.1, step=25_000, run_id="unit-run", passed=False):
    return {"HQNR": h, "ERGAS": e, "completed_update": step, "run_id": run_id, "joint_pass": passed}


def history(case="F01", *, h=.95, e=2.1):
    return {"server": "s1", "dataset": "WV3", "schedule": "H50", "seed": 261013101,
            "replica": "R1", "confirmation": False, "run_id": case,
            "loss_case": case, "optimizer_case": None, "completed_update": 25_000,
            "evaluations": [numeric(h, e, step, case) for step in (1000, 2000, 10_000, 20_000, 24_000, 25_000)]}


class RegistryTests(unittest.TestCase):
    def test_lanes_targets_seeds(self):
        self.assertEqual(set(plan.LANES), {"s1", "s2", "s3"})
        self.assertEqual([(v["dataset"], v["hqnr_min"], v["ergas_max"])
                          for v in plan.LANES.values()],
                         [("WV3", .958, 2.04), ("QB", .923, 3.5), ("GF2", .965, .522)])
        self.assertEqual(plan.LANES["s3"]["seeds"], (261013301, 261013302, 261013303))

    def test_exact_loss_grid_and_priorities(self):
        eps = (.001, .0003, .0001, .003, .001, .001, .001, .0003,
               .0001, .003, .0003, .0001, .003, .0003, .0001, .003)
        absolute = (1, 1, 1, 1, .3, .1, 3, .3, .3, .3, .1, .1, .1, 3, 3, 3)
        for i in range(16):
            self.assertEqual(plan.LOSS_CASES[f"F{i:02}"], (eps[i], absolute[i]))
        self.assertEqual(plan.LOSS_ORDER["s1"][:6], ("F00", "F01", "F04", "F07", "F02", "F06"))
        self.assertEqual(plan.LOSS_ORDER["s2"][:6], ("F00", "F02", "F08", "F11", "F01", "F07"))
        self.assertEqual(plan.LOSS_ORDER["s3"][:6], ("F00", "F06", "F13", "F01", "F14", "F04"))
        for order in plan.LOSS_ORDER.values():
            self.assertEqual(len(order), 16)
            self.assertEqual(set(order), set(plan.LOSS_CASES))

    def test_exact_optimizer_grid(self):
        expected = [(0.1, 1, 1), (.3, 1, 1), (3, 1, 1), (1, .5, 1), (1, 1.5, 1),
                    (1, 1, .1), (1, 1, 0), (.3, 1.5, 1), (.1, .5, 1)]
        self.assertEqual(list(plan.OPTIMIZER_CASES.values()), expected)
        self.assertEqual(plan.OPTIMIZER_ORDER["s2"], ("T00", "T01", "T07", "T03"))

    def test_cadence_counts_and_exact_endpoints(self):
        for schedule, count in (("H50", 27), ("H100", 52), ("H150", 77)):
            steps = plan.evaluation_steps(schedule)
            self.assertEqual(len(steps), count)
            self.assertEqual(steps, tuple(sorted(set(steps))))
            self.assertEqual(steps[0], 1000)
            self.assertEqual(steps[-1], plan.HORIZONS[schedule])
            self.assertIn(25_000, steps)
        for bad in (40_000, "H25", True):
            with self.assertRaises(ValueError):
                plan.evaluation_steps(bad)

    def test_no_missing_or_placeholder_baseline_defaults(self):
        plan.validate_baseline(baseline())
        for field in plan.BASELINE_REQUIRED:
            value = baseline()
            del value[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan.validate_baseline(value)
        value = baseline()
        value["teacher_architecture"]["width"] = "MUST_PIN_CURRENT_APPROVED"
        with self.assertRaises(ValueError):
            plan.validate_baseline(value)

    def test_invalid_numeric_baselines(self):
        for key, val in (("lr_A0", 0), ("lr_U0", float("nan")), ("weight_decay0", -1),
                         ("warmup_updates0", 50_000), ("optimizer_betas", [1, .999]),
                         ("microbatch_RR", True), ("FR_train_HW", [127, 128])):
            value = baseline()
            value[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan.validate_baseline(value)
        value = baseline()
        value["effective_batch_RR"] = 16
        with self.assertRaisesRegex(ValueError, "effective batch"):
            plan.validate_baseline(value)

    def test_all_cases_keep_full_stage1(self):
        for server in plan.LANES:
            for case in plan.LOSS_CASES:
                result = plan.resolve_case(server, baseline=baseline(), loss_case=case)
                self.assertGreater(result["lambda_abs"], 0)
                self.assertGreater(result["lambda_eps_RR"], 0)
                self.assertEqual(result["lambda_eps_RR"], result["lambda_eps_FR"])
                self.assertEqual(result["epsilon_radius_rr"], 2)
                self.assertEqual(result["epsilon_radius_fr"], 2)
                self.assertTrue(result["epsilon_every_update"])
                self.assertEqual(result["initialization"], "FRESH")
                self.assertIsNone(result["donor_checkpoint"])
                self.assertEqual(result["retention_policy"], "STRICT_TARGET_ONLY")
                self.assertFalse(result["stage2"])

    def test_baseline_relative_optimizer_not_compounded(self):
        value = baseline()
        # An unrelated existing value must never be mistaken for the baseline.
        value["lr_A"] = 999
        result = plan.resolve_case("s2", baseline=value, loss_case="F08", optimizer_case="T00")
        self.assertAlmostEqual(result["lr_A"], 1e-6)
        self.assertEqual(result["lr_U"], 1e-4)
        self.assertEqual(result["lambda_eps_FR"], .0001)
        self.assertEqual(result["lambda_abs"], .3)
        self.assertEqual(result["parent_recipe"], "F08")
        self.assertEqual(result["run_id"], "S1M13_48H_QB_S2_F08_T00_H50_R1_S261013201_v1")

    def test_long_confirmation_id_and_fresh_schedule(self):
        result = plan.resolve_case("s3", baseline=baseline(), loss_case="F06", optimizer_case="T04",
                                   schedule="H100", replica="R2")
        self.assertEqual(result["run_id"], "S1M13_48H_GF2_S3_F06_T04_H100_R2_S261013302_v1")
        self.assertEqual(result["planned_horizon"], 100_000)
        self.assertTrue(result["confirmation"])
        self.assertEqual(result["initialization"], "FRESH")

    def test_zero_decay_duplicates_are_non_runnable(self):
        value = baseline()
        value["weight_decay0"] = 0
        for case in ("T05", "T06"):
            result = plan.resolve_case("s1", baseline=value, optimizer_case=case)
            self.assertFalse(result["runnable"])
            self.assertEqual(result["status"], "SKIP_DUPLICATE_CONFIG")
        self.assertTrue(plan.resolve_case("s1", baseline=value, optimizer_case="T01")["runnable"])

    def test_retry_same_id_and_streams_case_independent(self):
        original = plan.resolve_case("s1", baseline=baseline())
        retry = plan.resolve_case("s1", baseline=baseline(), attempt=2)
        alternative = plan.resolve_case("s1", baseline=baseline(), loss_case="F15")
        self.assertEqual(original["run_id"], retry["run_id"])
        self.assertEqual(original["stream_seeds"], retry["stream_seeds"])
        self.assertEqual(original["stream_seeds"], alternative["stream_seeds"])
        self.assertEqual(len(set(original["stream_seeds"].values())), len(original["stream_seeds"]))
        self.assertEqual(original["stream_seeds"]["diagnostic_probe"], 261013900)

    def test_baseline_copy_and_invalid_run_choices(self):
        value = baseline()
        result = plan.resolve_case("s1", baseline=value)
        result["baseline_resolved"]["teacher_architecture"]["width"] = 999
        self.assertEqual(value["teacher_architecture"]["width"], 112)
        for kwargs in ({"replica": "R4"}, {"attempt": 0}, {"optimizer_case": "T09"}, {"schedule": "H25"}):
            with self.assertRaises(ValueError):
                plan.resolve_case("s1", baseline=value, **kwargs)
        with self.assertRaises(ValueError):
            plan.resolve_case("s4", baseline=value)


class QualificationTests(unittest.TestCase):
    def test_exact_inclusive_boundaries_all_lanes(self):
        for server in plan.LANES:
            result = qualify(evaluation(server), server)
            self.assertTrue(result["joint_pass"], result)
            self.assertEqual(result["D_inf"], 0)
            self.assertEqual(result["D1"], 0)

    def test_rounding_cannot_pass(self):
        self.assertFalse(qualify(evaluation(h=.9579999))["joint_pass"])
        self.assertFalse(qualify(evaluation("s3", e=.5220001), "s3")["joint_pass"])
        self.assertFalse(plan.target_gap(math.nextafter(.958, 0), 2.04, server="s1")["h_pass"])
        self.assertFalse(plan.target_gap(.965, math.nextafter(.522, math.inf), server="s3")["e_pass"])

    def test_one_goal_is_never_joint_pass(self):
        for h, e in ((.97, 2.2), (.94, 2.0)):
            self.assertFalse(qualify(evaluation(h=h, e=e))["joint_pass"])

    def test_missing_duplicate_and_unexpected_scene_fail(self):
        for kind in ("missing", "duplicate", "unexpected"):
            value = evaluation()
            if kind == "missing":
                value["rr"]["per_scene"].pop()
            elif kind == "duplicate":
                value["fr"]["per_scene"][1]["scene_id"] = "fr1"
            else:
                value["fr"]["per_scene"][1]["scene_id"] = "fr3"
            result = qualify(value)
            self.assertFalse(result["joint_pass"], kind)
            self.assertFalse(result["metrics_complete"], kind)

    def test_expected_list_cannot_be_empty_or_duplicate(self):
        for rr in ([], ["rr1", "rr1"]):
            result = plan.qualify_evaluation(evaluation(), server="s1", expected_rr_ids=rr,
                                            expected_fr_ids=["fr1", "fr2"])
            self.assertFalse(result["joint_pass"])

    def test_every_requested_metric_required(self):
        for branch, key in (("rr", "RMSE"), ("rr", "CC"), ("rr", "Q8"), ("fr", "JQM")):
            value = evaluation()
            del value[branch]["metrics"][key]
            result = qualify(value)
            self.assertFalse(result["joint_pass"])
            self.assertFalse(result["metrics_complete"])

    def test_scene_nan_and_inf_prevent_gate(self):
        for branch, key, val in (("rr", "RMSE", float("nan")), ("fr", "JQM", float("inf"))):
            value = evaluation()
            value[branch]["per_scene"][0]["metrics"][key] = val
            result = qualify(value)
            self.assertFalse(result["joint_pass"])
            self.assertFalse(result["metrics_finite"])

    def test_aggregate_must_match_scene_mean(self):
        value = evaluation()
        value["fr"]["per_scene"][0]["metrics"]["HQNR"] -= .001
        result = qualify(value)
        self.assertFalse(result["joint_pass"])
        self.assertIn("FR_AGGREGATE_MISMATCH:HQNR", result["reasons"])

    def test_all_four_snapshot_identities_must_match(self):
        for state in ("state_before", "rr_state", "fr_state", "state_after"):
            for key, bad in (("completed_update", 26_000), ("model_state_sha256", "b" * 64),
                             ("attempt", 2), ("inference_policy_digest", "changed"),
                             ("evaluator_revision", "other"), ("prior_rule_digest", "scale-four")):
                value = evaluation()
                value[state][key] = bad
                result = qualify(value)
                with self.subTest(state=state, key=key):
                    self.assertFalse(result["same_state_verified"])
                    self.assertFalse(result["joint_pass"])

    def test_missing_identity_cannot_be_replaced_by_verified_boolean(self):
        value = evaluation()
        del value["fr_state"]
        value["same_state_verified"] = True
        self.assertFalse(qualify(value)["joint_pass"])

    def test_outer_run_identity_is_cross_checked(self):
        value = evaluation()
        value["run_id"] = "different-run"
        self.assertFalse(qualify(value)["joint_pass"])

    def test_numerical_prior_cache_may_differ_with_same_rule(self):
        value = evaluation()
        value["rr"]["prior_cache_digest"] = "a" * 64
        value["fr"]["prior_cache_digest"] = "b" * 64
        self.assertTrue(qualify(value)["joint_pass"])

    def test_method_and_lane_eligibility_required(self):
        for field, bad in (("full_method_eligible", False), ("full_method_eligible", 1),
                           ("dataset", "QB"), ("server", "s3")):
            value = evaluation()
            value[field] = bad
            self.assertFalse(qualify(value)["joint_pass"])

    def test_invalid_target_values(self):
        for h, e in ((float("nan"), 2), (.97, float("inf")), (1.1, 2), (.97, -.1), (True, 2)):
            result = plan.target_gap(h, e, server="s1")
            self.assertFalse(result["target_pair_pass"])
            self.assertIsNone(result["D_inf"])


class RankingTests(unittest.TestCase):
    def test_qualified_representative_lexicographic_not_ergas_first(self):
        high_h = numeric(.97, 2.03, passed=True)
        low_e = numeric(.96, 1.9, passed=True)
        invalid = numeric(.999, 8, passed=False)
        self.assertIs(plan.representative([low_e, high_h, invalid]), high_h)
        self.assertIs(plan.rr_reserve([low_e, high_h]), low_e)
        self.assertIsNone(plan.representative([invalid]))

    def test_exact_ties_then_step_and_run_id(self):
        a = numeric(.96, 2., 2000, "b", True)
        b = numeric(.96, 2., 1000, "b", True)
        c = numeric(.96, 2., 1000, "a", True)
        d = numeric(.96, 2.01, 1, "a", True)
        self.assertIs(plan.representative([a, b, c, d]), c)

    def test_gap_uses_one_snapshot_and_is_not_retention_permission(self):
        one = numeric(.99, 3.)
        two = numeric(.90, 1.)
        closest = min([one, two], key=lambda r: plan.gap_rank(r, server="s1"))
        self.assertIs(closest, two)
        self.assertIsNone(plan.representative([one, two]))

    def test_pareto_front_keeps_both_tradeoffs_and_exact_ties(self):
        a, b, c = numeric(.97, 3.), numeric(.95, 2.), numeric(.94, 4.)
        tie = dict(a, run_id="tie")
        result = plan.pareto_front([a, b, c, tie])
        self.assertEqual(len(result), 3)
        self.assertNotIn(c, result)
        self.assertFalse(any(r["joint_pass"] for r in result))


class PruningTests(unittest.TestCase):
    def test_only_two_consistent_comparators_allow_prune(self):
        cur = history()
        before = [history("F00", h=.96, e=2.), history("F02", h=.97, e=2.)]
        self.assertFalse(plan.should_prune(cur, before[:1])["prune"])
        result = plan.should_prune(cur, before)
        self.assertTrue(result["prune"])
        self.assertEqual(result["common_steps"], [20_000, 24_000, 25_000])

    def test_baseline_confirmation_and_before25k_protected(self):
        before = [history("F02", h=.97, e=2.), history("F03", h=.97, e=2.)]
        cases = [history("F00"), dict(history(), replica="R2"),
                 dict(history(), confirmation=True), dict(history(), completed_update=24_000)]
        for cur in cases:
            self.assertFalse(plan.should_prune(cur, before)["prune"])

    def test_any_gap_improvement_blocks_pruning(self):
        cur = history()
        cur["evaluations"][-1]["ERGAS"] -= .001
        before = [history("F02", h=.97, e=2.), history("F03", h=.97, e=2.)]
        self.assertFalse(plan.should_prune(cur, before)["prune"])

    def test_wrong_seed_horizon_or_dataset_cannot_dominate(self):
        for key, value in (("seed", 261013102), ("schedule", "H100"), ("dataset", "QB")):
            before = [history("F02", h=.97, e=2.), dict(history("F03", h=.97, e=2.), **{key: value})]
            self.assertFalse(plan.should_prune(history(), before)["prune"])

    def test_missing_common_points_and_10k_blocks_pruning(self):
        before = [history("F02", h=.97, e=2.), history("F03", h=.97, e=2.)]
        before[1]["evaluations"] = before[1]["evaluations"][-2:]
        self.assertFalse(plan.should_prune(history(), before)["prune"])
        cur = history()
        cur["evaluations"] = [e for e in cur["evaluations"] if e["completed_update"] != 10_000]
        self.assertFalse(plan.should_prune(cur, before)["prune"])

    def test_technical_retries_not_independent_comparators(self):
        one = history("F02", h=.97, e=2.)
        two = copy.deepcopy(one)
        two["run_id"] += "-attempt2"
        self.assertFalse(plan.should_prune(history(), [one, two])["prune"])

    def test_completed_prior_runs_can_have_future_points(self):
        before = [history("F02", h=.97, e=2.), history("F03", h=.97, e=2.)]
        for run in before:
            run["completed_update"] = 50_000
            run["evaluations"].append(numeric(.98, 1.9, 50_000, run["run_id"]))
        self.assertTrue(plan.should_prune(history(), before)["prune"])


class BudgetTests(unittest.TestCase):
    def test_p90_estimate_includes_all_eval_and_artifact_costs(self):
        self.assertEqual(plan.estimate_remaining_seconds(remaining_updates=100,
            update_p90_seconds=.5, remaining_evaluations=2, rr_fr_eval_seconds=10,
            artifact_seconds=30), 100)

    def test_twenty_percent_safety_and_deadline_inclusive(self):
        result = plan.admit_new_run(elapsed_seconds=45 * 3600, estimated_to_25k_seconds=1000,
                                   estimated_full_seconds=9000)
        self.assertTrue(result["admitted"])
        self.assertEqual(result["estimated_finish_seconds"], 48 * 3600)
        result = plan.admit_new_run(elapsed_seconds=45 * 3600, estimated_to_25k_seconds=1000,
                                   estimated_full_seconds=9001)
        self.assertFalse(result["admitted"])

    def test_no_new_runs_at46h_even_if_fast(self):
        result = plan.admit_new_run(elapsed_seconds=46 * 3600, estimated_to_25k_seconds=1,
                                   estimated_full_seconds=1)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "NOT_STARTED_NO_NEW_AFTER_46H")

    def test_minimum25k_budget_and_closing_reserve(self):
        result = plan.admit_new_run(elapsed_seconds=45 * 3600, estimated_to_25k_seconds=10_000,
                                   estimated_full_seconds=11_000)
        self.assertEqual(result["reason"], "NOT_STARTED_BUDGET_25K")
        result = plan.admit_new_run(elapsed_seconds=45 * 3600, estimated_to_25k_seconds=1000,
                                   estimated_full_seconds=9000, closing_reserve_seconds=1)
        self.assertFalse(result["admitted"])

    def test_missing_measurements_do_not_invent_speed(self):
        with self.assertRaises(ValueError):
            plan.admit_new_run(elapsed_seconds=0, estimated_to_25k_seconds=0, estimated_full_seconds=0)
        with self.assertRaises(ValueError):
            plan.estimate_remaining_seconds(remaining_updates=100, update_p90_seconds=0,
                remaining_evaluations=1, rr_fr_eval_seconds=1, artifact_seconds=0)


if __name__ == "__main__":
    unittest.main()
