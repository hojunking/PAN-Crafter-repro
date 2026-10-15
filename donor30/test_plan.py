"""Data-free tests of lane, selection, pairing and 30h budget policy."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from . import plan


def point(h=.960, e=2.3, s=.98, *, server="s1", step=8000, name="a"):
    return {"HQNR": h, "ERGAS": e, "SCC": s, "run_id": name,
            "server": server, "dataset": plan.LANES[server]["dataset"],
            "completed_update": step, "checkpoint_path": "/tmp/" + name + ".pt",
            "state_hash": "a" * 64, "checkpoint_verified": True,
            "metrics_complete": True, "metrics_finite": True, "same_state_verified": True}


def donor(server="s1"):
    threshold = plan.LANES[server]["hqnr_min"]
    return plan.select_donor([point(h=threshold, server=server)], threshold)


class PlanTests(unittest.TestCase):
    def test_mask_policy_and_new_run_namespace_are_mandatory(self):
        for server in plan.LANES:
            cfg = plan.default_config(server)
            case = plan.resolve_donor_case(cfg)
            self.assertTrue(case['run_id'].startswith('D30_M2_'))
            self.assertEqual(case['loss_mask_policy'], plan.LOSS_MASK_POLICY)
            bad = deepcopy(cfg)
            bad.pop('loss_mask_policy')
            with self.assertRaises(ValueError):
                plan.validate_config(bad)
            case['loss_mask_policy'] = 'ALL_PIXELS'
            with self.assertRaises(ValueError):
                plan.validate_case(case, config=cfg)

    def test_both_lanes_load(self):
        self.assertEqual(set(plan.LANES), {"s1", "s3"})
        for server in plan.LANES:
            c = plan.default_config(server)
            plan.validate_config(c)
            self.assertEqual(c["budget"]["total_seconds"], 30 * 3600)
            self.assertEqual(c["baseline_resolved"]["server"], server)
        with self.assertRaises(ValueError):
            plan.default_config("s2")

    def test_no_torch_import(self):
        subprocess.run([sys.executable, "-c", "import sys; import donor30.plan; assert 'torch' not in sys.modules"],
                       cwd=plan.ROOT, check=True)

    def test_baseline_unmodified(self):
        for server in plan.LANES:
            c = plan.default_config(server)
            source = json.loads((plan.ROOT / c["baseline_path"]).read_text())
            self.assertEqual(c["baseline_resolved"], source)
            plan.resolve_donor_case(c)
            self.assertEqual(c["baseline_resolved"], source)

    def test_two_fresh_sensor_specific_donors(self):
        seeds = []
        for server in plan.LANES:
            c = plan.default_config(server)
            one, two = (plan.resolve_donor_case(c, replica) for replica in ("R1", "R2"))
            self.assertNotEqual(one["stream_seeds"], two["stream_seeds"])
            self.assertNotEqual(one["run_id"], two["run_id"])
            self.assertEqual(one["initialization"], "FRESH")
            self.assertIsNone(one["donor_checkpoint"])
            self.assertFalse(one["stage2"])
            self.assertEqual(one["lambda_abs"], 3. if server == "s1" else 1.)
            self.assertEqual(one["lambda_epsilon"], .003 if server == "s1" else .0003)
            self.assertEqual(one["epsilon_radius_rr"], 3.)
            seeds.extend([one["seed"], two["seed"]])
        self.assertEqual(len(set(seeds)), 4)

    def test_full_schedule_kept_for_early_donor(self):
        c = plan.default_config()
        resolved = plan.resolve_donor_case(c)
        self.assertEqual(resolved["planned_horizon"], 50000)
        self.assertEqual(len(resolved["evaluation_steps"]), 27)
        self.assertIn(8000, resolved["evaluation_steps"])
        self.assertIn(25000, resolved["evaluation_steps"])
        resolved["planned_horizon"] = 8000
        with self.assertRaises(ValueError):
            plan.validate_case(resolved, config=c)

    def test_inclusive_threshold_then_ergas(self):
        winner = point(h=.959, e=2.1, name="lower_ergas")
        higher_hqnr = point(h=.970, e=2.4, name="higher_hqnr")
        ineligible = point(h=.958999999999, e=1.9, name="below_gate")
        chosen = plan.select_donor([higher_hqnr, ineligible, winner], .959)
        self.assertEqual(chosen["run_id"], winner["run_id"])
        self.assertTrue(chosen["donor_hqnr_qualified"])
        self.assertEqual(chosen["donor_selection_status"], plan.QUALIFIED_DONOR)
        self.assertIsNone(chosen["donor_final_target_pass"])

    def test_ties_hqnr_scc_step_id(self):
        a = point(h=.960, e=2.2, s=.98, name="a")
        b = point(h=.961, e=2.2, s=.97, name="b")
        self.assertEqual(plan.select_donor([a, b], .959)["run_id"], "b")
        a["HQNR"] = b["HQNR"]
        self.assertEqual(plan.select_donor([a, b], .959)["run_id"], "a")
        a["SCC"] = b["SCC"]
        a["completed_update"] = 10000
        self.assertEqual(plan.select_donor([a, b], .959)["run_id"], "b")

    def test_fallback_explicit_and_never_qualified(self):
        a = point(h=.958, e=2.1)
        b = point(h=.957, e=2.0, name="b")
        selected = plan.select_donor([b, a], .959)
        self.assertEqual(selected["run_id"], "a")
        self.assertFalse(selected["donor_hqnr_qualified"])
        self.assertEqual(selected["donor_selection_status"], plan.EXPLORATORY_DONOR)
        self.assertIsNone(plan.select_donor([a, b], .959, allow_fallback=False))
        self.assertIsNone(plan.select_donor([], .959))

    def test_no_unverified_nonfinite_or_mixed_state_selection(self):
        for key, value in [("checkpoint_verified", False), ("same_state_verified", False),
                           ("metrics_complete", False), ("metrics_finite", False),
                           ("HQNR", float("nan")), ("ERGAS", float("inf")),
                           ("SCC", True), ("checkpoint_path", ""),
                           ("state_hash", "bad"), ("completed_update", 0)]:
            p = point()
            p[key] = value
            with self.subTest(key=key):
                self.assertIsNone(plan.select_donor([p], .959))
                with self.assertRaises(ValueError):
                    plan.selection_key(p, .959)

    def test_pairing_and_one_donor_across_cases(self):
        for server in plan.LANES:
            c, d = plan.default_config(server), donor(server)
            cases = plan.stage2_cases(c, d)
            self.assertEqual(len(cases), 6)
            self.assertEqual([x["stage2_mode"] for x in cases[:3]], c["stage2"]["core_order"])
            self.assertEqual([x["stage2_mode"] for x in cases[3:5]], c["stage2"]["second_seed_order"])
            self.assertEqual({x["donor_state_hash"] for x in cases}, {d["state_hash"]})
            self.assertEqual({x["donor_checkpoint"] for x in cases}, {d["checkpoint_path"]})
            self.assertTrue(all(x["shared_frozen_aligner"] and x["frozen_teacher"] for x in cases))
            self.assertEqual(cases[0]["stream_seeds"], cases[1]["stream_seeds"])
            self.assertEqual(cases[0]["stream_seeds"], cases[2]["stream_seeds"])
            self.assertEqual(cases[3]["stream_seeds"], cases[4]["stream_seeds"])
            self.assertNotEqual(cases[0]["stream_seeds"], cases[3]["stream_seeds"])
            self.assertTrue(cases[-1]["optional"])
            self.assertFalse(any(x["optional"] for x in cases[:-1]))
            self.assertEqual(len(plan.stage2_cases(c, d, include_optional=False)), 5)

    def test_wrong_server_donor_rejected(self):
        with self.assertRaises(ValueError):
            plan.stage2_cases(plan.default_config("s1"), donor("s3"))

    def test_fallback_status_propagates_to_stage2(self):
        c = plan.default_config()
        d = plan.select_donor([point(h=.958)], .959)
        case = plan.resolve_stage2_case(c, donor=d)
        self.assertEqual(case["donor_selection_status"], plan.EXPLORATORY_DONOR)
        self.assertFalse(case["donor_hqnr_qualified"])
        c["donor"]["allow_exploratory_fallback"] = False
        with self.assertRaises(ValueError):
            plan.resolve_stage2_case(c, donor=d)

    def test_stage2_selection_keeps_donor_lineage(self):
        p = point()
        p["donor_selection_status"] = plan.EXPLORATORY_DONOR
        chosen = plan.select_stage2([p], .959)
        self.assertEqual(chosen["donor_selection_status"], plan.EXPLORATORY_DONOR)
        self.assertTrue(chosen["model_hqnr_qualified"])

    def test_config_corruption_fails_closed(self):
        for section, key, value in [("donor", "hqnr_min", .958),
                                   ("donor", "loss_case", "DF00"),
                                   ("donor", "seeds", [1, 1]),
                                   ("stage2", "lambda_edge", .1),
                                   ("stage2", "lambda_edge", True),
                                   ("stage2", "calibration_split", "test"),
                                   ("stage2", "shared_frozen_donor", False),
                                   ("budget", "total_seconds", 120000)]:
            c = plan.default_config()
            c[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                plan.validate_config(c)

    def test_resolved_case_corruption_rejected(self):
        config = plan.default_config()
        for original, changes in [
                (plan.resolve_donor_case(config), {"lambda_rec": True, "epsilon_every_update": False,
                 "epsilon_distribution": "UNIFORM_RADIUS", "stream_seed_protocol": "old"}),
                (plan.resolve_stage2_case(config, "eq", donor=donor()), {"lambda_edge": True,
                 "calibration_split": "test", "donor_state_hash": "bad", "donor_hqnr_qualified": False,
                 "shared_frozen_aligner": False, "initialization": "INHERITED_STUDENT"})]:
            for key, value in changes.items():
                corrupted = deepcopy(original)
                corrupted[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    plan.validate_case(corrupted, config=config)

    def test_admission_safety_and_closeout(self):
        self.assertTrue(plan.admission_decision(0, 10000)["admitted"])
        self.assertTrue(plan.admission_decision(92400, 10000)["admitted"])
        self.assertFalse(plan.admission_decision(92400.01, 10000)["admitted"])
        self.assertFalse(plan.admission_decision(108000, 1)["admitted"])
        self.assertEqual(plan.admission_decision(120000, 1)["remaining_seconds"], 0)
        for elapsed, estimate in [(-1, 100), (0, 0), (False, 1), (0, float("inf"))]:
            with self.assertRaises(ValueError):
                plan.admission_decision(elapsed, estimate)

    def test_measured_timing_includes_full_evaluation(self):
        estimate = plan.estimated_run_seconds(.1, 20., saving_seconds=60, diagnostics_seconds=100)
        self.assertEqual(estimate, 5000 + 27 * 20 + 60 + 100)
        with self.assertRaises(ValueError):
            plan.estimated_run_seconds(.1, 20., horizon=8000)


if __name__ == "__main__":
    unittest.main()
