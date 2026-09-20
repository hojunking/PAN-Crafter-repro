"""CPU-only branch thresholds, seed ranking and deadline/budget regressions."""
from datetime import timedelta
import unittest

from qg40.plan import active_cases, blocks_for, case_for, teacher_for
from qg40.policy import (CampaignWindow, RuntimeObservation, SeedResult, admission,
                         c3_eligible, choose_student_trial, estimate_case_hours,
                         estimate_calibration_hours, promote_screen)


def branch_evidence():
    return dict(p0_passed=True, baseline_complete=True, diagnostics_complete=True,
                budget_available=True, evidence_files=["diagnostics.json"],
                a24r=dict(native_c_changed=True, diagonal_reproduced=True,
                          asset_sha_verified=True, lp_synchronized=True,
                          delta_h=.001, delta_ds=-.001, relative_delta_e=.005),
                b20=dict(has_h_eligible=True, e_goal_missed=True, gate_implementation_verified=True,
                         advantage_positive_fraction=[.1, .1], soft_hard_gradient_ratio_median=[.2, .2]),
                e10=dict(lp_phase_valid=True, band_order_valid=True, metric_valid=True,
                         late_delta_ds=.001, late_delta_dlambda=.001,
                         edge_hard_gradient_cosine=[-.1, -.1], weighted_edge_hard_gradient_ratio=[.1, .1]))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.window = CampaignWindow("2026-09-20T00:00:00Z")

    def test_branch_priority_thresholds_and_one_time_selection(self):
        evidence = branch_evidence()
        choose = lambda: choose_student_trial("s2", evidence, self.window.t0_utc)
        receipt = choose()
        self.assertEqual(receipt["profile"], "A24R")
        evidence["a24r"]["delta_h"] = .0009
        self.assertEqual(choose()["profile"], "B20")
        evidence["b20"]["soft_hard_gradient_ratio_median"] = [0., .2]
        self.assertEqual(choose()["profile"], "E10")
        evidence["e10"]["edge_hard_gradient_cosine"] = [0., -.1]
        self.assertEqual(choose()["profile"], "BASE")
        self.assertEqual(choose_student_trial("s2", evidence, self.window.t0_utc,
                                             previous=receipt)["profile"], "A24R")
        for key in ("p0_passed", "baseline_complete", "diagnostics_complete", "budget_available"):
            evidence = branch_evidence()
            evidence[key] = False
            self.assertEqual(choose()["profile"], "BASE")

    def test_c3_high_q_alone_insufficient(self):
        evidence = dict(q_ref=.46875)
        self.assertFalse(c3_eligible("s1", evidence)["eligible"])
        evidence.update(parent_teacher_50k_complete=True, parent_student_seeds_completed=[82001, 82002],
                        parent_students_official_complete=True, p0_passed=True, sign_valid=True,
                        units_valid=True, cache_valid=True, augmentation_valid=True, lp_ms_phase_valid=True,
                        response_gain_median=.2, weighted_consistency_rec_a_gradient_ratio=[.09, .09],
                        evidence_files=["teacher_diagnostics.json"])
        self.assertTrue(c3_eligible("s1", evidence)["eligible"])
        evidence["weighted_consistency_rec_a_gradient_ratio"] = [.1, .09]
        self.assertFalse(c3_eligible("s1", evidence)["eligible"])

    def test_shared_window_strict_admission_and_eval_debt(self):
        c = teacher_for("s1")
        at = self.window.t0_utc + timedelta(hours=29.5)
        allowed = admission(self.window, at, [c], p0_ready=True, evaluation_debt_hours=0.)
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["calibration_hours"], 2.5)
        self.assertEqual(allowed["block_hours"], 6.)
        self.assertFalse(admission(self.window, at, [c], p0_ready=True,
                                   evaluation_debt_hours=.5)["allowed"])
        self.assertEqual(admission(self.window, self.window.admission_cutoff_utc, [c],
                                   p0_ready=True, evaluation_debt_hours=0.)["reason"], "NOT_ADMITTED_CUTOFF")
        self.assertEqual(CampaignWindow.from_dict(self.window.to_dict()), self.window)
        with self.assertRaises(ValueError):
            CampaignWindow("2026-09-20T00:00:00")
        with self.assertRaises(ValueError):
            CampaignWindow.from_dict(dict(self.window.to_dict(), deadline_utc="2026-09-23T00:00:00+00:00"))

    def test_c3_full_chain_and_cutoff(self):
        c3 = [case for case in active_cases("s1", enable_c3=True) if case.reference_id == "QB_TB_C3"]
        args = dict(p0_ready=True, evaluation_debt_hours=1., c3_evidence_valid=True)
        with self.assertRaises(ValueError):
            admission(self.window, self.window.t0_utc, c3[:1], **args)
        result = admission(self.window, self.window.t0_utc, c3, **args)
        self.assertAlmostEqual(result["block_hours"], 3.5 + 2.5 + 2 * 2.2)
        self.assertAlmostEqual(result["required_hours"], result["block_hours"] + 5.)
        self.assertFalse(admission(self.window, self.window.c3_cutoff_utc, c3, **args)["allowed"])

    def test_trial_pair_cannot_admit_only_alternative(self):
        pair = next(block for block in blocks_for("s2", screen="B20") if "SCREEN" in block.block_id)
        cases = [case_for(run) for run in pair.run_ids]
        args = dict(p0_ready=True, evaluation_debt_hours=0.)
        with self.assertRaises(ValueError):
            admission(self.window, self.window.t0_utc, cases[1:], **args)
        result = admission(self.window, self.window.t0_utc, cases, **args)
        self.assertEqual(result["block_hours"], 5.)
        fixed_pair = next(block for block in blocks_for("s1") if block.block_id == "s1_BASE_01")
        with self.assertRaises(ValueError):
            admission(self.window, self.window.t0_utc, fixed_pair.run_ids[:1], **args)

    def test_runtime_estimates_do_not_mix_roles_or_broken_evaluation(self):
        case = teacher_for("s1")
        def observation(component, hours, **kwargs):
            return RuntimeObservation(server_id="s1", sensor="QB", role="T", width=112,
                                      depth=(1, 2, 3), component=component, hours=hours,
                                      completed=True, evaluation_valid=kwargs.get("valid", True))
        self.assertEqual(estimate_case_hours(case, [observation("TRAIN", 1.)]), 3.5)
        measured = [observation("TRAIN", 2.), observation("EVAL", 1.), observation("TRAIN", .01, valid=False)]
        self.assertEqual(estimate_case_hours(case, measured), 3.75)

    def test_screen_ranking_missing_target_and_no_eligible_rule(self):
        def row(seed, h, e, eligible=False):
            return SeedResult(seed, eligible, e if eligible else None, h if eligible else None,
                              h, e, official_complete=True, same_checkpoint_verified=True)
        base = [row(82006, .90, 4.), row(82007, .90, 4.)]
        alt = [row(82006, .902, 4.01), row(82007, .902, 4.01)]
        result = promote_screen("s2", "B20", base, alt)
        self.assertTrue(result["promoted"])
        self.assertIsNone(result["base"]["target_e_median"])
        self.assertFalse(promote_screen("s2", "B20", base, base)["promoted"])
        self.assertFalse(promote_screen("s2", "B20", base, alt[:1])["promoted"])
        base = [row(82006, .93, 3.6, True), row(82007, .90, 3.6)]
        alt = [row(82006, .90, 3.5), row(82007, .94, 3.5, True)]
        result = promote_screen("s2", "B20", base, alt)
        self.assertTrue(result["promoted"])
        self.assertTrue(result["different_eligible_seeds"])

    def test_controller_capacity_metadata_roundtrips_through_time_estimators(self):
        case = teacher_for("s1")
        common = dict(server_id="s1", sensor="QB", role="T", width=112,
            depth=[1, 2, 3], completed=True, evaluation_valid=True, profile="BASE",
            peak_training_memory_bytes=1024, memory_scope="training+scheduled_eval+diagnostics")
        rows = [dict(common, component=component, hours=hours)
                for component, hours in (("TRAIN", 2.), ("EVAL", 1.), ("CALIBRATION", .5))]
        self.assertEqual(estimate_case_hours(case, rows), 3.75)
        self.assertEqual(estimate_calibration_hours(case, rows), .625)
        result = admission(self.window, self.window.t0_utc, [case], p0_ready=True,
                           evaluation_debt_hours=0., observations=rows)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["block_hours"], 4.375)


if __name__ == "__main__":
    unittest.main()
