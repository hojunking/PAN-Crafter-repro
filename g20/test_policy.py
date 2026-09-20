from dataclasses import asdict, replace
from datetime import timedelta
import unittest

from g20 import plan as p
from g20 import policy as q


SHA = "a" * 64


def metrics(h=.955, e=.56, d=.02):
    return dict(HQNR=h, ERGAS=e, D_lambda=d)


def run(server, seed, reference, profile, exact=None, val=None):
    return q.RunResult(seed=seed, server_id=server, reference_id=reference, profile=profile,
                       exact50k=exact or metrics(), rr_val_selected=val or metrics(),
                       p0_passed=True, paired_identity_verified=True, official_complete=True,
                       same_checkpoint_verified=True, a_on=True,
                       evidence_files={f"{server}/{reference}/{profile}/{seed}.json": SHA},
                       u_init_sha256=SHA, data_view_sha256=SHA, source_sha256=SHA,
                       a_init_sha256=SHA, reference_sha256=SHA)


def pairs(server="s3", candidate="A1", seeds=p.SCREEN_SEEDS, h=.957, e=.559, d=.02):
    base, alt = q._expected_assignments(server, candidate)
    return [dict(seed=seed, base=run(server, seed, *base),
                 candidate=run(server, seed, *alt, exact=metrics(h, e, d), val=metrics(h, e, d)))
            for seed in seeds]


class WindowAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.w = q.CampaignWindow("2026-09-20T00:00:00Z")

    def admit(self, server="s3", hour=0, index=0, **kwargs):
        return q.admission(self.w, self.w.t0_utc + timedelta(hours=hour), p.blocks_for(server)[index].cases,
                           p0_ready=True, evaluation_debt_hours=0, **kwargs)

    def test_window_20_16_18(self):
        self.assertEqual(self.w.remaining_hours(self.w.t0_utc), 20)
        self.assertEqual(self.w.admission_cutoff_utc - self.w.t0_utc, timedelta(hours=16))
        self.assertEqual(self.w.train_finish_utc - self.w.t0_utc, timedelta(hours=18))
        self.assertEqual(q.CampaignWindow.from_dict(self.w.to_dict()), self.w)
        with self.assertRaises(ValueError):
            q.CampaignWindow("2026-09-20T00:00:00")
        with self.assertRaises(ValueError):
            q.CampaignWindow.from_dict(self.w.to_dict() | dict(campaign_id="old"))

    def test_initial_complete_reservations(self):
        self.assertEqual(self.admit("s1")["block_hours"], 8)
        self.assertEqual(self.admit("s2")["block_hours"], 10)
        self.assertEqual(self.admit()["block_hours"], 4.5)
        self.assertEqual(self.admit("s2")["calibration_hours"], 2)

    def test_equality_at18_allowed_but_after16_new_admission_forbidden(self):
        self.assertTrue(self.admit(hour=13.5)["allowed"])
        self.assertFalse(self.admit(hour=13.50001)["allowed"])
        self.assertEqual(self.admit(hour=16)["reason"], "NOT_ADMITTED_CUTOFF")
        self.assertEqual(self.admit(hour=20)["reason"], "PARTIAL_TIME_LIMIT")

    def test_evaluation_debt_included_and_invalid_debt_rejected(self):
        cases = p.blocks_for("s3")[0].cases
        result = q.admission(self.w, self.w.t0_utc + timedelta(hours=13), cases, p0_ready=True, evaluation_debt_hours=1)
        self.assertEqual(result["reason"], "NOT_ADMITTED_BUDGET")
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                q.admission(self.w, self.w.t0_utc, cases, p0_ready=True, evaluation_debt_hours=value)

    def test_teacher_or_partialpair_alone_rejected(self):
        for cases in (p.blocks_for("s1")[0].cases[:1], p.blocks_for("s2")[0].cases[:3],
                      p.blocks_for("s3")[0].cases[:2], tuple(reversed(p.blocks_for("s3")[0].cases))):
            with self.assertRaises(ValueError):
                q.admission(self.w, self.w.t0_utc, cases, p0_ready=True, evaluation_debt_hours=0)

    def test_s2_r3_waits_for_completed_r2_r4(self):
        self.assertEqual(self.admit("s2", index=1)["reason"], "NOT_ADMITTED_PRIORITY_COMPARISON")
        self.assertTrue(self.admit("s2", index=1, completed_run_ids=p.blocks_for("s2")[0].run_ids)["allowed"])

    def observation(self, case, component, hours, **kwargs):
        return asdict(q.RuntimeObservation(case.server_id, case.sensor, case.role, case.width, case.depth,
                                           component, hours, True, True, **kwargs))

    def test_measured_reservations_never_shrink(self):
        case = p.case_for("G20-S11")
        obs = [self.observation(case, "TRAIN", .4), self.observation(case, "EVAL", .1)]
        self.assertEqual(q.estimate_case_hours(case, obs), 1.5)
        slow = [self.observation(case, "TRAIN", 2), self.observation(case, "EVAL", 1)]
        self.assertEqual(q.estimate_case_hours(case, slow), 3.75)
        self.assertEqual(q.estimate_case_hours(case, slow[:1]), 4.)
        teacher = p.case_for("G20-T01")
        self.assertEqual(q.estimate_calibration_hours(teacher, [self.observation(teacher, "CALIBRATION", .1)]), 1.)
        self.assertEqual(q.estimate_calibration_hours(teacher, [self.observation(teacher, "CALIBRATION", 2)]), 2.5)

    def test_p90_safety_memory_observation_roundtrip(self):
        case = p.case_for("G20-S11")
        obs = [self.observation(case, "TRAIN_EVAL", hours, profile="A1", peak_training_memory_bytes=123,
                                memory_scope="train", gpu_uuid="fake") for hours in (1, 3)]
        self.assertAlmostEqual(q.estimate_case_hours(case, obs), 3.5)
        foreign = [dict(o, server_id="s4") for o in obs]
        self.assertEqual(q.estimate_case_hours(case, foreign), 1.5)

    def test_no_implicit_reuse(self):
        case = p.case_for("G20-T01")
        with self.assertRaises(ValueError):
            self.admit("s1", reuse_receipts={case.run_id: {"run_id": case.run_id}})
        receipt = dict(run_id=case.run_id, actual_updates=50000,
                       **{key: True for key in ("official_complete", "checksum_verified", "reference_equivalent",
                           "source_equivalent", "initialization_equivalent", "evaluation_equivalent", "calibration_validated")})
        self.assertEqual(self.admit("s1", reuse_receipts={case.run_id: receipt})["block_hours"], 6.)


class PairedTests(unittest.TestCase):
    def test_complete_paired_screen_passes(self):
        r = q.paired_screen("s3", "A1", pairs())
        self.assertEqual(r["status"], "PROMISING_PAIRED")
        self.assertTrue(all(r["criteria"].values()))
        self.assertFalse(r["statistical_significance_claim"])

    def test_local_reference_comparison_s1_s2(self):
        for server, candidate in (("s1", "R1"), ("s2", "R3"), ("s2", "R4")):
            rows = pairs(server, candidate)
            for row in rows:
                row["candidate"] = replace(row["candidate"], a_init_sha256="b" * 64, reference_sha256="c" * 64)
            self.assertEqual(q.paired_screen(server, candidate, rows)["status"], "PROMISING_PAIRED")

    def test_different_reference_across_seeds_is_not_one_condition(self):
        rows = pairs("s1", "R1")
        rows[0]["candidate"] = replace(rows[0]["candidate"], reference_sha256="c" * 64)
        self.assertEqual(q.paired_screen("s1", "R1", rows)["status"], "TECHNICAL_INVALID")

    def test_missing_val_or_unverified_never_pass(self):
        for field, value in (("rr_val_selected", None), ("official_complete", False), ("same_checkpoint_verified", False),
                             ("p0_passed", False), ("a_on", False), ("actual_updates", 49999),
                             ("n_evaluated", 49), ("evidence_files", {}), ("u_init_sha256", None)):
            rows = pairs()
            rows[0]["candidate"] = replace(rows[0]["candidate"], **{field: value})
            self.assertEqual(q.paired_screen("s3", "A1", rows)["status"], "TECHNICAL_INVALID", field)

    def test_pair_seed_identity_and_local_base_enforced(self):
        for field, value in (("seed", 93003), ("server_id", "s4"), ("reference_id", "R1"),
                             ("data_view_sha256", "b" * 64), ("source_sha256", "b" * 64),
                             ("u_init_sha256", "b" * 64), ("a_init_sha256", "b" * 64),
                             ("reference_sha256", "b" * 64)):
            rows = pairs()
            rows[0]["candidate"] = replace(rows[0]["candidate"], **{field: value})
            self.assertEqual(q.paired_screen("s3", "A1", rows)["status"], "TECHNICAL_INVALID", field)
        self.assertEqual(q.paired_screen("s3", "A1", pairs()[:1])["status"], "TECHNICAL_INVALID")
        self.assertEqual(q.paired_screen("s3", "A1", pairs(seeds=(93001, 93001)))["status"], "TECHNICAL_INVALID")

    def test_h_requires_both_positive_and_median(self):
        rows = pairs(h=.956)
        self.assertEqual(q.paired_screen("s3", "A1", rows)["status"], "NO_GAIN")
        rows[1]["candidate"] = replace(rows[1]["candidate"], exact50k=metrics(.954))
        self.assertFalse(q.paired_screen("s3", "A1", rows)["eligible"])

    def test_e_and_dlambda_guards_individual_and_median(self):
        for e, d in ((.566, .02), (.559, .0221), (.563, .02), (.559, .0211)):
            r = q.paired_screen("s3", "A1", pairs(e=e, d=d))
            self.assertEqual(r["status"], "HQNR_ONLY_TRADEOFF")
            self.assertFalse(r["eligible"])

    def test_validation_h_and_e_guard(self):
        for val in (metrics(.95, .559), metrics(.957, .567)):
            rows = pairs()
            for row in rows:
                row["candidate"] = replace(row["candidate"], rr_val_selected=val)
            self.assertEqual(q.paired_screen("s3", "A1", rows)["status"], "HQNR_ONLY_TRADEOFF")

    def test_joint_single_is_retest_not_default_promotion(self):
        rows = pairs(h=.954)
        rows[0]["candidate"] = replace(rows[0]["candidate"], exact50k=metrics(.965, .551))
        r = q.paired_screen("s3", "A1", rows)
        self.assertEqual(r["status"], "JOINT_SINGLE_RETEST")
        self.assertFalse(all(r["criteria"].values()))

    def test_target_only_joint_allows_retest_never_promising_or_confirmation(self):
        rows = pairs(h=.954)
        rows[0]['candidate'] = replace(rows[0]['candidate'], development_target=metrics(.965, .551))
        receipt = q.paired_screen('s3', 'A1', rows)
        self.assertEqual(receipt['status'], 'JOINT_SINGLE_RETEST')
        self.assertEqual(receipt['development_target_joint_seeds'], [93001])
        self.assertEqual(receipt['exact50k']['joint_seed_count'], 0)
        self.assertFalse(receipt['criteria']['exact_h'])
        invalid = rows.copy()
        invalid[0] = dict(rows[0], candidate=replace(rows[0]['candidate'], development_target={'HQNR': .99}))
        self.assertEqual(q.paired_screen('s3', 'A1', invalid)['status'], 'TECHNICAL_INVALID')

    def test_joint_requires_same_valid_point_strict_goal(self):
        self.assertFalse(q.joint(metrics(.964, .551)))
        self.assertFalse(q.joint(metrics(.965, .552)))
        self.assertTrue(q.joint(metrics(.965, .551)))
        self.assertEqual(q.joint_flags(metrics(.965, .521)), dict(joint=True, strong=True))

    def test_choice_promising_first_then_prespecified_tie(self):
        a1 = q.paired_screen("s3", "A1", pairs())
        a9 = q.paired_screen("s3", "A9", pairs(candidate="A9"))
        d = q.choose_screen("s3", [a9, a1], "2026-09-20T01:00:00Z")
        self.assertEqual(d["selected_candidate"], "A1")
        self.assertTrue(d["locked"])
        self.assertEqual(q.choose_screen("s3", [], "2026-09-20T02:00:00Z", previous=d), d)
        self.assertEqual(len(p.resolve_confirmation("s3", d, {"R0": SHA})), 4)

    def test_no_candidate_no_base_seed_fallback(self):
        receipts = [q.paired_screen("s3", c, pairs(candidate=c, h=.954)) for c in ("A1", "A9")]
        d = q.choose_screen("s3", receipts, "2026-09-20T01:00:00Z")
        self.assertIsNone(d["selected_candidate"])
        with self.assertRaises(ValueError):
            p.resolve_confirmation("s3", d, {"R0": SHA})

    def test_choice_waits_both_directions_except_budget_skipped_r3(self):
        a1 = q.paired_screen("s3", "A1", pairs())
        with self.assertRaises(ValueError):
            q.choose_screen("s3", [a1], "2026-09-20T01:00:00Z")
        r4 = q.paired_screen("s2", "R4", pairs("s2", "R4"))
        d = q.choose_screen("s2", [r4], "2026-09-20T01:00:00Z", not_admitted={"R3": SHA})
        self.assertEqual(d["selected_candidate"], "R4")

    def test_forged_receipt_rejected(self):
        r = q.paired_screen("s3", "A1", pairs(h=.954))
        r["status"], r["eligible"] = "PROMISING_PAIRED", True
        with self.assertRaises(ValueError):
            q.choose_screen("s3", [r], "2026-09-20T01:00:00Z")

    def test_four_seed_confirmation_and_joint_independent(self):
        rows = pairs(seeds=(*p.SCREEN_SEEDS, *p.CONFIRM_SEEDS))
        result = q.confirm_gain("s3", "A1", rows)
        self.assertTrue(result["reproducible_gain"])
        self.assertFalse(result["joint_reproduced"])
        for row in rows[-2:]:
            row["candidate"] = replace(row["candidate"], rr_val_selected=metrics(.965, .551))
        result = q.confirm_gain("s3", "A1", rows)
        self.assertTrue(result["joint_reproduced"])
        self.assertEqual(result["status"], "JOINT_REPRODUCED")
        self.assertEqual(q.confirm_gain("s3", "A1", rows[:2])["status"], "TECHNICAL_INVALID")

    def test_three_of_four_and_new_seed_positive(self):
        rows = pairs(seeds=(*p.SCREEN_SEEDS, *p.CONFIRM_SEEDS), h=.959)
        rows[-1]["candidate"] = replace(rows[-1]["candidate"], exact50k=metrics(.954, .559))
        self.assertTrue(q.confirm_gain("s3", "A1", rows)["reproducible_gain"])
        rows[-2]["candidate"] = replace(rows[-2]["candidate"], exact50k=metrics(.954, .559))
        self.assertFalse(q.confirm_gain("s3", "A1", rows)["reproducible_gain"])

    def test_transfer_requires_two_screen_passes_and_original_server(self):
        ref = q.paired_screen("s1", "R1", pairs("s1", "R1"))
        student = q.paired_screen("s4", "E1", pairs("s4", "E1"))
        receipt = q.lock_transfer(ref, student, "2026-09-20T01:00:00Z", reference_id="R1",
                                  reference_sha256=SHA, local_controls_verified=True,
                                  pair_identity_verified=True, evidence_files={"controls": SHA})
        cases = p.resolve_transfer(receipt, {"R1": SHA})
        self.assertEqual({c.server_id for c in cases}, {"s4"})
        with self.assertRaises(ValueError):
            q.lock_transfer(ref, student, "2026-09-20T01:00:00Z", reference_id="R2",
                            reference_sha256=SHA, local_controls_verified=True,
                            pair_identity_verified=True, evidence_files={"controls": SHA})


if __name__ == "__main__":
    unittest.main()
