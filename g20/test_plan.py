from dataclasses import asdict, replace
import unittest

from g20 import plan as p


SHA = "a" * 64


def decision(server, winner):
    return dict(campaign_id=p.CAMPAIGN_ID, server_id=server, decision="SCREEN_SELECTION",
                selected_candidate=winner, selected_status="PROMISING_PAIRED",
                selected_at_utc="2026-09-20T01:00:00+00:00", evidence_files={"measured.json": SHA})


class PlanTests(unittest.TestCase):
    def test_exact_finite_csv(self):
        self.assertEqual(len(p.CASES), 56)
        self.assertEqual(len(p.CORE_CASES), 32)
        self.assertEqual(sum(c.role == "T" for c in p.CORE_CASES), 4)
        self.assertEqual(sum(c.tier == "CONDITIONAL_CONFIRM" for c in p.CASES), 20)
        self.assertEqual(sum(c.tier == "CONDITIONAL_TRANSFER" for c in p.CASES), 4)
        self.assertEqual(len(p.verify_sources()), 2)
        self.assertEqual(p.case_for("G20-S01"), p.case_for(p.case_for("G20-S01").run_id))

    def test_every_server_is_explicit_gf2(self):
        self.assertTrue(all(c.sensor == "GF2" for c in p.CASES))
        self.assertEqual(p.SHEET_TABS["s1"], "GF2-s1")
        self.assertEqual(p.SHEET_TABS["s4"], "GF2-s4")
        for wrong in ("QB", "WV3", "gf2"):
            with self.assertRaises(ValueError):
                p.sensor_spec(wrong)

    def test_architecture_and_numerical_contract(self):
        for c in p.CORE_CASES:
            cfg = p.build_config(c)
            self.assertEqual(p.case_from_config(cfg), c)
            self.assertEqual(cfg["trainer"], "g20")
            self.assertNotIn("qg40", cfg)
            self.assertEqual(cfg["num_iter"], 50000)
            self.assertEqual(cfg["batch_size"], 48)
            self.assertEqual(cfg["mixed_precision"], "no")
            self.assertEqual(cfg["max_pixel"], 1023)
            self.assertEqual(cfg["model_args"]["hidden_size"], 112 if c.role == "T" else 104)
            self.assertEqual(cfg["model_args"]["depth"], [1, 2, 3] if c.role == "T" else [1, 2, 2])
            self.assertFalse(cfg["model_args"]["mode_modulation"])
            self.assertEqual(cfg["model_args"]["attn_locations"], [])
            self.assertEqual(cfg["g20"]["lp"]["decimation"], "2::4,2::4")
            self.assertIsNone(cfg["g20"]["tau_R"])
            self.assertIsNone(cfg["g20"]["q_ref"])

    def test_single_scalar_profiles(self):
        base = asdict(p.PROFILES["BASE"])
        expected = dict(A1=("student_a_peak_lr", 1e-6), A9=("student_a_peak_lr", 9e-6),
                        E1=("lambda_edge", .001), E4=("lambda_edge", .004),
                        K05=("beta", .05), K20=("beta", .2))
        for profile, (key, value) in expected.items():
            changed = asdict(p.PROFILES[profile])
            self.assertEqual({k for k in base if base[k] != changed[k]}, {key})
            self.assertEqual(changed[key], value)
        self.assertTrue(all(v.alpha == 1 for v in p.PROFILES.values()))

    def test_teacher_seed_and_scalar_only(self):
        for case_id, value in (("G20-T01", 1e-4), ("G20-T02", 1e-4), ("G20-T03", 3e-5), ("G20-T04", 3e-4)):
            cfg = p.build_config(case_id)
            self.assertEqual(cfg["seed"], 91002)
            self.assertEqual(cfg["g20"]["consistency_weight"], value)
            self.assertEqual(cfg["g20"]["corruption_seed"], 191002)
            self.assertEqual(cfg["g20"]["aligner_lr"], 1e-5)

    def test_student_a_lr_first_update_no_late_switch(self):
        for case_id, lr in (("G20-S12", 1e-6), ("G20-S13", 9e-6)):
            cfg = p.build_config(case_id)["g20"]
            self.assertEqual(cfg["aligner_lr"], lr)
            self.assertIsNone(cfg["a_lr_switch_completed_updates"])
            self.assertEqual(cfg["a_lr_after_multiplier"], 1.)
            self.assertEqual(cfg["consistency_weight"], 0.)

    def test_registry_readonly_profile(self):
        with self.assertRaises(TypeError):
            p.PROFILES["OTHER"] = p.PROFILES["BASE"]
        self.assertEqual(p.R0_PINS["tau_R"], .005695626139640808)
        self.assertTrue(all(p.valid_sha(s) for s in p.R0_DATA_PINS.values()))
        self.assertEqual(len(p.registry_document()["cases"]), 56)
        self.assertTrue(p.valid_sha(p.registry_sha256()))

    def test_core_package_has_all_local_reference_controls(self):
        b = p.blocks_for("s1")[0]
        self.assertEqual([c.case_id for c in b.cases], ["G20-T01", "G20-S01", "G20-S02", "G20-S03", "G20-S04"])
        self.assertEqual({(c.student_seed, c.reference_id) for c in b.cases if c.role == "S"},
                         {(93001, "R0"), (93001, "R1"), (93002, "R0"), (93002, "R1")})

    def test_s2_priority_full_comparison_precedes_c030(self):
        first, second = p.blocks_for("s2")
        self.assertEqual({c.reference_id for c in first.cases}, {"R2", "R4"})
        self.assertEqual(sum(c.role == "T" for c in first.cases), 2)
        self.assertEqual(sum(c.role == "S" for c in first.cases), 4)
        self.assertEqual({c.reference_id for c in second.cases}, {"R3"})
        self.assertEqual(len(second.cases), 3)

    def test_screen_seed_forward_reverse(self):
        for server in ("s3", "s4", "s5"):
            first, second = p.blocks_for(server)
            self.assertEqual([c.profile for c in first.cases], ["BASE", *p.AXES[server]])
            self.assertEqual([c.profile for c in second.cases], [*reversed(p.AXES[server]), "BASE"])
            self.assertEqual({c.seed for c in first.cases}, {93001})
            self.assertEqual({c.seed for c in second.cases}, {93002})

    def test_unresolved_never_defaults_to_base(self):
        for case in p.CASES:
            if case.tier != "CORE":
                self.assertIsNone(case.lambda_con)
                with self.assertRaises(ValueError):
                    p.build_config(case)

    def test_confirmation_all_servers_and_choices(self):
        for server, choices in p.AXES.items():
            for winner in choices:
                cases = p.resolve_confirmation(server, decision(server, winner), {ref: SHA for ref in p.REFERENCES})
                self.assertEqual(len(cases), 4)
                self.assertEqual({c.seed for c in cases}, {93011, 93012})
                self.assertEqual(len({c.run_id for c in cases}), 4)
                self.assertEqual(len(p.blocks_for(server, confirmation_cases=cases)[-1].cases), 4)
                for c in cases:
                    self.assertEqual(p.case_from_config(p.build_config(c)), c)

    def test_confirmation_rejects_foreign_missing_evidence(self):
        for change in (dict(selected_candidate="K20"), dict(selected_status="NO_GAIN"), dict(evidence_files={})):
            d = decision("s3", "A1") | change
            with self.assertRaises(ValueError):
                p.resolve_confirmation("s3", d, {"R0": SHA})
        with self.assertRaises(ValueError):
            p.resolve_confirmation("s3", decision("s3", "A1"), {})

    def test_resolved_scalar_and_hash_tampering_rejected(self):
        c = p.resolve_confirmation("s3", decision("s3", "A1"), {"R0": SHA})[1]
        for changed in (replace(c, beta=.2), replace(c, reference_sha256=None), replace(c, server_id="s4"), replace(c, student_seed=93013)):
            with self.assertRaises(ValueError):
                p.build_config(changed)
        with self.assertRaises(ValueError):
            p.build_config(replace(p.case_for("G20-S01"), beta=.2))

    def test_transfer_owner_single_scalar_full_four(self):
        receipt = dict(campaign_id=p.CAMPAIGN_ID, decision="TRANSFER_LOCK", server_id="s4", reference_id="R3",
                       profile="E4", selected_at_utc="2026-09-20T01:00:00Z", evidence_files={"proof": SHA},
                       reference_promising=True, student_screen_passed=True, local_controls_verified=True,
                       pair_identity_verified=True)
        cases = p.resolve_transfer(receipt, {"R3": SHA})
        self.assertEqual([(c.seed, c.profile) for c in cases], [(93001, "BASE"), (93001, "E4"), (93002, "E4"), (93002, "BASE")])
        self.assertEqual(len(p.blocks_for("s4", transfer_cases=cases)[-1].cases), 4)
        for bad in (receipt | dict(profile="A1"), receipt | dict(local_controls_verified=False), receipt | dict(server_id="s2")):
            with self.assertRaises(ValueError):
                p.resolve_transfer(bad, {"R3": SHA})
        with self.assertRaises(ValueError):
            p.blocks_for("s4", transfer_cases=cases[:2])


if __name__ == "__main__":
    unittest.main()
