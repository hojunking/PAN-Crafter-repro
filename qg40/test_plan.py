"""CPU-only registry and immutable sensor/config contract checks."""
from dataclasses import FrozenInstanceError, replace
import itertools
import unittest

from qg40.plan import (BASELINE_CASES, CASES, GRID_STEPS, SERVERS, SensorSpec,
                       active_cases, blocks_for, build_config, case_for, cases_for,
                       registry_document, registry_sha256, sensor_spec, teacher_for,
                       verify_sources)


class PlanTests(unittest.TestCase):
    def test_exact_sources_and_preserved_ids(self):
        self.assertEqual(len(verify_sources()), 4)
        self.assertEqual(len(CASES), 89)
        self.assertEqual(len({case.run_id for case in CASES}), 89)
        self.assertEqual(len(BASELINE_CASES), 11)
        self.assertEqual(sum(case.role == "T" for case in BASELINE_CASES), 3)
        self.assertEqual([len(cases_for(server)) for server in SERVERS], [14, 21, 18, 15, 21])
        self.assertTrue(all(case.run_id.startswith("QGBASE_") for case in BASELINE_CASES))
        self.assertEqual(teacher_for("s2"), teacher_for("QB_TA"))
        self.assertEqual(teacher_for("s5").server_id, "s3")
        self.assertEqual(len(registry_document()["actions"]), 7)
        self.assertEqual(len(registry_sha256()), 64)
        self.assertIn("Preparation", registry_document()["source_note"])
        self.assertFalse(any("QG40I_" in case.run_id for case in CASES))
        self.assertEqual(teacher_for("s2").server_id, "s4")
        for ref in ("QB_TC", "GF2_TB"):
            with self.assertRaises(ValueError):
                teacher_for(ref)

    def test_all_legal_branch_counts_and_crossed_pairs(self):
        for s2, s5 in itertools.product(("BASE", "A24R", "B20", "E10"), repeat=2):
            counts = []
            for server in SERVERS:
                profile = s2 if server == "s2" else s5 if server == "s5" else "BASE"
                active = active_cases(server, screen=profile, confirm=profile)
                counts.append(len(active))
                for kind in ("SCREEN", "CONFIRM"):
                    pairs = [block for block in blocks_for(server, screen=profile, confirm=profile)
                             if kind in block.block_id]
                    if pairs:
                        expected = [("BASE",), ("BASE",)] if profile == "BASE" else [
                            ("BASE", profile), (profile, "BASE")]
                        self.assertEqual([tuple(case_for(run).profile for run in block.run_ids)
                                          for block in pairs], expected)
            self.assertEqual(sum(counts), 31 + 4 * (s2 != "BASE") + 4 * (s5 != "BASE"))
        maximum = sum(len(active_cases(s, screen="A24R" if s in ("s2", "s5") else "BASE",
                                      confirm="A24R" if s in ("s2", "s5") else "BASE",
                                      enable_c3=s in ("s1", "s3"), include_reserve=True)) for s in SERVERS)
        self.assertEqual(maximum, 73)
        self.assertEqual(sum(case.tier == "TIME_RESERVE" for case in CASES), 28)

    def test_invalid_branch_and_mutation_rejected(self):
        for kwargs in (dict(screen="B20", confirm="E10"), dict(screen="UNKNOWN")):
            with self.assertRaises(ValueError):
                active_cases("s2", **kwargs)
        with self.assertRaises(ValueError):
            active_cases("s4", screen="B20")
        with self.assertRaises(ValueError):
            active_cases("s5", enable_c3=True)
        with self.assertRaises(FrozenInstanceError):
            CASES[0].profile = "C3"
        with self.assertRaises(ValueError):
            build_config(replace(CASES[0], width=999))

    def test_sensor_constants_and_no_fallback(self):
        qb, gf2 = sensor_spec("QB"), sensor_spec("GF2")
        self.assertEqual((qb.num_bands, qb.max_dn, qb.inverse_scale), (4, 2047, 1023.5))
        self.assertEqual((gf2.num_bands, gf2.max_dn, gf2.inverse_scale), (4, 1023, 511.5))
        self.assertFalse(gf2.is_bound)
        self.assertIsNone(gf2.band_order)
        with self.assertRaises(ValueError):
            gf2.split("train")
        with self.assertRaises(ValueError):
            sensor_spec("WV3")
        with self.assertRaises(ValueError):
            replace(gf2, max_dn=2047)
        bindings = {name: dict(path=f"/tmp/qg40_test/{name}.h5", sha256="1" * 64,
                               source_identity="test-source") for name in ("train", "val", "rr", "fr")}
        bound = gf2.bind(band_order=("B", "G", "R", "NIR"), splits=bindings,
                         source_provenance={"units": "DN"}, verify_files=False)
        self.assertTrue(bound.is_bound)
        self.assertEqual(SensorSpec.from_dict(bound.to_dict()), bound)
        self.assertEqual(bound.split("valid"), bound.split("val"))
        with self.assertRaises(ValueError):
            gf2.bind(band_order=bound.band_order, splits=bindings, source_provenance={})

    def test_every_config_keeps_recipe_and_unmeasured_null(self):
        self.assertEqual(len(GRID_STEPS), 50)
        self.assertIn(24240, GRID_STEPS)
        for case in CASES:
            config = build_config(case)
            meta = config["qg40"]
            self.assertEqual((config["num_bands"], config["batch_size"], config["num_iter"]), (4, 48, 50000))
            self.assertEqual(config["max_pixel"], sensor_spec(case.sensor).max_dn)
            self.assertEqual(config["model_args"]["depth"], [1, 2, 3] if case.role == "T" else [1, 2, 2])
            for key in ("dataset_manifest", "reference_manifest", "teacher_sha256", "tau_R", "q_ref", "q_cache_sha256"):
                self.assertIsNone(meta[key])
            self.assertEqual(meta["beta"], .2 if case.profile == "B20" else .1)
            self.assertEqual(meta["lambda_E"], .001 if case.profile == "E10" else .002)
            self.assertEqual(meta["a_lr_switch_completed_updates"], 24240 if case.profile == "A24R" else None)
            if case.role == "T":
                self.assertEqual(meta["consistency_weight"], 3e-4 if case.profile == "C3" else 1e-4)
            self.assertEqual(meta["window_path"], "work_dir/_qg40/campaign_window.json")


if __name__ == "__main__":
    unittest.main()
