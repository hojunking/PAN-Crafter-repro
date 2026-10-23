"""Sampling-specific deployment measurement and immutable cost evidence tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment_cost.test_store import measurement as old_measurement
from panda_next30 import plan, cost as C, best_publication as B
from panda_next30.contract import sampling_manifest
from panda_next30.cost.profiling import build_deployment, registered_profiles, account_student_lanczos4
from panda_next30.cost.timing import timing_for_config
from panda_next30.reconstruction import assert_sampling_contract
from panda_next30.tests.test_publication import persist


def measurement(cfg, step=1000):
    """Clearly synthetic proof for offline validation; never a production profile."""
    sig = C.signature(cfg, step)
    value = old_measurement(cfg["server"])
    value["deployment_signature_sha256"] = C.R.digest(sig)
    value["gpu_isolation_assertion"].update(
        mode="STANDALONE_ISOLATED_NO_TRAINER_GPU", training_state_retained_in_memory=False)
    value["deployment"] = {
        "phase": sig["phase"], "geometry_key": sig["geometry_key"],
        "pan_shift_scale": sig["pan_shift_scale"], "pan_execution": sig["pan_execution"],
        "student_sampling": deepcopy(sig["student_sampling"]),
        "reconstructor_sampling": deepcopy(sig["reconstructor_sampling"]),
        "student_input": deepcopy(sig["student_input"]), "student_warp": deepcopy(sig["student_warp"]),
        "reconstructor_input": deepcopy(sig["reconstructor_input"]),
        "training_teacher_included": False if sig["phase"] == "stage2" else None,
        "graph": "A+S" if sig["phase"] == "stage2" else "A+T",
    }
    component = "student" if sig["phase"] == "stage2" else "teacher"
    value["parameter_components"] = {
        "aligner": 105458, component: sig["reconstructor_sampling"]["student_parameters"]}
    value["params"] = sum(value["parameter_components"].values())
    for split in ("rr", "fr"):
        value["observations"][split]["full_pipeline"] = {"samples_ms": [12.] * 10}
        value["observations"][split]["sheet_fields"]["Params(M)"] = value["params"] / 1e6
    return value


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_profiles_cover_all_actual_operators_and_share_only_identical_graphs(self):
        for server in ("s1", "s3"):
            profiles = registered_profiles(server)
            cases = [plan.resolve_case(server, case, replica) for group in plan.groups(server)
                     for case, replica in group["runs"]]
            unique = {(cfg["policy_id"], cfg["sampling_variant"], cfg["input_variant"]) for cfg in cases}
            self.assertEqual(len(profiles), 1 + len(unique))
            self.assertEqual(profiles[0][0]["phase"], "donor")
            signatures = []
            for cfg, step in profiles[1:]:
                sig = C.signature(cfg, step)
                signatures.append(C.R.digest(sig))
                equivalent = [c for c in cases if (c["policy_id"], c["sampling_variant"], c["input_variant"])
                              == (cfg["policy_id"], cfg["sampling_variant"], cfg["input_variant"])]
                for other in equivalent:
                    self.assertEqual(sig, C.signature(other, other["planned_horizon"]))
                self.assertEqual(sig["student_sampling"], cfg["student_sampling"])
                self.assertEqual(sig["student_input"], cfg["student_input"])
                self.assertEqual(sig["student_warp"], cfg["student_warp"])
                self.assertIn("panda_next30/reconstruction.py", sig["source_sha256"])
                self.assertIn("panda_next30/warp.py", sig["source_sha256"])
            self.assertEqual(len(set(signatures)), len(unique))
        up = C.signature(plan.resolve_case("s1", "UBIL"), 1000)
        both = C.signature(plan.resolve_case("s1", "DAVG_UBIL"), 1000)
        self.assertEqual(up["student_sampling"]["student_parameters"],
                         both["student_sampling"]["student_parameters"])
        self.assertNotEqual(up, both)
        base = C.signature(plan.resolve_case("s1", "S100C30"), 1000)
        diff = C.signature(plan.resolve_case("s1", "S100DIFFC30"), 1000)
        self.assertEqual(diff["student_sampling"]["student_parameters"] -
                         base["student_sampling"]["student_parameters"], 1008)
        self.assertNotEqual(base, diff)
        self.assertEqual(base["student_warp"]["method"], "lanczos4")
        donor = C.signature(plan.resolve_case("s1", plan.DONORS[0]), 1000)
        self.assertIsNone(donor["student_warp"])
        self.assertIn("BICUBIC", donor["pan_execution"])

    def test_cpu_construction_is_actual_variant_and_excludes_frozen_training_teacher(self):
        for case in ("S020C30", "S100C30", "S100DIFFC30", "DAVG", "UBIL", "DAVG_UBIL"):
            cfg = plan.resolve_case("s1", case)
            model = build_deployment(cfg, None, 1000)
            actual = assert_sampling_contract(model.inference.student, cfg["sampling_variant"], input_variant=cfg["input_variant"])
            self.assertEqual(actual["parameter_count"], cfg["student_sampling"]["student_parameters"])
            self.assertIs(model.teacher, model.inference.student)
            self.assertEqual(list(model._modules), ["inference"])
            self.assertEqual(set(model.inference._modules), {"aligner", "student"})

    def test_same_parameter_count_does_not_admit_base_cost_for_davg(self):
        cfg = plan.resolve_case("s1", "DAVG")
        sig, value = C.signature(cfg, 1000), measurement(cfg)
        C.validate_measurement(value, sig)
        changed = deepcopy(value)
        changed["deployment"]["student_sampling"] = sampling_manifest("BASE")
        changed["deployment"]["reconstructor_sampling"] = sampling_manifest("BASE")
        self.assertEqual(changed["params"], value["params"])
        with self.assertRaisesRegex(ValueError, "sampling/scale/graph"):
            C.validate_measurement(changed, sig)
        wrong = deepcopy(cfg)
        wrong["student_sampling"] = sampling_manifest("BASE")
        with self.assertRaises(ValueError):
            C.signature(wrong, 1000)

    def test_actual_component_count_and_sampler_profile_integrity_are_required(self):
        cfg = plan.resolve_case("s1", "UBIL")
        sig, value = C.signature(cfg, 1000), measurement(cfg)
        for key, bad in (("student", 2658888), ("aligner", 0)):
            changed = deepcopy(value)
            changed["parameter_components"][key] = bad
            with self.assertRaisesRegex(ValueError, "parameter components"):
                C.validate_measurement(changed, sig)
        with tempfile.TemporaryDirectory() as root:
            saved = C.save_profile(cfg, 1000, root, value)
            self.assertEqual(C.find_profile(cfg, 30000, root), saved)
            self.assertIsNone(C.find_profile(plan.resolve_case("s1", "DAVG_UBIL"), 1000, root))
            changed_source = deepcopy(sig)
            changed_source["source_sha256"]["panda_next30/reconstruction.py"] = "a" * 64
            with patch.object(C, "signature", return_value=changed_source):
                self.assertIsNone(C.find_profile(cfg, 1000, root))
            self.assertEqual(len(list((Path(root) / "cost_profiles").glob("*.json"))), 1)

    def test_no_profile_blocks_cost_required_publication_without_fabricated_values(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, "DAVG")
            value = B.project(root, "s1", require_cost=True)
            self.assertTrue(value["conflicts"])
            self.assertEqual(value["cost"]["missing"][0]["reason"], "NO_VERIFIED_NEXT30_COST")
            self.assertEqual(value["rows"][0][C.R.INDEX["Params(M)"]], "")

    def test_lanczos_partial_mac_estimate_is_added_once_and_keeps_timing(self):
        value = {"observations": {"rr": {"pan_shape": [1, 1, 256, 256],
            "accounting": {"components_MAC_equivalent": {"convolution_MAC": 100},
                "mac_equivalent": 100, "arithmetic_flops_covered_estimate": 200, "omissions": []},
            "sheet_fields": {"FLOPs(G)": 100 / 1e9, "Cost_scope": "partial", "Infer(ms)": 7.5}}}}
        account_student_lanczos4(value)
        obs = value["observations"]["rr"]
        self.assertEqual(obs["accounting"]["mac_equivalent"], 100 + 16 * 256 * 256)
        self.assertEqual(obs["sheet_fields"]["Infer(ms)"], 7.5)
        self.assertEqual(obs["sheet_fields"]["FLOPs(G)"], (100 + 16 * 256 * 256) / 1e9)
        with self.assertRaises(ValueError):
            account_student_lanczos4(value)

    def test_fresh_timing_delegates_but_inherited_student_is_rejected(self):
        cfg = plan.resolve_case("s1", "DAVG")
        with patch("panda_next30.cost.timing.original_timing", return_value={"observed": True}) as original:
            self.assertEqual(timing_for_config(cfg, 1000, "/tmp"), {"observed": True})
            original.assert_called_once()
            changed = deepcopy(cfg)
            changed["inherited_updates"] = 25000
            with self.assertRaises(ValueError):
                timing_for_config(changed, 30000, "/tmp")


if __name__ == "__main__":
    unittest.main()
