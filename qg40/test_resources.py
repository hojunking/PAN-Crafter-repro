"""Resource admission tests use mocked nvidia-smi; no GPU initialization."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qg40.common import atomic_json, camp
from qg40.plan import CASES, case_for, teacher_for
from qg40.resources import (CAPACITY_MARGIN, DISK_HEADROOM_BYTES, GIB,
                            _disk_case, _model_footprint, assess_block, inventory)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher = teacher_for("QB_TA")
        self.student = next(case for case in CASES if case.server_id == "s4" and case.role == "S")
        atomic_json(camp(self.root, "s4") / "dataset_manifest.json",
                    dict(sensor="QB", num_bands=4, splits=dict(train=dict(count=4096))))
        self.measured = dict(disk=dict(status="MEASURED", free_bytes=100 * GIB), gpu=dict(status="MEASURED",
            selected_device=dict(index=0, uuid="GPU-test", name="test", total_bytes=24 * GIB, free_bytes=20 * GIB)))
        self.footprint = dict(parameters=1000, parameter_bytes=4000, state_bytes=4000,
                              device="cpu", precision="fp32", num_bands=4)

    def assess(self, cases=None, observations=(), measured=None):
        with patch("qg40.resources.inventory", return_value=measured or self.measured), \
                patch("qg40.resources._model_footprint", return_value=self.footprint):
            return assess_block(self.root, cases or [self.teacher, self.student], observations)

    def observation(self, case, **changes):
        return dict(server_id=case.server_id, sensor=case.sensor, role=case.role, width=case.width,
                    depth=list(case.depth), profile=case.profile, component="TRAIN", completed=True,
                    peak_training_memory_bytes=8 * GIB, **changes)

    def test_inventory_readonly_disk_and_gpu_units_visibility(self):
        output = "0, GPU-zero, GPU Zero, 24576, 1000\n1, GPU-one, GPU One, 49152, 40000\n"
        with patch("qg40.resources.shutil.disk_usage", return_value=(100, 25, 75)), \
                patch("qg40.resources.subprocess.run", return_value=SimpleNamespace(stdout=output)) as run, \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,0"}):
            result = inventory(self.root)
        self.assertEqual(result["disk"]["free_bytes"], 75)
        self.assertEqual(result["gpu"]["selected_device"]["uuid"], "GPU-one")
        self.assertEqual(result["gpu"]["devices"][1]["total_bytes"], 48 * GIB)
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_inventory_unavailable_na_and_disabled_visibility_are_explicit(self):
        with patch("qg40.resources.subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            result = inventory(self.root)
        self.assertEqual(result["gpu"]["status"], "UNKNOWN")
        self.assertIsNone(result["gpu"]["selected_device"])
        with patch("qg40.resources.subprocess.run", return_value=SimpleNamespace(stdout="0, GPU-zero, Test, 24000, N/A\n")), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            result = inventory(self.root)
        self.assertIsNone(result["gpu"]["selected_device"]["free_bytes"])
        with patch("qg40.resources.subprocess.run", return_value=SimpleNamespace(stdout="0, GPU-zero, Test, 24000, 20000\n")), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "-1"}):
            result = inventory(self.root)
        self.assertIsNone(result["gpu"]["selected_device"])

    def test_exact_checkpoint_accounting_and_actual_calibration_count(self):
        with patch("qg40.resources._model_footprint", return_value=self.footprint):
            row = _disk_case(self.root, self.teacher)
        self.assertEqual(row["checkpoint_weights"], 50)
        self.assertEqual(row["candidate_fullstates"], 2)
        self.assertEqual(row["restart_fullstates"], 3)
        self.assertEqual(row["simultaneous_last_fullstates"], 2)
        self.assertEqual(row["candidate_bytes"] + row["restart_bytes"] + row["last_bytes"], 74 * 4000)
        self.assertEqual(row["calibration_bytes"], 4096 * 112 + 3072 * 8)
        self.assertEqual(row["train_count"], 4096)

    def test_unknown_vram_is_uncertified_without_cross_server_barrier(self):
        result = self.assess()
        self.assertTrue(result["allowed"])
        self.assertEqual(result["status"], "LOCAL_RESOURCE_UNCERTIFIED")
        self.assertIsNone(result["budget"]["gpu_required_bytes"])
        self.assertFalse(result["budget"]["gpu_requirement_complete"])
        self.assertTrue(result["warnings"])

    def test_matching_measured_profile_peak_and_shortage(self):
        observations = [self.observation(self.teacher), self.observation(self.student)]
        result = self.assess(observations=observations)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["budget"]["gpu_required_bytes"], 10 * GIB)
        self.assertEqual(result["status"], "LOCAL_RESOURCE_AVAILABLE")
        limited = copy.deepcopy(self.measured)
        limited["gpu"]["selected_device"]["free_bytes"] = 9 * GIB
        result = self.assess(observations=observations, measured=limited)
        self.assertFalse(result["allowed"])
        self.assertIn("INSUFFICIENT_MEASURED_PROFILE_VRAM", result["reasons"])

    def test_other_profile_sensor_or_postrun_peak_cannot_certify_training(self):
        valid = self.observation(self.teacher)
        for changes in (dict(profile="C3"), dict(sensor="GF2"), dict(server_id="s1"),
                        dict(component="EVAL"), dict(completed=False), dict(gpu_uuid="GPU-other")):
            with self.subTest(changes=changes):
                result = self.assess([self.teacher], [dict(valid, **changes)])
                self.assertIsNone(result["budget"]["gpu_required_bytes"])
        result = self.assess([self.teacher], [dict(valid, peak_training_memory_bytes=None, mem_mb=9000)])
        self.assertIsNone(result["budget"]["gpu_required_bytes"])

    def test_disk_shortage_no_gpu_and_unknown_count_hold_only_local_block(self):
        limited = copy.deepcopy(self.measured)
        limited["disk"]["free_bytes"] = DISK_HEADROOM_BYTES
        result = self.assess(measured=limited)
        self.assertFalse(result["allowed"])
        self.assertIn("INSUFFICIENT_DISK", result["reasons"])
        missing_gpu = copy.deepcopy(self.measured)
        missing_gpu["gpu"]["selected_device"] = None
        self.assertIn("NO_VISIBLE_GPU", self.assess(measured=missing_gpu)["reasons"])
        atomic_json(camp(self.root, "s4") / "dataset_manifest.json", dict(sensor="QB", num_bands=4, splits={}))
        result = self.assess()
        self.assertIn("UNKNOWN_TRAIN_COUNT_FOR_CALIBRATION_DISK", result["reasons"])
        self.assertFalse(result["budget"]["disk_estimate_complete"])
        self.assertTrue(self.assess([self.student])["allowed"])

    def test_real_cpu_footprint_does_not_initialize_cuda_or_change_cpu_rng(self):
        import torch
        torch.set_num_threads(2)
        _model_footprint.cache_clear()
        before = torch.get_rng_state().clone()
        with patch("torch.cuda.init", side_effect=AssertionError("GPU initialization forbidden")):
            result = _model_footprint("T", 8, (1, 1, 1))
        self.assertEqual(result["state_bytes"], result["parameters"] * 4)
        self.assertEqual(result["device"], "cpu")
        self.assertGreater(result["parameters"], 100000)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
