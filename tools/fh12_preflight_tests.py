#!/usr/bin/env python3
"""Synthetic/mocked FH12 preflight tests; no real cache, GPU or window activation."""
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fh12.common import sha256
from fh12.data import DEFAULT_PATHS
from fh12.plan import CAMPAIGN_ID, SOURCE_PLAN
from tools.gen_fh12_configs import artifacts
from tools import fh12_preflight as P


class FH12PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fh12-preflight-test-")
        self.root = Path(self.tmp.name)
        for name, content in artifacts().items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        spec = self.root / SOURCE_PLAN
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text("synthetic preflight fixture; never used for production")
        hashes = {}
        for split, relative in DEFAULT_PATHS.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            size = 512 if split == "fr" else 256 if split == "rr" else 64
            n = 20 if split in ("rr", "fr") else 2
            with h5py.File(path, "w") as stream:
                stream.create_dataset("pan", shape=(n, 1, size, size), dtype="float32")
                stream.create_dataset("ms", shape=(n, 8, size // 4, size // 4), dtype="float32")
                stream.create_dataset("lms", shape=(n, 8, size, size), dtype="float32")
                if split != "fr":
                    stream.create_dataset("gt", shape=(n, 8, size, size), dtype="float32")
            hashes[P.HASH_KEYS[split]] = sha256(path)
        recipe = self.root / "assets/mix20h/reference_recipe.json"
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(json.dumps(dict(dataset_hashes=hashes)))
        self.deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat()

    def tearDown(self):
        self.tmp.cleanup()

    def activate_fixture(self, device="cpu"):
        path = self.root / "work_dir/_fh12/s1/window.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(campaign_id=CAMPAIGN_ID, server_id="s1", device=device, deadline_utc=self.deadline)))
        return path

    def test_read_only_without_window_reports_assets_and_writes_nothing(self):
        files_before = sorted(str(p) for p in self.root.rglob("*"))
        with patch.object(P, "prepare_data", side_effect=AssertionError("must not generate")), patch.object(P, "frontend_self_test", side_effect=AssertionError("must not run CUDA")):
            report = P.run_preflight(self.root, "s1", "cuda", check_only=True)
        self.assertEqual(report["status"], "CHECK_ONLY_NEEDS_PREPARATION")
        self.assertFalse(report["preflight_pass"])
        self.assertFalse(report["runtime_numerics_executed"])
        self.assertEqual(report["sources"]["splits"]["rr"]["count"], 20)
        self.assertEqual(report["registry"]["config_count"], 21)
        self.assertEqual(files_before, sorted(str(p) for p in self.root.rglob("*")))

    def test_no_implicit_window_activation(self):
        with self.assertRaisesRegex(ValueError, "no activated window"):
            P.run_preflight(self.root, "s1", "cpu")
        self.assertFalse((self.root / "work_dir").exists())

    def test_wrong_config_in_another_server_is_rejected(self):
        path = next((self.root / "config").glob("FH12_S5_T_*.yaml"))
        path.write_text(path.read_text().replace("batch_size: 48", "batch_size: 24"))
        with self.assertRaisesRegex(ValueError, "registry/configs"):
            P.run_preflight(self.root, "s1", "cpu", check_only=True)

    def test_wrong_native_hash_rejected_before_cache_generation(self):
        path = self.root / DEFAULT_PATHS["train"]
        with h5py.File(path, "r+") as stream:
            stream.attrs["tamper"] = "changed"
        self.activate_fixture()
        with patch.object(P, "prepare_data") as prepare:
            with self.assertRaisesRegex(ValueError, "native data SHA"):
                P.run_preflight(self.root, "s1", "cpu")
            prepare.assert_not_called()
        report = json.loads((self.root / "work_dir/_fh12/s1/preflight_report.json").read_text())
        self.assertFalse(report["preflight_pass"])

    def test_deadline_and_device_are_immutable(self):
        path = self.activate_fixture()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "deadline"):
            P.run_preflight(self.root, "s1", "cpu", "2099-01-01T00:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "device"):
            P.run_preflight(self.root, "s1", "cuda")
        self.assertEqual(before, path.read_bytes())

    def test_cuda_unavailable_is_not_silently_skipped(self):
        self.activate_fixture("cuda")
        with patch.object(P.torch.cuda, "is_available", return_value=False), patch.object(P, "prepare_data") as prepare:
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                P.run_preflight(self.root, "s1", "cuda")
            prepare.assert_not_called()

    def test_mocked_success_reports_actual_device_without_activating_new_clock(self):
        window = self.activate_fixture()
        before = window.read_bytes()
        with patch.object(P, "prepare_data", return_value={"fixture": True}) as prepare, \
             patch.object(P, "validate_prepared", return_value={"complete": True, "missing": []}), \
             patch.object(P, "frontend_self_test", return_value={"passed": True}) as frontend, \
             patch.object(P, "coupled_model_checks", return_value={"passed": True}), \
             patch.object(P, "numerical_routing_checks", return_value={"passed": True}), \
             patch.object(P, "training_smoke", return_value={"passed": True, "smoke_GPU": "not_run"}) as smoke, \
             patch.object(P, "evaluator_smoke", return_value={"passed": True}) as evaluator, \
             patch.object(P, "source_identity", return_value={"test": "mocked"}):
            report = P.run_preflight(self.root, "s1", "cpu", self.deadline)
        self.assertEqual(report["execution_device"], "cpu")
        self.assertTrue(report["preflight_pass"])
        prepare.assert_called_once_with(self.root, "s1", deadline_utc=self.deadline)
        frontend.assert_called_once_with(device="cpu")
        smoke.assert_called_once_with("s1", "cpu", data={"fixture": True}, deadline=self.deadline)
        evaluator.assert_called_once_with({"fixture": True}, deadline=self.deadline)
        self.assertEqual(window.read_bytes(), before)
        self.assertTrue((window.parent / "preflight_report.json").is_file())

    def test_production_coupled_checks_and_cpu_routing_execute(self):
        P.torch.set_num_threads(2)
        report = P.coupled_model_checks("s1")
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["all_layouts"]), 4)
        routing = P.numerical_routing_checks()
        self.assertTrue(routing["passed"])
        self.assertEqual(routing["n_tests"], 6)

    def test_cpu_training_smoke_runs_two_teacher_and_one_routed_student_updates(self):
        P.torch.set_num_threads(2)
        report = P.training_smoke("s4", "cpu")
        self.assertTrue(report["passed"])
        self.assertEqual(report["smoke_GPU"], "not_run")
        self.assertFalse(report["production_batch_and_architecture"])
        self.assertEqual(report["teacher_updates"], 2)
        self.assertEqual(report["student_updates"], 1)
        self.assertEqual(report["batch_size"], 2)
        self.assertTrue(report["teacher_frozen_verified"])
        self.assertIn("W112_D121", report["student_run"])

    def test_training_smoke_does_not_fallback_from_cuda(self):
        with patch.object(P.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "unavailable CUDA"):
                P.training_smoke("s1", "cuda")

    def test_largest_student_includes_local_reserve(self):
        self.assertEqual(P._largest_student("s1").input_layout, "PLH")
        self.assertEqual(P._largest_student("s2").input_layout, "PL")
        self.assertEqual(P._largest_student("s3").width, 112)
        self.assertEqual(P._largest_student("s4").width, 112)
        self.assertEqual(P._largest_student("s5").depth, (1, 2, 2))

    def test_evaluator_dependency_failure_is_detected_before_training(self):
        with patch("tools.metrics.eval_fr.load_dlpan", side_effect=FileNotFoundError("missing DLPan")):
            with self.assertRaisesRegex(FileNotFoundError, "missing DLPan"):
                P.evaluator_smoke({"splits": {}})

    def test_one_scene_official_evaluator_smoke_is_not_a_result(self):
        rng = np.random.default_rng(9)
        rr_path, fr_path = (self.root / DEFAULT_PATHS[key] for key in ("rr", "fr"))
        with h5py.File(rr_path, "r+") as f:
            gt = rng.uniform(100, 900, (8, 256, 256)).astype(np.float32)
            f["gt"][0] = gt
            f["lms"][0] = gt + rng.uniform(-20, 20, gt.shape)
        with h5py.File(fr_path, "r+") as f:
            f["lms"][0] = rng.uniform(100, 900, (8, 512, 512))
            f["pan"][0] = rng.uniform(100, 900, (1, 512, 512))
        data = {"splits": {key: {"dataroot": str(self.root / DEFAULT_PATHS[key])} for key in ("rr", "fr")}}
        wald = SimpleNamespace(interp23tap=lambda x, ratio: np.repeat(np.repeat(x, ratio, 0), ratio, 1))
        with patch("tools.metrics.eval_fr.load_dlpan", return_value=wald), \
             patch("tools.metrics.eval_fr.mtf_filter", side_effect=lambda x, *a: x):
            result = P.evaluator_smoke(data)
        self.assertTrue(result["passed"])
        self.assertFalse(result["official_result"])
        self.assertEqual(result["n_rr"], 1)
        self.assertEqual(result["n_fr"], 1)
        self.assertTrue(np.isfinite(result["rr"]["ergas"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
