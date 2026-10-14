"""CPU-only cost protocol tests. Never claims production GPU costs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiment_cost.measurement import (ROOT, _counter_mode, architecture_key,
                                        architecture_signature, profile_stage)


class Sources:
    def __init__(self):
        self.calls = []

    def batch(self, split, rows, *, device):
        self.calls.append((split, rows))
        size = 16 if split == "rr" else 20
        return SimpleNamespace(pan=torch.ones(1, 1, size, size, device=device),
            ms=torch.ones(1, 4, size//4, size//4, device=device),
            native_lms=torch.ones(1, 4, size, size, device=device),
            lineage=SimpleNamespace(sample_ids=["GF2:"+split+":"+str(rows[0])]))


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.aligner = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        self.teacher = nn.Conv2d(2, 4, 1, bias=False)
        self.calls = 0

    def native_prediction(self, batch):
        self.calls += 1
        y = self.teacher(self.aligner(batch.pan))
        y = y + F.interpolate(batch.ms, scale_factor=4, mode="bicubic", align_corners=False)
        return {"y": y}


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def profile(self, stage=None, **kwargs):
        return profile_stage(stage or Stage(), Sources(), device="cpu", warmup=1, repeats=2,
            expected_shapes={"rr": (1, 1, 16, 16), "fr": (1, 1, 20, 20)}, **kwargs)

    def test_lazy_import_without_torch(self):
        code = "import sys;import experiment_cost.measurement;assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    def test_signature_ignores_optimizer_seed_but_binds_architecture_sensor_server(self):
        b = json.loads((ROOT/"config/directshift48/s1.baseline.json").read_text())
        cfg = {"baseline_resolved": b, "seed": 1, "lr_A": 1e-5, "lambda_abs": 1}
        original = architecture_key(cfg)
        changed = copy.deepcopy(cfg)
        changed.update(seed=999, lr_A=1e-2, lambda_abs=3)
        self.assertEqual(original, architecture_key(changed))
        for field, value in (("server", "s3"), ("dataset", "GF2"), ("teacher_base", "NATIVE_LMS")):
            changed = copy.deepcopy(cfg)
            changed["baseline_resolved"][field] = value
            if field == "teacher_base":
                with self.assertRaises(ValueError): architecture_key(changed)
            else:
                self.assertNotEqual(original, architecture_key(changed))
        changed = copy.deepcopy(cfg)
        changed["baseline_resolved"]["teacher_architecture"]["width"] = 96
        self.assertNotEqual(original, architecture_key(changed))

    def test_legacy_and_direct_signatures_separate(self):
        direct = json.loads((ROOT/"config/directshift48/s1.baseline.json").read_text())
        legacy = json.loads((ROOT/"config/edgeres48/s1.baseline.json").read_text())
        self.assertNotEqual(architecture_key(direct), architecture_key(legacy))
        self.assertIsNone(architecture_signature(direct)["edge_match"])
        self.assertEqual(architecture_signature(legacy)["edge_match"]["radius_rr"], 4)

    def test_operator_counter_conv_linear_functional_and_resampling(self):
        x = torch.ones(1, 2, 8, 8)
        weight = torch.ones(3, 2, 3, 3)
        mode = _counter_mode()
        with torch.inference_mode(), mode:
            z = F.conv2d(x, weight, padding=1)
            mat = torch.ones(2, 4) @ torch.ones(4, 3)
            cubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            omitted = z.abs().mean()
        report = mode.report()
        parts = report["components_MAC_equivalent"]
        self.assertEqual(parts["convolution_MAC"], z.numel()*2*3*3)
        self.assertEqual(parts["matrix_MAC"], 2*4*3)
        self.assertEqual(parts["bicubic_32op_per_output_MAC_equivalent_ESTIMATE"], cubic.numel()*16)
        self.assertFalse(report["complete_arithmetic_coverage"])
        self.assertIn("aten.abs.default", report["unaccounted_operator_calls"])
        self.assertIn("aten.mean.default", report["unaccounted_operator_calls"])

    def test_transposed_conv_and_grid_count(self):
        x = torch.ones(1, 2, 4, 4)
        weight = torch.ones(2, 3, 3, 3)
        grid = torch.zeros(1, 6, 7, 2)
        mode = _counter_mode()
        with torch.inference_mode(), mode:
            F.conv_transpose2d(x, weight, stride=2)
            out = F.grid_sample(x, grid, mode="bicubic", align_corners=False)
        parts = mode.report()["components_MAC_equivalent"]
        self.assertEqual(parts["convolution_MAC"], x.numel()*3*3*3)
        self.assertEqual(parts["grid_sample_kernel_MAC_equivalent_ESTIMATE"], out.numel()*16)

    def test_whole_pipeline_params_and_profile_scope(self):
        stage = Stage()
        p = self.profile(stage)
        self.assertEqual(p["status"], "COMPLETE")
        self.assertEqual(p["params"], 18+8)
        self.assertEqual(stage.calls, 2*(1+2+1))
        self.assertEqual(p["sheet_cost"], p["observations"]["rr"]["sheet_fields"])
        self.assertIn("fresh architecture", p["sheet_cost"]["Cost_scope"])
        self.assertIn("PARTIAL", p["sheet_cost"]["Cost_scope"])
        self.assertIsNone(p["sheet_cost"]["Mem(MB)"])
        self.assertGreater(p["sheet_cost"]["Infer(ms)"], 0)
        self.assertEqual(p["optimizer_updates"], 0)
        self.assertEqual(p["model_state_sha256_before"], p["model_state_sha256_after"])
        json.dumps(p, allow_nan=False)

    def test_no_gradients_or_mode_rng_changes(self):
        stage = Stage().train()
        stage.teacher.eval()
        rng = torch.random.get_rng_state().clone()
        p = self.profile(stage)
        self.assertTrue(stage.training)
        self.assertFalse(stage.teacher.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in stage.parameters()))

    def test_official_shapes_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "observation shape"):
            profile_stage(Stage(), Sources(), device="cpu", warmup=0, repeats=1)

    def test_invalid_repeat_count(self):
        for repeats in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                profile_stage(Stage(), Sources(), device="cpu", repeats=repeats)

    def test_cuda_requires_external_isolation_before_any_cuda_work(self):
        with self.assertRaisesRegex(ValueError, "isolation"):
            profile_stage(Stage(), Sources(), device="cuda")

    def test_deadline_is_not_complete_and_restores_mode(self):
        stage = Stage().train()
        with self.assertRaises(TimeoutError): self.profile(stage, deadline_unix=time.time()-1)
        self.assertTrue(stage.training)

    def test_model_mutation_rejected(self):
        class MutatingStage(Stage):
            def native_prediction(self, batch):
                self.aligner.weight.add_(1)
                return super().native_prediction(batch)
        with self.assertRaisesRegex(RuntimeError, "changed model"):
            self.profile(MutatingStage())

    def test_fp32_required(self):
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.profile(Stage().double())

    def test_actual_direct_and_legacy_native_paths_count_all_module_convolutions(self):
        from directshift48.test_method import batch as direct_batch, model as direct_model
        from panda_method.test_edge_guided import batch as legacy_batch, stage as legacy_model
        for factory, make_batch, legacy in ((direct_model, direct_batch, False),
                                            (legacy_model, legacy_batch, True)):
            stage = factory().eval()
            b = make_batch(side=32, batch_size=1, split="test")
            hooked = [0]
            def hook(layer, inputs, output):
                hooked[0] += (inputs[0].numel()*layer.weight[0].numel()
                              if isinstance(layer, nn.ConvTranspose2d) else output.numel()*layer.weight[0].numel())
            handles = [m.register_forward_hook(hook) for m in stage.modules()
                       if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
            mode = _counter_mode()
            try:
                with torch.inference_mode(), mode:
                    stage.native_prediction(b)
            finally:
                for handle in handles: handle.remove()
            counted = mode.report()["components_MAC_equivalent"]["convolution_MAC"]
            if legacy:
                self.assertGreater(counted, hooked[0])  # Functional Scharr matching, not hidden by module hooks.
            else:
                self.assertEqual(counted, hooked[0])
            self.assertGreater(mode.report()["components_MAC_equivalent"]["matrix_MAC"], 0)
            self.assertIn("grid_sample_kernel_MAC_equivalent_ESTIMATE", mode.report()["components_MAC_equivalent"])


if __name__ == "__main__":
    unittest.main()
