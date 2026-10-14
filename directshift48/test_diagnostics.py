"""CPU diagnostic completeness, meaning and state-preservation tests."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

import torch

from .diagnostics import (axis20, benchmark_inference, response_statistics,
                          run_probes, validate_probe_spec)
from .evaluation import model_state_hash
from .test_method import batch
from .test_training import trainer


class Sources:
    sensor, bands, peak = "GF2", 4, 1023.

    def __init__(self):
        self.calls = []

    def manifest_sha(self, split):
        return "a"*64

    def batch(self, split, rows, *, device):
        self.calls.append((split, tuple(rows)))
        sample = batch(side=32, sensor="GF2", batch_size=len(rows),
            resolution="FR" if split == "fr" else "RR",
            split={"train": "train", "val": "validation", "rr": "test", "fr": "test"}[split])
        return replace(sample, lineage=replace(sample.lineage,
            sample_ids=tuple("GF2:%s:%d" % (split, i) for i in rows)))


def spec(gradients=True):
    value = {"rows": {"train": [0], "val": [0], "rr": [0], "fr": [0]},
             "max_wall_seconds": 60, "response_radii": [.25, .5, 1., 2., 3.]}
    if gradients:
        value.update(gradient_rows=[[0, 1]], gradient_microbatch=1, probe_seed=91)
    return value


class DirectDiagnosticsTests(unittest.TestCase):
    def test_axis20_and_response_gain_sign(self):
        points = axis20()
        self.assertEqual(len(points), 20)
        self.assertEqual({row["radius"] for row in points}, {.25, .5, 1., 2., 3.})
        ideal = [{**r, "du": [-v for v in r["epsilon_dy_dx"]]} for r in points]
        measures = response_statistics(ideal)
        self.assertEqual(measures["gain"], 1.)
        self.assertEqual(measures["EPE"], 0.)
        self.assertEqual(measures["MAE"], 0.)
        zero = response_statistics([{**r, "du": [0., 0.]} for r in points])
        self.assertEqual(zero["gain"], 0.)
        self.assertAlmostEqual(zero["EPE"], 1.35)

    def test_probes_preserve_all_training_state_and_keep_FR_diagnostic_only(self):
        run, sources = trainer(), Sources()
        first = next(run.stage.parameters())
        first.grad = torch.ones_like(first)
        old_grad = first.grad.clone()
        old_rng = torch.random.get_rng_state().clone()
        old_eps = run.stage._epsilon_rng.get_state().clone()
        old_hash = model_state_hash(run.stage)
        before_opt = copy.deepcopy(run.optimizer.state_dict())
        result = run_probes(run, sources, spec(), device="cpu", evaluator_pin={"revision": "unit"})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["completed_input_count"], 4)
        self.assertEqual(result["model_state_sha256"], old_hash)
        self.assertEqual(result["model_state_sha256_after"], old_hash)
        self.assertEqual(run.updates, 0)
        self.assertEqual(run.stage.forward_calls, 0)
        self.assertTrue(run.stage.training)
        self.assertTrue(torch.equal(first.grad, old_grad))
        self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(old_eps, run.stage._epsilon_rng.get_state()))
        self.assertEqual(run.optimizer.state_dict(), before_opt)
        for split, response in result["response"].items():
            self.assertEqual(len(response["rows"]), 20)
            self.assertEqual(response["summary"]["all"]["gain"], 0.)
            self.assertEqual(set(response["summary"]["by_radius"]), {"0.25", "0.5", "1.0", "2.0", "3.0"})
        self.assertFalse(result["physical_displacement_ground_truth"])
        self.assertFalse(result["full_method_eligible"])
        self.assertFalse(result["metric_evaluator_invoked"])
        self.assertEqual(result["gradient"]["n_inputs"], 2)
        self.assertFalse(result["gradient"]["uses_evaluation_inputs"])
        grads = result["gradient"]["term_gradient_norm"]
        self.assertEqual(grads["weighted_abs"]["teacher"], 0.)
        self.assertEqual(grads["weighted_eps_rr"]["teacher"], 0.)
        self.assertGreater(grads["weighted_eps_rr"]["aligner"], 0.)
        self.assertNotIn("RGB_GT", result["geometry"]["fr"]["rows"][0]["discrepancy"]["CURRENT"])
        json.dumps(result, allow_nan=False)

    def test_deadline_returns_explicit_incomplete_and_preserves_weights(self):
        run, sources = trainer(), Sources()
        initial = model_state_hash(run.stage)
        result = run_probes(run, sources, spec(), deadline_unix=time.time()-1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["completed_input_count"], 0)
        self.assertEqual(model_state_hash(run.stage), initial)
        self.assertEqual(sources.calls, [])
        self.assertIn("BUDGET", result["incomplete_reason"])

    def test_partial_probe_failure_also_restores_mode_and_rng(self):
        run = trainer()
        eps = run.stage._epsilon_rng.get_state().clone()
        with patch("directshift48.diagnostics._check_deadline", side_effect=[None, None, TimeoutError("partial")]):
            result = run_probes(run, Sources(), spec(gradients=False))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["response"]["train"]["rows"]), 1)
        self.assertIsNone(result["response"]["train"]["summary"])
        self.assertTrue(run.stage.training)
        self.assertTrue(torch.equal(eps, run.stage._epsilon_rng.get_state()))

    def test_benchmark_measured_total_actual_params_and_state_preserved(self):
        run = trainer()
        before = model_state_hash(run.stage)
        result = benchmark_inference(run, Sources(), device="cpu", warmup=0, repeats=2,
                                     rows={"rr": 0, "fr": 0})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["model_state_sha256_after"], before)
        self.assertEqual(result["trainable_params"], sum(p.numel() for p in run.stage.parameters()))
        self.assertEqual(result["trainable_params"], result["aligner_params"]+result["teacher_params"])
        for value in result["observations"].values():
            self.assertEqual(len(value["full_pipeline"]["samples_ms"]), 2)
            self.assertGreater(value["full_pipeline"]["median_ms"], 0.)
            self.assertNotIn("EdgeMatch_only", value)
        self.assertIsNone(result["FLOPs"])
        self.assertIsNone(result["Mem_MB"])
        json.dumps(result, allow_nan=False)

    def test_invalid_spec_and_benchmark_deadline(self):
        bad = spec()
        bad["response_radii"] = [.25, .5, 1., 2.]
        with self.assertRaises(ValueError):
            validate_probe_spec(bad)
        bad = spec()
        bad["rows"]["rr"] = [20]
        with self.assertRaises(ValueError):
            validate_probe_spec(bad)
        result = benchmark_inference(trainer(), Sources(), device="cpu", warmup=0, repeats=1,
                                     rows={"rr": 0, "fr": 0}, deadline_unix=time.time()-1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["observations"], {})


if __name__ == "__main__":
    unittest.main()
