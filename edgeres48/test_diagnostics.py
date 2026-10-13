"""Bounded synthetic numeric probes; no real data, files, optimizer updates."""
import copy
import json
import unittest

import numpy as np
import torch

from .diagnostics import (axis16, benchmark_inference, gradient_probe, response_statistics,
                          response_summary, run_probes, validate_probe_spec)
from .evaluation import model_state_hash
from .test_evaluation import SyntheticSources, engine, metrics


def probe_spec():
    return {"rows": {key: [0] for key in ("train", "val", "rr", "fr")},
            "gradient_rows": [[0, 1], [2, 3]], "gradient_microbatch": 1,
            "probe_seed": 261013900, "response_domains": ["normal", "clean_support"],
            "clean_cnn_view_margin": 10, "counterfactual_metrics": "official_subset",
            "max_wall_seconds": 120}


class ResponseTests(unittest.TestCase):
    def test_axis16_integer_fractional_separation_and_shared_denominator(self):
        rows = []
        for point in axis16():
            eps = np.asarray(point["epsilon_dy_dx"])
            rows.append({**point, "db": (-.6*eps).tolist(), "dr": (-.3*eps).tolist(),
                         "du": (-.9*eps).tolist()})
        result = response_summary(rows)
        self.assertEqual(len(rows), 16)
        self.assertEqual(result["by_offset_type"]["integer"]["n"], 8)
        self.assertEqual(result["by_offset_type"]["fractional"]["n"], 8)
        self.assertAlmostEqual(result["all"]["g_edge"], .6)
        self.assertAlmostEqual(result["all"]["g_residual"], .3)
        self.assertAlmostEqual(result["all"]["g_total"], .9)
        self.assertAlmostEqual(result["all"]["gain_additivity_error"], 0)
        self.assertGreater(result["all"]["EPE_improvement_residual"], 0)

    def test_full_edge_response_is_not_reported_as_learned_residual(self):
        rows = [{**p, "db": [-x for x in p["epsilon_dy_dx"]], "dr": [0, 0],
                 "du": [-x for x in p["epsilon_dy_dx"]]} for p in axis16()]
        result = response_statistics(rows)
        self.assertEqual(result["g_edge"], 1.)
        self.assertEqual(result["g_residual"], 0.)
        self.assertEqual(result["g_total"], 1.)
        self.assertEqual(result["EPE_total"], 0.)

    def test_missing_nonfinite_and_inconsistent_decomposition_reject(self):
        with self.assertRaises(ValueError):
            response_statistics([])
        row = {"epsilon_dy_dx": [1., 0.], "db": [-1., 0.], "dr": [0., 0.], "du": [-.5, 0.]}
        with self.assertRaises(ValueError):
            response_statistics([row])
        row["du"] = [float("nan"), 0.]
        with self.assertRaises(ValueError):
            response_statistics([row])


class ProbeTests(unittest.TestCase):
    def test_full_numeric_domains_no_updates_rng_or_images(self):
        trainer, sources = engine(), SyntheticSources()
        before = model_state_hash(trainer.stage)
        rng, eps_rng = torch.random.get_rng_state().clone(), trainer.stage._epsilon_rng.get_state().clone()
        out = run_probes(trainer, sources, probe_spec(), metric_functions=metrics())
        self.assertEqual(out["status"], "COMPLETE")
        self.assertFalse(out["official_selection_input"])
        self.assertFalse(out["full_method_eligible"])
        self.assertFalse(out["images_written"])
        self.assertEqual(out["model_state_sha256"], before)
        self.assertEqual(out["model_state_sha256_after"], before)
        self.assertEqual(trainer.updates, 1)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(eps_rng, trainer.stage._epsilon_rng.get_state()))
        self.assertEqual(trainer.stage.forward_calls, 0)
        for split in ("train", "val", "rr", "fr"):
            self.assertEqual(set(out["response"][split]), {"normal", "clean_support"})
            for domain in ("normal", "clean_support"):
                section = out["response"][split][domain]
                self.assertEqual(len(section["rows"]), 16)
                self.assertEqual(section["summary"]["all"]["g_residual"], 0.)
            geom = out["geometry"][split]["rows"][0]
            self.assertIsNone(geom["cos_b_delta"])
            self.assertEqual(set(geom["discrepancy"]), {"CURRENT", "PRIOR_ONLY", "ZERO"})
        self.assertEqual(len(out["counterfactual"]), 6)
        self.assertTrue(all(not r["full_method_eligible"] for r in out["counterfactual"]))
        json.dumps(out, allow_nan=False)

    def test_effective_gradient_microbatch_mean_and_scope_norms(self):
        trainer, sources = engine(), SyntheticSources()
        original_grads = [p.grad for p in trainer.stage.parameters()]
        result = gradient_probe(trainer, sources, rows=[[0, 1]], microbatch=1, seed=103)
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["completed_effective_batches"], 1)
        self.assertEqual(set(result["scopes"]), {"whole", "fc2", "fc1", "body", "stem"})
        # Fresh Teacher's zero output projection may also make rec->A zero at
        # initialization; the direct Scharr route must still reach FC2.
        self.assertGreater(result["scopes"]["fc2"]["raw_norm"]["abs"], 0.)
        self.assertEqual(result["scopes"]["fc1"]["raw_norm"]["rec"], 0.)
        self.assertEqual(result["scopes"]["body"]["raw_norm"]["rec"], 0.)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual([p.grad for p in trainer.stage.parameters()], original_grads)

    def test_gradient_coefficients_are_weighted_but_not_renormalized(self):
        trainer, sources = engine(), SyntheticSources()
        trainer.stage.lambda_abs, trainer.stage.lambda_eps = .3, .0001
        result = gradient_probe(trainer, sources, rows=[[0]], microbatch=1, seed=31)
        for scope in result["scopes"].values():
            self.assertAlmostEqual(scope["weighted_norm"]["abs"], .3 * scope["raw_norm"]["abs"])
            self.assertAlmostEqual(scope["weighted_norm"]["eps"], .0001 * scope["raw_norm"]["eps"])

    def test_explicit_wall_budget_returns_partial_not_fake_complete(self):
        result = run_probes(engine(), SyntheticSources(), probe_spec(), metric_functions=metrics(), deadline_unix=1)
        self.assertEqual(result["status"], "PARTIAL_BUDGET")
        self.assertIn("incomplete_reason", result)

    def test_probe_spec_requires_explicit_registered_rows_and_settings(self):
        base = probe_spec()
        for key in base:
            invalid = copy.deepcopy(base)
            del invalid[key]
            with self.assertRaises(ValueError):
                validate_probe_spec(invalid)
        invalid = copy.deepcopy(base)
        invalid["rows"]["fr"] = [20]
        with self.assertRaises(ValueError):
            validate_probe_spec(invalid)

    def test_cold_matching_timing_measured_whole_pipeline(self):
        trainer = engine()
        before = model_state_hash(trainer.stage)
        result = benchmark_inference(trainer, SyntheticSources(), device="cpu", warmup=0, repeats=2,
                                     rows={"rr": 0, "fr": 0})
        self.assertEqual(model_state_hash(trainer.stage), before)
        self.assertEqual(result["edge_match_params"], 0)
        self.assertGreater(result["trainable_params"], 0)
        self.assertIsNone(result["FLOPs"])
        for split in ("rr", "fr"):
            self.assertEqual(len(result["observations"][split]["full_pipeline"]["samples_ms"]), 2)
            self.assertGreater(result["observations"][split]["full_pipeline"]["median_ms"], 0)


if __name__ == "__main__":
    unittest.main()
