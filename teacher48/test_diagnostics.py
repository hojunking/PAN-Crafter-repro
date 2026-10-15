"""Synthetic CPU probes: no campaign data, outputs, or Google access."""
import copy
from dataclasses import replace
import json
import random
import unittest

import numpy as np
import torch

from panda_method.test_stages import batch, stage1
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from .diagnostics import axis16, response_statistics, run_probes, validate_probe_spec
from .training import Trainer, TrainingConfig, model_state_hash


torch.set_num_threads(1)


class SyntheticSources:
    def __init__(self):
        self.samples = {group: [{"sample_id": f"{group}-sample", "scene_id": f"{group}-scene",
                                "source_scene_id": "synthetic-A"}]
                        for group in ("rr_eval", "fr_eval", "rr_train", "fr_train")}
        self.calls = []

    def batch(self, group, indices, *, device="cpu"):
        self.calls.append((group, tuple(indices)))
        original = batch(group[:2].upper(), split="train" if group.endswith("train") else "test")
        row = self.samples[group][0]
        lineage = replace(original.lineage, scene_ids=(row["source_scene_id"],),
                          sample_ids=(row["sample_id"],))
        tensors = {name: getattr(original, name)[:1].to(device) if getattr(original, name) is not None else None
                   for name in ("pan", "ms", "native_lms", "gt", "lpan", "prior_dy_dx")}
        return replace(original, lineage=lineage, **tensors)


def fixture(layout="P0"):
    config = TrainingConfig(seed=261013101, horizon=10, lambda_abs=.3, lambda_eps=.001,
        optimizer="AdamW", optimizer_betas=(.9, .999), optimizer_eps=1e-8,
        lr_U=1e-4, lr_A=1e-5, weight_decay=.01, warmup_updates=0,
        min_lr_U=0., min_lr_A=0., effective_batch_RR=1, effective_batch_FR=1,
        microbatch_RR=1, microbatch_FR=1, precision="fp32", scheduler="warmup_cosine")
    stage = stage1()
    if layout != "P0":
        stage.teacher = build_reconstructor(ReconstructionSpec(4, 8, (1, 1, 1), layout, "BICUBIC_MS"),
                                            seed=42, role="T")
    trainer = Trainer(stage, config)
    sources = SyntheticSources()
    baseline = {"probes": {"rr_eval_ids": ["rr_eval-scene"], "fr_eval_ids": ["fr_eval-scene"],
        "rr_train_sample_id": "rr_train-sample", "fr_train_sample_id": "fr_train-sample", "max_samples": 1}}
    return trainer, sources, baseline


class ResponseTests(unittest.TestCase):
    def test_axis16_exact_offsets(self):
        points = axis16()
        self.assertEqual(len(points), 16)
        self.assertEqual({p["radius"] for p in points}, {.25, .5, 1., 2.})
        for radius in (.25, .5, 1., 2.):
            self.assertEqual({tuple(p["epsilon_dy_dx"]) for p in points if p["radius"] == radius},
                             {(radius, 0.), (-radius, 0.), (0., radius), (0., -radius)})

    def test_ideal_gain_mae_epe(self):
        rows = [{**p, "delta_dy_dx": [-v for v in p["epsilon_dy_dx"]]} for p in axis16()]
        value = response_statistics(rows)
        self.assertEqual(value["gain"], 1.)
        self.assertEqual(value["gain_x"], 1.)
        self.assertEqual(value["gain_y"], 1.)
        self.assertEqual(value["component_mae"], 0.)
        self.assertEqual(value["epe"], 0.)

    def test_wrong_sign_has_negative_gain(self):
        rows = [{**p, "delta_dy_dx": p["epsilon_dy_dx"]} for p in axis16()]
        self.assertEqual(response_statistics(rows)["gain"], -1.)

    def test_unexcited_axis_gain_is_null_not_zero(self):
        point = axis16()[0]
        result = response_statistics([{**point, "delta_dy_dx": [0., 0.]}])
        self.assertIsNone(result["gain_x"])
        self.assertEqual(result["gain_y"], 0.)


class FixedProbeTests(unittest.TestCase):
    def test_explicit_registered_ids_and_budget_required(self):
        _, sources, baseline = fixture()
        for key in baseline["probes"]:
            spec = dict(baseline["probes"])
            del spec[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_probe_spec(spec, sources)
        for change in ({"rr_eval_ids": []}, {"fr_eval_ids": ["unknown"]},
                       {"rr_train_sample_id": "rr_eval-sample"}, {"max_samples": 0},
                       {"rr_eval_ids": ["rr_eval-scene", "rr_eval-scene"]}):
            with self.assertRaises(ValueError):
                validate_probe_spec({**baseline["probes"], **change}, sources)

    def test_numeric_probe_completeness_and_nonofficial_labels(self):
        trainer, sources, baseline = fixture()
        result = run_probes(trainer, sources, baseline)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["source_group_counts"], {"rr_eval": 1, "fr_eval": 1,
                                                        "rr_train": 1, "fr_train": 1})
        for branch in ("rr", "fr"):
            response = result["response"][branch]
            self.assertEqual(len(response["rows"]), 16)
            self.assertEqual(len(response["summary"]["by_radius"]), 4)
            self.assertEqual(len(response["summary"]["by_radius_axis"]), 16)
            self.assertEqual(response["summary"]["all_radii"]["gain"], 0.)
            self.assertTrue(response["prior_unchanged_under_epsilon"])
        self.assertEqual(len(result["spatial_discrepancy"]), 6)
        self.assertEqual(len(result["inference_interventions"]), 6)
        self.assertFalse(result["official_selection_input"])
        self.assertFalse(result["images_written"])
        self.assertFalse(result["physical_displacement_ground_truth"])
        for row in result["spatial_discrepancy"]:
            self.assertFalse(row["independent_geometric_evidence"])
        for row in result["inference_interventions"]:
            self.assertFalse(row["official_metric"])
            self.assertFalse(row["fr_reference_is_ground_truth"])
        self.assertEqual(set(result["gradient_diagnostics"]["raw_loss"]), {"rec", "abs_rr", "eps_rr", "eps_fr"})
        self.assertFalse(result["gradient_diagnostics"]["evaluation_data_in_backward"])

    def test_preserves_states_grad_buffers_modes_and_all_rng(self):
        trainer, sources, baseline = fixture()
        trainer.stage.train()
        trainer.stage.teacher.eval()  # Mixed module modes must survive too.
        first = next(trainer.stage.aligner.parameters())
        first.grad = torch.ones_like(first)
        grads = [None if p.grad is None else p.grad.clone() for p in trainer.stage.parameters()]
        state = model_state_hash(trainer.stage)
        modes = [m.training for m in trainer.stage.modules()]
        torch_rng, python_rng, numpy_rng = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
        epsilon_rng = trainer.objective.rng_state()
        extra = copy.deepcopy(trainer.stage.get_extra_state())
        result = run_probes(trainer, sources, baseline)
        self.assertEqual(state, result["model_state_sha256"])
        self.assertEqual(state, model_state_hash(trainer.stage))
        self.assertEqual(modes, [m.training for m in trainer.stage.modules()])
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(numpy_rng[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
        self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
        after = trainer.objective.rng_state()
        self.assertEqual(after["calls"], epsilon_rng["calls"])
        for name in after["generators"]:
            self.assertTrue(torch.equal(after["generators"][name], epsilon_rng["generators"][name]))
        self.assertEqual(extra["forward_calls"], trainer.stage.forward_calls)
        self.assertTrue(torch.equal(extra["epsilon_rng"], trainer.stage.get_extra_state()["epsilon_rng"]))
        for before, parameter in zip(grads, trainer.stage.parameters()):
            if before is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(before, parameter.grad, atol=0, rtol=0)

    def test_probe_repeatability_and_separate_gradient_train_population(self):
        trainer, sources, baseline = fixture()
        original = trainer.gradient_diagnostics
        seen = []
        def record(rr, fr, **kwargs):
            seen.append((rr.lineage.split, fr.lineage.split, rr.lineage.sample_ids, fr.lineage.sample_ids))
            return original(rr, fr, **kwargs)
        trainer.gradient_diagnostics = record
        first = run_probes(trainer, sources, baseline)
        second = run_probes(trainer, sources, baseline)
        for result in (first, second):
            result.pop("elapsed_seconds")
        self.assertEqual(first, second)
        self.assertEqual(seen, [("train", "train", ("rr_train-sample",), ("fr_train-sample",))] * 2)

    def test_lpan_layout_uses_synchronized_warp(self):
        trainer, sources, baseline = fixture("PLH")
        seen = []
        def hook(module, args, kwargs):
            low = kwargs["aligned_lpan"]
            self.assertIsNotNone(low)
            self.assertEqual(low.shape, args[0].shape)
            seen.append(low.detach().clone())
        handle = trainer.stage.teacher.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            result = run_probes(trainer, sources, baseline)
        finally:
            handle.remove()
        self.assertEqual(len(result["inference_interventions"]), 6)
        self.assertGreaterEqual(len(seen), 6)
        self.assertFalse(torch.equal(seen[0], seen[1]))


if __name__ == "__main__":
    unittest.main()
