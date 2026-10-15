"""Disposable CPU-only objective/update checks; no training runs or uploads."""
import copy
from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from panda_method.test_stages import batch, stage1
from .training import (BranchObjective, Trainer, TrainingConfig, learning_rates,
                       model_state_hash, stream_seeds)


torch.set_num_threads(1)


def config(**changes):
    values = dict(seed=261013201, horizon=6, lambda_abs=.3, lambda_eps=.0003,
                  optimizer="AdamW", optimizer_betas=(.9, .999), optimizer_eps=1e-8,
                  lr_U=1e-4, lr_A=1e-5, weight_decay=1e-4, warmup_updates=2,
                  min_lr_U=1e-7, min_lr_A=1e-7, effective_batch_RR=2,
                  effective_batch_FR=2, microbatch_RR=2, microbatch_FR=2,
                  precision="fp32", scheduler="warmup_cosine")
    values.update(changes)
    return TrainingConfig(**values)


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        result = stage1()
        with torch.no_grad():
            result.teacher.backbone.output[-1].weight.normal_(std=.01)
    return result


def take(observation, indices):
    lineage = replace(observation.lineage,
                      scene_ids=tuple(observation.lineage.scene_ids[i] for i in indices),
                      sample_ids=tuple(observation.lineage.sample_ids[i] for i in indices))
    tensors = {name: getattr(observation, name)[indices] if getattr(observation, name) is not None else None
               for name in ("pan", "ms", "native_lms", "prior_dy_dx", "gt", "lpan")}
    return replace(observation, lineage=lineage, **tensors)


class ConfigTests(unittest.TestCase):
    def test_no_invented_baseline_and_positive_full_losses(self):
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            TrainingConfig.from_mapping({"seed": 1})
        for name in ("lambda_abs", "lambda_eps", "lr_A", "lr_U"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                config(**{name: 0})
        with self.assertRaisesRegex(ValueError, "precision"):
            config(precision="amp")
        with self.assertRaises(ValueError):
            config(warmup_updates=6)
        with self.assertRaises(ValueError):
            config(min_lr_A=.1)
        values = dict(config().__dict__, gradient_clip=1.)
        with self.assertRaisesRegex(ValueError, "clipping"):
            TrainingConfig.from_mapping(values)

    def test_pinned_one_based_cosine_and_warmup(self):
        c = config()
        self.assertEqual(learning_rates(c, 1), {"U": c.lr_U / 2, "A": c.lr_A / 2})
        self.assertEqual(learning_rates(c, 2), {"U": c.lr_U, "A": c.lr_A})
        self.assertEqual(learning_rates(c, 6), {"U": c.min_lr_U, "A": c.min_lr_A})
        self.assertGreater(learning_rates(c, 3)["U"], learning_rates(c, 5)["U"])
        c = config(warmup_updates=0)
        self.assertEqual(learning_rates(c, 1)["A"], c.lr_A)
        self.assertEqual(learning_rates(c, 6)["A"], c.min_lr_A)
        with self.assertRaises(ValueError):
            learning_rates(c, 7)

    def test_named_streams_independent_reproducible(self):
        seeds = stream_seeds(123)
        self.assertEqual(seeds, stream_seeds(123))
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertNotEqual(seeds, stream_seeds(124))


class ObjectiveTests(unittest.TestCase):
    def test_exact_four_terms_shared_A_and_RR_only_U(self):
        stage = model()
        obj = BranchObjective(stage, config())
        a_calls, t_calls = [], []
        a_hook = stage.aligner.register_forward_hook(lambda *_: a_calls.append(1))
        t_hook = stage.teacher.register_forward_hook(lambda *_: t_calls.append(1))
        rr, fr = obj.evaluate(batch(), "rr"), obj.evaluate(batch("FR", side=24), "fr")
        a_hook.remove()
        t_hook.remove()
        self.assertEqual(len(a_calls), 4)
        self.assertEqual(len(t_calls), 1)
        raw = {**rr["raw"], **fr["raw"]}
        torch.testing.assert_close(rr["loss"] + fr["loss"], raw["rec"] + .3 * raw["abs_rr"]
                                   + .0003 * (raw["eps_rr"] + raw["eps_fr"]))
        self.assertFalse(fr["correction"].requires_grad)
        self.assertTrue(rr["correction"].requires_grad)
        self.assertLessEqual(float(rr["epsilon"].norm(dim=-1).max()), 2.)
        self.assertEqual(stage.forward_calls, 0)  # Core's legacy single RNG never used.

    def test_each_gradient_route_and_detached_epsilon_target(self):
        for name in ("rec", "abs_rr", "eps_rr", "eps_fr"):
            with self.subTest(name=name):
                stage = model()
                obj = BranchObjective(stage, config())
                branch = "fr" if name == "eps_fr" else "rr"
                out = obj.evaluate(batch(branch.upper()), branch)
                out["raw"][name].backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in stage.aligner.parameters()
                                       if p.grad is not None), 0.)
                t_grads = [p.grad for p in stage.teacher.parameters() if p.grad is not None]
                if name == "rec":
                    self.assertGreater(sum(float(g.abs().sum()) for g in t_grads), 0.)
                else:
                    self.assertEqual(t_grads, [])

    def test_rr_consumption_does_not_change_fr_stream_and_coefficients_do_not_change_draws(self):
        first = BranchObjective(model(), config())
        second = BranchObjective(model(), config(lambda_eps=.003, lambda_abs=3.))
        first.evaluate(batch(), "rr")
        first.evaluate(batch(), "rr")
        fr_a = first.evaluate(batch("FR"), "fr")["epsilon"]
        fr_b = second.evaluate(batch("FR"), "fr")["epsilon"]
        torch.testing.assert_close(fr_a, fr_b, rtol=0, atol=0)
        fresh = BranchObjective(model(), config())
        torch.testing.assert_close(fresh.evaluate(batch(), "rr")["epsilon"],
                                   second.evaluate(batch(), "rr")["epsilon"], rtol=0, atol=0)

    def test_reject_benchmark_train_and_allow_explicit_diagnostics_only(self):
        obj = BranchObjective(model(), config())
        with self.assertRaises(ValueError):
            obj.evaluate(batch("FR", split="test"), "fr")
        out = obj.evaluate(batch("FR", split="test"), "fr", training=False,
                           epsilon=torch.tensor([[.5, 0.], [0., .5]]))
        self.assertTrue(torch.isfinite(out["loss"]))


class TrainerTests(unittest.TestCase):
    def test_accumulation_means_match_large_branch_batches_unequal_RR_FR_sizes(self):
        rr, fr = batch(), take(batch("FR", side=24), [0, 1, 0])
        full = Trainer(model(), config(effective_batch_FR=3, microbatch_FR=3))
        micro = Trainer(model(), config(effective_batch_FR=3, microbatch_FR=1, microbatch_RR=1))
        trace_f = full.step([rr], [fr])
        trace_m = micro.step([take(rr, [0]), take(rr, [1])],
                             [take(fr, [0]), take(fr, [1]), take(fr, [2])])
        for key in ("loss", "raw_rec", "raw_abs_rr", "raw_eps_rr", "raw_eps_fr"):
            self.assertAlmostEqual(trace_f[key], trace_m[key], places=6, msg=key)
        for p, q in zip(full.stage.parameters(), micro.stage.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-5, atol=1e-7)
        self.assertEqual(trace_m["rr_branch_microbatches"], 2)
        self.assertEqual(trace_m["fr_branch_microbatches"], 3)
        self.assertAlmostEqual(trace_m["loss"], trace_m["raw_rec"] + .3 * trace_m["raw_abs_rr"]
                               + .0003 * (trace_m["raw_eps_rr"] + trace_m["raw_eps_fr"]), places=7)

    def test_exact_RAM_resume_optimizer_streams_and_no_global_rng_consumption(self):
        trainer = Trainer(model(), config())
        before = torch.random.get_rng_state().clone()
        trainer.step([batch()], [batch("FR")])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        saved = trainer.snapshot_state()
        trace = trainer.step([batch()], [batch("FR")])
        expected = model_state_hash(trainer.stage)
        restored = Trainer(model(), config())
        restored.restore_state(saved)
        actual = restored.step([batch()], [batch("FR")])
        self.assertEqual(model_state_hash(restored.stage), expected)
        for key in ("loss", "raw_eps_rr", "raw_eps_fr", "lr_U", "lr_A", "step"):
            self.assertEqual(actual[key], trace[key], key)

    def test_validation_error_no_update_and_nonfinite_attempt_poisoned(self):
        trainer = Trainer(model(), config())
        before = model_state_hash(trainer.stage)
        with self.assertRaises(ValueError):
            trainer.step([take(batch(), [0])], [batch("FR")])
        self.assertEqual(trainer.updates, 0)
        self.assertEqual(model_state_hash(trainer.stage), before)
        with self.assertRaises(ValueError):
            trainer.step([batch()], [batch("FR", split="test")])
        self.assertEqual(model_state_hash(trainer.stage), before)
        self.assertTrue(trainer.poisoned)
        self.assertEqual(trainer.objective.calls, {"rr": 0, "fr": 0})
        with self.assertRaises(RuntimeError):
            trainer.step([batch()], [batch("FR")])

    def test_evaluation_preserves_modes_rng_and_hash_detects_mutation(self):
        trainer = Trainer(model(), config())
        trainer.stage.train()
        trainer.stage.teacher.eval()
        modes = [m.training for m in trainer.stage.modules()]
        rng = torch.random.get_rng_state().clone()
        objective = copy.deepcopy(trainer.objective.rng_state())
        with trainer.evaluation_context() as sha:
            self.assertEqual(sha, model_state_hash(trainer.stage))
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(trainer.stage.training)
            torch.rand(4)
            random.random()
            np.random.random()
        self.assertEqual(modes, [m.training for m in trainer.stage.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(objective["generators"]["rr"],
                                    trainer.objective.rng_state()["generators"]["rr"]))
        with self.assertRaisesRegex(RuntimeError, "changed"):
            with trainer.evaluation_context():
                next(trainer.stage.aligner.parameters()).add_(1.)
        self.assertTrue(trainer.poisoned)

    def test_numeric_grad_probe_does_not_consume_streams_or_parameter_grads(self):
        trainer = Trainer(model(), config())
        trainer.step([batch()], [batch("FR")])
        sha = model_state_hash(trainer.stage)
        grads = [p.grad.clone() if p.grad is not None else None for p in trainer.stage.parameters()]
        before = trainer.objective.rng_state()
        result = trainer.gradient_diagnostics(batch(split="test"), batch("FR", split="test"),
                    epsilon_rr=torch.tensor([[.5, .5], [-.5, .5]]),
                    epsilon_fr=torch.tensor([[.5, 0.], [0., -.5]]))
        self.assertEqual(sha, model_state_hash(trainer.stage))
        self.assertEqual(set(result["raw_A_grad_norm"]), {"rec", "abs_rr", "eps_rr", "eps_fr"})
        self.assertEqual(len(result["raw_A_grad_cosine"]), 6)
        self.assertTrue(torch.equal(before["generators"]["rr"],
                                    trainer.objective.rng_state()["generators"]["rr"]))
        for expected, p in zip(grads, trainer.stage.parameters()):
            if expected is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(expected, p.grad, rtol=0, atol=0)

    def test_run_callbacks_stop_before_next_update(self):
        trainer = Trainer(model(), config())
        seen = []
        count = trainer.run(lambda step: [batch()], lambda step: [batch("FR")],
                            after_update=lambda engine, trace: seen.append(trace["step"]),
                            should_stop=lambda engine: engine.updates >= 2)
        self.assertEqual(count, 2)
        self.assertEqual(seen, [1, 2])


if __name__ == "__main__":
    unittest.main()
