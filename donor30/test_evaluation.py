"""Injected-metric CPU tests: reference identity, state, scope, completeness."""
from __future__ import annotations

import copy
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from directshift48.test_evaluation import SyntheticSources, metrics
from directshift48.training import DirectStage1Trainer
from panda_method.calibration import canonical_sha256, state_dict_sha256
from ta2.evaluation import digest
from . import plan
from .evaluation import (model_identity, model_state_hash, official_evaluate,
                         paused_model, qualify_evaluation, expected_ids)
from .test_method import setup, model
from .training import StudentTrainer


torch.set_num_threads(1)


def engine(phase="donor"):
    if phase == "donor":
        stage = model(lambda_abs=1., lambda_eps=.0003)
        trainer = DirectStage1Trainer(stage, torch.optim.AdamW([
            {"params": stage.aligner.parameters(), "lr": 1e-5},
            {"params": stage.teacher.parameters(), "lr": 1e-4}]), gradient_log_interval=100)
    else:
        stage, _, _, _ = setup("eq")
        trainer = StudentTrainer(stage, torch.optim.AdamW(stage.student.parameters(), lr=1e-4))
    trainer.updates = 1
    c = plan.resolve_donor_case(plan.default_config("s3"))
    spec = stage.teacher.spec
    c["baseline_resolved"]["teacher_architecture"] = {"bands": spec.bands, "width": spec.width, "depth": list(spec.depth)}
    if phase == "stage2":
        c.update(phase="stage2", stage2=True, stage2_mode="eq", lambda_edge=1.,
            student_architecture={"bands": stage.student.spec.bands, "width": stage.student.spec.width,
                                  "depth": list(stage.student.spec.depth)},
            donor_state_hash=canonical_sha256({"A": state_dict_sha256(stage.aligner),
                                              "T": state_dict_sha256(stage.teacher)}))
    trainer.config, trainer.evaluator_pin = c, {}
    return trainer


def evaluate(trainer, sources=None, functions=None, **kwargs):
    return official_evaluate(trainer, sources or SyntheticSources(),
        metric_functions=functions or metrics(), evaluator_revision="TEST", **kwargs)


class EvaluationTests(unittest.TestCase):
    def test_complete_same_state_both_phases_and_student_identity(self):
        sources = SyntheticSources()
        for phase in ("donor", "stage2"):
            trainer = engine(phase)
            before = model_state_hash(trainer.stage)
            record = evaluate(trainer, sources)
            self.assertEqual(before, model_state_hash(trainer.stage))
            self.assertTrue(record["qualification"]["joint_pass"])
            self.assertTrue(record["qualification"]["metrics_complete"])
            self.assertTrue(record["qualification"]["same_state_verified"])
            self.assertFalse(record["weights_saved"])
            self.assertTrue(record["metric_function_injection"])
            for split in ("rr", "fr"):
                self.assertEqual([x["sample_id"] for x in record[split]["per_scene"]], list(expected_ids(sources, split)))
                self.assertEqual(record[split+"_state"], record["state_before"])
                self.assertTrue(all(len(r["raw_prediction_sha256"]) == 64 for r in record[split]["per_scene"]))
            self.assertEqual(record["state_before"], record["state_after"])
            if phase == "stage2":
                with torch.no_grad():
                    next(trainer.stage.student.parameters()).add_(.01)
                self.assertNotEqual(before, model_state_hash(trainer.stage))

    def test_original_unmasked_references_and_current_pixel_shift(self):
        trainer, sources = engine(), SyntheticSources()
        with torch.no_grad():
            trainer.stage.aligner.fc2.bias.copy_(torch.tensor([.5, -.25]))
        funcs, calls = metrics(), {"rr": 0, "fr": 0}
        for split, function in list(funcs.items()):
            def checked(raw, arrays, split=split, function=function):
                self.assertEqual(raw.shape, (4, 32, 32))
                expected = sources.raw(split, [calls[split]])
                for key in expected:
                    np.testing.assert_array_equal(arrays[key], expected[key][0])
                calls[split] += 1
                return function(raw, arrays)
            funcs[split] = checked
        record = evaluate(trainer, sources, funcs)
        self.assertEqual(calls, {"rr": 20, "fr": 20})
        for split in ("rr", "fr"):
            self.assertTrue(all(r["correction_dy_dx"] == [.5, -.25] for r in record[split]["per_scene"]))

    def test_stage2_official_inference_never_calls_teacher(self):
        trainer = engine("stage2")
        with patch.object(trainer.stage.teacher, "forward", side_effect=AssertionError("Teacher used")):
            record = evaluate(trainer)
        self.assertTrue(record["qualification"]["same_state_verified"])

    def test_rng_modes_gradients_optimizer_and_update_preserved(self):
        for phase in ("donor", "stage2"):
            trainer = engine(phase)
            param = next(trainer.stage.teacher.parameters() if phase == "donor" else trainer.stage.student.parameters())
            param.grad = torch.ones_like(param)
            random_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
            funcs = metrics()
            old = funcs["fr"]
            def draw(p, a):
                random.random(); np.random.rand(); torch.rand(1)
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(trainer.stage.training)
                return old(p, a)
            funcs["fr"] = draw
            evaluate(trainer, functions=funcs)
            self.assertEqual(random_before, random.getstate())
            np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
            self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
            self.assertTrue(trainer.stage.training)
            self.assertTrue(bool((param.grad == 1).all()))
            self.assertEqual(trainer.updates, 1)
            self.assertEqual(len(trainer.optimizer.state), 0)

    def test_nonfinite_incomplete_wrong_ids_and_injection_fail_closed(self):
        trainer, sources = engine(), SyntheticSources()
        with self.assertRaisesRegex(ValueError, "synthetic TEST"):
            official_evaluate(trainer, sources, metric_functions=metrics(), evaluator_revision="PRODUCTION")
        with self.assertRaises(FloatingPointError):
            evaluate(trainer, sources, metrics(ergas=float("nan")))
        with self.assertRaisesRegex(TimeoutError, "INCOMPLETE"):
            evaluate(trainer, sources, deadline_unix=0)
        original = sources.batch
        def wrong(split, ids, **kwargs):
            result = original(split, ids, **kwargs)
            return replace(result, lineage=replace(result.lineage, sample_ids=("wrong",)))
        with patch.object(sources, "batch", side_effect=wrong):
            with self.assertRaisesRegex(ValueError, "image identity"):
                evaluate(trainer, sources)

    def test_qualification_rejects_missing_mixed_and_wrong_aggregate(self):
        trainer, sources = engine(), SyntheticSources()
        record = evaluate(trainer, sources)
        for kind in ("missing", "mixed", "aggregate", "preflight"):
            changed = copy.deepcopy(record)
            if kind == "missing": changed["rr"]["per_scene"].pop()
            elif kind == "mixed": changed["fr_state"]["model_state_sha256"] = "b"*64
            elif kind == "aggregate": changed["fr"]["metrics"]["HQNR"] += .001
            else: changed["full_method_eligible"] = False
            result = qualify_evaluation(changed, server="s3", expected_rr_ids=expected_ids(sources, "rr"),
                                        expected_fr_ids=expected_ids(sources, "fr"))
            self.assertFalse(result["joint_pass"])
        failed = evaluate(trainer, sources, metrics(hqnr=.964999999999))
        self.assertFalse(failed["qualification"]["joint_pass"])

    def test_weight_optimizer_and_update_mutations_poison(self):
        for kind in ("weights", "optimizer", "update"):
            trainer = engine("stage2")
            functions = metrics()
            def mutate(p, a):
                if kind == "weights": next(trainer.stage.student.parameters()).add_(.01)
                elif kind == "optimizer": trainer.optimizer.param_groups[0]["lr"] *= .99
                else: trainer.updates += 1
                return metrics()["rr"](p, a)
            functions["rr"] = mutate
            with self.assertRaisesRegex(RuntimeError, "changed"):
                evaluate(trainer, functions=functions)
            self.assertTrue(trainer.poisoned)

    def test_paused_restores_epsilon_and_gradients_after_failure(self):
        trainer = engine()
        param = next(trainer.stage.aligner.parameters())
        param.grad = torch.ones_like(param)
        epsilon_rng = trainer.stage._epsilon_rng.get_state().clone()
        with self.assertRaisesRegex(ValueError, "interrupted"):
            with paused_model(trainer, gradients=True):
                torch.rand(4, generator=trainer.stage._epsilon_rng)
                trainer.stage.forward_calls += 1
                param.grad.zero_()
                raise ValueError("interrupted")
        self.assertEqual(trainer.stage.forward_calls, 0)
        self.assertTrue(torch.equal(epsilon_rng, trainer.stage._epsilon_rng.get_state()))
        self.assertTrue(bool((param.grad == 1).all()))
        self.assertFalse(trainer.poisoned)

    def test_registered_architecture_source_and_donor_identity_checked(self):
        trainer, sources = engine("stage2"), SyntheticSources()
        trainer.config["student_architecture"]["width"] = 104
        with self.assertRaisesRegex(ValueError, "Student architecture"):
            model_identity(trainer, sources)
        trainer = engine("stage2")
        trainer.config["donor_state_hash"] = "b"*64
        with self.assertRaisesRegex(ValueError, "Selected donor"):
            model_identity(trainer, sources)


if __name__ == "__main__":
    unittest.main()
