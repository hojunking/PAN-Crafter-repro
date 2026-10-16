"""CPU W112 training/evaluation integration; synthetic metrics never publish."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from directshift48.method import DirectPANAligner
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
from panda_stage2_condq.evaluation import checkpoint_rank, official_evaluate, validate_result
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.spec import Stage2Spec
from panda_stage2_condq.training import StudentTrainer, TrainingSpec


def rr_batch(count=1, *, shift=0):
    side = 64
    pan = torch.linspace(-.8, .8, side * side).reshape(1, 1, side, side).repeat(count, 1, 1, 1)
    ms = torch.linspace(-.5, .5, 8 * 16 * 16).reshape(1, 8, 16, 16).repeat(count, 1, 1, 1)
    lms = torch.linspace(-.6, .6, 8 * side * side).reshape(1, 8, side, side).repeat(count, 1, 1, 1)
    gt = lms * .7 + .1
    lineage = EdgeBatchLineage("train", "RR", "a" * 64,
        tuple(f"WV3:train:{i + shift}" for i in range(count)), "WV3", "PAN_RR",
        ((False, False, 0),) * count)
    return EdgeObservationBatch(pan, ms, lms, lineage, gt)


def stage_and_trainer(*, micro=1, total=1, correction=(.5, -.25), delta=1e-6, seed=71):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(32)
        aligner = DirectPANAligner(8)
    with torch.no_grad():
        aligner.fc2.bias.copy_(torch.tensor(correction))
    binding = {"sensor": "WV3", "source_manifest_sha256": "a" * 64,
               "aligner_state_sha256": state_dict_sha256(aligner),
               "teacher_state_sha256": "b" * 64, "donor_checkpoint_id": "c" * 64,
               "sample_count": 100, "source_split": "RR_train_only",
               "sample_ids_sha256": canonical_sha256(sorted(f"WV3:train:{i}" for i in range(100))),
               "reconstruction_spec": {"bands": 8, "width": 112, "depth": [1, 2, 3],
                   "input_layout": "P0", "base_reference": "BICUBIC_MS"},
               "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": [4, 2, 1], "roi_margin": 8},
               "warp": "bicubic_border_FP32_align_corners_false_current_pan_px"}
    stage = ConditionalQStage2(Stage2Spec("WV3", "gt_only", seed, 0, 0, 0, delta),
                              aligner, donor_binding=binding)
    training = TrainingSpec(effective_batch_size=total, microbatch_size=micro,
                            total_updates=20, warmup_updates=0, gradient_log_interval=2)
    return StudentTrainer(stage, training, evaluation_pin={})


class EvalSources:
    sensor, bands, peak = "WV3", 8, 2047.

    def manifest_sha(self, split):
        return "a" * 64

    def batch(self, split, rows, *, device="cpu"):
        if len(rows) != 1:
            raise ValueError("One fixture scene at a time")
        base = rr_batch()
        resolution = "FR" if split == "fr" else "RR"
        line = replace(base.lineage, split="test", resolution=resolution,
                       sample_ids=(f"WV3:{split}:{rows[0]}",), grid_name="PAN_" + resolution)
        return replace(base, lineage=line, gt=None if split == "fr" else base.gt)

    def raw(self, split, rows):
        batch = self.batch(split, rows)
        result = {k: ((getattr(batch, name) + 1) * self.peak / 2).numpy().copy()
                  for k, name in (("pan", "pan"), ("ms", "ms"), ("lms", "native_lms"))}
        if batch.gt is not None:
            result["gt"] = ((batch.gt + 1) * self.peak / 2).numpy().copy()
        return result


def metrics():
    return {"rr": lambda p, a: {"ergas": 2.1, "scc": .98, "sam": 2., "psnr": 39.,
                                 "ssim": .97, "q8": .95, "rmse": 11., "cc": .99},
            "fr": lambda p, a: {"hqnr": .958, "d_s": .03, "d_lambda": .01,
                                 "jqm": .86, "jqm_weights": [.125] * 8,
                                 "jqm_weight_source": "SYNTHETIC_TEST_ONLY"}}


class TrainingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_optimizer_only_student_and_diagnostics(self):
        trainer = stage_and_trainer()
        before_a = state_dict_sha256(trainer.stage.aligner)
        before_s = state_dict_sha256(trainer.stage.student)
        first = trainer.step(rr_batch())
        self.assertEqual(before_a, state_dict_sha256(trainer.stage.aligner))
        self.assertNotEqual(before_s, state_dict_sha256(trainer.stage.student))
        self.assertTrue(all(p.grad is None for p in trainer.stage.aligner.parameters()))
        self.assertEqual(first["gradient"]["aligner"], 0.)
        self.assertTrue(first["gradient"]["diagnostic_computed"])
        self.assertGreater(first["gradient"]["hard_norm"], 0)
        self.assertEqual(first["gradient"]["beta_soft_norm"], 0)
        self.assertIsNone(first["gradient"]["hard_soft_cosine"])
        self.assertTrue(trainer.step(rr_batch())["gradient"]["diagnostic_computed"])
        self.assertFalse(trainer.step(rr_batch())["gradient"]["diagnostic_computed"])
        trainer.optimizer.add_param_group({"params": [trainer.stage.aligner.fc2.bias]})
        with self.assertRaisesRegex(ValueError, "Student"):
            trainer.assert_contract()

    def test_batch4_and_microbatch1x4_gradients_agree(self):
        joined, split = stage_and_trainer(micro=4, total=4), stage_and_trainer(micro=1, total=4)
        batch = rr_batch(4)
        one = joined.step_accumulated([batch])
        two = split.step_accumulated([rr_batch(1, shift=i) for i in range(4)])
        self.assertAlmostEqual(one["loss"], two["loss"], places=6)
        for (name, a), (_, b) in zip(joined.stage.student.named_parameters(), split.stage.student.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-7, msg=name)

    def test_support_failure_prevents_any_optimizer_update(self):
        trainer = stage_and_trainer(correction=(7., 0.))
        before = state_dict_sha256(trainer.stage.student)
        with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
                trainer.step(rr_batch())
            step.assert_not_called()
        self.assertEqual(trainer.updates, 0)
        self.assertFalse(trainer.optimizer.state)
        self.assertEqual(before, state_dict_sha256(trainer.stage.student))
        self.assertTrue(all(p.grad is None for p in trainer.stage.student.parameters()))

    def test_late_invalid_microbatch_discards_partial_gradients(self):
        trainer = stage_and_trainer(micro=1, total=2)
        before = state_dict_sha256(trainer.stage.student)
        invalid = rr_batch(1, shift=1)
        invalid = replace(invalid, lineage=replace(invalid.lineage, split="test"))
        with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step:
            with self.assertRaises(ValueError):
                trainer.step_accumulated([rr_batch(), invalid])
            step.assert_not_called()
        self.assertEqual(before, state_dict_sha256(trainer.stage.student))
        self.assertTrue(all(p.grad is None for p in trainer.stage.student.parameters()))
        self.assertEqual(trainer.updates, 0)

    def test_checkpoint_exact_reload_and_incompatible_contracts_rejected(self):
        trainer = stage_and_trainer()
        trainer.step(rr_batch())
        saved = trainer.state_dict(stream_state={"cursor": 7})
        fresh = stage_and_trainer()
        self.assertEqual(fresh.load_state_dict(saved), {"cursor": 7})
        self.assertEqual(fresh.updates, 1)
        self.assertEqual(state_dict_sha256(fresh.stage.student), state_dict_sha256(trainer.stage.student))
        trainer.step(rr_batch())
        fresh.step(rr_batch())
        self.assertEqual(state_dict_sha256(fresh.stage.student), state_dict_sha256(trainer.stage.student))
        for key, value in (("schema", "DONOR30_STAGE2_OLD_EQ"), ("updates", -1)):
            bad = deepcopy(saved)
            bad[key] = value
            with self.assertRaises(ValueError):
                stage_and_trainer().load_state_dict(bad)
        with self.assertRaisesRegex(ValueError, "contract"):
            stage_and_trainer(delta=2e-6).load_state_dict(saved)
        bad = deepcopy(saved)
        bad["stage"]["_extra_state"]["calibration_artifact_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            stage_and_trainer().load_state_dict(bad)

    def test_optimizer_hyperparameter_tampering_is_rejected(self):
        trainer = stage_and_trainer()
        trainer.step(rr_batch())
        saved = trainer.state_dict()
        saved["optimizer"]["param_groups"][0]["betas"] = (.5, .9)
        with self.assertRaises(ValueError):
            stage_and_trainer().load_state_dict(saved)

    def test_missing_allmissing_or_negative_adam_moments_are_rejected(self):
        trainer = stage_and_trainer()
        trainer.step(rr_batch())
        saved = trainer.state_dict()
        ids = saved["optimizer"]["param_groups"][0]["params"]
        self.assertEqual(set(saved["optimizer"]["state"]), set(ids))
        for mutation in ("missing_one", "missing_all", "negative_variance"):
            bad = deepcopy(saved)
            if mutation == "missing_one":
                del bad["optimizer"]["state"][ids[0]]
            elif mutation == "missing_all":
                bad["optimizer"]["state"] = {}
            else:
                bad["optimizer"]["state"][ids[0]]["exp_avg_sq"].flatten()[0] = -1.
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    stage_and_trainer().load_state_dict(bad)

    def test_update_zero_rejects_fabricated_adam_moments(self):
        fresh = stage_and_trainer()
        saved = fresh.state_dict()
        fresh.step(rr_batch())
        saved["optimizer"]["state"] = deepcopy(fresh.optimizer.state_dict()["state"])
        for item in saved["optimizer"]["state"].values():
            item["step"].zero_()
        with self.assertRaises(ValueError):
            stage_and_trainer().load_state_dict(saved)

    def test_atomic_checkpoint_roundtrip_no_overwrite_and_file_sha(self):
        from panda_stage2_condq.checkpoint import save_checkpoint, load_checkpoint
        trainer = stage_and_trainer()
        trainer.step(rr_batch())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.pt"
            written = save_checkpoint(path, trainer, stream_state={"cursor": 9})
            fresh = stage_and_trainer()
            loaded = load_checkpoint(path, written["sha256"], fresh)
            self.assertEqual(loaded["stream_state"], {"cursor": 9})
            self.assertEqual(state_dict_sha256(fresh.stage.student), state_dict_sha256(trainer.stage.student))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, trainer)
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_checkpoint(path, "0" * 64, fresh)
            self.assertEqual(list(Path(temp).glob(".condq-checkpoint-*")), [])

    def test_condq_optimizer_freezes_teacher_and_eval_skips_teacher_and_cdf(self):
        from panda_stage2_condq.tests.test_method import fixture, batch
        stage = fixture("panda_condq")
        trainer = StudentTrainer(stage, TrainingSpec(effective_batch_size=1,
            microbatch_size=1, total_updates=20, warmup_updates=0), evaluation_pin={})
        before_t = state_dict_sha256(stage.teacher)
        before_a = state_dict_sha256(stage.aligner)
        trace = trainer.step(batch())
        self.assertEqual(before_t, state_dict_sha256(stage.teacher))
        self.assertEqual(before_a, state_dict_sha256(stage.aligner))
        self.assertTrue(all(p.grad is None for p in stage.teacher.parameters()))
        self.assertGreater(trace["gradient"]["hard_norm"], 0)
        self.assertEqual(trace["gradient"]["teacher"], 0)
        with patch.object(stage.teacher, "forward", side_effect=AssertionError("Teacher at inference")), \
                patch.object(stage.stats, "lookup_exact", side_effect=AssertionError("CDF at inference")):
            record = official_evaluate(trainer, EvalSources(), metric_functions=metrics(),
                                       evaluator_revision="TEST", evaluator_pin={})
        self.assertTrue(record["same_state_verified"])

    def test_evaluation_preserves_state_modes_rng_and_original_references(self):
        trainer, sources = stage_and_trainer(), EvalSources()
        trainer.step(rr_batch())
        before = (state_dict_sha256(trainer.stage.student), torch.random.get_rng_state().clone(),
                  random.getstate(), deepcopy(np.random.get_state()))
        modes = [m.training for m in trainer.stage.modules()]
        gradients = [None if p.grad is None else p.grad.clone() for p in trainer.stage.parameters()]
        functions, calls = metrics(), {"rr": 0, "fr": 0}
        for split, fn in list(functions.items()):
            def check(raw, arrays, split=split, fn=fn):
                self.assertEqual(raw.shape, (8, 64, 64))
                expected = sources.raw(split, [calls[split]])
                for key in expected:
                    np.testing.assert_array_equal(arrays[key], expected[key][0])
                calls[split] += 1
                self.assertFalse(torch.is_grad_enabled())
                torch.rand(2); random.random(); np.random.random()
                return fn(raw, arrays)
            functions[split] = check
        with patch.object(trainer.stage, "forward", side_effect=AssertionError("training loss evaluated")):
            record = official_evaluate(trainer, sources, metric_functions=functions,
                                       evaluator_revision="TEST", evaluator_pin={})
        self.assertEqual(calls, {"rr": 20, "fr": 20})
        self.assertTrue(validate_result(record))
        self.assertEqual(before[0], state_dict_sha256(trainer.stage.student))
        self.assertTrue(torch.equal(before[1], torch.random.get_rng_state()))
        self.assertEqual(before[2], random.getstate())
        np.testing.assert_array_equal(before[3][1], np.random.get_state()[1])
        self.assertEqual(modes, [m.training for m in trainer.stage.modules()])
        for old, parameter in zip(gradients, trainer.stage.parameters()):
            if old is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(old, parameter.grad, rtol=0, atol=0)
        self.assertEqual({tuple(r["correction_dy_dx"]) for r in record["fr"]["per_scene"]}, {(.5, -.25)})
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            checkpoint_rank(record)
        # Unit-test the comparator only; this modified fixture is never saved or published.
        def rank(hqnr, scc, ergas):
            value = deepcopy(record)
            value["metric_function_injection"] = False
            for split, key, number in (("fr", "HQNR", hqnr), ("rr", "SCC", scc), ("rr", "ERGAS", ergas)):
                value[split]["metrics"][key] = number
                for row in value[split]["per_scene"]:
                    row["metrics"][key] = number
            return checkpoint_rank(value)
        self.assertGreater(rank(.959, .90, 1.), rank(.960, .99, 3.))
        self.assertGreater(rank(.960, .98, 1.), rank(.960, .99, 3.))
        self.assertGreater(rank(.960, .99, 1.), rank(.960, .99, 2.))
        self.assertGreater(rank(.959, .90, 3.), rank(.958999999, .99, 1.))
        self.assertGreater(rank(.958, .90, 3.), rank(.957, .99, 1.))
        self.assertEqual(rank(.960, .80, 2.), rank(.960, .99, 2.))
        with self.assertRaisesRegex(ValueError, "synthetic TEST"):
            official_evaluate(trainer, sources, metric_functions=metrics(), evaluator_revision="PRODUCTION")


if __name__ == "__main__":
    unittest.main()
