"""Student optimizer isolation, accumulation and checkpoint roundtrip tests."""
from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

from panda_method.calibration import state_dict_sha256
from .test_method import setup
from .training import StudentTrainer


def split(rr):
    return [replace(rr, pan=rr.pan[i:i+1], ms=rr.ms[i:i+1], native_lms=rr.native_lms[i:i+1],
                    gt=rr.gt[i:i+1], lineage=replace(rr.lineage,
                        sample_ids=rr.lineage.sample_ids[i:i+1], augmentations=rr.lineage.augmentations[i:i+1]))
            for i in range(len(rr.pan))]


def engine(mode="eq"):
    stage, _, rr, _ = setup(mode)
    optimizer = torch.optim.AdamW(stage.student.parameters(), lr=1e-4)
    return StudentTrainer(stage, optimizer, effective_batch_size=2, microbatch_size=1), rr


class TrainerTests(unittest.TestCase):
    def test_step_frozen_hashes_and_checkpoint_next_update(self):
        trainer, rr = engine()
        a, t = state_dict_sha256(trainer.stage.aligner), state_dict_sha256(trainer.stage.teacher)
        before = state_dict_sha256(trainer.stage.student)
        result = trainer.step_accumulated(split(rr))
        self.assertEqual(result["update"], 1)
        self.assertEqual(result["gradient_norm"]["aligner"], 0.)
        self.assertEqual(a, state_dict_sha256(trainer.stage.aligner))
        self.assertEqual(t, state_dict_sha256(trainer.stage.teacher))
        self.assertNotEqual(before, state_dict_sha256(trainer.stage.student))
        saved = trainer.state_dict()
        second, _ = engine()
        second.load_state_dict(saved)
        trainer.step_accumulated(split(rr))
        second.step_accumulated(split(rr))
        self.assertEqual(state_dict_sha256(trainer.stage.student), state_dict_sha256(second.stage.student))

    def test_accumulation_equals_full_batch(self):
        trainer, rr = engine("gt")
        stage, _, _, _ = setup("gt")
        other = StudentTrainer(stage, torch.optim.AdamW(stage.student.parameters(), lr=1e-4))
        result_a = trainer.step_accumulated(split(rr))
        result_b = other.step(rr)
        self.assertAlmostEqual(result_a["loss"], result_b["loss"], places=6)
        for a, b in zip(trainer.stage.student.parameters(), other.stage.student.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_rejected_batch_no_partial_update(self):
        trainer, rr = engine()
        before = state_dict_sha256(trainer.stage.student)
        bad = split(rr)
        bad[-1] = replace(bad[-1], lineage=replace(bad[-1].lineage, source_manifest_sha256="c"*64))
        with self.assertRaises(ValueError):
            trainer.step_accumulated(bad)
        self.assertEqual(trainer.updates, 0)
        self.assertFalse(trainer.poisoned)
        self.assertEqual(before, state_dict_sha256(trainer.stage.student))
        self.assertTrue(all(p.grad is None for p in trainer.stage.student.parameters()))

    def test_optimizer_failure_poison_and_strict_contract(self):
        trainer, rr = engine()
        with patch.object(trainer.optimizer, "step", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                trainer.step_accumulated(split(rr))
        self.assertTrue(trainer.poisoned)
        with self.assertRaises(RuntimeError):
            trainer.state_dict()
        trainer, _ = engine()
        saved = trainer.state_dict()
        saved["effective_batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "contract"):
            trainer.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
