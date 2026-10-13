"""CPU v2 optimizer/accumulation contracts, never launches a campaign."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import torch

from panda_method.calibration import state_dict_sha256
from panda_method.edge_guided import EdgeGuidedStage1, ResidualPANAligner
from panda_method.edge_match import EdgeMatchSpec, EdgeMatcher
from panda_method.test_edge_guided import batch, teacher, SOURCES
from teacher48.edge_training import EdgeStage1Trainer


torch.set_num_threads(1)


def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(814)
        return EdgeGuidedStage1(ResidualPANAligner(4, view_margin=0), teacher(),
            EdgeMatcher(EdgeMatchSpec.campaign_v2('QB', radius=1)), sources=SOURCES,
            roi_margin=3, boundary_policy='report', epsilon_seed=117,
            lambda_abs=.3, lambda_eps=.0003)


def trainer(*, cadence=1000, effective=None, micro=None, adam=False):
    stage = model()
    optimizer = (torch.optim.AdamW(stage.parameters(), lr=1e-4, weight_decay=.01) if adam else
                 torch.optim.SGD(stage.parameters(), lr=1e-4, momentum=.9))
    return EdgeStage1Trainer(stage, optimizer, gradient_log_interval=cadence,
                             effective_batch_size=effective, microbatch_size=micro)


def slice_batch(rr, start, stop):
    return replace(rr, pan=rr.pan[start:stop], ms=rr.ms[start:stop],
        native_lms=rr.native_lms[start:stop], gt=rr.gt[start:stop],
        lineage=replace(rr.lineage, sample_ids=rr.lineage.sample_ids[start:stop],
                        augmentations=rr.lineage.augmentations[start:stop]))


class EdgeTrainingV2Tests(unittest.TestCase):
    def test_effective48_micro12_accum4_equals_full_batch_one_step(self):
        rr = batch(side=16, batch_size=48)
        whole, accum = trainer(), trainer(effective=48, micro=12)
        mb = [slice_batch(rr, i, i+12) for i in range(0, 48, 12)]
        with patch.object(accum.optimizer, 'step', wraps=accum.optimizer.step) as step:
            accumulated = accum.step_accumulated(mb)
        full = whole.step(rr)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(accumulated['effective_batch_size'], 48)
        self.assertEqual(accumulated['microbatch_sizes'], [12]*4)
        self.assertEqual(accum.updates, 1)
        self.assertEqual(accum.stage.forward_calls, 4)
        self.assertTrue(torch.equal(whole.stage._epsilon_rng.get_state(), accum.stage._epsilon_rng.get_state()))
        for key in ('loss', 'rec', 'abs_rr', 'weighted_abs', 'eps_rr', 'weighted_eps_rr'):
            self.assertAlmostEqual(full[key], accumulated[key], places=6)
        self.assertEqual(full['alignment']['shifted']['b'], accumulated['alignment']['shifted']['b'])
        for p, q in zip(whole.stage.parameters(), accum.stage.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-8)

    def test_unequal_microbatch_means_are_sample_weighted(self):
        rr = batch(side=16, batch_size=8)
        whole, accum = trainer(), trainer()
        a, b = whole.step(rr), accum.step_accumulated([slice_batch(rr, 0, 3), slice_batch(rr, 3, 8)],
                                                     expected_effective_batch=8)
        self.assertAlmostEqual(a['loss'], b['loss'], places=6)
        for p, q in zip(whole.stage.parameters(), accum.stage.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-8)

    def test_partial_accumulation_failure_rewinds_rng_and_no_weight_update(self):
        run, rr = trainer(), batch(side=16, batch_size=4)
        before = (state_dict_sha256(run.stage.aligner), state_dict_sha256(run.stage.teacher))
        rng = run.stage._epsilon_rng.get_state().clone()
        bad = replace(slice_batch(rr, 2, 4), lineage=replace(slice_batch(rr, 2, 4).lineage, split='validation'))
        with self.assertRaises(ValueError):
            run.step_accumulated([slice_batch(rr, 0, 2), bad])
        self.assertEqual(run.updates, 0)
        self.assertEqual(run.stage.forward_calls, 0)
        self.assertTrue(torch.equal(rng, run.stage._epsilon_rng.get_state()))
        self.assertEqual(before, (state_dict_sha256(run.stage.aligner), state_dict_sha256(run.stage.teacher)))
        self.assertTrue(all(p.grad is None for p in run.stage.parameters()))
        self.assertFalse(run.poisoned)
        self.assertEqual(run.optimizer.state_dict()['state'], {})

    def test_pinned_batch_sizes_and_optimizer_names_enforced(self):
        run, rr = trainer(effective=48, micro=12), batch(side=16, batch_size=48)
        with self.assertRaises(ValueError):
            run.step(rr)
        with self.assertRaises(ValueError):
            run.step_accumulated([slice_batch(rr, 0, 12)], expected_effective_batch=12)
        with self.assertRaises(ValueError):
            run.step_accumulated([])
        current = trainer()
        saved = current.state_dict()
        stage = model()
        reversed_optimizer = torch.optim.SGD(list(reversed(list(stage.parameters()))), lr=1e-4, momentum=.9)
        other = EdgeStage1Trainer(stage, reversed_optimizer, gradient_log_interval=1000)
        before = state_dict_sha256(stage.aligner)
        with self.assertRaisesRegex(ValueError, 'name/group order'):
            other.load_state_dict(saved)
        self.assertEqual(before, state_dict_sha256(stage.aligner))

    def test_actual_update_norm_and_weighted_gradient_scopes(self):
        run, rr = trainer(adam=True), batch(side=16, batch_size=2)
        before = [p.detach().clone() for p in run.stage.aligner.parameters()]
        result = run.step(rr)
        actual = sum(float((p.detach()-old).double().square().sum())
                     for p, old in zip(run.stage.aligner.parameters(), before)) ** .5
        self.assertEqual(result['optimizer_update_norm']['aligner'], actual)
        self.assertEqual(set(result['optimizer_update_norm']['aligner_scopes']),
                         {'whole', 'fc2', 'fc1', 'body', 'stem'})
        self.assertAlmostEqual(result['term_gradient_norm']['weighted_abs']['aligner'],
                               .3*result['term_gradient_norm']['abs_rr']['aligner'], places=6)
        self.assertEqual(result['term_gradient_norm']['weighted_abs']['teacher'], 0.)
        self.assertEqual(result['term_gradient_norm']['weighted_eps_rr']['teacher'], 0.)
        self.assertEqual(result['term_gradient_cosine']['fc1']['rec__weighted_abs'], None)
        json.dumps(result, allow_nan=False)

    def test_logging_does_not_modify_update_and_resume_is_identical(self):
        rr = batch(side=16, batch_size=4)
        pieces = [slice_batch(rr, 0, 2), slice_batch(rr, 2, 4)]
        logged, sparse = trainer(cadence=1), trainer(cadence=1000)
        for _ in range(2):
            a, b = logged.step_accumulated(pieces), sparse.step_accumulated(pieces)
            self.assertEqual(a['loss'], b['loss'])
        self.assertIsNone(b['term_gradient_norm'])
        for p, q in zip(logged.stage.parameters(), sparse.stage.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        saved = sparse.state_dict()
        restored = trainer(cadence=1000)
        restored.load_state_dict(saved)
        self.assertEqual(sparse.step_accumulated(pieces), restored.step_accumulated(pieces))
        tampered = copy.deepcopy(saved)
        tampered['stage']['_extra_state']['pipeline']['loss_coefficients'] = [1., 1., .001]
        with self.assertRaisesRegex(ValueError, 'pipeline'):
            restored.load_state_dict(tampered)

    def test_optimizer_failure_poisoned_not_claimed_rolled_back(self):
        run = trainer()
        with patch.object(run.optimizer, 'step', side_effect=RuntimeError('synthetic partial optimizer')):
            with self.assertRaisesRegex(RuntimeError, 'partial optimizer'):
                run.step(batch(side=16))
        self.assertTrue(run.poisoned)
        self.assertEqual(run.updates, 0)
        with self.assertRaises(RuntimeError):
            run.step(batch(side=16))
        with self.assertRaises(RuntimeError):
            run.state_dict()


if __name__ == '__main__':
    unittest.main()
