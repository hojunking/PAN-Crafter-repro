"""MASK2 is an explicit, separately calibrated extension, not a full-image rename."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from donor30.masking import masked_mean
from donor30.plan import LOSS_MASK_POLICY
from pa.warp import warp_support_mask
from panda_stage2_condq.calibration import CalibrationBuilder, CalibrationStats
from panda_stage2_condq.losses import compute_loss
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.spec import MASK_POLICY, MASK2_POLICY, MASK2_CALIBRATION_SCHEMA
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from panda_stage2_condq.checkpoint import save_checkpoint, load_checkpoint
from panda_stage2_condq.evaluation import official_evaluate, validate_result
from panda_stage2_condq.tests.test_calibration import binding
from panda_stage2_condq.tests.test_losses import config, images, FixedRanks
from panda_stage2_condq.tests.test_method import fixture, batch
from directshift48.method import ScharrAbsolute
from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics


def support(shape=(64, 64), count=2):
    return warp_support_mask(*shape, torch.tensor([[0., 0.], [4., -3.5]])[:count])


class Mask2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_explicit_identity_and_default_manifest(self):
        full, masked = config(), config(mask_policy=MASK2_POLICY)
        self.assertEqual(MASK2_POLICY, LOSS_MASK_POLICY)
        self.assertEqual(full.mask_policy, MASK_POLICY)
        self.assertNotEqual(full.method_version, masked.method_version)
        self.assertNotEqual(full.loss_schema, masked.loss_schema)
        self.assertNotEqual(full.calibration_schema, masked.calibration_schema)
        self.assertEqual(masked.manifest()['loss_mask_policy'], MASK2_POLICY)
        with self.assertRaises(ValueError):
            config(mask_policy='old-mask')

    def test_hard_soft_match_stage1_mask_reduction_and_outside_gradient_zero(self):
        s, t, gt, pan = images()
        mask = support()
        spec = config(mask_policy=MASK2_POLICY)
        out = compute_loss(s, t, gt, pan, spec=spec, scharr=ScharrAbsolute('WV3'),
                           stats=FixedRanks(), loss_mask=mask, return_maps=True)
        maps = out['maps']
        hard = masked_mean(maps['w_h'] * maps['e_s'], mask)
        soft = masked_mean((1-maps['d_e'])*maps['a_T']*maps['delta_ts'], mask)
        torch.testing.assert_close(out['loss'], hard + spec.beta*soft)
        torch.testing.assert_close(out['hard_weighted'], hard)
        torch.testing.assert_close(out['soft_unscaled'], soft)
        out['loss'].backward()
        self.assertEqual(float(s.grad.masked_select(~mask.expand_as(s)).abs().sum()), 0.)
        self.assertGreater(float(s.grad.masked_select(mask.expand_as(s)).abs().sum()), 0.)
        self.assertFalse(out['loss_mask'].requires_grad)
        # Mask cannot hide corruption in a loss image.
        corrupt = s.detach().clone(); corrupt[..., 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compute_loss(corrupt, t, gt, pan, spec=spec, loss_mask=mask, stats=FixedRanks())

    def test_gt_only_equal_sample_not_global_pixel_weighted(self):
        s, _, gt, pan = images()
        s = torch.stack([gt[0] + 1, gt[1] + 3]).requires_grad_()
        mask = support()
        out = compute_loss(s, None, gt, pan, spec=config('gt_only', mask_policy=MASK2_POLICY),
                           loss_mask=mask)
        self.assertAlmostEqual(float(out['loss']), 2.)
        self.assertNotAlmostEqual(float((s-gt).abs().mean(1, keepdim=True)[mask].mean()), 2.)

    def test_microbatch_exact_equal_sample_masked_accumulation(self):
        s, t, gt, pan = images()
        mask, spec = support(), config('panda_e', mask_policy=MASK2_POLICY)
        p_full = torch.tensor(.2, requires_grad=True)
        compute_loss(s.detach()+p_full, t, gt, pan, spec=spec, stats=FixedRanks(),
                     loss_mask=mask)['loss'].backward()
        p_micro = torch.tensor(.2, requires_grad=True)
        for index in range(2):
            sl = slice(index, index+1)
            out = compute_loss(s[sl].detach()+p_micro, t[sl], gt[sl], pan[sl], spec=spec,
                               stats=FixedRanks(), loss_mask=mask[sl])
            (out['loss']/2).backward()
        torch.testing.assert_close(p_full.grad, p_micro.grad)

    def test_policy_missing_empty_and_invalid_roi_masks_fail_closed(self):
        s, t, gt, pan = images()
        spec = config('panda_e', mask_policy=MASK2_POLICY)
        for mask in (None, torch.zeros_like(pan, dtype=torch.bool), torch.ones_like(pan)):
            with self.subTest(mask=type(mask)), self.assertRaises(ValueError):
                compute_loss(s, t, gt, pan, spec=spec, stats=FixedRanks(), loss_mask=mask)
        mask = support(); mask[..., 9, 9] = False
        with self.assertRaisesRegex(ValueError, 'SUPPORT_VIOLATION'):
            compute_loss(s, t, gt, pan, spec=spec, stats=FixedRanks(), loss_mask=mask)
        with self.assertRaisesRegex(ValueError, 'Full-image control'):
            compute_loss(s, t, gt, pan, spec=config('panda_e'), stats=FixedRanks(), loss_mask=support())
        with self.assertRaisesRegex(ValueError, 'mask policy mismatch'):
            compute_loss(s, t, gt, pan, spec=spec,
                         stats=SimpleNamespace(tau_e=.25, mask_policy=MASK_POLICY), loss_mask=support())

    def test_calibration_valid_error_only_roi_q_unchanged_and_cross_policy_rejected(self):
        ids = ['a', 'b']; mask = support()
        e = torch.where(mask, torch.ones_like(mask, dtype=torch.float32), 1000.)
        g, q = torch.zeros_like(e), torch.ones_like(e)
        collector = CalibrationBuilder(ids, binding(ids), mask_policy=MASK2_POLICY,
                                       k=2, memory_budget_bytes=8*1024**2)
        collector.add(ids, e, g, q, sensor='WV3', loss_mask=mask)
        stats = collector.finalize()
        self.assertEqual(stats.tau_e, 1.)
        self.assertEqual(stats.schema, MASK2_CALIBRATION_SCHEMA)
        self.assertEqual(stats.mask_policy, MASK2_POLICY)
        self.assertEqual(stats.metadata['n_e'], int(mask.sum()))
        self.assertEqual(stats.metadata['n_q'], 2*48*48)
        self.assertEqual(stats.metadata['valid_e_counts_by_sample'],
                         [[key, int(mask[i].sum())] for i, key in enumerate(ids)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'masked.npz'
            stats.save(path)
            loaded = CalibrationStats.load(path, expected_binding=binding(ids), expected_mask_policy=MASK2_POLICY)
            self.assertEqual(stats.artifact_sha256, loaded.artifact_sha256)
            with self.assertRaisesRegex(ValueError, 'mask policy mismatch'):
                CalibrationStats.load(path, expected_binding=binding(ids), expected_mask_policy=MASK_POLICY)

    def test_masked_global_median_is_pixel_not_average_of_sample_medians(self):
        ids, mask = ['a', 'b'], support()
        e = torch.stack([torch.ones(1, 64, 64), torch.ones(1, 64, 64)*3])
        collector = CalibrationBuilder(ids, binding(ids), mask_policy=MASK2_POLICY,
                         include_condq=False, memory_budget_bytes=2*1024**2)
        collector.add(ids, e, sensor='WV3', loss_mask=mask)
        self.assertEqual(collector.finalize().tau_e, 1.)  # first sample has more valid pixels

    def test_calibration_order_independence_and_empty_support_poisoning(self):
        ids, mask = ['a', 'b'], support()
        e = torch.arange(2*64*64, dtype=torch.float32).reshape(2, 1, 64, 64)
        make = lambda: CalibrationBuilder(ids, binding(ids), mask_policy=MASK2_POLICY,
                        include_condq=False, memory_budget_bytes=2*1024**2)
        first, second = make(), make()
        first.add(ids, e, sensor='WV3', loss_mask=mask)
        second.add(ids[1:], e[1:], sensor='WV3', loss_mask=mask[1:])
        second.add(ids[:1], e[:1], sensor='WV3', loss_mask=mask[:1])
        self.assertEqual(first.finalize().artifact_sha256, second.finalize().artifact_sha256)
        failed = make()
        with self.assertRaises(ValueError):
            failed.add(ids, e, sensor='WV3', loss_mask=torch.zeros_like(mask))
        with self.assertRaises(RuntimeError):
            failed.finalize()

    def test_actual_stage_mask_and_full_inference_same_no_cross_reload(self):
        full = fixture('gt_only')
        masked = ConditionalQStage2(replace(full.spec, mask_policy=MASK2_POLICY),
            copy.deepcopy(full.aligner), donor_binding=full.donor_binding)
        rr = batch()
        out = masked(rr)
        expected = warp_support_mask(64, 64, out['correction'])
        self.assertTrue(torch.equal(expected, out['loss_mask']))
        torch.testing.assert_close(out['loss'], masked_mean((out['student']['y']-rr.gt).abs(), expected))
        with self.assertRaisesRegex(ValueError, 'Strict same-contract'):
            masked.load_state_dict(full.state_dict())
        # Masking changes only train reductions, not model inference or its references.
        torch.testing.assert_close(full.native_prediction(rr)['y'], masked.native_prediction(rr)['y'],
                                   rtol=0, atol=0)

    def test_actual_stage_rejects_full_image_stats_for_masked_loss(self):
        full = fixture('panda_e')
        with self.assertRaisesRegex(ValueError, 'mask policy mismatch'):
            ConditionalQStage2(replace(full.spec, mask_policy=MASK2_POLICY),
                full.aligner, full.teacher, donor_binding=full.donor_binding, stats=full.stats)

    def test_periodic_audit_still_rejects_frozen_version_mutation_each_update(self):
        trainer = StudentTrainer(fixture('gt_only'), TrainingSpec(
            effective_batch_size=1, microbatch_size=1, full_audit_interval=100))
        trainer.step(batch())
        with torch.no_grad():
            trainer.stage.aligner.fc2.bias.add_(.01)
        with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, 'version changed'):
                trainer.step(batch())
            step.assert_not_called()

    def test_periodic_audit_checkpoint_detects_data_alias_mutation(self):
        trainer = StudentTrainer(fixture('gt_only'), TrainingSpec(
            effective_batch_size=1, microbatch_size=1, full_audit_interval=100))
        trainer.step(batch())
        # Deliberate unsupported .data alias write bypasses Tensor version counters.
        trainer.stage.aligner.fc2.bias.data.add_(.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.pt'
            with self.assertRaisesRegex(ValueError, 'Aligner state changed'):
                save_checkpoint(path, trainer)
            self.assertFalse(path.exists())

    def test_masked_checkpoint_with_same_state_evaluation_and_schema_tamper_guard(self):
        def make():
            full = fixture('gt_only')
            stage = ConditionalQStage2(replace(full.spec, mask_policy=MASK2_POLICY),
                full.aligner, donor_binding=full.donor_binding)
            return StudentTrainer(stage, TrainingSpec(effective_batch_size=1,
                microbatch_size=1, full_audit_interval=100), evaluation_pin={})
        source = make()
        source.step(batch())
        record = official_evaluate(source, EvalSources(), metric_functions=metrics(),
                                   evaluator_revision='TEST')
        self.assertEqual(record['method_version'], source.stage.spec.method_version)
        self.assertEqual(record['pipeline']['mask_policy'], MASK2_POLICY)
        self.assertEqual(source.contract()['mask_policy'], MASK2_POLICY)
        self.assertTrue(validate_result(record))
        corrupt = copy.deepcopy(record)
        corrupt['pipeline']['loss_mask_policy'] = MASK_POLICY
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            validate_result(corrupt)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'masked.pt'
            saved = save_checkpoint(path, source, evaluation=record)
            target = make()
            load_checkpoint(path, saved['sha256'], target)
            self.assertEqual(target.updates, source.updates)
            torch.testing.assert_close(target.stage.native_prediction(batch())['y'],
                                       source.stage.native_prediction(batch())['y'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
