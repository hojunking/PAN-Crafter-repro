"""Support geometry, zero invalid-pixel gradients, lineage and inference parity."""
import copy
from dataclasses import replace
import math
import unittest

import torch

from directshift48.test_method import batch, model as legacy_model
from pa.warp import warp_support_mask
from .masking import masked_mean, MaskedDonorStage1, LOSS_MASK_POLICY
from .method import (loss_terms, reference_maps, calibrate_training,
                     ExactDirectCalibrator, calibration_binding)
from .test_method import model, setup


class MaskTests(unittest.TestCase):
    def test_four_tap_geometry_independent_of_image_values(self):
        shifts = torch.tensor([[0., 0.], [.25, -.75], [3.25, -4.5], [-5.25, 4.1]], requires_grad=True)
        mask = warp_support_mask(32, 32, shifts)
        expected = torch.tensor([[[all(0 <= v <= 31 for v in
            (math.floor(y + dy) - 1, math.floor(y + dy) + 2,
             math.floor(x + dx) - 1, math.floor(x + dx) + 2))
            for x in range(32)] for y in range(32)] for dy, dx in shifts.detach().tolist()])[:, None]
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(mask.requires_grad)
        self.assertEqual(int(mask[0].sum()), 29 * 29)  # Conservative even at integer zero.

    def test_per_sample_count_not_full_image_or_pooled_batch_denominator(self):
        x = torch.stack((torch.ones(2, 4, 4), torch.full((2, 4, 4), 3.))).requires_grad_()
        mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        mask[0, :, 1, 1] = True
        mask[1, :, :3, :3] = True
        result = masked_mean(x, mask)
        self.assertEqual(float(result), 2.)
        partitioned = sum(masked_mean(x[i:i+1], mask[i:i+1]) for i in range(2)) / 2
        torch.testing.assert_close(result, partitioned)
        result.backward()
        invalid = ~mask.expand_as(x)
        self.assertEqual(float(x.grad[invalid].abs().sum()), 0.)
        self.assertTrue(bool((x.grad[~invalid] > 0).all()))
        torch.testing.assert_close(masked_mean(x, torch.ones_like(mask)), x.mean())

    def test_empty_or_wrong_mask_fails_not_silent_zero_loss(self):
        x = torch.ones(2, 4, 32, 32)
        mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
        mask[1] = False
        with self.assertRaisesRegex(ValueError, 'EMPTY_SUPPORT'):
            masked_mean(x, mask)
        with self.assertRaises(ValueError):
            masked_mean(x, mask.float())
        with self.assertRaises(ValueError):
            masked_mean(x, mask.expand_as(x))

    def test_donor_rec_has_zero_output_gradient_outside_support_both_sensors(self):
        for sensor in ('WV3', 'GF2'):
            stage, rr = model(sensor), batch(sensor=sensor)
            with torch.no_grad():
                stage.aligner.fc2.bias.copy_(torch.tensor([3.25, -2.75]))
            out = stage(rr)
            y, mask = out['native']['y'], out['loss_mask']
            grad, = torch.autograd.grad(out['rec'], y, retain_graph=True)
            self.assertEqual(float(grad[~mask.expand_as(y)].abs().sum()), 0.)
            self.assertGreater(float(grad[mask.expand_as(y)].abs().sum()), 0.)
            torch.testing.assert_close(out['rec'], masked_mean((y - rr.gt).abs(), mask))
            changed = rr.gt.clone()
            changed[~mask.expand_as(changed)] += 100
            # No augmentation, GT or spatial border error can enter reconstruction.
            changed_out = stage(replace(rr, gt=changed))
            torch.testing.assert_close(changed_out['rec'], out['rec'])
            torch.testing.assert_close(changed_out['abs_rr'], out['abs_rr'])

    def test_all_student_modes_zero_invalid_output_gradient(self):
        stage, _, rr, scales = setup()
        mask = warp_support_mask(32, 32, torch.tensor([[3.25, -2.75], [-.25, .75]]))
        for mode in ('gt', 'e', 'eq', 'uniform'):
            s = rr.gt.clone().requires_grad_()
            t, gt = rr.gt + .7, rr.gt + .3
            result = loss_terms(stage.spatial, s, t, gt, rr.pan,
                                mode=mode, calibration=scales, valid_mask=mask)
            grad, = torch.autograd.grad(result['loss'], s)
            self.assertEqual(float(grad[~mask.expand_as(s)].abs().sum()), 0.)
            self.assertGreater(float(grad[mask.expand_as(s)].abs().sum()), 0.)
            altered = s.detach().clone()
            altered[~mask.expand_as(s)] += 100
            compared = loss_terms(stage.spatial, altered, t, gt, rr.pan,
                                 mode=mode, calibration=scales, valid_mask=mask)
            torch.testing.assert_close(result['loss'], compared['loss'])

    def test_calibration_border_errors_do_not_enter_tau_e_or_tau_q(self):
        stage, rr = model(), batch()
        mask = warp_support_mask(32, 32, torch.tensor([[3.25, -2.75], [-.25, .75]]))
        baseline = ExactDirectCalibrator(calibration_binding(stage), stage.spatial,
                                        expected_sample_ids=rr.lineage.sample_ids)
        changed = ExactDirectCalibrator(calibration_binding(stage), stage.spatial,
                                       expected_sample_ids=rr.lineage.sample_ids)
        teacher = rr.gt + .25
        corrupted = teacher.clone()
        corrupted[~mask.expand_as(teacher)] += 100
        for c, t in ((baseline, teacher), (changed, corrupted)):
            c.update(t, rr.gt, rr.pan, split='train', sample_ids=rr.lineage.sample_ids, valid_mask=mask)
        original, altered = baseline.finalize(), changed.finalize()
        self.assertEqual(original, altered)
        self.assertEqual(original.n_e, int(mask.sum()))
        self.assertEqual(original.n_q, 2 * 16 * 16)

    def test_native_predictions_unchanged_and_old_checkpoints_rejected(self):
        for sensor in ('WV3', 'GF2'):
            old, new = legacy_model(sensor), model(sensor)
            for stage in (old, new):
                with torch.no_grad():
                    stage.aligner.fc2.bias.copy_(torch.tensor([.25, -.75]))
            for res in ('RR', 'FR'):
                rr = batch(sensor=sensor, resolution=res, split='test')
                torch.testing.assert_close(old.native_prediction(rr)['y'], new.native_prediction(rr)['y'],
                                           rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'pipeline'):
                new.load_state_dict(old.state_dict())
            roundtrip = model(sensor)
            roundtrip.load_state_dict(copy.deepcopy(new.state_dict()))
            self.assertEqual(roundtrip.pipeline_manifest()['rec_reduction'], LOSS_MASK_POLICY)
            with self.assertRaises(TypeError):
                calibrate_training(old, [], expected_sample_ids=('sample',))


if __name__ == '__main__':
    unittest.main()
