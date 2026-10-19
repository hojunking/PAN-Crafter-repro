"""Actual W112/D123 geometry, nonzero residual, objective and gradient contracts."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from directshift48.method import ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import compute_loss, reference_maps
from panda_stage2_condq.tests.test_method import fixture as original_fixture, batch
from panda_midpoint.calibration import bind_donor
from panda_midpoint.method import (MidpointStage2, frozen_midpoint_predictions,
                                   midpoint_support, reconstruct_midpoint)
from panda_midpoint.spec import MidpointSpec, geometry_manifest


def fixture(*, shift=(1., -.5), beta=.05):
    original = original_fixture(shift=shift)
    # Avoid passing geometry tests trivially because a fresh output head is zero.
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        original.teacher.backbone.output[-1].weight.normal_(0., .003)
    binding = original.donor_binding
    binding['teacher_state_sha256'] = state_dict_sha256(original.teacher)
    donor = bind_donor(LoadedDonor(original.aligner, original.teacher, binding, {}))
    spec = MidpointSpec('WV3', 'panda_condq', 345, .25, beta, .5, 1e-6)
    rr = batch()
    out = frozen_midpoint_predictions(donor.aligner, donor.teacher, rr, sensor='WV3')
    e, g, q = reference_maps(ScharrAbsolute('WV3'), out['teacher_y'],
                             out['gt_midpoint'], out['pan_aligned'])
    collector = CalibrationBuilder(['WV3:train:0'], donor.binding, k=5,
        memory_budget_bytes=4*1024**2, include_condq=True, mask_policy=spec.mask_policy)
    collector.add(['WV3:train:0'], e, g, q, sensor='WV3', loss_mask=out['loss_mask'])
    stage = MidpointStage2(spec, donor.aligner, donor.teacher,
                          donor_binding=donor.binding, stats=collector.finalize())
    return stage


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_two_edges_meet_halfway_under_sampling_sign_convention(self):
        pan, ms = torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)
        pan[..., 32, 30], ms[..., 32, 32] = 1., 1.
        # PAN edge at x30 must move to MS edge at x32: sampling dx is -2.
        u = torch.tensor([[0., -2.]])
        torch.testing.assert_close(warp_pan(pan, u), ms, atol=0, rtol=0)
        left, right = warp_pan(pan, u/2), warp_pan(ms, -u/2)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(int(left[0, 0, 32].argmax()), 31)

    def test_original_aligner_inputs_original_teacher_then_transport(self):
        stage, rr = fixture(), batch()
        a_args, t_args = [], []
        ha = stage.aligner.register_forward_pre_hook(lambda m, args: a_args.append(args))
        ht = stage.teacher.register_forward_pre_hook(lambda m, args: t_args.append(args))
        out = stage.frozen_predictions(rr)
        ha.remove(); ht.remove()
        self.assertIs(a_args[0][0], rr.pan)
        self.assertIs(a_args[0][1], rr.native_lms)
        u = torch.tensor([[1., -.5]])
        torch.testing.assert_close(out['correction'], u)
        torch.testing.assert_close(out['pan_aligned'], warp_pan(rr.pan, u/2))
        torch.testing.assert_close(out['gt_midpoint'], warp_pan(rr.gt, -u/2))
        torch.testing.assert_close(t_args[0][0], warp_pan(rr.pan, u))
        self.assertIs(t_args[0][1], rr.ms)
        self.assertIs(t_args[0][2], rr.native_lms)
        torch.testing.assert_close(out['teacher_y'], warp_pan(out['teacher']['y'], -u/2))
        with torch.no_grad():
            wrong = stage.teacher(out['pan_aligned'], rr.ms, rr.native_lms)['y']
        self.assertGreater(float((wrong-out['teacher_y']).abs().max()), .01)
        self.assertFalse(any(t.requires_grad for t in (out['teacher_y'], out['gt_midpoint'])))

    def test_shifted_input_and_residual_base_nonzero_network(self):
        stage, rr = fixture(), batch()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(24)
            stage.student.backbone.output[-1].weight.normal_(0., .003)
        f = stage.frozen_predictions(rr)
        base = F.interpolate(rr.ms, scale_factor=4, mode='bicubic', align_corners=False)
        expected_base = warp_pan(base, -f['u']/2)
        seen = []
        hook = stage.student.backbone.input.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        got = reconstruct_midpoint(stage.student, f['pan_aligned'], f['ms_base_midpoint'], rr.ms)
        hook.remove()
        expected_input = torch.cat((f['pan_aligned'], expected_base), 1)
        torch.testing.assert_close(seen[0], expected_input, atol=0, rtol=0)
        residual = stage.student.backbone(f['pan_aligned'], None, rr.ms,
                                          rr.pan.new_ones(1), x_in=expected_input)
        self.assertGreater(float(residual.abs().max()), .01)
        torch.testing.assert_close(got['y'], expected_base+residual, atol=0, rtol=0)
        self.assertGreater(float((got['y']-(base+residual)).abs().max()), .01)

    def test_existing_loss_formula_with_midpoint_arguments_and_frozen_gradients(self):
        stage, rr = fixture(), batch()
        out = stage(rr, return_maps=True)
        expected = compute_loss(out['student']['y'], out['teacher_y'], out['gt_midpoint'],
            out['pan_aligned'], spec=stage.spec, scharr=stage.spatial, stats=stage.stats,
            return_maps=True, loss_mask=out['loss_mask'])
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(out[key], value, atol=0, rtol=0)
        out['loss'].backward()
        self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher) for p in m.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in stage.student.parameters()))
        stage.assert_frozen_binding()

    def test_three_beta_cases_have_identical_fresh_weights_and_geometry(self):
        hashes = []
        for beta in (.05, .10, .20):
            stage = fixture(beta=beta)
            hashes.append(stage.student_initial_state_sha256)
            self.assertEqual(stage.pipeline_manifest()['midpoint_geometry'], geometry_manifest())
            self.assertEqual(stage.student.spec.depth, (1, 2, 3))
            self.assertEqual(stage.student.spec.width, 112)
        self.assertEqual(len(set(hashes)), 1)

    def test_bidirectional_and_teacher_two_stage_support_not_single_pan_mask(self):
        u = torch.tensor([[1., -.5], [-2.5, 1.2]])
        got = midpoint_support(64, 64, u, include_teacher=True)['loss_mask']
        expected = (warp_support_mask(64, 64, u/2) & warp_support_mask(64, 64, -u/2)
                    & two_stage_support_mask(64, 64, u, -u/2))
        self.assertTrue(torch.equal(got, expected))
        self.assertFalse(torch.equal(got, warp_support_mask(64, 64, u)))
        self.assertTrue(bool(got[..., 7:-7, 7:-7].all()))
        with self.assertRaisesRegex(ValueError, 'SUPPORT_VIOLATION'):
            midpoint_support(64, 64, torch.tensor([[20., 0.]]), include_teacher=True)

    def test_zero_shift_recovers_original_forward_with_nonzero_output(self):
        stage, rr = fixture(shift=(0., 0.)), batch()
        with torch.no_grad():
            stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
            original = stage.student(rr.pan, rr.ms, rr.native_lms)['y']
            midpoint = stage.inference_model()(rr)['y']
        torch.testing.assert_close(midpoint, original, atol=1e-6, rtol=1e-6)

    def test_inference_two_warps_no_teacher_gt_cdf_or_output_return(self):
        stage, rr = fixture(), batch(split='test')
        inference = stage.inference_model().eval()
        self.assertEqual(set(inference._modules), {'aligner', 'student'})
        with (patch.object(stage.teacher, 'forward', side_effect=AssertionError('Teacher inference')),
              patch.object(stage.stats, 'lookup_exact', side_effect=AssertionError('CDF inference'))):
            with patch('panda_midpoint.method.warp_pan', wraps=warp_pan) as warp:
                got = inference(replace(rr, gt=torch.tensor(float('nan'))))
                self.assertEqual(warp.call_count, 2)
                torch.testing.assert_close(warp.call_args_list[0].args[1], got['u']/2)
                torch.testing.assert_close(warp.call_args_list[1].args[1], -got['u']/2)
            torch.testing.assert_close(got['y'], inference(rr)['y'], atol=0, rtol=0)
        self.assertNotIn('loss_mask', got)
        # Fresh S residual is zero: the unreturned output must be the shifted base.
        torch.testing.assert_close(got['y'], got['ms_base'], atol=0, rtol=0)

    def test_old_geometry_and_checkpoint_rejected_before_tensor_mutation(self):
        stage = fixture()
        binding = stage.donor_binding
        binding.pop('midpoint_geometry')
        with self.assertRaisesRegex(ValueError, 'geometry'):
            MidpointStage2(stage.spec, stage.aligner, stage.teacher,
                donor_binding=binding, stats=stage.stats)
        state = deepcopy(stage.state_dict())
        state['_extra_state']['method_version'] = 'LEGACY'
        with self.assertRaisesRegex(ValueError, 'Strict same-contract'):
            stage.load_state_dict(state)
        stage.assert_frozen_binding()

    def test_actual_rr256_fr512_shapes_no_hidden_scale_factor(self):
        stage = fixture()
        inference = stage.inference_model().eval()
        for side, resolution in ((256, 'RR'), (512, 'FR')):
            rr = batch(side=side, resolution=resolution, split='test')
            got = inference(rr)
            self.assertEqual(tuple(got['y'].shape), (1, 8, side, side))
            torch.testing.assert_close(got['u'], torch.tensor([[1., -.5]]))
            torch.testing.assert_close(got['pan_correction'], torch.tensor([[.5, -.25]]))
            torch.testing.assert_close(got['ms_correction'], torch.tensor([[-.5, .25]]))
            self.assertFalse(got['y'].requires_grad)


if __name__ == '__main__':
    unittest.main()
