"""Real W112/D123 transport, inverse-gradient and frozen Teacher contracts."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_method.calibration import state_dict_sha256
from panda_method.edge_data import augment_edge_batch
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import compute_loss, reference_maps
from panda_stage2_condq.tests.test_method import fixture as old_fixture, batch
from panda_warp30.spec import WarpSpec, FRAMES, geometry_manifest
from panda_warp30.contract import CAMPAIGN_ID
from panda_warp30.calibration import bind_donor
from panda_warp30.method import WarpStage2, frozen_predictions, warp_support
from panda_warp30.reconstruction import base_from_batch, reconstruct_with_base


def make_spec(frame='pan', scale=1., upsampler='interp23'):
    return WarpSpec('WV3', 'panda_condq', 345, .25, .05, .5, 1e-6,
                    upsampler=upsampler, frame=frame, shift_scale=scale)


def make_donor(upsampler='interp23', shift=(1., -.5)):
    old = old_fixture(shift=shift)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        old.teacher.backbone.output[-1].weight.normal_(0., .003)
    binding = old.donor_binding
    binding['teacher_state_sha256'] = state_dict_sha256(old.teacher)
    binding['warp30_teacher'] = {'upsampler': upsampler, 'campaign_id': CAMPAIGN_ID}
    return LoadedDonor(old.aligner, old.teacher, binding, {})


def fixture(frame='pan', scale=1., upsampler='interp23', shift=(1., -.5)):
    spec = make_spec(frame, scale, upsampler)
    donor = bind_donor(make_donor(upsampler, shift), spec)
    rr = batch()
    f = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
    e, g, q = reference_maps(ScharrAbsolute('WV3'), f['teacher_y'], f['gt_loss'], f['pan_loss'])
    builder = CalibrationBuilder(['WV3:train:0'], donor.binding, k=5,
        memory_budget_bytes=4*1024**2, include_condq=True, mask_policy=spec.mask_policy)
    builder.add(['WV3:train:0'], e, g, q, sensor='WV3', loss_mask=f['loss_mask'])
    stage = WarpStage2(spec, donor.aligner, donor.teacher,
                      donor_binding=donor.binding, stats=builder.finalize())
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    return stage


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_explicit_registered_axes_and_manifests(self):
        for frame in FRAMES:
            spec = make_spec(frame)
            manifest = spec.manifest()
            self.assertEqual(manifest['warp30_geometry'], geometry_manifest(spec))
            self.assertEqual(manifest['student_constructor_spec']['base_reference'], 'BICUBIC_MS')
            self.assertIn('INTERP23', manifest['student']['base_reference'])
        for kwargs in ({'frame': 'unknown'}, {'shift_scale': 4.}, {'shift_scale': True},
                       {'upsampler': 'dlpan_warp'}, {'shift_scale': float('nan')}):
            with self.assertRaises(ValueError):
                replace(make_spec(), **kwargs)

    def test_all_frames_correct_inputs_targets_output_inverse_with_nonzero_head(self):
        rr = batch()
        for frame in FRAMES:
            with self.subTest(frame=frame):
                stage = fixture(frame)
                out = stage(rr, return_maps=True)
                u, rho = out['u'], stage.spec.pan_fraction
                p, m = rho*u, (rho-1)*u
                base = base_from_batch(rr, 'interp23')
                pan = rr.pan if rho == 0 else warp_pan(rr.pan, p)
                moved_base = base if rho == 1 else warp_pan(base, m)
                torch.testing.assert_close(out['pan_input'], pan)
                torch.testing.assert_close(out['ms_base_frame'], moved_base)
                raw = reconstruct_with_base(stage.student, pan, moved_base, rr.ms)
                self.assertGreater(float((raw['y']-moved_base).abs().max()), .01)
                expected = warp_pan(raw['y'], -m) if stage.spec.returns_output else raw['y']
                torch.testing.assert_close(out['student']['y'], expected, atol=0, rtol=0)
                gt = warp_pan(rr.gt, m) if frame == 'mid' else rr.gt
                qpan = pan if frame == 'mid' else warp_pan(rr.pan, u)
                torch.testing.assert_close(out['gt_loss'], gt)
                torch.testing.assert_close(out['pan_loss'], qpan)
                terms = compute_loss(expected, out['teacher_y'], gt, qpan,
                    spec=stage.spec, scharr=stage.spatial, stats=stage.stats,
                    loss_mask=out['loss_mask'], return_maps=True)
                torch.testing.assert_close(out['loss'], terms['loss'], atol=0, rtol=0)
                torch.testing.assert_close(out['maps']['q'], terms['maps']['q'], atol=0, rtol=0)

    def test_teacher_always_original_full_shift_same_function_for_all_student_policies(self):
        rr, donor = batch(), make_donor()
        u = donor.aligner(rr.pan, rr.native_lms)
        fixed = reconstruct_with_base(donor.teacher, warp_pan(rr.pan, u),
                                      base_from_batch(rr, 'interp23'), rr.ms)['y']
        for frame in FRAMES:
            for scale in (0., .25, .5, 1.):
                spec = make_spec(frame, scale)
                with patch.object(donor.teacher, 'forward', side_effect=AssertionError('legacy base')):
                    f = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
                torch.testing.assert_close(f['teacher']['y'], fixed, atol=0, rtol=0)
                expected = warp_pan(fixed, -.5*scale*u) if frame == 'mid' and scale else fixed
                torch.testing.assert_close(f['teacher_y'], expected, atol=0, rtol=0)

    def test_inverse_is_inside_autograd_only_student_receives_gradients(self):
        for frame in ('mid_return', 'ms_return'):
            stage = fixture(frame)
            out = stage(batch())
            y, raw = out['student']['y'], out['student']['y_frame']
            self.assertTrue(y.requires_grad and raw.requires_grad)
            self.assertIsNot(y, raw)
            grad = torch.autograd.grad(out['loss'], raw, retain_graph=True)[0]
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().sum()), 0.)
            out['loss'].backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum())
                                for p in stage.student.parameters()))
            self.assertTrue(all(p.grad is None for module in (stage.aligner, stage.teacher)
                                for p in module.parameters()))
            stage.assert_frozen_binding()

    def test_zero_strength_all_frames_collapse_output_loss_mask_and_cues(self):
        for upsampler in ('bicubic', 'interp23'):
            reference = None
            for frame in FRAMES:
                stage = fixture(frame, 0., upsampler)
                out = stage(batch(), return_maps=True)
                values = (out['student']['y'], out['loss'], out['loss_mask'], out['maps']['q'],
                          out['gt_loss'], out['teacher_y'])
                if reference is None:
                    reference = tuple(v.detach().clone() for v in values)
                else:
                    for got, expected in zip(values, reference):
                        torch.testing.assert_close(got, expected, atol=0, rtol=0)
                self.assertIs(out['student']['y'], out['student']['y_frame'])

    def test_return_masks_track_two_stages_not_net_zero(self):
        u = torch.tensor([[1., -.5], [-2., 1.]])
        for frame in ('mid_return', 'ms_return'):
            spec = make_spec(frame)
            rho, d = spec.pan_fraction, u
            p, m, back = rho*d, (rho-1)*d, (1-rho)*d
            expected = (warp_support_mask(64,64,back) if rho == 0 else
                        two_stage_support_mask(64,64,p,back))
            expected &= two_stage_support_mask(64,64,m,back)
            expected &= warp_support_mask(64,64,d) & warp_support_mask(64,64,u)
            mask = warp_support(64,64,u,spec,include_teacher=True)['loss_mask']
            self.assertTrue(torch.equal(mask, expected))
            self.assertFalse(torch.equal(mask, warp_support_mask(64,64,u)))
        with self.assertRaisesRegex(ValueError, 'SUPPORT_VIOLATION'):
            warp_support(64,64,torch.tensor([[20., 0.]]),make_spec('ms_return'),include_teacher=True)

    def test_augmented_student_train_inference_parity_and_frozen_teacher_base(self):
        for upsampler in ('bicubic','interp23'):
            stage = fixture('mid_return', .5, upsampler)
            for rotation in range(4):
                rr = augment_edge_batch(batch(), hflip=True, vflip=True, rot90=rotation)
                out = stage(rr)
                inferred = stage.inference_model()(rr)
                torch.testing.assert_close(out['student']['y'], inferred['y'], atol=0,rtol=0)
                torch.testing.assert_close(out['teacher']['ms_base'], base_from_batch(rr,upsampler))

    def test_inference_no_gt_teacher_cdf_and_current_pixel_units_rr_fr(self):
        for frame in FRAMES:
            stage = fixture(frame, .5)
            inference = stage.inference_model()
            self.assertEqual(set(inference._modules), {'aligner','student'})
            with patch.object(stage.teacher,'forward',side_effect=AssertionError('Teacher')), \
                 patch.object(stage.stats,'lookup_exact',side_effect=AssertionError('CDF')):
                for side, resolution in ((256,'RR'),(512,'FR')):
                    rr = batch(side=side,resolution=resolution,split='test')
                    out = inference(replace(rr,gt=torch.tensor(float('nan'))))
                    self.assertEqual(out['y'].shape,(1,8,side,side))
                    torch.testing.assert_close(out['u'],torch.tensor([[1.,-.5]]))
                    torch.testing.assert_close(out['pan_correction'],out['u']*.5*stage.spec.pan_fraction)
                    self.assertFalse(out['y'].requires_grad)
                    self.assertNotIn('loss_mask',out)

    def test_wrong_donor_geometry_scale_base_and_restart_rejected_before_mutation(self):
        stage = fixture('mid_return')
        before = state_dict_sha256(stage.student)
        for changed in ('geometry','teacher','campaign'):
            binding = stage.donor_binding
            if changed == 'geometry': binding['warp30_geometry']['shift_scale'] = .5
            if changed == 'teacher': binding['warp30_teacher']['upsampler'] = 'bicubic'
            if changed == 'campaign': binding['warp30_teacher']['campaign_id'] = 'old'
            with self.assertRaises(ValueError):
                WarpStage2(stage.spec,stage.aligner,stage.teacher,donor_binding=binding,stats=stage.stats)
        saved = deepcopy(stage.state_dict())
        saved['_extra_state']['pipeline']['warp30_geometry']['frame'] = 'mid'
        with self.assertRaisesRegex(ValueError,'Strict same-contract'):
            stage.load_state_dict(saved)
        self.assertEqual(state_dict_sha256(stage.student),before)

    def test_inverse_trainer_exact_resume_next_optimizer_update(self):
        from donor30.artifacts import _same
        from panda_stage2_condq.training import TrainingSpec
        from panda_warp30.engine import WarpStudentTrainer
        training = TrainingSpec(total_updates=3, warmup_updates=0,
            microbatch_size=1, effective_batch_size=1, full_audit_interval=1)
        trainer = WarpStudentTrainer(fixture('mid_return'), training)
        trainer.step_accumulated([batch()])
        saved = deepcopy(trainer.state_dict())
        restored = WarpStudentTrainer(fixture('mid_return'), training)
        restored.load_state_dict(saved)
        trainer.step_accumulated([batch()])
        restored.step_accumulated([batch()])
        self.assertTrue(_same(trainer.state_dict(), restored.state_dict()))
        incompatible = WarpStudentTrainer(fixture('ms_return'), training)
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(saved)


if __name__ == '__main__':
    unittest.main()
