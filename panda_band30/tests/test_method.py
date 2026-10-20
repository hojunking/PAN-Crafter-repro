"""Real W112/D123 transport, inverse-gradient and frozen Teacher contracts."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_method.calibration import state_dict_sha256, canonical_sha256
from panda_method.edge_data import augment_edge_batch
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import compute_loss, reference_maps
from panda_stage2_condq.tests.test_method import fixture as old_fixture, batch
from panda_band30.spec import BandSpec, FRAMES, geometry_manifest, teacher_binding
from panda_band30.aligner import BandPANAligner, aligned_pan_views, support_for_shifts
from panda_band30.reconstruction import build_reconstructor
FRAMES = tuple(f for f in FRAMES if f != 'band_pan')
from panda_band30.contract import CAMPAIGN_ID
from panda_band30.calibration import bind_donor
from panda_band30.method import BandStage2, frozen_predictions, warp_support, q_pan_reference
from panda_band30.reconstruction import base_from_batch, reconstruct_with_base


def make_spec(frame='pan', scale=1., upsampler='interp23'):
    band = frame == 'band_pan'
    return BandSpec('WV3', 'panda_condq', 345, .25, .05, .5, 1e-6,
        upsampler=upsampler, frame=frame, shift_scale=scale,
        aligner_outputs=16 if band else 2, pan_channels=8 if band else 1,
        alignment_target='MEAN8_MATCHED_BAND_SCHARR_LOSSES' if band else 'SCHARR_OF_MEAN8_GT')


def make_donor(upsampler='interp23', shift=(1., -.5), outputs=2):
    spec = make_spec('band_pan' if outputs == 16 else 'pan', upsampler=upsampler)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        aligner = BandPANAligner(outputs)
    teacher = build_reconstructor(spec.reconstruction_spec, seed=12, role='T')
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        teacher.backbone.output[-1].weight.normal_(0., .003)
        shifts = torch.as_tensor(shift).float()
        if outputs == 16 and shifts.shape == (2,):
            shifts = shifts.expand(8, 2)
        aligner.fc2.bias.copy_(shifts.reshape(-1))
    aligner.requires_grad_(False).eval()
    teacher.requires_grad_(False).eval()
    binding = {'sensor': 'WV3', 'source_split': 'RR_train_only',
        'sample_ids_sha256': canonical_sha256(['WV3:train:0']), 'source_manifest_sha256': 'a'*64,
        'aligner_state_sha256': state_dict_sha256(aligner), 'teacher_state_sha256': state_dict_sha256(teacher),
        'donor_checkpoint_id': 'c'*64,
        'reconstruction_spec': {'bands':8,'width':112,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
        'scharr': {'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[4,2,1],'roi_margin':8},
        'warp':'bicubic_border_FP32_align_corners_false_current_pan_px',
        'band30_teacher': teacher_binding(spec),
        'band30_reconstruction_spec': spec.reconstruction_spec.manifest()}
    return LoadedDonor(aligner, teacher, binding, {})


def fixture(frame='pan', scale=1., upsampler='interp23', shift=(1., -.5)):
    spec = make_spec(frame, scale, upsampler)
    donor = bind_donor(make_donor(upsampler, shift, outputs=spec.aligner_outputs), spec)
    rr = batch()
    f = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
    e, g, q = reference_maps(ScharrAbsolute('WV3'), f['teacher_y'], f['gt_loss'], f['pan_loss'])
    builder = CalibrationBuilder(['WV3:train:0'], donor.binding, k=5,
        memory_budget_bytes=4*1024**2, include_condq=True, mask_policy=spec.mask_policy)
    builder.add(['WV3:train:0'], e, g, q, sensor='WV3', loss_mask=f['loss_mask'])
    stage = BandStage2(spec, donor.aligner, donor.teacher,
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
            self.assertEqual(manifest['band30_geometry'], geometry_manifest(spec))
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
        for frame in ('mid_return',):
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
        for frame in ('mid_return',):
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
            warp_support(64,64,torch.tensor([[20., 0.]]),make_spec('mid_return'),include_teacher=True)

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
            if changed == 'geometry': binding['band30_geometry']['shift_scale'] = .5
            if changed == 'teacher': binding['band30_teacher']['upsampler'] = 'bicubic'
            if changed == 'campaign': binding['band30_teacher']['campaign_id'] = 'old'
            with self.assertRaises(ValueError):
                BandStage2(stage.spec,stage.aligner,stage.teacher,donor_binding=binding,stats=stage.stats)
        saved = deepcopy(stage.state_dict())
        saved['_extra_state']['pipeline']['band30_geometry']['frame'] = 'mid'
        with self.assertRaisesRegex(ValueError,'Strict same-contract'):
            stage.load_state_dict(saved)
        self.assertEqual(state_dict_sha256(stage.student),before)

    def test_inverse_trainer_exact_resume_next_optimizer_update(self):
        from donor30.artifacts import _same
        from panda_stage2_condq.training import TrainingSpec
        from panda_band30.engine import BandStudentTrainer
        training = TrainingSpec(total_updates=3, warmup_updates=0,
            microbatch_size=1, effective_batch_size=1, full_audit_interval=1)
        trainer = BandStudentTrainer(fixture('mid_return'), training)
        trainer.step_accumulated([batch()])
        saved = deepcopy(trainer.state_dict())
        restored = BandStudentTrainer(fixture('mid_return'), training)
        restored.load_state_dict(saved)
        trainer.step_accumulated([batch()])
        restored.step_accumulated([batch()])
        self.assertTrue(_same(trainer.state_dict(), restored.state_dict()))
        incompatible = BandStudentTrainer(fixture('pan'), training)
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(saved)

    def test_band_routing_q_mean_before_scharr_and_all_support_intersection(self):
        shifts = torch.tensor([[.2 + k*.15, -.8+k*.1] for k in range(8)])
        stage = fixture('band_pan', shift=shifts)
        rr = batch()
        out = stage(rr, return_maps=True)
        self.assertEqual(out['u'].shape, (1,8,2))
        expected = aligned_pan_views(rr.pan, shifts.unsqueeze(0))
        torch.testing.assert_close(out['pan_input'], expected)
        torch.testing.assert_close(out['student']['x_in'][:,:8], expected)
        torch.testing.assert_close(out['student']['x_in'][:,8:], base_from_batch(rr,'interp23'))
        qpan = expected[:,[4,2,1]].mean(1,keepdim=True)
        torch.testing.assert_close(out['pan_loss'], qpan)
        _,_,q = reference_maps(stage.spatial, out['teacher_y'], rr.gt, qpan)
        torch.testing.assert_close(out['maps']['q'], q)
        mask = support_for_shifts(64,64,out['u'])
        self.assertTrue(torch.equal(out['loss_mask'],mask))
        self.assertEqual(mask.shape,(1,1,64,64))
        torch.testing.assert_close(out['teacher']['pan_aligned'],expected)
        out['loss'].backward()
        self.assertTrue(all(p.grad is None for m in (stage.aligner,stage.teacher) for p in m.parameters()))
        self.assertTrue(all(p.grad is not None for p in stage.student.parameters()))
        self.assertGreater(float(stage.student.backbone.input.weight.grad.abs().sum()),0.)
        bad=out['u'].clone();bad[:,7,0]=20
        with self.assertRaisesRegex(ValueError,'SUPPORT_VIOLATION'):
            warp_support(64,64,bad,stage.spec,include_teacher=True)

    def test_band_equal_shifts_collapse_q_masks_and_fresh_split_stem(self):
        rr = batch()
        global_stage,band_stage = fixture(),fixture('band_pan')
        global_out,band_out = global_stage(rr,return_maps=True),band_stage(rr,return_maps=True)
        torch.testing.assert_close(global_out['pan_loss'],band_out['pan_loss'],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(global_out['maps']['q'],band_out['maps']['q'],atol=1e-7,rtol=1e-6)
        self.assertTrue(torch.equal(global_out['loss_mask'],band_out['loss_mask']))
        # Equal PAN copies recover the old stem's function, not just zero-output identity.
        g=global_stage.student.backbone.input(global_out['student']['x_in'])
        b=band_stage.student.backbone.input(band_out['student']['x_in'])
        torch.testing.assert_close(g,b,atol=2e-6,rtol=2e-5)
        self.assertEqual(band_stage.student.backbone.input.in_channels,16)
        self.assertNotEqual(global_stage.stats.binding,band_stage.stats.binding)

    def test_band_gt_free_rr_fr_inference_and_nonzero_shift_units(self):
        shifts=torch.tensor([[.25*k,-.125*k] for k in range(8)])
        stage=fixture('band_pan',shift=shifts)
        inference=stage.inference_model()
        self.assertEqual(set(inference._modules),{'aligner','student'})
        with patch.object(stage.stats,'lookup_exact',side_effect=AssertionError('CDF')):
            for side,resolution in ((256,'RR'),(512,'FR')):
                rr=batch(side=side,resolution=resolution,split='test')
                out=inference(replace(rr,gt=torch.tensor(float('nan'))))
                self.assertEqual(out['y'].shape,(1,8,side,side))
                torch.testing.assert_close(out['u'][0],shifts)
                torch.testing.assert_close(out['pan_correction'][0],shifts)
                self.assertFalse(out['y'].requires_grad)

    def test_band_policy_and_stale_q_calibration_rejected(self):
        for updates in ({'frame':'mid'}, {'shift_scale':.5}, {'upsampler':'bicubic'},
                        {'pan_channels':1}, {'alignment_target':'SCHARR_OF_MEAN8_GT'}):
            with self.assertRaises(ValueError):
                replace(make_spec('band_pan'),**updates)
        with self.assertRaises(ValueError):
            make_spec('ms_return')
        stage=fixture('band_pan')
        for key in ('band30_q_reference','band30_reconstruction_spec'):
            bad=stage.donor_binding;bad.pop(key)
            with self.assertRaises(ValueError):
                BandStage2(stage.spec,stage.aligner,stage.teacher,donor_binding=bad,stats=stage.stats)
        saved=deepcopy(stage.state_dict())
        saved['_extra_state']['pipeline']['band30_q_reference']['pan_reference']='single'
        with self.assertRaisesRegex(ValueError,'Strict same-contract'):
            stage.load_state_dict(saved)

    def test_band_masked_loss_full48_equals_four_micro12_with_gradient(self):
        stage=fixture('band_pan')
        gen=torch.Generator().manual_seed(56)
        y=torch.rand(48,8,64,64,generator=gen,requires_grad=True)
        t=torch.rand(48,8,64,64,generator=gen)
        gt=torch.rand(48,8,64,64,generator=gen)
        pan=torch.rand(48,1,64,64,generator=gen)
        shifts=torch.rand(48,8,2,generator=gen)*2-1
        mask=warp_support(64,64,shifts,stage.spec,include_teacher=True)['loss_mask']
        common=dict(spec=stage.spec,scharr=stage.spatial,stats=stage.stats)
        full=compute_loss(y,t,gt,pan,loss_mask=mask,**common)['loss']
        full_grad=torch.autograd.grad(full,y)[0]
        ymicro=y.detach().clone().requires_grad_()
        accumulated=sum(compute_loss(ymicro[i:i+12],t[i:i+12],gt[i:i+12],pan[i:i+12],
            loss_mask=mask[i:i+12],**common)['loss']/4 for i in range(0,48,12))
        micro_grad=torch.autograd.grad(accumulated,ymicro)[0]
        torch.testing.assert_close(full,accumulated,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(full_grad,micro_grad,atol=1e-10,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
