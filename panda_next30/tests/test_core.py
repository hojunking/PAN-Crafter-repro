"""Execute each new sampler through unchanged Hard training and exact resume."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from donor30.artifacts import _same
from donor30.masking import masked_mean
from directshift48.method import ScharrAbsolute
from panda_band30.tests.test_method import make_donor
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_next30 import plan, calibration as C, reporting as R
from panda_next30.contract import CASES, sampling_manifest
from panda_next30.engine import ScheduleTrainingSpec, MatureStudentTrainer
from panda_next30.method import MatureStage2, frozen_predictions
from panda_next30.spec import MatureSpec


def fixture(case):
    p = CASES[case]
    server='s3' if case=='S020C30H050' else 's1'
    cfg=plan.resolve_case(server,case,'R1')
    spec=MatureSpec(**cfg['stage2_spec'])
    donor = C.bind_donor(make_donor(shift=(1.,-.5)), spec)
    rr = batch(); key = spec.geometry_keys[0]
    frozen = frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
    e,g,q = reference_maps(ScharrAbsolute('WV3'),frozen['teacher_y'],rr.gt,frozen['pan_loss'])
    builder = CalibrationBuilder(['WV3:train:0'], C.binding_for_geometry(donor.binding,key),
        k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    builder.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=frozen['loss_mask'])
    stage = MatureStage2(spec,donor.aligner,donor.teacher,donor_binding=donor.binding,
                        stats_bank={key:builder.finalize()})
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    training = ScheduleTrainingSpec(lr=p['lr_S'],total_updates=p['total_updates'],microbatch_size=1,effective_batch_size=1)
    trainer=MatureStudentTrainer(stage,training)
    cfg['donor_binding']=deepcopy(donor.binding)
    trainer.config=cfg
    return trainer


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_each_sampling_variant_keeps_hard_loss_frozen_teacher_and_exact_resume(self):
        for case in ('S020C30','DAVG','UBIL','DAVG_UBIL','S100DIFFC30','S020C30H050'):
            with self.subTest(case=case):
                trainer = fixture(case); stage = trainer.stage; rr = batch()
                with patch.object(stage.student.backbone,'forward',wraps=stage.student.backbone.forward) as forward:
                    out = stage(rr, return_maps=True)
                self.assertEqual(forward.call_count, 1)
                maps = out['maps']; w = 1+stage.spec.alpha*maps['d_e']+stage.spec.gamma*maps['d_e']*maps['r_q_bar']
                expected = masked_mean(w*(out['student']['y']-rr.gt).abs().mean(1,keepdim=True),out['loss_mask'])
                torch.testing.assert_close(out['loss'],expected,rtol=0,atol=0)
                self.assertEqual(stage.deployment_manifest()['student_sampling'],sampling_manifest(stage.spec.sampling_variant, stage.spec.input_variant))
                self.assertEqual(sum(p.numel() for p in stage.student.parameters()),sampling_manifest(stage.spec.sampling_variant, stage.spec.input_variant)['student_parameters'])
                trainer.step_accumulated([rr])
                self.assertTrue(all(p.grad is None for m in (stage.aligner,stage.teacher) for p in m.parameters()))
                saved = trainer.state_dict(); trainer.step_accumulated([rr])
                expected = deepcopy(stage.student.state_dict())
                trainer.load_state_dict(saved); trainer.step_accumulated([rr])
                self.assertTrue(_same(stage.student.state_dict(),expected))
                self.assertEqual(trainer.updates, 2)

    def test_registry_binds_positive_lanczos_geometry_inputs_and_teacher(self):
        from panda_next30.contract import WARP_MANIFEST, input_manifest
        from panda_next30.spec import geometry_manifest
        for server in ('s1','s3'):
            for g in plan.groups(server):
                for case,replica in g['runs']:
                    cfg=plan.resolve_case(server,case,replica)
                    spec=MatureSpec(**cfg['stage2_spec'])
                    self.assertEqual(cfg['initialization'],'FRESH')
                    self.assertEqual(cfg['student_warp'],WARP_MANIFEST)
                    self.assertEqual(cfg['student_input'],input_manifest(spec.input_variant))
                    self.assertEqual(cfg['rotation_consistency_weight'],0.)
                    self.assertEqual(spec.beta,0.)
                    self.assertGreater(cfg['shift_scale'],0.)
                    self.assertEqual(spec.teacher_reconstruction_spec.pan_channels,1)
                    self.assertEqual(spec.reconstruction_spec.pan_channels,
                                     2 if spec.input_variant=='CORRECTED_DIFF' else 1)
                    self.assertEqual(geometry_manifest(spec.geometry_keys[0])['student_warp'],WARP_MANIFEST)
                    if spec.input_variant=='CORRECTED_DIFF':
                        self.assertEqual(cfg['shift_scale'],1.)
        with self.assertRaises(ValueError): ScheduleTrainingSpec(inherited_updates=25000)

    def test_lanczos_student_teacher_bicubic_and_difference_are_distinct_contracts(self):
        from panda_next30.method import student_pan_inputs, warp_support
        from panda_next30.warp import warp_pan
        from pa.warp import warp_pan as old_warp
        from panda_band30.reconstruction import reconstruct_with_base
        trainer=fixture('S100DIFFC30'); stage=trainer.stage; rr=batch()
        frozen=stage.frozen_predictions(rr)
        torch.testing.assert_close(frozen['pan_input'],warp_pan(rr.pan,frozen['u']),rtol=0,atol=0)
        expected_teacher=reconstruct_with_base(stage.teacher,old_warp(rr.pan,frozen['u']),
                                               frozen['ms_base_frame'],rr.ms)['y']
        torch.testing.assert_close(frozen['teacher_y'],expected_teacher,rtol=0,atol=0)
        inputs=student_pan_inputs(frozen['pan_input'],rr.pan,'CORRECTED_DIFF')
        self.assertEqual(inputs.shape[1],2)
        torch.testing.assert_close(inputs[:,1:2],frozen['pan_input']-rr.pan,rtol=0,atol=0)
        self.assertFalse(torch.equal(frozen['pan_input'],old_warp(rr.pan,frozen['u'])))
        out=stage.native_prediction(rr)
        self.assertEqual(out['y'].shape,rr.gt.shape)
        torch.testing.assert_close(out['pan_correction'],out['u'],rtol=0,atol=0)
        self.assertEqual(stage.deployment_manifest()['student_input']['channels'],10)
        warp_support(64,64,torch.tensor([[3.9,-3.9]],dtype=torch.float32),'s100')
        with self.assertRaisesRegex(ValueError,'SUPPORT_VIOLATION'):
            warp_support(64,64,torch.tensor([[4.1,0]],dtype=torch.float32),'s100')

    def test_actual_evaluation_pipeline_and_old_calibration_rejection(self):
        from types import SimpleNamespace
        from panda_next30.evaluation import configured_pipeline
        from panda_next30.contract import WARP_MANIFEST
        from panda_sampling30.calibration import base_protocol as old_protocol
        trainer=fixture('S100DIFFC30'); stage=trainer.stage
        source=SimpleNamespace(sensor='WV3',manifest_sha=lambda split: stage.donor_binding['source_manifest_sha256'])
        actual=configured_pipeline(trainer,source)
        self.assertEqual(actual['student_warp'],WARP_MANIFEST)
        self.assertEqual(actual['student_input']['channels'],10)
        self.assertEqual(actual['teacher']['pan_channels'],1)
        old_binding=deepcopy(stage.donor_binding); old_binding['mature30_base']=old_protocol()
        with self.assertRaises(ValueError):
            C.validate_base_binding(old_binding,stage.spec)
        corrupted=deepcopy(trainer.config); corrupted['student_input']['channels']=9
        trainer.config=corrupted
        with self.assertRaises(ValueError):
            configured_pipeline(trainer,source)

    def test_lanczos_shift_gradient_support_dc_and_integer_convention(self):
        from panda_next30.warp import warp_pan, warp_support_mask
        from panda_warp_candidates.ops import windowed_sinc_shift
        image=torch.randn(1,1,9,10,dtype=torch.float64,requires_grad=True)
        shift=torch.tensor([[.23,-.37]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(warp_pan,(image,shift),fast_mode=True))
        torch.testing.assert_close(warp_pan(image,shift),windowed_sinc_shift(image,shift),rtol=0,atol=0)
        torch.testing.assert_close(warp_pan(torch.ones_like(image),shift),torch.ones_like(image),rtol=1e-14,atol=1e-14)
        ramp=torch.arange(64*64,dtype=torch.float64).reshape(1,1,64,64)
        d=torch.tensor([[1.,-2.]],dtype=torch.float64)
        actual=warp_pan(ramp,d)
        torch.testing.assert_close(actual[...,8:-8,8:-8],ramp[...,9:-7,6:-10],rtol=1e-14,atol=1e-12)
        mask=warp_support_mask(64,64,d)
        self.assertTrue(bool(mask[...,7:-7,7:-7].all()))
        self.assertFalse(bool(mask[...,0,:].any()))


if __name__ == '__main__':
    unittest.main()
