"""Actual positive PAN warp, unchanged single-view Hard and complete Student resume."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from donor30.masking import masked_mean
from pa.warp import warp_pan, warp_support_mask
from panda_band30.tests.test_method import make_donor
from panda_band30.tests.test_calibration import SyntheticInventory
from panda_band30.reconstruction import base_from_batch, reconstruct_with_base
from panda_method.calibration import state_dict_sha256, canonical_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps, compute_loss
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.training import TrainingSpec
from panda_scale30 import calibration as C, plan
from panda_scale30.contract import CASES
from panda_scale30.engine import MatureStudentTrainer, build_fresh_stage2
from panda_scale30.method import MatureStage2, applied_correction, frozen_predictions, warp_support
from panda_scale30.spec import MatureSpec, policy_manifest, view_manifest


def make_spec(case='S020BASE'):
    row = CASES[case]
    return MatureSpec('WV3','panda_condq',345,row['alpha'],0.,row['gamma'],1e-6,
                      policy_id=row['policy_id'])


def make_fixture(case='S020BASE', shift=(1.,-.5)):
    spec, rr = make_spec(case), batch()
    donor = C.bind_donor(make_donor(shift=shift),spec)
    key = spec.geometry_keys[0]
    frozen = frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
    e,g,q = reference_maps(ScharrAbsolute('WV3'),frozen['teacher_y'],rr.gt,frozen['pan_loss'])
    collector = CalibrationBuilder(['WV3:train:0'],C.binding_for_geometry(donor.binding,key),
        k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    collector.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=frozen['loss_mask'])
    bank = {key:collector.finalize()}
    stage = MatureStage2(spec,donor.aligner,donor.teacher,donor_binding=donor.binding,stats_bank=bank)
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    return stage,donor,bank


def make_trainer(case='S020BASE'):
    stage,_,_ = make_fixture(case)
    training = TrainingSpec(lr=CASES[case]['lr_S'],total_updates=50000,warmup_updates=0,
        microbatch_size=1,effective_batch_size=1,full_audit_interval=1)
    return MatureStudentTrainer(stage,training)


class MethodCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_registered_positive_scale_and_no_new_loss_contract(self):
        for case,row in CASES.items():
            spec = make_spec(case)
            self.assertEqual(spec.geometry_keys,(row['policy_id'].lower(),))
            self.assertEqual(spec.geometry_at_completed(50000),spec.geometry_at_next(0))
            self.assertFalse(spec.paired_views)
            self.assertEqual(spec.consistency_lambda,0.)
            self.assertEqual(view_manifest(spec)['student_views'],1)
            self.assertEqual(policy_manifest(spec)['phases'][0]['geometry']['scale'],row['scale'])
        for values in ({'policy_id':'ZERO'},{'policy_id':'ROTCONS'},{'beta':.05},
                       {'alpha':.4},{'gamma':0.},{'frame':'mid'}):
            with self.assertRaises(ValueError):
                replace(make_spec(),**values)
        for key in ('s000','s025','s100','cap085'):
            with self.assertRaises(ValueError):
                applied_correction(torch.ones(1,2),key)
        with self.assertRaises(ValueError):
            applied_correction(torch.ones(1,2,dtype=torch.float64),'s020')
        with self.assertRaises(ValueError):
            make_spec().geometry_at_next(50000)

    def test_true_warp_at_all_scales_teacher_gt_base_stay_unchanged(self):
        rr,donor = batch(),make_donor(shift=(2.,-1.))
        base = base_from_batch(rr,'interp23')
        u = donor.aligner(rr.pan,rr.native_lms)
        teacher = reconstruct_with_base(donor.teacher,warp_pan(rr.pan,u),base,rr.ms)['y']
        for scale in (.1,.2,.3,.4,.5):
            key = f's{round(scale*100):03d}'
            frozen = frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
            torch.testing.assert_close(frozen['pan_correction'],scale*u,rtol=0,atol=0)
            torch.testing.assert_close(frozen['pan_input'],warp_pan(rr.pan,scale*u),rtol=0,atol=0)
            self.assertFalse(torch.equal(frozen['pan_input'],rr.pan))
            torch.testing.assert_close(frozen['teacher_y'],teacher,rtol=0,atol=0)
            torch.testing.assert_close(frozen['ms_base_frame'],base,rtol=0,atol=0)
            self.assertIs(frozen['gt_loss'],rr.gt)
            support = warp_support_mask(64,64,scale*u)&warp_support_mask(64,64,u)
            self.assertTrue(torch.equal(frozen['loss_mask'],support))
            self.assertEqual(int(torch.count_nonzero(frozen['ms_correction'])),0)
            self.assertEqual(int(torch.count_nonzero(frozen['output_correction'])),0)
        with self.assertRaisesRegex(ValueError,'SUPPORT_VIOLATION'):
            warp_support(64,64,torch.tensor([[20.,0.]]),'s010')

    def test_actual_single_forward_existing_hard_formula_and_frozen_gradients(self):
        for case in ('S020BASE','S020H050','S020H200'):
            stage,_,_ = make_fixture(case)
            rr = batch()
            with patch.object(stage.student.backbone,'forward',wraps=stage.student.backbone.forward) as student, \
                 patch.object(stage.aligner,'forward',wraps=stage.aligner.forward) as aligner:
                out = stage(rr,return_maps=True)
            self.assertEqual(student.call_count,1)
            self.assertEqual(aligner.call_count,1)
            maps = out['maps']
            weights = 1+stage.spec.alpha*maps['d_e']+stage.spec.gamma*maps['d_e']*maps['r_q_bar']
            expected = masked_mean(weights*(out['student']['y']-rr.gt).abs().mean(1,keepdim=True),
                                   out['loss_mask'])
            torch.testing.assert_close(out['loss'],expected,rtol=0,atol=0)
            torch.testing.assert_close(out['loss'],out['hard_weighted'],rtol=0,atol=0)
            self.assertEqual(float(out['soft_weighted']),0.)
            self.assertNotIn('student_rotated',out)
            self.assertNotIn('consistency_weighted',out)
            out['loss'].backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in stage.student.parameters()))
            self.assertTrue(all(p.grad is None for m in (stage.aligner,stage.teacher) for p in m.parameters()))

    def test_half_scale_matches_existing_mature_method_numerically(self):
        from panda_mature30.tests.test_schedule_core import make_fixture as historical_fixture
        stage,_,_ = make_fixture('S050BASE')
        historical,_,_ = historical_fixture('HALF')
        rr = batch()
        current,old = stage(rr,return_maps=True),historical(rr,return_maps=True)
        for key in ('loss','hard_weighted','teacher_y','pan_loss','loss_mask','w_h','r_q_bar'):
            torch.testing.assert_close(current[key],old[key],rtol=0,atol=0)
        torch.testing.assert_close(current['student']['y'],old['student']['y'],rtol=0,atol=0)

    def test_positive_inference_same_rr_fr_scale_no_teacher_gt_or_calibration(self):
        stage,_,_ = make_fixture('S010LR150H050')
        model = stage.inference_model()
        self.assertEqual(set(model._modules),{'aligner','student'})
        with patch.object(stage.teacher.backbone,'forward',side_effect=AssertionError('Teacher forbidden')), \
             patch.object(stage.student.backbone,'forward',wraps=stage.student.backbone.forward) as student:
            for resolution in ('RR','FR'):
                rr = batch(side=64,resolution=resolution,split='test')
                out = model(replace(rr,gt=torch.tensor(float('nan'))))
                torch.testing.assert_close(out['pan_correction'],.1*out['u'],rtol=0,atol=0)
                torch.testing.assert_close(out['pan_aligned'],warp_pan(rr.pan,.1*out['u']),rtol=0,atol=0)
                self.assertGreater(int(torch.count_nonzero(out['pan_correction'])),0)
                self.assertNotIn('loss_mask',out)
                self.assertFalse(out['y'].requires_grad)
            self.assertEqual(student.call_count,2)

    def test_exact_resume_rejects_scale_coeff_and_lr_changes(self):
        trainer = make_trainer('S020LR150H050')
        first = trainer.step_accumulated([batch()])
        self.assertEqual(first['student_views'],1)
        self.assertEqual(first['consistency_lambda'],0.)
        self.assertEqual(first['loss'],first['hard_weighted'])
        self.assertEqual(first['gradient']['beta_soft_norm'],0.)
        state = deepcopy(trainer.state_dict())
        restored = make_trainer('S020LR150H050')
        restored.load_state_dict(deepcopy(state))
        trainer.step_accumulated([batch()])
        restored.step_accumulated([batch()])
        self.assertTrue(_same(trainer.state_dict(),restored.state_dict()))
        for case in ('S010LR150H050','S020LR150','S020H050'):
            wrong = make_trainer(case)
            before = state_dict_sha256(wrong.stage.student)
            with self.assertRaises(ValueError):
                wrong.load_state_dict(deepcopy(state))
            self.assertEqual(before,state_dict_sha256(wrong.stage.student))
        state['stage']['_completed_update_state'].add_(1)
        with self.assertRaisesRegex(ValueError,'schedule state'):
            restored.load_state_dict(state)

    def test_same_scale_calibration_reused_across_existing_hard_coefficients_only(self):
        stage,donor,bank = make_fixture('S020BASE')
        for case in ('S020H050','S020H200','S020LR150','S020LR150H050'):
            C.validate_stats_bank(bank,donor.binding,make_spec(case),full_hash=True)
        with self.assertRaises(ValueError):
            C.validate_stats_bank(bank,donor.binding,make_spec('S010BASE'))
        bank['s020']._q[0].add_(1.)
        with self.assertRaisesRegex(ValueError,'immutable calibration'):
            stage._assert_contract()

    def test_real_calibration_save_load_and_wrong_geometry_rejection(self):
        spec,donor,inventory = make_spec('S020BASE'),make_donor(),SyntheticInventory()
        donor.binding.update(sample_count=3,sample_ids_sha256=canonical_sha256(list(inventory.sample_ids)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'s020.npz'
            with patch.object(C,'code_pin',return_value={'synthetic':'fixed'}):
                record = C.calibrate_geometry(donor,inventory,spec=spec,geometry_key='s020',output=path,
                    k=5,batch_size=2,memory_budget_bytes=8*1024**2)
            self.assertEqual(record['sample_count'],3)
            bank = C.load_stats_bank(donor,make_spec('S020H050'),{'s020':record},memory_budget_bytes=8*1024**2)
            self.assertEqual(set(bank),{'s020'})
            with self.assertRaisesRegex(ValueError,'binding mismatch'):
                C.load_stats_bank(donor,make_spec('S010BASE'),{'s010':record},memory_budget_bytes=8*1024**2)

    def test_engine_rejects_registry_drift_before_training(self):
        cfg = plan.resolve_case('s3','S020LR150H050')
        cfg['stage2_spec']['seed'] = cfg['stream_seeds']['S_init'] = 345
        _,donor,bank = make_fixture('S020LR150H050')
        cfg['donor_binding'] = deepcopy(donor.binding)
        trainer = build_fresh_stage2(cfg,donor,bank)
        self.assertEqual(trainer.training.lr,1.5e-4)
        for field,value in (('shift_scale',0.),('student_views',2),
                            ('rotation_consistency_weight',.05),('lr_U',1e-4)):
            changed = deepcopy(cfg)
            changed[field] = value
            with self.assertRaises(ValueError):
                build_fresh_stage2(changed,donor,bank)


if __name__ == '__main__':
    unittest.main()
