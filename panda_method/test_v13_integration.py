"""Value-level v1.3 contracts across geometry, training and frozen deployment."""
import copy
from dataclasses import replace
import unittest

import torch

from . import METHOD_VERSION
from .calibration import canonical_sha256, state_dict_sha256
from .data import augment_batch, bind_scene_priors
from .stages import Stage2, calibration_binding
from .test_data import fixtures, lineage, observations
from .test_stages import batch, calibrated, recon, stage1


class V13IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rr_fr_native_and_epsilon_heads_receive_identical_raw_branch_prior(self):
        model=stage1()
        rr=replace(batch(),prior_dy_dx=torch.tensor([[1.,0.],[3.,0.]],requires_grad=True))
        fr=replace(batch('FR',side=24),prior_dy_dx=torch.tensor([[4.,.5],[-2.,6.]],requires_grad=True))
        inputs=[]
        h=model.aligner.fc1.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().clone()))
        out=model(rr,fr); h.remove()
        self.assertEqual(len(inputs),4)
        for i,expected in enumerate((rr.prior_dy_dx,rr.prior_dy_dx,fr.prior_dy_dx,fr.prior_dy_dx)):
            torch.testing.assert_close(inputs[i][:,-2:],expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(inputs[0][1,-2:],torch.tensor([1.,0.])))
        torch.testing.assert_close(out['native']['correction'],rr.prior_dy_dx,rtol=0,atol=0)
        torch.testing.assert_close(out['c0_fr'],fr.prior_dy_dx,rtol=0,atol=0)
        out['loss'].backward()
        self.assertIsNone(rr.prior_dy_dx.grad)
        self.assertIsNone(fr.prior_dy_dx.grad)

    def test_verified_grid_phase_and_image_augmentation_reach_same_head_and_output(self):
        binding,_=fixtures()
        b=bind_scene_priors(**observations(dtype=torch.float32),lineage=lineage(binding),binding=binding)
        b=augment_batch(b,hflip=True,vflip=False,rot90=1)
        a=stage1().aligner
        inputs=[]
        hook=a.fc1.register_forward_pre_hook(lambda m,args:inputs.append(args[0].detach().clone()))
        correction,details=a(b.pan,b.native_lms,b.prior_dy_dx,return_details=True)
        hook.remove()
        torch.testing.assert_close(inputs[0][:,-2:],b.prior_dy_dx,rtol=0,atol=0)
        torch.testing.assert_close(correction,b.prior_dy_dx,rtol=0,atol=0)
        torch.testing.assert_close(details['prior'],b.prior_dy_dx,rtol=0,atol=0)

    def test_native_diagnostics_reuse_the_same_aligner_forward(self):
        model=stage1(); counts=[]
        hook=model.aligner.register_forward_hook(lambda *_:counts.append(1))
        detailed=model.native_prediction(batch(),return_aligner_details=True)
        hook.remove()
        self.assertEqual(len(counts),1)
        self.assertIn('aligner_details',detailed)
        plain=model.native_prediction(batch())
        self.assertNotIn('aligner_details',plain)
        for key in plain:
            if isinstance(plain[key],torch.Tensor):
                torch.testing.assert_close(plain[key],detailed[key],rtol=0,atol=0)

    def test_v12_stage_checkpoint_is_rejected_before_parameter_mutation(self):
        model=stage1(); old=copy.deepcopy(model.state_dict())
        old['_extra_state']['method']='PANDA_MethodPlan_v1.2_2026-10-13'
        old['_extra_state']['pipeline']['method']='PANDA_MethodPlan_v1.2_2026-10-13'
        old['_extra_state']['pipeline'].pop('aligner_conditioning')
        old.pop('aligner._conditioning_revision')
        old['aligner.fc1.weight'].fill_(17)
        before=state_dict_sha256(model.aligner)
        with self.assertRaisesRegex(ValueError,'method'):
            model.load_state_dict(old)
        self.assertEqual(before,state_dict_sha256(model.aligner))
        self.assertEqual(model.get_extra_state()['method'],METHOD_VERSION)

    def test_v12_calibration_cannot_cross_the_semantic_boundary_even_with_zero_head(self):
        model,rr,scales=calibrated()
        current=calibration_binding(model)
        old_state={k:v for k,v in model.aligner.state_dict().items() if k!='_conditioning_revision'}
        old_pipeline=copy.deepcopy(model.pipeline_manifest())
        old_pipeline.pop('aligner_conditioning')
        old_pipeline['method']='PANDA_MethodPlan_v1.2_2026-10-13'
        stale=replace(scales,binding=replace(current,aligner_sha256=state_dict_sha256(old_state),
                                            pipeline_sha256=canonical_sha256(old_pipeline)))
        # c=b numerically at fresh initialization does NOT imply method equivalence.
        with torch.no_grad():
            torch.testing.assert_close(model.native_prediction(rr)['correction'],rr.prior_dy_dx,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'stale'):
            Stage2(model,recon(role='S'),stale)

    def test_stage2_and_student_deployment_keep_raw_signed_conditioning(self):
        model,rr,scales=calibrated()
        second=Stage2(model,recon(role='S'),scales)
        seen=[]
        h=second.aligner.fc1.register_forward_pre_hook(lambda m,a:seen.append(a[0][:,-2:].detach().clone()))
        second(rr)
        fr=batch('FR',split='test',side=24)
        second.inference_model()(fr)
        h.remove()
        torch.testing.assert_close(seen[0],rr.prior_dy_dx,rtol=0,atol=0)
        torch.testing.assert_close(seen[1],fr.prior_dy_dx,rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in second.aligner.parameters()))


if __name__=='__main__':
    unittest.main()
