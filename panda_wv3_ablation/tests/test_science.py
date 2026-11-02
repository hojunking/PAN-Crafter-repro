from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from directshift48.method import ScharrAbsolute,DirectTrainingSources
from donor30.artifacts import _same
from donor30.masking import masked_mean
from panda_band30.stage1 import BandStage1
from panda_band30.tests.test_method import make_donor
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps,compute_loss
from panda_stage2_condq.spec import Stage2Spec,MASK2_POLICY
from panda_tune30.calibration import bind_donor,binding_for_geometry
from panda_tune30.method import frozen_predictions
from panda_method.calibration import state_dict_sha256
from pa.warp import warp_pan
from panda_wv3_ablation import plan,assets,model,training


def fixture():
    spec=assets.anchor_spec();donor=bind_donor(make_donor(),spec);rr=batch()
    f=frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key='s100')
    e,g,q=reference_maps(ScharrAbsolute('WV3'),f['teacher_y'],rr.gt,f['pan_loss'])
    builder=CalibrationBuilder(['WV3:train:0'],binding_for_geometry(donor.binding,'s100'),
        k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=MASK2_POLICY)
    builder.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=f['loss_mask'])
    return donor,builder.finalize(),rr,f


class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_stage2_cases_execute_real_forward_backward_with_frozen_donor(self):
        donor,stats,rr,f=fixture()
        sha_a=state_dict_sha256(donor.aligner);sha_t=state_dict_sha256(donor.teacher)
        for cfg in plan.queue('s1')[:9]:
            with self.subTest(case=cfg['case_id']):
                m=model.Stage2(cfg,donor,stats)
                with patch.object(m.teacher.backbone,'forward',wraps=m.teacher.backbone.forward) as t:
                    terms=m(rr);terms['loss'].backward()
                no_t=cfg['case_id'] in ('S2_B0_CORR_GT','S2_B1_DIFF_GT','S2_D2_NO_E_CUE')
                self.assertEqual(t.call_count,0 if no_t else 1)
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.student.parameters()))
                self.assertTrue(all(p.grad is None for p in donor.aligner.parameters()))
                self.assertTrue(all(p.grad is None for p in donor.teacher.parameters()))
                self.assertEqual(state_dict_sha256(donor.aligner),sha_a)
                self.assertEqual(state_dict_sha256(donor.teacher),sha_t)
                self.assertEqual(float(terms['valid_fraction']),float(f['loss_mask'].float().mean()))
                if not cfg['alpha'] and not cfg['gamma']:
                    torch.testing.assert_close(terms['loss'],terms['rec'],atol=0,rtol=0)

    def test_existing_loss_and_gradients_match_for_single_cues_and_full_control(self):
        donor,stats,rr,f=fixture()
        for key,alpha,gamma in [('S2_B2_E_ONLY',.25,0),('S2_B3_EQ_ONLY',0,.25),('S2_D4_TWO_PANS',.25,.25)]:
            c=plan.resolve('s1',key,1)
            # Synthetic full control is NOT a registered production job.
            c['student_input']='CORRECTED_DIFF'
            m=model.Stage2(c,donor,stats)
            with torch.no_grad():m.student.backbone.output[-1].weight.normal_(0,.001)
            terms=m(rr)
            y=m.reconstruct(rr,f['pan_input'],f['ms_base_frame'])['y']
            spec=Stage2Spec('WV3','panda_e' if gamma==0 else 'panda_condq',1,alpha,0,gamma,1e-6,mask_policy=MASK2_POLICY)
            old=compute_loss(y,f['teacher_y'],rr.gt,f['pan_loss'],spec=spec,
                             scharr=ScharrAbsolute('WV3'),stats=stats,loss_mask=f['loss_mask'])
            torch.testing.assert_close(terms['loss'],old['loss'],atol=0,rtol=0)
            p=m.student.backbone.input.weight
            g1=torch.autograd.grad(terms['loss'],p)[0];g2=torch.autograd.grad(old['loss'],p)[0]
            torch.testing.assert_close(g1,g2,atol=0,rtol=0)

    def test_global_rank_matches_pooled_ecdf_and_mean_weight_preserves_gain(self):
        donor,stats,rr,f=fixture()
        e,g,q=reference_maps(ScharrAbsolute('WV3'),f['teacher_y'],rr.gt,f['pan_loss'])
        pooled=torch.cat(stats._q).sort().values
        expected=torch.searchsorted(pooled,q.contiguous(),right=True).double().div(len(pooled)).float()
        torch.testing.assert_close(model.global_rank(stats,q),expected,atol=0,rtol=0)
        cfg=plan.resolve('s1','S2_D4_TWO_PANS',1)
        full=model.adaptive_weight(e,g,q,stats,f['loss_mask'],cfg)
        cfg=plan.resolve('s1','S2_D3_MEAN_WEIGHT',1)
        mean=model.adaptive_weight(e,g,q,stats,f['loss_mask'],cfg)
        torch.testing.assert_close(masked_mean(full,f['loss_mask']),masked_mean(mean,f['loss_mask']))
        self.assertEqual(mean.unique().numel(),1)
        self.assertFalse(mean.requires_grad)

    def test_input_initialization_shared_and_two_pan_reparameterization(self):
        donor,stats,rr,f=fixture()
        models=[model.Stage2(plan.resolve('s1',key,1),donor,stats) for key in
                ('S2_B0_CORR_GT','S2_B1_DIFF_GT','S2_D4_TWO_PANS')]
        corr,diff,two=models
        self.assertEqual(sum(p.numel() for p in diff.student.parameters())-sum(p.numel() for p in corr.student.parameters()),1008)
        self.assertTrue(torch.equal(diff.student.backbone.input.weight[:,1],torch.zeros_like(diff.student.backbone.input.weight[:,1])))
        for m in models[1:]:
            torch.testing.assert_close(m.native_prediction(rr)['y'],corr.native_prediction(rr)['y'],atol=0,rtol=0)
        with torch.no_grad():
            diff.student.backbone.input.weight[:,1:2].normal_(0,.02)
            diff.student.backbone.output[-1].weight.normal_(0,.002)
            two.student.load_state_dict(diff.student.state_dict())
            w=diff.student.backbone.input.weight
            two.student.backbone.input.weight[:,0:1].copy_(w[:,0:1]+w[:,1:2])
            two.student.backbone.input.weight[:,1:2].copy_(-w[:,1:2])
        torch.testing.assert_close(two.native_prediction(rr)['y'],diff.native_prediction(rr)['y'],atol=2e-6,rtol=2e-6)

    def test_stage1_loss_removal_matches_existing_band30_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered=synthetic_sources(Path(tmp));rr=registered.batch([0])
            g=torch.Generator().manual_seed(42)
            rr=replace(rr,**{k:torch.rand(getattr(rr,k).shape,generator=g)*1.6-.8 for k in ('pan','ms','native_lms','gt')})
            for key in ('S1_A2_REC','S1_A3_REC_SPA','S1_A4_REC_EPS'):
                c=plan.resolve('s3',key,1);m=model.Stage1(c,registered.inventory.manifest())
                with torch.no_grad():m.teacher.backbone.output[-1].weight.normal_(0,.001)
                original=BandStage1(deepcopy(m.aligner),deepcopy(m.teacher),upsampler='interp23',
                    sources=DirectTrainingSources('WV3',registered.sources.manifest_sha('train')),
                    epsilon_seed=c['stream_seeds']['rr_epsilon'],band_augmentation_seed=c['stream_seeds']['rr_band_augmentation'],
                    lambda_abs=2.,lambda_eps=.003)
                with patch.object(m.aligner,'forward',wraps=m.aligner.forward) as calls:
                    out=m(rr)
                old=original(rr)
                old_loss=old['rec']+c['lambda_abs']*old['abs_rr']+c['lambda_epsilon']*old['eps_rr']
                self.assertEqual(calls.call_count,2 if c['lambda_epsilon'] else 1)
                torch.testing.assert_close(out['loss'],old_loss,atol=0,rtol=0)
                out['loss'].backward();old_loss.backward()
                torch.testing.assert_close(m.aligner.fc2.weight.grad,original.aligner.fc2.weight.grad,atol=0,rtol=0)

    def test_scharr_integer_fractional_sign_and_constant_ties(self):
        from panda_wv3_ablation.matching import ScharrAlignment
        match=ScharrAlignment();torch.manual_seed(91)
        pan=torch.nn.functional.avg_pool2d(torch.rand(2,1,68,68),5,stride=1)
        shifts=torch.tensor([[1.,-2.],[-.5,.75]])
        target=warp_pan(pan,shifts).expand(-1,8,-1,-1)
        actual=match(pan,target)
        torch.testing.assert_close(actual,shifts,atol=0,rtol=0)
        zero=match(torch.zeros_like(pan),torch.zeros_like(target))
        torch.testing.assert_close(zero,torch.zeros_like(shifts),atol=0,rtol=0)
        self.assertFalse(actual.requires_grad)
        # Complete support, including every searched corner; +6 would violate it.
        for y in (-5.,5.):
            for x in (-5.,5.):model.support(pan,pan.new_tensor([[y,x]]).expand(len(pan),2))
        with self.assertRaisesRegex(ValueError,'SUPPORT_VIOLATION'):
            model.support(pan,pan.new_tensor([[6.,6.]]).expand(len(pan),2))

    def test_complete_resume_replays_optimizer_and_data_and_epsilon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_sources(root,count=3)
            c=plan.resolve('s3','S1_A4_REC_EPS',1)
            # Reduced batch ONLY in this CPU fixture; production validate rejects it.
            c.update(batch=1,microbatch=1)
            def build():
                with patch('panda_wv3_ablation.plan.validate'):
                    t=training.Trainer(c,model.Stage1(c,registered.inventory.manifest()))
                return t,training.stream_for(c,registered,'cpu')
            t,s=build();t.step(s.next_update());training.save(root/'resume.pt',t,s,{'test':True})
            expected=t.step(s.next_update());weights=deepcopy(t.model.state_dict());opt=deepcopy(t.optimizer.state_dict());stream=s.state_dict()
            t2,s2=build();training.load(root/'resume.pt',t2,s2,{'test':True});got=t2.step(s2.next_update())
            self.assertEqual(expected,got);self.assertTrue(_same(weights,t2.model.state_dict()))
            self.assertTrue(_same(opt,t2.optimizer.state_dict()));self.assertTrue(_same(stream,s2.state_dict()))
            with self.assertRaisesRegex(ValueError,'Resume'):
                training.load(root/'resume.pt',t2,s2,{'test':False})

    def test_student_resume_keeps_calibration_frozen_weights_and_optimizer(self):
        donor,stats,rr,_=fixture();c=plan.resolve('s1','S2_B2_E_ONLY',1);c.update(batch=1,microbatch=1)
        class Stream:
            index=0
            def next_update(self):self.index+=1;return [rr]
            def state_dict(self):return {'index':self.index}
            def load_state_dict(self,s):self.index=s['index']
        def build():
            with patch('panda_wv3_ablation.plan.validate'):
                return training.Trainer(c,model.Stage2(c,donor,stats)),Stream()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'student.pt';t,s=build();t.step(s.next_update())
            training.save(path,t,s,{'test':True});expected=t.step(s.next_update())
            weights=deepcopy(t.model.state_dict());opt=deepcopy(t.optimizer.state_dict())
            t2,s2=build();training.load(path,t2,s2,{'test':True});actual=t2.step(s2.next_update())
            self.assertEqual(actual,expected);self.assertTrue(_same(weights,t2.model.state_dict()))
            self.assertTrue(_same(opt,t2.optimizer.state_dict()));self.assertEqual(s.index,s2.index)
            t2.audit()
