"""Scientific equivalence, zero endpoints and complete restart contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from donor30.artifacts import _same
from donor30.masking import masked_mean
from directshift48.method import ScharrAbsolute
from panda_band30.tests.test_method import make_donor
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.spec import MASK2_POLICY
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_stage2_condq.tests.test_method import batch
from panda_tune30.calibration import bind_donor, binding_for_geometry
from panda_tune30.method import frozen_predictions
from panda_wv3_ablation import assets as anchored_assets
from panda_wv3_ablation_rgb import plan as rgb_plan
from panda_wv3_main_rgb import model as original_model
from panda_wv3_main_rgb import plan as original_plan
from panda_wv3_hyper import model, plan, training


def fixture():
    donor=bind_donor(make_donor(),anchored_assets.anchor_spec())
    donor.binding['alignment_target']='SCHARR_OF_RGB_MEAN_GT'
    donor.binding=binding_for_geometry(donor.binding,'s100')
    donor.selection_receipt={'synthetic':True,'uses_test_for_selection':False}
    rr=batch()
    frozen=frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key='s100')
    e,g,q=reference_maps(ScharrAbsolute('WV3'),frozen['teacher_y'],rr.gt,frozen['pan_loss'])
    builder=CalibrationBuilder(['WV3:train:0'],donor.binding,
        k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=MASK2_POLICY)
    builder.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=frozen['loss_mask'])
    return donor,builder.finalize(),rr,frozen,(e,g,q)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_round_major_counts_and_paired_recipe_invariants(self):
        for server,count,stage in (('s1',30,'stage2'),('s3',33,'stage1')):
            queue=plan.queue(server)
            self.assertEqual(len(queue),count)
            self.assertEqual(len({c['run_id'] for c in queue}),count)
            self.assertEqual([c['repeat'] for c in queue],sum(([r]*(count//3) for r in (1,2,3)),[]))
            for repeat in (1,2,3):
                block=[c for c in queue if c['repeat']==repeat]
                self.assertEqual(len({plan.digest(c['stream_seeds']) for c in block}),1)
                for cfg in block:
                    plan.validate(cfg)
                    self.assertEqual(cfg['stage'],stage)
                    self.assertEqual(cfg['scale'],1.)
                    self.assertEqual(cfg['beta'],0.)
                    self.assertEqual(cfg['teacher_width'],112)
                    self.assertFalse(cfg['weight_averaging'])
                    self.assertFalse(cfg['inference_tta'])
                    self.assertFalse(cfg['rotation_consistency'])
                    self.assertEqual(cfg['primary_step'],29000 if server=='s1' else 50000)
                    old=(original_plan.resolve('s1','M0_OURS',repeat) if server=='s1'
                         else rgb_plan.resolve('s3','S1_A5_ALL',repeat))
                    for key in ('stream_seeds','lr','lr_aligner','warp','batch','microbatch',
                                'epsilon_radius','augmentation','horizon'):
                        self.assertEqual(cfg[key],old[key])
            self.assertEqual(sum(bool(c['existing_control_case']) for c in queue),9)
        with self.assertRaises(ValueError):plan.resolve('s1','H1_REF',1)
        altered=plan.queue('s1')[0];altered['alpha']=.123
        with self.assertRaises(ValueError):plan.validate(altered)

    def test_positive_path_is_identical_to_original_main_loss_and_gradient(self):
        donor,stats,rr,_,_=fixture()
        cfg=plan.resolve('s1','H2_REF',1)
        original=original_model.Stage2(cfg,donor,stats)
        new=model.Stage2(cfg,donor,stats)
        self.assertTrue(_same(original.student.state_dict(),new.student.state_dict()))
        with torch.no_grad():original.student.backbone.output[-1].weight.normal_(0,.002)
        new.student.load_state_dict(original.student.state_dict())
        oldterms=original(rr);terms=new(rr)
        self.assertEqual(set(terms),set(oldterms))
        for key in terms:torch.testing.assert_close(terms[key],oldterms[key],atol=0,rtol=0)
        oldterms['loss'].backward();terms['loss'].backward()
        for p,q in zip(original.student.parameters(),new.student.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_zero_endpoints_keep_support_and_literal_weight_equation(self):
        donor,stats,rr,frozen,(e,g,q)=fixture()
        before={role:state_dict_sha256(getattr(donor,role)) for role in ('aligner','teacher')}
        for case in ('H2_A0','H2_G0','H2_GT','H2_A100','H2_G100'):
            cfg=plan.resolve('s1',case,1);net=model.Stage2(cfg,donor,stats)
            with patch.object(net.teacher.backbone,'forward',wraps=net.teacher.backbone.forward) as t:
                terms=net(rr)
            self.assertEqual(t.call_count,0 if case=='H2_GT' else 1)
            self.assertEqual(float(terms['valid_fraction']),float(frozen['loss_mask'].float().mean()))
            rank=torch.zeros_like(e);roi=(...,slice(8,-8),slice(8,-8))
            rank[roi]=stats.lookup_exact(g[roi],q[roi])
            cue=e/(e+stats.tau_e).clamp_min(torch.finfo(torch.float32).tiny)
            expected=1+cfg['alpha']*cue+cfg['gamma']*cue*rank
            y=net.reconstruct(rr,frozen['pan_input'],frozen['ms_base_frame'])['y']
            error=(y-rr.gt).abs().mean(1,keepdim=True)
            torch.testing.assert_close(terms['loss'],masked_mean(expected*error,frozen['loss_mask']),atol=0,rtol=0)
            terms['loss'].backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in net.student.parameters()))
            if case=='H2_GT':
                torch.testing.assert_close(terms['loss'],terms['rec'],atol=0,rtol=0)
                self.assertEqual(float(terms['weight_mean']),1.)
            donor.assert_frozen()
            self.assertEqual(before,{r:state_dict_sha256(getattr(donor,r)) for r in before})

    def test_stage1_complete_resume_replays_optimizer_data_and_epsilon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_sources(root,count=3)
            cfg=plan.resolve('s3','H1_E010',1);cfg.update(batch=1,microbatch=1)
            def build():
                with patch('panda_wv3_hyper.plan.validate'):
                    trainer=training.Trainer(cfg,model.Stage1(cfg,registered.inventory.manifest()))
                return trainer,training.stream_for(cfg,registered,'cpu')
            t,s=build();t.step(s.next_update());training.save(root/'resume.pt',t,s,{'test':True})
            expected=t.step(s.next_update());weights=deepcopy(t.model.state_dict());opt=deepcopy(t.optimizer.state_dict())
            t2,s2=build();training.load(root/'resume.pt',t2,s2,{'test':True})
            self.assertEqual(expected,t2.step(s2.next_update()))
            self.assertTrue(_same(weights,t2.model.state_dict()))
            self.assertTrue(_same(opt,t2.optimizer.state_dict()))
            self.assertTrue(_same(s.state_dict(),s2.state_dict()))

    def test_stage2_complete_resume_retains_frozen_donor_and_calibration(self):
        donor,stats,rr,_,_=fixture();cfg=plan.resolve('s1','H2_GT',1);cfg.update(batch=1,microbatch=1)
        class Stream:
            index=0
            def next_update(self):self.index+=1;return [rr]
            def state_dict(self):return {'index':self.index}
            def load_state_dict(self,s):self.index=s['index']
        def build():
            with patch('panda_wv3_hyper.plan.validate'):
                return training.Trainer(cfg,model.Stage2(cfg,donor,stats)),Stream()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt';t,s=build();t.step(s.next_update())
            training.save(path,t,s,{'test':True});expected=t.step(s.next_update())
            weights=deepcopy(t.model.state_dict());opt=deepcopy(t.optimizer.state_dict())
            t2,s2=build();training.load(path,t2,s2,{'test':True})
            self.assertEqual(expected,t2.step(s2.next_update()))
            self.assertTrue(_same(weights,t2.model.state_dict()))
            self.assertTrue(_same(opt,t2.optimizer.state_dict()))
            self.assertEqual(s.index,s2.index);t2.audit()


if __name__=='__main__':unittest.main()
