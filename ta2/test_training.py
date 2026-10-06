"""CPU resume, paired streams and complete-batch optimizer contracts."""
import copy
import unittest

import torch
from torch import nn

from ta2.common import digest
from ta2.model import TeacherModel, state_hash
from ta2.plan import Plan, resolved_config
from ta2.test_model_loss import sample
from ta2.training import UpdateEngine, configure_runtime, cosine_factor, objective_inputs, preserve_runtime


class TinyU(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv=nn.Conv2d(bands+1,bands,1)
    def forward(self,pan,lpan,ms,mode,x_in=None):
        return self.conv(x_in)


class Bundle:
    def __init__(self,bands=8):
        self.values=sample(bands=bands,count=96)
        self.datasets={'train':list(range(96))}
    def batch(self,split,ids,rots=None,device='cpu'):
        return {k:torch.stack([torch.rot90(v[i].flip((-1,-2)),r,(-2,-1)) if rots is not None else v[i]
                for i,r in zip(ids,rots if rots is not None else [0]*len(ids))]).to(device)
                for k,v in self.values.items()}
    def case_shifts(self,cfg,split):
        return torch.tensor([.2,-.4]).expand(96,-1)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.plan=Plan()
    def engine(self,case='TA2-M07',micro=12):
        row=next(r for r in self.plan.queue('s1') if r['case_id']==case)
        cfg=resolved_config(row,micro)
        configure_runtime(cfg['seed'],'cpu')
        model=TeacherModel(8,cfg['seed'],cfg)
        model.backbone=TinyU(8)
        return UpdateEngine(model,Bundle(),cfg,'cpu')
    def test_exact_cpu_resume_and_isolated_streams(self):
        uninterrupted=self.engine()
        first=uninterrupted.update()
        snapshot=uninterrupted.state_dict()
        expected=uninterrupted.update()
        expected_hash=state_hash(uninterrupted.model)
        resumed=self.engine(); resumed.load_state_dict(snapshot)
        actual=resumed.update()
        self.assertEqual(expected_hash,state_hash(resumed.model))
        for key in ('sample_ids','rotations','rec','epsilon','struct','learning_rates'):
            self.assertEqual(expected[key],actual[key])
        self.assertEqual(first['learning_rates']['A'],0.)
        self.assertEqual(first['actual_parameter_update']['A'],0.)
        no_a=self.engine('TA2-B00')
        self.assertEqual(no_a.update()['sample_ids'],first['sample_ids'])
        self.assertEqual(no_a.update()['rotations'],expected['rotations'])
    def test_microbatch_preserves_mean_update(self):
        a=self.engine(micro=12); b=self.engine(micro=48)
        for _ in range(2):
            x=a.update(); y=b.update()
            for key in ('rec','epsilon','struct'):
                self.assertAlmostEqual(x[key],y[key],places=6)
        for p,q in zip(a.model.parameters(),b.model.parameters()):
            self.assertTrue(torch.allclose(p,q,atol=2e-7,rtol=1e-5))
    def test_b07_canonical_transform_keeps_gradient(self):
        engine=self.engine('TA2-B07')
        engine.model.global_shift.data[:]=torch.tensor([.2,-.3])
        inputs=objective_inputs(engine.model,engine.bundle,engine.cfg,[0,1],[0,1],'cpu')
        self.assertTrue(inputs['fixed_shift'].requires_grad)
        inputs['fixed_shift'][0].sum().backward()
        self.assertTrue(torch.equal(engine.model.global_shift.grad,torch.tensor([-1.,-1.])))
    def test_preserve_runtime_restores_grad_modes_rng(self):
        e=self.engine()
        e.model.train(); e.model.aligner.eval()
        param=next(e.model.parameters()); param.grad=torch.ones_like(param)
        state=torch.get_rng_state().clone()
        with preserve_runtime(e.model):
            e.model.eval(); param.grad=None; torch.rand(5)
        self.assertTrue(e.model.training); self.assertFalse(e.model.aligner.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(param.grad,torch.ones_like(param)))
    def test_schedule_and_config_resume_guard(self):
        self.assertEqual(cosine_factor(0,50000),0)
        self.assertEqual(cosine_factor(100,50000),1)
        self.assertEqual(cosine_factor(50000,50000),0)
        self.assertNotEqual(cosine_factor(50000,100000),0)
        a=self.engine(micro=12); b=self.engine(micro=24)
        with self.assertRaisesRegex(ValueError,'config'):
            b.load_state_dict(a.state_dict())
    def test_plan_counts_and_no_s2_s4(self):
        self.assertEqual(len(self.plan.runs),468)
        self.assertEqual(sum(r['updates'] for r in self.plan.runs),25200000)
        for server in ('s2','s4'):
            with self.assertRaises(ValueError):self.plan.queue(server)


if __name__=='__main__':unittest.main()
