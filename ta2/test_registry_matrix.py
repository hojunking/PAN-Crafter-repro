"""Real52-case resolution acrossall3 sensors, independently of launch smoke."""
import unittest

import torch

from ta2.losses import compute_objective
from ta2.model import TeacherModel
from ta2.plan import Plan,resolved_config
from ta2.test_model_loss import sample


class RegistryMatrixTests(unittest.TestCase):
    def test_all_156_registered_objectives_forward_backward(self):
        torch.set_num_threads(1)
        plan=Plan(); count=0
        for server in ('s1','s3','s5'):
            for row in (r for r in plan.queue(server) if r['replica']==1):
                with self.subTest(server=server,case=row['case_id']):
                    cfg=resolved_config(row); model=TeacherModel(cfg['bands'],cfg['seed'],cfg)
                    inputs=sample(bands=cfg['bands'])
                    extras={}
                    if model.policy not in ('BYPASS','ZERO','LEARNED','GLOBAL_LEARNED'):
                        extras['fixed_shift']=torch.tensor([[.1,-.2]])
                    if cfg['band_weights']=='CAL_FROZEN':extras['band_weights']=torch.ones(2,cfg['bands'])
                    if cfg['struct_descriptor']=='PSEUDO_HUBER':extras['pseudo_shift']=torch.tensor([[.1,-.2]])
                    output=compute_objective(model,inputs,cfg,2,epsilon=torch.tensor([[.25,-.5]]),
                        auxiliary_choice=56 if cfg['auxiliary']=='CROP' else 2,**extras)
                    output['total'].backward()
                    self.assertTrue(torch.isfinite(output['total']))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    count+=1
        self.assertEqual(count,156)


if __name__=='__main__':unittest.main()
