from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from panda_wv3_ablation import evaluation,plan


class EvaluationTests(unittest.TestCase):
    def test_same_state_complete_original_references_no_fr_gt(self):
        torch.set_num_threads(1)
        calls=[]
        class Sources:
            sensor='WV3';peak=2047;bands=8
            def batch(self,split,ids,device=None):
                h=256 if split=='rr' else 512
                return SimpleNamespace(pan=torch.zeros(1,1,h,h),native_lms=torch.zeros(1,8,h,h),
                    lineage=SimpleNamespace(sample_ids=(f'WV3:{split}:{ids[0]}',)),gt=None)
            def raw(self,split,ids):
                calls.append((split,ids[0]));a={'pan':np.zeros((1,1)),'lms':np.zeros((1,1)),'ms':np.zeros((1,1))}
                if split=='rr':a['gt']=np.zeros((1,1))
                return a
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.aligner=torch.nn.Linear(1,1);self.student=torch.nn.Linear(1,1)
            def native_prediction(self,b):return {'y':b.native_lms,'u':torch.zeros(1,2)}
        cfg=plan.queue('s1')[0];m=Model();m.train()
        trainer=SimpleNamespace(cfg=cfg,model=m,updates=1000,audit=lambda:None)
        def rr(raw,arrays):
            self.assertIn('gt',arrays)
            return {k:1. for k in ('ergas','scc','sam','psnr','ssim','q8','rmse','cc')}
        def fr(raw,arrays):
            self.assertNotIn('gt',arrays)
            return {k:.9 for k in ('hqnr','d_s','d_lambda','jqm')}
        with patch('panda_wv3_ablation.evaluation.evaluator_identity',return_value={'synthetic':True}), \
             patch('panda_wv3_ablation.evaluation.metric_functions_for',return_value={'rr':rr,'fr':fr}):
            record=evaluation.evaluate(trainer,Sources(),{'synthetic':True})
        evaluation.verify_record(record,cfg)
        self.assertEqual(calls,[(s,i) for s in ('rr','fr') for i in range(20)])
        self.assertTrue(m.training);self.assertTrue(record['same_state_verified'])
