#!/usr/bin/env python3
"""CPU-only FH12 evaluator/profile contracts, without production artifacts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from fh12.evaluation import rr_metrics, infer
from fh12.model import build_model, state_hash
from fh12.postrun import profile_model


class SampleData(torch.utils.data.Dataset):
    has_gt=True
    def __len__(self): return 1
    def __getitem__(self,index):
        gen=torch.Generator().manual_seed(14)
        gt=torch.rand((8,256,256),generator=gen)*2-1
        ms=torch.rand((8,64,64),generator=gen)*2-1
        lp=torch.rand((1,64,64),generator=gen)*2-1
        pan=torch.rand((1,256,256),generator=gen)*2-1
        return gt,gt.clone(),ms,lp,pan,torch.tensor([0,0,0,0])


class EvaluationTests(unittest.TestCase):
    def test_profile_covers_full_model_and_preserves_state(self):
        model,_=build_model('PLH',8,[1,1,1],71001,role='T')
        before=state_hash(model.state_dict())
        cost=profile_model(model,SampleData(),'cpu')
        self.assertEqual(before,state_hash(model.state_dict()))
        self.assertEqual(cost['params_m'],sum(p.numel() for p in model.parameters())/1e6)
        self.assertGreater(cost['frontend_mac_equivalent_g'],0)
        self.assertAlmostEqual(cost['flops_g'],cost['backbone_aligner_thop_macs_g']+cost['frontend_mac_equivalent_g'])
        self.assertGreater(cost['infer_ms'],0)
        self.assertIsNone(cost['mem_mb'])
        self.assertTrue(cost['frontend_included'])
        self.assertTrue(cost['flops_is_estimate'])

    def test_inference_restores_train_mode_and_never_feeds_gt(self):
        class Model(torch.nn.Module):
            def forward(self,pan,ms,lp):
                self.assert_shapes=(pan.shape,ms.shape,lp.shape)
                return {'y':pan.expand(-1,8,-1,-1)*3,'delta':torch.zeros(len(pan),2)}
        model=Model().train()
        sr,delta=infer(model,SampleData(),'cpu')
        self.assertTrue(model.training)
        self.assertEqual(sr.shape,(1,8,256,256))
        self.assertEqual(delta.shape,(1,2))
        self.assertGreaterEqual(sr.min(),0)
        self.assertLessEqual(sr.max(),2047)
        self.assertTrue(np.any(sr!=np.round(sr)))

    def test_rr_rejects_19_scene_subset(self):
        data=np.empty((19,8,256,256),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'20x8x256x256'):
            rr_metrics(data,data)

    def test_rr_rejects_nonfinite_discarded_border(self):
        data=np.ones((20,8,256,256),dtype=np.float32)
        data[0,0,0,0]=np.nan
        with self.assertRaisesRegex(FloatingPointError,'discarded borders'):
            rr_metrics(data,data)

    def test_rr_has_exact_crop_order_and_all20_scene_aggregation(self):
        gt=np.broadcast_to(np.arange(256,dtype=np.float32)[None,None,None,:]+100,
                           (20,8,256,256)).copy()
        sr=gt+1
        seen=[]
        def q(ref,pred,block,shift):
            seen.append((ref.shape,float(ref[0,0,0]),float(ref[-1,-1,0]),block,shift))
            self.assertTrue(np.all(pred-ref==1))
            return .9,None
        with patch('fh12.evaluation.q2n',side_effect=q), \
             patch('fh12.evaluation.sam',return_value=2.), \
             patch('fh12.evaluation.ergas',return_value=2.1), \
             patch('fh12.evaluation.scc_dlpan',return_value=.98), \
             patch('fh12.evaluation.psnr_global',return_value=40.), \
             patch('fh12.evaluation.ssim_skimage',return_value=.97):
            result=rr_metrics(sr,gt)
        self.assertEqual(seen,[((215,215,8),120.,334.,32,32)]*20)
        self.assertEqual(result['n_scenes'],20)
        self.assertTrue(result['official_complete'])
        self.assertAlmostEqual(result['q8'],.9)
        self.assertEqual(len(result['per_scene']),20)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
