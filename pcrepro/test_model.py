import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from pcrepro.model import build_model,Eq11Attention,parameter_counts


class ModelTests(unittest.TestCase):
    def test_fixed_gather_matches_exact_neighbours(self):
        x=torch.arange(2*3*5*6,dtype=torch.float32).reshape(2,3,5,6)
        actual=Eq11Attention.gather(x)
        padded=F.pad(x,(1,1,1,1))
        expected=torch.stack([padded[:,:,y:y+5,z:z+6] for y in range(3) for z in range(3)],2)
        self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(any('gather' in n or 'dep_conv' in n for n,_ in Eq11Attention(8,2,4).named_parameters()))

    def test_raw_pan_joint_kv_modequery_and_lpan_independence(self):
        torch.manual_seed(41);attn=Eq11Attention(8,2,4)
        x=torch.rand(2,8,8,8);ms=torch.rand(2,4,8,8);pan=torch.rand(2,1,8,8)
        got={}
        handles=[module.register_forward_pre_hook(lambda _m,args,key=name:got.__setitem__(key,args[0].detach().clone()))
                 for name,module in [('q',attn.q),('pan',attn.kv_pan),('ms',attn.kv_ms)]]
        first=attn(x,ms,torch.zeros_like(pan),pan,torch.tensor([1,0]))
        second=attn(x,ms,torch.full_like(pan,9000),pan,torch.tensor([1,0]))
        for handle in handles:handle.remove()
        self.assertTrue(torch.equal(first[0],second[0]));self.assertTrue(torch.equal(first[1],second[1]))
        self.assertTrue(torch.equal(got['pan'][:,:4],pan.repeat(1,4,1,1)))
        self.assertTrue(torch.equal(got['q'][0,:4],ms[0]))
        self.assertTrue(torch.equal(got['q'][1,:4],pan[1].repeat(4,1,1)))
        changed=attn(x,ms,None,pan*.7,torch.tensor([1,0]))
        self.assertFalse(torch.equal(first[0],changed[0]))

    def test_ms_inference_has_no_lp_and_zero_head_gives_msbase(self):
        model=build_model(4,seed=12,max_pixel=1023,hidden_size=8,depth=(1,1,1),num_heads=2)
        ms=torch.rand(2,4,4,4);pan=torch.rand(2,1,16,16)
        with patch.object(model,'pan_residual_base',side_effect=AssertionError('MS must not useLP')):
            actual=model(pan,ms)
        self.assertTrue(torch.equal(actual,F.interpolate(ms,scale_factor=4,mode='bicubic',align_corners=False)))
        self.assertEqual(model.input.in_channels,5)

    def test_pan_base_matches_original_dn_gaussian(self):
        from tools.repair_lpan import make_lpan
        model=build_model(4,seed=4,max_pixel=1023,hidden_size=8,depth=(1,1,1),num_heads=2)
        raw=torch.randint(0,1024,(2,1,32,32)).float();pan=raw.mul(2/1023).sub(1)
        low=torch.from_numpy(make_lpan(raw.double().numpy()).astype(np.float32)).mul(2/1023).sub(1)
        expected=F.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False).repeat(1,4,1,1)
        actual=model.pan_residual_base(pan,raw)
        self.assertTrue(torch.allclose(actual,expected,atol=2e-7,rtol=0))
        with self.assertRaises(ValueError):model.pan_residual_base(pan,raw+1)

    def test_model_no_panda_and_measured_parameters(self):
        for bands,maxdn in [(4,1023),(8,2047)]:
            model=build_model(bands,seed=2025,max_pixel=maxdn)
            counts=parameter_counts(model)
            self.assertEqual(counts['total'],counts['trainable'])
            self.assertEqual(model.input.in_channels,bands+1)
            self.assertEqual(len([m for m in model.modules() if isinstance(m,Eq11Attention)]),3)
            self.assertFalse(hasattr(model,'aligner'))
            self.assertEqual(model.depth,(2,2,4));self.assertEqual(model.dec_depth,(2,2))


if __name__=='__main__':unittest.main()
