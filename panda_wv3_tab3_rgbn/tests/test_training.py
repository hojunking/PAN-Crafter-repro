"""Exercise ablated gradients and exact optimizer/stream/RNG continuation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from donor30.artifacts import _same, _cpu
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha
from panda_wv3_tab3_rgbn import plan, model, training


def synthetic_source(root):
    root=Path(root);rng=np.random.default_rng(47);p=root/'train.h5'
    with h5py.File(p,'w') as h:
        for name,ch,side in [('pan',1,64),('ms',8,16),('lms',8,64),('gt',8,64)]:
            h[name]=rng.uniform(300,1500,(2,ch,side,side)).astype('float32')
    catalog=root/'catalog.json'
    catalog.write_text(json.dumps({'schema':'ABLR2_SENSOR_SOURCES_v1','sensors':{'WV3':{
        'band_order':['Coastal','B','G','Y','R','RE','NIR1','NIR2'],
        'splits':{'train':{'path':'train.h5','sha256':file_sha(p)}}}}}))
    return RegisteredRRTrain.from_catalog(catalog,root=root,sensor='WV3')


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_each_ablation_has_expected_losses_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=synthetic_source(tmp);batch=source.batch([0])
            teacher_initial=None
            for case in plan.stage1_cases('s3'):
                cfg=plan.resolve('s3',case,1);net=model.Stage1(cfg,source.inventory.manifest())
                current=_cpu(net.teacher.state_dict())
                if teacher_initial is None:teacher_initial=current
                else:self.assertTrue(_same(teacher_initial,current))
                rng=net.epsilon_rng.get_state().clone()
                with patch('pa.warp.warp_pan',side_effect=AssertionError('Bicubic PAN path used')):
                    terms=net(batch);terms['loss'].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.teacher.parameters()))
                if case=='B_REC':
                    # Historical zero-initialized reconstruction head blocks the
                    # PAN gradient at update0; after fitting T once, rec must reach A.
                    torch.optim.SGD(net.teacher.parameters(),lr=1e-3).step()
                    net.zero_grad(set_to_none=True)
                    terms=net(batch);terms['loss'].backward()
                if case!='A_SCHARR':
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.aligner.parameters()))
                else:self.assertEqual(sum(p.numel() for p in net.aligner.parameters()),0)
                if not cfg['lambda_abs']:self.assertEqual(float(terms['spatial']),0)
                if not cfg['lambda_epsilon']:
                    self.assertEqual(float(terms['epsilon']),0)
                    self.assertTrue(torch.equal(rng,net.epsilon_rng.get_state()))
                torch.testing.assert_close(terms['loss'],terms['rec']+cfg['lambda_abs']*terms['spatial']+cfg['lambda_epsilon']*terms['epsilon'])

    def test_resume_before_first_update_and_after_update(self):
        for case in ('A_SCHARR','OURS'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                source=synthetic_source(tmp);cfg=deepcopy(plan.resolve('s3',case,1))
                cfg.update(batch=1,microbatch=1)
                cfg['config_sha256']=plan.digest({k:v for k,v in cfg.items() if k!='config_sha256'})
                with patch.object(plan,'validate'):
                    trainer=training.Trainer(cfg,model.Stage1(cfg,source.inventory.manifest()))
                stream=training.stream_for(cfg,source,'cpu');path=Path(tmp)/'resume.pt'
                training.save(path,trainer,stream,{'synthetic':True})
                training.load(path,trainer,stream,{'synthetic':True})
                trainer.step(stream.next_update())
                training.save(path,trainer,stream,{'synthetic':True})
                expected=trainer.step(stream.next_update());expected_hash=training.model_hash(trainer.model)
                training.load(path,trainer,stream,{'synthetic':True})
                if case=='A_SCHARR':trainer.model.aligner._cache.clear()
                actual=trainer.step(stream.next_update())
                self.assertEqual(expected,actual)
                self.assertEqual(expected_hash,training.model_hash(trainer.model))


if __name__=='__main__':unittest.main()
