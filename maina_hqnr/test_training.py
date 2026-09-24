"""Unit tests isolate all RNG; no real asset, GPU, controller or upload writes."""
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from fh12.training import BatchStream,rng_state,save_resume_bundle
from maina_hqnr.data import stream_manifest,completed_updates,epoch_loader
from maina_hqnr.training import verify_restored_state,verify_optimizer_updates,make_optimizer,make_scheduler,GRID_STEPS
from maina_hqnr.diagnostics import isolated_probe,rng_equal,assert_tree_equal,smoke_roundtrip


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):return 96
    def __getitem__(self,index):
        i,rot=index
        gen=torch.Generator().manual_seed(i*4+rot)
        gt=torch.rand(8,16,16,generator=gen)
        pan=torch.rand(1,16,16,generator=gen)
        ms=torch.rand(8,4,4,generator=gen);lp=torch.rand(1,4,4,generator=gen)
        return gt,gt,ms,lp,pan,torch.tensor([i,rot,1,1])


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Conv2d(1,8,1);self.aligner=torch.nn.Linear(1,2)
    def forward(self,pan,ms,lp):
        delta=self.aligner(pan.mean((2,3)))
        y=self.backbone(pan)+delta.sum(1)[:,None,None,None]*pan
        return dict(y=y,delta=delta)


def config():
    return dict(seed=73101,optimizer='AdamW',learning_rate=1e-4,weight_decay=.01,
        betas=[.9,.999],eps=1e-8,fh20r1=dict(aligner_lr=3e-6),
        num_warmup=100,num_iter=50000,lr_scheduler='cosine')


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_fifty_grid(self):
        self.assertEqual(len(GRID_STEPS),50);self.assertEqual(GRID_STEPS[:2],(1010,2020))
        self.assertEqual(GRID_STEPS[-2:],(49490,50000))

    def test_stream_digest_entire_consumed_grid_original_order(self):
        n,batch,seed,count=101,48,73101,9
        state=rng_state();reported=stream_manifest(n,batch,seed,count)
        self.assertTrue(rng_equal(state,rng_state()))
        stream=BatchStream(n,batch,seed);pairs=[]
        for _ in range(count):
            if stream.cursor==stream.count:stream.new_epoch()
            pairs.extend(stream.remaining_batches()[0]);stream.cursor+=1
        expected=hashlib.sha256(np.asarray(pairs,dtype='<i8').tobytes()).hexdigest()
        self.assertEqual(reported['sample_view_sequence_sha256'],expected)
        self.assertEqual(completed_updates(stream),count)

    def test_epoch_loader_explicit_view_and_rng_isolation(self):
        stream=BatchStream(96,48,17);state=rng_state()
        batch=next(iter(epoch_loader(TinyDataset(),stream,17,workers=0,pin_memory=False)))
        self.assertEqual(batch[-1][:,:2].tolist(),[list(x) for x in stream.remaining_batches()[0]])
        self.assertTrue(rng_equal(state,rng_state()))

    def test_probe_restores_python_numpy_torch_and_model_mode_on_error(self):
        model=TinyNet().train();state=rng_state()
        with self.assertRaises(RuntimeError):
            with isolated_probe(model):
                model.eval();random.random();np.random.rand();torch.rand(2)
                raise RuntimeError('test')
        self.assertTrue(model.training);self.assertTrue(rng_equal(state,rng_state()))

    def test_fullstate_roundtrip_optimizer_rng_stream(self):
        cfg=config();case=dict(alpha=1.,beta=.1,lambda_E=.002)
        model,teacher=TinyNet(),TinyNet();teacher.eval().requires_grad_(False)
        with isolated_probe(model,teacher):
            report=smoke_roundtrip(model,teacher,TinyDataset(),torch.ones(96,4),
                                  .4532603621482849,.012118559330701828,case,cfg,'cpu',workers=0)
        self.assertTrue(report['passed']);self.assertEqual(report['max_abs_error'],0.)

    def test_resume_rejects_weights_only_wrong_cursor_and_identity(self):
        stream=BatchStream(96,48,73101);model=TinyNet()
        scheduler=make_scheduler(make_optimizer(model,config()),config())
        state=dict(full_state=True,precision='fp32',update=0,source='fixed')
        self.assertEqual(verify_restored_state(state,dict(source='fixed'),stream,scheduler),0)
        for changes in (dict(full_state=False),dict(update=1),dict(source='wrong')):
            with self.assertRaises(ValueError):verify_restored_state(dict(state,**changes),dict(source='fixed'),stream,scheduler)

    def test_original_scheduler_update_boundaries(self):
        opt=make_optimizer(TinyNet(),config());sch=make_scheduler(opt,config())
        self.assertEqual([g['lr'] for g in opt.param_groups],[0.,0.])
        for _ in range(100):opt.step();sch.step()
        self.assertEqual([g['lr'] for g in opt.param_groups],[1e-4,3e-6])

    def test_resume_optimizer_counters_match_actual_updates(self):
        model=TinyNet();opt=make_optimizer(model,config())
        verify_optimizer_updates(opt,0)
        for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
        opt.step();verify_optimizer_updates(opt,1)
        with self.assertRaises(ValueError):verify_optimizer_updates(opt,2)
        with self.assertRaises(ValueError):verify_optimizer_updates(opt,0)


if __name__=='__main__':unittest.main()
