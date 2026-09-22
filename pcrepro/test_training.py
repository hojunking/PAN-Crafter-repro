import copy
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pcrepro.model import build_model,state_hash
from pcrepro.training import (mars_loss,cosine_factor,make_optimizer,make_scheduler,
    rng_state,restore_rng,preserve_runtime,val_improved,should_pause,verify_metadata,verify_resume_environment)
from pcrepro.data import PairedBatchStream
from pcrepro.plan import build_config,Case
from pcrepro.common import atomic_json


class StreamDataset:
    split='train';spec={'shapes':{'ms':[4,16,16]}};manifest_sha256='fixture'
    def __len__(self):return 103


class Recording(torch.nn.Module):
    num_bands=4
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.7));self.last=None
    def forward(self,pan,ms,mode=None,pan_dn=None):
        self.last=(pan.detach().clone(),ms.detach().clone(),mode.detach().clone(),pan_dn.detach().clone())
        return self.weight*torch.ones(len(pan),4,*pan.shape[-2:])


class TrainingTests(unittest.TestCase):
    def test_mars_duplicate_after_geometry_sum_not_half(self):
        model=Recording();pan=torch.full((3,1,8,8),.2);ms=torch.zeros(3,4,2,2);gt=torch.full((3,4,8,8),.1)
        result=mars_loss(model,pan,ms,gt,pan)
        self.assertAlmostEqual(float(result['loss']),1.1,places=6)
        self.assertEqual(result['mars_count'],6)
        self.assertTrue(torch.equal(model.last[0][:3],model.last[0][3:]))
        self.assertTrue(torch.equal(model.last[2],torch.tensor([1.,1.,1.,0.,0.,0.])))
        result['loss'].backward();self.assertAlmostEqual(float(model.weight.grad),2.,places=6)

    def test_both_tasks_produce_real_model_gradients(self):
        model=build_model(4,seed=32,max_pixel=1023,hidden_size=8,depth=(1,1,1),num_heads=2)
        raw=torch.randint(0,1024,(2,1,16,16)).float();pan=raw*(2/1023)-1
        values=mars_loss(model,pan,torch.rand(2,4,4,4)*2-1,torch.rand(2,4,16,16)*2-1,raw)
        head=model.output[-1].weight
        for name in ('ms_loss','pan_loss'):
            gradient=torch.autograd.grad(values[name],head,retain_graph=True)[0]
            self.assertGreater(float(gradient.norm()),0.)

    def test_warmup_and_exact_endpoint_lr(self):
        self.assertEqual(cosine_factor(0),0.);self.assertEqual(cosine_factor(100),1.);self.assertEqual(cosine_factor(50000),0.)
        self.assertGreater(cosine_factor(49999),0.)
        for bad in (-1,50001,True):
            with self.assertRaises(ValueError):cosine_factor(bad)
        model=Recording();optimizer=make_optimizer(model,build_config(Case('s3',0,'GF2')))
        scheduler=make_scheduler(optimizer);before=model.weight.detach().clone()
        model.weight.grad=torch.ones_like(model.weight);optimizer.step();scheduler.step()
        self.assertTrue(torch.equal(before,model.weight))
        self.assertEqual(float(optimizer.state[model.weight]['step']),1.)

    def test_resume_sampler_optimizer_rng_exact(self):
        torch.manual_seed(13);np.random.seed(14);random.seed(15)
        model=Recording();optimizer=make_optimizer(model,build_config(Case('s3',0,'GF2')));scheduler=make_scheduler(optimizer)
        stream=PairedBatchStream(StreamDataset(),4,2025)
        def step():
            if stream.cursor==stream.count:stream.new_epoch()
            meta=torch.tensor(stream.batch_at(stream.cursor));verify_metadata(stream,meta)
            optimizer.zero_grad(set_to_none=True)
            target=torch.rand(())+np.random.rand()+random.random()+int(meta[0,0])*.001
            (model.weight-target).square().backward();optimizer.step();scheduler.step();stream.advance()
        for _ in range(5):step()
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                                rng=rng_state(),sampler=stream.state_dict()))
        for _ in range(8):step()
        expected=state_hash(model.state_dict());expected_batch=stream.batch_at(stream.cursor)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        stream.load_state_dict(saved['sampler']);restore_rng(saved['rng'])
        for _ in range(8):step()
        self.assertEqual(state_hash(model.state_dict()),expected);self.assertEqual(stream.batch_at(stream.cursor),expected_batch)

    def test_resume_refuses_gpu_or_runtime_policy_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy={'recipe_sha256':'fixture','policy':{'tf32_matmul':False}}
            value={'gpu':'Fixture GPU A','runtime_policy':policy}
            atomic_json(Path(tmp)/'meta/training_start_manifest.json',value)
            self.assertEqual(verify_resume_environment(tmp,policy,'Fixture GPU A'),value)
            with self.assertRaisesRegex(ValueError,'GPU/runtime policy'):
                verify_resume_environment(tmp,policy,'Fixture GPU B')
            with self.assertRaisesRegex(ValueError,'GPU/runtime policy'):
                verify_resume_environment(tmp,{'policy':{'tf32_matmul':True}},'Fixture GPU A')

    def test_val_ties_and_rng_preservation(self):
        self.assertFalse(val_improved({'ergas':2.},{'ergas':2.}))
        self.assertTrue(val_improved({'ergas':2.},{'ergas':1.99}))
        with self.assertRaises(FloatingPointError):val_improved(None,{'ergas':float('nan')})
        model=Recording().train();before=rng_state()
        with preserve_runtime(model):
            model.eval();torch.rand(5);random.random();np.random.rand()
        self.assertTrue(model.training);self.assertTrue(torch.equal(before['torch'],torch.get_rng_state()))

    def test_stop_controls_do_not_truncate_run_by_cycle_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'control.json'
            for command in ('CONTINUE','STOP_AFTER_CURRENT_RUN','STOP_AFTER_CYCLE'):
                atomic_json(path,dict(command=command));self.assertFalse(should_pause(path))
            atomic_json(path,dict(command='STOP_NOW_SAFE'));self.assertTrue(should_pause(path))
            atomic_json(path,dict(command='CONTINUE'));self.assertTrue(should_pause(path,True))


if __name__=='__main__':unittest.main()
