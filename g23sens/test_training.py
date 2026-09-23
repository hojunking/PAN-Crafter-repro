"""Serialized optimizer/scheduler/RNG resume contracts, without real training."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from g23sens.common import atomic_json,sha256,object_sha,RuntimePaused
from g23sens.model import (initial_snapshot,build_model,load_model,state_hash)
from g23sens.diagnostics import rng_state,restore_rng
from g23sens.training import (make_optimizer,make_scheduler,verify_restored_state,
    routed_backward,GRID_STEPS,smoke_validate)
from g23sens.test_numerics import config,tiny_pair,objective
from pa.aligner import PANGlobalAligner


class Cursor:
    def __init__(self,cursor=0):self.cursor=cursor


class TrainingContracts(unittest.TestCase):
    def test_grid_primary_and_lower_step_tie(self):
        self.assertEqual(len(GRID_STEPS),50)
        self.assertEqual(GRID_STEPS[-2:],(49490,50000))
        self.assertEqual(min([dict(update=2020,val_ergas=2.),dict(update=1010,val_ergas=2.)],
            key=lambda row:(row['val_ergas'],row['update']))['update'],1010)

    def test_serialized_exact_resume_same_grad_and_update(self):
        cfg=config();model,teacher,batch=tiny_pair()
        opt=make_optimizer(model,cfg);sch=make_scheduler(opt,cfg)
        def step():
            opt.zero_grad(set_to_none=True)
            _,terms=objective(model,teacher,batch)
            routed_backward(model,terms);opt.step();sch.step()
        step()
        packed=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                        scheduler=sch.state_dict(),rng=rng_state()),packed)
        step();expected=copy.deepcopy(model.state_dict());expectedopt=copy.deepcopy(opt.state_dict())
        packed.seek(0);restored=torch.load(packed,map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
        sch.load_state_dict(restored['scheduler']);restore_rng(restored['rng'])
        step()
        self.assertEqual(state_hash(expected),state_hash(model.state_dict()))
        self.assertEqual(sch.last_epoch,2)
        for key,item in opt.state_dict()['state'].items():
            for name,value in item.items():
                self.assertTrue(torch.equal(value,expectedopt['state'][key][name]))

    def test_restore_checks_source_and_sampler_cursor(self):
        context={'source_identity':{'content':'pinned'},'init_U_sha256':'fixed'}
        state=dict(full_state=True,precision='fp32',update=19,**context)
        scheduler=mock.Mock(last_epoch=19)
        self.assertEqual(verify_restored_state(state,context,Cursor(19),scheduler),19)
        for broken in (dict(state,update=18),dict(state,precision='fp16'),
                       dict(state,source_identity={'content':'different'}),dict(state,full_state=False)):
            with self.assertRaises(ValueError):verify_restored_state(broken,context,Cursor(19),scheduler)

    def test_new_arms_clone_snapshot_independently(self):
        cfg=config()
        teacher=mock.Mock(aligner=PANGlobalAligner(8),aligner_margin=4)
        snapshot=initial_snapshot(cfg,teacher)
        first=build_model(cfg,snapshot);second=build_model(cfg,snapshot)
        self.assertEqual(state_hash(first.state_dict()),state_hash(second.state_dict()))
        self.assertFalse(set(map(id,first.parameters()))&set(map(id,second.parameters())))
        before=state_hash(second.state_dict())
        with torch.no_grad():next(first.backbone.parameters()).add_(1.)
        self.assertEqual(state_hash(second.state_dict()),before)
        self.assertEqual(state_hash(snapshot['U']),snapshot['hashes']['U'])
        changed=copy.deepcopy(snapshot)
        changed['U'][next(iter(changed['U']))].add_(1.)
        with self.assertRaises(ValueError):build_model(cfg,changed)

    def test_candidate_strict_load_hashes(self):
        cfg=config();teacher=mock.Mock(aligner=PANGlobalAligner(8),aligner_margin=4)
        model=build_model(cfg,initial_snapshot(cfg,teacher))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/'model.safetensors'
            save_file(model.state_dict(),str(path))
            identity=dict(update=50000,model_sha256=sha256(path),state_hash=state_hash(model.state_dict()))
            atomic_json(folder/'identity.json',identity)
            restored,got=load_model(cfg,folder)
            self.assertEqual(got,identity)
            self.assertEqual(state_hash(model.state_dict()),state_hash(restored.state_dict()))
            atomic_json(folder/'identity.json',dict(identity,model_sha256='0'*64))
            with self.assertRaises(ValueError):load_model(cfg,folder)

    def test_cpu_mocked_device_loop_safe_pause_and_exact_resume(self):
        """Run the real persistence loop; CUDA calls alone are mocked, not evidence
        of actual GPU acceptance. No production path or asset is accessed."""
        from contextlib import ExitStack
        from g23sens.training import train_run
        from g23sens.common import read_json
        from g23sens.plan import make_case
        model,teacher,batch=tiny_pair();teacher.aligner_margin=4
        snapshot=dict(policy='UNIT_FIXTURE',seed=1234,
            U=copy.deepcopy(model.backbone.state_dict()),A=copy.deepcopy(model.aligner.state_dict()),
            post_constructor_torch_rng=torch.get_rng_state(),hashes=dict(
                U=state_hash(model.backbone.state_dict()),A=state_hash(model.aligner.state_dict())))
        class Dataset:
            def __init__(self,*args,**kwargs):pass
            def close(self):pass
        class Stream:
            def __init__(self,*args,**kwargs):self.cursor=0;self.manifest={'fixture':True}
            def next_batch(self):return batch
            def advance(self):self.cursor+=1
            def state_dict(self):return {'cursor':self.cursor}
            def load_state_dict(self,value):self.cursor=value['cursor']
            def fixed_probe(self):return batch
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as patches:
            root=Path(tmp);wd=root/'run';cfg=config();case=make_case('s4',0,'BASE')
            bindings={'server':'s4','teacher':{'checkpoint_sha256':'t'*64},'cue':{'raw_q_sha256':'q'*64}}
            bindpath=root/'bindings.json';atomic_json(bindpath,bindings)
            cfg['g23sens'].update(case=case,bindings_path=str(bindpath),binding_sha256=object_sha(bindings))
            cfg['work_dir']=str(wd);cfgpath=root/'cfg.json';atomic_json(cfgpath,cfg)
            def pause_at_two(case,root,update):
                if update>=2:raise RuntimePaused('unit stop')
            tensor_to=torch.Tensor.to
            def cpu_to(value,*args,**kwargs):
                args=tuple(torch.device('cpu') if isinstance(arg,torch.device) and arg.type=='cuda' else arg for arg in args)
                if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
                return tensor_to(value,*args,**kwargs)
            for name,value in dict(is_available=True,is_initialized=False,is_current_stream_capturing=False,
                                   max_memory_allocated=0).items():
                patches.enter_context(mock.patch.object(torch.cuda,name,return_value=value))
            patches.enter_context(mock.patch.object(torch.Tensor,'to',cpu_to))
            patches.enter_context(mock.patch.object(torch.cuda,'manual_seed_all'))
            patches.enter_context(mock.patch.object(torch.cuda,'synchronize'))
            for target in ('g23sens.plan.validate_config','g23sens.controller.authorize_train',
                           'g23sens.assets.validate_bindings','g23sens.training.validate_architecture'):
                patches.enter_context(mock.patch(target))
            patches.enter_context(mock.patch('g23sens.common.apply_runtime_policy',return_value={'unit':True}))
            patches.enter_context(mock.patch('g23sens.common.source_identity',return_value={'unit':'pinned'}))
            stop=patches.enter_context(mock.patch('g23sens.common.check_runtime',side_effect=pause_at_two))
            patches.enter_context(mock.patch('g23sens.assets.load_training_assets',return_value={
                'raw_q':torch.full((3,4),.3,dtype=torch.float64),'teacher_run_path':'unit-only'}))
            patches.enter_context(mock.patch('g23sens.data.NativeDataset',Dataset))
            patches.enter_context(mock.patch('g23sens.data.BatchStream',Stream))
            patches.enter_context(mock.patch('g23sens.training.load_run_model',side_effect=lambda *a:(copy.deepcopy(teacher),{'tag_meta':{'step':24240}})))
            patches.enter_context(mock.patch('g23sens.training._cycle_initialization',return_value=snapshot))
            patches.enter_context(mock.patch('g23sens.training.build_model',side_effect=lambda *a:copy.deepcopy(model)))
            patches.enter_context(mock.patch('g23sens.training.GRID_STEPS',(1,2)))
            patches.enter_context(mock.patch('g23sens.training.DIAGNOSTIC_STEPS',(0,2)))
            patches.enter_context(mock.patch('g23sens.evaluation.evaluate_validation',return_value={'ergas':2.,'scc':.99,'hqnr':None}))
            self.assertEqual(train_run(cfgpath,root=root),75)
            state=torch.load(wd/'last/training_state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['update'],2);self.assertEqual(state['sampler']['cursor'],2)
            self.assertFalse(read_json(wd/'meta/training_status.json')['training_complete'])
            old_candidate=sha256(wd/'candidates/2/model.safetensors')
            def pause_at_three(case,root,update):
                if update>=3:raise RuntimePaused('unit stop after resume')
            stop.side_effect=pause_at_three
            self.assertEqual(train_run(cfgpath,root=root,resume=True),75)
            state=torch.load(wd/'last/training_state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['update'],3);self.assertEqual(state['sampler']['cursor'],3)
            self.assertEqual(state['scheduler']['last_epoch'],3)
            self.assertEqual(sha256(wd/'candidates/2/model.safetensors'),old_candidate)
            # A partially failed optimizer update must retain the earlier,
            # coherent on-disk model/optimizer/sampler transaction byte-for-byte.
            before_fault={}
            stop.side_effect=lambda *args:False
            def partial_step(optimizer,*args,**kwargs):
                before_fault['state']=sha256(wd/'last/training_state.pt')
                before_fault['identity']=sha256(wd/'last/identity.json')
                with torch.no_grad():optimizer.param_groups[0]['params'][0].add_(123.)
                raise RuntimeError('simulated partial optimizer mutation')
            with mock.patch.object(torch.optim.AdamW,'step',partial_step):
                with self.assertRaisesRegex(RuntimeError,'partial optimizer mutation'):
                    train_run(cfgpath,root=root,resume=True)
            self.assertEqual(sha256(wd/'last/training_state.pt'),before_fault['state'])
            self.assertEqual(sha256(wd/'last/identity.json'),before_fault['identity'])
            status=read_json(wd/'meta/training_status.json')
            self.assertEqual(status['status'],'TECHNICAL_FAILURE')
            self.assertFalse(status['safe_update_boundary'])
            self.assertEqual(status['last_valid_checkpoint_update'],3)


if __name__=='__main__':unittest.main()
