"""CPU numerical contracts; these tests do not constitute GPU preflight."""
import copy
import ast
from pathlib import Path
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml

from g23sens.model import fresh_backbone, state_hash
from g23sens.training import (make_criterion, make_optimizer, make_scheduler,
    forward_objectives, routed_gradients, routed_backward, qweight_from_raw)
from g23sens.diagnostics import fixed_probe, isolated_probe
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.model import PAModel
from pa.losses import output_edge_loss_per_sample
from model.pancrafter_paper import PANCrafterPaper


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml'


def config():
    with BASE.open() as handle:
        cfg = yaml.safe_load(handle)
    cfg['g23sens'] = dict(case=dict(resolved_values=dict(
        A_peak_lr=3e-6, tau_R_used=cfg['kdv']['rec']['tau'])))
    return cfg


class TinyU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,3,padding=1)
    def forward(self, pan, lp, ms, switches):
        base = torch.nn.functional.interpolate(ms, scale_factor=4, mode='bicubic')
        return self.conv(torch.cat((pan,base),1))


class TinyA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.09,-.07]))
        self.gain = nn.Parameter(torch.tensor(.13))
    def forward(self, pan, base):
        return self.weight[None].expand(len(pan),-1)+self.gain*pan.mean((1,2,3))[:,None]


def tiny_pair():
    torch.manual_seed(9)
    model = PAModel(TinyU(), TinyA(), aligner_margin=0)
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    with torch.no_grad():
        teacher.backbone.conv.weight.mul_(.2)
        teacher.aligner.weight.add_(.12)
    batch = dict(gt=torch.randn(3,1,16,16),ms=torch.randn(3,1,4,4),
                 lpan=torch.randn(3,1,4,4),pan=torch.randn(3,1,16,16),
                 meta=torch.tensor([[0,0,1,1],[1,2,1,1],[2,3,1,1]]))
    return model,teacher,batch


def objective(model, teacher, batch, criterion=None, edge=.002):
    criterion = criterion or GTAnchoredReconstructionKD(.3)
    weights = qweight_from_raw(torch.full((3,4),.3),.3)
    out,terms = forward_objectives(model,teacher,criterion,weights,
        *(batch[k] for k in ('gt','ms','lpan','pan','meta')),edge)
    return out,terms


class NumericalContracts(unittest.TestCase):
    def test_exact_legacy_constructor_not_named_initialization(self):
        cfg=config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'])
            expected=PANCrafterPaper(**cfg['model_args'])
            rng=torch.get_rng_state()
        model,after=fresh_backbone(cfg)
        self.assertEqual(state_hash(model.state_dict()),state_hash(expected.state_dict()))
        self.assertTrue(torch.equal(rng,after))
        cfg['seed']+=1
        different,_=fresh_backbone(cfg)
        self.assertNotEqual(state_hash(model.state_dict()),state_hash(different.state_dict()))

    def test_tau_scaling_once(self):
        for scale in (.5,1,2):
            cfg=config()
            raw=cfg['kdv']['rec']['tau']
            cfg['kdv']['rec']['tau_scale']=scale
            cfg['g23sens']['case']['resolved_values']['tau_R_used']=raw*scale
            self.assertEqual(float(make_criterion(cfg).tau),raw*scale)
            if scale != 1:
                cfg['kdv']['rec']['tau']*=scale
                with self.assertRaises(ValueError): make_criterion(cfg)

    def test_actual_optimizer_and_scheduler_arms(self):
        for lr in (1e-6,3e-6,6e-6):
            cfg=config();cfg['kdv']['aligner_lr']=lr
            cfg['g23sens']['case']['resolved_values']['A_peak_lr']=lr
            model,_,_=tiny_pair()
            opt=make_optimizer(model,cfg)
            self.assertEqual([g['lr'] for g in opt.param_groups],[1e-4,lr])
            sch=make_scheduler(opt,cfg)
            self.assertEqual([g['lr'] for g in opt.param_groups],[0.,0.])
            for _ in range(100): opt.step();sch.step()
            self.assertEqual([g['lr'] for g in opt.param_groups],[1e-4,lr])

    def test_qref_changes_only_weight_not_raw(self):
        raw=torch.full((2,4),.3276133416220546,dtype=torch.float64)
        weights=[qweight_from_raw(raw,float(raw[0,0])*s) for s in (.5,1,2)]
        self.assertEqual(len({w.stats['q_raw_sha256'] for w in weights}),1)
        self.assertEqual(len({w.stats['w_sha256'] for w in weights}),3)
        for w,want in zip(weights,(1/3,.5,2/3)):
            self.assertAlmostEqual(float(w.tables['q'][0,0]),want,places=7)

    def test_legacy_formula_per_sample_exact(self):
        model,teacher,batch=tiny_pair();crit=GTAnchoredReconstructionKD(.3)
        out,terms=objective(model,teacher,batch,crit)
        rec=crit(out['y'],out['y_t'],batch['gt'],return_maps=True)
        e=(out['y']-batch['gt']).abs().mean(1,keepdim=True)
        k=(out['y']-out['y_t']).abs().mean(1,keepdim=True)
        h=(rec.maps['hard_weight']*e).mean((1,2,3))
        kd=(rec.maps['soft_weight']*k).mean((1,2,3))
        edge=output_edge_loss_per_sample(out['y'],batch['gt'])
        self.assertTrue(torch.equal(terms['L_U'],(h+kd+.002*.5*edge).mean()))
        self.assertTrue(torch.equal(terms['L_A'],(.5*h).mean()))

    def test_forbidden_A_gradient_paths_removed_in_routing(self):
        model,teacher,batch=tiny_pair()
        _,terms=objective(model,teacher,batch)
        gu,ga=routed_gradients(model,terms,retain_graph=True)
        altered=dict(terms,L_U=terms['H'].mean()+31*terms['K'].mean()+45*terms['weighted_E'])
        gu2,ga2=routed_gradients(model,altered,retain_graph=True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ga,ga2)))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(gu,gu2)))
        direct=torch.autograd.grad(terms['L_U'],tuple(model.aligner.parameters()),retain_graph=True)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(ga,direct)))
        routed_backward(model,terms)
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(model.aligner.parameters(),ga)))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_actual_legacy_routing_method_gradient_parity(self):
        # Execute the checked-in original method itself without constructing a
        # KDVTrainer (which would touch old ledgers, datasets and CUDA).
        source=ast.parse((ROOT/'train_kdv.py').read_text())
        klass=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='KDVTrainer')
        method=next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='_qrecon_backward')
        scope={'torch':torch}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(ROOT/'train_kdv.py'),'exec'),scope)
        first,teacher,batch=tiny_pair();second=copy.deepcopy(first)
        _,terms=objective(first,teacher,batch);_,legacy_terms=objective(second,teacher,batch)
        routed_backward(first,terms)
        owner=SimpleNamespace(accelerator=SimpleNamespace(gradient_accumulation_steps=1))
        scope['_qrecon_backward'](owner,dict(_L_U_t=legacy_terms['L_U'],_L_A_t=legacy_terms['L_A']),second,True)
        self.assertTrue(all(torch.equal(a.grad,b.grad) for a,b in zip(first.parameters(),second.parameters())))

    def test_diagnostic_restores_rng_grad_and_mode(self):
        model,teacher,batch=tiny_pair();cfg=config();opt=make_optimizer(model,cfg)
        model.train()
        for p in model.parameters(): p.grad=torch.ones_like(p)
        before=torch.get_rng_state().clone();npstate=np.random.get_state();py=random.getstate()
        manifest=fixed_probe(model,teacher,GTAnchoredReconstructionKD(.3),
            qweight_from_raw(torch.full((3,4),.3),.3),batch,opt,step=0,
            lambda_edge=.002,initial_aligner_state=copy.deepcopy(model.aligner.state_dict()))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters()))
        self.assertTrue(manifest['gradient_routing']['A_forbidden_soft_edge_offset_invariance'])
        self.assertEqual(manifest['aligner_change_l2'],0.)


if __name__=='__main__': unittest.main()
