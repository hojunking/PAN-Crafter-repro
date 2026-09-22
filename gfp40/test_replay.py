import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from g20.model import state_hash
from gfp40.diagnostics import native_quality, perturbation_probe, _stats
from gfp40.replay import (crossed_model, factorial_effects, leave_one_scene_out,
    native_split_probe, save_fixed_visuals, FIXED_SCENES, FIXED_CROPS)


class TinyReplay(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(5, 4, 1)
        self.aligner = torch.nn.Linear(1, 2)

    def forward(self, pan, ms, lp, delta_override=None):
        delta = self.aligner(pan.mean((2,3))) if delta_override is None else delta_override
        base = F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)
        value = self.backbone(torch.cat((base, pan), 1)) + delta[:, :1, None, None]*.01
        return dict(y=base+value, delta=delta, ms_base=base)


class Views:
    gammas = (1., 1.25)
    native_gamma_id = 0
    def get_view(self, index, rot=0, augment=True, gamma_id=None):
        gamma_id = self.native_gamma_id if gamma_id is None else gamma_id
        pan = torch.linspace(-.5, .5, 64).reshape(1,8,8)*self.gammas[gamma_id]
        ms = torch.zeros(4,2,2)
        return (torch.zeros(4,8,8), torch.zeros(4,8,8), ms, torch.zeros(1,2,2), pan,
                torch.tensor([index,rot,1,1,gamma_id,gamma_id,0]))


class Native(torch.utils.data.Dataset):
    augment = False
    spec = SimpleNamespace(band_order=['blue','green','red','nir'])
    def __init__(self, split='rr', count=20):
        self.split, self.has_gt, self.count = split, split != 'fr', count
    def __len__(self): return self.count
    def __getitem__(self, i):
        size = 256 if self.split != 'val' else 8
        gt = torch.zeros(4,size,size)
        rest = (gt, torch.zeros(4,size//4,size//4),torch.zeros(1,size//4,size//4),
                torch.zeros(1,size,size),torch.tensor([i,0,0,0]))
        return (gt,)+rest if self.has_gt else rest


class ReplayTests(unittest.TestCase):
    def test_crossing_preserves_originals_and_is_independent(self):
        ctrl, mix = TinyReplay(), TinyReplay()
        before = [state_hash(m.state_dict()) for m in (ctrl,mix)]
        sources = dict(C=ctrl,M=mix)
        for combo in ('CC','CM','MC','MM'):
            model = crossed_model(ctrl,mix,combo)
            self.assertEqual(state_hash(model.aligner.state_dict()),state_hash(sources[combo[0]].aligner.state_dict()))
            self.assertEqual(state_hash(model.backbone.state_dict()),state_hash(sources[combo[1]].backbone.state_dict()))
            with torch.no_grad(): next(model.parameters()).add_(1)
        self.assertEqual(before,[state_hash(m.state_dict()) for m in (ctrl,mix)])
        with self.assertRaises(ValueError): crossed_model(ctrl,mix,'BEST')

    def test_factorial_algebra_signs_and_no_official_candidates(self):
        rows = {combo:dict(fr={k:val for k in ('hqnr','d_s','d_lambda')},
                          rr={k:val for k in ('ergas','scc')})
                for combo,val in [('CC',1.),('CM',3.),('MC',4.),('MM',10.)]}
        result = factorial_effects(rows)
        self.assertEqual(result['effects']['hqnr'],dict(U_at_CTRL_A=2.,A_at_CTRL_U=3.,interaction=4.,total=9.))
        self.assertTrue(result['diagnostic_only'])

    def test_leaveoneout_never_changes_twenty_scene_aggregate(self):
        summary = dict(per_scene=[dict(hqnr=float(i)) for i in range(20)])
        result = leave_one_scene_out(summary)
        self.assertEqual(result['metrics']['hqnr']['full20_mean'],9.5)
        self.assertEqual(result['metrics']['hqnr']['leave_one_out_means'][0],10.)
        self.assertFalse(result['scene_exclusion_allowed'])
        with self.assertRaises(ValueError): leave_one_scene_out(dict(per_scene=summary['per_scene'][:19]))

    def test_quality_without_clipping_or_hiding_overshoot(self):
        y = torch.full((2,4,8,8),2.)
        quality = native_quality(y,torch.zeros_like(y),torch.zeros_like(y))
        self.assertEqual(quality['mean']['pixel_mae'],2.)
        self.assertEqual(quality['mean']['band_mae'],[2.]*4)
        self.assertEqual(quality['mean']['output_overshoot'],1.)
        self.assertEqual(quality['mean']['residual_energy'],4.)
        self.assertIn('p95',_stats([1.,2.,3.]))

    def test_perturbation_native_id_zero_and_cfreeze_no_mutation(self):
        model = TinyReplay().train()
        before = state_hash(model.state_dict())
        result = perturbation_probe(model,Views(),[0,1],device='cpu')
        self.assertEqual(result['gamma_sweep']['1.0']['output_mae']['mean'],0.)
        self.assertEqual(result['gamma_sweep']['1.0']['correction_shift']['mean'],0.)
        self.assertGreater(result['gamma_sweep']['1.25']['output_mae']['mean'],0.)
        self.assertEqual(before,state_hash(model.state_dict()))
        self.assertTrue(model.training)

    def test_native_quality_population_and_parent_drift(self):
        model = TinyReplay()
        one = native_split_probe(model,Native('val',3),'cpu')
        two = native_split_probe(model,Native('val',3),'cpu',parent_delta=one['corrections'])
        self.assertEqual(two['quality']['n_samples'],3)
        self.assertEqual(two['drift_from_parent']['maximum'],0.)
        self.assertEqual(two['independent_estimator'],'ESTIMATOR_NOT_AVAILABLE')

    def test_fixed_scene_visuals_have_raw_error_and_no_fr_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = save_fixed_visuals(TinyReplay(),dict(rr=Native(),fr=Native('fr')),tmp,'cpu')
            root = Path(tmp)
            self.assertEqual(manifest['scene_indices'],list(FIXED_SCENES))
            self.assertEqual(len(list(root.glob('rr/scene_*/crop_*_rr_error.png'))),len(FIXED_SCENES)*len(FIXED_CROPS))
            self.assertFalse(list(root.glob('fr/**/*rr_error*')))
            self.assertFalse(manifest['FR_has_GT'])


if __name__ == '__main__': unittest.main()
