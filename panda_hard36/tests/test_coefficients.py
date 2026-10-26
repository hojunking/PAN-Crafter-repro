"""Verify alpha/gamma-only controls, exact uniform-L1 gradient and resume on CPU.

The donor and RR image are deterministic synthetic fixtures. Production model
widths, warps, Hard calculation, calibration, optimizer and checkpoint I/O are
executed; only the test batch size is reduced to one.
"""
from collections import Counter
from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from donor30.masking import masked_mean
from panda_band30.tests.test_method import make_donor
from panda_band30.reconstruction import reconstruct_with_base
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.tests.test_method import batch
from panda30r.tests.test_engine import MemoryStream
from panda_hard36 import plan, calibration as calibration
from panda_hard36.contract import CASES, CAMPAIGN_ID, WARP_MANIFEST
from panda_hard36.engine import (
    ScheduleTrainingSpec, build_fresh_stage2, advance, save_checkpoint,
    load_checkpoint,
)
from panda_hard36.evaluation import configured_pipeline
from panda_hard36.method import frozen_predictions
from panda_hard36.spec import MatureSpec
from panda_hard36.warp import warp_pan
from pa.warp import warp_pan as bicubic_warp_pan


EVALUATOR_PIN = {'test': 'explicit_cpu_synthetic'}
CODE_PIN = {'synthetic_test_pin.py': 'd' * 64}


def training_fixture(case, replica='R1'):
    """Use the real registry/builder with a real, small conditional-q bank."""
    cfg = plan.resolve_case('s3', case, replica)
    cfg['training_spec'].update(microbatch_size=1, effective_batch_size=1)
    spec = MatureSpec(**cfg['stage2_spec'])
    donor = calibration.bind_donor(make_donor(shift=(1., -.5)), spec)
    rr = batch()
    key = spec.geometry_keys[0]
    frozen = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key=key)
    e, g, q = reference_maps(ScharrAbsolute('WV3'), frozen['teacher_y'], rr.gt,
                             frozen['pan_loss'])
    collector = CalibrationBuilder(
        ['WV3:train:0'], calibration.binding_for_geometry(donor.binding, key),
        k=5, memory_budget_bytes=8 * 1024**2, include_condq=True,
        mask_policy=spec.mask_policy,
    )
    collector.add(['WV3:train:0'], e, g, q, sensor='WV3',
                  loss_mask=frozen['loss_mask'])
    cfg['donor_binding'] = deepcopy(donor.binding)
    return build_fresh_stage2(cfg, donor, {key: collector.finalize()},
                              evaluator_pin=EVALUATOR_PIN, device='cpu')


class HardCoefficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_registry_is_only_six_new_alpha_gamma_pairs_with_fixed_beta_zero(self):
        expected={(0.,0.),(.25,0.),(0.,.5),(.125,.25),(.5,.5),(.25,1.)}
        self.assertEqual({(v['alpha'],v['gamma']) for v in CASES.values()},expected)
        from panda_tune30 import plan as parent
        for server in ('s1','s3'):
            self.assertEqual(plan.donor_receipt_path(server),parent.donor_receipt_path(server))
            configs=[plan.resolve_case(server,c,r) for g in plan.groups(server) for c,r in g['runs']]
            self.assertEqual(len(configs),12)
            self.assertEqual(sum(c['planned_horizon'] for c in configs),360000)
            self.assertEqual(plan.default_config(server)['budget']['total_seconds'],36*3600)
            self.assertFalse({c['case'] for c in configs} & {c for g in parent.groups(server) for c,r in g['runs']})
            for cfg in configs:
                self.assertEqual(cfg['stage2_spec']['beta'],0.)
                self.assertEqual(cfg['student_architecture']['width'],112)
                self.assertEqual(cfg['shift_scale'],1.)
                self.assertEqual(cfg['sampling_variant'],'UBIL')
                self.assertEqual(cfg['input_variant'],'CORRECTED_DIFF')
                self.assertEqual(cfg['lr_S'],1e-4)
                self.assertEqual(cfg['planned_horizon'],30000)
                self.assertEqual(cfg['initialization'],'FRESH')
                self.assertEqual(cfg['stage2_spec']['variant'],'panda_e' if cfg['stage2_spec']['gamma']==0 else 'panda_condq')
            with (Path(plan.__file__).parent/f'cases_{server}.csv').open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)),list(csv.DictReader(io.StringIO(plan.catalog_csv(server)))))

    def test_every_setting_executes_exact_registered_hard_only_loss(self):
        init=None
        for case in CASES:
            with self.subTest(case=case):
                trainer=training_fixture(case);stage=trainer.stage;rr=batch()
                current=deepcopy(stage.student.state_dict())
                if init is None: init=current
                else: self.assertTrue(_same(init,current))
                a=deepcopy(stage.aligner.state_dict());t=deepcopy(stage.teacher.state_dict())
                out=stage(rr,return_maps=True);maps=out['maps']
                alpha,gamma=CASES[case]['alpha'],CASES[case]['gamma']
                error=(out['student']['y']-rr.gt).abs().mean(1,keepdim=True)
                weight=1+alpha*maps['d_e']+gamma*maps['r_q_bar']*maps['d_e']
                torch.testing.assert_close(out['loss'],masked_mean(weight*error,out['loss_mask']),rtol=1e-7,atol=1e-8)
                self.assertEqual(float(out['soft_weighted']),0.)
                if alpha==gamma==0:
                    plain=masked_mean(error,out['loss_mask'])
                    torch.testing.assert_close(out['loss'],plain,rtol=0,atol=0)
                    actual=torch.autograd.grad(out['loss'],out['student']['y'],retain_graph=True)[0]
                    expected=torch.autograd.grad(plain,out['student']['y'])[0]
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                result=trainer.step_accumulated([rr])
                self.assertEqual(result['soft_weighted'],0.)
                self.assertEqual(result['consistency_lambda'],0.)
                self.assertTrue(_same(a,stage.aligner.state_dict()))
                self.assertTrue(_same(t,stage.teacher.state_dict()))
                self.assertFalse(_same(current,stage.student.state_dict()))
                for key in ('pan_input','ms_base_frame','loss_mask'):
                    torch.testing.assert_close(stage.frozen_predictions(rr)[key],frozen_predictions(stage.aligner,stage.teacher,rr,geometry_key='s100')[key],rtol=0,atol=0)

    def test_zero_and_positive_gamma_complete_state_resume(self):
        for case in ('S100DIFFUBILC30AG000','S100DIFFUBILC30G100'):
            trainer=training_fixture(case);stream=MemoryStream(0);advance(trainer,stream)
            pins={'code_pin':CODE_PIN,'evaluator_pin':EVALUATOR_PIN}
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'step1.pt'
                artifact=save_checkpoint(path,trainer,stream,**pins)
                resumed=training_fixture(case);other=MemoryStream(0)
                load_checkpoint(path,artifact['file_sha256'],resumed,other,**pins)
                advance(trainer,stream);advance(resumed,other)
                self.assertTrue(_same(trainer.stage.student.state_dict(),resumed.stage.student.state_dict()))
                self.assertTrue(_same(trainer.optimizer.state_dict(),resumed.optimizer.state_dict()))
                self.assertEqual(stream.state_dict(),other.state_dict())

    def test_soft_loss_cannot_be_enabled_and_training_timing_separates_q_lookup(self):
        from dataclasses import replace
        from panda_hard36.runtime import Campaign
        for case in CASES:
            cfg=plan.resolve_case('s1',case)
            spec=MatureSpec(**cfg['stage2_spec'])
            with self.assertRaises(ValueError): replace(spec,beta=.05)
        zero=plan.resolve_case('s1','S100DIFFUBILC30AG000')
        e=plan.resolve_case('s1','S100DIFFUBILC30G000')
        q=plan.resolve_case('s1','S100DIFFUBILC30G100')
        self.assertEqual(Campaign.timing_key(zero),Campaign.timing_key(e))
        self.assertNotEqual(Campaign.timing_key(zero),Campaign.timing_key(q))
