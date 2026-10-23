"""Execute each new sampler through unchanged Hard training and exact resume."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from donor30.artifacts import _same
from donor30.masking import masked_mean
from directshift48.method import ScharrAbsolute
from panda_band30.tests.test_method import make_donor
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_sampling30 import plan, calibration as C, reporting as R
from panda_sampling30.contract import CASES, sampling_manifest
from panda_sampling30.engine import ScheduleTrainingSpec, MatureStudentTrainer
from panda_sampling30.method import MatureStage2, frozen_predictions
from panda_sampling30.spec import MatureSpec


def fixture(case):
    p = CASES[case]
    spec = MatureSpec('WV3','panda_condq',345,p['alpha'],0.,p['gamma'],1e-6,
                     policy_id=p['policy_id'],total_updates=30000,sampling_variant=case)
    donor = C.bind_donor(make_donor(shift=(1.,-.5)), spec)
    rr = batch(); key = spec.geometry_keys[0]
    frozen = frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
    e,g,q = reference_maps(ScharrAbsolute('WV3'),frozen['teacher_y'],rr.gt,frozen['pan_loss'])
    builder = CalibrationBuilder(['WV3:train:0'], C.binding_for_geometry(donor.binding,key),
        k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    builder.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=frozen['loss_mask'])
    stage = MatureStage2(spec,donor.aligner,donor.teacher,donor_binding=donor.binding,
                        stats_bank={key:builder.finalize()})
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    training = ScheduleTrainingSpec(lr=1e-4,total_updates=30000,microbatch_size=1,effective_batch_size=1)
    return MatureStudentTrainer(stage,training)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_each_sampling_variant_keeps_hard_loss_frozen_teacher_and_exact_resume(self):
        for case in CASES:
            with self.subTest(case=case):
                trainer = fixture(case); stage = trainer.stage; rr = batch()
                with patch.object(stage.student.backbone,'forward',wraps=stage.student.backbone.forward) as forward:
                    out = stage(rr, return_maps=True)
                self.assertEqual(forward.call_count, 1)
                maps = out['maps']; w = 1+.25*maps['d_e']+.5*maps['d_e']*maps['r_q_bar']
                expected = masked_mean(w*(out['student']['y']-rr.gt).abs().mean(1,keepdim=True),out['loss_mask'])
                torch.testing.assert_close(out['loss'],expected,rtol=0,atol=0)
                self.assertEqual(stage.deployment_manifest()['student_sampling'],sampling_manifest(case))
                self.assertEqual(sum(p.numel() for p in stage.student.parameters()),sampling_manifest(case)['student_parameters'])
                trainer.step_accumulated([rr])
                self.assertTrue(all(p.grad is None for m in (stage.aligner,stage.teacher) for p in m.parameters()))
                saved = trainer.state_dict(); trainer.step_accumulated([rr])
                expected = deepcopy(stage.student.state_dict())
                trainer.load_state_dict(saved); trainer.step_accumulated([rr])
                self.assertTrue(_same(stage.student.state_dict(),expected))
                self.assertEqual(trainer.updates, 2)

    def test_registry_is_fresh_architecture_only_and_reuses_paired_controls(self):
        from panda_schedule30 import plan as control
        before = {s:deepcopy(control.default_config(s)) for s in ('s1','s3')}
        for server in ('s1','s3'):
            self.assertEqual(len(plan.groups(server)),6)
            for g in plan.groups(server):
                for case,replica in g['runs']:
                    cfg=plan.resolve_case(server,case,replica); R._config(cfg)
                    base=control.resolve_case(server,'S020C30',replica)
                    for key in ('seed','stream_seeds','training_spec','augmentation','student_architecture','shift_scale'):
                        self.assertEqual(cfg[key],base[key],key)
                    self.assertEqual(cfg['initialization'],'FRESH')
                    self.assertEqual(cfg['comparison_control']['run_id'],base['run_id'])
                    self.assertEqual(cfg['rotation_consistency_weight'],0.)
                    self.assertEqual(cfg['stage2_spec']['beta'],0.)
                    changed=deepcopy(cfg); changed['student_sampling']=sampling_manifest('BASE')
                    with self.assertRaises(ValueError): R._config(changed)
            self.assertEqual(control.default_config(server),before[server])
        with self.assertRaises(ValueError): ScheduleTrainingSpec(inherited_updates=25000)


if __name__ == '__main__':
    unittest.main()
