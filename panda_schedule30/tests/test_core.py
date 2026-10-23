"""Schedule endpoints, genuine PAN warp/Hard loss, and full-state fork invariants."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from donor30.artifacts import _same, capture_rng
from donor30.masking import masked_mean
from directshift48.method import ScharrAbsolute
from panda_band30.tests.test_method import make_donor
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.training import TrainingSpec, rng_state
from panda_scale30 import plan as old_plan
from panda_scale30.tests.test_method_core import make_trainer as old_trainer
from panda_schedule30 import plan, calibration as C
from panda_schedule30.contract import CASES, BUDGET
from panda_schedule30.engine import ScheduleTrainingSpec, MatureStudentTrainer
from panda_schedule30.fork import _transfer_validated
from panda_schedule30.method import MatureStage2, frozen_predictions
from panda_schedule30.spec import MatureSpec, policy_manifest


def fixture(case='S020C30'):
    p=CASES[case]
    spec=MatureSpec('WV3','panda_condq',345,p['alpha'],0.,p['gamma'],1e-6,
                   policy_id=p['policy_id'],total_updates=p['total_updates'])
    donor=C.bind_donor(make_donor(shift=(1.,-.5)),spec)
    rr=batch(); key=spec.geometry_keys[0]
    f=frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
    e,g,q=reference_maps(ScharrAbsolute('WV3'),f['teacher_y'],rr.gt,f['pan_loss'])
    builder=CalibrationBuilder(['WV3:train:0'],C.binding_for_geometry(donor.binding,key),
          k=5,memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    builder.add(['WV3:train:0'],e,g,q,sensor='WV3',loss_mask=f['loss_mask'])
    stage=MatureStage2(spec,donor.aligner,donor.teacher,donor_binding=donor.binding,
                      stats_bank={key:builder.finalize()})
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    training=ScheduleTrainingSpec(lr=p['lr_S'],total_updates=p['total_updates'],
        min_lr=p['min_lr'],warmup_updates=0 if p['inherited_updates'] else 100,
        inherited_updates=p['inherited_updates'],microbatch_size=1,effective_batch_size=1)
    return MatureStudentTrainer(stage,training)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_registered_queues_do_not_mutate_old_campaign(self):
        original={s:deepcopy(old_plan.default_config(s)) for s in ('s1','s3')}
        self.assertIsNone(BUDGET['total_seconds'])
        for server in ('s1','s3'):
            for group in plan.groups(server):
                for case,replica in group['runs']:
                    cfg=plan.resolve_case(server,case,replica); row=CASES[case]
                    self.assertEqual(cfg['planned_horizon'],row['total_updates'])
                    self.assertEqual(cfg['geometry']['total_updates'],row['total_updates'])
                    self.assertEqual(cfg['evaluation_steps'][-1],row['total_updates'])
                    self.assertTrue(all(s>row['inherited_updates'] for s in cfg['evaluation_steps']))
                    self.assertIn(cfg['shift_scale'],(.1,.2))
                    self.assertEqual(cfg['student_views'],1)
                    self.assertEqual(cfg['rotation_consistency_weight'],0.)
                    self.assertEqual(cfg['stage2_spec']['beta'],0.)
                    if row['source_case']:
                        self.assertEqual(cfg['continuation_updates'],10000)
                    spec=MatureSpec(**cfg['stage2_spec'])
                    self.assertEqual(policy_manifest(spec),cfg['geometry'])
                    with self.assertRaises(ValueError): spec.geometry_at_next(row['total_updates'])
            self.assertEqual(plan.default_config(server)['joint_target'],{'HQNR':.9585,'ERGAS':2.040})
            self.assertEqual(old_plan.default_config(server),original[server])

    def test_cosine_is_rescaled_not_50k_early_stop(self):
        short=ScheduleTrainingSpec(total_updates=30000)
        old=TrainingSpec(total_updates=50000)
        self.assertEqual(short.rate(100),1e-4)
        self.assertEqual(short.rate(30000),0.)
        self.assertGreater(old.rate(30000),3e-5)
        self.assertLess(short.rate(25000),old.rate(25000))
        tail=ScheduleTrainingSpec(lr=2e-5,min_lr=1e-6,total_updates=35000,warmup_updates=0,inherited_updates=25000)
        self.assertEqual(tail.rate(25000),old.rate(25000))
        self.assertEqual(tail.rate(25001),2e-5)
        self.assertEqual(tail.rate(35000),1e-6)
        rates=[tail.rate(i) for i in (25001,26000,30000,34000,35000)]
        self.assertEqual(rates,sorted(rates,reverse=True))
        with self.assertRaises(ValueError): tail.rate(35001)
        with self.assertRaises(ValueError): replace(tail,inherited_updates=20000)

    def test_single_view_gt_hard_and_frozen_donor_preserved(self):
        trainer=fixture(); rr=batch(); stage=trainer.stage
        with patch.object(stage.student.backbone,'forward',wraps=stage.student.backbone.forward) as f:
            out=stage(rr,return_maps=True)
        self.assertEqual(f.call_count,1)
        maps=out['maps']; w=1+.25*maps['d_e']+.5*maps['d_e']*maps['r_q_bar']
        expected=masked_mean(w*(out['student']['y']-rr.gt).abs().mean(1,keepdim=True),out['loss_mask'])
        torch.testing.assert_close(out['loss'],expected,rtol=0,atol=0)
        self.assertNotIn('student_rotated',out)
        trainer.step_accumulated([rr])
        self.assertEqual(trainer.updates,1)
        self.assertEqual(stage.completed_update,1)
        self.assertTrue(all(p.grad is None for m in (stage.aligner,stage.teacher) for p in m.parameters()))
        state=trainer.state_dict(); trainer.step_accumulated([rr]); expected=deepcopy(stage.student.state_dict())
        trainer.load_state_dict(state); trainer.step_accumulated([rr])
        self.assertTrue(_same(stage.student.state_dict(),expected))

    def test_explicit_fork_keeps_moments_rng_stream_and_rejects_changed_objective(self):
        source=old_trainer(); source.step_accumulated([batch()])
        # Synthetic state at a declared source boundary; production additionally
        # validates the real source file/hash and full sampler progress.
        source.updates=25000; source.stage.set_completed_update(25000)
        source.optimizer.param_groups[0]['lr']=source.training.rate(25000)
        for value in source.optimizer.state.values(): value['step'].fill_(25000)
        # Source fixture has warmup0; use same original warmup100 as production.
        source.training=replace(source.training,warmup_updates=100)
        source.optimizer.param_groups[0]['lr']=source.training.rate(25000)
        target=fixture('S020F25C35')
        class Stream:
            def state_dict(self): return {'exact_sample_order':[8,2,1],'cursor':3}
        stream=Stream(); payload={'stream':stream.state_dict(),'rng':capture_rng()}
        before=deepcopy(source.optimizer.state_dict())
        receipt=_transfer_validated(source,target,stream,payload)
        self.assertTrue(_same(target.optimizer.state_dict(),before))
        self.assertEqual(target.updates,25000)
        self.assertEqual(receipt['next_lr'],2e-5)
        target.step_accumulated([batch()])
        self.assertEqual(target.updates,25001)
        self.assertEqual(target.optimizer.param_groups[0]['lr'],2e-5)
        self.assertTrue(all(int(v['step'])==25001 for v in target.optimizer.state.values()))
        state=target.state_dict(); target.step_accumulated([batch()]); expected=deepcopy(target.stage.student.state_dict())
        target.load_state_dict(state); target.step_accumulated([batch()])
        self.assertTrue(_same(target.stage.student.state_dict(),expected))
        with self.assertRaises(ValueError): _transfer_validated(source,target,stream,payload)


if __name__=='__main__': unittest.main()
