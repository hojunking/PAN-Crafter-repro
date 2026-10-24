"""Full-scale queue contrasts and real single-model Hard updates on CPU."""
from copy import deepcopy
import csv
import io
from pathlib import Path
import unittest

import torch

from donor30.artifacts import _same
from panda_follow30 import plan
from panda_follow30.contract import CASES, CAMPAIGN_ID, NEXT_CAMPAIGN_ID
from panda_follow30.engine import MatureStudentTrainer, ScheduleTrainingSpec
from panda_follow30.method import MatureStage2
from panda_follow30.spec import MatureSpec
from panda_next30.tests.test_core import fixture
from panda_stage2_condq.tests.test_method import batch


class RegisteredCaseTests(unittest.TestCase):
    def test_every_case_is_full_scale_without_new_loss_or_averaging(self):
        self.assertEqual(len(CASES),8)
        for server, count, updates in [('s1',10,310000),('s3',12,450000)]:
            configs=[plan.resolve_case(server,c,r) for g in plan.groups(server) for c,r in g['runs']]
            self.assertEqual(len(configs),count)
            self.assertEqual(sum(c['continuation_updates'] for c in configs),updates)
            for cfg in configs:
                self.assertEqual(cfg['shift_scale'],1.)
                self.assertEqual(cfg['policy_id'],'S100')
                self.assertEqual(cfg['student_warp']['method'],'lanczos4')
                self.assertEqual(cfg['student_views'],1)
                self.assertEqual(cfg['stage2_spec']['beta'],0.)
                self.assertEqual(cfg['rotation_consistency_weight'],0.)
                self.assertFalse(cfg['inference_tta'])
                self.assertEqual(cfg['model_variant'],'RAW')
                self.assertEqual(cfg['weight_averaging'],'EXCLUDED_USER_DECISION_20261024')
                self.assertNotIn('swa',cfg)

    def test_lr_hard_sampling_and_input_comparisons_change_only_intended_knobs(self):
        pairs=[('S100DIFFC50','S100DIFFLR150C50',{'lr_S'}),
               ('S100DIFFC50','S100DIFFH050C50',{'alpha','gamma'}),
               ('S100DIFFC30','S100DIFFUBILC30',{'sampling_variant'}),
               ('S100C30','S100UBILC30',{'sampling_variant'}),
               ('S100C30','S100DIFFC30',{'input_variant'}),
               ('S100DIFFC30','S100DIFFC50',{'total_updates'})]
        for left,right,expected in pairs:
            with self.subTest(pair=(left,right)):
                self.assertEqual({k for k in CASES[left] if CASES[left][k]!=CASES[right][k]},expected)

    def test_comparison_references_are_registered_same_seed(self):
        from panda_next30 import plan as previous
        for server in ('s1','s3'):
            for group in plan.groups(server):
                for case,replica in group['runs']:
                    cfg=plan.resolve_case(server,case,replica)
                    control=cfg['comparison_control']
                    other=plan if control['campaign_id']==CAMPAIGN_ID else previous
                    reference=other.resolve_case(server,control['case'],replica)
                    self.assertEqual(reference['seed'],cfg['seed'])
                    self.assertEqual(reference['server'],server)

    def test_continuations_bind_exact_next30_same_seed_15k_state(self):
        from panda_next30 import plan as previous
        from panda_next30.engine import ScheduleTrainingSpec as ParentTrainingSpec
        for server in ('s1','s3'):
            for replica in ('R1','R2'):
                cfg=plan.resolve_case(server,'S100DIFFF15C30',replica)
                parent=previous.resolve_case(server,'S100DIFFC30',replica)
                self.assertEqual(cfg['source_run_id'],parent['run_id'])
                self.assertEqual(cfg['source_campaign_id'],NEXT_CAMPAIGN_ID)
                self.assertEqual(cfg['source_completed_update'],15000)
                self.assertTrue(str(plan.source_checkpoint_path(cfg)).endswith('step_0015000.pt'))
                schedule=ScheduleTrainingSpec(**cfg['training_spec'])
                original=ParentTrainingSpec(**parent['training_spec'])
                self.assertEqual(schedule.rate(15000),original.rate(15000))
                self.assertEqual(schedule.rate(15001),2e-5)
                self.assertAlmostEqual(schedule.rate(30000),1e-6)

    def test_r3_is_paired_full_scale_diff_vs_corrected_only(self):
        for server in ('s1','s3'):
            diff=plan.resolve_case(server,'S100DIFFC30','R3')
            base=plan.resolve_case(server,'S100C30','R3')
            earlier=plan.resolve_case(server,'S100DIFFC50','R1')
            self.assertEqual(diff['seed'],261018103)
            self.assertEqual(diff['stream_seeds'],base['stream_seeds'])
            self.assertNotEqual(diff['stream_seeds'],earlier['stream_seeds'])
            self.assertEqual(diff['shift_scale'],base['shift_scale'])

    def test_catalogs_match_executable_queues(self):
        for server in ('s1','s3'):
            with (Path(plan.__file__).parent/f'cases_{server}.csv').open(newline='') as handle:
                saved=list(csv.DictReader(handle))
            self.assertEqual(saved,list(csv.DictReader(io.StringIO(plan.catalog_csv(server)))))


class HardExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_new_hard_and_corrected_only_ubil_update_and_resume(self):
        for server,case in [('s3','S100DIFFH050C50'),('s1','S100UBILC30')]:
            with self.subTest(case=case):
                cfg=plan.resolve_case(server,case,'R1')
                template=fixture('S100DIFFC30')
                stage=MatureStage2(MatureSpec(**cfg['stage2_spec']),deepcopy(template.stage.aligner),
                    deepcopy(template.stage.teacher),donor_binding=template.stage._donor_binding,
                    stats_bank=template.stage.stats_bank)
                trainer=MatureStudentTrainer(stage,ScheduleTrainingSpec(**{
                    **cfg['training_spec'],'microbatch_size':1,'effective_batch_size':1}))
                trainer.config=cfg
                fixed_a=deepcopy(stage.aligner.state_dict());fixed_t=deepcopy(stage.teacher.state_dict())
                first=trainer.step_accumulated([batch()])
                saved=trainer.state_dict()
                trainer.step_accumulated([batch()])
                expected=deepcopy(stage.student.state_dict())
                trainer.load_state_dict(saved)
                trainer.step_accumulated([batch()])
                self.assertTrue(_same(expected,stage.student.state_dict()))
                self.assertTrue(_same(fixed_a,stage.aligner.state_dict()))
                self.assertTrue(_same(fixed_t,stage.teacher.state_dict()))
                self.assertEqual(first['loss'],first['hard_weighted'])
                self.assertEqual(first['soft_weighted'],0.)
                self.assertEqual(first['student_views'],1)
                self.assertNotIn('swa',saved)


if __name__=='__main__':
    unittest.main()
