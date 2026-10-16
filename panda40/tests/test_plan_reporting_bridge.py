"""Actual plan adapters; synthetic smoke MUST remain ineligible for publication."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from panda40 import plan, cost, reporting as R, publication as P
from panda40.evaluation import official_evaluate, configured_pipeline
from panda40.stage1 import build_stage1
from panda_stage2_condq.tests.test_training_evaluation import (
    EvalSources, metrics, stage_and_trainer, rr_batch)
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.spec import Stage2Spec
from panda_stage2_condq.training import StudentTrainer, TrainingSpec


class PlanBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_production_plan_configs_match_evaluation_and_cost_adapters(self):
        for server, case in (('s1', 'D_A1'), ('s1', 'S_A1'), ('s3', 'G')):
            with self.subTest(case=case):
                cfg = plan.resolve_case(server, case)
                source = EvalSources()
                if server == 's1':
                    trainer = build_stage1(cfg, source)
                    trainer.updates = 1
                else:
                    donor = stage_and_trainer().stage
                    stage = ConditionalQStage2(Stage2Spec(**cfg['stage2_spec']),
                        donor.aligner, donor_binding=donor.donor_binding)
                    trainer = StudentTrainer(stage, TrainingSpec(effective_batch_size=1,
                        microbatch_size=1, total_updates=20, warmup_updates=0), evaluation_pin={})
                    trainer.step(rr_batch())
                    trainer.config = cfg
                trainer.evaluator_pin = {}
                self.assertEqual(R._config(cfg), cfg)
                self.assertEqual(cost.signature(cfg)['phase'], cfg['phase'])
                self.assertTrue(configured_pipeline(trainer, source))
                ev = official_evaluate(trainer, source, metric_functions=metrics(),
                                       evaluator_revision='TEST')
                self.assertFalse(ev['full_method_eligible'])
                self.assertFalse(ev['qualification']['full_method_eligible'])
                self.assertFalse(ev['qualification']['joint_pass'])
                # Persist the exact native output envelope. Do not change TEST
                # to a production digest or flip synthetic eligibility flags.
                with tempfile.TemporaryDirectory() as root:
                    directory = Path(root)/'runs'/cfg['run_id']/'attempt_01'
                    R.atomic_json(directory/'config.json', cfg)
                    R.atomic_json(directory/'evaluation_0000001.json', ev)
                    R.emit_evaluation(root, cfg, ev)
                    view = P.project(root, server, require_cost=True)
                    self.assertTrue(view['conflicts'])
                    self.assertEqual(view['rows'], [])
                    self.assertIsNone(cost.find_profile(cfg, root))

    def test_final_watcher_exit_requires_stable_terminal_readback(self):
        from panda40.tests.test_reporting import config, persist
        with tempfile.TemporaryDirectory() as root:
            cfg = config(); persist(root, cfg)
            state = {'status': 'FINISHED_REGISTERED_QUEUE', 'elapsed_seconds': 100.}
            R.atomic_json(Path(root)/'campaign_state.json', state)
            view = P.project_campaign(root, 's1')
            self.assertFalse(P.terminal_upload_complete(root, view, {}, state))
            self.assertTrue(P.terminal_upload_complete(root, view, {'readback_verified': True}, state))
            bad = deepcopy(view); bad['conflicts'].append('MISSING_COST')
            self.assertFalse(P.terminal_upload_complete(root, bad, {'readback_verified': True}, state))
            persist(root, cfg, 2000, .961, 2.04)
            self.assertFalse(P.terminal_upload_complete(root, view, {'readback_verified': True}, state))
            R.atomic_json(Path(root)/'campaign_state.json', {'status': 'RUNNING'})
            self.assertFalse(P.terminal_upload_complete(root, view, {'readback_verified': True}, state))

    def test_watcher_flushes_twice_before_terminal_exit_without_network(self):
        from panda40.tests.test_reporting import config, persist
        from panda40.tests.test_transport import FakeGoogle
        with tempfile.TemporaryDirectory() as root:
            cfg = config(); persist(root, cfg)
            R.atomic_json(Path(root)/'campaign_state.json', {'status': 'FINISHED_REGISTERED_QUEUE'})
            value = P.project_campaign(root, 's1')
            fake = FakeGoogle(value)
            with patch('sheet_publish_v2.transport.GoogleSheets', return_value=fake), \
                    patch('time.sleep') as sleep:
                code = P.publish(root, 's1', apply=True, watch=True, require_cost=False)
            self.assertEqual(code, 0)
            sleep.assert_called_once_with(120)
            self.assertEqual(sum(url.endswith(':append') for _, url, _ in fake.calls), 1)
            self.assertTrue((Path(root)/'sheet_publish_panda40'/'terminal_upload.json').is_file())


if __name__ == '__main__':
    unittest.main()
