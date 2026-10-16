from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from directshift48.training import DirectStage1Trainer
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.tests.test_training_evaluation import (
    EvalSources, metrics, stage_and_trainer, rr_batch)
from s2dshift48.tests.test_stage1 import build
from panda40.tests.test_reporting import config
from panda40 import evaluation as E


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_direct_s2d_and_stage2_same_native_protocol(self):
        for kind in ('direct', 's2d', 'stage2'):
            with self.subTest(kind=kind):
                if kind == 'stage2':
                    trainer = stage_and_trainer()
                    trainer.step(rr_batch())
                    cfg = config('s3', 'stage2')
                else:
                    stage = build(sensor='WV3', baseline=kind == 'direct',
                                  lambda_abs=3., lambda_eps=.003)
                    trainer = DirectStage1Trainer(stage, torch.optim.AdamW(stage.parameters(), lr=1e-4),
                                                  gradient_log_interval=100)
                    trainer.updates = 1
                    cfg = config()
                    cfg['aligner_kind'] = kind
                cfg.pop('alignment_protocol_digest', None)
                trainer.config = cfg
                trainer.evaluator_pin = {}
                sources, funcs, refs = EvalSources(), metrics(), []
                original_rr = funcs['rr']
                def record_rr(raw, arrays):
                    np.testing.assert_array_equal(arrays['pan'], sources.raw('rr', [len(refs)])['pan'][0])
                    self.assertEqual(raw.shape, (8, 64, 64))
                    refs.append(1)
                    return original_rr(raw, arrays)
                funcs['rr'] = record_rr
                before = E.model_state_hash(trainer.stage)
                result = E.official_evaluate(trainer, sources, metric_functions=funcs, evaluator_revision='TEST')
                self.assertEqual(before, E.model_state_hash(trainer.stage))
                self.assertEqual(len(refs), 20)
                self.assertTrue(result['qualification']['same_state_verified'])
                self.assertTrue(result['qualification']['metrics_complete'])
                self.assertFalse(result['full_method_eligible'])
                self.assertFalse(result['qualification']['joint_pass'])
                self.assertEqual(result['state_before'], result['state_after'])
                self.assertEqual(result['rr']['per_scene'][0]['alignment_kind'], type(trainer.stage.aligner).__name__)

    def test_mask_and_aligner_mismatch_rejected(self):
        trainer = stage_and_trainer()
        trainer.config = config('s3', 'stage2')
        trainer.config.pop('alignment_protocol_digest', None)
        trainer.config['loss_mask_policy'] = 'silently_changed'
        with self.assertRaisesRegex(ValueError, 'loss mask'):
            E.configured_pipeline(trainer, EvalSources())
        stage = build(sensor='WV3', lambda_abs=3., lambda_eps=.003)
        trainer = DirectStage1Trainer(stage, torch.optim.AdamW(stage.parameters(), lr=1e-4),
                                      gradient_log_interval=100)
        trainer.config = config()
        with self.assertRaisesRegex(ValueError, 'Aligner architecture'):
            E.configured_pipeline(trainer, EvalSources())

    def test_evaluation_mutation_poisoned_and_no_injection_production(self):
        trainer = stage_and_trainer(); trainer.config = config('s3', 'stage2')
        trainer.config.pop('alignment_protocol_digest', None)
        with self.assertRaisesRegex(ValueError, 'synthetic TEST'):
            E.official_evaluate(trainer, EvalSources(), metric_functions=metrics())
        with self.assertRaisesRegex(RuntimeError, 'changed during paused'):
            with E.paused_model(trainer):
                trainer.stage.student.backbone.output[-1].bias.add_(1.)
        self.assertTrue(trainer.poisoned)


if __name__ == '__main__':
    unittest.main()
