"""Full-frame official-reference invariants, tested without GPU or cloud."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch

from panda_stage2_condq.tests.test_training_evaluation import (
    EvalSources, metrics, stage_and_trainer, rr_batch)
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.training import StudentTrainer
from panda_stage2_condq.spec import MASK2_POLICY
from panda30r.tests.test_reporting import config
from panda30r import evaluation as E


def masked_trainer():
    source = stage_and_trainer()
    stage = ConditionalQStage2(replace(source.stage.spec, mask_policy=MASK2_POLICY),
        deepcopy(source.stage.aligner), donor_binding=source.stage.donor_binding)
    trainer = StudentTrainer(stage, source.training, evaluation_pin={})
    trainer.config = config('s3', 'stage2')
    trainer.config.pop('alignment_protocol_digest', None)
    return trainer


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_mask_training_only_full20_unchanged_original_references(self):
        trainer = masked_trainer(); trainer.step(rr_batch())
        sources, funcs, observed = EvalSources(), metrics(), {'rr':0,'fr':0}
        for split in ('rr','fr'):
            original = funcs[split]
            def check(raw, arrays, split=split, original=original):
                self.assertEqual(raw.shape, (8,64,64))
                np.testing.assert_array_equal(arrays['pan'],sources.raw(split,[observed[split]])['pan'][0])
                observed[split] += 1
                return original(raw,arrays)
            funcs[split] = check
        before = E.model_state_hash(trainer.stage)
        result = E.official_evaluate(trainer,sources,metric_functions=funcs,evaluator_revision='TEST')
        self.assertEqual(observed,{'rr':20,'fr':20})
        self.assertEqual(before,E.model_state_hash(trainer.stage))
        self.assertTrue(result['qualification']['same_state_verified'])
        self.assertTrue(result['qualification']['metrics_complete'])
        self.assertFalse(result['full_method_eligible'])
        self.assertFalse(result['qualification']['joint_pass'])
        self.assertEqual(result['state_before']['engine_kind'],'stage2_condq')

    def test_wrong_mask_engine_and_source_rejected(self):
        for key,value,pattern in (('loss_mask_policy','FULL','MASK2'),
                                  ('engine_kind','mature_teacher','phase')):
            trainer = masked_trainer(); trainer.config[key] = value
            with self.assertRaisesRegex(ValueError,pattern):
                E.configured_pipeline(trainer,EvalSources())
        trainer = masked_trainer()
        wrong = EvalSources(); wrong.sensor = 'GF2'
        with self.assertRaisesRegex(ValueError,'sensor'):
            E.configured_pipeline(trainer,wrong)

    def test_fresh_direct_fallback_uses_original_mask2_native_graph(self):
        from directshift48.training import DirectStage1Trainer
        from s2dshift48.tests.test_stage1 import build
        stage = build(sensor='WV3', baseline=True, lambda_abs=3., lambda_eps=.003)
        trainer = DirectStage1Trainer(stage,torch.optim.AdamW(stage.parameters(),lr=1e-4),
                                      gradient_log_interval=100)
        trainer.updates=1
        trainer.config=config('s1','donor')
        trainer.config.update(engine_kind='fresh_donor')
        trainer.config.pop('alignment_protocol_digest',None)
        trainer.evaluator_pin={}
        value=E.official_evaluate(trainer,EvalSources(),metric_functions=metrics(),evaluator_revision='TEST')
        self.assertTrue(value['qualification']['same_state_verified'])
        self.assertFalse(value['full_method_eligible'])
        self.assertEqual(value['state_before']['engine_kind'],'fresh_donor')

    def test_injected_metrics_are_never_production(self):
        trainer = masked_trainer()
        with self.assertRaisesRegex(ValueError,'synthetic TEST'):
            E.official_evaluate(trainer,EvalSources(),metric_functions=metrics())

    def test_imported_donor_replay_is_zero_local_and_not_new_run_evidence(self):
        from panda30r.engine import _DonorReplay
        from panda_stage2_condq.donor import LoadedDonor
        from panda_method.calibration import state_dict_sha256
        base=stage_and_trainer()
        aligner=deepcopy(base.stage.aligner).requires_grad_(False).eval()
        teacher=deepcopy(base.stage.student).requires_grad_(False).eval()
        binding=deepcopy(base.stage.donor_binding)
        binding.update(teacher_state_sha256=state_dict_sha256(teacher),donor_step=7000,
                       donor_checkpoint_sha256='c'*64)
        donor=LoadedDonor(aligner,teacher,binding,
                         {'source_pipeline':base.stage.pipeline_manifest()})
        cfg=config('s1','donor')
        cfg.update(engine_kind='donor_replay',start_update=7000)
        cfg.pop('alignment_protocol_digest',None)
        trainer=_DonorReplay(donor,cfg,{})
        result=E.official_evaluate(trainer,EvalSources(),metric_functions=metrics(),evaluator_revision='TEST')
        self.assertEqual(result['state_before']['completed_update'],0)
        self.assertEqual(result['state_before']['lifetime_update'],7000)
        self.assertTrue(result['qualification']['same_state_verified'])
        self.assertFalse(result['full_method_eligible'])

    def test_native_maturity_and_tail_before_after_freeze_full20_context(self):
        from panda30r.tests.test_engine import fixture
        for kind,update in (('mature_teacher',1),('tail_fc2_eps',1),('tail_fc2_abs',5000)):
            with self.subTest(kind=kind,update=update):
                trainer=fixture(kind,update=update)
                # Add production reporting metadata to the explicitly synthetic
                # imported-moments fixture before sealing its test contract.
                cfg=config('s3','donor' if kind=='mature_teacher' else 'stage2')
                for key in ('start_update','planned_horizon','training_spec','epsilon_seed','seed'):
                    cfg[key]=deepcopy(trainer.config[key])
                cfg.update(engine_kind=kind,run_id='P30R_TEST_'+kind)
                cfg.pop('alignment_protocol_digest',None)
                trainer.config=cfg
                trainer._initial_contract=trainer.contract()
                trainer.assert_contract()
                flags=[p.requires_grad for p in trainer.stage.parameters()]
                rng=trainer.stage._epsilon_rng.get_state().clone()
                optimizer=E._tree_hash(trainer.optimizer.state_dict())
                value=E.official_evaluate(trainer,EvalSources(),metric_functions=metrics(),evaluator_revision='TEST')
                self.assertTrue(value['qualification']['same_state_verified'])
                self.assertTrue(value['qualification']['metrics_complete'])
                self.assertEqual(value['state_before']['lifetime_update'],cfg['start_update']+update)
                self.assertEqual(flags,[p.requires_grad for p in trainer.stage.parameters()])
                self.assertTrue(torch.equal(rng,trainer.stage._epsilon_rng.get_state()))
                self.assertEqual(optimizer,E._tree_hash(trainer.optimizer.state_dict()))
                trainer.assert_contract()

    def test_paused_context_restores_grad_flags_and_poisons_weight_mutation(self):
        trainer = masked_trainer()
        flags = [p.requires_grad for p in trainer.stage.parameters()]
        with E.paused_model(trainer):
            trainer.stage.aligner.fc2.bias.requires_grad_(True)
        self.assertEqual(flags,[p.requires_grad for p in trainer.stage.parameters()])
        with self.assertRaisesRegex(RuntimeError,'changed during paused'):
            with E.paused_model(trainer):
                trainer.stage.student.backbone.output[-1].bias.add_(1.)
        self.assertTrue(trainer.poisoned)


if __name__ == '__main__':
    unittest.main()
