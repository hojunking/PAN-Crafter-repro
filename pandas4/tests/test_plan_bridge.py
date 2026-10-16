"""Real registered s4 configs against adapters, using CPU-only synthetic data.

Metric injection is always TEST/ineligible. These checks never access Google,
produce a publishable cost, or launch a production experiment.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from directshift48.method import DirectPANAligner
from pa.warp import warp_support_mask
from panda40.stage1 import build_stage1
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from panda_stage2_condq.tests.test_method import fixture
from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics, rr_batch
from pandas4 import plan, cost, evaluation as E, reporting as R, publication as P


def stage2_trainer(cfg):
    """Exercise real method coefficients; replace only population/training size."""
    spec = Stage2Spec(**cfg['stage2_spec'])
    donor = fixture('panda_condq')
    stats = None
    if spec.variant != 'gt_only':
        ids = ['WV3:train:0']
        collector = CalibrationBuilder(ids, donor.donor_binding, k=5,
            memory_budget_bytes=4*1024**2, include_condq=spec.variant == 'panda_condq',
            mask_policy=spec.mask_policy)
        error = torch.full((1, 1, 64, 64), .25)
        zeros = torch.zeros_like(error) if spec.variant == 'panda_condq' else None
        support = (warp_support_mask(64, 64, torch.tensor([[1., -.5]]))
                   if spec.mask_policy == MASK2_POLICY else None)
        collector.add(ids, error, zeros, zeros, sensor='WV3', loss_mask=support)
        stats = collector.finalize()
    stage = ConditionalQStage2(spec, donor.aligner,
                              None if spec.variant == 'gt_only' else donor.teacher,
                              donor_binding=donor.donor_binding, stats=stats)
    trainer = StudentTrainer(stage, TrainingSpec(effective_batch_size=1,
        microbatch_size=1, total_updates=20, warmup_updates=0), evaluation_pin={})
    trainer.config = deepcopy(cfg)
    return trainer


class PlanBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_registered_cases_admit_actual_stage_and_cost_contracts(self):
        source = EvalSources()
        signatures, initial_students, bindings = [], [], []
        cases = [('D_A1', 'R1'), ('D_A1', 'R2'), ('D_A3', 'R1')]
        cases += [(case, 'R1') for case in plan.STAGE2]
        for case, replica in cases:
            with self.subTest(case=case, replica=replica):
                cfg = plan.resolve_case('s4', case, replica)
                self.assertEqual(R._config(cfg), cfg)
                signature = cost.signature(cfg)
                self.assertEqual(signature['server'], 's4')
                self.assertEqual(signature['aligner'], 'DirectPANAligner')
                if cfg['phase'] == 'donor':
                    trainer = build_stage1(cfg, source)
                    trainer.updates = 1  # Synthetic inference-only state, never production evidence.
                    self.assertEqual(trainer.stage.lambda_abs, cfg['lambda_abs'])
                    self.assertEqual(trainer.stage.lambda_eps, cfg['lambda_epsilon'])
                    self.assertIs(type(trainer.stage.aligner), DirectPANAligner)
                else:
                    trainer = stage2_trainer(cfg)
                    self.assertEqual(trainer.stage.spec.manifest()['loss_mask_policy'], cfg['loss_mask_policy'])
                    for key in ('variant', 'alpha', 'beta', 'gamma', 'seed', 'mask_policy'):
                        self.assertEqual(getattr(trainer.stage.spec, key), cfg['stage2_spec'][key])
                    initial_students.append(trainer.stage.student_initial_state_sha256)
                    bindings.append(R.digest(trainer.stage.donor_binding))
                    signatures.append(signature)
                    result = trainer.step(rr_batch())
                    self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
                    self.assertTrue(all(p.grad is None for p in trainer.stage.aligner.parameters()))
                trainer.evaluator_pin = {}
                self.assertTrue(E.configured_pipeline(trainer, source))
                identity = E.model_identity(trainer, source, evaluator_revision='TEST')
                self.assertEqual(identity['campaign_id'], plan.CAMPAIGN_ID)
                self.assertEqual(identity['method_revision'], plan.METHOD_REVISION)
                self.assertEqual(identity['server'], 's4')
        # Loss variants, coefficients and support are training-only changes;
        # the deployed A+S graph and paired R1 Student initialization are fixed.
        self.assertEqual(len({R.digest(sig) for sig in signatures}), 1)
        self.assertEqual(len(set(initial_students)), 1)
        self.assertEqual(len(set(bindings)), 1)

    def test_real_plan_direct_masked_and_full_summaries_stay_test_only(self):
        for case in ('D_A1', 'D_A3', 'G', 'Q', 'Q_FULL'):
            with self.subTest(case=case):
                cfg = plan.resolve_case('s4', case)
                source = EvalSources()
                if cfg['phase'] == 'donor':
                    trainer = build_stage1(cfg, source)
                    trainer.updates = 1
                else:
                    trainer = stage2_trainer(cfg)
                    trainer.step(rr_batch())
                trainer.evaluator_pin = {}
                ev = E.official_evaluate(trainer, source, metric_functions=metrics(),
                                         evaluator_revision='TEST')
                self.assertTrue(ev['qualification']['same_state_verified'])
                self.assertTrue(ev['qualification']['metrics_complete'])
                self.assertFalse(ev['full_method_eligible'])
                self.assertFalse(ev['qualification']['joint_pass'])
                with tempfile.TemporaryDirectory() as root:
                    directory = Path(root)/'runs'/cfg['run_id']/'attempt_01'
                    R.atomic_json(directory/'config.json', cfg)
                    R.atomic_json(directory/'evaluation_0000001.json', ev)
                    R.emit_evaluation(root, cfg, ev)
                    view = P.project(root, 's4', require_cost=True)
                    self.assertTrue(view['conflicts'])
                    self.assertEqual(view['rows'], [])
                    self.assertIsNone(cost.find_profile(cfg, root))
