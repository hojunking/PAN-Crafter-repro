"""Actual CPU Stage2 integration with synthetic H5 and explicitly mocked GPU cost.

No production data, GPU launch, sheet write or claimed numerical cost is used.
"""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from donor24 import plan as donor_plan
from donor24.evaluation import model_state_hash
from donor30.artifacts import SCHEMA as DONOR_SCHEMA
from donor30.runtime import build_trainer
from panda_stage2_condq.calibration import CalibrationStats
from panda_stage2_condq.checkpoint import load_checkpoint
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics
from ta2.evaluation import file_sha

from . import plan
from .runtime import Campaign, BudgetExpired
from .evaluation import official_evaluate


class Stage2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.temp.name)
        cls.registered = synthetic_sources(cls.fixture_root,count=12)
        cfg = donor_plan.resolve_case(donor_plan.default_config('s3'),'A1')
        trainer = build_trainer(cfg,cls.registered.sources)
        trainer.updates = 7  # import fixture only, never advertised as trained donor
        identity = {'model_state_sha256':model_state_hash(trainer.stage),
                    'completed_update':7,'run_id':cfg['run_id'],'dataset':'WV3'}
        payload = {'schema':DONOR_SCHEMA,'config':cfg,'trainer':trainer.state_dict(),
                   'identity':identity,'state_hash':identity['model_state_sha256']}
        cls.donor_path = cls.fixture_root/'synthetic_donor.pt'
        torch.save(payload,cls.donor_path)
        cls.donor_sha = file_sha(cls.donor_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp_run = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_run.cleanup)
        with patch('panda40.runtime.evaluator_identity',return_value={}), \
             patch('panda40.runtime.code_pin',return_value={'TEST':'a'*64}):
            self.campaign = Campaign('s3',output=self.temp_run.name,device='cpu')
        self.campaign.sources = self.registered.sources
        self.campaign.registered = self.registered
        self.campaign.state['donor'] = {'checkpoint_path':str(self.donor_path),
            'checkpoint_sha256':self.donor_sha,'exploratory':True}

    def fake_cost(self,cfg,sources,**kwargs):
        self.assertEqual(kwargs['cost_root'],self.campaign.root)
        self.assertIs(sources,self.registered.sources)
        trainer = kwargs['retained_trainer']
        self.assertTrue(all(p.device.type=='cpu' for p in trainer.stage.parameters()))
        if getattr(trainer.stage,'stats',None) is not None:
            self.assertEqual(trainer.stage.stats.device.type,'cpu')
        return {'TEST_ONLY_NO_NUMERIC_COST':True}

    def build(self,case):
        cfg = plan.resolve_case('s3',case)
        with patch('panda40.cost.ensure_cost',side_effect=self.fake_cost) as cost:
            trainer,stream = self.campaign.build(cfg)
            cost.assert_called_once()
        return cfg,trainer,stream

    def test_all_stage2_cases_build_with_same_frozen_donor_and_receipt_API(self):
        identities,initial = [],[]
        for case in ('G','E','ES','Q','Q_FULL','Q_WEAK'):
            with self.subTest(case=case):
                cfg,trainer,stream = self.build(case)
                trainer.assert_contract()
                identities.append(trainer.stage.donor_binding)
                initial.append(trainer.stage.student_initial_state_sha256)
                self.assertEqual(trainer.stage.spec.mask_policy,cfg['loss_mask_policy'])
                self.assertEqual(trainer.config,cfg)
                if case=='G':
                    self.assertIsNone(trainer.stage.teacher)
                    self.assertIsNone(trainer.stage.stats)
                    self.assertIsNone(cfg['calibration_sha256'])
                else:
                    receipt = json.loads(Path(cfg['calibration_path']).with_suffix('.json').read_text())
                    self.assertEqual(receipt['sha256'],receipt['file_sha256'])
                    self.assertEqual(receipt['sha256'],file_sha(cfg['calibration_path']))
                    self.assertEqual(trainer.stage.stats.mask_policy,cfg['loss_mask_policy'])
                    self.assertEqual(receipt['artifact_sha256'],trainer.stage.stats.artifact_sha256)
                del trainer,stream
        self.assertTrue(all(value==identities[0] for value in identities))
        self.assertEqual(len(set(initial)),1)
        masked = json.loads((self.campaign.root/'calibration'/'MASK2.json').read_text())
        full = json.loads((self.campaign.root/'calibration'/'FULL.json').read_text())
        self.assertLess(masked['n_e'],full['n_e'])
        self.assertEqual(masked['n_q'],full['n_q'])
        self.assertNotEqual(masked['artifact_sha256'],full['artifact_sha256'])

    def test_gt_only_skips_all_calibration_and_teacher_construction(self):
        with patch.object(self.campaign,'calibration',side_effect=AssertionError('GT calibration')), \
             patch('panda_stage2_condq.donor.build_reconstructor',side_effect=AssertionError('GT Teacher')):
            _,trainer,_ = self.build('G')
        trainer.assert_contract()

    def test_calibration_receipt_recovery_replay_and_tampering_fail_closed(self):
        path,receipt = self.campaign.calibration(plan.LOSS_MASK_POLICY)
        again_path,again = self.campaign.calibration(plan.LOSS_MASK_POLICY)
        self.assertEqual(path,again_path)
        self.assertEqual(receipt['artifact_sha256'],again['artifact_sha256'])
        path.with_suffix('.json').unlink()  # synthetic temporary test receipt only
        _,recovered = self.campaign.calibration(plan.LOSS_MASK_POLICY)
        self.assertTrue(recovered['recovered_complete_artifact'])
        self.assertEqual(recovered['sha256'],receipt['file_sha256'])
        corrupt = deepcopy(recovered); corrupt['mask_policy']=plan.MASK_POLICY
        path.with_suffix('.json').write_text(json.dumps(corrupt))
        with self.assertRaisesRegex(ValueError,'receipt donor/policy/content'):
            self.campaign.calibration(plan.LOSS_MASK_POLICY)

    def test_calibration_never_rebinds_an_artifact_to_changed_source_or_donor(self):
        self.campaign.calibration(plan.LOSS_MASK_POLICY)
        self.campaign.state['donor']['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            self.campaign.calibration(plan.LOSS_MASK_POLICY)

    def test_mask_setting_disagreement_fails_before_any_calibration_or_cost(self):
        cfg=plan.resolve_case('s3','Q')
        cfg['stage2_spec']['mask_policy']=plan.MASK_POLICY
        with patch.object(self.campaign,'calibration',side_effect=AssertionError('calibration ran')), \
             patch('panda40.cost.ensure_cost',side_effect=AssertionError('cost ran')):
            with self.assertRaisesRegex(ValueError,'mask policy mismatch'):
                self.campaign.build(cfg)

    def test_full_control_calibration_reserved_separately_and_budget_checked(self):
        self.assertEqual(self.campaign.calibration_overhead(plan.resolve_case('s3','G')),0.)
        self.assertGreater(self.campaign.calibration_overhead(plan.resolve_case('s3','Q')),0.)
        self.campaign.calibration(plan.LOSS_MASK_POLICY)
        self.assertEqual(self.campaign.calibration_overhead(plan.resolve_case('s3','Q')),0.)
        self.assertGreater(self.campaign.calibration_overhead(plan.resolve_case('s3','Q_FULL')),0.)
        self.campaign.config['budget']['total_seconds']=self.campaign.elapsed()+1900
        with patch('panda_stage2_condq.runtime.calibrate',side_effect=AssertionError('beyond budget')):
            with self.assertRaisesRegex(BudgetExpired,'complete FULL calibration'):
                self.campaign.calibration(plan.MASK_POLICY)
        self.assertFalse((self.campaign.root/'calibration'/'FULL.npz').exists())

    def test_cpu_48sample_update_eval_checkpoint_and_resume_no_external_side_effects(self):
        cfg,trainer,stream=self.build('Q')
        report,_=self.campaign.trace(trainer,stream)
        self.assertEqual(trainer.updates,1)
        self.assertGreater(report['loss'],0.)
        self.assertEqual(trainer.training.effective_batch_size,48)
        trainer.assert_contract()
        # Real A+S forwards, but explicit synthetic metrics and 64px fixtures.
        class Sources(EvalSources):
            def manifest_sha(inner,split):
                return self.registered.inventory.source_manifest_sha256
        evaluation=official_evaluate(trainer,Sources(),evaluator_pin={},
            metric_functions=metrics(),evaluator_revision='TEST',device='cpu')
        self.assertTrue(evaluation['qualification']['same_state_verified'])
        path=self.campaign.root/'synthetic_checkpoint.pt'
        # Tests the campaign save adapter without pretending injected metrics
        # qualify for retention/publication. Stage2 native checkpoint handles RNG.
        artifact=self.campaign._save(trainer,stream,path,cfg)
        _,fresh,fresh_stream=self.build('Q')
        restored=load_checkpoint(path,artifact['sha256'],fresh)
        fresh_stream.load_state_dict(restored['stream_state'])
        self.assertEqual(fresh.updates,1)
        self.assertEqual(stream.state_dict(),fresh_stream.state_dict())
        self.assertEqual(fresh.stage.get_extra_state(),trainer.stage.get_extra_state())
        self.assertFalse((self.campaign.root/'outbox').exists())


if __name__=='__main__':
    unittest.main()
