"""Offline integration of runtime/checkpoint/outbox contracts, never a launch."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from panda40 import plan, reporting
from panda40.runtime import Campaign, BudgetExpired
from panda40.stage1 import build_stage1, Stage1Stream, advance_stage1
from panda40.tests.test_reporting import evaluation as fixture_evaluation
from panda40.tests_stage1 import PINS
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_stage2_condq.tests.test_training_evaluation import stage_and_trainer, rr_batch
from panda40.evaluation import model_state_hash


def checked_fixture(trainer):
    result = fixture_evaluation(trainer.config, trainer.updates)
    for key in ('state_before', 'rr_state', 'fr_state', 'state_after'):
        result[key]['model_state_sha256'] = model_state_hash(trainer.stage)
    for split in ('rr', 'fr'):
        for i, row in enumerate(result[split]['per_scene']):
            row['raw_prediction_sha256'] = f'{i:064x}'
    result['elapsed_seconds'] = .01
    return result


class RuntimeCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = object.__new__(Campaign)
        self.campaign.root = self.root
        self.campaign.device = 'cpu'
        self.campaign.code = PINS['code_pin']
        self.campaign.evaluator_pin = PINS['evaluator_pin']

    def stage1(self, kind='direct'):
        registered = synthetic_sources(self.root, count=3)
        cfg = plan.resolve_case('s1','D_A1' if kind == 'direct' else 'S_A1')
        cfg['baseline_resolved'].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
        trainer = build_stage1(cfg,registered.sources)
        stream = Stage1Stream(registered.sources,cfg,device='cpu')
        advance_stage1(trainer,stream)
        return trainer,stream,cfg

    def test_both_stage1_saved_artifacts_validate_exact_publication_identity(self):
        for kind in ('direct','s2d'):
            trainer,stream,cfg = self.stage1(kind)
            evaluation = checked_fixture(trainer)
            artifact = self.campaign._save(trainer,stream,self.root/(kind+'.pt'),cfg,evaluation)
            self.assertEqual(artifact['identity'],evaluation['state_before'])
            record = reporting.make_record(cfg,evaluation,artifact,root=self.root)
            reporting.validate_record(record)
            self.assertTrue(record['source']['extra']['weights_saved'])
            self.assertEqual(record['row'][reporting.INDEX['Checkpoint_SHA']],artifact['file_sha256'])

    def test_stage2_full_checkpoint_schema_and_publication_identity(self):
        trainer = stage_and_trainer()
        trainer.config = plan.resolve_case('s3','G')
        trainer.step(rr_batch())
        stream_state = {'epoch':0,'cursor':1,'contract':'SYNTHETIC_STREAM_ONLY'}
        stream = SimpleNamespace(count=100,state_dict=lambda:deepcopy(stream_state))
        cfg,evaluation = trainer.config,checked_fixture(trainer)
        artifact = self.campaign._save(trainer,stream,self.root/'student.pt',cfg,evaluation)
        reporting.validate_record(reporting.make_record(cfg,evaluation,artifact,root=self.root))
        payload = torch.load(self.root/'student.pt',map_location='cpu',weights_only=False)
        self.assertEqual(payload['trainer']['stream_state'],stream_state)
        self.assertEqual(payload['trainer']['updates'],1)
        self.assertEqual(artifact['identity'],evaluation['state_before'])
        self.assertTrue(artifact['weights_saved'])
        self.assertFalse(artifact['predictions_reverified'])
        self.assertTrue((self.root/'student.json').exists())

    def test_save_rejects_partial_scene_list_before_creating_checkpoint(self):
        trainer,stream,cfg = self.stage1()
        evaluation = checked_fixture(trainer)
        evaluation['fr']['per_scene'].pop()
        path = self.root/'never.pt'
        with self.assertRaisesRegex(ValueError,'all20'):
            self.campaign._save(trainer,stream,path,cfg,evaluation)
        self.assertFalse(path.exists())

    def test_stage2_trace_restores_sampler_after_preoptimizer_failure(self):
        before = {'cursor':0}
        state = dict(before)
        def advance():
            state['cursor'] += 1
            return [object()]
        stream = SimpleNamespace(state_dict=lambda:dict(state),next_update=advance,
            load_state_dict=lambda value:state.update(value))
        trainer = SimpleNamespace(config={'phase':'stage2'},updates=0,poisoned=False,
            step_accumulated=lambda batches:(_ for _ in ()).throw(ValueError('pre-update')))
        with self.assertRaisesRegex(ValueError,'pre-update'):
            self.campaign.trace(trainer,stream)
        self.assertEqual(state,before)

    def test_preflight_disposes_checkpoint_and_validates_record_without_publishing(self):
        trainer,stream,cfg = self.stage1()
        self.campaign.state = {'timing':{}}
        self.campaign.check = lambda:None
        self.campaign.persist = lambda:None
        self.campaign.build = lambda config:(trainer,stream)
        self.campaign.trace = lambda *args:({},.01)
        self.campaign.evaluate = lambda trainer:checked_fixture(trainer)
        # Build is deliberately injected; bind the disposable config like the
        # real build would, while leaving one genuinely completed saved update.
        def build(resolved):
            trainer.config = deepcopy(resolved)
            stream.config = deepcopy(resolved)
            return trainer,stream
        self.campaign.build = build
        self.campaign.preflight_cases([cfg])
        self.assertIn('direct',self.campaign.state['timing'])
        receipt = json.loads((self.root/'preflight_direct.json').read_text())
        self.assertTrue(receipt['weights_discarded'])
        self.assertTrue(receipt['local_outbox_envelope_validated'])
        self.assertFalse(receipt['published_to_outbox'])
        self.assertFalse(list(self.root.glob('.preflight-checkpoint-*')))
        self.assertFalse((self.root/'outbox').exists())

    def test_build_failure_persists_attempt_before_construction_and_primary_exception(self):
        cfg = plan.resolve_case('s1','D_A1')
        self.campaign.state = {'runs':[]}
        self.campaign.persist = lambda:None
        self.campaign.check = lambda:None
        def build(resolved):
            run = self.campaign.state['runs'][-1]
            directory = Path(run['directory'])
            self.assertTrue((directory/'requested_config.json').exists())
            self.assertEqual(json.loads((directory/'run_state.json').read_text())['status'],'RUNNING')
            raise ValueError('ORIGINAL_BUILD_FAILURE')
        self.campaign.build = build
        with patch('panda40.reporting.emit_status',side_effect=RuntimeError('SECONDARY_OUTBOX_FAILURE')):
            with self.assertRaisesRegex(ValueError,'ORIGINAL_BUILD_FAILURE'):
                self.campaign.run_case(cfg)
        result = self.campaign.state['runs'][-1]
        self.assertEqual(result['status'],'FAILED')
        self.assertIn('SECONDARY_OUTBOX_FAILURE',result['failure_reporting_error'])


if __name__ == '__main__':
    unittest.main()
