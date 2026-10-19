"""Completed-donor metadata tests; fixtures are not real experiment evidence.

The helper receives an already strict-loaded donor in production. These small
fixtures isolate the additional metadata checks without repeating GPU training.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from panda_warp30 import plan
from panda_warp30.runtime import plain, validate_completed_donor_receipt
from panda_warp30.stage1 import CHECKPOINT_SCHEMA
from ta2.evaluation import file_sha


class DonorReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)/'attempt_01'
        self.directory.mkdir()
        (self.directory/'models').mkdir()
        self.cfg = plain(plan.resolve_case('s3', 'S1_U23'))
        self.state_hash = 'c' * 64
        self.step = self.cfg['evaluation_steps'][0]
        self.path = self.directory/'models'/f'step_{self.step:07d}.pt'
        self.evidence_path = self.directory/f'evaluation_{self.step:07d}.json'
        identity = {'model_state_sha256': self.state_hash, 'completed_update': self.step}
        self.evaluation = {'state_before': identity,
            'fr': {'metrics': {'HQNR': .961}}, 'rr': {'metrics': {'ERGAS': 2.1, 'SCC': .99}}}
        self.payload = {'schema': CHECKPOINT_SCHEMA, 'config': deepcopy(self.cfg),
            'trainer': {'updates': self.step}, 'evaluation': deepcopy(self.evaluation),
            'model_state_sha256': self.state_hash, 'state_hash': self.state_hash}
        torch.save(self.payload, self.path)
        self.sha = file_sha(self.path)
        saved = []
        for step in self.cfg['evaluation_steps']:
            row = {'run_id': self.cfg['run_id'], 'attempt': 1, 'seed': self.cfg['seed'],
                'completed_update': step, 'HQNR': .961 if step == self.step else .95,
                'ERGAS': 2.1, 'SCC': .99, 'state_hash': self.state_hash,
                'checkpoint_verified': True, 'metrics_complete': True, 'metrics_finite': True,
                'same_state_verified': True, 'file_sha256': self.sha,
                'checkpoint_path': str(self.directory/'models'/f'step_{step:07d}.pt'),
                'evaluation_path': str(self.directory/f'evaluation_{step:07d}.json')}
            saved.append(row)
        self.selection = {'best_selected': deepcopy(saved[0]), 'best_hqnr': deepcopy(saved[0]),
                          'saved': saved, 'policy': plan.POLICY.manifest()}
        self.finished = {k: self.cfg[k] for k in ('run_id', 'case', 'replica', 'attempt', 'phase')}
        self.finished.update(status='COMPLETED', step=50000, directory=str(self.directory),
                             selection=self.selection)
        self.donor = SimpleNamespace(checkpoint_metadata={
            'checkpoint_path': str(self.path), 'checkpoint_sha256': self.sha,
            'run_id': self.cfg['run_id'], 'completed_update': self.step,
            'donor_model_state_sha256': self.state_hash}, binding={
                'donor_checkpoint_id': self.sha,
                'warp30_teacher': {'campaign_id': self.cfg['campaign_id'], 'upsampler': 'interp23'}})
        self.write_metadata()

    def write_metadata(self):
        for path, value in (
            (self.directory/'requested_config.json', self.cfg),
            (self.directory/'config.json', self.cfg),
            (self.directory/'run_state.json', self.finished),
            (self.directory/'models/selection.json', self.selection),
            (self.evidence_path, self.evaluation)):
            path.write_text(json.dumps(value))

    def validate(self):
        return validate_completed_donor_receipt(self.cfg, self.finished, self.donor)

    def test_normal_earlier_best_after_full_horizon_is_accepted(self):
        self.assertLess(self.step, 50000)
        self.assertEqual(self.validate(), self.cfg)

    def test_incomplete_or_wrong_identity_completion_rejected(self):
        for key, value in (('step', 49999), ('step', True), ('status', 'STOPPED'),
                           ('run_id', 'another_run'), ('phase', 'stage2')):
            old = self.finished[key]
            self.finished[key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'completion'):
                self.validate()
            self.finished[key] = old

    def test_selected_metrics_must_equal_embedded_checkpoint(self):
        for key, value in (('HQNR', .97), ('ERGAS', 1.9), ('SCC', .9)):
            original = deepcopy(self.selection)
            self.selection['best_selected'][key] = value
            self.selection['saved'][0][key] = value
            self.write_metadata()
            with self.subTest(metric=key), self.assertRaisesRegex(ValueError, 'metric values'):
                self.validate()
            self.selection.clear()
            self.selection.update(original)

    def test_requested_resolved_and_embedded_configs_are_bound(self):
        for name in ('requested_config.json', 'config.json'):
            path = self.directory/name
            changed = deepcopy(self.cfg)
            changed['upsampler'] = 'bicubic'
            path.write_text(json.dumps(changed))
            with self.subTest(file=name), self.assertRaisesRegex(ValueError, 'configuration'):
                self.validate()
            self.write_metadata()
        changed = deepcopy(self.payload)
        changed['config']['upsampler'] = 'bicubic'
        torch.save(changed, self.path)
        sha = file_sha(self.path)
        self.selection['saved'][0]['file_sha256'] = sha
        self.selection['best_selected']['file_sha256'] = sha
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'payload configuration'):
            self.validate()

    def test_best_pointer_and_full_evaluation_schedule_required(self):
        original = deepcopy(self.selection)
        self.selection['best_selected'] = deepcopy(self.selection['saved'][-1])
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'maximum HQNR'):
            self.validate()
        self.selection.clear()
        self.selection.update(original)
        self.selection['saved'].pop()
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'full registered evaluation schedule'):
            self.validate()

    def test_checkpoint_step_sha_and_external_evaluation_are_bound(self):
        altered = deepcopy(self.payload)
        altered['trainer']['updates'] += 1
        torch.save(altered, self.path)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.validate()
        sha = file_sha(self.path)
        self.selection['saved'][0]['file_sha256'] = sha
        self.selection['best_selected']['file_sha256'] = sha
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'payload configuration/update'):
            self.validate()
        torch.save(self.payload, self.path)
        sha = file_sha(self.path)
        self.selection['saved'][0]['file_sha256'] = sha
        self.selection['best_selected']['file_sha256'] = sha
        self.write_metadata()
        changed = deepcopy(self.evaluation)
        changed['fr']['metrics']['HQNR'] = .99
        self.evidence_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'external and embedded'):
            self.validate()

    def test_loaded_pair_must_match_actual_selected_file(self):
        self.donor.binding['warp30_teacher']['upsampler'] = 'bicubic'
        with self.assertRaisesRegex(ValueError, 'Imported donor pair'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
