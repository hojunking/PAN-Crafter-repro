import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('s1loss9_recovery.py'))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.failed = dict(run_id='failed', status='FAILED', error=R.ERROR,
                           phase='donor', poisoned=False, step=11197)
        self.state = {'runs': [self.failed, dict(run_id='good', status='COMPLETED', step=30000)]}
        self.configs = [dict(run_id='failed', planned_horizon=50000),
                        dict(run_id='child', planned_horizon=30000),
                        dict(run_id='good', planned_horizon=30000)]

    def test_partial_completion_is_explicit_and_failed_record_preserved(self):
        original = copy.deepcopy(self.state)
        R.validate_accounting(self.state, self.configs, 'failed', 'child')
        self.assertEqual(original, self.state)
        self.assertNotEqual(R.TERMINAL, 'FINISHED_REGISTERED_QUEUE')

    def test_unknown_failure_and_poisoned_optimizer_rejected(self):
        for patch in ({'error': 'CUDA out of memory'}, {'poisoned': True},
                      {'status': 'COMPLETED'}, {'stop_checkpoint_error': 'failed save'}):
            state = copy.deepcopy(self.state)
            state['runs'][0].update(patch)
            with self.assertRaises(ValueError):
                R.validate_accounting(state, self.configs, 'failed', 'child')

    def test_started_child_cannot_be_hidden(self):
        self.state['runs'].append(dict(run_id='child', status='RUNNING'))
        with self.assertRaises(ValueError):
            R.validate_accounting(self.state, self.configs, 'failed', 'child')

    def test_missing_duplicate_or_incomplete_successor_blocks_transition(self):
        for runs in ([self.failed], self.state['runs'] + [self.state['runs'][1]],
                     [self.failed, dict(run_id='good', status='STOPPED', step=20000)]):
            with self.assertRaises(ValueError):
                R.validate_accounting({'runs': runs}, self.configs, 'failed', 'child')

    def test_final_evidence_hash_and_same_state_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / 'runs/good/attempt_01'
            (d / 'models').mkdir(parents=True)
            checkpoint = d / 'models/step_0030000.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            ev = {'qualification': dict(same_state_verified=True, metrics_complete=True, metrics_finite=True),
                  'state_before': {'model_state_sha256': 'model'}}
            meta = {'artifact': dict(checkpoint_verified=True, completed_update=30000,
                                    file_sha256=R.sha(checkpoint), model_state_sha256='model'),
                    'config': {'run_id': 'good'}}
            (d / 'evaluation_0030000.json').write_text(json.dumps(ev))
            checkpoint.with_suffix('.json').write_text(json.dumps(meta))
            R.verify_final_evidence(root, self.configs[-1])
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                R.verify_final_evidence(root, self.configs[-1])


if __name__ == '__main__':
    unittest.main()
