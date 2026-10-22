"""No training/GPU processes are launched by pending-controller tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_scale30 import wait_previous as W


class WaiterTests(unittest.TestCase):
    def test_active_previous_is_only_waited_for(self):
        self.assertEqual(W.prior_decision({'Running': True}, {'status': 'RUNNING'}),
                         'WAITING_PREVIOUS')

    def test_abnormal_or_incomplete_previous_never_launches(self):
        valid = {'Running': False, 'Status': 'exited', 'ExitCode': 0}
        for container in ({**valid, 'ExitCode': 1}, {**valid, 'Status': 'created'}):
            with self.assertRaisesRegex(RuntimeError, 'exit normally'):
                W.prior_decision(container, {'status': 'FINISHED_REGISTERED_QUEUE', 'runs': []})
        for status in ('RUNNING', 'STOPPED_SAFE', 'FAILED', None):
            with self.assertRaisesRegex(RuntimeError, 'terminal receipt'):
                W.prior_decision(valid, {'status': status})
        for status in ('RUNNING', 'STOPPED', 'FAILED', None):
            with self.assertRaisesRegex(RuntimeError, 'incomplete run'):
                W.prior_decision(valid, {'status': 'FINISHED_REGISTERED_QUEUE',
                                         'runs': [{'status': status}]})

    def test_success_requires_successful_campaign_and_completed_runs(self):
        valid = {'Running': False, 'Status': 'exited', 'ExitCode': 0}
        for status in W.SUCCESSFUL:
            self.assertEqual(W.prior_decision(valid, {'status': status,
                            'runs': [{'status': 'COMPLETED'}]}), 'READY_TO_CHECK_GPU')

    def test_source_changes_and_added_inventory_prevent_queued_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'source.py'
            path.write_text('pass\n')
            pins = {'source.py': W.hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(W, 'source_pin', return_value=pins):
                W.check_pin(root, pins)
            with patch.object(W, 'source_pin', return_value={**pins, 'new.py': 'a'*64}):
                with self.assertRaisesRegex(RuntimeError, 'inventory changed'):
                    W.check_pin(root, pins)
            path.write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                W.check_pin(root, pins)


if __name__ == '__main__':
    unittest.main()
