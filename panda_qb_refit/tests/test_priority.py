from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_refit.prerequisites import priority_evidence, sha
from panda_qb_refit.priority import validate_finish_container, advance


class PriorityTests(unittest.TestCase):
    def test_boundary_requires_exact_container_and_normal_exit(self):
        request = {'package': 'panda_tune30', 'target_run_id': 'target',
                   'original_receipt': {'image': 'image'}}
        receipt = {'container_id': 'id', 'image': 'image', 'target_run': 'target'}
        current = {'Id': 'id', 'Image': 'image', 'Config': {'Cmd': [
            'python', '-m', 'panda_sensor5.wv3', '--server', 's1', '--package', 'panda_tune30',
            '--mode', 'finish-current', '--target-run', 'target']},
            'State': {'Running': True, 'Status': 'running', 'ExitCode': 0}}
        self.assertFalse(validate_finish_container(request, receipt, current))
        current['State'].update(Running=False, Status='exited')
        self.assertTrue(validate_finish_container(request, receipt, current))
        for change in ('exit', 'id', 'image', 'target'):
            invalid = deepcopy(current)
            if change == 'exit': invalid['State']['ExitCode'] = 1
            if change == 'id': invalid['Id'] = 'replacement'
            if change == 'image': invalid['Image'] = 'other'
            if change == 'target': invalid['Config']['Cmd'][-1] = 'next-run'
            with self.assertRaises(ValueError):
                validate_finish_container(request, receipt, invalid)

    def test_no_priority_bypass_before_final_evidence_or_after_wv3_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / 'work_dir/panda_qb_refit/priority_s1'
            directory.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'WAIT:'):
                priority_evidence(root)
            request = {'server': 's1', 'package': 'panda_tune30', 'target_run_id': 'target',
                       'policy': 'USER_20261030_FINISH_CURRENT_THEN_QB_THEN_WV3'}
            (directory / 'request.json').write_text(json.dumps(request))
            folder = root / 'wv3'; folder.mkdir()
            state = {'status': 'PAUSED_FOR_SENSOR5', 'runs': [{'status': 'COMPLETED'}]}
            state_file = folder / 'campaign_state.json'
            state_file.write_text(json.dumps(state))
            completed = {'run_id': 'target', 'step': 30000}
            proof = {'request_sha256': sha(directory / 'request.json'),
                     'container_exit_code': 0, 'state_sha256': sha(state_file), 'completed_run': completed}
            (directory / 'boundary_complete.json').write_text(json.dumps(proof))
            with patch('panda_sensor5.transition.campaign_root', return_value=folder), \
                    patch('panda_sensor5.transition.state_of', side_effect=lambda *a: json.loads(state_file.read_text())), \
                    patch('panda_sensor5.transition.completed_run', return_value=completed) as final:
                self.assertEqual(priority_evidence(root), proof)
                final.side_effect = ValueError('Missing final evaluation')
                with self.assertRaisesRegex(ValueError, 'Missing final'):
                    priority_evidence(root)
                final.side_effect = None
                state['runs'].append({'status': 'RUNNING'})
                state_file.write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'WV3 changed'):
                    priority_evidence(root)

    def test_qb_must_close_sheet_before_remaining_wv3(self):
        with tempfile.TemporaryDirectory() as temp, patch('panda_qb_refit.priority.ROOT', Path(temp)), \
                patch('panda_qb_refit.priority.DIRECTORY', Path(temp)), \
                patch('panda_qb_refit.priority.priority_evidence'), \
                patch('panda_qb_refit.priority.ensure_qb_waiter', return_value=123), \
                patch('panda_qb_refit.priority.terminal', return_value={'completed_count': 9, 'early_stopped_count': 1}), \
                patch('panda_sensor5.queue.advance') as resume:
            from panda_qb_refit.registry import CAMPAIGN_ID
            path = Path(temp) / 'work_dir/panda_qb_refit/launcher_s1/status.json'
            path.parent.mkdir(parents=True)
            chain = {'stage': 'QB_REFIT'}
            path.write_text(json.dumps({'stage': 'WAIT_SHEET_READBACK'}))
            advance(chain, {})
            self.assertEqual(chain['stage'], 'QB_REFIT'); resume.assert_not_called()
            path.write_text(json.dumps({'stage': 'COMPLETED', 'campaign_id': CAMPAIGN_ID}))
            with self.assertRaisesRegex(ValueError, 'Sheet'):
                advance(chain, {})
            path.write_text(json.dumps({'stage': 'COMPLETED', 'campaign_id': CAMPAIGN_ID, 'sheet_verified': True}))
            advance(chain, {})
            self.assertEqual(chain['stage'], 'RESUME_TUNE30'); resume.assert_not_called()
            self.assertEqual((chain['qb_completed'], chain['qb_early_stopped']), (9, 1))
            advance(chain, {})
            resume.assert_called_once()


if __name__ == '__main__':
    unittest.main()
