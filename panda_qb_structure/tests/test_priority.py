import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_structure import priority as P


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def container(mode='resume-all', target=None, *, running=False, exit_code=1):
    receipt = {'container': 'wv3', 'container_id': 'owned', 'image': 'image',
               'package': 'panda_tune30', 'mode': mode, 'target_run': target}
    args = ['python', '-m', 'panda_sensor5.wv3', '--server', 's1',
            '--package', 'panda_tune30', '--mode', mode]
    if target:
        args += ['--target-run', target]
    current = {'Id': 'owned', 'Image': 'image', 'Config': {'Cmd': args},
               'State': {'Running': running, 'Status': 'running' if running else 'exited',
                         'ExitCode': exit_code, 'OOMKilled': False}}
    return receipt, current


class PriorityTests(unittest.TestCase):
    def test_intentional_exit_one_requires_complete_resume_artifact(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'ROOT', Path(temp)):
            root = Path(temp)
            folder = root / 'work_dir/wv3/runs/target/attempt_01'
            models = folder / 'models'; models.mkdir(parents=True)
            checkpoint = models / 'stop_0001234.pt'; checkpoint.write_bytes(b'complete resume state')
            run = {'run_id': 'target', 'status': 'STOPPED', 'step': 1234, 'poisoned': False,
                   'directory': '/workspace/work_dir/wv3/runs/target/attempt_01'}
            pointer = {'checkpoint_path': '/workspace/work_dir/wv3/runs/target/attempt_01/models/stop_0001234.pt',
                       'file_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       'completed_update': 1234, 'campaign_resume_complete': True,
                       'checkpoint_verified': True}
            dump(models / 'latest_resume.json', pointer)
            dump(folder / 'run_state.json', run)
            state = {'status': 'STOPPED_SAFE', 'reason': 'STOP_REQUESTED', 'runs': [run]}
            receipt, current = container(exit_code=1)
            request = {'original_receipt': receipt, 'target_run_id': 'target'}
            proof = P.authenticate_safe_stop(request, current, state)
            self.assertEqual(proof['checkpoint']['step'], 1234)
            checkpoint.write_bytes(b'incomplete or changed')
            with self.assertRaisesRegex(ValueError, 'resume proof'):
                P.authenticate_safe_stop(request, current, state)

    def test_safe_stop_rejects_oom_unexpected_exit_or_different_target(self):
        receipt, current = container(exit_code=1)
        request = {'original_receipt': receipt, 'target_run_id': 'target'}
        state = {'status': 'STOPPED_SAFE', 'reason': 'STOP_REQUESTED',
                 'runs': [{'run_id': 'other', 'status': 'STOPPED'}]}
        with self.assertRaisesRegex(ValueError, 'changed or failed'):
            P.authenticate_safe_stop(request, current, state)
        current['State']['OOMKilled'] = True
        with self.assertRaisesRegex(ValueError, 'checkpoint-safe'):
            P.authenticate_safe_stop(request, current, state)
        current['State'].update(OOMKilled=False, ExitCode=137)
        with self.assertRaisesRegex(ValueError, 'checkpoint-safe'):
            P.authenticate_safe_stop(request, current, state)

    def test_reused_owner_pid_never_receives_a_signal(self):
        expected = {'pid': 10, 'generation': 'old', 'argv': ['python', '-m', 'panda_qb_refit.priority'],
                    'cwd': str(P.ROOT), 'uid': 1}
        with patch.object(P, 'process_identity', return_value={**expected, 'generation': 'new'}), \
                patch.object(P.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                P.stop_owner(expected)
            kill.assert_not_called()

    def test_appearing_owner_prevents_unowned_signal(self):
        with patch.object(P, 'matching_processes', return_value=[{'pid': 12}]), \
                patch.object(P.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'appeared'):
                P.stop_owner(None)
            kill.assert_not_called()

    def test_finish_current_requires_normal_exit_and_final_model_proof(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'DIRECTORY', Path(temp)):
            receipt, current = container('finish-current', 'target', running=True, exit_code=0)
            dump(Path(temp) / 'finish_current.json', receipt)
            request = {'package': 'panda_tune30', 'target_run_id': 'target'}
            with patch.object(P, 'inspect', return_value=current), \
                    patch('panda_sensor5.transition.completed_run') as complete:
                self.assertIsNone(P.finish_current_complete(request))
                complete.assert_not_called()
                current['State'].update(Running=False, Status='exited')
                complete.side_effect = ValueError('Missing final same-state checkpoint')
                with self.assertRaisesRegex(ValueError, 'Missing final'):
                    P.finish_current_complete(request)
                current['State']['ExitCode'] = 1
                with self.assertRaisesRegex(ValueError, 'complete normally'):
                    P.finish_current_complete(request)

    def test_no_premature_qb_launch(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'DIRECTORY', Path(temp)), \
                patch.object(P, 'finish_current_complete', return_value=None) as complete, \
                patch.object(P, 'ensure_process') as spawn:
            chain = {'stage': 'FINISH_CURRENT'}
            P.advance(chain, {})
            self.assertEqual(chain['stage'], 'FINISH_CURRENT')
            spawn.assert_not_called()
            complete.return_value = {'run_id': 'target', 'step': 30000}
            P.advance(chain, {})
            self.assertEqual(chain['stage'], 'QB_STRUCTURE')
            spawn.assert_not_called()
            self.assertEqual(json.loads((Path(temp) / 'boundary_complete.json').read_text())['step'], 30000)

    def test_qb_wait_and_closeout_precede_wv3_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'DIRECTORY', Path(temp)), \
                patch.object(P, 'qb_complete', return_value=None) as complete, \
                patch.object(P, 'ensure_process', return_value={'pid': 8}) as spawn, \
                patch.object(P, 'launch_wv3') as launch:
            chain = {'stage': 'QB_STRUCTURE'}
            P.advance(chain, {})
            self.assertEqual(chain['stage'], 'QB_STRUCTURE')
            spawn.assert_called_once_with(P.ENQUEUE_ARGS, 'qb_waiter.json', 'qb_waiter.log')
            launch.assert_not_called()
            complete.return_value = {'completed_count': 10, 'early_stopped_count': 0}
            P.advance(chain, {})
            self.assertEqual(chain['stage'], 'RESUME_TUNE30')
            launch.assert_not_called()

    def test_container_collision_rejected(self):
        receipt, current = container()
        current['Id'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'identity/command'):
            P.authenticate_container(current, receipt, 'resume-all')

    def test_saved_own_launch_is_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'DIRECTORY', Path(temp)), \
                patch('panda_sensor5.wv3_launch.launch') as launch:
            receipt, current = container('finish-current', 'target', running=True)
            dump(Path(temp) / 'finish_current.json', receipt)
            with patch.object(P, 'inspect', return_value=current):
                self.assertEqual(P.launch_wv3('panda_tune30', 'finish-current', 'target'), receipt)
            launch.assert_not_called()

    def test_failed_launch_is_not_restarted_in_a_loop(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'ROOT', Path(temp)):
            path = Path(temp) / P.OUTPUT / 'launcher_s1/status.json'
            dump(path, {'stage': 'FAILED_LAUNCH', 'campaign_id': P.CAMPAIGN_ID})
            with self.assertRaisesRegex(ValueError, 'launch/training failed'):
                P.qb_complete()

    def test_qb_completion_requires_sheet_and_own_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(P, 'ROOT', Path(temp)), \
                patch('panda_qb_structure.prerequisites.terminal', return_value={'complete': True}) as terminal:
            path = Path(temp) / P.OUTPUT / 'launcher_s1/status.json'
            status = {'stage': 'COMPLETED', 'campaign_id': P.CAMPAIGN_ID}
            dump(path, status)
            with self.assertRaisesRegex(ValueError, 'Sheet closeout'):
                P.qb_complete()
            terminal.assert_not_called()
            dump(path, {**status, 'sheet_verified': True})
            self.assertEqual(P.qb_complete(), {'complete': True})
            terminal.assert_called_once_with(Path(temp), 's1')


if __name__ == '__main__':
    unittest.main()
