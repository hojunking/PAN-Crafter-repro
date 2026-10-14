"""Process safety tests: all signals, process launches and Docker calls mocked."""
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from . import operations as O


def evidence():
    return {'pid': 12345, 'starttime': '123', 'argv': ['python', '-m', 'directshift48', 'run', '--server', 's1'],
            'container_id': 'abc', 'container_name': 'panda-directshift48-s1', 'server': 's1'}


class OperationsTests(unittest.TestCase):
    def mocked_pause(self, stack, *, status='\nState:\tT (stopped)\n'):
        guard = MagicMock()
        guard.poll.return_value = None
        stack.enter_context(patch.object(O, '_argv', return_value=evidence()['argv']))
        stack.enter_context(patch.object(O, '_starttime', return_value='123'))
        stack.enter_context(patch.object(O.subprocess, 'Popen', return_value=guard))
        stack.enter_context(patch.object(O.time, 'sleep'))
        stack.enter_context(patch.object(Path, 'read_text', return_value=status))
        kill = stack.enter_context(patch.object(O.os, 'kill'))
        return guard, kill

    def test_body_error_always_resumes_exact_trainer(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            guard, kill = self.mocked_pause(stack)
            with self.assertRaisesRegex(RuntimeError, 'profiling failed'):
                with O.paused_trainer(evidence(), directory):
                    raise RuntimeError('profiling failed')
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(12345, signal.SIGSTOP), (12345, signal.SIGCONT)])
            self.assertEqual(len(list(Path(directory).glob('*.done'))), 1)
            guard.wait.assert_called_once_with(timeout=5)

    def test_guard_failure_never_stops_training(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            guard, kill = self.mocked_pause(stack)
            guard.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'watchdog failed'):
                with O.paused_trainer(evidence(), directory):
                    self.fail('Should not enter lease')
            kill.assert_not_called()

    def test_pid_reuse_before_pause_refuses_all_signals(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, kill = self.mocked_pause(stack)
            stack.enter_context(patch.object(O, '_starttime', return_value='changed'))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                with O.paused_trainer(evidence(), directory):
                    self.fail('Should not enter lease')
            kill.assert_not_called()

    def test_stop_timeout_still_resumes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, kill = self.mocked_pause(stack, status='\nState:\tR (running)\n')
            stack.enter_context(patch.object(O.time, 'monotonic', side_effect=[0., 6.]))
            with self.assertRaises(TimeoutError):
                with O.paused_trainer(evidence(), directory):
                    self.fail('Should not enter lease')
            self.assertEqual(kill.call_args.args, (12345, signal.SIGCONT))

    def test_disappeared_trainer_does_not_mask_body_exception(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, kill = self.mocked_pause(stack)
            stack.enter_context(patch.object(O, '_starttime', side_effect=['123', FileNotFoundError()]))
            with self.assertRaisesRegex(RuntimeError, 'original'):
                with O.paused_trainer(evidence(), directory):
                    raise RuntimeError('original')
            self.assertEqual([call.args for call in kill.call_args_list], [(12345, signal.SIGSTOP)])
            self.assertEqual(len(list(Path(directory).glob('*.done'))), 1)

    def test_resume_failure_keeps_independent_guard_active(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            guard, kill = self.mocked_pause(stack)
            kill.side_effect = [None, PermissionError('test resume failure')]
            with self.assertRaises(PermissionError):
                with O.paused_trainer(evidence(), directory):
                    pass
            self.assertFalse(list(Path(directory).glob('*.done')))
            guard.wait.assert_not_called()

    def test_watchdog_resume_and_pid_birth_guard(self):
        for start, signals in (('123', 1), ('changed', 0), (FileNotFoundError(), 0)):
            with self.subTest(start=start), patch.object(O.time, 'monotonic', return_value=0), \
                    patch.object(O, '_starttime', side_effect=start if isinstance(start, Exception) else None,
                                 return_value=start), patch.object(O.os, 'kill') as kill:
                O.resume_guard(12345, '123', '/nonexistent/cancel', 67890, 0)
                self.assertEqual(kill.call_count, signals)
                if signals:
                    kill.assert_called_once_with(12345, signal.SIGCONT)

    def test_inspection_rejects_other_gpu_jobs_and_accepts_exact_trainer(self):
        raw = {'State': {'Running': True, 'Paused': False}, 'Id': 'abc'}
        for gpu, expected in (({12345}, True), ({12345, 99999}, False)):
            with self.subTest(gpu=gpu), patch.object(O.subprocess, 'check_output',
                    side_effect=[json.dumps([raw]), 'PID COMMAND\n12345 python\n']), \
                    patch.object(O, '_argv', return_value=evidence()['argv']), \
                    patch.object(O, '_starttime', return_value='123'), \
                    patch.object(O, '_gpu_pids', return_value=gpu):
                if expected:
                    self.assertEqual(O.inspect_trainer('container', 's1')['pid'], 12345)
                else:
                    with self.assertRaisesRegex(ValueError, 'Other GPU'):
                        O.inspect_trainer('container', 's1')

    def test_already_measured_never_pauses(self):
        with patch.object(O, 'configurations', return_value=[{}]), \
                patch('experiment_cost.store.find_profile', return_value={'verified': True}), \
                patch.object(O, 'paused_trainer', side_effect=AssertionError('No pause')):
            self.assertEqual(O.measure_needed('/unused', 's1', 'container')['status'], 'ALREADY_MEASURED')

    def test_failed_measurement_does_not_stop_publisher(self):
        with patch.object(O, 'measure_needed', side_effect=RuntimeError('profile failure')), \
                patch.object(O.os, 'kill') as kill:
            with self.assertRaises(RuntimeError):
                O.start('/unused', 's1', 'container', credentials='unused')
            kill.assert_not_called()

    def test_bad_credentials_fail_before_old_publisher_is_signalled(self):
        with patch.object(O, 'measure_needed', return_value={'status': 'ALREADY_MEASURED'}), \
                patch('experiment_cost.publication.enriched_projection',
                      return_value={'conflicts': [], 'destination': {'sheet_id': 1}}), \
                patch('sheet_publish_v2.transport.GoogleSheets', side_effect=ValueError('invalid credentials')), \
                patch('experiment_cost.publication.competing_publishers') as competitors, \
                patch.object(O.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'invalid credentials'):
                O.start('/unused', 's1', 'container', credentials='unused')
            competitors.assert_not_called()
            kill.assert_not_called()

    def test_foreign_remote_rows_fail_before_cutover_or_cloud_write(self):
        from directshift48.publication import combined_projection
        from directshift48.test_publication import persist
        from directshift48.test_reporting import record
        from sheet_publish_v2.test_transport import Fake, snapshot
        from sheet_publish_v2.transport import TransportError
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record())
            project = combined_projection(directory, 's1', [])
            remote = snapshot(project['rows'], project['notes'])
            # Preserve valid shape/schema, but make this row foreign ownership.
            from directshift48.publication import INDEX
            remote['rows'][0][INDEX['Campaign_ID']] = 'FOREIGN_CAMPAIGN'
            adapter = Fake(remote)
            with patch.object(O, 'measure_needed', return_value={'status': 'ALREADY_MEASURED'}), \
                    patch('experiment_cost.publication.enriched_projection', return_value=project), \
                    patch('sheet_publish_v2.transport.GoogleSheets', return_value=adapter), \
                    patch('experiment_cost.publication.competing_publishers') as competitors, \
                    patch.object(O.os, 'kill') as kill:
                with self.assertRaises(TransportError):
                    O.start(directory, 's1', 'container', credentials='unused')
                self.assertFalse(adapter.writes)
                competitors.assert_not_called()
                kill.assert_not_called()

    def test_s3_default_lane_path(self):
        path = O.campaign_root('s3')
        self.assertEqual(path.parts[-2:], ('s3', 'GF2'))

    def test_readonly_repo_and_nested_writable_workdir_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = [{'Type': 'bind', 'Source': str(root), 'Destination': '/workspace', 'RW': False},
                      {'Type': 'bind', 'Source': str(root/'work_dir'),
                       'Destination': '/workspace/work_dir', 'RW': True}]
            path = root/'work_dir/experiment_cost/s3/requests/request.json'
            self.assertEqual(O._writable_container_path(path, mounts, repository_root=root),
                             '/workspace/work_dir/experiment_cost/s3/requests/request.json')
            with self.assertRaisesRegex(ValueError, 'writable bind'):
                O._writable_container_path(root/'profiles/test.json', mounts, repository_root=root)
            with self.assertRaisesRegex(ValueError, 'inside the mounted repository'):
                O._writable_container_path(root.parent/'outside.json', mounts, repository_root=root)

    def test_more_specific_readonly_or_different_host_mount_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = [{'Type': 'bind', 'Source': str(root), 'Destination': '/workspace', 'RW': False},
                      {'Type': 'bind', 'Source': str(root/'work_dir'),
                       'Destination': '/workspace/work_dir', 'RW': True}]
            path = root/'work_dir/experiment_cost/s3/request.json'
            extra = {'Type': 'bind', 'Source': str(root/'work_dir/experiment_cost'),
                     'Destination': '/workspace/work_dir/experiment_cost', 'RW': False}
            with self.assertRaisesRegex(ValueError, 'writable bind'):
                O._writable_container_path(path, mounts+[extra], repository_root=root)
            extra.update(Source=str(root/'different_location'), RW=True)
            with self.assertRaisesRegex(ValueError, 'different host directory'):
                O._writable_container_path(path, mounts+[extra], repository_root=root)

    def test_unwritable_mount_fails_before_inspecting_or_pausing_trainer(self):
        raw = {'Mounts': [{'Type': 'bind', 'Source': str(O.ROOT), 'Destination': '/workspace', 'RW': False}]}
        with patch.object(O, 'configurations', return_value=[{}]), \
                patch('experiment_cost.store.find_profile', return_value=None), \
                patch.object(O.subprocess, 'check_output', return_value=json.dumps([raw])), \
                patch.object(O, 'inspect_trainer', side_effect=AssertionError('Must validate first')), \
                patch.object(O.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'writable bind'):
                O.measure_needed('/unused', 's1', 'container')
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
