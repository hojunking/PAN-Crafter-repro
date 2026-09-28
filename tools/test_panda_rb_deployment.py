"""Read-only/mock launcher admission checks; never invoke Docker/GPU start."""
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from panda_rb.deployment import git_mounts, legacy_processes, start, snapshot, DEFAULT_IMAGE


class DeploymentSafetyTests(unittest.TestCase):
    def test_one_checkout_cannot_admit_two_server_slots(self):
        with tempfile.TemporaryDirectory() as tmp, patch('panda_rb.deployment.validate_plan'):
            root = Path(tmp); owner = root / 'work_dir/_panda_rb/local_owner.json'
            owner.parent.mkdir(parents=True); owner.write_text(json.dumps({'server': 's1'}))
            with patch('panda_rb.deployment.subprocess.check_output') as process:
                with self.assertRaisesRegex(RuntimeError, 'different B01 server'):
                    start('s3', root=root)
                process.assert_not_called()

    def test_snapshot_has_readonly_nested_bind_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / '.git').mkdir()
            with patch('panda_rb.deployment.source_files', return_value={}), patch(
                    'panda_rb.deployment.subprocess.check_output', return_value='commit\n'):
                release, digest = snapshot(root)
                for directory in ('work_dir', 'data', 'pan_h5', '.git'):
                    self.assertTrue((release / directory).is_dir())
                self.assertEqual(snapshot(root), (release, digest))

    def test_regular_git_mount_is_readable_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / '.git').mkdir()
            with patch('panda_rb.deployment.subprocess.check_output', side_effect=[str(root / '.git'), '.git']):
                self.assertEqual(git_mounts(root), [root / '.git'])

    def test_worktree_mount_includes_pointer_gitdir_and_common(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / 'PAN-Crafter-run'; root.mkdir()
            common = parent / 'main/.git'; gitdir = common / 'worktrees/run'; gitdir.mkdir(parents=True)
            (root / '.git').write_text('gitdir: ' + str(gitdir))
            with patch('panda_rb.deployment.subprocess.check_output', side_effect=[str(gitdir), str(common)]):
                self.assertEqual(set(git_mounts(root)), {root / '.git', gitdir, common})

    def test_missing_git_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                git_mounts(tmp)

    def test_idle_legacy_queue_is_detected_without_gpu_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / 'PAN-Crafter'; root.mkdir()
            proc = parent / 'proc'; proc.mkdir()
            def fake(pid, cwd, *args):
                item = proc / str(pid); item.mkdir()
                (item / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in args) + b'\0')
                (item / 'cwd').symlink_to(cwd)
            fake(101, root, 'python', 'tools/fh20r1_runner.py', 'run', '--secret', 'not-printed')
            fake(102, root, 'python', 'tools/panda_rb_runner.py', 'run')
            fake(103, root, 'bash', '-lc', 'rg tools/fh20r1_runner.py')
            unrelated = parent / 'unrelated'; unrelated.mkdir()
            fake(104, unrelated, 'python', 'tools/other_runner.py')
            self.assertEqual(legacy_processes(root, proc), [dict(pid=101, script='fh20r1_runner.py')])

    def test_sibling_pinned_runtime_controller_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / 'PAN-Crafter'; root.mkdir()
            sibling = parent / 'PAN-Crafter-runtime-old'; sibling.mkdir()
            proc = parent / 'proc'; item = proc / '500'; item.mkdir(parents=True)
            (item / 'cmdline').write_bytes(b'python\0tools/r2_runner.py\0run\0')
            (item / 'cwd').symlink_to(sibling)
            self.assertEqual(legacy_processes(root, proc), [dict(pid=500, script='r2_runner.py')])

    def test_existing_attempt_container_found_by_campaign_labels(self):
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            if args[:3] == ['docker', 'image', 'inspect']:
                return json.dumps([{'Id': DEFAULT_IMAGE}])
            if args[:2] == ['docker', 'ps']:
                return 'running-attempt-two-id'
            raise AssertionError('Must not query/launch after existing attempt detection')
        with tempfile.TemporaryDirectory() as tmp, patch('panda_rb.deployment.validate_plan'), patch('panda_rb.deployment.subprocess.check_output', side_effect=command):
            report = start('s1', root=tmp)
        self.assertTrue(report['already_running'])
        self.assertIn('label=panda_rb.server=s1', calls[1])
        self.assertIn('label=panda_rb.campaign=B01', calls[1])

    def test_mock_launch_preserves_attempt_logs_mounts_git_readonly_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); (root / 'tools').mkdir(); (root / '.git').mkdir()
            (root / 'tools/_watchdog.sh').write_text('# PANDA_RB_LOCAL_OWNER\n')
            release = root / 'frozen'; release.mkdir()
            commands = []
            def command(args, **kwargs):
                commands.append(args)
                if args[:3] == ['docker', 'image', 'inspect']:
                    return json.dumps([{'Id': DEFAULT_IMAGE}])
                if args[:3] == ['docker', 'ps', '-a']:
                    return 'panda-rb-b01-s1\n'
                if args[:2] == ['docker', 'ps'] or args[0] == 'nvidia-smi':
                    return ''
                if args[:2] == ['docker', 'run']:
                    return 'new-container-id\n'
                raise AssertionError(args)
            stack.enter_context(patch('panda_rb.deployment.validate_plan'))
            stack.enter_context(patch('panda_rb.deployment.snapshot', return_value=(release, 'release-sha')))
            stack.enter_context(patch('panda_rb.deployment.legacy_processes', return_value=[]))
            stack.enter_context(patch('panda_rb.deployment.git_mounts', return_value=[root / '.git']))
            stack.enter_context(patch('panda_rb.deployment.asset_mounts', return_value=[]))
            stack.enter_context(patch('panda_rb.deployment.subprocess.check_output', side_effect=command))
            report = start('s1', root=root)
            self.assertEqual(report['container'], 'panda-rb-b01-s1-attempt2')
            launch = commands[-1]
            self.assertIn(f'{root / ".git"}:{root / ".git"}:ro', launch)
            self.assertIn(f'{release}:{root}:ro', launch)
            self.assertIn('panda_rb.server=s1', launch)
            self.assertIn('panda_rb.campaign=B01', launch)
            self.assertNotIn('rm', launch)
            self.assertTrue((root / 'work_dir/_panda_rb/local_owner.json').is_file())

    def test_idle_legacy_owner_rejects_before_any_admission_guard_change(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch('panda_rb.deployment.validate_plan'))
            stack.enter_context(patch('panda_rb.deployment.subprocess.check_output', side_effect=[json.dumps([{'Id': DEFAULT_IMAGE}]), '', '']))
            stack.enter_context(patch('panda_rb.deployment.legacy_processes', return_value=[dict(pid=42, script='fh20r1_runner.py')]))
            with self.assertRaisesRegex(RuntimeError, 'Existing PAN queue'):
                start('s1', root=tmp)
            self.assertFalse((Path(tmp) / 'work_dir/_panda_rb/local_owner.json').exists())


if __name__ == '__main__':
    unittest.main()
