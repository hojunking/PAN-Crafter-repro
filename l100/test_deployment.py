"""Read-only launch contract and isolated runtime binding regression."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from l100 import deployment
from l100.common import ROOT


class DeploymentTests(unittest.TestCase):
    def test_training_pause_exit_code_propagates(self):
        from tools.l100_runner import main
        from l100.policy import CampaignWindow
        from contextlib import ExitStack, redirect_stdout
        import io
        with ExitStack() as stack:
            stack.enter_context(patch('l100.common.apply_runtime_policy'))
            stack.enter_context(patch('l100.controller.verify_registration'))
            stack.enter_context(patch('l100.controller.authorize_train'))
            stack.enter_context(patch('l100.controller.shared_window', return_value=CampaignWindow('2026-09-21T00:00:00+00:00')))
            stack.enter_context(patch('l100.training.train_run', return_value=75))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main(['train', '--server', 's3', '--config', 'unused.yaml']), 75)
            self.assertEqual(json.loads(output.getvalue())['exit_code'], 75)

    def test_dry_run_does_not_create_window_or_run(self):
        before = (ROOT / 'work_dir/_l100/campaign_window.json').exists()
        for server in ('s3', 's4', 's5'):
            output = subprocess.check_output([sys.executable, str(ROOT / 'tools/l100_runner.py'),
                'start', '--server', server, '--dry-run'], text=True)
            self.assertEqual(json.loads(output)['state']['server_id'], server)
        self.assertEqual((ROOT / 'work_dir/_l100/campaign_window.json').exists(), before)

    def test_no_per_server_implicit_clock(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/l100_runner.py'),
            'start', '--server', 's4', '--t0', 'now'], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reset clocks separately', result.stderr)

    def test_existing_tracked_gspread_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repo'
            root.mkdir()
            (root / 'data').mkdir()
            (root / 'gspread').mkdir()
            (root / 'gspread/local-key.json').write_text('{}')
            head = 'a' * 40
            target = root.parent / f'repo-runtime-l100-s4-{head[:12]}'
            (target / 'gspread').mkdir(parents=True)
            helper = target / 'gspread/gspread_upload.py'
            helper.write_text('# tracked helper')
            def checked(command, **kwargs):
                if command[1] == 'rev-parse':
                    return head + '\n'
                if command[1] == 'ls-files':
                    return b'tools/l100_start.sh\x00'
                if command[1] == 'diff':
                    return ''
                self.fail(str(command))
            with patch('l100.common.source_identity', return_value={'files': {}}), \
                    patch('l100.deployment.subprocess.check_output', side_effect=checked):
                actual = deployment.frozen_checkout(root, 's4')
                self.assertEqual(actual, target)
                self.assertEqual(helper.read_text(), '# tracked helper')
                self.assertEqual((target / 'gspread/local-key.json').resolve(), root / 'gspread/local-key.json')
                self.assertEqual((target / 'data').resolve(), root / 'data')
                self.assertEqual((target / 'work_dir').resolve(), root / 'work_dir')

    def test_uncommitted_release_is_not_deployed(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('l100.common.source_identity', return_value={'files': {'l100/training.py': 'x'}}), \
                 patch('l100.deployment.subprocess.check_output', side_effect=['a' * 40, b'']):
                with self.assertRaisesRegex(ValueError, 'Commit'):
                    deployment.frozen_checkout(temp, 's3')


if __name__ == '__main__':
    unittest.main()
