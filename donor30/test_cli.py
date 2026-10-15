"""Entry points must be explicit; a review/catalog is never a launch."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import __main__ as cli, plan


class EntryPointTests(unittest.TestCase):
    def test_catalog_both_lanes_is_read_only(self):
        for server in ('s1', 's3'):
            out = io.StringIO()
            with patch('subprocess.Popen') as spawn, redirect_stdout(out):
                self.assertEqual(cli.main(['catalog', '--server', server]), 0)
            value = json.loads(out.getvalue())
            self.assertEqual(value['server'], server)
            self.assertTrue(value['donor']['allow_exploratory_fallback'])
            spawn.assert_not_called()

    def test_busy_gpu_never_constructs_campaign(self):
        with patch('directshift48.__main__.require_idle_gpu', side_effect=RuntimeError('busy')), \
                patch('donor30.runtime.Campaign') as campaign:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                cli.main(['run', '--server', 's1'])
            campaign.assert_not_called()

    def test_publish_default_remains_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch('donor30.publication.publish', return_value=0) as publish:
            self.assertEqual(cli.main(['publish', '--server', 's3', '--root', tmp]), 0)
            args, kwargs = publish.call_args
            self.assertEqual(args, (Path(tmp), 's3'))
            self.assertFalse(kwargs['apply'])
            self.assertFalse(kwargs['watch'])

    def test_automatic_publisher_needs_explicit_launch_authority(self):
        with patch('donor30.publication.start_publisher') as start, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(['publish-auto', '--server', 's1'])
            start.assert_not_called()

    def test_automatic_publisher_does_not_replace_writer_after_failed_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Test artifacts only, outside the repository and live campaign.
            from directshift48.reporting import atomic_json
            atomic_json(Path(tmp) / 'campaign_state.json', {'status': 'FAILED'})
            with patch('donor30.publication.start_publisher') as start:
                with self.assertRaisesRegex(RuntimeError, 'preflight stopped'):
                    cli.main(['publish-auto', '--server', 's1', '--root', tmp, '--apply'])
                start.assert_not_called()

    def test_shell_syntax_and_no_destructive_job_commands(self):
        for script in ('run_donor30.sh', 'run_donor30_docker.sh'):
            path = plan.ROOT / 'scripts' / script
            subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)
            content = path.read_text()
            for command in ('docker stop', 'docker kill', 'docker rm', 'pkill', 'kill -9', 'rm -rf'):
                self.assertNotIn(command, content)


if __name__ == '__main__':
    unittest.main()
