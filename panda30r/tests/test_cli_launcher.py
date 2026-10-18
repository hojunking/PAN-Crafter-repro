"""CLI and shell launch contracts, using fake executables only (no CUDA/Sheets)."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from panda30r.__main__ import main

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ('run_panda30r.sh', 'run_panda30r_docker.sh')


class CLITests(unittest.TestCase):
    def test_unsupported_or_missing_server_rejected(self):
        for command in ('catalog', 'check', 'run'):
            for args in ([command], [command, '--server', 's2']):
                with self.subTest(args=args), patch('sys.stderr', new=io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
                    self.assertEqual(raised.exception.code, 2)

    def test_mounts_selects_wv3_not_legacy_s3_gf2(self):
        output = io.StringIO()
        with patch('edgeres48.docker_mounts.targets', return_value=['/data/wv3.h5']) as targets:
            with redirect_stdout(output):
                self.assertEqual(main(['mounts']), 0)
        targets.assert_called_once_with(ROOT, 's1')
        self.assertEqual(output.getvalue(), '/data/wv3.h5\n')

    def test_help_needs_no_training_stack(self):
        code = '''
import importlib.abc, sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','h5py','google','panda40','panda_stage2_condq'):
            raise AssertionError('Help loaded training dependency: '+fullname)
sys.meta_path.insert(0, RejectTraining())
from panda30r.__main__ import main
raise SystemExit(main(['--help']))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_catalog_dispatch_is_server_specific_and_read_only(self):
        fake = types.ModuleType('panda30r.plan')
        fake.catalog_csv = Mock(side_effect=lambda server: 'server\n' + server)
        fake.default_config = Mock(side_effect=lambda server: {'server': server})
        # Dependency import traps ensure catalog never invokes runtime/evaluation.
        with patch.dict(sys.modules, {'panda30r.plan': fake}), \
                patch('panda30r.plan', fake, create=True):
            for server in ('s1', 's3', 's4'):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(['catalog', '--server', server, '--csv']), 0)
                self.assertEqual(output.getvalue(), 'server\n' + server + '\n')
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['catalog', '--server', server]), 0)
        self.assertEqual(fake.catalog_csv.call_count, 3)
        self.assertEqual(fake.default_config.call_count, 3)


class ShellLauncherTests(unittest.TestCase):
    """Copies just launch scripts into a temporary fake checkout for side effects."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        (self.root / 'scripts').mkdir()
        for name in LAUNCHERS:
            shutil.copyfile(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        self.capture = self.root / 'calls.jsonl'
        self.data = self.root / 'registered.h5'
        self.data.touch()
        self.dlpan = self.root / 'DLPan-Toolbox'
        wald = self.dlpan / '01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
        wald.parent.mkdir(parents=True)
        wald.touch()
        self.env = {**os.environ, 'PATH': str(self.bin) + ':' + os.environ['PATH'],
                    'P30_CAPTURE': str(self.capture), 'P30_MOUNT': str(self.data),
                    'PANCRAFTER_DLPAN': str(self.dlpan), 'P30_GPU_BUSY': '0',
                    'P30_EXISTS': '0', 'P30_PREFLIGHT_FAIL': '0',
                    'P30_AUTH_FAIL': '0', 'P30_START_FAIL': '0'}
        # Each fake records all arguments then performs only its narrow stub.
        self.executable('fake-python', '''import json, os, sys
args = sys.argv[1:]
with open(os.environ['P30_CAPTURE'], 'a') as out:
    out.write(json.dumps(['python'] + args) + '\\n')
if args[:3] == ['-m','panda30r','mounts']:
    print(os.environ['P30_MOUNT'])
if args and args[0] == '-c' and 'require_idle_gpu' in args[1]:
    sys.exit(int(os.environ['P30_GPU_BUSY']))
if args and args[0] == '-c' and 'service_account' in args[1]:
    sys.exit(int(os.environ['P30_AUTH_FAIL']))
''')
        self.env['PAN_HOST_PYTHON'] = str(self.bin / 'fake-python')
        self.env['PAN_PYTHON'] = str(self.bin / 'fake-python')
        self.executable('docker', '''import json, os, sys
args = sys.argv[1:]
with open(os.environ['P30_CAPTURE'], 'a') as out:
    out.write(json.dumps(['docker'] + args) + '\\n')
if args[:2] == ['container','inspect']:
    sys.exit(0 if os.environ['P30_EXISTS'] == '1' else 1)
if args[:2] == ['run','--rm']:
    sys.exit(int(os.environ['P30_PREFLIGHT_FAIL']))
if args[:2] == ['run','--detach']:
    sys.exit(int(os.environ['P30_START_FAIL']))
''')
        # Never spawn a real persistent watcher in a unit test.
        self.executable('nohup', '''import json, os, sys
with open(os.environ['P30_CAPTURE'], 'a') as out:
    out.write(json.dumps(['nohup'] + sys.argv[1:]) + '\\n')
''')

    def executable(self, name, source):
        path = self.bin / name
        path.write_text('#!' + sys.executable + '\n' + source)
        path.chmod(0o700)

    def run_script(self, name, *args):
        return subprocess.run(['bash', str(self.root / 'scripts' / name), *args],
                              env=self.env, text=True, capture_output=True, timeout=20)

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def test_shell_syntax_and_no_destructive_commands(self):
        for name in LAUNCHERS:
            path = ROOT / 'scripts' / name
            subprocess.run(['bash', '-n', str(path)], check=True)
            for forbidden in ('docker stop', 'docker rm', 'pkill', 'killall', '--replace'):
                self.assertNotIn(forbidden, path.read_text())

    def test_bad_arguments_stop_before_any_external_command(self):
        for name in LAUNCHERS:
            for args in ((), ('s2',), ('s1', '--replace'), ('s4', '--output', '/tmp/foreign')):
                with self.subTest(name=name, args=args):
                    self.assertEqual(self.run_script(name, *args).returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_host_forwards_each_explicit_server_without_publisher(self):
        for server in ('s1', 's3', 's4'):
            result = self.run_script('run_panda30r.sh', server)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.calls()[-1], ['python', '-m', 'panda30r', 'run', '--server', server])
        self.assertFalse(any('publication' in str(call) for call in self.calls()))

    def test_docker_launch_mounts_and_preflight_order(self):
        result = self.run_script('run_panda30r_docker.sh', 's3')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        docker_runs = [call for call in calls if call[:2] == ['docker', 'run']]
        self.assertEqual(len(docker_runs), 2)
        preflight, launch = docker_runs
        self.assertIn('--rm', preflight)
        self.assertNotIn('--gpus', preflight)
        self.assertEqual(preflight[-6:], ['python', '-m', 'panda30r', 'check', '--server', 's3'])
        self.assertIn('--detach', launch)
        self.assertIn('--gpus', launch)
        self.assertEqual(launch[-3:], ['bash', 'scripts/run_panda30r.sh', 's3'])
        self.assertIn('panda30r-wv3-s3', launch)
        for command in (preflight, launch):
            self.assertIn(f'type=bind,src={self.root},dst=/workspace,readonly', command)
            self.assertIn(f'type=bind,src={self.data},dst={self.data},readonly', command)
            self.assertIn(f'type=bind,src={self.dlpan},dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly', command)
        self.assertNotIn(f'type=bind,src={self.root}/work_dir,dst=/workspace/work_dir', preflight)
        self.assertIn(f'type=bind,src={self.root}/work_dir,dst=/workspace/work_dir', launch)
        gpu_check = next(i for i, call in enumerate(calls) if 'require_idle_gpu' in str(call))
        self.assertLess(calls.index(preflight), gpu_check)
        self.assertLess(gpu_check, calls.index(launch))
        self.assertTrue((self.root / 'work_dir/panda30r/PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1/s3/WV3').is_dir())

    def test_existing_container_refuses_without_preflight_or_start(self):
        self.env['P30_EXISTS'] = '1'
        result = self.run_script('run_panda30r_docker.sh', 's4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Nothing removed or reset', result.stderr)
        self.assertFalse(any(call[:2] == ['docker', 'run'] for call in self.calls()))

    def test_failures_do_not_start_training_or_publisher(self):
        for failure in ('P30_AUTH_FAIL', 'P30_PREFLIGHT_FAIL', 'P30_GPU_BUSY'):
            self.env[failure] = '1'
            before = len(self.calls())
            result = self.run_script('run_panda30r_docker.sh', 's1')
            self.assertNotEqual(result.returncode, 0, failure)
            new = self.calls()[before:]
            self.assertFalse(any(call[:3] == ['docker', 'run', '--detach'] for call in new))
            self.assertFalse(any(call[0] == 'nohup' for call in new))
            self.env[failure] = '0'

    def test_missing_evaluator_and_invalid_mount_fail_before_launch(self):
        for variable, value in (('PANCRAFTER_DLPAN', str(self.root / 'absent')),
                                ('P30_MOUNT', str(self.root / 'missing.h5'))):
            previous = self.env[variable]
            self.env[variable] = value
            before = len(self.calls())
            result = self.run_script('run_panda30r_docker.sh', 's1')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(call[:2] == ['docker', 'run'] for call in self.calls()[before:]))
            self.env[variable] = previous

    def test_failed_detached_start_does_not_start_publisher(self):
        self.env['P30_START_FAIL'] = '1'
        result = self.run_script('run_panda30r_docker.sh', 's4')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'nohup' for call in self.calls()))


if __name__ == '__main__':
    unittest.main()
