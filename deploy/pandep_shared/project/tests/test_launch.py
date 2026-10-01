import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('pan_shared_host_launcher', Path(__file__).resolve().parents[1] / 'launch.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def fixture(self, root, server='s2'):
        project, legacy, storage = root / 'new', root / 'legacy', root / 'storage'
        project.mkdir(); legacy.mkdir(); storage.mkdir()
        (legacy / 'gspread').mkdir()
        (legacy / 'gspread/server.txt').write_text(server + '\n')
        credentials = legacy / 'gspread/account.json'
        credentials.write_text('{"fixture":"not-a-real-credential"}')
        sensors = {}
        for sensor in ('WV3', 'GF2', 'QB'):
            splits = {}
            for split in ('train', 'val', 'rr', 'fr'):
                source = storage / (sensor + '_' + split + '.h5')
                source.write_bytes(b'fixture-source')
                link = legacy / source.name
                link.symlink_to(source)
                splits[split] = {'path': link.name}
            sensors[sensor] = {'splits': splits, 'source_provenance': {}}
        for split in ('train', 'val'):
            source = storage / ('QB_raw_' + split + '.h5')
            source.write_bytes(b'raw-QB-fixture')
            (legacy / source.name).symlink_to(source)
            sensors['QB']['source_provenance']['raw_' + split + '_path'] = source.name
        catalog = project / 'vendor_reference/model_source/ablr2/sensor_sources.json'
        catalog.parent.mkdir(parents=True)
        catalog.write_text(json.dumps({'schema': 'ABLR2_SENSOR_SOURCES_v1', 'sensors': sensors}))
        return project, legacy, credentials, storage

    def test_all_sources_and_credentials_are_ro_and_project_separate(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, storage = self.fixture(Path(name))
            layout = launcher.resolve_layout(legacy, credentials, project=project)
            self.assertEqual(len(layout['sources']), 14)
            command = launcher.build_command('start', layout, stamp='test')
            mounts = [command[i + 1] for i, value in enumerate(command) if value == '--mount']
            for source in storage.iterdir():
                self.assertIn(f'type=bind,src={source},dst={source},readonly', mounts)
            self.assertIn(f'type=bind,src={credentials},dst={credentials},readonly', mounts)
            self.assertIn('--pull=never', command)
            self.assertEqual(command[command.index('--gpus') + 1], 'all')
            self.assertEqual(command[command.index('--pid') + 1], 'host')
            self.assertIn('PYTHONPATH=' + str(project / 'src'), command)
            self.assertNotIn(str(legacy) + '/src', command)

    def test_controls_are_cpu_network_free_and_not_existing_container_exec(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, _, _ = self.fixture(Path(name))
            layout = launcher.resolve_layout(legacy, project=project)
            for verb in ('status', 'pause', 'stop'):
                command = launcher.build_command(verb, layout, stamp=verb)
                self.assertNotIn('--gpus', command)
                self.assertEqual(command[command.index('--network') + 1], 'none')
                self.assertIn('--rm', command)
                self.assertNotIn('exec', command)
                self.assertNotIn('kill', command)
            self.assertIn('--after-current-update', launcher.build_command('stop', layout))

    def test_controls_remain_available_without_raw_data(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, _, storage = self.fixture(Path(name))
            for source in storage.iterdir():
                source.unlink()
            layout = launcher.resolve_layout(legacy, project=project, require_sources=False)
            self.assertEqual(layout['sources'], [])
            self.assertNotIn('--gpus', launcher.build_command('stop', layout))

    def test_child_cli_accepts_every_launcher_command(self):
        from pan_shared.cli import parser
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, _ = self.fixture(Path(name))
            layout = launcher.resolve_layout(legacy, credentials, project=project)
            for verb in launcher.COMMANDS:
                command = launcher.build_command(verb, layout)
                child = command[command.index('pan_shared.cli') + 1:]
                parsed = parser().parse_args(child)
                self.assertEqual(parsed.command, verb)
                self.assertEqual(parsed.work_root, layout['work'])

    def test_resume_is_explicit_detached_gpu_command(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, _ = self.fixture(Path(name))
            layout = launcher.resolve_layout(legacy, credentials, project=project)
            command = launcher.build_command('resume', layout, microbatch=24)
            self.assertIn('--detach', command)
            self.assertIn('--until-operator-stop', command)
            self.assertEqual(command[command.index('--microbatch') + 1], '24')
            self.assertNotIn('--rm', command)
            preserved = launcher.build_command('resume', layout, microbatch=None)
            self.assertNotIn('--microbatch', preserved)

    def test_wrong_server_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, _ = self.fixture(Path(name), server='s1')
            with self.assertRaisesRegex(ValueError, 'BLOCKED_SERVER'):
                launcher.resolve_layout(legacy, credentials, project=project)

    def test_project_overlap_and_credentials_inside_project_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, _ = self.fixture(Path(name))
            with self.assertRaisesRegex(ValueError, 'physically separate'):
                launcher.resolve_layout(legacy, credentials, project=legacy)
            secret = project / 'secret.json'
            secret.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'CREDENTIAL_PATH'):
                launcher.resolve_layout(legacy, secret, project=project)

    def test_work_root_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, storage = self.fixture(Path(name))
            (project / 'work_dir').mkdir()
            (project / 'work_dir' / launcher.CAMPAIGN).symlink_to(storage, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                launcher.resolve_layout(legacy, credentials, project=project)

    def test_image_missing_never_pulls(self):
        with patch.object(launcher.subprocess, 'run') as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ''
            with self.assertRaisesRegex(RuntimeError, 'no automatic pull'):
                launcher.verify_image()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:3], ['docker', 'image', 'inspect'])

    def test_sheet_sync_uses_only_explicit_target(self):
        with tempfile.TemporaryDirectory() as name:
            project, legacy, credentials, _ = self.fixture(Path(name))
            command = launcher.build_command('sync-sheet', launcher.resolve_layout(legacy, credentials, project=project))
            self.assertEqual(command[command.index('--sheet-id') + 1], '1198707876')
            self.assertNotIn('--gpus', command)
            self.assertNotIn('--network', command)

    def test_mount_paths_reject_unrepresentable_delimiters(self):
        with self.assertRaises(ValueError):
            launcher._bind('/tmp/path,readonly', '/tmp/ok', True)
