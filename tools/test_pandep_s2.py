"""Stdlib-only isolated tests; no real project, data, sheet, GPU, or Docker writes."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('pandep_bootstrap', Path(__file__).with_name('pandep_s2.py'))
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.repo = self.parent / 'legacy'
        self.project = self.repo / 'deploy/pandep_shared/project'
        self.project.mkdir(parents=True)
        (self.repo / 'gspread').mkdir()
        (self.repo / 'gspread/server.txt').write_text('s2\n')
        (self.project / 'src/pan_shared').mkdir(parents=True)
        (self.project / 'src/pan_shared/__init__.py').write_text('# frozen fixture\n')
        (self.project / 'launch.py').write_text('# frozen launcher fixture\n')
        (self.project / 'verification').mkdir()
        source = {'schema': 'PANDEP_SOURCE_v1', 'legacy_source_commit': 'fixture',
                  'files': {str(path.relative_to(self.project)): bootstrap.file_sha(path)
                            for path in self.project.rglob('*.py')}}
        source['source_sha256'] = bootstrap.canonical_sha(source)
        receipt = {'schema': 'PANDEP_LOCAL_CPU_VERIFICATION_v1', 'status': 'PASS',
                   'source': source, 'unit_tests': {'success': True, 'test_count': 1, 'returncode': 0,
                                                  'errors': [], 'failures': [], 'skipped': []}}
        (self.project / 'verification/local_cpu_tests.json').write_text(json.dumps(receipt))
        self.manifest = {'schema': 'PANDEP_GIT_DISTRIBUTION_v1', 'campaign_id': bootstrap.CAMPAIGN,
                         'source_sha256': source['source_sha256'],
                         'files': {str(path.relative_to(self.project)): {
                             'sha256': bootstrap.file_sha(path), 'size_bytes': path.stat().st_size}
                             for path in self.project.rglob('*') if path.is_file()}}
        self.manifest_path = self.project.parent / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.target = self.parent / 'pan_deploy_shared'

    def run_main(self, command, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = bootstrap.main([command, '--legacy-root', str(self.repo), *args])
        return code, out.getvalue(), err.getvalue()

    def test_verified_snapshot_and_idempotent_deployment_preserves_generated_files(self):
        project, manifest = bootstrap.read_distribution(self.repo)
        self.assertTrue(bootstrap.deploy(project, manifest, self.target))
        work = self.target / 'work_dir/experiment'
        work.mkdir(parents=True)
        (work / 'checkpoint.bin').write_bytes(b'owned experiment')
        (self.target / 'manual-note.txt').write_text('keep')
        self.assertFalse(bootstrap.deploy(project, manifest, self.target))
        self.assertEqual((work / 'checkpoint.bin').read_bytes(), b'owned experiment')
        self.assertEqual((self.target / 'manual-note.txt').read_text(), 'keep')

    def test_changed_existing_source_is_not_overwritten(self):
        bootstrap.deploy(self.project, self.manifest, self.target)
        (self.target / 'launch.py').write_text('operator edits')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'differs'):
            bootstrap.deploy(self.project, self.manifest, self.target)
        self.assertEqual((self.target / 'launch.py').read_text(), 'operator edits')

    def test_existing_untracked_python_import_hook_or_package_rejected(self):
        bootstrap.deploy(self.project, self.manifest, self.target)
        for relative in ('sitecustomize.py', 'argparse.pyc', 'src/pan_shared/extra.py', 'argparse/__init__.py'):
            with self.subTest(relative=relative):
                path = self.target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# unverified code')
                with self.assertRaisesRegex(bootstrap.DeploymentError, 'unverified executable/source'):
                    bootstrap.verify_files(self.target, self.manifest)
                path.unlink()

    def test_existing_untracked_vendor_file_rejected_but_artifacts_and_cache_preserved(self):
        bootstrap.deploy(self.project, self.manifest, self.target)
        for relative in ('work_dir/export/model.py', 'dist/candidate.py', '__pycache__/launch.pyc',
                         'src/pan_shared/__pycache__/__init__.pyc', 'manual-notes.txt'):
            path = self.target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('keep generated artifact')
        bootstrap.verify_files(self.target, self.manifest)
        vendor = self.target / 'vendor_reference/unregistered.json'
        vendor.parent.mkdir(parents=True)
        vendor.write_text('{}')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'unverified executable/source'):
            bootstrap.verify_files(self.target, self.manifest)
        self.assertEqual((self.target / 'work_dir/export/model.py').read_text(), 'keep generated artifact')

    def test_corrupt_payload_rejected_before_copy(self):
        (self.project / 'launch.py').write_text('corrupt')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'differs'):
            bootstrap.read_distribution(self.repo)
        self.assertFalse(self.target.exists())

    def test_extra_payload_file_rejected(self):
        (self.project / 'secret.json').write_text('not allowed')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'inventory'):
            bootstrap.read_distribution(self.repo)

    def test_source_receipt_must_match_even_with_updated_file_manifest(self):
        (self.project / 'launch.py').write_text('different code')
        self.manifest['files']['launch.py'] = {'sha256': bootstrap.file_sha(self.project / 'launch.py'),
                                              'size_bytes': (self.project / 'launch.py').stat().st_size}
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'source receipt hash'):
            bootstrap.read_distribution(self.repo)

    def test_zero_test_receipt_rejected(self):
        path = self.project / 'verification/local_cpu_tests.json'
        value = json.loads(path.read_text())
        value['unit_tests']['test_count'] = 0
        path.write_text(json.dumps(value))
        self.manifest['files']['verification/local_cpu_tests.json'] = {
            'sha256': bootstrap.file_sha(path), 'size_bytes': path.stat().st_size}
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'receipt required'):
            bootstrap.read_distribution(self.repo)

    def test_traversal_and_noncanonical_relative_names_rejected(self):
        for value in ('../escape', '/absolute', 'a/../b', './a', 'a//b', 'a\\b', '', '.'):
            with self.subTest(value=value), self.assertRaises(bootstrap.DeploymentError):
                bootstrap.safe_relative(value)

    def test_symlink_payload_and_tracked_target_rejected(self):
        source = self.project / 'launch.py'
        source.unlink()
        source.symlink_to('/etc/hosts')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'symlink'):
            bootstrap.read_distribution(self.repo)
        self.target.symlink_to(self.project, target_is_directory=True)
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'sibling'):
            bootstrap.resolve_target(self.repo, self.target)

    def test_descendant_ancestor_and_legacy_targets_rejected(self):
        for target in (self.repo, self.repo / 'new-project', self.parent, Path('/')):
            with self.subTest(target=target), self.assertRaises(bootstrap.DeploymentError):
                bootstrap.resolve_target(self.repo, target)

    def test_publish_race_never_replaces_existing_target(self):
        staged = self.parent / 'private-stage'
        staged.mkdir()
        (staged / 'new').write_text('payload')
        self.target.mkdir()
        (self.target / 'operator-data').write_text('preserve')
        with self.assertRaisesRegex(bootstrap.DeploymentError, 'not replaced'):
            bootstrap.publish_noreplace(staged, self.target)
        self.assertEqual((self.target / 'operator-data').read_text(), 'preserve')
        self.assertTrue((staged / 'new').exists())

    def test_dry_run_on_s1_has_no_writes_or_subprocess_and_reports_blocker(self):
        (self.repo / 'gspread/server.txt').write_text('s1\n')
        before = sorted(str(path) for path in self.parent.rglob('*'))
        with mock.patch.object(bootstrap.subprocess, 'run') as child:
            code, out, err = self.run_main('start', '--dry-run')
        self.assertEqual(code, 0, err)
        plan = json.loads(out)
        self.assertTrue(plan['server_would_block'])
        self.assertTrue(plan['would_deploy'])
        self.assertFalse(plan['writes'])
        self.assertEqual(before, sorted(str(path) for path in self.parent.rglob('*')))
        child.assert_not_called()

    def test_real_start_on_wrong_server_refused_before_deploy(self):
        (self.repo / 'gspread/server.txt').write_text('s1')
        with mock.patch.object(bootstrap.subprocess, 'run') as child:
            code, out, err = self.run_main('start')
        self.assertEqual(code, 2)
        self.assertIn('BLOCKED_SERVER', err)
        self.assertFalse(self.target.exists())
        child.assert_not_called()

    def test_start_uses_verified_sibling_and_default_credentials_no_imports(self):
        with mock.patch.object(bootstrap.subprocess, 'run', return_value=mock.Mock(returncode=0)) as child:
            code, out, err = self.run_main('start')
        self.assertEqual(code, 0, err)
        child.assert_called_once()
        args = child.call_args.args[0]
        self.assertEqual(args[1], str(self.target / 'launch.py'))
        self.assertEqual(args[-1], 'start')
        self.assertEqual(args[args.index('--credentials') + 1], str(self.repo / 'gspread/account.json'))
        self.assertEqual(child.call_args.kwargs['cwd'], self.target)
        self.assertFalse((self.target / 'gspread').exists())

    def test_controls_never_deploy_absent_target(self):
        for command in ('status', 'pause', 'resume', 'stop', 'sync-sheet'):
            with self.subTest(command=command), mock.patch.object(bootstrap.subprocess, 'run') as child:
                code, out, err = self.run_main(command)
                self.assertEqual(code, 2)
                self.assertIn('NOT_DEPLOYED', err)
                self.assertFalse(self.target.exists())
                child.assert_not_called()

    def test_resume_preserves_microbatch_and_control_commands_omit_secrets(self):
        command = bootstrap.launcher_argv('resume', self.repo, self.target)
        self.assertNotIn('--microbatch', command)
        for name in ('status', 'pause', 'stop'):
            command = bootstrap.launcher_argv(name, self.repo, self.target)
            self.assertNotIn('--credentials', command)
        command = bootstrap.launcher_argv('resume', self.repo, self.target, microbatch=24)
        self.assertEqual(command[-3:], ['resume', '--microbatch', '24'])

    def test_launcher_failure_return_code_is_preserved(self):
        with mock.patch.object(bootstrap.subprocess, 'run', return_value=mock.Mock(returncode=7)):
            code, out, err = self.run_main('start')
        self.assertEqual(code, 7)


if __name__ == '__main__':
    unittest.main()
