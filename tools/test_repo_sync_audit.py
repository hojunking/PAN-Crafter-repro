"""Synthetic repositories only; never touch the live repository index."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools import repo_sync_audit as audit


class SyncAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'synthetic@example.invalid')
        self.git('config', 'user.name', 'Synthetic tests')
        self.write('model.py', 'original\n')
        self.write('research_log/plan.md', 'immutable plan\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'synthetic baseline')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path

    def test_same_head_does_not_hide_local_changes_or_extra_code(self):
        before = audit.inventory(self.root)
        self.write('model.py', 'modified\n')
        self.write('new.py', 'extra\n')
        after = audit.inventory(self.root)
        diff = audit.compare(before, after)
        self.assertTrue(diff['same_head'])
        self.assertFalse(diff['same_shared_files'])
        self.assertEqual(diff['changed'], ['model.py'])
        self.assertEqual(diff['right_only'], ['new.py'])

    def test_credentials_data_models_and_server_marker_are_excluded(self):
        for name in ('gspread/account.json', 'gspread/server.txt', '.env', 'data/t.py',
                     'work_dir/status.json', 'analysis_runs/a.py', 'model.safetensors',
                     'config/_po10_gate_N3.yaml'):
            self.write(name, 'private')
            self.assertIsNone(audit.shared_kind(name))
        value = audit.inventory(self.root)
        self.assertEqual(len(value['files']), 2)

    def test_links_are_hashed_without_reading_targets(self):
        target = self.root / 'outside_secret'
        target.write_text('not read')
        os.symlink(target, self.root / 'linked.py')
        first = audit.inventory(self.root)['files']['linked.py']
        target.write_text('different secret')
        second = audit.inventory(self.root)['files']['linked.py']
        self.assertEqual(first, second)
        self.assertEqual(first['kind'], 'symlink')

    def test_exact_moves_create_relative_links_with_original_content(self):
        old = self.root / 'research_log/plan.md'
        moved = self.root / 'research_log/past/plan.md'
        moved.parent.mkdir()
        old.rename(moved)
        self.assertIn('research_log/plan.md', audit.review(self.root)['exact_content_relocations'])
        result = audit.compatibility_links(self.root)
        self.assertEqual(result['created'], 1)
        self.assertEqual(old.read_text(), 'immutable plan\n')
        self.assertEqual(os.readlink(old), 'past/plan.md')
        self.assertEqual(audit.compatibility_links(self.root)['created'], 0)

    def test_changed_archive_is_not_an_exact_move(self):
        (self.root / 'research_log/plan.md').unlink()
        self.write('research_log/past/plan.md', 'different')
        result = audit.compatibility_links(self.root)
        self.assertEqual(result['created'], 0)
        self.assertEqual(result['unmatched'], ['research_log/plan.md'])

    def test_secret_scan_reports_only_path_and_flag(self):
        self.write('bad.json', json.dumps({'refresh_token': 'synthetic-placeholder-token-0123456789'}))
        value = audit.inventory(self.root)
        self.assertEqual(value['summary']['secret_pattern_flags'], ['bad.json'])
        self.assertNotIn('synthetic-placeholder', json.dumps(value))

    def test_environment_files_are_included_and_mode_changes_detected(self):
        for name in ('Dockerfile.env', '.dockerignore', 'LICENSE',
                     'research_log/report.html', 'results_log/manifest.sha256'):
            self.assertIsNotNone(audit.shared_kind(name))
        before = audit.inventory(self.root)
        (self.root / 'model.py').chmod(0o755)
        self.assertEqual(audit.compare(before, audit.inventory(self.root))['changed'], ['model.py'])

    def test_inventory_output_never_overwrites_an_existing_file(self):
        target = self.write('prior.json', '{}')
        with self.assertRaises(SystemExit):
            audit.main(['--root', str(self.root), 'inventory', '--output', str(target)])
        self.assertEqual(target.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
