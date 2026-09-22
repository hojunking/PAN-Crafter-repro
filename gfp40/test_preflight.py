"""Read-only B20 handoff/source evidence checks (no GPU or runtime launch)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fh12.common import atomic_json, object_sha, sha256
from gfp40.preflight import audit_b20_source, inspect_b20_handoff


class PriorSourceTests(unittest.TestCase):
    def test_runtime_identity_binds_transitive_data_upload_and_archive_helpers(self):
        from gfp40.common import ROOT, source_identity
        identity = source_identity(ROOT)
        for name in ('l100/references.py', 'gspread/gspread_upload.py',
                     'gspread/sheet_categories.py', 'fh12/source_documents.py'):
            self.assertEqual(identity['files'][name], sha256(ROOT / name))

    def fixture(self, root):
        original = root / 'b20-source'
        (original / 'gfb20').mkdir(parents=True)
        (original / 'shared.py').write_text('VALUE = 1\n')
        (original / 'gfb20/losses.py').write_text('old_declared_method = True\n')
        (root / 'gfp40').mkdir()
        (root / 'gfp40/losses.py').write_text('new_distribution_only = True\n')
        files = {name: sha256(original / name) for name in ('shared.py', 'gfb20/losses.py')}
        origin = dict(files=files, content_sha256=object_sha(files),
            git_release='060ba8d209925854f1f3c30496c6366c4cb09016')
        identity = root / 'original-preflight.json'
        atomic_json(identity, dict(source_identity=origin))
        current = dict(files={'shared.py': files['shared.py'],
            'gfp40/losses.py': sha256(root / 'gfp40/losses.py')}, git_release='5d9f4d6')
        return original, identity, current

    def test_actual_archive_evidence_and_diff_not_commit_equivalence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, identity, current = self.fixture(root)
            binding = dict(root=str(original), source_identity_path=str(identity))
            before = sha256(identity)
            with patch('gfp40.assets.NUMERICAL_PATHS', ('shared.py',)), patch(
                    'gfp40.preflight.source_identity', return_value=current), patch(
                    'gfp40.assets.source_compatibility', return_value={'passed': True}):
                result = audit_b20_source(root, 's3', binding)
            self.assertTrue(result['complete'])
            self.assertFalse(result['commit_equivalence_assumed'])
            self.assertIn('060ba8', result['original_recorded_git_release'])
            self.assertIn('-old_declared_method', result['campaign_deltas']['losses.py']['diff'])
            self.assertEqual(before, sha256(identity))
            self.assertFalse(result['previous_campaign_written'])

    def test_missing_archive_or_altered_original_or_core_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'preserve B20'):
                audit_b20_source(root, 's5')
            original, identity, current = self.fixture(root)
            binding = dict(root=str(original), source_identity_path=str(identity))
            with patch('gfp40.assets.NUMERICAL_PATHS', ('shared.py',)), patch(
                    'gfp40.preflight.source_identity', return_value=current):
                current['files']['shared.py'] = 'f' * 64
                with self.assertRaisesRegex(ValueError, 'numerical implementation'):
                    audit_b20_source(root, 's3', binding)
                (original / 'gfb20/losses.py').write_text('altered old code\n')
                with self.assertRaisesRegex(ValueError, 'archive bytes differ'):
                    audit_b20_source(root, 's3', binding)

    def test_local_old_pair_is_only_reported_not_reclassified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'work_dir/GFB20_GF2_S5_C07_NATIVE0_FIXTURE'
            atomic_json(run / 'meta/training_status.json',
                dict(status='RUNNING', actual_updates=100, training_complete=False))
            result = inspect_b20_handoff(root, 's5')
            self.assertEqual(len(result['cases']), 1)
            self.assertFalse(result['cases'][0]['reused_as_p40'])
            self.assertEqual(result['cases'][0]['new_training_count'], 0)
            self.assertFalse(result['previous_campaign_written'])


if __name__ == '__main__':
    unittest.main()
