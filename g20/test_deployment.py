"""Source-only packaging checks; no live cron, startup or git operations."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from g20 import deployment
from g20.plan import CORE_CASES, SOURCE_SHAS
from qg40.plan import SOURCE_SHAS as QG_SOURCE_SHAS


class DeploymentTests(unittest.TestCase):
    def fixture(self, root):
        names = ['g20/__init__.py', 'g20/README.md', 'g20/REFERENCES.md',
            'qg40/model.py', 'qg40/sensor_sources.json', 'qg40/SENSOR_SOURCE_EVIDENCE.md',
            'tools/g20_runner.py', 'tools/g20_start.sh', 'tools/qg40_runner.py',
            'reporting_extra/__init__.py', 'reporting_extra/sensor_sheet.py', 'reporting_extra/sensor_layout.py',
            'config/g20/G20_Registry.json', *SOURCE_SHAS, *QG_SOURCE_SHAS]
        names += ['config/g20/' + case.run_id + '.yaml' for case in CORE_CASES]
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('isolated source fixture\n')
        return set(names)

    def test_bundle_has_core_configs_frozen_dependencies_no_runtime_or_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.fixture(root)
            for name in ['qg40/campaign_window.json', 'work_dir/live/model.safetensors', 'credentials.json']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('DO_NOT_EXPORT')
            with patch('g20.common.source_identity', return_value={'files': {'pa/warp.py': 'a'*64}}), \
                 patch.object(deployment, 'spawn', side_effect=AssertionError('No activation')), \
                 patch.object(deployment, 'install_recovery', side_effect=AssertionError('No cron')):
                result = deployment.bundle(root)
            self.assertFalse(result['activation_performed'])
            with tarfile.open(result['path']) as archive:
                entries = {item.name: archive.extractfile(item).read() for item in archive.getmembers()}
            manifest = json.loads(entries.pop('g20/package_manifest.json'))
            self.assertEqual(set(entries), expected)
            self.assertEqual(manifest['required_existing_files'], {'pa/warp.py': 'a'*64})
            self.assertEqual(manifest['files'], {name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()})
            self.assertFalse((root / 'work_dir/_g20/campaign_window.json').exists())
            self.assertNotIn(b'DO_NOT_EXPORT', b''.join(entries.values()))

    def test_bundle_rejects_symlink_before_reading_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            secret = root / 'credentials.json'
            secret.write_text('secret')
            (root / 'g20/private.py').symlink_to(secret)
            with self.assertRaises(ValueError):
                deployment.bundle(root)

    def test_cron_installer_preserves_other_jobs_and_is_idempotent(self):
        current = ['# Existing experiment job\n* * * * * /srv/old/start\n']
        def run(cmd, **kwargs):
            self.assertEqual(cmd, ['crontab', '-'])
            current[0] = kwargs['input']
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(deployment, '_read_crontab', side_effect=lambda: (True, current[0])), \
             patch.object(deployment.subprocess, 'run', side_effect=run) as execute:
            self.assertEqual(deployment.install_recovery(directory, 's2')['status'], 'INSTALLED')
            self.assertEqual(deployment.install_recovery(directory, 's2')['status'], 'ALREADY_INSTALLED')
            self.assertEqual(execute.call_count, 1)
            self.assertTrue(current[0].startswith('# Existing experiment job\n* * * * * /srv/old/start\n'))
            self.assertNotIn('kill', current[0])
