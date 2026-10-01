import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
from unittest.mock import patch
from panda_rb_m12 import transfer
from panda_rb_m12.plan import CAMPAIGN
from panda_rb_m12.common import sha256


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'work_dir/_panda_rb/20261001/B02_M12/train/s3/R1/QFULL'
        self.row = dict(server='s3', run_id='registered')
        self.a = patch.object(transfer, 'training_runs', return_value=[self.row]); self.a.start(); self.addCleanup(self.a.stop)
        self.b = patch.object(transfer, 'run_dir', return_value=self.base); self.b.start(); self.addCleanup(self.b.stop)

    def bundle(self, relative=None, content=b'{}', duplicate=False, wrong_hash=False):
        relative = relative or str((self.base / 'native/metrics.json').relative_to(self.root))
        manifest = dict(schema='PANDA_M12_RETURN_v1', campaign_id=CAMPAIGN,
            report=dict(servers=['s3'], completed_students=1),
            files={relative: 'a'*64 if wrong_hash else hashlib.sha256(content).hexdigest()})
        path = self.root / 'return.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('manifest.json', json.dumps(manifest)); z.writestr(relative, content)
            if duplicate:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); z.writestr(relative, content)
        return path

    def test_explicit_activation(self):
        with self.assertRaises(PermissionError):
            transfer.import_package('absent', 'a'*64, self.root)

    def test_verified_import_idempotent(self):
        path = self.bundle()
        first = transfer.import_package(path, sha256(path), self.root, True)
        second = transfer.import_package(path, sha256(path), self.root, True)
        self.assertEqual(first['added_files'], 1); self.assertEqual(second['added_files'], 0)

    def test_mismatched_sender_sha(self):
        path = self.bundle()
        with self.assertRaises(ValueError):
            transfer.import_package(path, 'f'*64, self.root, True)

    def test_corrupt_member(self):
        path = self.bundle(wrong_hash=True)
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)
        self.assertFalse(self.base.exists())

    def test_zip_traversal(self):
        path = self.bundle('../escaped.json')
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)

    def test_duplicates(self):
        path = self.bundle(duplicate=True)
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)

    def test_no_overwrite(self):
        path = self.bundle(); target = self.base / 'native/metrics.json'
        target.parent.mkdir(parents=True); target.write_text('original')
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)
        self.assertEqual(target.read_text(), 'original')

    def test_no_executable(self):
        path = self.bundle(str((self.base / 'meta/payload.py').relative_to(self.root)))
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)

    def test_counts_tensor_copied_without_deserialization(self):
        relative = str((self.base / 'diagnostics/consumed_sample_view_counts.pt').relative_to(self.root))
        path = self.bundle(relative, b'opaque counted-view tensor bytes')
        self.assertEqual(transfer.import_package(path, sha256(path), self.root, True)['added_files'], 1)

    def test_resume_state_still_forbidden(self):
        relative = str((self.base / 'diagnostics/training_state.pt').relative_to(self.root))
        path = self.bundle(relative)
        with self.assertRaises(ValueError):
            transfer.import_package(path, sha256(path), self.root, True)


if __name__ == '__main__':
    unittest.main()
