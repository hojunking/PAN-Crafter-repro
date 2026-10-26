"""Host inspection must preserve the signed container receipt and verify bytes."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_hard36.donor import inspect_source_artifact


class DonorInspectionTests(unittest.TestCase):
    def test_container_path_maps_without_mutating_receipt_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / 'work_dir/donor.pt'
            local.parent.mkdir()
            local.write_bytes(b'original donor bytes')
            receipt = {'selected': {
                'checkpoint_path': '/workspace/work_dir/donor.pt',
                'file_sha256': hashlib.sha256(local.read_bytes()).hexdigest()}}
            unchanged = deepcopy(receipt)
            # Receipt-schema validation is covered by the immutable selector;
            # this fixture isolates path resolution and actual file hashing.
            with patch('panda_hard36.donor.validate_receipt', side_effect=lambda r: r):
                self.assertEqual(inspect_source_artifact(receipt, directory), local)
                self.assertEqual(receipt, unchanged)
                local.write_bytes(b'tampered donor bytes')
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    inspect_source_artifact(receipt, directory)
                local.unlink()
                with self.assertRaisesRegex(ValueError, 'missing'):
                    inspect_source_artifact(receipt, directory)
