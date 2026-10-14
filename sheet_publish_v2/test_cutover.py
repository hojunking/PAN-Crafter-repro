import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from . import cutover


class CutoverTests(unittest.TestCase):
    def test_refuse_training_other_user_and_unknown_argv(self):
        base = {'argv': ['/usr/bin/python', '-m', 'edgeres48', 'publish', '--server', 's1', '--watch'],
                'uid': cutover.os.getuid(), 'cwd': '/tmp'}
        cutover.verify_publisher(base, 's1', '/tmp')
        for change in ({'argv': ['/usr/bin/python', '-m', 'edgeres48', 'run', '--server', 's1']},
                       {'uid': -1}, {'cwd': '/'},
                       {'argv': base['argv'] + ['--output', '/tmp']}):
            with self.assertRaises(RuntimeError):
                cutover.verify_publisher({**base, **change}, 's1', '/tmp')
        with self.assertRaises(RuntimeError):
            cutover.verify_publisher(base, 's2', '/tmp')

    def test_backup_contains_original_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root, audit = repo / 'lane', repo / 'audit'
            root.mkdir()
            (repo / 'watched.py').write_text('unchanged\n')
            pin = {'watched.py': hashlib.sha256(b'unchanged\n').hexdigest()}
            (root / 'admission.json').write_text(json.dumps({'code': pin}))
            (root / 'records.jsonl').write_bytes(b'{"a":1}\n')
            (root / 'outbox').mkdir()
            (root / 'outbox' / 'EGR2-example.json').write_bytes(b'{"a":1}\n')
            path, manifest = cutover.backup_sources(root, audit, repo)
            self.assertTrue(path.is_file())
            with tarfile.open(manifest['archive']) as archive:
                for item in manifest['files']:
                    data = archive.extractfile(item['path']).read()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
                    self.assertEqual(data, (root / item['path']).read_bytes())
            self.assertEqual(pin, cutover.code_proof(root, repo))
            (repo / 'watched.py').write_text('modified')
            with self.assertRaises(RuntimeError):
                cutover.code_proof(root, repo)

    def test_pid_reuse_prevents_signal(self):
        expected = {'pid': 100, 'start_ticks': 10,
                    'argv': ['/usr/bin/python', '-m', 'edgeres48', 'publish', '--server', 's1', '--watch'],
                    'uid': cutover.os.getuid(), 'cwd': '/tmp'}
        with mock.patch.object(cutover.os, 'pidfd_open', return_value=42, create=True), \
             mock.patch.object(cutover.os, 'close'), \
             mock.patch.object(cutover, 'process_identity', return_value={**expected, 'start_ticks': 11}), \
             mock.patch.object(cutover.signal, 'pidfd_send_signal', create=True) as send:
            with self.assertRaises(RuntimeError):
                cutover.stop_verified(100, expected, server='s1', repo='/tmp')
            send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
