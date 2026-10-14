"""Server/source/architecture identity and isolated read-only cost proof."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from directshift48.test_reporting import config
from .test_publication import profile
from . import store as S


def measurement(server='s1'):
    value = profile()['measurement']
    value.update(optimizer_updates=0, model_state_sha256_before='a'*64,
                 model_state_sha256_after='a'*64, batch=1, precision='FP32',
                 environment={'device': 'cuda'},
                 gpu_isolation_assertion={'server': server, 'training_state_retained_in_memory': True,
                                          'other_compute_pids': []})
    for split, size in (('rr', 256), ('fr', 512)):
        value['observations'][split]['pan_shape'] = [1, 1, size, size]
    return value


class StoreTests(unittest.TestCase):
    def test_roundtrip_server_local_immutable_archive(self):
        for server in ('s1', 's3'):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                cfg = config(server)
                saved = S.save_profile(cfg, measurement(server), server=server, cost_root=directory)
                found = S.find_profile(cfg, server=server, cost_root=directory)
                self.assertEqual(saved, found)
                self.assertEqual(saved['server'], server)
                self.assertEqual(len(list((Path(directory)/'archive').glob('*.json'))), 1)
                self.assertEqual(saved['signature']['dataset'], 'WV3' if server == 's1' else 'GF2')

    def test_same_graph_different_loss_or_seed_reuses_cost(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            first = S.save_profile(cfg, measurement(), server='s1', cost_root=directory)
            altered = deepcopy(cfg)
            altered.update(seed=1234, lambda_epsilon=.003, lambda_abs=.3, lr_A=.1)
            second = S.find_profile(altered, server='s1', cost_root=directory)
            self.assertEqual(first['profile_sha256'], second['profile_sha256'])

    def test_foreign_server_and_dataset_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            S.save_profile(config(), measurement(), server='s1', cost_root=directory)
            with self.assertRaisesRegex(ValueError, 'server mismatch'):
                S.find_profile(config(), server='s3', cost_root=directory)
            self.assertIsNone(S.find_profile(config('s3'), server='s3', cost_root=directory))

    def test_inference_source_change_invalidates_profile(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            S.save_profile(cfg, measurement(), server='s1', cost_root=directory)
            changed = S.architecture_signature(cfg)
            changed['code_sha256']['directshift48/method.py'] = 'b'*64
            with patch.object(S, 'architecture_signature', return_value=changed):
                self.assertIsNone(S.find_profile(cfg, server='s1', cost_root=directory))

    def test_integrity_tamper_and_wrong_identity_fail_closed(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            item = S.save_profile(cfg, measurement(), server='s1', cost_root=directory)
            path = Path(item['path'])
            original = json.loads(path.read_text())
            tampered = deepcopy(original)
            tampered['measurement']['observations']['rr']['sheet_fields']['Infer(ms)'] = 999.
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                S.find_profile(cfg, server='s1', cost_root=directory)
            tampered = deepcopy(original)
            tampered['server'] = 's3'
            tampered.pop('profile_sha256')
            tampered['profile_sha256'] = S._digest(tampered)
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'architecture/source identity'):
                S.find_profile(cfg, server='s1', cost_root=directory)

    def test_invalid_proofs_are_rejected_before_any_write(self):
        bad = [measurement() for _ in range(8)]
        for key in ('model_state_sha256_before', 'model_state_sha256_after'):
            bad[0].pop(key)
        bad[1]['model_state_sha256_after'] = 'b'*64
        bad[2]['gpu_isolation_assertion']['server'] = 's3'
        bad[3]['gpu_isolation_assertion']['other_compute_pids'] = [1234]
        bad[4]['environment']['device'] = 'cpu'
        bad[5]['optimizer_updates'] = 1
        bad[6]['observations']['rr']['pan_shape'] = [1, 1, 64, 64]
        bad[7]['gpu_isolation_assertion'] = True
        for number, value in enumerate(bad):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    S.save_profile(config(), value, server='s1', cost_root=directory)
                self.assertFalse(list(Path(directory).rglob('*.json')))

    def test_cpu_nullable_memory_is_not_a_production_cost_profile(self):
        value = measurement()
        value['observations']['rr']['sheet_fields']['Mem(MB)'] = None
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            S.save_profile(config(), value, server='s1', cost_root=directory)


if __name__ == '__main__':
    unittest.main()
