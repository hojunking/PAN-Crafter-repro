"""CPU-only checks of explicitly authorized s1/R0 variance acceptance."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fh12.common import atomic_json, object_sha, read_json, sha256
from fh12.data import AUGMENTATION, RECIPE
from g20 import gpu_exception as ex
from g20 import references as ref
from g20.common import CAMPAIGN_ID, camp
from qg40.references import _probe
from qg40.test_references import ProbeDataset, ProbeModel


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.release = dict(files={'g20/gpu_exception.py': 'release'}, content_sha256='release')
        mocked = patch.object(ex, 'source_identity', return_value=self.release)
        mocked.start(); self.addCleanup(mocked.stop)
        self.origin = dict(schema='QG40_REFERENCE_v1', reference_id='GF2_TA', server='s3',
            teacher_run_id='original-r0', teacher_checkpoint_sha256='checkpoint',
            q_cache_sha256='q-cache', q_shape=[3072, 4], source_identity=dict(content_sha256='origin-source'))
        mocked = patch.dict(ref.R0_PINS, teacher_checkpoint_sha256='checkpoint',
            q_cache_sha256='q-cache', calibration_id=object_sha(self.origin))
        mocked.start(); self.addCleanup(mocked.stop)
        registration = camp(self.root, 's1') / 'registration.json'
        atomic_json(registration, dict(source_identity='original', clock='unchanged'))
        self.auth = dict(schema=ex.AUTH_SCHEMA, campaign_id=CAMPAIGN_ID, server_id='s1',
            reference_id='R0', user_approved=True, approval_text='승인한다',
            original_registration_sha256=sha256(registration), consumer_source_identity=self.release,
            gpu_name=ex.GPU_NAME, teacher_checkpoint_sha256='checkpoint', q_cache_sha256='q-cache',
            origin_manifest_sha256=object_sha(self.origin), policy=copy.deepcopy(ex.POLICY))
        atomic_json(ex.authorization_path(self.root), self.auth)
        self.local = dict(sensor='GF2', num_bands=4, max_pixel=1023, mtf_sensor='GF2',
            band_order=['B', 'G', 'R', 'NIR'], recipe=RECIPE, augmentation=AUGMENTATION,
            splits={split: dict(count=3072, sha256='data', lpan_sha256='lp', shapes={},
                sample_order_sha256='order') for split in ('train', 'val', 'rr', 'fr')})
        self.q = np.full((3072, 4), .46873, dtype=np.float32)

    def test_missing_authorization_returns_none_and_cannot_measure(self):
        ex.authorization_path(self.root).unlink()
        self.assertIsNone(ex.authorization(self.root, 's1', self.origin))
        with self.assertRaisesRegex(ValueError, 'authorization is required'):
            ex._bindings(self.q, self.origin, self.local, self.root, 's1', None)

    def test_exact_authorization_and_read_only_cpu_validation(self):
        with patch.object(torch.cuda, 'get_device_name') as device:
            self.assertEqual(ex.authorization(self.root, 's1', self.origin), self.auth)
        device.assert_not_called()
        with patch.object(torch.cuda, 'get_device_name', return_value=ex.GPU_NAME):
            self.assertEqual(ex.authorization(self.root, 's1', self.origin, 'cuda'), self.auth)

    def test_authorization_rejects_scope_source_policy_or_approval_changes(self):
        variations = [('campaign_id', 'another'), ('server_id', 's2'), ('reference_id', 'R1'),
            ('user_approved', False), ('approval_text', ''), ('original_registration_sha256', 'changed'),
            ('consumer_source_identity', {'content_sha256': 'different'}), ('gpu_name', 'A100'),
            ('teacher_checkpoint_sha256', 'changed'), ('q_cache_sha256', 'changed'),
            ('origin_manifest_sha256', 'changed'), ('policy', {'strict_atol': 1.})]
        for key, value in variations:
            with self.subTest(field=key):
                atomic_json(ex.authorization_path(self.root), dict(self.auth, **{key: value}))
                with self.assertRaisesRegex(ValueError, 'authorization identity/policy'):
                    ex.authorization(self.root, 's1', self.origin)

    def test_wrong_server_and_wrong_original_reference_are_rejected(self):
        atomic_json(ex.authorization_path(self.root, 's2'), self.auth)
        with self.assertRaisesRegex(ValueError, 'pinned original s1/R0'):
            ex.authorization(self.root, 's2', self.origin)
        for key, value in [('reference_id', 'R1'), ('server', 's1'),
                           ('teacher_checkpoint_sha256', 'altered'), ('q_cache_sha256', 'altered'),
                           ('teacher_run_id', 'altered')]:
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'pinned original s1/R0'):
                ex.authorization(self.root, 's1', dict(self.origin, **{key: value}))

    def test_cuda_gpu_identity_is_mandatory_when_consumed(self):
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            ex.authorization(self.root, 's1', self.origin, 'cpu')
        with patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA A100'):
            with self.assertRaisesRegex(ValueError, 'GPU differs'):
                ex.authorization(self.root, 's1', self.origin, 'cuda')
        with self.assertRaises((ValueError, TypeError)):
            ex._gpu(None)

    def test_changed_registration_and_current_source_are_rejected(self):
        with patch.object(ex, 'source_identity', return_value={'files': {'changed': 'sha'}}):
            with self.assertRaisesRegex(ValueError, 'authorization identity/policy'):
                ex.authorization(self.root, 's1', self.origin)
        atomic_json(camp(self.root, 's1') / 'registration.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'authorization identity/policy'):
            ex.authorization(self.root, 's1', self.origin)

    def test_second_authorization_preserves_and_binds_first_and_metric_approval(self):
        original_path = ex.authorization_path(self.root, version=1)
        original_bytes = original_path.read_bytes()
        metric_path = camp(self.root, 's1') / 'metric_exception_authorization.json'
        metric = dict(consumer_source_identity=self.release, user_approved=True)
        atomic_json(metric_path, metric)
        second_path = ex.authorization_path(self.root)
        self.assertEqual(second_path.name, 'parity_exception_authorization_v2.json')
        second = dict(self.auth, previous_authorization_sha256=sha256(original_path),
                      metric_authorization_sha256=sha256(metric_path), policy=ex.policy_for(self.root))
        atomic_json(second_path, second)
        with patch('g20.metric_exception.authorization', return_value=metric):
            self.assertEqual(ex.authorization(self.root, 's1', self.origin), second)
            self.assertEqual(original_path.read_bytes(), original_bytes)
            for key in ('previous_authorization_sha256', 'metric_authorization_sha256'):
                atomic_json(second_path, dict(second, **{key: '0' * 64}))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'supplemental'):
                    ex.authorization(self.root, 's1', self.origin)
            atomic_json(second_path, second)
            metric_path.unlink()
            self.assertEqual(ex.authorization_path(self.root), second_path)
            with self.assertRaises(FileNotFoundError):
                ex.authorization(self.root, 's1', self.origin)

    def test_second_authorization_does_not_ignore_invalid_metric_approval(self):
        metric_path = camp(self.root, 's1') / 'metric_exception_authorization.json'
        atomic_json(metric_path, {'user_approved': False})
        atomic_json(ex.authorization_path(self.root), dict(self.auth, policy=ex.policy_for(self.root)))
        with patch('g20.metric_exception.authorization', side_effect=ValueError('invalid approval')):
            with self.assertRaisesRegex(ValueError, 'invalid approval'):
                ex.authorization(self.root, 's1', self.origin)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        AuthorizationTests.setUp(self)
        self.dataset, self.model = ProbeDataset(), ProbeModel()
        self.original_probe = _probe(self.model, self.dataset, 'cpu')
        self.original_probe['output'] -= np.float32(1e-4)
        self.original_probe['delta'] -= np.float32(2e-5)
        bypass = patch.object(ex, '_gpu', return_value=ex.GPU_NAME)
        bypass.start(); self.addCleanup(bypass.stop)

    def measure(self):
        return ex.measure(self.model, self.dataset, self.q, self.original_probe,
            self.origin, self.local, self.root, 's1', 'cpu')

    def validate(self, receipt):
        return ex.validate(receipt, self.q, self.origin, self.local, self.root, 's1')

    def verify(self, receipt):
        return ex.verify_local(self.model, self.dataset, self.q, self.original_probe,
            receipt, self.origin, self.local, self.root, 's1', 'cpu')

    def test_measured_exception_keeps_raw_failed_comparisons_and_original_values(self):
        old_q = self.q.copy()
        old_probe = {key: value.copy() for key, value in self.original_probe.items()}
        receipt, local_probe = self.measure()
        self.assertFalse(receipt['passed'])
        self.assertEqual(receipt['status'], 'ACCEPTED_WITH_PARITY_EXCEPTION')
        self.assertEqual(receipt['policy']['strict_atol'], 3e-6)
        self.assertEqual(receipt['policy']['strict_rtol'], 2e-5)
        for key in ('q', 'output', 'delta'):
            self.assertEqual(receipt['historical_comparison'][key]['status'], 'FAILED')
            self.assertFalse(receipt['historical_comparison'][key]['passed'])
            self.assertGreater(receipt['historical_comparison'][key]['failed_count'], 0)
            self.assertTrue(receipt['same_local_repeat'][key]['passed'])
        np.testing.assert_array_equal(self.q, old_q)
        for key, value in old_probe.items():
            np.testing.assert_array_equal(self.original_probe[key], value)
        self.assertNotEqual(ex.probe_identity(local_probe), ex.probe_identity(old_probe))
        self.assertIs(self.validate(receipt), receipt)
        self.assertTrue(self.verify(receipt)['passed'])
        self.assertTrue(ex.verify_probe_binding(receipt, old_probe))

    def test_full_train_count_is_distinct_from_calibration_subset_count(self):
        full_q = np.full((19809, 4), .46873, dtype=np.float32)
        local = copy.deepcopy(self.local)
        local['splits']['train']['count'] = 19809
        origin = dict(self.origin, q_shape=[19809, 4])
        np.testing.assert_array_equal(ex._q_inputs(full_q, local, origin), ex.parity_indices(19809))
        for count in (16, 3072, 19808):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'full-train'):
                ex._q_inputs(full_q[:count], local, origin)

    def test_forged_historical_pass_and_invented_measurements_fail(self):
        receipt, _ = self.measure()
        modifications = [('passed', True), ('status', 'PASSED'), ('accepted', False),
                         ('local_data_identity', 'changed'), ('authorization_sha256', 'changed')]
        for key, value in modifications:
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'identity/policy/status'):
                self.validate(dict(receipt, **{key: value}))
        for group in ('historical_comparison', 'same_local_repeat', 'array_identity', 'protocol'):
            bad = copy.deepcopy(receipt)
            if group == 'historical_comparison':
                bad[group]['q'].update(passed=True, status='PASSED', failed_count=0)
            elif group == 'same_local_repeat':
                bad[group]['q']['max_abs_error'] = 1e-8
            elif group == 'array_identity':
                bad[group]['local_q']['sha256'] = 'invented'
            else:
                bad[group]['indices'][0] = (bad[group]['indices'][0] + 1) % 3072
            with self.subTest(group=group), self.assertRaises(ValueError):
                self.validate(bad)

    def test_historical_caps_finite_values_and_same_local_repeat_are_enforced(self):
        receipt, _ = self.measure()
        arrays = ex._read_arrays(receipt, self.root, 's1')
        for key in ('q', 'output', 'delta'):
            bad = {name: value.copy() for name, value in arrays.items()}
            bad['local_' + key] = bad['historical_' + key] + np.float32(ex.POLICY['max_abs_error'][key] * 2)
            bad['repeat_' + key] = bad['local_' + key].copy()
            with self.subTest(cap=key), self.assertRaisesRegex(ValueError, 'variance cap exceeded'):
                ex._summaries(bad)
            bad = {name: value.copy() for name, value in arrays.items()}
            bad['repeat_' + key] += np.float32(.01)
            with self.subTest(repeat=key), self.assertRaisesRegex(ValueError, 'same-local repeat failed'):
                ex._summaries(bad)
        for key in ('historical_q', 'local_q', 'repeat_q', 'pan', 'historical_output', 'local_delta'):
            bad = {name: value.copy() for name, value in arrays.items()}
            bad[key].flat[0] = np.nan
            with self.subTest(nonfinite=key), self.assertRaises(ValueError):
                ex._summaries(bad)

    def test_all_historical_pass_does_not_create_exception(self):
        self.q[:] = np.float32(.46875)
        self.original_probe = _probe(self.model, self.dataset, 'cpu')
        with self.assertRaisesRegex(ValueError, 'historical FAILED'):
            self.measure()
        self.assertFalse(ex._measurement_path(self.root, 's1').exists())

    def test_repeat_must_also_remain_within_historical_variance_caps(self):
        receipt, _ = self.measure()
        arrays = ex._read_arrays(receipt, self.root, 's1')
        arrays['local_q'] = arrays['historical_q'] + np.float32(4.8e-5)
        arrays['repeat_q'] = arrays['historical_q'] + np.float32(5.2e-5)
        self.assertTrue(ex._comparison(arrays['repeat_q'], arrays['local_q'])['passed'])
        with self.assertRaisesRegex(ValueError, 'historical variance cap exceeded: q'):
            ex._summaries(arrays)

    def test_second_policy_accepts_recorded_runtime_delta_without_changing_first_policy(self):
        self.original_probe['delta'] -= np.float32(1e-4)
        with self.assertRaisesRegex(ValueError, 'variance cap exceeded: delta'):
            self.measure()
        original_auth_path = ex.authorization_path(self.root, version=1)
        metric_path = camp(self.root, 's1') / 'metric_exception_authorization.json'
        metric = dict(consumer_source_identity=self.release, user_approved=True)
        atomic_json(metric_path, metric)
        second = dict(self.auth, policy=ex.policy_for(self.root),
                      previous_authorization_sha256=sha256(original_auth_path),
                      metric_authorization_sha256=sha256(metric_path))
        atomic_json(ex.authorization_path(self.root), second)
        with patch('g20.metric_exception.authorization', return_value=metric):
            receipt, _ = self.measure()
            self.assertEqual(receipt['policy']['max_abs_error']['delta'], 2e-4)
            self.assertEqual(ex.POLICY['max_abs_error']['delta'], 5e-5)
            self.assertFalse(receipt['historical_comparison']['delta']['passed'])
            self.assertTrue(self.verify(receipt)['passed'])
            self.assertIs(self.validate(receipt), receipt)
            arrays = ex._read_arrays(receipt, self.root, 's1')
            arrays['local_delta'] = arrays['historical_delta'] + np.float32(2.1e-4)
            arrays['repeat_delta'] = arrays['local_delta'].copy()
            with self.assertRaisesRegex(ValueError, 'variance cap exceeded: delta'):
                ex._summaries(arrays, ex.POLICY_V2)

    def test_changed_probe_cache_or_measurement_file_cannot_be_consumed(self):
        receipt, _ = self.measure()
        changed_probe = {key: value.copy() for key, value in self.original_probe.items()}
        changed_probe['output'].flat[0] += .1
        with self.assertRaisesRegex(ValueError, 'original portable probe changed'):
            ex.verify_probe_binding(receipt, changed_probe)
        self.q[ex.parity_indices(len(self.q))[0], 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'measurements or historical failure'):
            self.validate(receipt)
        self.q[ex.parity_indices(len(self.q))[0], 0] = np.float32(.46873)
        wrong = dict(receipt, measurements_sha256='changed')
        with self.assertRaisesRegex(ValueError, 'measurement file changed'):
            self.validate(wrong)

    def test_fresh_replay_enforces_original_strict_tolerances(self):
        receipt, _ = self.measure()
        self.assertTrue(self.verify(receipt)['passed'])
        with patch.object(ex, '_measure_q', return_value=np.full((16, 4), .46879, np.float32)):
            with self.assertRaisesRegex(ValueError, 'fresh local replay failed'):
                self.verify(receipt)
        current = _probe(self.model, self.dataset, 'cpu')
        current['output'] += np.float32(1e-4)
        with patch.object(ex, '_measure_probe', return_value=current):
            with self.assertRaisesRegex(ValueError, 'fresh local replay failed'):
                self.verify(receipt)

    def test_measurement_rejects_bad_q_metadata_and_original_probe_inputs(self):
        self.original_probe['pan'].flat[0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'probe inputs differ'):
            self.measure()
        with patch.object(self.dataset, 'get_view', side_effect=lambda index, view, augment:
                (*self.dataset.base(index)[:5], torch.tensor([index, view, 0, 1]))):
            with self.assertRaisesRegex(ValueError, 'view/base-ID metadata'):
                ex._measure_q(self.model, self.dataset, self.q, 'cpu', None)
        for value in (-1, np.inf):
            with patch.object(ex, 'axis16_q', return_value=(torch.full((16,), value), None, None)):
                with self.assertRaisesRegex(ValueError, 'online q is invalid'):
                    ex._measure_q(self.model, self.dataset, self.q, 'cpu', None)

    def test_measurement_preserves_nested_modes_and_rng(self):
        self.model.train()
        rng = torch.get_rng_state().clone()
        ex._measure_q(self.model, self.dataset, self.q, 'cpu', None)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_late_measurement_does_not_write_receipt(self):
        with self.assertRaises(TimeoutError):
            ex.measure(self.model, self.dataset, self.q, self.original_probe, self.origin,
                self.local, self.root, 's1', 'cpu', '2000-01-01T00:00:00+00:00')
        self.assertFalse(ex._measurement_path(self.root, 's1').exists())


if __name__ == '__main__':
    unittest.main()
