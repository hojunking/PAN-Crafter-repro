"""Preflight guard fixtures confined to temporary directories; no real readiness."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qg40 import preflight
from qg40.common import atomic_json, camp, object_sha, read_json, sha256
from qg40.plan import sensor_spec


class PreflightGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = 's5'
        self.deadline = '2999-01-01T00:00:00+00:00'
        self.folder = camp(self.root, self.server)
        bindings = {name: dict(path=str(self.root / f'{name}.h5'), sha256='d' * 64,
                              source_identity=f'synthetic-fixture-{name}')
                    for name in ('train', 'val', 'rr', 'fr')}
        self.spec = sensor_spec('GF2').bind(band_order=('B', 'G', 'R', 'NIR'), splits=bindings,
                    source_provenance={'fixture': 'Synthetic test metadata, not source provenance'},
                    verify_files=False)
        self.spec_path = self.folder / 'sensor_spec.json'
        atomic_json(self.spec_path, self.spec.to_dict())
        self.release = dict(files={'external/DLPan/wald_utilities.py': 'a' * 64}, fixture=True)
        self.manifest = dict(schema='QG40_DATA_v1', sensor='GF2', num_bands=4, max_pixel=1023,
                             fixture=True, splits=bindings)
        self.manifest_path = self.folder / 'dataset_manifest.json'
        atomic_json(self.manifest_path, self.manifest)
        self.receipt = dict(complete=True, status='LOCAL_READY', fixture=True,
                            source_identity=self.release, sensor_spec_sha256=object_sha(self.spec.to_dict()),
                            dataset_manifest_sha256=object_sha(self.manifest), real_native_forward=True)
        self.receipt_path = self.folder / 'preflight.json'
        atomic_json(self.receipt_path, self.receipt)

    def guards(self, stack, *, release=None, prepared=None):
        stack.enter_context(patch.object(preflight, 'source_identity', return_value=release or self.release))
        prepare = stack.enter_context(patch('qg40.data.prepare_data', return_value=self.manifest if prepared is None else prepared))
        # Any accidental native-data or model path is a test failure, not real work.
        stack.enter_context(patch('qg40.data.build_dataset', side_effect=AssertionError('No real data in guard tests')))
        stack.enter_context(patch('qg40.model.build_model', side_effect=AssertionError('No model execution in guard tests')))
        stack.enter_context(patch('qg40.evaluation.FRMetrics', side_effect=AssertionError('No FR execution in guard tests')))
        stack.enter_context(patch.object(preflight, 'method_checks', side_effect=AssertionError('No nested preflight execution')))
        stack.enter_context(patch('torch.cuda.is_available', side_effect=AssertionError('No GPU access in guard tests')))
        return prepare

    def execute(self):
        return preflight.execute(self.root, self.server, self.spec_path, self.deadline, device='cpu')

    def test_missing_or_unavailable_external_dlpan_rejected_before_preparation(self):
        original = sha256(self.receipt_path)
        for value in (None, 'UNAVAILABLE', '', 123):
            with self.subTest(value=value), ExitStack() as stack:
                release = copy.deepcopy(self.release)
                if value is None:
                    release['files'].pop('external/DLPan/wald_utilities.py')
                else:
                    release['files']['external/DLPan/wald_utilities.py'] = value
                prepare = self.guards(stack, release=release)
                with self.assertRaisesRegex(ValueError, 'DLPan evaluation dependency'):
                    self.execute()
                prepare.assert_not_called()
                self.assertEqual(sha256(self.receipt_path), original)

    def test_cached_receipt_with_changed_source_is_rejected(self):
        release = copy.deepcopy(self.release)
        release['files']['external/DLPan/wald_utilities.py'] = 'b' * 64
        with ExitStack() as stack:
            prepare = self.guards(stack, release=release)
            with self.assertRaisesRegex(ValueError, 'source/sensor binding changed'):
                self.execute()
            prepare.assert_not_called()
        self.assertEqual(read_json(self.receipt_path), self.receipt)

    def test_cached_receipt_with_changed_sensor_spec_is_rejected(self):
        changed = self.spec.to_dict()
        changed['band_order'] = ['R', 'G', 'B', 'NIR']
        atomic_json(self.spec_path, changed)
        with ExitStack() as stack:
            prepare = self.guards(stack)
            with self.assertRaisesRegex(ValueError, 'source/sensor binding changed'):
                self.execute()
            prepare.assert_not_called()
        self.assertEqual(read_json(self.receipt_path), self.receipt)

    def test_cached_receipt_manifest_hash_must_match_actual_preparation(self):
        changed = dict(self.manifest, changed_after_readiness=True)
        with ExitStack() as stack:
            prepare = self.guards(stack, prepared=changed)
            with self.assertRaisesRegex(ValueError, 'dataset manifest changed'):
                self.execute()
            prepare.assert_called_once()
        self.assertEqual(read_json(self.receipt_path), self.receipt)

    def test_cached_alias_must_match_prepared_manifest_even_with_valid_receipt_hash(self):
        atomic_json(self.manifest_path, dict(self.manifest, altered_alias=True))
        with ExitStack() as stack:
            prepare = self.guards(stack)
            with self.assertRaisesRegex(ValueError, 'dataset manifest changed'):
                self.execute()
            prepare.assert_called_once()
        self.assertEqual(read_json(self.receipt_path), self.receipt)

    def test_successful_cached_reuse_revalidates_bound_sources_and_cache(self):
        original = sha256(self.receipt_path)
        with ExitStack() as stack:
            prepare = self.guards(stack)
            self.assertEqual(self.execute(), self.receipt)
            prepare.assert_called_once_with(self.root, self.server, self.spec, deadline_utc=self.deadline)
        self.assertEqual(sha256(self.receipt_path), original)

    def test_successful_cached_reuse_accepts_preparation_manifest_path(self):
        with ExitStack() as stack:
            prepare = self.guards(stack, prepared=self.manifest_path)
            self.assertEqual(self.execute(), self.receipt)
            prepare.assert_called_once()

    def test_P1_distribution_timeout_preserves_local_ready(self):
        original = sha256(self.receipt_path)
        with ExitStack() as stack:
            self.guards(stack)
            stack.enter_context(patch('qg40.input_distribution.capture_input_distributions',
                                      side_effect=TimeoutError('P1 budget exceeded')))
            self.assertEqual(self.execute(), self.receipt)
        self.assertEqual(sha256(self.receipt_path), original)
        report = read_json(self.folder / 'diagnostics/input_distribution.json')
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertFalse(report['p0_gate'])

    def test_phase_coordinate_failure_prevents_ready_reuse(self):
        with ExitStack() as stack:
            prepare = self.guards(stack)
            stack.enter_context(patch('qg40.phase_audit.phase_coordinate_checks',
                                      side_effect=ValueError('P0 bad phase coordinates')))
            with self.assertRaisesRegex(ValueError, 'P0 bad phase'):
                self.execute()
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
