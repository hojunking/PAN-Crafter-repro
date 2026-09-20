"""Reference integrity CPU tests; no campaign artifacts or GPUs are touched."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from fh12.common import atomic_json, object_sha, read_json, sha256
from fh12.data import RECIPE, AUGMENTATION
from qg40.model import state_hash
from qg40.test_references import ProbeModel, ProbeDataset
from qg40.reference_parity import verify_q_cache, identity_for
from qg40.calibration import select_calibration_indices
from g20 import references as ref


class DataIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def dataset(self, name, compression=None, changed=False):
        path, lp = self.root / (name + '.h5'), self.root / (name + '_lp.h5')
        with h5py.File(path, 'w') as f:
            for key, channels, side in [('pan', 1, 8), ('lms', 4, 8), ('ms', 4, 2), ('gt', 4, 8)]:
                x = np.arange(2 * channels * side * side, dtype=np.float32).reshape(2, channels, side, side)
                if changed and key == 'ms':
                    x[0, 0, 0, 0] += .125
                f.create_dataset(key, data=x, compression=compression)
        with h5py.File(lp, 'w') as f:
            f.create_dataset('lpan', data=np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2), compression=compression)
        splits = {split: dict(dataroot=str(path), lpan_path=str(lp), sha256=sha256(path),
            lpan_sha256=sha256(lp), count=2, shapes={'test': [2]}, sample_order_sha256='order')
            for split in ('train', 'val', 'rr', 'fr')}
        return dict(sensor='GF2', num_bands=4, max_pixel=1023, mtf_sensor='GF2',
            band_order=['B', 'G', 'R', 'NIR'], recipe=RECIPE, augmentation=AUGMENTATION, splits=splits)

    def test_reserialization_allowed_only_with_exact_ordered_tensor_equality(self):
        a, b = self.dataset('a'), self.dataset('b', 'gzip')
        self.assertNotEqual(a['splits']['train']['sha256'], b['splits']['train']['sha256'])
        report = ref.verify_data_equivalence(a, b)
        self.assertTrue(report['passed'])
        self.assertEqual(report['splits']['train']['data']['method'], 'canonical_ordered_tensors')
        with self.assertRaisesRegex(ValueError, 'canonical ordered tensors'):
            ref.verify_data_equivalence(a, self.dataset('c', 'gzip', changed=True))

    def test_missing_origin_requires_manifest_bound_canonical_receipt(self):
        a, b = self.dataset('a'), self.dataset('b', 'gzip')
        proof = ref._tensor_receipt(a, None)
        for key in ('dataroot', 'lpan_path'):
            Path(a['splits']['train'][key]).unlink()
        with self.assertRaisesRegex(ValueError, 'requires original pinned tensor proof'):
            ref.verify_data_equivalence(a, b)
        self.assertTrue(ref.verify_data_equivalence(a, b, origin_tensor_receipt=proof)['passed'])
        wrong = copy.deepcopy(proof); wrong['origin_data_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'not bound'):
            ref.verify_data_equivalence(a, b, origin_tensor_receipt=wrong)

    def test_wrong_sensor_dn_band_order_or_changed_bytes_fail(self):
        a, b = self.dataset('a'), self.dataset('b')
        for key, value in [('sensor', 'QB'), ('max_pixel', 2047), ('band_order', ['R', 'G', 'B', 'NIR'])]:
            altered = copy.deepcopy(b); altered[key] = value
            with self.assertRaises(ValueError):
                ref.verify_data_equivalence(a, altered)
        b['splits']['train']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            ref.verify_data_equivalence(a, b)

    def test_r0_requires_original_appendix_pins(self):
        a = self.dataset('a')
        with self.assertRaisesRegex(ValueError, 'Appendix B'):
            ref.verify_data_equivalence(a, a, r0=True)

    def test_authenticated_old_manifest_lp_hash_handles_reserialization(self):
        import hashlib
        a, b = self.dataset('a'), self.dataset('b', 'gzip')
        with h5py.File(a['splits']['train']['lpan_path'], 'r') as cache:
            digest = hashlib.sha256(cache['lpan'][:].astype('<f4').tobytes()).hexdigest()
        for split in a['splits']:
            a['splits'][split]['lpan_canonical_sha256'] = digest
            # Source MS/GT bytes remain identical; only LP is reserialized.
            b['splits'][split]['dataroot'] = a['splits'][split]['dataroot']
            b['splits'][split]['sha256'] = a['splits'][split]['sha256']
        Path(a['splits']['train']['lpan_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'requires original pinned tensor proof'):
            ref.verify_data_equivalence(a, b)
        report = ref.verify_data_equivalence(a, b, origin_authenticated=True)
        self.assertEqual(report['splits']['train']['lp']['method'], 'authenticated_origin_LP_float32_sha256')
        a['splits']['train']['lpan_canonical_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'manifest-bound LP'):
            ref.verify_data_equivalence(a, b, origin_authenticated=True)


class OriginPinsTests(unittest.TestCase):
    def test_aliases_owner_seed_and_coefficients_are_distinct(self):
        self.assertEqual(ref.REFERENCE_SPECS['R0'][2], 91001)
        self.assertEqual(ref.REFERENCE_SPECS['R1'][2], 91002)
        self.assertNotEqual(ref.REFERENCE_SPECS['R1'][:2], ref.REFERENCE_SPECS['R2'][:2])
        self.assertEqual([ref.REFERENCE_SPECS[r][3] for r in ('R2', 'R3', 'R4')], [1e-4, 3e-5, 3e-4])

    def test_r0_rejects_an_unpinned_old_teacher(self):
        cfg = dict(seed=91001, qg40=dict(role='T', input_layout='P0'), model_args=dict(hidden_size=112, depth=[1, 2, 3]))
        origin = dict(teacher_update=50000, sensor='GF2', num_bands=4, max_pixel=1023,
                      teacher_layout='P0', reference_id='GF2_TA', server='s3')
        with self.assertRaisesRegex(ValueError, 'source snapshot pin'):
            ref._check_origin('R0', origin, cfg, {}, '.')
        origin.update({key: ref.R0_PINS[key] for key in ('teacher_checkpoint_sha256', 'q_cache_sha256', 'tau_R', 'q_ref')})
        with self.assertRaisesRegex(ValueError, 'calibration ID'):
            ref._check_origin('R0', origin, cfg, {}, '.')
        origin['calibration_id'] = ref.R0_PINS['calibration_id']
        origin['source_identity'] = dict(numeric_method_revision='wrong')
        with self.assertRaisesRegex(ValueError, 'calibration ID'):
            ref._check_origin('R0', origin, cfg, {}, '.')
        with patch.dict(ref.R0_PINS, calibration_id=object_sha(origin)):
            with self.assertRaisesRegex(ValueError, 'source bundle'):
                ref._check_origin('R0', origin, cfg, {}, '.')

    def test_probe_checks_both_output_and_native_c(self):
        dataset = ProbeDataset()
        probe = ref._probe(ProbeModel(), dataset, 'cpu')
        self.assertTrue(ref._check_probe(ProbeModel(), probe, 'cpu')['passed'])
        for key in ('output', 'delta'):
            bad = {k: v.copy() for k, v in probe.items()}; bad[key] += .1
            with self.assertRaisesRegex(ValueError, 'numerical parity'):
                ref._check_probe(ProbeModel(), bad, 'cpu')


class PortableReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.release = dict(content_sha256='test', files={})
        from g20.common import camp
        self.camp = camp
        for name in ('source_identity',):
            p = patch('g20.common.' + name, return_value=self.release); p.start(); self.addCleanup(p.stop)
        p = patch('g20.references._check_origin'); p.start(); self.addCleanup(p.stop)
        p = patch('g20.references._model', side_effect=lambda *a, **kw: ProbeModel().eval()); p.start(); self.addCleanup(p.stop)
        p = patch('g20.data.build_dataset', return_value=ProbeDataset()); p.start(); self.addCleanup(p.stop)
        folder = ref.reference_path('R1', 's1', self.root).parent; folder.mkdir(parents=True)
        # Source bytes can be identical between relocatable manifests without a
        # numerical fallback. Real canonical H5 coverage is tested separately.
        data_path, lp_path = folder / 'data.h5', folder / 'lp.h5'
        data_path.write_bytes(b'immutable synthetic data'); lp_path.write_bytes(b'immutable synthetic LP')
        self.data = dict(sensor='GF2', num_bands=4, max_pixel=1023, mtf_sensor='GF2',
            band_order=['B', 'G', 'R', 'NIR'], recipe=RECIPE, augmentation=AUGMENTATION,
            splits={s: dict(dataroot=str(data_path), lpan_path=str(lp_path), sha256=sha256(data_path),
                lpan_sha256=sha256(lp_path), count=3072, shapes={}, sample_order_sha256='ordered')
                for s in ('train', 'val', 'rr', 'fr')})
        manifest_data = self.camp(self.root, 's1') / 'dataset_manifest.json'; atomic_json(manifest_data, self.data)
        atomic_json(self.camp(self.root, 's2') / 'dataset_manifest.json', self.data)
        cfg_path = folder / 'config.yaml'; cfg_path.write_text(yaml.safe_dump(dict(seed=91002)))
        cfg = yaml.safe_load(cfg_path.read_text())
        weights = folder / 'model.safetensors'; state = ProbeModel().state_dict(); save_file(state, str(weights))
        identity = dict(update=50000, full_state=True, config_sha256=object_sha(cfg),
            model_sha256=sha256(weights), source_identity=self.release, data_sha256=object_sha(self.data), state_hash=state_hash(state))
        full = folder / 'training_state.pt'; torch.save(dict(identity, model_state=state), full)
        identity['training_state_sha256'] = sha256(full)
        identity_path = folder / 'identity.json'; atomic_json(identity_path, identity)
        self.q = np.full((3072, 4), .46875, np.float32); indices = select_calibration_indices(3072)
        q_path = folder / 'q.npz'; np.savez(q_path, q=self.q, calibration_indices=indices)
        cal_path = folder / 'calibration.json'
        origin = dict(schema='G20_REFERENCE_v1', reference_id='R1', teacher_run_id='synthetic-R1',
            source_identity=self.release, teacher_checkpoint=str(weights), teacher_checkpoint_sha256=sha256(weights),
            teacher_training_state=str(full), teacher_training_state_sha256=sha256(full),
            teacher_config=str(cfg_path), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(cfg_path),
            teacher_checkpoint_identity=str(identity_path), teacher_checkpoint_identity_sha256=sha256(identity_path),
            dataset_manifest_path=str(manifest_data), dataset_manifest_sha256=sha256(manifest_data),
            q_cache_path=str(q_path), q_cache_sha256=sha256(q_path), q_shape=[3072, 4],
            tau_R=.05, q_ref=.46875, calibration_indices_sha256=object_sha(indices.tolist()), calibration_path=str(cal_path))
        shapes = {key: list(value.shape) for key, value in state.items()}
        origin.update(architecture_state_shapes=shapes, architecture_state_shapes_sha256=object_sha(shapes),
                      architecture_forward=ref.FORWARD_CONTRACT)
        cal = {key: origin[key] for key in ('tau_R', 'q_ref', 'q_shape', 'calibration_indices_sha256',
                                           'teacher_checkpoint_sha256', 'source_identity')}
        cal.update(schema='G20_CALIBRATION_v1', synthetic_test=False); atomic_json(cal_path, cal)
        origin['calibration_sha256'] = sha256(cal_path)
        receipt = verify_q_cache(ProbeModel(), ProbeDataset(), self.q, reference_identity=identity_for(origin),
            data_identity=object_sha(ref.data_signature(self.data)), device='cpu')
        parity_path = folder / 'parity.json'; atomic_json(parity_path, receipt)
        origin.update(q_cache_parity_path=str(parity_path), q_cache_parity_sha256=sha256(parity_path))
        self.origin = origin
        atomic_json(ref.reference_path('R1', 's1', self.root), origin)

    def test_full_payload_local_readback_and_frozen_online_load(self):
        origin, q, _, _ = ref.validate_reference('R1', 's1', self.root)
        np.testing.assert_array_equal(q, self.q)
        path = ref.adopt_reference(ref.reference_path('R1', 's1', self.root), 'R1', 's2', self.root)
        bridge = read_json(path)
        self.assertTrue(bridge['q_cache_online_parity']['passed'])
        self.assertTrue(bridge['output_parity']['passed'])
        model, _, _ = ref.load_reference('R1', 's2', self.root, device='cpu')
        self.assertFalse(model.training)
        self.assertEqual(origin['g20_reference_id'], 'R1')

    def test_mutated_fullstate_or_q_cache_cannot_be_consumed(self):
        with open(self.origin['q_cache_path'], 'ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
            ref.validate_reference('R1', 's1', self.root)

    def test_portable_archive_roundtrip_checksum_and_wrong_alias(self):
        with patch('g20.references._tensor_receipt', return_value=None):
            archive = ref.export_reference('R1', 's1', self.root)
        target = ref.import_reference(archive, 'R1', 's2', self.root)
        self.assertTrue(read_json(target)['complete'])
        with self.assertRaisesRegex(ValueError, 'Wrong reference archive alias'):
            ref.import_reference(archive, 'R2', 's2', self.root)

    def test_archive_parent_traversal_rejected(self):
        archive = self.root / 'bad.tar'
        with tarfile.open(archive, 'w') as package:
            info = tarfile.TarInfo('../escape'); info.size = 1; package.addfile(info, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            ref.import_reference(archive, 'R1', 's2', self.root)

    def test_inventory_does_not_claim_validation_or_write(self):
        rows = ref.inventory_references(self.root)
        self.assertEqual(rows[0]['status'], 'DISCOVERED_NOT_VALIDATED')
        self.assertEqual(len(rows), 1)

    def test_fresh_calibration_publishes_only_new_reference_measurements(self):
        from g20.plan import teacher_for, build_config
        from g20.calibration import calibrate
        case = teacher_for('R2')
        cfg = build_config(case)
        cfg['g20']['dataset_manifest'] = str(self.camp(self.root, 's2') / 'dataset_manifest.json')
        run = self.root / 'work_dir' / case.run_id
        (run / 'meta').mkdir(parents=True)
        cfg_path = run / 'meta/config.resolved.yaml'; cfg_path.write_text(yaml.safe_dump(cfg))
        candidate = run / 'candidates/50000'; candidate.mkdir(parents=True)
        model = ProbeModel(); weights = candidate / 'model.safetensors'; save_file(model.state_dict(), str(weights))
        identity = dict(update=50000, full_state=True, config_sha256=object_sha(cfg),
            model_sha256=sha256(weights), source_identity=self.release, data_sha256=object_sha(self.data),
            state_hash=state_hash(model.state_dict()))
        state = candidate / 'training_state.pt'; torch.save(dict(identity, model_state=model.state_dict()), state)
        identity['training_state_sha256'] = sha256(state); atomic_json(candidate / 'identity.json', identity)
        indices = select_calibration_indices(3072)
        cal = dict(schema='G20_CALIBRATION_v1', synthetic_test=False, tau_R=.123,
                   q_ref=.46875, q_shape=[3072, 4], calibration_indices_sha256=object_sha(indices.tolist()))
        arrays = dict(q=self.q, calibration_indices=indices)
        with patch('g20.common.load_checkpoint_model', return_value=(model, identity)), \
             patch('g20.training.runtime_context', return_value=(
                 {'deadline_utc': '2999-01-01T00:00:00+00:00'}, '2000-01-01T00:00:00+00:00')), \
             patch('g20.calibration.compute_calibration', return_value=(cal, arrays)) as measured:
            destination = calibrate(case.run_id, self.root, 's2', device='cpu')
        result = read_json(destination)
        self.assertEqual(result['reference_id'], 'R2')
        self.assertEqual(result['teacher_alias'], 'GF2_G20_S2_TB_C100')
        self.assertEqual(result['tau_R'], .123)
        self.assertNotEqual(result['tau_R'], ref.R0_PINS['tau_R'])
        self.assertEqual(result['architecture_forward']['width'], 112)
        self.assertEqual(measured.call_args.args[2].shape, (3072,))
        self.assertEqual(measured.call_args.kwargs['deadline_utc'], '2999-01-01T00:00:00+00:00')
        with patch('g20.training.runtime_context', return_value=(
                {'deadline_utc': '2000-01-01T00:00:00+00:00'}, '1999-01-01T00:00:00+00:00')), \
             patch('g20.calibration.compute_calibration') as late:
            with self.assertRaises(TimeoutError):
                calibrate(case.run_id, self.root, 's2', device='cpu')
            late.assert_not_called()


if __name__ == '__main__':
    unittest.main()
