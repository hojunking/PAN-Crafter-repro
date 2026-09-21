"""Synthetic CPU tests of the actual gain, LP, view and AXIS16 paths."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from fh12.calibration import CONSTANT_ALIGNER_Q, axis16_q
from fh12.data import RECIPE, canonical_sha, sha256_file
from g20.data import G20Dataset
from g20.plan import sensor_spec
from gfb20.augmentation import gaussian_detail_view, prepare_augmentation, verify_augmentation_cache
from gfb20.calibration import (compute_mixed_calibration, verify_native_slice,
                               weighted_median_tokens, _native_paths)
from gfb20.data import GFB20Dataset
from pa.warp import warp_pan
from tools.repair_lpan import make_lpan


def synthetic_dataset(folder, count=17):
    folder = Path(folder)
    rng = np.random.default_rng(487)
    pan = rng.uniform(0, 1023, (count, 1, 64, 64)).astype(np.float64)
    source, lp = folder / 'train.h5', folder / 'native_lp.h5'
    with h5py.File(source, 'w') as dst:
        dst['pan'] = pan
        dst['gt'] = np.zeros((count, 4, 64, 64), dtype=np.float32)
        dst['lms'] = np.zeros((count, 4, 64, 64), dtype=np.float32)
        dst['ms'] = np.zeros((count, 4, 16, 16), dtype=np.float32)
    with h5py.File(lp, 'w') as dst:
        dst['lpan'] = make_lpan(pan).astype(np.float32)
        dst.attrs['source_sha256'] = sha256_file(source)
        dst.attrs['recipe_sha256'] = canonical_sha(RECIPE)
    spec = replace(sensor_spec('GF2'), band_order=('blue', 'green', 'red', 'nir'))
    return G20Dataset(source, lp, spec=spec, split='train'), pan


class ConstantTeacher(torch.nn.Module):
    bands = 4

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict_delta(self, pan, base):
        return pan.new_zeros((len(pan), 2)) + self.anchor

    def forward(self, pan, ms, lp):
        return {'y': pan.expand(-1, 4, -1, -1) + self.anchor}


class PanViewsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gamma1_is_original_object(self):
        value = np.arange(64, dtype=np.float64).reshape(1, 1, 8, 8)
        self.assertIs(gaussian_detail_view(value, 1), value)
        with self.assertRaises(ValueError):
            gaussian_detail_view(value.astype(np.float32), 0)

    def test_constant_symmetric_center_and_no_clamp(self):
        constant = np.full((1, 1, 64, 64), 237., dtype=np.float64)
        np.testing.assert_allclose(gaussian_detail_view(constant, 0), constant, atol=1e-12)
        impulse = np.zeros_like(constant)
        impulse[0, 0, 32, 32] = 1023
        high = gaussian_detail_view(impulse, 2)
        self.assertGreater(high.max(), 1023)
        self.assertLess(high.min(), 0)
        patch_data = high[0, 0, 29:36, 29:36]
        np.testing.assert_array_equal(patch_data, patch_data[::-1, ::-1])

    def test_exact_weighted_tokens_not_median_of_patch_medians(self):
        values = np.array([[0, 10, 20], [100, 110, 120]], dtype=np.float32)
        self.assertEqual(weighted_median_tokens(values, 1), 60)
        values = np.array([[[0, 0, 100], [1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        self.assertEqual(weighted_median_tokens(values, 1), float(np.median(values[:, [0, 1, 1, 2]])))
        self.assertNotEqual(weighted_median_tokens(values, 1), float(np.mean(np.median(values, axis=-1))))

    def test_cache_native_parity_geometry_lp_phase_and_control(self):
        with tempfile.TemporaryDirectory() as folder:
            native, source = synthetic_dataset(folder, 2)
            path = prepare_augmentation(native, Path(folder) / 'views')
            manifest = verify_augmentation_cache(path, native)
            self.assertEqual(path, prepare_augmentation(native, Path(folder) / 'views'))
            mixed = GFB20Dataset(native, path, 'PANMIX')
            control = GFB20Dataset(native, path, 'NAT_MIXCAL')
            row, base = mixed[(0, 1, 1)], native.get_view(0, 1, augment=True)
            for got, expected in zip(row[:5], base[:5]):
                self.assertTrue(torch.equal(got, expected))
            for position in (0, 1, 2):
                self.assertTrue(torch.equal(mixed[(0, 1, 2)][position], base[position]))
            self.assertEqual(control[(0, 1, 2)][-1].tolist(), [0, 1, 1, 1, 1, 2])
            self.assertEqual(mixed[(0, 1, 2)][-1].tolist(), [0, 1, 1, 1, 2, 2])
            self.assertTrue(torch.equal(control[(0, 1, 2)][4], base[4]))
            gain = gaussian_detail_view(source[:1], 2)
            expected_lp = np.rot90(make_lpan(gain)[0, :, ::-1, ::-1], 1, axes=(1, 2)).copy()
            expected_lp = torch.from_numpy(expected_lp.astype(np.float32)).mul_(2 / 1023).sub_(1)
            self.assertTrue(torch.equal(mixed[(0, 1, 2)][3], expected_lp))
            wrong_phase = make_lpan(np.rot90(gain[:, :, ::-1, ::-1], 1, axes=(2, 3)))
            self.assertFalse(np.allclose(wrong_phase[0], (expected_lp.numpy()+1)*1023/2))
            self.assertEqual(manifest['population'], 2)

    def test_no_augmented_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder, 1)
            native.split = 'rr'
            adapter = GFB20Dataset(native)
            with self.assertRaisesRegex(ValueError, 'remain native'):
                adapter.get_view(0, gamma_id=0)
            with self.assertRaises(ValueError):
                prepare_augmentation(native, Path(folder) / 'views')

    def test_full_native_parity_fails_closed(self):
        original = np.full((17, 4), CONSTANT_ALIGNER_Q, np.float32)
        changed = original.copy()
        changed[-1, -1] += .001
        with self.assertRaisesRegex(ValueError, 'parity failed'):
            verify_native_slice(changed, original)
        self.assertEqual(verify_native_slice(original, original)['population'], 17)

    def test_real_axis16_receives_gain_before_warp(self):
        class Recorder(ConstantTeacher):
            def __init__(self):
                super().__init__()
                self.inputs = []
            def predict_delta(self, pan, base):
                self.inputs.append(pan.detach().clone())
                return super().predict_delta(pan, base)
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder, 1)
            manifest = prepare_augmentation(native, Path(folder) / 'views')
            dataset = GFB20Dataset(native, manifest, 'PANMIX')
            row = dataset.get_view(0, 2, augment=True, gamma_id=2)
            pan, ms = row[4][None], row[2][None]
            model = Recorder()
            q, _, _ = axis16_q(model, pan, ms)
            self.assertTrue(torch.equal(model.inputs[0], pan))
            self.assertTrue(torch.equal(model.inputs[1], warp_pan(pan, pan.new_tensor([[.25, 0.]]))))
            self.assertAlmostEqual(q.item(), CONSTANT_ALIGNER_Q)

    def test_complete_calibration_pixel_median_resume_and_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder)
            augmentation = prepare_augmentation(native, Path(folder) / 'views')
            dataset = GFB20Dataset(native, augmentation, 'PANMIX')
            teacher = ConstantTeacher()
            teacher.train().requires_grad_(True)
            indices = np.array([0, 7, 16])
            native_q = np.full((17, 4), CONSTANT_ALIGNER_Q, dtype=np.float32)
            progress = Path(folder) / 'calibration.h5'
            report, cache = compute_mixed_calibration(teacher, dataset, indices, native_q,
                device='cpu', progress_path=progress, synthetic_test=True)
            self.assertEqual(cache['q'].shape, (17, 4, 3))
            np.testing.assert_array_equal(cache['q'][:, :, 1], native_q)
            self.assertTrue(teacher.training)
            self.assertTrue(teacher.anchor.requires_grad)
            errors = np.stack([np.stack([(dataset.get_view(int(i), gamma_id=gamma)[4]
                .expand(4, -1, -1) - dataset.base(int(i))[0]).abs().mean(0).numpy()
                for gamma in range(3)]) for i in indices])
            self.assertEqual(report['tau_R'], weighted_median_tokens(errors, 1))
            self.assertEqual(report['q_ref'], CONSTANT_ALIGNER_Q)
            with patch('gfb20.calibration.axis16_q', side_effect=AssertionError('should resume')):
                again, reused = compute_mixed_calibration(teacher, dataset, indices, native_q,
                    device='cpu', progress_path=progress, synthetic_test=True)
            self.assertEqual(report, again)
            np.testing.assert_array_equal(cache['q'], reused['q'])
            with h5py.File(progress, 'r+') as dirty:
                dirty['q'][0, 0, 0] += .1
            with self.assertRaisesRegex(ValueError, 'chunk changed'):
                compute_mixed_calibration(teacher, dataset, indices, native_q,
                    device='cpu', progress_path=progress, synthetic_test=True)

    def test_deadline_and_production_count_and_batch_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder, 1)
            with self.assertRaises(TimeoutError):
                prepare_augmentation(native, Path(folder) / 'views', '2000-01-01T00:00:00Z')
            path = prepare_augmentation(native, Path(folder) / 'views')
            dataset = GFB20Dataset(native, path, 'PANMIX')
            q = np.full((1, 4), CONSTANT_ALIGNER_Q, np.float32)
            with self.assertRaisesRegex(ValueError, 'train19809'):
                compute_mixed_calibration(ConstantTeacher(), dataset, [0], q, device='cpu')
            with self.assertRaisesRegex(ValueError, 'batch16'):
                compute_mixed_calibration(ConstantTeacher(), dataset, [0], q,
                    device='cpu', batch_size=64, synthetic_test=True)

    def test_partial_deadline_resume_reuses_only_completed_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder, 1)
            path = prepare_augmentation(native, Path(folder) / 'views')
            dataset = GFB20Dataset(native, path, 'PANMIX')
            progress = Path(folder) / 'progress.h5'
            q = np.full((1, 4), CONSTANT_ALIGNER_Q, np.float32)
            with patch('gfb20.calibration.check_deadline', side_effect=[None, None, TimeoutError('pause')]):
                with self.assertRaises(TimeoutError):
                    compute_mixed_calibration(ConstantTeacher(), dataset, [0], q,
                        device='cpu', progress_path=progress, synthetic_test=True)
            with h5py.File(progress, 'r') as state:
                self.assertTrue(state['e_checksums'][0, 0])
                self.assertTrue(state['q_checksums'][0, 0, 0])
                self.assertFalse(state['q_checksums'][0, 1, 0])
            report, arrays = compute_mixed_calibration(ConstantTeacher(), dataset, [0], q,
                device='cpu', progress_path=progress, synthetic_test=True)
            self.assertEqual(arrays['q'].shape, (1, 4, 3))
            self.assertTrue(report['native_slice_parity']['passed'])

    def test_original_relative_manifest_requires_bound_absolute_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder, 1)
            original = dict(teacher_checkpoint='weights/model.safetensors', q_cache_path='reference/q.npz')
            with self.assertRaisesRegex(ValueError, 'binder-resolved'):
                _native_paths(original)
            info = dict(native_manifest=original, teacher_checkpoint=native.raw_h5_path,
                        q_cache_path=native.lp_path)
            paths = _native_paths(info)
            self.assertEqual(paths['teacher_checkpoint'], native.raw_h5_path)
            self.assertEqual(original['q_cache_path'], 'reference/q.npz')

    def test_qref_uses_calibration_ids_not_full_train(self):
        class LocalTeacher(ConstantTeacher):
            def predict_delta(self, pan, base):
                return torch.stack((pan[:, 0, 32, 32] * .2,
                                    pan[:, 0, 16, 16] * .1), dim=1)
        with tempfile.TemporaryDirectory() as folder:
            native, _ = synthetic_dataset(folder)
            path = prepare_augmentation(native, Path(folder) / 'views')
            dataset = GFB20Dataset(native, path, 'PANMIX')
            model = LocalTeacher()
            q = np.empty((17, 4), np.float32)
            for view in range(4):
                for start in range(0, 17, 16):
                    rows = [native.get_view(i, view, augment=True) for i in range(start, min(17, start+16))]
                    q[start:start+16, view] = axis16_q(model,
                        torch.stack([r[4] for r in rows]), torch.stack([r[2] for r in rows]))[0].numpy()
            indices = np.array([0, 7, 16])
            report, arrays = compute_mixed_calibration(model, dataset, indices, q,
                device='cpu', synthetic_test=True)
            self.assertEqual(report['q_ref'], weighted_median_tokens(arrays['q'][indices], 2))
            self.assertNotEqual(report['q_ref'], weighted_median_tokens(arrays['q'], 2))


if __name__ == '__main__':
    unittest.main()
