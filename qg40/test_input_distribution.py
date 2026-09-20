"""CPU-only distribution/coordinate fixtures, no real source mutation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from qg40.common import object_sha, read_json, sha256
from qg40.input_distribution import capture_input_distributions, record_input_distributions
from qg40.phase_audit import phase_coordinate_checks, qb_phase_residuals


class InputDistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(42)
        self.native = dict(pan=rng.uniform(0, 1023, (3, 1, 16, 16)).astype(np.float32),
            ms=rng.uniform(0, 1023, (3, 4, 4, 4)).astype(np.float32),
            lms=rng.uniform(0, 1023, (3, 4, 16, 16)).astype(np.float32),
            gt=rng.uniform(0, 1023, (3, 4, 16, 16)).astype(np.float32))
        self.manifest = dict(sensor='GF2', max_pixel=1023, band_order=['B', 'G', 'R', 'NIR'], splits={})
        for split in ('train', 'val', 'rr', 'fr'):
            source, lp = self.root / f'{split}.h5', self.root / f'{split}_lp.h5'
            with h5py.File(source, 'w') as f:
                for key, value in self.native.items():
                    f[key] = value if split != 'fr' or key != 'gt' else np.full_like(value, np.nan)
            with h5py.File(lp, 'w') as f:
                f['lpan'] = self.native['pan'][:, :, 2::4, 2::4]
            self.manifest['splits'][split] = dict(dataroot=str(source), lpan_path=str(lp),
                sha256=sha256(source), lpan_sha256=sha256(lp))

    def test_full_selected_moments_covariance_and_DN_inverse(self):
        result = capture_input_distributions(self.manifest, max_samples=3, max_quantile_pixels=768)
        stats = result['splits']['train']['radiometry']['gt']
        flat = self.native['gt'].astype(np.float64).transpose(1, 0, 2, 3).reshape(4, -1)
        np.testing.assert_allclose(stats['raw_DN']['mean'], flat.mean(1), rtol=1e-12)
        np.testing.assert_allclose(stats['raw_DN']['covariance'], np.cov(flat, bias=True), atol=1e-9)
        np.testing.assert_allclose(stats['normalized']['mean'], (flat * 2 / 1023 - 1).mean(1), atol=1e-12)
        np.testing.assert_allclose(stats['raw_DN']['quantiles']['p5'], np.quantile(flat, .05, axis=1), atol=1e-12)
        self.assertFalse(result['normalization_fitted'])
        self.assertFalse(result['train_test_mixed'])

    def test_bounded_reproducible_sampling_FR_no_GT_and_RNG_isolated(self):
        before_torch = torch.get_rng_state().clone()
        before_numpy = np.random.get_state()
        args = dict(max_samples=2, max_quantile_pixels=16)
        first = capture_input_distributions(self.manifest, **args)
        second = capture_input_distributions(self.manifest, **args)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        for old, new in zip(before_numpy, np.random.get_state()):
            np.testing.assert_equal(old, new)
        for name, split in first['splits'].items():
            self.assertEqual(split['selected_count'], 2)
            self.assertEqual(len(set(split['sample_ids'])), 2)
            self.assertLessEqual(split['radiometry']['pan']['quantile_samples_per_band'], 16)
            self.assertEqual(sha256(self.manifest['splits'][name]['dataroot']),
                             self.manifest['splits'][name]['sha256'])
        self.assertNotIn('gt', first['splits']['fr']['radiometry'])
        self.assertFalse(first['splits']['fr']['has_reconstruction_GT'])

    def test_energy_HP_signed_difference_and_fixed_lowtexture_rule(self):
        from torch.nn import functional as F
        report = capture_input_distributions(self.manifest, max_samples=1,
                                            max_quantile_pixels=16, low_texture_threshold=100)
        split = report['splits']['train']
        i = split['sample_ids'][0]
        pan = torch.from_numpy(self.native['pan'][i])[None]
        lp = F.interpolate(pan[:, :, 2::4, 2::4], size=(16, 16), mode='bicubic', align_corners=False)
        expected = float((((pan - lp) * 2 / 1023) ** 2).mean())
        self.assertAlmostEqual(split['energy']['means']['HP']['mean_square'], expected, places=6)
        self.assertEqual(split['energy']['low_texture_ratio'], 1.)
        self.assertFalse(split['energy']['threshold_fitted'])

    def test_P1_timeout_is_incomplete_receipt_not_exception_or_data_mutation(self):
        original = copy.deepcopy(self.manifest)
        receipt = record_input_distributions(self.root, 's5', self.manifest, budget_seconds=0)
        self.assertFalse(receipt['complete'])
        self.assertEqual(receipt['status'], 'INCOMPLETE')
        self.assertFalse(receipt['p0_gate'])
        self.assertEqual(self.manifest, original)
        saved = read_json(self.root / 'work_dir/_qg40/s5/diagnostics/input_distribution.json')
        self.assertEqual(saved['dataset_manifest_sha256'], object_sha(self.manifest))

    def test_P1_publication_failure_is_nonblocking(self):
        with patch('qg40.input_distribution.capture_input_distributions', side_effect=TimeoutError('budget')):
            with patch('qg40.input_distribution.atomic_json', side_effect=OSError('full disk')):
                receipt = record_input_distributions(self.root, 's5', self.manifest)
        self.assertEqual(receipt['reason'], 'budget')
        self.assertEqual(receipt['publication_error'], 'full disk')


class PhaseAuditTests(unittest.TestCase):
    def test_analytic_phase_unit_is_one_HR_not_one_LR_pixel(self):
        result = phase_coordinate_checks()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['one_phase_index_hr_pixels'], 1)
        self.assertEqual(result['one_phase_index_lr_spacing'], .25)
        self.assertEqual(result['one_lr_index_hr_pixels'], 4)

    def test_wrong_LP_phase_is_rejected(self):
        from tools.repair_lpan import make_lpan
        def shifted(pan):
            return make_lpan(np.roll(pan, 1, axis=-1))
        with patch('tools.repair_lpan.make_lpan', side_effect=shifted):
            with self.assertRaisesRegex(ValueError, 'LP ramp phase'):
                phase_coordinate_checks()

    def test_bounded_raw_QB_residual_audit_detects_phase_without_repair(self):
        from scipy.signal import fftconvolve
        from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'raw.h5'
            gt = np.random.default_rng(14).uniform(0, 2047, (2, 4, 64, 64))
            kernel = genmtf_matlab(GNYQ_TABLE['QB'], 4, 41)
            low = np.stack([np.stack([fftconvolve(np.pad(x[b], 20, mode='edge'),
                kernel[:, :, b][::-1, ::-1], mode='valid') for b in range(4)]) for x in gt])
            with h5py.File(raw, 'w') as f:
                f['gt'], f['ms'] = gt, low[:, :, 1::4, 2::4]
            original = sha256(raw)
            manifest = dict(sensor='QB', band_order=['B', 'G', 'R', 'NIR'], source_provenance={
                f'raw_{split}_{key}': value for split in ('train', 'val')
                for key, value in (('path', str(raw)), ('sha256', original))})
            result = qb_phase_residuals(manifest, max_samples=1)
            for row in result['splits'].values():
                self.assertEqual(len(row['sample_ids']), 1)
                self.assertEqual(row['rows'][0]['minimum_residual_phase'], [1, 2])
                self.assertGreater(row['rows'][0]['phase2_mae'], 0)
            self.assertEqual(sha256(raw), original)


if __name__ == '__main__':
    unittest.main()
