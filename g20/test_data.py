import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from g20.data import (AUGMENTATION, RECIPE, G20Dataset, _create_lp_cache, build_dataset,
                      canonical_band_indices, canonical_sha, prepare_data, sha256_file, verify_qb_msfix)
from g20.data import verify_lp_cache
from g20.plan import sensor_spec


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.lp = self.root / 'source.h5', self.root / 'lp.h5'
        self.spec = replace(sensor_spec('GF2'), band_order=('B', 'G', 'R', 'NIR'))
        rng = np.random.default_rng(4)
        self.native = dict(pan=rng.uniform(0, 1023, (2, 1, 64, 64)),
                           ms=rng.uniform(0, 1023, (2, 4, 16, 16)),
                           lms=rng.uniform(0, 1023, (2, 4, 64, 64)),
                           gt=rng.uniform(0, 1023, (2, 4, 64, 64)))
        with h5py.File(self.source, 'w') as f:
            for k, v in self.native.items():
                f[k] = v
        self.source_sha = sha256_file(self.source)
        _create_lp_cache(self.source, self.lp, self.source_sha)

    def dataset(self, augment=False):
        return G20Dataset(self.source, self.lp, spec=self.spec, split='train', augment=augment)

    def manifest(self):
        return dict(schema='G20_DATA_v1', sensor='GF2', num_bands=4, max_pixel=1023,
                    band_order=['B', 'G', 'R', 'NIR'], mtf_sensor='GF2', recipe=RECIPE,
                    augmentation_sha256=canonical_sha(AUGMENTATION),
                    splits={'train': dict(dataroot=str(self.source), lpan_path=str(self.lp),
                                          sha256=self.source_sha, lpan_sha256=sha256_file(self.lp),
                                          source_identity='test-source-two-patches')})

    def test_gf2_normalization_and_explicit_fixed_views(self):
        before = torch.get_rng_state().clone()
        d = self.dataset()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(d.base_count, 2)
        gt, lms, ms, lp, pan, meta = d.base(0)
        np.testing.assert_allclose(pan.numpy(), self.native['pan'][0] * 2 / 1023 - 1, atol=2e-7)
        self.assertEqual(meta.tolist(), [0, 0, 0, 0])
        rotated = d[(0, 1)]
        for base, view in zip((gt, lms, ms, lp, pan), rotated[:-1]):
            np.testing.assert_array_equal(view, np.rot90(base.numpy()[:, ::-1, ::-1], 1, axes=(1, 2)))
        self.assertEqual(rotated[-1].tolist(), [0, 1, 1, 1])
        self.assertEqual(len(self.dataset(augment=True)), 8)
        self.assertEqual(sha256_file(self.source), self.source_sha)

    def test_lp_is_native_pan_gaussian_before_view(self):
        from tools.repair_lpan import make_lpan
        with h5py.File(self.lp) as f:
            np.testing.assert_array_equal(f['lpan'][:], make_lpan(self.native['pan'].astype(np.float64)).astype(np.float32))
            self.assertEqual(f['lpan'].dtype, np.dtype('float32'))
            self.assertEqual(f.attrs['recipe_sha256'], canonical_sha(RECIPE))

    def test_manifest_sensor_hash_and_band_order_fail_closed(self):
        d = build_dataset(self.manifest(), 'train')
        self.assertEqual(d.max_pixel, 1023)
        for key, value in [('max_pixel', 2047), ('num_bands', 8), ('mtf_sensor', 'QB')]:
            manifest = self.manifest()
            manifest[key] = value
            with self.assertRaises(ValueError):
                build_dataset(manifest, 'train')
        m = self.manifest()
        m['splits']['train']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            build_dataset(m, 'train')
        with self.assertRaises(ValueError):
            canonical_band_indices(None)
        with self.assertRaises(ValueError):
            prepare_data(self.root, 's5', sensor_spec('GF2'))

    def test_lp_corruption_rejected_even_with_matching_source_attrs(self):
        self.assertEqual(len(verify_lp_cache(self.source, self.lp, self.source_sha)['sample_order_sha256']), 64)
        self.lp.chmod(0o600)
        with h5py.File(self.lp, 'r+') as f:
            f['lpan'][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'correspondence'):
            verify_lp_cache(self.source, self.lp, self.source_sha)

    def test_qb_msfix_requires_unchanged_gt_pan_and_actual_recipe(self):
        from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE
        from tools.repair_qb_ms import mtf_down
        class Wald:
            def interp23tap(self, a, ratio):
                return a.repeat(ratio, axis=0).repeat(ratio, axis=1)
        fixed = self.root / 'fixed.h5'
        kernel = genmtf_matlab(GNYQ_TABLE['QB'], 4, 41)
        with h5py.File(fixed, 'w') as f:
            for key in ('gt', 'pan'):
                f[key] = self.native[key]
            ms = np.stack([mtf_down(x, kernel) for x in self.native['gt']])
            f['ms'] = ms
            f['lms'] = ms.repeat(4, axis=2).repeat(4, axis=3)
        def verify():
            return verify_qb_msfix(self.source, fixed, raw_sha256=self.source_sha,
                                   fixed_sha256=sha256_file(fixed), band_order=('B', 'G', 'R', 'NIR'), wald=Wald())
        self.assertEqual(verify()['samples_verified'], 2)
        with h5py.File(fixed, 'r+') as f:
            f['pan'][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'changed pan'):
            verify()
        self.assertEqual(sha256_file(self.source), self.source_sha)


if __name__ == '__main__':
    unittest.main()
