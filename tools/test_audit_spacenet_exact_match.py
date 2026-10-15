import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import tifffile

from tools.audit_spacenet_exact_match import (audit, d4, index_candidates,
                                             match_patch, query_hashes, read_tiff)


class ExactMatchTest(unittest.TestCase):
    def setUp(self):
        self.source = np.random.default_rng(873).integers(0, 2048, (8, 45, 53), dtype=np.uint16)

    def search(self, patch, source=None, cap=4096, max_matches=256):
        source = self.source if source is None else source
        candidates, counts = index_candidates(source, query_hashes(patch), 7, cap)
        return match_patch(source, patch, candidates, counts, cap, max_matches)

    def test_all_d4_even_and_odd_and_nonsquare(self):
        for height, width in [(8, 8), (7, 7), (6, 9)]:
            crop = self.source[:, 12:12 + height, 17:17 + width]
            for flip in (False, True):
                for k in range(4):
                    patch = d4(crop, k, flip)
                    result = self.search(patch)
                    self.assertEqual(result['status'], 'EXACT_MATCH')
                    self.assertTrue(result['search_complete'])
                    self.assertEqual(len(result['matches']), 1)
                    hit = result['matches'][0]
                    self.assertEqual((hit['source_y'], hit['source_x']), (12, 17))
                    aligned = d4(patch, hit['patch_to_source_rot90_ccw'], hit['patch_to_source_flip_x'])
                    np.testing.assert_array_equal(aligned, crop)

    def test_no_match_even_if_anchor_matches(self):
        patch = self.source[:, 12:20, 17:25].copy()
        patch[0, 0, 0] += 1
        result = self.search(patch)
        self.assertEqual(result['status'], 'NO_EXACT_MATCH')
        self.assertGreater(result['full_crop_comparisons'], 0)

    def test_duplicates_retained(self):
        source = self.source.copy()
        source[:, 25:33, 30:38] = source[:, 12:20, 17:25]
        result = self.search(source[:, 12:20, 17:25], source)
        self.assertEqual({(m['source_y'], m['source_x']) for m in result['matches']}, {(12, 17), (25, 30)})

    def test_cap_never_reports_complete_negative(self):
        source = np.zeros((8, 20, 20), dtype=np.uint16)
        patch = np.zeros((8, 8, 8), dtype=np.uint16)
        result = self.search(patch, source, cap=1)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_MATCH_FOUND')
        self.assertFalse(result['search_complete'])
        self.assertTrue(result['anchor_candidates_truncated'])

    def test_match_output_cap_explicit(self):
        source = np.zeros((8, 20, 20), dtype=np.uint16)
        result = self.search(np.zeros((8, 8, 8), dtype=np.uint16), source, max_matches=2)
        self.assertEqual(result['status'], 'EXACT_MATCH')
        self.assertEqual(len(result['matches']), 2)
        self.assertTrue(result['matches_truncated'])
        self.assertFalse(result['search_complete'])

    def test_float_dn_exact_supported_and_fraction_rejected(self):
        patch = self.source[:, 12:20, 17:25].astype(np.float64)
        self.assertEqual(self.search(patch)['status'], 'EXACT_MATCH')
        patch[0, 0, 0] += .25
        result = match_patch(self.source, patch, {}, {})
        self.assertEqual(result['status'], 'NO_EXACT_MATCH_UINT16_DN_INCOMPATIBLE')

    def test_full_h5_audit_and_tiff_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            tiff = p / 'source.tif'
            tifffile.imwrite(tiff, self.source.transpose(1, 2, 0), photometric='minisblack', metadata={'axes': 'YXS'})
            loaded, meta = read_tiff(tiff)
            np.testing.assert_array_equal(loaded, self.source)
            h5 = p / 'patches.h5'
            with h5py.File(h5, 'w') as f:
                f['gt'] = self.source[None, :, 12:20, 17:25].astype(np.float64)
            result = audit(tiff, [('synthetic', h5, 'gt')], block_rows=7)
            self.assertEqual(result['summary'], {'EXACT_MATCH': 1})
            self.assertEqual(len(result['source']['sha256']), 64)
            self.assertEqual(result['datasets'][0]['rows'][0]['row'], 0)
            with self.assertRaises(ValueError):
                read_tiff(tiff, max_pixels=1)


if __name__ == '__main__':
    unittest.main()
