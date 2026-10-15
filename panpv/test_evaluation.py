import unittest
import numpy as np
from panpv.evaluation import RRAuditAccumulator, valid_windows


class RRAuditTests(unittest.TestCase):
    def test_perfect_and_infinite_psnr_explicit(self):
        image = np.random.default_rng(3).integers(1, 1800, (8, 64, 64))
        acc = RRAuditAccumulator()
        acc.add(image, image, np.ones((64, 64), dtype=bool), sample_id='a')
        result = acc.result()
        self.assertEqual(result['valid_windows'], 4)
        self.assertEqual(result['metrics']['ergas'], 0)
        self.assertEqual(result['metrics']['rmse'], 0)
        self.assertAlmostEqual(result['metrics']['q8'], 1, places=6)
        self.assertAlmostEqual(result['metrics']['ssim'], 1, places=6)
        self.assertIsNone(result['metrics']['psnr'])
        self.assertEqual(result['undefined_window_counts']['psnr'], 4)

    def test_common_mask_and_global_phase(self):
        mask = np.ones((64, 64), dtype=bool)
        mask[0, 0] = False
        self.assertEqual(len(list(valid_windows(mask))), 3)
        self.assertEqual(len(list(valid_windows(mask, origin=(1, 1)))), 1)
        self.assertEqual(len(list(valid_windows(np.ones((31, 31), bool)))), 0)

    def test_duplicate_and_bad_shapes_rejected(self):
        image = np.ones((8, 32, 32))
        acc = RRAuditAccumulator()
        acc.add(image, image, np.ones((32, 32), bool), sample_id=0)
        with self.assertRaises(ValueError):
            acc.add(image, image, np.ones((32, 32), bool), sample_id=0)
        with self.assertRaises(ValueError):
            acc.add(image[:3], image[:3], np.ones((32, 32), bool), sample_id=1)
        with self.assertRaises(ValueError):
            list(valid_windows(np.full((32, 32), .5)))


if __name__ == '__main__':
    unittest.main()
