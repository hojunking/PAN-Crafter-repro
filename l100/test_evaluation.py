"""Exercise the same audited GF2 kernels, not a new L100 metric implementation."""
import unittest

import g20.evaluation as audited
import l100.evaluation as current


class EvaluationTests(unittest.TestCase):
    def test_all_numerical_entrypoints_are_the_audited_objects(self):
        for name in current.__all__:
            self.assertIs(getattr(current, name), getattr(audited, name), name)

    def test_signed_ds_validation_not_replaced_with_a_scalar_proxy(self):
        from g20.test_evaluation import record
        metric = record(100000)['fr']
        current.validate_signed_ds(metric)
        metric['signed_ds']['delta'][0][0] += .01
        with self.assertRaises(ValueError):
            current.validate_signed_ds(metric)


if __name__ == '__main__':
    unittest.main()
