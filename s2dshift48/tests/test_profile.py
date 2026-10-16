import unittest
import torch

from s2dshift48.profile import compare


class ProfileTests(unittest.TestCase):
    def test_actual_counts_scope_and_rng_restoration(self):
        state = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        report = compare(bands=8, sides=(64,), repeats=1, warmup=0, threads=1)
        direct, s2d = report['rows']
        self.assertEqual(direct['parameters'], 105458)
        self.assertEqual(s2d['parameters'], 124898)
        self.assertEqual(direct['conv_linear_macs'] - s2d['conv_linear_macs'], 3981312)
        self.assertEqual(s2d['conv_linear_macs'], sum(x['macs'] for x in s2d['module_costs']))
        self.assertEqual(s2d['conv_linear_flops_2mac'], 2 * s2d['conv_linear_macs'])
        self.assertGreater(s2d['cpu_latency_ms_median'], 0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        self.assertIn('PAN_warp', report['excluded_from_macs'])

    def test_invalid_shapes_limits_rejected(self):
        for kwargs in ({'bands': 3}, {'sides': [63]}, {'sides': []}, {'repeats': 0}, {'threads': 0}):
            with self.assertRaises(ValueError):
                compare(**kwargs)


if __name__ == '__main__':
    unittest.main()
