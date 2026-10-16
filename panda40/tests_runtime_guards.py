"""Clock/deadline regression checks with no experiment or external writes."""
import unittest
from unittest.mock import patch

from .runtime import Campaign, BudgetExpired


class RuntimeBudgetTests(unittest.TestCase):
    def campaign(self):
        campaign = object.__new__(Campaign)
        campaign.state = {'started_unix':10000.,'elapsed_seconds':0.}
        campaign.initial_elapsed = 0.
        campaign.tick = 0.
        campaign.stop_requested = False
        campaign.config = {'budget':{'total_seconds':144000.,'closeout_seconds':1800.}}
        return campaign

    def test_wall_clock_forward_then_backward_never_reduces_elapsed(self):
        campaign = self.campaign()
        with patch('panda40.runtime.time.time',side_effect=[10000.,20000.,9000.]), \
                patch('panda40.runtime.time.monotonic',side_effect=[1.,2.,3.]):
            values = [campaign.elapsed() for _ in range(3)]
        self.assertEqual(values,[1.,10000.,10000.])

    def test_persisted_elapsed_and_monotonic_continue_to_count_downtime(self):
        campaign = self.campaign()
        campaign.initial_elapsed = 700.
        campaign.state['elapsed_seconds'] = 700.
        with patch('panda40.runtime.time.time',return_value=9000.), \
                patch('panda40.runtime.time.monotonic',side_effect=[10.,50.]):
            self.assertEqual(campaign.elapsed(),710.)
            self.assertEqual(campaign.elapsed(),750.)

    def test_closeout_is_reserved_and_hard_deadline_cannot_be_reset(self):
        campaign = self.campaign()
        campaign.state['elapsed_seconds'] = 142200.
        with patch('panda40.runtime.time.time',return_value=10000.), \
                patch('panda40.runtime.time.monotonic',return_value=1.):
            with self.assertRaisesRegex(BudgetExpired,'CLOSEOUT'):
                campaign.check()
            campaign.check(reserve=False)
            campaign.state['elapsed_seconds'] = 144000.
            with self.assertRaisesRegex(BudgetExpired,'CLOSEOUT'):
                campaign.check(reserve=False)
            # Even corrupting the persisted value downward cannot erase the
            # current process's observed elapsed-time high-water mark.
            campaign.state['elapsed_seconds'] = 0.
            with self.assertRaises(BudgetExpired):
                campaign.check(reserve=False)

    def test_explicit_stop_is_respected_even_before_time_budget(self):
        campaign = self.campaign()
        campaign.stop_requested = True
        with patch('panda40.runtime.time.time',return_value=10000.), \
                patch('panda40.runtime.time.monotonic',return_value=1.):
            with self.assertRaisesRegex(BudgetExpired,'STOP_REQUESTED'):
                campaign.check(reserve=False)


if __name__ == '__main__':
    unittest.main()
