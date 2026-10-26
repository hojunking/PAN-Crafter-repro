import unittest
from panda_sensor5.wv3 import active_clock, controlled_class

class TestWV3Control(unittest.TestCase):
    def test_first_pause_does_not_consume_elapsed(self):
        state={'started_unix':100,'elapsed_seconds':50}
        c=active_clock(state,1000)
        self.assertEqual(c['active_seconds_at_open'],50)
        self.assertEqual(c['excluded_gap_seconds'],850)
        self.assertEqual(state['started_unix'],100)

    def test_dataset_wait_excluded_after_controlled_close(self):
        state={'started_unix':100,'elapsed_seconds':70,
               'sensor5_control':{'last_closed_unix':1000}}
        self.assertEqual(active_clock(state,100000)['excluded_gap_seconds'],99000)
        self.assertEqual(active_clock(state,100000)['active_seconds_at_open'],70)

    def test_future_clock_rejected(self):
        with self.assertRaises(ValueError):
            active_clock({'started_unix':100,'elapsed_seconds':100},100)

    def test_registered_campaign_only(self):
        with self.assertRaises(ValueError):
            controlled_class('other')
