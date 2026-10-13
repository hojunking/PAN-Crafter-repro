import json
from pathlib import Path
import unittest

from .plan import resolve_case
from .stream import learning_rates

ROOT=Path(__file__).resolve().parents[1]


class ScheduleTests(unittest.TestCase):
    def config(self,schedule='H50'):
        baseline=json.loads((ROOT/'config/edgeres48/s1.baseline.json').read_text())
        return resolve_case('s1',baseline=baseline,schedule=schedule)

    def test_warmup_and_endpoint(self):
        c=self.config()
        self.assertEqual(learning_rates(c,1)['A'],c['lr_A']/100)
        self.assertEqual(learning_rates(c,100)['T'],c['lr_U'])
        self.assertEqual(learning_rates(c,50000),{'A':0.,'T':0.})
        self.assertGreater(learning_rates(c,101)['T'],0)
        self.assertLess(learning_rates(c,101)['T'],c['lr_U'])

    def test_long_horizon_not_continuation(self):
        short,long=self.config(),self.config('H100')
        self.assertEqual(learning_rates(short,50000)['T'],0)
        self.assertGreater(learning_rates(long,50000)['T'],0)

    def test_update_domain(self):
        for update in (0,-1,50001,1.5,True):
            with self.assertRaises(ValueError): learning_rates(self.config(),update)


if __name__=='__main__': unittest.main()
