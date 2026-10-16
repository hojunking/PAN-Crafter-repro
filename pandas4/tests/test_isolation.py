"""S4 adapters may not change live s1/s3 globals or need Torch on sheet host."""
from copy import deepcopy
import subprocess
import sys
import unittest

from panda40 import reporting as original_reporting, publication as original_publication
from pandas4 import reporting, publication
from pandas4.tests.test_reporting import config


class IsolationTests(unittest.TestCase):
    def test_campaign_modules_are_isolated(self):
        self.assertEqual(original_reporting.CAMPAIGN_ID,
                         'PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1')
        self.assertEqual(original_reporting.RECORD_PREFIX, 'P40-')
        self.assertEqual(set(original_publication.LANES), {'s1', 's3'})
        self.assertEqual(set(publication.LANES), {'s4'})
        self.assertIsNot(reporting.build_record, original_reporting.build_record)
        for server in ('s1', 's3'):
            with self.assertRaises(ValueError): reporting.destination(server)
        with self.assertRaises(ValueError): original_reporting._config(config())
        for key, value in (('campaign_id', original_reporting.CAMPAIGN_ID),
                           ('run_id', 'P40_FOREIGN'), ('aligner_kind', 's2d')):
            cfg = deepcopy(config()); cfg[key] = value
            with self.assertRaises(ValueError): reporting._config(cfg)

    def test_publisher_and_cost_import_without_torch(self):
        code = ('import sys; import pandas4.publication, pandas4.transport, pandas4.cost; '
                'assert "torch" not in sys.modules; print("TORCH_FREE")')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), 'TORCH_FREE')
