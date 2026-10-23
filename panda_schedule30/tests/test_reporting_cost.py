from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_schedule30 import plan, reporting as R
from panda_schedule30.cost import geometry_key, signature
from panda_schedule30.cost.timing import timing_for_config
from panda_schedule30.evaluation import target_gap


class ReportingCostTests(unittest.TestCase):
    def test_all_registered_horizons_and_continuation_lineage(self):
        for server in ('s1', 's3'):
            for group in plan.groups(server):
                for case, replica in group['runs']:
                    cfg = plan.resolve_case(server, case, replica)
                    R._config(cfg)
                    end = cfg['planned_horizon']
                    lineage = R.update_lineage(cfg, end)
                    self.assertEqual(lineage['local_updates'], cfg['continuation_updates'])
                    self.assertEqual(lineage['lifetime_updates'], end)
                    self.assertEqual(geometry_key(cfg, end), cfg['policy_id'].lower())
                    with self.assertRaises(ValueError):
                        geometry_key(cfg, end+1)

    def test_schedule_cannot_be_relabelled_in_evidence(self):
        cfg = plan.resolve_case('s1', 'S020F25C35')
        for key, wrong in (('inherited_updates', 0), ('planned_horizon', 50000),
                           ('source_case', 'S010BASE'), ('start_update', 25000)):
            changed = deepcopy(cfg)
            changed[key] = wrong
            with self.assertRaises(ValueError, msg=key):
                R._config(changed)
        self.assertTrue(target_gap(.9585, 2.040, server='s1')['target_pair_pass'])
        self.assertFalse(target_gap(.95849, 2.039, server='s1')['target_pair_pass'])

    def test_deployment_cost_is_shared_only_for_identical_inference_geometry(self):
        short = plan.resolve_case('s1', 'S020C30')
        fork = plan.resolve_case('s1', 'S020F25C35')
        small = plan.resolve_case('s1', 'S010C30')
        self.assertEqual(signature(short, 1000), signature(fork, 26000))
        self.assertNotEqual(signature(short, 1000), signature(small, 1000))

    def test_continuation_time_counts_only_observed_new_updates(self):
        cfg = plan.resolve_case('s1', 'S020F25C35')
        blank = {'sheet_fields': {'Eval(h)': 0.01}, 'proof': {},
                 'omitted': {'Train(h)': 'noncontiguous original prefix'}}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'runs'/cfg['run_id']/'attempt_01'/'train.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(''.join(json.dumps({'step': s, 'seconds_including_IO': t})+'\n'
                                    for s, t in ((25001, 2.), (25002, 3.), (25003, 5.))))
            with patch('panda_schedule30.cost.timing.original_timing', side_effect=lambda *a: deepcopy(blank)):
                value = timing_for_config(cfg, 25002, root)
                self.assertAlmostEqual(value['sheet_fields']['Train(h)'], 5./3600)
                self.assertEqual(value['proof']['Train(h)']['contiguous_updates'], 2)
                self.assertEqual(value['sheet_fields']['Eval(h)'], .01)
                path.write_text(json.dumps({'step':25002, 'seconds_including_IO':3.})+'\n')
                broken = timing_for_config(cfg, 25002, root)
                self.assertNotIn('Train(h)', broken['sheet_fields'])
                self.assertIn('Noncontiguous', broken['omitted']['Train(h)'])


if __name__ == '__main__':
    unittest.main()
