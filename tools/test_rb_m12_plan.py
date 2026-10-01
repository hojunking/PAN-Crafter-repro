"""Production plan/deployment isolation tests (no GPU, data or network)."""
import copy
import unittest
from unittest.mock import patch
from panda_rb_m12 import plan
from panda_rb_m12.deployment import start


class PlanTests(unittest.TestCase):
    def test_registered_counts(self):
        result = plan.validate_plan()
        self.assertEqual(result['training_runs'], 120)
        self.assertEqual(result['stress_curves'], 288)

    def test_configs_all_ten_interventions(self):
        for case in plan.training_runs('s1')[:10]:
            cfg = plan.build_config(case['run_id'])
            self.assertEqual(cfg['seed'], case['seed'])
            self.assertNotIn('fh20r1', cfg)
            self.assertNotIn('panda_rb', cfg)
            self.assertEqual(cfg['panda_rb_m12']['alpha'], 0. if case['case_id'] == 'H0' else 1.)
            self.assertEqual(cfg['num_iter'], 50000)

    def test_seed_drift_rejected(self):
        reg = plan.registry(); reg['seed_table']['s1'][0] += 1
        with patch.object(plan, 'registry', return_value=reg), self.assertRaises(ValueError):
            plan.validate_plan()

    def test_selector_drift_rejected(self):
        reg = plan.registry(); reg['training_runs'][0]['secondary_selection'] = 'BEST_HQNR'
        with patch.object(plan, 'registry', return_value=reg), self.assertRaises(ValueError):
            plan.validate_plan()

    def test_route_drift_rejected(self):
        reg = plan.registry(); reg['templates'][4]['aligner_weight'] = 'QFULL'
        with patch.object(plan, 'registry', return_value=reg), self.assertRaises(ValueError):
            plan.validate_plan()

    def test_lr_drift_rejected(self):
        reg = plan.registry(); reg['training']['lr_A'] = 1e-4
        with patch.object(plan, 'registry', return_value=reg), self.assertRaises(ValueError):
            plan.validate_plan()

    def test_no_unknown_or_legacy_run(self):
        with self.assertRaises(ValueError):
            plan.case_for('RB01_WV3_S1_R1_QFULL')
        with self.assertRaises(ValueError):
            plan.training_runs('s2')

    def test_start_requires_activation(self):
        with self.assertRaises(PermissionError):
            start('s1')

    def test_all_servers_dry_no_subprocess_or_writes(self):
        with patch('panda_rb_m12.deployment.subprocess.check_output', side_effect=AssertionError), \
             patch('panda_rb_m12.deployment.atomic_json', side_effect=AssertionError):
            for server in plan.SERVERS:
                self.assertEqual(start(server, dry_run=True)['training_runs'], 40)


if __name__ == '__main__':
    unittest.main()
