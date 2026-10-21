"""Registered ZERO30 scientific choices; no data, GPU or cloud access."""
from copy import deepcopy
import csv
import io
import unittest

from panda_mature30 import plan as historical
from panda_zero30 import plan
from panda_zero30.contract import BUDGET, CAMPAIGN_ID
from panda_zero30.evaluation import LANES, target_gap


class PlanTests(unittest.TestCase):
    def test_ordered_two_seed_zero_and_compute_matched_pairs(self):
        expected = [
            ('zero_replicates', False, [('ZERO', 'R1'), ('ZERO', 'R2')]),
            ('rotation_pair_r1', False, [('PAIR0', 'R1'), ('ROTCONS', 'R1')]),
            ('rotation_pair_r2', True, [('PAIR0', 'R2'), ('ROTCONS', 'R2')]),
        ]
        for server in ('s1', 's3'):
            groups = plan.groups(server)
            self.assertEqual([(g['name'], g['optional'], g['runs']) for g in groups], expected)
            self.assertTrue(all(g['admission'] == 'WHOLE_GROUP_OR_SKIP' for g in groups))
            self.assertTrue(all(case not in plan.DONORS for g in groups for case, _ in g['runs']))

    def test_only_registered_cases_replicas_and_servers(self):
        for server, case, replica, attempt in (
            ('s2', 'ZERO', 'R1', 1), ('s1', 'FULL', 'R1', 1),
            ('s1', 'ZERO', 'R3', 1), ('s1', 'ZERO', 'R1', 0),
            ('s1', 'ZERO', 'R1', True), ('s1', plan.DONORS[0], 'R2', 1),
        ):
            with self.subTest(server=server, case=case, replica=replica, attempt=attempt):
                with self.assertRaises(ValueError):
                    plan.resolve_case(server, case, replica, attempt=attempt)

    def test_cases_preserve_baseline_streams_and_horizon_but_no_soft(self):
        for server in ('s1', 's3'):
            for replica in ('R1', 'R2'):
                source = historical.resolve_case(server, 'HALF', replica)
                for case in ('ZERO', 'PAIR0', 'ROTCONS'):
                    cfg = plan.resolve_case(server, case, replica)
                    for name in ('baseline_resolved', 'seed', 'stream_seeds', 'augmentation',
                                 'training_spec', 'evaluation_steps', 'student_architecture'):
                        self.assertEqual(cfg[name], source[name], name)
                    self.assertEqual(cfg['planned_horizon'], 50000)
                    self.assertEqual(cfg['shift_scale'], 0.)
                    self.assertEqual(cfg['stage2_spec']['beta'], 0.)
                    self.assertFalse(cfg['stage1_retrained'])
                    self.assertFalse(cfg['inference_tta'])
                    self.assertEqual(cfg['student_views'], 1 if case == 'ZERO' else 2)
                    self.assertEqual(cfg['rotation_consistency_weight'], .05 if case == 'ROTCONS' else 0.)
                    self.assertEqual(cfg['upsampler'], 'interp23')
                    self.assertEqual(cfg['frame'], 'pan')
                    self.assertEqual(cfg['campaign_id'], CAMPAIGN_ID)
                    self.assertTrue(cfg['run_id'].startswith('PZ30_WV3_' + server.upper() + '_'))

    def test_new_target_is_consistent_and_annotations_do_not_change_hqnr_rank(self):
        for server in ('s1', 's3'):
            self.assertEqual(plan.default_config(server)['joint_target'], {'HQNR': .959, 'ERGAS': 2.045})
            self.assertEqual(LANES[server]['hqnr_min'], .959)
            self.assertEqual(LANES[server]['ergas_max'], 2.045)
            self.assertTrue(target_gap(.959, 2.045, server=server)['target_pair_pass'])
            self.assertFalse(target_gap(.959 - 1e-8, 2.045, server=server)['target_pair_pass'])
            self.assertFalse(target_gap(.959, 2.045 + 1e-8, server=server)['target_pair_pass'])
            self.assertEqual(target_gap(.959, 2.045, server=server)['D_inf'], 0.)
            for group in plan.groups(server):
                for case, replica in group['runs']:
                    cfg = plan.resolve_case(server, case, replica)
                    self.assertEqual((cfg['hqnr_min'], cfg['ergas_max']), (.959, 2.045))
        high = {'HQNR': .961, 'ERGAS': 2.30, 'completed_update': 4000}
        balanced = {'HQNR': .959, 'ERGAS': 2.04, 'completed_update': 30000}
        self.assertGreater(plan.rank(high), plan.rank(balanced))

    def test_original_campaign_is_unchanged_and_configs_are_independent(self):
        before = {s: deepcopy(historical.default_config(s)) for s in ('s1', 's3')}
        old_rows = {s: historical.resolve_case(s, 'HALF') for s in ('s1', 's3')}
        for server in ('s1', 's3'):
            cfg = plan.default_config(server)
            plan.validate_config(cfg)
            cfg['groups'][0]['runs'].append(('UNREGISTERED', 'R1'))
            with self.assertRaises(ValueError):
                plan.validate_config(cfg)
            row = plan.resolve_case(server, 'ZERO')
            row['baseline_resolved']['teacher_architecture']['width'] = -1
            self.assertNotEqual(plan.resolve_case(server, 'ZERO')['baseline_resolved']['teacher_architecture']['width'], -1)
            self.assertEqual(historical.default_config(server), before[server])
            self.assertEqual(historical.resolve_case(server, 'HALF'), old_rows[server])
            self.assertEqual(old_rows[server]['hqnr_min'], .960)

    def test_receipt_paths_are_same_server_and_local_without_reselection(self):
        for server in ('s1', 's3'):
            cfg = plan.default_config(server)
            self.assertFalse(cfg['allow_cross_server_donor'])
            self.assertIn('NO_RESELECTION', cfg['donor_policy'])
            self.assertEqual(cfg['donor_receipt'], plan.donor_receipt_path(server))
            self.assertIn('/' + server + '/WV3/donors/rre_selection.json', cfg['donor_receipt'])

    def test_catalog_has_six_cases_and_no_new_stage1(self):
        for server in ('s1', 's3'):
            rows = list(csv.DictReader(io.StringIO(plan.catalog_csv(server))))
            self.assertEqual(len(rows), 6)
            self.assertEqual([r['order'] for r in rows], [str(i) for i in range(1, 7)])
            self.assertEqual([r['optional'] for r in rows], ['False'] * 4 + ['True'] * 2)
            self.assertTrue(all(r['updates'] == '50000' for r in rows))
        self.assertEqual(BUDGET['total_seconds'], 30 * 3600)
        self.assertEqual(BUDGET['closeout_seconds'], 1800)
        self.assertEqual(BUDGET['safety_factor'], 1.12)


if __name__ == '__main__':
    unittest.main()
