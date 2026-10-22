"""Registered positive-scale queues and existing-Hard/optimizer choices."""
from copy import deepcopy
import csv
import io
import unittest

from panda_mature30 import plan as historical
from panda_zero30 import plan as zero_historical
from panda_scale30 import plan
from panda_scale30.contract import BUDGET, CAMPAIGN_ID, CASES
from panda_scale30.evaluation import LANES, target_gap


class PlanTests(unittest.TestCase):
    def test_predeclared_per_server_queues_with_local_controls(self):
        expected = {
            's1': [('S020BASE','R1'),('S010BASE','R1'),('S020LR075','R1'),
                   ('S030BASE','R1'),('S040BASE','R1'),('S050BASE','R1'),
                   ('S020BASE','R2'),('S010BASE','R2')],
            's3': [('S020BASE','R1'),('S020H050','R1'),('S020LR150','R1'),
                   ('S020H200','R1'),('S020LR150H050','R1'),('S010LR150H050','R1'),
                   ('S020BASE','R2'),('S020LR150H050','R2')],
        }
        for server in ('s1','s3'):
            groups = plan.groups(server)
            self.assertEqual([r for g in groups for r in g['runs']], expected[server])
            self.assertEqual([g['optional'] for g in groups], [False]*7+[True])
            self.assertTrue(all(g['admission']=='WHOLE_GROUP_OR_SKIP' for g in groups))
            self.assertTrue(all(c not in plan.DONORS for g in groups for c,_ in g['runs']))

    def test_rejects_unregistered_server_case_replica_attempt(self):
        for server, case, replica, attempt in (
            ('s2','S020BASE','R1',1), ('s1','ZERO','R1',1),
            ('s1','ROTCONS','R1',1), ('s1','S020LR150','R1',1),
            ('s3','S050BASE','R1',1), ('s1','S020BASE','R3',1),
            ('s1','S020BASE','R1',0), ('s1','S020BASE','R1',True),
            ('s1',plan.DONORS[0],'R2',1),
        ):
            with self.subTest(server=server,case=case,replica=replica,attempt=attempt):
                with self.assertRaises(ValueError):
                    plan.resolve_case(server,case,replica,attempt=attempt)

    def test_same_seed_baseline_c4_horizon_with_only_registered_tuning(self):
        for server in ('s1','s3'):
            for group in plan.groups(server):
                for case, replica in group['runs']:
                    cfg = plan.resolve_case(server,case,replica)
                    source = historical.resolve_case(server,'HALF',replica)
                    params = CASES[case]
                    for name in ('baseline_resolved','seed','stream_seeds','augmentation',
                                 'evaluation_steps','student_architecture'):
                        self.assertEqual(cfg[name],source[name],name)
                    training = deepcopy(source['training_spec'])
                    training['lr'] = params['lr_S']
                    self.assertEqual(cfg['training_spec'],training)
                    self.assertEqual(cfg['lr_U'],params['lr_S'])
                    self.assertEqual(cfg['lr_S'],params['lr_S'])
                    self.assertEqual(cfg['planned_horizon'],50000)
                    self.assertEqual(cfg['shift_scale'],params['scale'])
                    self.assertGreaterEqual(cfg['shift_scale'],.1)
                    self.assertLessEqual(cfg['shift_scale'],.5)
                    self.assertEqual(cfg['stage2_spec']['alpha'],params['alpha'])
                    self.assertEqual(cfg['stage2_spec']['gamma'],params['gamma'])
                    self.assertEqual(cfg['stage2_spec']['beta'],0.)
                    self.assertFalse(cfg['stage1_retrained'])
                    self.assertFalse(cfg['inference_tta'])
                    self.assertEqual(cfg['student_views'],1)
                    self.assertEqual(cfg['rotation_consistency_weight'],0.)
                    self.assertEqual(cfg['pair_transform'],'NONE')
                    self.assertEqual(cfg['campaign_id'],CAMPAIGN_ID)
                    self.assertTrue(cfg['run_id'].startswith('PS30_WV3_'+server.upper()+'_'))

    def test_same_checkpoint_joint_target_does_not_change_hqnr_ranking(self):
        for server in ('s1','s3'):
            self.assertEqual(plan.default_config(server)['joint_target'],{'HQNR':.959,'ERGAS':2.045})
            self.assertEqual(LANES[server]['hqnr_min'],.959)
            self.assertEqual(LANES[server]['ergas_max'],2.045)
            self.assertTrue(target_gap(.959,2.045,server=server)['target_pair_pass'])
            self.assertFalse(target_gap(.959-1e-8,2.045,server=server)['target_pair_pass'])
            self.assertFalse(target_gap(.959,2.045+1e-8,server=server)['target_pair_pass'])
        self.assertGreater(plan.rank({'HQNR':.961,'ERGAS':2.3,'completed_update':4000}),
                           plan.rank({'HQNR':.959,'ERGAS':2.04,'completed_update':30000}))

    def test_historical_configs_unchanged_and_registry_copies_independent(self):
        mature_before = {s:deepcopy(historical.default_config(s)) for s in ('s1','s3')}
        zero_before = {s:deepcopy(zero_historical.default_config(s)) for s in ('s1','s3')}
        for server in ('s1','s3'):
            cfg = plan.default_config(server)
            plan.validate_config(cfg)
            cfg['cases']['S020BASE']['scale'] = 0.
            with self.assertRaises(ValueError):
                plan.validate_config(cfg)
            row = plan.resolve_case(server,'S020BASE')
            row['tuning_parameters']['scale'] = 0.
            self.assertEqual(plan.resolve_case(server,'S020BASE')['shift_scale'],.2)
            self.assertEqual(historical.default_config(server),mature_before[server])
            self.assertEqual(zero_historical.default_config(server),zero_before[server])

    def test_local_donor_reuse_catalog_and_budget(self):
        for server in ('s1','s3'):
            cfg = plan.default_config(server)
            self.assertFalse(cfg['allow_cross_server_donor'])
            self.assertIn('NO_RESELECTION',cfg['donor_policy'])
            self.assertIn('/'+server+'/WV3/donors/rre_selection.json',cfg['donor_receipt'])
            rows = list(csv.DictReader(io.StringIO(plan.catalog_csv(server))))
            self.assertEqual(len(rows),8)
            self.assertTrue(all(row['updates']=='50000' for row in rows))
            self.assertTrue(all(row['views']=='1' and row['lambda_cons']=='0.0' for row in rows))
            self.assertEqual([row['optional'] for row in rows],['False']*7+['True'])
        self.assertEqual(BUDGET['total_seconds'],30*3600)
        self.assertEqual(BUDGET['closeout_seconds'],1800)
        self.assertEqual(BUDGET['safety_factor'],1.12)


if __name__ == '__main__':
    unittest.main()
