import copy
import unittest
from unittest.mock import patch
from panda_rb import plan


class PlanTests(unittest.TestCase):
    def test_supplied_json_csv_schedule_agree(self):
        result = plan.validate_plan()
        self.assertEqual(result['training_runs'], 24)
        self.assertEqual(result['stress_curves'], 48)

    def test_s2_s4_rejected(self):
        for server in ('s2', 's4', 'S1', ''):
            with self.assertRaises(ValueError): plan.schedule(server)

    def test_exact_case_order(self):
        expected = {'s1': ['QFULL','QMEAN','QSHUF','QESUR','QESUR','QSHUF','QMEAN','QFULL'],
                    's3': ['QMEAN','QFULL','QESUR','QSHUF','QSHUF','QESUR','QFULL','QMEAN'],
                    's5': ['QSHUF','QMEAN','QESUR','QFULL','QFULL','QESUR','QMEAN','QSHUF']}
        for server, cases in expected.items():
            self.assertEqual([r['case_id'] for r in plan.training_runs(server)], cases)
            self.assertEqual(plan.schedule(server)[-1]['action'], 'STOP_FOR_REVIEW')

    def test_config_no_old_campaign_deadline(self):
        row = plan.training_runs()[0]
        cfg = plan.build_config(row['run_id'])
        self.assertNotIn('fh20r1', cfg)
        self.assertNotIn('budget_path', cfg['panda_rb'])
        self.assertEqual(cfg['model_args']['hidden_size'], 104)
        self.assertEqual(cfg['model_args']['depth'], [1,2,2])
        self.assertEqual(cfg['mixed_precision'], 'no')
        self.assertEqual(cfg['batch_size'], 48)
        self.assertEqual(cfg['panda_rb']['teacher_alias'], 'F1')
        self.assertEqual(cfg['panda_rb']['primary_selection'], 'EXACT_50000')

    def test_paired_configs_only_registered_variation(self):
        cfgs = [plan.build_config(r['run_id']) for r in plan.training_runs('s1')[:4]]
        common = []
        for cfg in cfgs:
            cfg = copy.deepcopy(cfg); cfg.pop('work_dir')
            for key in ('run_id','case_id'): cfg['panda_rb'].pop(key)
            common.append(cfg)
        self.assertTrue(all(x == common[0] for x in common))

    def test_no_alias_unknown_run(self):
        with self.assertRaises(ValueError): plan.case_for(plan.TEACHER_RUN)

    def test_actual_registry_null_not_fabricated(self):
        self.assertIsNone(plan.registry()['method_anchor']['teacher_checkpoint_sha256'])

    def test_fixed_grid(self):
        points = plan.shifts()
        self.assertEqual(len(points),49)
        self.assertEqual(sum(p['radius_hr']==0 for p in points),1)
        self.assertEqual(len(plan.VAL_GRID),50)
        self.assertNotIn(10000,plan.VAL_GRID)

if __name__ == '__main__': unittest.main()
