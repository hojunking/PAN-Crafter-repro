import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from directshift48.reporting import INDEX
from panda_wv3_main_rgb import plan as main
from panda_wv3_hyper import plan as previous
from panda_wv3_main_sensitivity import plan, reuse, enqueue
from panda_wv3_main_sensitivity import training, runtime, assets, publication, evaluation


class CampaignTests(unittest.TestCase):
    def test_paired_ofat_matches_main_except_the_declared_coefficients(self):
        queue = plan.queue('s1')
        self.assertEqual(len(queue), 27)
        self.assertEqual(len({c['run_id'] for c in queue}), 27)
        self.assertEqual([c['repeat'] for c in queue], [1]*9+[2]*9+[3]*9)
        self.assertEqual(sum(bool(c['existing_control_case']) for c in queue), 3)
        for repeat in (1, 2, 3):
            block = [c for c in queue if c['repeat'] == repeat]
            self.assertEqual({(c['alpha'], c['gamma']) for c in block},
                {(a, .25) for a in (0, .125, .25, .5, 1)} |
                {(.25, g) for g in (0, .125, .25, .5, 1)})
            for c in block:
                plan.validate(c)
                old = main.resolve('s1', 'M5_LR110', repeat)
                for key in reuse.COMMON_SCIENCE + reuse.STAGE2_SCIENCE:
                    if key not in ('alpha', 'gamma'):
                        self.assertEqual(c[key], old[key], key)
                self.assertFalse(c['diagnostic_only'])
                self.assertEqual(c['lr'], .00011)
                self.assertEqual(c['primary_step'], 29000)
                self.assertEqual(c['horizon'], 30000)
        with self.assertRaises(ValueError): plan.queue('s3')
        self.assertNotIn('H2_GT', [c['case_id'] for c in queue])

    def test_inherited_engine_binds_new_plan_without_changing_old_campaign(self):
        for module in (training, runtime, assets, publication):
            self.assertIs(module.plan, plan)
            self.assertTrue(module.__name__.startswith('panda_wv3_main_sensitivity.'))
        self.assertEqual(previous.resolve('s1', 'H2_REF', 1)['lr'], .0001)
        self.assertNotEqual(previous.CAMPAIGN_ID, plan.CAMPAIGN_ID)
        self.assertEqual(runtime.initial_state('s1')['campaign_id'], plan.CAMPAIGN_ID)

    def test_reuse_accepts_only_actual_main_controls(self):
        for c in plan.queue('s1'):
            source = reuse.source_spec(c)
            if c['case_id'] == 'H2_REF':
                self.assertEqual(source[2]['case_id'], 'M5_LR110')
                reuse._compare_science(c, source[2])
                with self.assertRaises(ValueError):
                    reuse._compare_science(c, main.resolve('s1', 'M0_OURS', c['repeat']))
            else:
                self.assertIsNone(source)

    def test_sheet_rows_are_distinct_and_describe_lr_and_reporting(self):
        cfg = plan.resolve('s1', 'H2_A0', 1)
        row, note = publication.make_row(cfg, 'QUEUED')
        old_row, _ = publication.make_row(previous.resolve('s1', 'H2_A0', 1), 'QUEUED')
        self.assertNotEqual(row[INDEX['Result_ID']], old_row[INDEX['Result_ID']])
        self.assertEqual(row[INDEX['Campaign_ID']], plan.CAMPAIGN_ID)
        self.assertIn('MAIN-SENS-LR110', row[INDEX['Experiment']])
        self.assertEqual(row[INDEX['U_peak_lr']], .00011)
        n = json.loads(note)
        self.assertEqual(n['reporting']['decimal_places'], 4)
        self.assertEqual(n['configuration']['reporting_policy']['normal_low_results'], 'INCLUDE')

    def test_launcher_isolates_output_and_does_not_mutate_historical_runs(self):
        image = {'Id':'sha256:example','Config':{'Env':['PATH=/opt/conda/bin']}}
        result = SimpleNamespace(stdout=json.dumps([image]), returncode=0)
        with patch.object(enqueue.subprocess, 'run', return_value=result), \
                patch('edgeres48.docker_mounts.targets', return_value=[Path('/data/WV3.h5')]), \
                patch.object(enqueue, 'verify', return_value={'source_sha256':'test'}):
            cmd = enqueue.command('s1', 'test', detached=True)
        mounts = [cmd[i+1] for i,v in enumerate(cmd) if v == '--mount']
        self.assertEqual([m for m in mounts if not m.endswith(',readonly')],
            [f'type=bind,src={plan.ROOT}/work_dir/panda_wv3_main_sensitivity,dst=/workspace/work_dir/panda_wv3_main_sensitivity'])
        self.assertIn('panda-wv3-main-sensitivity-s1-v1', cmd)


if __name__ == '__main__':
    unittest.main()
