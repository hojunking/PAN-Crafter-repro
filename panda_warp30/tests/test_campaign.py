"""Offline queue, budget, original-metric selection and command contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from panda_warp30 import plan, runtime
from panda_warp30.__main__ import main

ROOT = Path(__file__).resolve().parents[2]


class PlanTests(unittest.TestCase):
    def test_fresh_two_donors_six_core_six_optional_no_old_assets(self):
        cfg = plan.default_config('s3')
        plan.validate_config(cfg)
        rows = list(csv.DictReader(io.StringIO(plan.catalog_csv())))
        self.assertEqual(len(rows), 14)
        self.assertEqual([r['case'] for r in rows[:2]], ['S1_U23','S1_BIC'])
        self.assertEqual([r['case'] for r in rows[2:8]], list(plan.CORE))
        self.assertEqual(sum(r['optional']=='True' for r in rows), 6)
        self.assertFalse(cfg['allow_old_donor_reuse'])
        self.assertFalse(cfg['allow_cross_server_donor'])
        self.assertEqual(cfg['budget']['total_seconds'], 30*3600)
        self.assertEqual(cfg['budget']['closeout_seconds'], 1800)
        self.assertEqual(cfg['sheet_selection'],'ALL_OFFICIAL_EVALUATED_CHECKPOINTS')
        self.assertEqual(cfg['groups'][-1]['admission'],'whole_group_or_skip')
        for server in ('s1','s2','s4','s5'):
            with self.assertRaises(ValueError): plan.default_config(server)
        changed = deepcopy(cfg); changed['budget']['total_seconds'] += 1
        with self.assertRaises(ValueError): plan.validate_config(changed)

    def test_same_seed_loss_architecture_horizon_in_every_comparison(self):
        cases = [plan.resolve_case('s3', c) for c in plan.CASES]
        for cfg in cases:
            self.assertEqual(cfg['stage2_spec']['alpha'], .25)
            self.assertEqual(cfg['stage2_spec']['beta'], .05)
            self.assertEqual(cfg['stage2_spec']['gamma'], .5)
            self.assertEqual(cfg['student_architecture'], {'bands':8,'width':112,'depth':[1,2,3]})
            for key in ('stream_seeds','training_spec','augmentation','planned_horizon','evaluation_steps'):
                self.assertEqual(cfg[key], cases[0][key])
            self.assertEqual(cfg['evaluation_steps'][-1], 50000)
            self.assertEqual(cfg['initialization'], 'FRESH')
            self.assertEqual(cfg['start_update'], 0)
        donors = [plan.resolve_case('s3', c) for c in plan.DONORS]
        self.assertEqual(donors[0]['stream_seeds'], donors[1]['stream_seeds'])
        self.assertEqual(donors[0]['baseline_resolved'], donors[1]['baseline_resolved'])
        for cfg in donors:
            self.assertEqual((cfg['lambda_abs'],cfg['lambda_epsilon']), (2.,.003))
        self.assertEqual(plan.resolve_case('s3','U23_MID')['shift_scale'], 1.)
        self.assertEqual(plan.resolve_case('s3','U23_PAN_S050')['shift_scale'], .5)
        with self.assertRaises(ValueError): plan.resolve_case('s3','BIC_PAN','R2')

    def test_hqnr_first_even_if_ergas_threshold_fails(self):
        a={'HQNR':.9591,'ERGAS':2.00,'completed_update':1000}
        b={'HQNR':.9601,'ERGAS':2.20,'completed_update':2000}
        self.assertGreater(plan.rank(b),plan.rank(a))
        b['HQNR']=a['HQNR']; self.assertLess(plan.rank(b),plan.rank(a))
        b['ERGAS']=a['ERGAS']; self.assertLess(plan.rank(b),plan.rank(a))
        ev={'fr':{'metrics':{'HQNR':.9601}},'rr':{'metrics':{'ERGAS':2.2,'SCC':.5}},
            'state_before':{'completed_update':2000}}
        row, reasons=runtime.improvement(ev, {'best_selected':a,'best_hqnr':a})
        self.assertEqual(row['ERGAS'],2.2)
        self.assertIn('BEST_HQNR_THEN_ERGAS',reasons)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now=1000.
        for target,value in (('code_pin',{}),('evaluator_identity',{})):
            p=patch.object(runtime,target,return_value=value); p.start(); self.addCleanup(p.stop)

    def campaign(self):
        return runtime.Campaign('s3',output=self.tmp.name,device='cpu',clock=lambda:self.now)

    def test_deadline_and_elapsed_survive_restart_and_no_budget_reset(self):
        c=self.campaign(); c.persist()
        deadline=c.deadline_unix
        self.now += 20*3600
        restored=self.campaign()
        self.assertEqual(restored.deadline_unix,deadline)
        self.assertGreaterEqual(restored.elapsed_seconds(),20*3600)
        restored.persist()
        # Backward wall-clock movement cannot recover already consumed time.
        self.now -= 3600
        second=self.campaign()
        self.assertGreaterEqual(second.elapsed_seconds(),20*3600)
        self.now = 1000+29.5*3600
        with self.assertRaisesRegex(runtime.BudgetExpired,'DEADLINE'): second.check()
        second.closeout=True
        second.check()
        self.now=deadline
        with self.assertRaises(runtime.BudgetExpired): second.check()

    def test_invalid_elapsed_fails_closed(self):
        c=self.campaign(); c.persist()
        for value in (-1, float('nan'), True, 'zero'):
            state=deepcopy(c.state);state['elapsed_seconds']=value
            c.state_path.write_text(json.dumps(state))
            with self.assertRaises(ValueError): self.campaign()

    def test_optional_whole_group_admission_and_completed_runs_excluded(self):
        c=self.campaign();group=plan.groups()[-1]
        # Three hours remaining cannot admit four one-hour experiments.
        self.now += 26.5*3600
        with patch.object(c,'estimate_seconds',return_value=3600):
            self.assertFalse(c.admit_group(group))
            self.assertEqual(c.state['groups'][-1]['required_seconds'],4*3600)
            for case,replica in group['runs'][:2]:
                c.state['runs'].append({'run_id':plan.resolve_case('s3',case,replica)['run_id'],
                                        'status':'COMPLETED'})
            self.assertTrue(c.admit_group(group))
            self.assertEqual(len(c.state['groups'][-1]['pending']),2)

    def test_measured_update_evaluation_save_time_and_reserve_all_count(self):
        c=self.campaign();cfg=plan.resolve_case('s3','U23_PAN')
        n=len(cfg['evaluation_steps'])
        c.state['timing'][c.timing_key(cfg)]={
            'updates_observed':100,'update_p90':.1,'eval_seconds':20,'save_seconds':2}
        self.assertAlmostEqual(c.estimate_seconds(cfg),(50000*.1+n*22+600)*1.2)

    def test_controller_prepares_donors_then_core_and_reports_optional_skips(self):
        c=self.campaign();calls=[]
        with patch.object(c,'admission',side_effect=lambda _:calls.append('admission')), \
             patch.object(c,'prepare_donors',side_effect=lambda:calls.append('donors')), \
             patch.object(c,'estimate_seconds',return_value=1e9), \
             patch.object(c,'run_case',side_effect=lambda cfg:calls.append(cfg['case'])):
            state=c.run()
        self.assertEqual(calls,['admission','donors',*plan.CORE])
        self.assertEqual(state['status'],'FINISHED_CORE_OPTIONALS_BUDGET_SKIPPED')
        self.assertEqual(len(state['groups']),3)

    def test_expiration_never_claims_completed_queue(self):
        c=self.campaign()
        with patch.object(c,'admission'), \
             patch.object(c,'prepare_donors',side_effect=runtime.BudgetExpired('test')), \
             patch.object(c,'run_case') as run:
            state=c.run()
        run.assert_not_called()
        self.assertEqual(state['status'],'STOPPED_SAFE')
        self.assertEqual(state['reason'],'test')


class CLITests(unittest.TestCase):
    def test_direct_run_also_requires_local_server_marker_before_gpu(self):
        with tempfile.TemporaryDirectory() as root, patch.object(plan,'ROOT',Path(root)), \
             patch('directshift48.__main__.require_idle_gpu') as gpu:
            with self.assertRaisesRegex(ValueError,'server marker'):
                main(['run','--server','s3','--device','cpu'])
            gpu.assert_not_called()

    def test_catalog_and_wv3_mounts_without_launch(self):
        out=io.StringIO()
        with redirect_stdout(out): self.assertEqual(main(['catalog','--server','s3','--csv']),0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(out.getvalue())))),14)
        with patch('edgeres48.docker_mounts.targets',return_value=['/data/wv3.h5']) as target:
            with redirect_stdout(io.StringIO()): self.assertEqual(main(['mounts']),0)
        target.assert_called_once_with(ROOT,'s1')

    def test_shell_syntax_no_destructive_commands_and_wrong_server_refused(self):
        for name in ('run_panda_warp30.sh','run_panda_warp30_docker.sh'):
            path=ROOT/'scripts'/name
            subprocess.run(['bash','-n',str(path)],check=True)
            source=path.read_text()
            for forbidden in ('docker stop','docker rm','pkill','killall','--replace'):
                self.assertNotIn(forbidden,source)
            result=subprocess.run(['bash',str(path),'s1'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2)
            if (ROOT/'gspread/server.txt').read_text().strip()!='s3':
                result=subprocess.run(['bash',str(path),'s3'],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('server marker',result.stderr)


if __name__=='__main__': unittest.main()
