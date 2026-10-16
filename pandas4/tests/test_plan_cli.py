"""Registry and read-only entrypoint safety; never invoke a training launcher."""
import csv
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from pandas4 import plan
from pandas4.__main__ import main


class PlanTests(unittest.TestCase):
    def test_exact_trio_and_paired_anchor_order(self):
        self.assertEqual(plan.donor_queue(), [('D_A1','R1'),('D_A1','R2'),('D_A3','R1')])
        groups = plan.stage2_groups()
        self.assertEqual([x['name'] for x in groups],
            ['G','Q','E','ES','Q_ALOW','Q_BHIGH','Q_GHIGH','Q_FULL'])
        self.assertEqual(sum(len(x['runs']) for x in groups if x['anchor']), 4)
        rows = list(csv.DictReader(StringIO(plan.catalog_csv())))
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({(x['case'],x['replica']) for x in rows}), 13)
        self.assertTrue(all(x['horizon']=='50000' for x in rows))
        for case, replica in plan.donor_queue():
            cfg = plan.resolve_case('s4', case, replica)
            self.assertEqual(json.loads(json.dumps(cfg)), cfg)
        plan.validate_config(json.loads(json.dumps(plan.default_config())))

    def test_nonregistered_configuration_rejected(self):
        for server,case,replica in [('s1','D_A1','R1'),('s3','Q','R1'),
                                    ('s4','D_A3','R2'),('s4','ES','R2'),('s4','Q','R3')]:
            with self.assertRaises(ValueError):
                plan.resolve_case(server, case, replica)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                plan.resolve_case('s4','G',attempt=value)
        cfg = plan.default_config()
        cfg['budget']['total_seconds'] += 1
        with self.assertRaises(ValueError):
            plan.validate_config(cfg)

    def test_gate_then_ergas_never_scc_or_sheet_threshold(self):
        def row(h,e,step=1000,scc=.1):
            return dict(HQNR=h,ERGAS=e,completed_update=step,SCC=scc)
        gate = row(.959,3.,scc=.1)
        self.assertIs(plan.choose([row(.958999,1.,scc=1.),gate]),gate)
        lower_e = row(.9591,2.1)
        self.assertIs(plan.choose([row(.970,2.2),lower_e]),lower_e)
        better_h = row(.958,2.4)
        self.assertIs(plan.choose([row(.957,2.),better_h]),better_h)
        self.assertEqual(plan.default_config()['sheet_selection']['HQNR_min'],.960)

    def test_one_factor_sensitivities_and_no_aux_student_inputs(self):
        q = plan.resolve_case('s4','Q')['stage2_spec']
        for case,field in [('Q_ALOW','alpha'),('Q_BHIGH','beta'),('Q_GHIGH','gamma')]:
            other = plan.resolve_case('s4',case)['stage2_spec']
            self.assertEqual([k for k in q if q[k]!=other[k]], [field])
        for group in plan.stage2_groups():
            for case, replica in group['runs']:
                cfg = plan.resolve_case('s4',case,replica)
                from panda_stage2_condq.spec import Stage2Spec
                spec = Stage2Spec(**cfg['stage2_spec']).manifest()
                self.assertFalse(spec['stage2_epsilon'])
                self.assertFalse(spec['separate_spatial_loss'])
                self.assertEqual(cfg['baseline_resolved']['teacher_input_layout'],'P0')


class CLITests(unittest.TestCase):
    def test_catalog_has_no_run_or_gpu_action(self):
        with patch('pandas4.runtime.Campaign') as campaign, patch('builtins.print') as output:
            self.assertEqual(main(['catalog','--csv']),0)
        campaign.assert_not_called()
        self.assertIn('D_A3',output.call_args.args[0])

    def test_mounts_and_help_without_training_imports(self):
        for argv in (['mounts'],['--help']):
            code = '''
import importlib.abc, sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','h5py','google','panda40','panda_stage2_condq'):
            raise AssertionError('Host entrypoint loaded training dependency: '+fullname)
sys.meta_path.insert(0,RejectTraining())
from pandas4.__main__ import main
raise SystemExit(main(ARGS))
'''.replace('ARGS',repr(argv))
            result = subprocess.run([sys.executable,'-c',code],cwd=plan.ROOT,
                text=True,capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_launchers_syntax_and_no_replacement_workflow(self):
        for name in ('run_pandas4.sh','run_pandas4_docker.sh'):
            path = plan.ROOT/'scripts'/name
            result = subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run(['bash',str(path),'--invalid'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            for forbidden in ('docker stop','docker rm','pkill','killall','run_panda40.sh'):
                self.assertNotIn(forbidden,path.read_text())
        self.assertIn('--server s4', (plan.ROOT/'scripts/run_pandas4_docker.sh').read_text())


if __name__ == '__main__':
    unittest.main()
