#!/usr/bin/env python
"""CPU-only FH12 registry, admission and mocked lifecycle tests; no live queues/GPU."""
import csv
import io
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fh12.plan import (CASES,SERVERS,GRID_STEPS,SOURCE_PLAN,cases_for,teacher_for,
                       registry_rows,build_config,registry_sha256)
from tools.gen_fh12_configs import artifacts,generate
from tools import fh12_runner as R


class RegistryTests(unittest.TestCase):
    def test_cardinality(self):
        self.assertEqual(len(registry_rows()),31)
        self.assertEqual(len(CASES),21)
        self.assertEqual(sum(c.role=='T' for c in CASES),5)
        self.assertEqual(sum(c.role=='S' and c.tier=='CORE' for c in CASES),11)
        self.assertEqual(sum(c.tier=='RESERVE_D' for c in CASES),3)
        self.assertEqual(sum(c.tier=='RESERVE_W' for c in CASES),2)
        self.assertEqual(len({c.run_id for c in CASES}),21)

    def test_exact_order_and_seed(self):
        self.assertEqual([c.input_layout for c in cases_for('s4')],['PLH','PLH','P0','PL','PH','PLH','PLH'])
        self.assertEqual([c.order for c in cases_for('s4')],[10,30,40,50,60,70,80])
        for c in CASES:
            self.assertEqual(c.teacher_seed,71002 if c.server_id=='s5' else 71001)
            if c.role=='S':
                self.assertEqual(c.student_seed,72002 if c.server_id=='s5' else 72001)
                self.assertIn(f'_TS{c.teacher_seed}_SS{c.student_seed}_',c.run_id)

    def test_teacher_is_local_exact50k(self):
        for c in CASES:
            cfg=build_config(c)
            self.assertEqual(cfg['fh12']['teacher_run_id'],teacher_for(c.server_id).run_id)
            self.assertNotIn('T0',json.dumps(cfg))
            if c.role=='T':
                self.assertIsNone(cfg['fh12']['teacher_checkpoint'])
                self.assertIsNone(cfg['fh12']['reference_manifest'])
            else:
                self.assertTrue(cfg['fh12']['teacher_checkpoint'].endswith('/candidates/50000/model.safetensors'))

    def test_portability_and_precision(self):
        for c in CASES:
            cfg=build_config(c)
            self.assertNotIn('/home/',json.dumps(cfg))
            self.assertEqual(cfg['mixed_precision'],'no')
            self.assertEqual(cfg['num_iter'],50000)
            self.assertEqual(cfg['batch_size'],48)
            self.assertEqual(cfg['fh12']['candidate_grid'],list(GRID_STEPS))
            self.assertEqual(cfg['model_args']['attn_locations'],[])
        self.assertEqual(len(GRID_STEPS),50)
        self.assertEqual(GRID_STEPS[-2:],(49490,50000))

    def test_csv_and_generated_files(self):
        files=artifacts()
        self.assertEqual(len([p for p in files if p.endswith('.yaml')]),21)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(files['config/queues/FH12_case_registry.csv'])))),31)
        self.assertEqual(generate(ROOT,check=True),[])

    def test_initial_reservations(self):
        expected={'s1':9.6,'s2':10.2,'s3':6.5,'s4':9.3,'s5':8.7}
        for server in SERVERS:
            actual=sum(c.reservation_hours for c in cases_for(server) if c.tier=='CORE')+1.25+.75
            self.assertAlmostEqual(actual,expected[server])

    def test_reserve_formula_no_double_count(self):
        gate=R.admission(10,2,3,.2,reserve=True)
        self.assertAlmostEqual(gate['required_hours'],.75+1.15*(2+3+.2))
        self.assertFalse(R.admission(6,2,3,.2,reserve=True)['admitted'])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fh12-runner-test-')
        self.root=Path(self.temp.name)
        generate(self.root)
        spec=self.root/SOURCE_PLAN; spec.parent.mkdir(parents=True); spec.write_text('test spec')
        self.proc=patch.object(R,'process_inventory',return_value=[]); self.proc.start()
        self.clock=patch.object(R.time,'time',return_value=100000.); self.clock.start()
        self.server='s4'; self.camp=R.campaign_dir(self.root,self.server)

    def tearDown(self):
        self.clock.stop(); self.proc.stop(); self.temp.cleanup()

    def prepare(self):
        return R.prepare(self.root,self.server,device='cpu',upload=False)

    def test_dry_run_has_no_clock_or_hold(self):
        value=R.prepare(self.root,self.server,dry_run=True)
        self.assertEqual(value['status'],'DRY_RUN')
        self.assertFalse((self.root/'work_dir').exists())

    def test_restart_cannot_reset_clock(self):
        first=self.prepare()['window']
        with patch.object(R.time,'time',return_value=110000.):
            second=self.prepare()['window']
        self.assertEqual(first,second)
        self.assertEqual(R.timestamp(first['deadline_utc'])-R.timestamp(first['window_start_utc']),43200)

    def test_changed_config_rejected(self):
        self.prepare()
        path=self.root/f'config/{teacher_for(self.server).run_id}.yaml'
        path.write_text(path.read_text()+'# changed\n')
        with self.assertRaises(ValueError): self.prepare()

    def test_foreign_hold_preserved(self):
        hold=self.root/'work_dir/_eval_phase/hold.json'; R.write_json(hold,{'protocol_id':'OTHER'})
        before=hold.read_bytes()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(hold.read_bytes(),before)

    def test_unknown_legacy_controller_refused(self):
        with patch.object(R,'process_inventory',return_value=[dict(script='_run_cases.sh',controller=True,pid=999)]):
            with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.camp.exists())

    def test_known_m20_can_drain_without_kill(self):
        R.write_json(self.root/'work_dir/_qrc24_mix20h/plan_manifest.json',{})
        with patch.object(R,'process_inventory',return_value=[dict(script='mix20h_runner.py',controller=True,pid=999)]):
            self.prepare()
        self.assertTrue(R.hold_owned(R.read_json(self.root/'work_dir/_eval_phase/hold.json'),self.server))
        self.assertTrue(R.wait_resources(self.root,self.server,R.load_window(self.root,self.server),{},False))

    def test_busy_worker_waits_no_spawn(self):
        self.prepare()
        with patch.object(R,'process_inventory',return_value=[dict(script='main.py',controller=False,pid=123)]):
            self.assertFalse(R.wait_resources(self.root,self.server,R.load_window(self.root,self.server),{},False))
        self.assertEqual(R.read_json(self.camp/'status.json')['status'],'DRAINING_EXISTING')

    def test_drain_over_budget_does_not_start_preflight(self):
        self.prepare()
        with patch.object(R.time,'time',return_value=100000.+4*3600),patch.object(R,'command') as cmd:
            R.run_campaign(self.root,self.server,wait=False)
        cmd.assert_not_called()
        self.assertEqual(R.read_json(self.camp/'status.json')['status'],'DEFERRED_BUDGET')

    def test_training_pause_retained_not50k(self):
        self.prepare()
        with patch.object(R,'command',side_effect=[0,75]) as cmd:
            R.run_campaign(self.root,self.server,wait=False)
        self.assertEqual(cmd.call_count,2)
        state=R.read_json(self.camp/'status.json')
        self.assertEqual(state['status'],'PAUSED_DEADLINE')
        self.assertNotIn('training_complete',state['runs'][teacher_for(self.server).run_id])

    def test_fresh_retry_without_state_forbidden(self):
        self.prepare(); teacher=teacher_for(self.server)
        R.write_json(self.camp/'status.json',dict(status='RUNNING',preflight_complete=True,
            runs={teacher.run_id:dict(status='TRAINING',role='T',tier='CORE',training_started=True)}))
        with patch.object(R,'command') as cmd:
            R.run_campaign(self.root,self.server,wait=False)
        cmd.assert_not_called()
        state=R.read_json(self.camp/'status.json')
        self.assertEqual(state['status'],'FAILED_TEACHER')
        self.assertIn('never_fresh_retry',state['runs'][teacher.run_id]['reason'])

    def test_exact50k_bytes_verified(self):
        case=teacher_for(self.server); folder=self.root/'work_dir'/case.run_id/'candidates/50000'
        folder.mkdir(parents=True); (folder/'model.safetensors').write_bytes(b'checkpoint')
        R.write_json(folder/'identity.json',dict(step=50000,model_sha256=R.sha(folder/'model.safetensors')))
        self.assertTrue(R.train_complete(self.root,case))
        (folder/'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError): R.train_complete(self.root,case)

    def test_server_conflict(self):
        path=self.root/'gspread/server.txt';path.parent.mkdir();path.write_text('s4\n')
        self.assertEqual(R.detect_server(self.root),'s4')
        with self.assertRaises(ValueError): R.detect_server(self.root,'s1')

    def publish_exact(self,case,grid_complete=True):
        folder=self.root/'work_dir'/case.run_id/'candidates/50000'
        folder.mkdir(parents=True,exist_ok=True); (folder/'model.safetensors').write_bytes(b'checkpoint')
        R.write_json(folder/'identity.json',dict(update=50000,model_sha256=R.sha(folder/'model.safetensors')))
        R.write_json(folder.parent.parent/'official/raw_grid.json',dict(complete=grid_complete))
        last=folder.parent.parent/'last'; last.mkdir(exist_ok=True)
        (last/'training_state.pt').write_bytes(b'full-state-placeholder')

    def test_exact_saved_before_metric_commit_finishes_by_resume(self):
        self.prepare(); case=teacher_for(self.server); self.publish_exact(case,False)
        R.write_json(self.camp/'status.json',dict(status='RUNNING',preflight_complete=True,runs={}))
        with patch.object(R,'command',return_value=75) as cmd:
            R.run_campaign(self.root,self.server,wait=False)
        args=cmd.call_args.args[1]
        self.assertEqual(args[0],'tools/fh12_train.py')
        self.assertEqual(args[-1],'--resume')

    def test_complete_training_pending_report_does_not_train_again(self):
        self.prepare(); case=teacher_for(self.server); self.publish_exact(case,True)
        R.write_json(self.camp/'status.json',dict(status='EVALUATION_PENDING',preflight_complete=True,
                     runs={case.run_id:dict(role='T',tier='CORE',status='EVALUATION_PENDING',training_started=True)}))
        with patch.object(R,'command',return_value=2) as cmd:
            R.run_campaign(self.root,self.server,wait=False)
        self.assertEqual(cmd.call_count,1)
        self.assertEqual(cmd.call_args.args[1][0],'tools/fh12_postrun.py')
        self.assertEqual(R.read_json(self.camp/'status.json')['status'],'EVALUATION_PENDING')

    def test_full_pipeline_local_dependency_and_upload_failure(self):
        self.prepare(); state=dict(status='RUNNING',preflight_complete=True,runs={})
        R.write_json(self.camp/'status.json',state)
        window=R.read_json(self.camp/'window.json'); window['upload_enabled']=True
        R.write_json(self.camp/'window.json',window)
        calls=[]
        def fake_command(root,args,log):
            calls.append(args)
            if args[0]=='tools/fh12_train.py':
                case=next(c for c in CASES if c.run_id==Path(args[args.index('--config')+1]).stem)
                self.publish_exact(case)
                return 0
            if args[0]=='tools/fh12_postrun.py':
                R.write_json(root/'work_dir'/args[1]/'official/postrun_status.json',
                    dict(official_complete=True,status='EVAL_COMPLETE_UPLOAD_PENDING',sheet_uploaded=False))
                return 0
            if args[0]=='tools/fh12_calibrate.py':
                teacher=teacher_for(self.server)
                self.assertEqual(args[args.index('--teacher-run')+1],teacher.run_id)
                R.write_json(self.camp/'references'/teacher.run_id/'reference_manifest.json',
                             dict(teacher_run_id=teacher.run_id))
                return 0
            raise AssertionError(args)
        with patch.object(R,'command',side_effect=fake_command):
            R.run_campaign(self.root,self.server,wait=False)
        calibration=[a for a in calls if a[0]=='tools/fh12_calibrate.py']
        self.assertEqual(len(calibration),1)
        train=[a for a in calls if a[0]=='tools/fh12_train.py']
        self.assertEqual([Path(a[a.index('--config')+1]).stem for a in train],[c.run_id for c in cases_for(self.server)])
        state=R.read_json(self.camp/'status.json')
        self.assertEqual(state['status'],'DONE')
        self.assertTrue(all(r['upload_pending'] for r in state['runs'].values()))

    def test_runner_does_not_import_gpu_framework_or_calibration(self):
        command=[sys.executable,'-c',
                 'import sys; import tools.fh12_runner; assert "torch" not in sys.modules; '
                 'assert "fh12.calibration" not in sys.modules']
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
        source=(ROOT/'tools/fh12_runner.py').read_text()
        self.assertNotIn('from fh12.calibration import',source)

    def test_closed_window_does_not_relaunch_training(self):
        self.prepare()
        R.write_json(self.camp/'status.json',dict(status='DONE',runs={}))
        with patch.object(R,'command') as cmd:
            self.assertEqual(R.run_campaign(self.root,self.server,wait=False),0)
        cmd.assert_not_called()


class ShellGuardTests(unittest.TestCase):
    def exercise(self,filename,activated=False,incomplete=False):
        """Execute only the real new guard block, never any legacy shell body."""
        source=(ROOT/'tools'/filename).read_text()
        begin=source.index('# Explicitly activated FH12' if filename=='_watchdog.sh' else '# FH12 is opt-in:')
        end=source.index('if [ -f "$REPO/work_dir/_qrc24_mix20h/plan_manifest.json" ]; then',begin)
        block=source[begin:end]
        # Foreground the watchdog's detached child only in this isolated fixture
        # so the fake interpreter's call is deterministic without sleeping.
        block=block.replace('setsid nohup ','').replace('9>&- &','9>&-')
        with tempfile.TemporaryDirectory(prefix='fh12-shell-test-') as temp:
            root=Path(temp); (root/'tools').mkdir(); (root/'work_dir').mkdir()
            (root/'tools/fh12_runner.py').write_text(
                'import json, pathlib, sys\npathlib.Path("invoked.json").write_text(json.dumps(sys.argv[1:]))\n')
            if activated or incomplete:
                (root/'work_dir/_fh12').mkdir()
                (root/'work_dir/_fh12/local_server.txt').write_text('s3\n')
            if activated:
                camp=root/'work_dir/_fh12/s3';camp.mkdir();(camp/'window.json').write_text('{"status":"DONE"}')
            env=dict(os.environ,REPO=str(root),LOG=str(root/'work_dir/test.log'),PYTHON=sys.executable)
            result=subprocess.run(['bash','-c','exec 9>"$REPO/work_dir/test.lock"\n'+block+'\nprintf LEGACY\n'],
                                  cwd=root,env=env,capture_output=True,text=True,check=True)
            call=json.loads((root/'invoked.json').read_text()) if (root/'invoked.json').exists() else None
            return result.stdout,call

    def test_pull_without_registration_never_launches(self):
        for filename in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(filename=filename):
                stdout,call=self.exercise(filename)
                self.assertEqual(stdout,'LEGACY');self.assertIsNone(call)

    def test_registered_even_closed_window_never_falls_through(self):
        for filename in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(filename=filename):
                stdout,call=self.exercise(filename,activated=True)
                self.assertNotIn('LEGACY',stdout)
                self.assertEqual(call,['run','--server','s3'])

    def test_partial_registration_fails_closed(self):
        for filename in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(filename=filename):
                stdout,call=self.exercise(filename,incomplete=True)
                self.assertNotIn('LEGACY',stdout);self.assertIsNone(call)


if __name__=='__main__':
    unittest.main(verbosity=2)
