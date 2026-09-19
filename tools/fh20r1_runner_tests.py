"""No GPU/network/runtime mutation: FH20R1 ledger, lifecycle and atomic-block tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fh20r1 import ledger as L
from fh20r1.plan import (CAMPAIGN_ID,SOURCE_PLAN,SOURCE_CASES,cases_for,blocks_for,build_config,case_for,
                        active_cases,N2_TEACHER_RUN)
from tools import fh20r1_runner as R


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fh20r1-test-');self.root=Path(self.temp.name)
        self.now=100000.;self.clock=patch('time.time',side_effect=lambda:self.now);self.clock.start()
        self.inventory=patch.object(R,'inventory',return_value=[]);self.inventory.start()
        (self.root/SOURCE_PLAN).parent.mkdir(parents=True);(self.root/SOURCE_PLAN).write_text('fixture plan')
        (self.root/SOURCE_CASES).write_bytes((ROOT/SOURCE_CASES).read_bytes())
        self.server='s4';self.calls=[];self.seconds_per_train=3600.;self.donor_available=True
        self.diag_branch='STANDARD';self.unavailable_run=None

    def tearDown(self):self.clock.stop();self.inventory.stop();self.temp.cleanup()

    def setup_server(self,server=None,upload=True):
        if server:self.server=server
        (self.root/'config').mkdir(exist_ok=True)
        for c in cases_for(self.server):
            (self.root/f'config/{c.run_id}.yaml').write_text(yaml.safe_dump(build_config(c)))
        R.prepare(self.root,self.server,device='cpu',upload=upload)
        self.directory=L.camp(self.root,self.server)

    def timing(self,run,start,end,kind='train',sid='segment',committed=50000):
        wd=self.root/'work_dir'/run;identity=wd/'last/identity.json';state=identity.parent/'training_state.pt'
        state.parent.mkdir(parents=True,exist_ok=True);state.write_bytes(b'fullstate')
        item=dict(update=committed,config_sha256='config',training_state_sha256=L.digest(state))
        L.write(identity,item)
        doc=dict(run_id=run,campaign_id=CAMPAIGN_ID,valid_fullstate=True,config_sha256='config',
            full_state_identity_path='last/identity.json',training_state_sha256=item['training_state_sha256'],
            committed_update=committed,segments=[dict(id=sid,kind=kind,start_utc=L.iso(start),end_utc=L.iso(end),
            seconds=end-start,update_from=0,update_to=committed)])
        L.write(wd/'meta/timing_committed.json',doc)
        return doc

    def fake(self,root,args,log):
        self.calls.append(args);tool=args[0]
        if tool=='tools/fh20r1_reference.py':
            if args[1]=='import':
                L.write(self.directory/'imported_refs'/('F'+self.server[1:])/'bridge_manifest.json',
                        dict(complete=True,parity=dict(status='PASS'),teacher_checkpoint_sha256='origin'))
            elif args[1]=='donor':L.write(self.directory/'donor_manifest.json',dict(donor_available=self.donor_available))
            elif args[1]=='calibrate':
                self.now+=100.;L.write(self.directory/'imported_refs/N2PL/bridge_manifest.json',dict(complete=True))
            else:raise AssertionError(args)
        elif tool=='tools/fh20r1_preflight.py':
            L.write(self.directory/'preflight_report.json',dict(preflight_pass=True,
                registry=dict(source_plan_sha256=L.digest(root/SOURCE_PLAN),source_cases_sha256=L.digest(root/SOURCE_CASES)),
                disk=dict(per_run_estimate_bytes={c.run_id:1024**2 for c in cases_for(self.server)},minimum_safety_bytes=2*1024**3),
                batch_smoke=dict(actual_batch48_smoke_status='NOT_RUN_CPU_DIAGNOSTIC',batch_size=2,device='cpu')))
        elif tool=='tools/fh20r1_diagnose.py':
            self.now+=100.;L.write(self.directory/'diagnostics/report.json',dict(complete=True,branch=self.diag_branch))
        elif tool=='tools/fh20r1_train.py':
            run=Path(args[args.index('--config')+1]).stem
            if run==self.unavailable_run:return 69
            start=self.now;self.now+=self.seconds_per_train
            self.timing(run,start,self.now)
            L.write(root/'work_dir'/run/'meta/training_status.json',dict(training_complete=True,actual_updates=50000))
            L.write(root/'work_dir'/run/'official/raw_grid.json',dict(complete=True))
        elif tool=='tools/fh20r1_postrun.py':
            run=args[1]
            if '--upload-only' not in args:self.now+=10.
            L.write(root/'work_dir'/run/'official/postrun_status.json',dict(official_complete=True,actual_updates=50000,sheet_uploaded='--upload-only' in args))
        else:raise AssertionError(args)
        return 0

    def run_fake(self):
        with patch.object(R,'command',side_effect=self.fake):return R.run_campaign(self.root,self.server,wait=False)

    def trained(self):return [Path(a[a.index('--config')+1]).stem for a in self.calls if a[0]=='tools/fh20r1_train.py']


class LedgerTests(Fixture):
    def test_union_not_parent_plus_child(self):
        self.assertEqual(L.union_seconds([(0,10),(3,6),(9,15),(20,30)]),25)

    def test_dedup_and_wait_exclusion(self):
        self.setup_server();self.now+=100
        kwargs=dict(interval_id='one',kind='train',start_utc=100000,end_utc=100060,evidence='fullstate')
        self.assertTrue(L.record_interval(self.root,self.server,**kwargs));self.assertFalse(L.record_interval(self.root,self.server,**kwargs))
        L.record_interval(self.root,self.server,interval_id='nested',kind='eval',start_utc=100010,end_utc=100050,evidence='candidate')
        L.record_interval(self.root,self.server,interval_id='wait',kind='waiting',start_utc=100060,end_utc=100100,evidence='inventory')
        report=L.report(self.root,self.server)
        self.assertEqual(report['effective_seconds'],60);self.assertEqual(report['waiting_seconds'],40)
        self.assertEqual(report['train_seconds'],60);self.assertEqual(report['eval_seconds'],40)

    def test_forbidden_prior_future_categories(self):
        self.setup_server();self.now+=100
        for kind,start,end in [('train',99999,100001),('eval',100000,100200),('sleep',100000,100010),('upload',100000,100010)]:
            with self.subTest(kind=kind,start=start),self.assertRaises(ValueError):
                L.record_interval(self.root,self.server,interval_id='bad',kind=kind,start_utc=start,end_utc=end,evidence='bad')

    def test_id_collision_rejected(self):
        self.setup_server();self.now+=100
        L.record_interval(self.root,self.server,interval_id='a',kind='train',start_utc=100000,end_utc=100010,evidence='one')
        with self.assertRaises(ValueError):
            L.record_interval(self.root,self.server,interval_id='a',kind='train',start_utc=100000,end_utc=100020,evidence='two')

    def test_committed_timing_replay_counts_once(self):
        self.setup_server();self.now+=100;run=active_cases(self.server)[0].run_id
        self.timing(run,100000,100100)
        self.assertEqual(L.ingest_timing(self.root,self.server,run),1)
        self.assertEqual(L.ingest_timing(self.root,self.server,run),0)
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],100)

    def test_corrupt_fullstate_not_credited(self):
        self.setup_server();self.now+=100;run=active_cases(self.server)[0].run_id
        self.timing(run,100000,100100);(self.root/'work_dir'/run/'last/training_state.pt').write_bytes(b'bad')
        with self.assertRaises(ValueError):L.ingest_timing(self.root,self.server,run)
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],0)

    def test_unsaved_update_not_credited(self):
        self.setup_server();self.now+=100;run=active_cases(self.server)[0].run_id
        doc=self.timing(run,100000,100100);doc['segments'][0]['update_to']=50001
        L.write(self.root/'work_dir'/run/'meta/timing_committed.json',doc)
        with self.assertRaises(ValueError):L.ingest_timing(self.root,self.server,run)


class RunnerTests(Fixture):
    def test_prepare_dryrun_does_not_activate(self):
        (self.root/'config').mkdir()
        for c in cases_for('s4'):(self.root/f'config/{c.run_id}.yaml').write_text(yaml.safe_dump(build_config(c)))
        value=R.prepare(self.root,'s4',dry_run=True)
        self.assertEqual(value['status'],'DRY_RUN');self.assertFalse((self.root/'work_dir').exists())

    def test_budget_restart_no_clock_reset_or_deadline(self):
        self.setup_server();first=L.read(self.directory/'campaign_budget.json');self.now+=500000
        second=R.prepare(self.root,self.server,device='cpu')['budget']
        self.assertEqual(first,second);self.assertIsNone(first['hard_deadline'])

    def test_old_fh12_window_preserved_and_hold_archived(self):
        prior=self.root/'work_dir/_fh12/s4/window.json';L.write(prior,dict(deadline_utc='original12h'))
        oldhold=dict(protocol_id='FH12_FUTURE_ADMISSIONS_HOLD_v1',server='s4')
        L.write(self.root/'work_dir/_eval_phase/hold.json',oldhold);before=prior.read_bytes()
        self.setup_server()
        self.assertEqual(prior.read_bytes(),before)
        self.assertEqual(L.read(self.directory/'takeover_manifest.json')['previous_hold'],oldhold)
        self.assertEqual(L.read(self.root/'work_dir/_eval_phase/hold.json')['protocol_id'],R.HOLD)

    def test_foreign_hold_preserved(self):
        L.write(self.root/'work_dir/_eval_phase/hold.json',dict(protocol_id='UNKNOWN',server='s4'))
        with self.assertRaises(ValueError):self.setup_server()
        self.assertEqual(L.read(self.root/'work_dir/_eval_phase/hold.json')['protocol_id'],'UNKNOWN')

    def test_drain_old_fh12_controller_no_credit(self):
        self.setup_server();state={}
        with patch.object(R,'inventory',return_value=[dict(script='fh12_runner.py',waiting_controller=False,pid=999)]):
            self.assertFalse(R.wait_resources(self.root,self.server,L.read(self.directory/'campaign_budget.json'),state,False))
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],0)

    def test_core_not_cutoff_at20(self):
        self.setup_server();self.seconds_per_train=7200.;self.run_fake()
        self.assertEqual(len(self.trained()),20)
        self.assertEqual(L.read(self.directory/'status.json')['status'],'COMPLETE_20HPLUS')

    def test_reserve_quartet_completed_after20_insideblock(self):
        self.setup_server();self.seconds_per_train=3000.;self.run_fake()
        # Core20 ->16.75h, reserve quartet ->20.09h. All4 finish, nextreserve omitted.
        self.assertEqual(len(self.trained()),24)
        self.assertEqual(L.read(self.directory/'status.json')['status'],'COMPLETE_20HPLUS')

    def test_finite_exhaustion_below20_not_fake_complete(self):
        self.setup_server();self.seconds_per_train=100.;self.run_fake()
        self.assertEqual(len(self.trained()),28)
        self.assertEqual(L.read(self.directory/'status.json')['status'],'CAPACITY_EXHAUSTED_BELOW20')

    def test_s1_fallback_no_a10_immutable(self):
        self.setup_server('s1');self.diag_branch='S1_NO_A_SUPPORT_OR_INCONCLUSIVE';self.seconds_per_train=7200.;self.run_fake()
        trained=[case_for(r) for r in self.trained()]
        self.assertEqual(len(trained),14);self.assertTrue(all(c.profile=='BASE' for c in trained))
        self.assertEqual({c.depth for c in trained},{(1,2,1),(1,2,2)})
        with self.assertRaises(ValueError):R.branch(self.root,'s1','S1_A_SUPPORT','later favorable result')

    def test_s2_blocked_donor_no_teacher_or_n2_student(self):
        self.setup_server('s2');self.donor_available=False;self.seconds_per_train=7200.;self.run_fake()
        self.assertEqual(len(self.trained()),10)
        self.assertTrue(all(case_for(r).teacher_alias=='F2' for r in self.trained()))
        self.assertEqual(L.read(self.directory/'status.json')['status'],'COMPLETE_20HPLUS_WITH_BLOCKED_DONOR')

    def test_s2_ready_teacher_and_calibration_before_students(self):
        self.setup_server('s2');self.seconds_per_train=7200.;self.run_fake()
        self.assertEqual(self.trained()[0],N2_TEACHER_RUN);self.assertEqual(len(self.trained()),11)
        calibration=next(i for i,a in enumerate(self.calls) if a[:2]==['tools/fh20r1_reference.py','calibrate'])
        first_student=next(i for i,a in enumerate(self.calls) if a[0]=='tools/fh20r1_train.py' and '_S_PL_' in a[2])
        self.assertLess(calibration,first_student)

    def test_s2_partial_f2_baseline_reused_after_fallback(self):
        self.setup_server('s2');self.seconds_per_train=7200.
        first=blocks_for('s2')[0];self.unavailable_run=first.primary_order[1]
        self.run_fake();self.assertEqual(self.trained().count(first.primary_order[0]),1)
        self.assertIn(first.alternative_orders['S2_DONOR_OR_REFERENCE_UNAVAILABLE'][1],self.trained())

    def test_complete_restart_no_duplicate_training_or_credit(self):
        self.setup_server();self.seconds_per_train=7200.;self.run_fake();before=L.report(self.root,self.server)
        self.calls=[];self.run_fake();self.assertEqual(self.calls,[])
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],before['effective_seconds'])

    def test_upload_elapsed_excluded(self):
        self.setup_server();c=active_cases(self.server)[0];budget=L.read(self.directory/'campaign_budget.json');budget['upload_enabled']=True
        state=dict(runs={});original=self.fake
        def slow_upload(root,args,log):
            if '--upload-only' in args:self.now+=36000
            return original(root,args,log)
        with patch.object(R,'command',side_effect=slow_upload):R.run_case(self.root,self.server,c,state,budget)
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],3610)

    def test_failed_upload_cli_cannot_reuse_stale_verified_receipt(self):
        self.setup_server();c=active_cases(self.server)[0]
        L.write(self.root/'work_dir'/c.run_id/'official/postrun_status.json',dict(sheet_uploaded=True))
        row=dict(status='DONE',upload_pending=False,upload_status='VERIFIED')
        with patch.object(R,'command',return_value=2):R.upload_case(self.root,c,row)
        self.assertTrue(row['upload_pending']);self.assertEqual(row['upload_status'],'PENDING')
        self.assertIn('exit_code=2',row['upload_error'])

    def test_semantic_reuse_zero_new_credit(self):
        self.setup_server(upload=False);c=active_cases(self.server)[0]
        L.write(self.directory/'reuse_manifest.json',dict(runs={c.run_id:dict(validated=True,official_complete=True,identity_sha256='proven',source_run_id='old-run')}))
        with patch.object(R,'command') as cmd:
            self.assertTrue(R.run_case(self.root,self.server,c,dict(runs={}),L.read(self.directory/'campaign_budget.json')))
        cmd.assert_not_called();self.assertEqual(L.report(self.root,self.server)['effective_seconds'],0)

    def test_reuse_upload_link_only_no_training_credit(self):
        self.setup_server();c=active_cases(self.server)[0]
        L.write(self.directory/'reuse_manifest.json',dict(runs={c.run_id:dict(validated=True,
            official_complete=True,identity_sha256='proven',source_run_id='old-run')}))
        state=dict(runs={})
        with patch.object(R,'command',side_effect=self.fake):
            R.run_case(self.root,self.server,c,state,L.read(self.directory/'campaign_budget.json'))
        self.assertEqual(len(self.calls),1);self.assertIn('--upload-only',self.calls[0])
        self.assertEqual(state['runs'][c.run_id]['status'],'DONE_REUSED')
        self.assertEqual(L.read(self.root/'work_dir'/c.run_id/'meta/reuse_reference.json')['source_run_id'],'old-run')
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],0)

    def test_upload_pending_stops_training_without_completion_certification(self):
        self.setup_server(upload=False);self.seconds_per_train=7200.;self.run_fake()
        self.assertEqual(L.read(self.directory/'status.json')['status'],'WORK_COMPLETE_UPLOAD_PENDING')
        completion=L.read(self.directory/'completion_report.json')
        self.assertTrue(completion['work_complete']);self.assertFalse(completion['all_requested_results_uploaded'])
        self.calls=[];self.run_fake();self.assertEqual(self.calls,[])
        before=L.report(self.root,self.server)['effective_seconds']
        with patch.object(R,'command',side_effect=self.fake):R.retry_upload(self.root,self.server)
        self.assertTrue(all('--upload-only' in a for a in self.calls))
        self.assertEqual(L.read(self.directory/'status.json')['status'],'COMPLETE_20HPLUS')
        self.assertEqual(L.report(self.root,self.server)['effective_seconds'],before)

    def test_readiness_before_first_train_and_cpu_not_gpu_claim(self):
        self.setup_server();self.seconds_per_train=7200.;original=self.fake;observed=[]
        def probe(root,args,log):
            if args[0]=='tools/fh20r1_train.py':
                doc=L.read(self.directory/'readiness_report.json');self.assertTrue(doc)
                observed.append(doc)
            return original(root,args,log)
        with patch.object(R,'command',side_effect=probe):R.run_campaign(self.root,self.server,wait=False)
        self.assertFalse(observed[0]['launch_ready']);self.assertEqual(observed[0]['status'],'CPU_DIAGNOSTIC_ONLY')
        self.assertEqual(observed[0]['core_run_count'],20)
        self.assertEqual(observed[0]['next_run_ids'],list(blocks_for(self.server)[0].primary_order))
        self.assertTrue(observed[0]['checks']['source_hashes_match'])
        self.assertEqual(observed[0]['numeric_method_revision'],'FH12_SYNC_FREQ_NATIVE_TEACHER_v1')
        self.assertEqual(len(observed[0]['queue_sha256']),64)
        self.assertEqual(observed[0]['uploader_schema']['mapping'],'header_name')

    def test_admission_events_include_case_block_and_reuse(self):
        self.setup_server();self.seconds_per_train=7200.;self.run_fake()
        rows=[json.loads(line) for line in (self.directory/'admission_events.jsonl').read_text().splitlines()]
        starts=[r for r in rows if r['event']=='CASE_START'];done=[r for r in rows if r['event']=='CASE_COMPLETE']
        self.assertEqual(len(starts),20);self.assertEqual(len(done),20)
        self.assertEqual(len({r['event_id'] for r in rows}),len(rows))
        self.assertEqual({r['run_id'] for r in starts},{r['run_id'] for r in done})
        self.assertIn('BLOCK_COMPLETE',{r['event'] for r in rows})
        self.assertIn('RESERVE_REMAINDER_SKIPPED',{r['event'] for r in rows})

    def test_s2_ready_reference_restart_does_not_reprobe_donor(self):
        self.setup_server('s2');self.seconds_per_train=7200.;self.run_fake()
        state=L.read(self.directory/'status.json');state['status']='RUNNING';L.write(self.directory/'status.json',state)
        self.calls=[];self.donor_available=False;self.run_fake()
        self.assertFalse(any(a[:2]==['tools/fh20r1_reference.py','donor'] for a in self.calls))
        self.assertEqual(L.read(self.directory/'branch_record.json')['condition'],'S2_N2PL_REFERENCE_READY')

    def test_teacher_pause_does_not_switch_to_fallback(self):
        self.setup_server('s2');original=self.fake
        def paused(root,args,log):
            if args[0]=='tools/fh20r1_train.py':return 75
            return original(root,args,log)
        with patch.object(R,'command',side_effect=paused):self.assertEqual(R.run_campaign(self.root,self.server,wait=False),2)
        self.assertEqual(L.read(self.directory/'status.json')['status'],'PAUSED')
        self.assertEqual(L.read(self.directory/'branch_record.json')['condition'],'PRIMARY')

    def test_teacher_failure_does_not_switch_to_fallback(self):
        self.setup_server('s2');original=self.fake
        def failed(root,args,log):
            if args[0]=='tools/fh20r1_train.py':return 1
            return original(root,args,log)
        with patch.object(R,'command',side_effect=failed):self.assertEqual(R.run_campaign(self.root,self.server,wait=False),2)
        self.assertEqual(L.read(self.directory/'status.json')['status'],'BLOCK_FAILED')
        self.assertEqual(L.read(self.directory/'branch_record.json')['condition'],'PRIMARY')

    def test_cuda_readiness_requires_real_batch48_and_parity(self):
        self.setup_server();self.fake(self.root,['tools/fh20r1_reference.py','import'],None)
        self.fake(self.root,['tools/fh20r1_preflight.py'],None)
        budget=dict(L.read(self.directory/'campaign_budget.json'),device='cuda')
        state=dict(runs={},blocks={});chosen=dict(condition='STANDARD')
        with patch.object(R,'gpu_processes',return_value=[]):
            with self.assertRaises(RuntimeError):R.readiness(self.root,self.server,state,budget,chosen)
            preflight=L.read(self.directory/'preflight_report.json')
            preflight['batch_smoke']=dict(actual_batch48_smoke_status='PASS',batch_size=48,device='cuda')
            L.write(self.directory/'preflight_report.json',preflight)
            ready=R.readiness(self.root,self.server,state,budget,chosen)
            self.assertTrue(ready['launch_ready'])
            path=self.directory/'imported_refs/F4/bridge_manifest.json'
            bridge=L.read(path);bridge['parity']['status']='FAILED';L.write(path,bridge)
            with self.assertRaises(RuntimeError):R.readiness(self.root,self.server,state,budget,chosen)

    def test_cli_rejects_real_cpu_campaign_activation(self):
        with patch.object(R,'ROOT',self.root),patch.object(R,'detect_server',return_value='s4'), \
             patch.object(sys,'argv',['fh20r1_runner.py','start','--device','cpu']),patch.object(R,'prepare') as prepare:
            with self.assertRaises(ValueError):R.main()
        prepare.assert_not_called();self.assertFalse((self.root/'work_dir').exists())

    def test_cli_rejects_cpu_budget_execution(self):
        self.setup_server()
        with patch.object(R,'ROOT',self.root),patch.object(R,'detect_server',return_value=self.server), \
             patch.object(sys,'argv',['fh20r1_runner.py','run']),patch.object(R,'run_campaign') as run:
            with self.assertRaises(ValueError):R.main()
        run.assert_not_called()

    def test_calibration_infrastructure_failure_is_not_absence_fallback(self):
        self.setup_server('s2');original=self.fake
        def failed(root,args,log):
            if args[:2]==['tools/fh20r1_reference.py','calibrate']:return 1
            return original(root,args,log)
        with patch.object(R,'command',side_effect=failed),self.assertRaises(RuntimeError):
            R.run_campaign(self.root,self.server,wait=False)
        self.assertEqual(self.trained(),[N2_TEACHER_RUN])
        self.assertEqual(L.read(self.directory/'branch_record.json')['condition'],'PRIMARY')

    def test_disk_shortage_stops_before_atomic_block_without_deletion(self):
        self.setup_server();sentinel=self.root/'work_dir/keep-original';sentinel.write_bytes(b'keep')
        with patch.object(R.shutil,'disk_usage',return_value=type('Usage',(),dict(free=1))()):
            self.assertEqual(self.run_fake(),3)
        self.assertFalse(self.trained());self.assertEqual(sentinel.read_bytes(),b'keep')
        self.assertEqual(L.read(self.directory/'status.json')['status'],'DISK_BLOCKED')

    def test_failed_training_no_automatic_retry(self):
        self.setup_server();c=active_cases(self.server)[0];state=dict(runs={c.run_id:dict(status='FAILED')})
        with patch.object(R,'command') as cmd:self.assertFalse(R.run_case(self.root,self.server,c,state,L.read(self.directory/'campaign_budget.json')))
        cmd.assert_not_called()

    def test_module_import_no_gpu_framework(self):
        subprocess.run([sys.executable,'-c','import sys; import tools.fh20r1_runner; assert "torch" not in sys.modules'],cwd=ROOT,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)


class ShellGuardTests(unittest.TestCase):
    def exercise(self,name,activated=False,incomplete=False):
        source=(ROOT/'tools'/name).read_text();begin=source.index('# Explicit FH20R1 registration only;')
        marker='# Explicitly activated FH12' if name=='_watchdog.sh' else '# FH12 is opt-in:'
        end=source.index(marker,begin);block=source[begin:end]
        block=block.replace('setsid nohup ','').replace('9>&- &','9>&-')
        with tempfile.TemporaryDirectory(prefix='fh20r1-shell-') as tmp:
            root=Path(tmp);(root/'tools').mkdir();(root/'work_dir').mkdir()
            (root/'tools/fh20r1_runner.py').write_text('import json,pathlib,sys\npathlib.Path("called.json").write_text(json.dumps(sys.argv[1:]))\n')
            if activated or incomplete:
                (root/'work_dir/_fh20r1').mkdir();(root/'work_dir/_fh20r1/local_server.txt').write_text('s2\n')
            if activated:
                directory=root/'work_dir/_fh20r1/s2';directory.mkdir();(directory/'campaign_budget.json').write_text('{"closed":true}')
            env=dict(os.environ,REPO=str(root),LOG=str(root/'work_dir/log'),PYTHON=sys.executable)
            result=subprocess.run(['bash','-c','exec 9>"$REPO/work_dir/lock"\n'+block+'\nprintf LEGACY\n'],
                                  cwd=root,env=env,capture_output=True,text=True,check=True)
            return result.stdout,json.loads((root/'called.json').read_text()) if (root/'called.json').exists() else None

    def test_pull_only_does_not_activate(self):
        for name in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(name=name):self.assertEqual(self.exercise(name),('LEGACY',None))

    def test_registration_even_closed_never_falls_back(self):
        for name in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(name=name):
                out,call=self.exercise(name,activated=True)
                self.assertNotIn('LEGACY',out);self.assertEqual(call,['run','--server','s2'])

    def test_partial_registration_preserves_hold(self):
        for name in ('_watchdog.sh','_run_cases.sh'):
            with self.subTest(name=name):
                out,call=self.exercise(name,incomplete=True)
                self.assertNotIn('LEGACY',out);self.assertIsNone(call)


if __name__=='__main__':unittest.main(verbosity=2)
