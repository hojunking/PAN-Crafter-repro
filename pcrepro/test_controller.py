"""Offline controller protocol tests: no subprocess/GPU/legacy actions run."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcrepro import controller as C
from pcrepro.common import atomic_json, camp, read, run_dir
from pcrepro.plan import CAMPAIGN_ID, RECIPE_SHA256, Case, cases_for, case_for, cycle_seed


@contextmanager
def mocked_runner(root, case_action):
    with ExitStack() as stack:
        stack.enter_context(patch('pcrepro.controller.verify_registration',return_value=dict(bindings_path='fixture.json',upload=False)))
        stack.enter_context(patch('pcrepro.handoff.begin',return_value=dict(status='COMPLETE')))
        stack.enter_context(patch('pcrepro.handoff.verify_stopped',return_value=dict(complete=True)))
        stack.enter_context(patch('pcrepro.handoff.gpu_processes',return_value=[]))
        stack.enter_context(patch('pcrepro.handoff.inventory',return_value=[]))
        stack.enter_context(patch('pcrepro.preflight.disk_guard',return_value=dict(allowed=True)))
        action=stack.enter_context(patch('pcrepro.controller._case',side_effect=case_action))
        report=stack.enter_context(patch('pcrepro.reporting.cycle_report',return_value={}))
        stack.enter_context(patch('pcrepro.controller.time.sleep',side_effect=AssertionError('test must not busy-wait')))
        yield action,report


class CasePolicyTests(unittest.TestCase):
    def test_seed_schedule_and_local_dataset_orders(self):
        orders=dict(s3=('WV3','WV2','QB','GF2'),s4=('QB','GF2','WV3','WV2'),s5=('GF2','WV3','WV2','QB'))
        for server,order in orders.items():
            self.assertEqual(tuple(c.dataset for c in cases_for(server,0)),order)
            self.assertTrue(all(c.seed==2025 for c in cases_for(server,0)))
            for cycle in (1,2,1234567):
                self.assertEqual(cycle_seed(server,cycle),1000000+3*(cycle-1)+int(server[1])-3)
                for case in cases_for(server,cycle):self.assertEqual(case_for(case.run_id),case)
            wv2=Case(server,2,'WV2')
            self.assertEqual(wv2.source_run_id,Case(server,2,'WV3').run_id)
            self.assertEqual(wv2.updates,0)

    def test_no_seed_wrap_or_foreign_server(self):
        with self.assertRaises(ValueError):cycle_seed('s3',2**32)
        for server in ('s1','s2','s6'):
            with self.assertRaises(ValueError):Case(server,0,'WV3')
        for cycle in (-1,True,.5):
            with self.assertRaises(ValueError):cycle_seed('s3',cycle)
        with self.assertRaises(ValueError):case_for('PCREPRO_WV3_s3_C000001_S2025_F50K_v1')


class ControllerTests(unittest.TestCase):
    def test_controls_are_local_and_validate_commands_before_writes(self):
        with tempfile.TemporaryDirectory() as root:
            for command in C.CONTROLS:self.assertEqual(C.control(root,'s3',command)['command'],command)
            self.assertFalse(camp(root,'s4').exists())
            with self.assertRaises(ValueError):C.control(root,'s3','KILL_NOW')
            with self.assertRaises(ValueError):C.control(root,'s1','STOP_NOW_SAFE')
            # Control requests must durably journal without replacing the requested timestamp.
            import json
            events=[json.loads(line) for line in (camp(root,'s3')/'control_ledger.jsonl').read_text().splitlines()]
            self.assertEqual([e['command'] for e in events],list(C.CONTROLS))
            self.assertTrue(all(e['event']=='CONTROL' and e['at_utc'] for e in events))

    def test_registered_bindings_and_recipe_and_source_are_immutable(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'bindings.json';atomic_json(path,dict(datasets={'GF2':'original'}))
            value=dict(campaign_id=CAMPAIGN_ID,server='s3',recipe_sha256=RECIPE_SHA256,
                source_identity={'sha':'original'},bindings_path=str(path),bindings_sha256=C.object_sha(read(path)))
            reg=camp(root,'s3')/'registration.json';atomic_json(reg,value)
            with patch('pcrepro.controller.apply_runtime_policy'),patch('pcrepro.controller.source_identity',return_value={'sha':'original'}):
                self.assertEqual(C.verify_registration(root,'s3'),value)
                for key,corrupt in [('server','s4'),('campaign_id','OTHER'),('recipe_sha256','changed'),('source_identity',{'sha':'changed'}),('bindings_sha256','changed')]:
                    atomic_json(reg,dict(value,**{key:corrupt}))
                    with self.subTest(key=key),self.assertRaises(ValueError):C.verify_registration(root,'s3')
                atomic_json(reg,value);atomic_json(path,dict(datasets={'GF2':'replacement'}))
                with self.assertRaises(ValueError):C.verify_registration(root,'s3')

    def test_stop_now_safe_before_new_admission(self):
        with tempfile.TemporaryDirectory() as root:
            C.control(root,'s3','STOP_NOW_SAFE')
            with mocked_runner(root,lambda *_:self.fail('must not start case')) as (action,_):
                self.assertEqual(C.run(root,'s3'),75);action.assert_not_called()
            self.assertEqual(C._state(root,'s3')['index'],0)

    def test_stop_after_current_resumes_only_active_run(self):
        with tempfile.TemporaryDirectory() as root:
            current=Case('s3',0,'WV3');state=C._state(root,'s3');state['active_run']=current.run_id
            C._save(root,'s3',state);C.control(root,'s3','STOP_AFTER_CURRENT_RUN');seen=[]
            def action(root,server,state,case,binding):seen.append(case);return 'TERMINAL'
            with mocked_runner(root,action):self.assertEqual(C.run(root,'s3'),75)
            self.assertEqual(seen,[current]);self.assertEqual(C._state(root,'s3')['index'],1)

    def test_stop_after_cycle_finishes_four_segments_not_four_trainings(self):
        with tempfile.TemporaryDirectory() as root:
            C.control(root,'s4','STOP_AFTER_CYCLE');seen=[]
            def action(root,server,state,case,binding):
                seen.append(case)
                if len(seen)>4:self.fail('must stop at cycle boundary')
                return 'TERMINAL'
            with mocked_runner(root,action) as (_,report):self.assertEqual(C.run(root,'s4'),75)
            self.assertEqual(seen,list(cases_for('s4',0)))
            self.assertEqual(sum(c.stage=='TRAIN' for c in seen),3)
            state=C._state(root,'s4');self.assertEqual((state['cycle'],state['index']),(1,0))
            report.assert_called_once_with(Path(root).resolve(),'s4',0)

    def test_next_cycle_is_new_case_seed_and_control_can_stop_it(self):
        with tempfile.TemporaryDirectory() as root:
            seen=[]
            def action(root,server,state,case,binding):
                seen.append(case)
                if len(seen)>5:self.fail('unexpected extra case')
                if case.cycle==1:C.control(root,server,'STOP_AFTER_CURRENT_RUN')
                return 'TERMINAL'
            with mocked_runner(root,action):self.assertEqual(C.run(root,'s5'),75)
            self.assertEqual(seen[:4],list(cases_for('s5',0)))
            self.assertEqual(seen[4].seed,1000002);self.assertEqual(seen[4].dataset,'GF2')

    def test_pause_keeps_cursor_and_same_run_for_resume(self):
        with tempfile.TemporaryDirectory() as root:
            state=C._state(root,'s3');state.update(cycle=4,index=2);C._save(root,'s3',state)
            with mocked_runner(root,lambda *_:'PAUSE') as (action,_):self.assertEqual(C.run(root,'s3'),75)
            self.assertEqual((C._state(root,'s3')['cycle'],C._state(root,'s3')['index']),(4,2))
            self.assertEqual(action.call_args.args[3],Case('s3',4,'QB'))

    def test_crash_after_last_cursor_save_finishes_cycle_without_retraining(self):
        with tempfile.TemporaryDirectory() as root:
            state=C._state(root,'s3');state.update(cycle=0,index=4);C._save(root,'s3',state)
            C.control(root,'s3','STOP_AFTER_CYCLE')
            with mocked_runner(root,lambda *_:self.fail('completed cycle cannot retrain')) as (action,report):
                self.assertEqual(C.run(root,'s3'),75);action.assert_not_called()
            self.assertEqual((C._state(root,'s3')['cycle'],C._state(root,'s3')['index']),(1,0))
            report.assert_called_once()

    def test_all_training_datasets_blocked_pauses_without_busy_loop(self):
        with tempfile.TemporaryDirectory() as root:
            state=C._state(root,'s3');state['blocked_datasets']={k:'invalid source' for k in ('WV3','QB','GF2')};C._save(root,'s3',state)
            with mocked_runner(root,lambda *_:self.fail('no runnable datasets')) as (action,_):
                self.assertEqual(C.run(root,'s3'),75);action.assert_not_called()
            self.assertEqual(C._state(root,'s3')['status'],'PAUSED_NO_RUNNABLE_DATASET')

    def test_resource_pause_never_advances_cursor_or_starts_child(self):
        with tempfile.TemporaryDirectory() as root:
            with mocked_runner(root,lambda *_:self.fail('GPU busy')) as (action,_), \
                    patch('pcrepro.handoff.gpu_processes',return_value=[9876]):
                self.assertEqual(C.run(root,'s3'),75);action.assert_not_called()
            self.assertEqual(C._state(root,'s3')['index'],0)

    def test_failed_wv3_blocks_only_same_cycle_wv2_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',2,'WV2');state=C._state(root,'s3')
            # A previous-cycle success cannot satisfy this cycle dependency.
            atomic_json(run_dir(Case('s3',1,'WV3'),root)/'meta/training_status.json',dict(training_complete=True,actual_updates=50000))
            with patch('pcrepro.controller._status_row') as status,patch('pcrepro.controller._command') as command:
                self.assertEqual(C._case(root,'s3',state,c,'fixture'),'TERMINAL')
                command.assert_not_called();self.assertEqual(status.call_args.args[3],'BLOCKED_DEPENDENCY')
            self.assertNotIn('QB',state['blocked_datasets']);self.assertNotIn('GF2',state['blocked_datasets'])

    def test_training_io_retry_cap_two_preserves_same_run_and_recipe(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'QB');state=C._state(root,'s3');wd=run_dir(c,root)
            atomic_json(wd/'resume/index.json',dict(snapshots=['fixture']))
            atomic_json(wd/'meta/training_status.json',dict(status='FAILED_IO',actual_updates=123))
            calls=[]
            def command(root,server,args,*_):calls.append(args);return (0,0.) if args[0]=='preflight' else (74,1.)
            with patch('pcrepro.controller._command',side_effect=command),patch('pcrepro.controller._config',return_value=wd/'cfg'), \
                    patch('pcrepro.controller._status_row'):
                self.assertEqual(C._case(root,'s3',state,c,'fixture'),'TERMINAL')
            train=[a for a in calls if a[0]=='train']
            self.assertEqual(len(train),3);self.assertTrue(all('--resume' in args for args in train))
            self.assertEqual(state['runs'][c.run_id]['io_retries'],2)
            self.assertIn('QB',state['blocked_datasets']);self.assertNotIn('GF2',state['blocked_datasets'])

    def test_completed_training_is_not_started_again_when_postrun_retried(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'WV3');state=C._state(root,'s3');wd=run_dir(c,root)
            atomic_json(wd/'meta/training_status.json',dict(training_complete=True,actual_updates=50000))
            atomic_json(wd/'official/summary.json',dict(complete=True,status='COMPLETE'))
            with patch('pcrepro.controller._command',return_value=(0,0.)) as command, \
                    patch('pcrepro.controller._config',return_value=wd/'cfg'),patch('pcrepro.upload.spool_run'):
                self.assertEqual(C._case(root,'s3',state,c,'fixture'),'TERMINAL')
            self.assertEqual([call.args[2][0] for call in command.call_args_list],['preflight','postrun'])


class DeferredEvaluationTests(unittest.TestCase):
    def pending(self,root,cases):
        state=C._state(root,'s3')
        state['pending_evaluations']={c.run_id:dict(config=str(run_dir(c,root)/'config.json'),attempts=0) for c in cases}
        return state

    def test_cycle_completion_preserves_evaluation_debt_and_allows_later_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'WV3');state=self.pending(root,[c]);state['index']=4
            with patch('pcrepro.reporting.cycle_report'):self.assertTrue(C.finish_cycle(root,'s3',state))
            restored=C._state(root,'s3')
            self.assertEqual(restored['cycle'],1);self.assertIn(c.run_id,restored['pending_evaluations'])
            self.assertTrue(restored['evaluation_retry_due'])
            atomic_json(run_dir(c,root)/'official/summary.json',dict(complete=True))
            with patch('pcrepro.controller._command',return_value=(0,0)) as command,patch('pcrepro.controller._status_row'):
                self.assertEqual(C.retry_evaluations(root,'s3',restored),0)
            self.assertEqual(command.call_args.args[2],['postrun','--config',str(run_dir(c,root)/'config.json')])
            self.assertNotIn(c.run_id,C._state(root,'s3')['pending_evaluations'])
            self.assertEqual(C._state(root,'s3')['cycle'],1)

    def test_transient_evaluation_failure_rotates_without_starving_others(self):
        with tempfile.TemporaryDirectory() as root:
            cases=[Case('s3',0,'WV3'),Case('s3',0,'QB')];state=self.pending(root,cases)
            with patch('pcrepro.controller._command',return_value=(74,1)) as command:
                self.assertEqual(C.retry_evaluations(root,'s3',state,limit=1),0)
            self.assertEqual(command.call_count,1)
            self.assertEqual(list(state['pending_evaluations']),[cases[1].run_id,cases[0].run_id])
            self.assertEqual(state['pending_evaluations'][cases[0].run_id]['attempts'],1)
            self.assertFalse(state['blocked_datasets'])

    def test_deterministic_evaluation_failure_is_isolated_from_training_datasets(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'GF2');state=self.pending(root,[c])
            with patch('pcrepro.controller._command',return_value=(1,1)):
                C.retry_evaluations(root,'s3',state)
            self.assertNotIn(c.run_id,state['pending_evaluations'])
            self.assertIn(c.run_id,state['blocked_evaluations']);self.assertFalse(state['blocked_datasets'])

    def test_safe_pause_keeps_checkpoint_evaluation_debt(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'QB');state=self.pending(root,[c])
            with patch('pcrepro.controller._command',return_value=(75,1)):
                self.assertEqual(C.retry_evaluations(root,'s3',state),75)
            self.assertIn(c.run_id,C._state(root,'s3')['pending_evaluations'])
            C.control(root,'s3','STOP_NOW_SAFE')
            with patch('pcrepro.controller._command') as command:
                self.assertEqual(C.retry_evaluations(root,'s3',state),75);command.assert_not_called()

    def test_foreign_server_evaluation_debt_rejected_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as root:
            state=self.pending(root,[Case('s4',0,'GF2')])
            with patch('pcrepro.controller._command') as command:
                with self.assertRaises(ValueError):C.retry_evaluations(root,'s3',state)
                command.assert_not_called()


if __name__=='__main__':unittest.main()
