"""Unlimited fresh runtime and wait-only predecessor handoff guards."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from panda_next30 import plan, runtime as R, transition as T
from panda_schedule30 import plan as predecessor


class Clock:
    def __init__(self):
        self.wall, self.monotonic = 200000., 50000.

    def __call__(self):
        return self.wall


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(R, 'code_pin', return_value={'synthetic.py':'d'*64}))
        self.stack.enter_context(patch.object(R, 'evaluator_identity', return_value={'protocol':'SYNTHETIC'}))
        self.stack.enter_context(patch.object(R.time, 'monotonic', side_effect=lambda:self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)))
        self.campaign = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)

    def test_no_wall_deadline_even_after_long_pause_and_restart(self):
        c = self.campaign
        started = c.state['started_unix']
        self.clock.wall += 365*24*3600
        self.clock.monotonic += 365*24*3600
        c.check()
        c.persist()
        restarted = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)
        restarted.check()
        self.assertEqual(restarted.state['started_unix'], started)
        self.assertIsNone(restarted.deadline_unix)
        self.assertGreaterEqual(restarted.elapsed_seconds(), 365*24*3600)

    def test_stop_and_disk_guards_remain(self):
        self.campaign.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, 'STOP_REQUESTED'):
            self.campaign.check()
        self.campaign.stop_requested = False
        with patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired, 'DISK_CLOSEOUT'):
                self.campaign.check()

    def test_architecture_input_and_warp_have_distinct_throughput_keys(self):
        base = plan.resolve_case('s1', 'S020C30', 'R1')
        repeat = plan.resolve_case('s1', 'S020C30', 'R2')
        self.assertEqual(self.campaign.timing_key(base), self.campaign.timing_key(repeat))
        for case in ('DAVG','UBIL','DAVG_UBIL','S100DIFFC30'):
            cfg = plan.resolve_case('s1', case, 'R1')
            self.assertNotEqual(self.campaign.timing_key(base), self.campaign.timing_key(cfg))
        warp = deepcopy(base)
        warp['student_warp'] = {'method':'different'}
        self.assertNotEqual(self.campaign.timing_key(base), self.campaign.timing_key(warp))

    def test_fresh_only_build_rejects_source_import(self):
        case,replica = plan.groups('s1')[0]['runs'][0]
        cfg = plan.resolve_case('s1', case, replica)
        cfg['inherited_updates'] = 25000
        with self.assertRaisesRegex(ValueError, 'fresh Student'):
            self.campaign.build(cfg)

    def test_all_finite_groups_admitted_without_time_budget(self):
        for group in self.campaign.config['groups']:
            self.assertTrue(self.campaign.admit_group(group))
            self.assertIsNone(self.campaign.state['groups'][-1]['remaining_seconds'])

    def test_trace_preserves_uncheckpointed_tail_as_evidence(self):
        directory = Path(self.temp.name)/'trace'
        directory.mkdir()
        rows = [{'step':step, 'local_update':step, 'seconds_including_IO':.1} for step in (1,2,3)]
        path = directory/'train.jsonl'
        original = ''.join(json.dumps(row)+'\n' for row in rows)
        path.write_text(original)
        R.recover_trace(directory, 2)
        self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [1,2])
        self.assertEqual(next(directory.glob('train_before_resume_*.jsonl')).read_text(), original)


class TransitionTests(unittest.TestCase):
    @staticmethod
    def completed(server):
        return {'status':'FINISHED_REGISTERED_QUEUE', 'runs':[
            {'case':case, 'replica':replica, 'status':'COMPLETED'}
            for group in predecessor.groups(server) for case,replica in group['runs']]}

    def test_every_running_case_is_preserved(self):
        for server in ('s1','s3'):
            for case,replica in [(c,r) for g in predecessor.groups(server) for c,r in g['runs']]:
                campaign = {'status':'RUNNING', 'runs':[{'case':case, 'replica':replica, 'status':'RUNNING'}]}
                original = deepcopy(campaign)
                self.assertEqual(T.transition_decision(server, {'Running':True}, campaign),
                                 'WAITING_CURRENT_RUN_COMPLETION')
                self.assertEqual(campaign, original)
                skipped = T.unstarted_groups(server, campaign)
                active = next(g['name'] for g in predecessor.groups(server) if (case,replica) in g['runs'])
                self.assertNotIn(active, skipped)
                self.assertEqual(len(skipped), len(predecessor.groups(server))-1)

    def test_safe_stop_waits_for_operator_resume(self):
        self.assertEqual(T.transition_decision('s1', {'Running':False,'Status':'exited','ExitCode':0},
            {'status':'STOPPED_SAFE', 'runs':[]}), 'WAITING_PREVIOUS_OPERATOR_RESUME')

    def test_complete_or_explicitly_superseded_queue_admits_verification(self):
        for server in ('s1','s3'):
            complete = self.completed(server)
            self.assertEqual(T.transition_decision(server, {'Running':False,'Status':'exited','ExitCode':0},
                             complete), 'VERIFY_MIGRATION_RECEIPTS')
            complete['runs'] = complete['runs'][:1]
            groups = T.unstarted_groups(server, complete)
            complete['pending_amendments'] = [{'group':group, 'skip_groups':groups,
                'request_id':T.REQUEST_ID, 'reason':'User superseded pending cases'} for group in groups]
            complete['status'] = 'FINISHED_WITH_EXPLICIT_SKIPS'
            self.assertEqual(T.transition_decision(server, {'Running':False,'Status':'exited','ExitCode':0},
                             complete), 'VERIFY_MIGRATION_RECEIPTS')

    def test_abnormal_incomplete_duplicate_or_invalid_skip_fails_closed(self):
        for error in ('exit','missing','failed_run','duplicate','bad_skip'):
            campaign = self.completed('s1')
            container = {'Running':False,'Status':'exited','ExitCode':0}
            if error == 'exit': container['ExitCode'] = 1
            elif error == 'missing': campaign['runs'].pop()
            elif error == 'failed_run': campaign['runs'][0]['status'] = 'FAILED'
            elif error == 'bad_skip': campaign['pending_amendments'] = [{'group':'unknown','request_id':'x','reason':'x','skip_groups':['unknown']}]
            else: campaign['runs'].append(deepcopy(campaign['runs'][0]))
            with self.assertRaises(RuntimeError):
                T.transition_decision('s1', container, campaign)

    def test_pending_control_archives_previous_request_and_removes_raced_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/'work_dir/panda_schedule30'/T.PREVIOUS_CAMPAIGN/'s1/WV3'
            directory.mkdir(parents=True)
            original = {'request_id':'earlier-user-request', 'reason':'Earlier explicit request', 'skip_groups':[]}
            control = directory/'pending_control.json'
            T.atomic_json(control, original)
            state = {'runs':[]}
            first = T.arm_pending_control(root, 's1', state)
            self.assertEqual(len(first['skip_groups']),len(predecessor.groups('s1')))
            self.assertEqual(json.loads(next(directory.glob('pending_control_before_next30_*')).read_text()), original)
            state['runs'].append({'case':'S020F25C35','replica':'R1','status':'RUNNING'})
            second = T.arm_pending_control(root, 's1', state)
            self.assertEqual(len(second['skip_groups']), len(first['skip_groups'])-1)
            self.assertEqual(state['runs'][0]['status'],'RUNNING')
            with patch.object(T, 'atomic_json') as write:
                T.arm_pending_control(root, 's1', state)
                write.assert_not_called()

    def test_prepare_only_has_no_source_pin_or_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'gspread').mkdir(); (root/'gspread/server.txt').write_text('s1')
            path = root/'work_dir/panda_schedule30'/T.PREVIOUS_CAMPAIGN/'s1/WV3/campaign_state.json'
            T.atomic_json(path, {'status':'RUNNING','runs':[{'case':'S020F25C35','replica':'R1','status':'RUNNING'}]})
            with patch.object(T,'inspect_owned_container',return_value={'State':{'Running':True}}), \
                 patch.object(T,'retire_sampling_watcher',return_value={'status':'RETIRED','trainer_signalled':False}), \
                 patch.object(T.subprocess,'run') as launch:
                receipt = T.prepare_migration(root,'s1')
            launch.assert_not_called()
            self.assertFalse(receipt['source_pin_registered'])
            self.assertFalse(receipt['trainer_signalled'])
            self.assertFalse(receipt['successor_launched'])
            self.assertNotIn(predecessor.groups('s1')[0]['name'],receipt['pending_control']['skip_groups'])
            compact = T.concise_receipt({**receipt,'source_pin':{'large':'a'*1000},
                'sampling_watcher':{'status':'RETIRED','pid':123,'previous_receipt':{'source_pin':{'large':'x'*1000}}}})
            self.assertNotIn('source_pin',json.dumps(compact))

    def test_sampling_retirement_refuses_existing_container_or_foreign_pid(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(T.subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')), \
                 patch.object(T.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'container already exists'):
                    T.retire_sampling_watcher(root,'s1')
                kill.assert_not_called()
            directory = root/'work_dir/panda_sampling30_launchers/s1'
            directory.mkdir(parents=True); (directory/'pid').write_text(str(os.getpid()))
            T.atomic_json(directory/'transition_receipt.json',{'status':'WAITING_PREVIOUS_QUEUE_COMPLETION'})
            with patch.object(T.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='No such object')), \
                 patch.object(T.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'another command'):
                    T.retire_sampling_watcher(root,'s1')
                kill.assert_not_called()

    def test_migration_proves_final_states_deduplicates_old_cases_but_keeps_l4_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = {'runs':[]}
            for case in ('S020C30','S010C30'):
                cfg = json.loads(json.dumps(predecessor.resolve_case('s1',case,'R1')))
                directory = root/'work_dir/panda_schedule30'/T.PREVIOUS_CAMPAIGN/'s1/WV3/runs'/cfg['run_id']/'attempt_01'
                (directory/'models').mkdir(parents=True)
                step = cfg['planned_horizon']
                checkpoint = directory/'models'/f'step_{step:07d}.pt'
                checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT HASH ONLY')
                identity = {'run_id':cfg['run_id'], 'seed':cfg['seed'], 'completed_update':step,
                            'model_state_sha256':'a'*64}
                evaluation = {'state_before':identity, 'state_after':identity}
                artifact = {'completed_update':step, 'file_sha256':T.file_sha(checkpoint),
                            'checkpoint_path':str(checkpoint),'identity':identity,'model_state_sha256':'a'*64}
                checkpoint.with_suffix('.json').write_text(json.dumps({'artifact':artifact,'config':cfg,'evaluation':evaluation}))
                (directory/f'evaluation_{step:07d}.json').write_text(json.dumps(evaluation))
                campaign['runs'].append({'case':case,'replica':'R1','run_id':cfg['run_id'],
                    'status':'COMPLETED','step':step,'directory':str(directory)})
            receipt = T.migration_receipts(root, campaign, 's1')
            self.assertEqual(len(receipt['completed_runs']),2)
            self.assertEqual(len(receipt['successor_skips']),1)
            self.assertIn('S010C30',receipt['successor_skips'][0]['run_id'])
            self.assertTrue(all(row['reference_only_no_weights_imported'] and not row['reposted_to_sheet']
                                for row in receipt['completed_runs']))
            metadata_path = Path(receipt['completed_runs'][0]['metadata_path'])
            original = json.loads(metadata_path.read_text())
            wrong = deepcopy(original); wrong['config']['dataset']='GF2'
            metadata_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'provenance'):
                T.migration_receipts(root, campaign, 's1')
            metadata_path.write_text(json.dumps(original))
            Path(receipt['completed_runs'][0]['checkpoint_path']).write_bytes(b'CORRUPTED')
            with self.assertRaisesRegex(ValueError,'provenance'):
                T.migration_receipts(root, campaign, 's1')

    def test_migration_skip_never_suppresses_started_successor(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(R,'code_pin',return_value={}), patch.object(R,'evaluator_identity',return_value={}):
                campaign = R.Campaign('s1',output=temporary,device='cpu')
            group = plan.groups('s1')[0]
            cfg = plan.resolve_case('s1',*group['runs'][0])
            row = {'group':group['name'],'run_id':cfg['run_id'],'reason':'Synthetic migration'}
            campaign.migration = {'successor_skips':[row]}
            self.assertTrue(campaign.migration_skip(group))
            self.assertTrue(campaign.migration_skip(group))
            self.assertEqual(campaign.state['migration_skips'],[row])
            campaign.state['runs'].append({'run_id':cfg['run_id'],'status':'RUNNING'})
            with self.assertRaisesRegex(ValueError,'already started'):
                campaign.migration_skip(group)


if __name__ == '__main__':
    unittest.main()
