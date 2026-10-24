"""Queue preservation and complete-state continuation are executable guards."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from donor30.artifacts import _same, capture_rng
from panda_next30 import plan as predecessor
from panda_next30.tests.test_core import fixture as source_fixture
from panda_stage2_condq.tests.test_method import batch
from panda_follow30 import plan, transition as T, runtime as R
from panda_follow30.engine import MatureStudentTrainer, ScheduleTrainingSpec
from panda_follow30.fork import _transfer_validated
from panda_follow30.method import MatureStage2
from panda_follow30.spec import MatureSpec


class TransitionTests(unittest.TestCase):
    def complete(self, server):
        return {'status':'FINISHED_REGISTERED_QUEUE', 'runs':[
            {'case':case,'replica':replica,'status':'COMPLETED'}
            for g in predecessor.groups(server) for case,replica in g['runs']]}

    @staticmethod
    def exited():
        return {'Running':False,'Status':'exited','ExitCode':0}

    def pruned(self, server):
        state=self.complete(server)
        groups=T.group_policy(server)
        skipped=[name for name,row in groups.items() if row['replaceable']]
        state['runs']=[row for row in state['runs'] if not any(
            (row['case'],row['replica']) in groups[name]['group']['runs'] for name in skipped)]
        state['pending_amendments']=[{'group':name,'request_id':T.REQUEST_ID,
            'reason':T.REASON,'skip_groups':skipped} for name in skipped]
        state['status']='FINISHED_WITH_EXPLICIT_SKIPS'
        return state

    def test_complete_or_exactly_pruned_queues_are_admitted(self):
        for server in ('s1','s3'):
            for state in (self.complete(server),self.pruned(server)):
                self.assertEqual(T.transition_decision(server,self.exited(),state),
                                 'VERIFY_MIGRATION_RECEIPTS')
                state['runs'].pop()
                with self.assertRaisesRegex(RuntimeError,'incomplete'):
                    T.transition_decision(server,self.exited(),state)

    def test_every_started_group_and_full_scale_control_are_kept(self):
        for server in ('s1','s3'):
            for group in predecessor.groups(server):
                case,replica=group['runs'][0]
                for status in ('RUNNING','STOPPED','FAILED','COMPLETED'):
                    state={'status':'RUNNING','runs':[{'case':case,'replica':replica,'status':status}]}
                    before=deepcopy(state)
                    skipped=T.unstarted_groups(server,state)
                    self.assertNotIn(group['name'],skipped)
                    self.assertTrue(all(T.group_policy(server)[name]['replaceable'] for name in skipped))
                    self.assertEqual(state,before)
            no_runs=T.unstarted_groups(server,{'runs':[]})
            self.assertEqual(len(no_runs),len(predecessor.groups(server))-4)

    def test_wrong_request_reason_full_scale_skip_and_status_fail_closed(self):
        for error in ('request','reason','full_scale','started','status','duplicate','historical'):
            state=self.pruned('s1')
            row=state['pending_amendments'][0]
            if error=='request': row['request_id']='unrelated-request'
            elif error=='reason': row['reason']='different-scope'
            elif error=='full_scale': row['skip_groups'].append('14_s100c30_r2')
            elif error=='started': state['runs'].append({'case':'S020C30','replica':'R1','status':'COMPLETED'})
            elif error=='status': state['status']='FINISHED_REGISTERED_QUEUE'
            elif error=='historical': state['migration_skips']=[{'group':'07_davg_r1'}]
            else: state['runs'].append(deepcopy(state['runs'][0]))
            with self.assertRaises((RuntimeError,ValueError),msg=error):
                T.transition_decision('s1',self.exited(),state)

    def test_proven_historical_low_scale_migration_is_preserved(self):
        state=self.pruned('s3')
        name=state['pending_amendments'].pop()['group']
        case,replica=T.group_policy('s3')[name]['group']['runs'][0]
        historical={'group':name,'run_id':predecessor.resolve_case('s3',case,replica)['run_id'],
                    'predecessor_run_id':'original_schedule_run','reason':'Completed predecessor case'}
        state['migration_skips']=[historical]
        state['migration_receipt']={'successor_skips':[deepcopy(historical)]}
        self.assertEqual(T.transition_decision('s3',self.exited(),state),'VERIFY_MIGRATION_RECEIPTS')
        state['migration_receipt']['successor_skips']=[]
        with self.assertRaisesRegex(RuntimeError,'immutable migration'):
            T.transition_decision('s3',self.exited(),state)

    def test_running_predecessor_is_read_only_wait(self):
        state={'status':'RUNNING','runs':[{'case':'UBIL','replica':'R2','step':1}]}
        before=deepcopy(state)
        self.assertTrue(T.transition_decision('s1',{'Running':True},state).startswith('WAITING_'))
        self.assertEqual(state,before)
        self.assertEqual(T.transition_decision('s1',self.exited(),{'status':'STOPPED_SAFE'}),
                         'WAITING_PREVIOUS_OPERATOR_RESUME')

    def test_pending_refresh_is_idempotent_and_removes_raced_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            directory=root/'work_dir/panda_next30'/T.PREVIOUS_CAMPAIGN/'s1/WV3'
            state={'status':'RUNNING','runs':[]}
            first=T.arm_pending_control(root,'s1',state)
            self.assertEqual(len(first['skip_groups']),14)
            state['runs']=[{'case':'S010C30','replica':'R2','status':'RUNNING','step':1}]
            T.atomic_json(directory/'campaign_state.json',state)
            # The fresh durable state protects a raced run even when the caller is stale.
            second=T.arm_pending_control(root,'s1',{'status':'RUNNING','runs':[]})
            self.assertEqual(len(second['skip_groups']),13)
            self.assertNotIn('11_s010c30_r2',second['skip_groups'])
            with patch.object(T,'atomic_json') as write:
                self.assertEqual(T.arm_pending_control(root,'s1',state),second)
                write.assert_not_called()
            T.atomic_json(directory/'pending_control.json',{'request_id':'another','reason':'x','skip_groups':[]})
            with self.assertRaisesRegex(ValueError,'Another predecessor'):
                T.arm_pending_control(root,'s1',state)

    def test_readonly_inspection_and_explicit_apply_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            directory=root/'work_dir/panda_next30'/T.PREVIOUS_CAMPAIGN/'s1/WV3'
            T.atomic_json(directory/'campaign_state.json',{'status':'RUNNING','runs':[]})
            with patch.object(T,'inspect_owned_container',return_value={'State':{'Running':True}}), \
                 patch.object(T,'verify_predecessor_pin',return_value={'verified':True}), \
                 patch.object(T.subprocess,'run') as launch:
                result=T.inspect_transition(root,'s1')
                self.assertEqual(len(result['proposed_skip_groups']),14)
                self.assertFalse((directory/'pending_control.json').exists())
                self.assertEqual(result['source_mutations'],[])
                changed=T.inspect_transition(root,'s1',apply_pending=True)
                self.assertTrue((directory/'pending_control.json').exists())
                self.assertEqual(changed['source_mutations'],['pending_control.json'])
                launch.assert_not_called()

    def test_admitted_source_changes_block_pending_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'source.py';source.write_text('original')
            directory=root/'work_dir/panda_next30'/T.PREVIOUS_CAMPAIGN/'s1/WV3'
            T.atomic_json(directory/'admission.json',{'config':predecessor.default_config('s1'),
                'code':{'source.py':T.file_sha(source)}})
            self.assertEqual(T.verify_predecessor_pin(root,'s1')['source_files_verified'],1)
            source.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'source changed'):
                T.verify_predecessor_pin(root,'s1')

    def test_migration_requires_complete_same_seed_15k_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);state=self.pruned('s1')
            (root/'source.py').write_text('original')
            T.atomic_json(root/'work_dir/panda_next30'/T.PREVIOUS_CAMPAIGN/'s1/WV3/admission.json',
                          {'config':predecessor.default_config('s1'),
                           'code':{'source.py':T.file_sha(root/'source.py')}})
            def evidence(cfg, directory, step):
                checkpoint=directory/'models'/f'step_{step:07d}.pt'
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                checkpoint.write_bytes(f"{cfg['run_id']}:{step}".encode())
                ident={'run_id':cfg['run_id'],'seed':cfg['seed'],'completed_update':step,
                       'model_state_sha256':'fixture-model-hash'}
                evaluation={'state_before':ident,'state_after':ident}
                artifact={'identity':ident,'completed_update':step,
                          'model_state_sha256':'fixture-model-hash',
                          'file_sha256':T.file_sha(checkpoint),'checkpoint_path':str(checkpoint)}
                T.atomic_json(checkpoint.with_suffix('.json'),
                              {'config':cfg,'artifact':artifact,'evaluation':evaluation})
                T.atomic_json(directory/f'evaluation_{step:07d}.json',evaluation)
            for row in state['runs']:
                cfg=predecessor.resolve_case('s1',row['case'],row['replica'])
                directory=root/'work_dir/panda_next30'/T.PREVIOUS_CAMPAIGN/'s1/WV3/runs'/cfg['run_id']/'attempt_01'
                row.update(step=cfg['planned_horizon'],run_id=cfg['run_id'],directory=str(directory))
                evidence(cfg,directory,cfg['planned_horizon'])
                if row['case']=='S100DIFFC30':evidence(cfg,directory,15000)
            receipt=T.migration_receipts(root,state,'s1')
            self.assertEqual(len(receipt['completed_runs']),4)
            self.assertEqual(len(receipt['required_continuation_parents']),2)
            self.assertEqual({p['completed_update'] for p in receipt['required_continuation_parents']},{15000})
            self.assertFalse(receipt['running_case_interrupted'])
            source=Path(receipt['required_continuation_parents'][0]['checkpoint_path'])
            source.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checkpoint SHA'):
                T.migration_receipts(root,state,'s1')
            source.unlink()
            with self.assertRaisesRegex(ValueError,'required checkpoint'):
                T.migration_receipts(root,state,'s1')

    def test_source_prefix_is_not_charged_or_required_in_new_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            R.recover_trace(directory,15000,15000)
            path=Path(directory)/'train.jsonl'
            rows=[{'step':15001,'local_update':1,'seconds_including_IO':.1},
                  {'step':15002,'local_update':2,'seconds_including_IO':.1}]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            R.recover_trace(directory,15001,15000)
            self.assertEqual(len(path.read_text().splitlines()),1)
            self.assertEqual(len(list(Path(directory).glob('train_before_resume_*'))),1)


class ForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_diff_source15k_preserves_weights_moments_and_stream(self):
        source=source_fixture('S100DIFFC30')
        source.step_accumulated([batch()])
        source.updates=15000;source.stage.set_completed_update(15000)
        source.optimizer.param_groups[0]['lr']=source.training.rate(15000)
        for item in source.optimizer.state.values():item['step'].fill_(15000)
        cfg=plan.resolve_case('s1','S100DIFFF15C30','R1')
        spec=MatureSpec(**cfg['stage2_spec'])
        stage=MatureStage2(spec,deepcopy(source.stage.aligner),deepcopy(source.stage.teacher),
                          donor_binding=deepcopy(source.stage._donor_binding),
                          stats_bank=source.stage.stats_bank)
        training=ScheduleTrainingSpec(**{**cfg['training_spec'],
                                        'microbatch_size':1,'effective_batch_size':1})
        target=MatureStudentTrainer(stage,training);target.config=cfg
        class Stream:
            def state_dict(self):return {'order':[2,4,1],'cursor':3}
        stream=Stream();payload={'stream':stream.state_dict(),'rng':capture_rng()}
        before=deepcopy(source.optimizer.state_dict())
        proof=_transfer_validated(source,target,stream,payload)
        self.assertTrue(_same(source.optimizer.state_dict(),before))
        self.assertTrue(_same(target.optimizer.state_dict(),before))
        self.assertTrue(_same(target.stage.student.state_dict(),source.stage.student.state_dict()))
        self.assertEqual(target.updates,15000)
        self.assertEqual(proof['next_lr'],2e-5)
        target.step_accumulated([batch()])
        self.assertEqual(target.updates,15001)
        self.assertEqual(target.optimizer.param_groups[0]['lr'],2e-5)
        self.assertTrue(all(int(v['step'])==15001 for v in target.optimizer.state.values()))
        with self.assertRaises(ValueError):_transfer_validated(source,target,stream,payload)

    def test_cosine_tail_resumes_complete_single_model_state(self):
        source=source_fixture('S100DIFFC30');source.step_accumulated([batch()])
        source.updates=15000;source.stage.set_completed_update(15000)
        source.optimizer.param_groups[0]['lr']=source.training.rate(15000)
        for item in source.optimizer.state.values():item['step'].fill_(15000)
        cfg=plan.resolve_case('s3','S100DIFFF15C30','R1')
        stage=MatureStage2(MatureSpec(**cfg['stage2_spec']),deepcopy(source.stage.aligner),
                          deepcopy(source.stage.teacher),donor_binding=source.stage._donor_binding,
                          stats_bank=source.stage.stats_bank)
        training=ScheduleTrainingSpec(**{**cfg['training_spec'],
                                        'microbatch_size':1,'effective_batch_size':1})
        target=MatureStudentTrainer(stage,training);target.config=cfg
        class Stream:
            def state_dict(self):return {'synthetic_order':[2,1]}
        stream=Stream()
        _transfer_validated(source,target,stream,{'stream':stream.state_dict(),'rng':capture_rng()})
        target.step_accumulated([batch()])
        saved=target.state_dict(stream_state=stream.state_dict())
        self.assertNotIn('swa',saved)
        target.step_accumulated([batch()])
        expected=deepcopy(target.stage.student.state_dict())
        target.load_state_dict(saved)
        self.assertEqual(target.updates,15001)
        target.step_accumulated([batch()])
        self.assertTrue(_same(target.stage.student.state_dict(),expected))
        self.assertEqual(target.optimizer.param_groups[0]['lr'],target.training.rate(15002))


if __name__ == '__main__':unittest.main()
