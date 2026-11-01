"""Failure evidence, dependency isolation and explicit control-only recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_qb40 import failures,plan,prerequisites,recovery,runner
from panda_qb40.tests.test_runner import FakeCampaign,complete
from panda_sensor_refine.orchestration import sha


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def failed_fixture(root):
    source=plan.ROOT/'config/bglobal_rr/qb_original_lms.json'
    target=root/'config/bglobal_rr/qb_original_lms.json'
    target.parent.mkdir(parents=True);target.write_bytes(source.read_bytes())
    folder=root/plan.G.OUTPUT/plan.G.CAMPAIGN_ID/'s1/QB'
    cfg=plan.resolve_case('s1','QB5_S1_A200_E100')
    directory=folder/'runs'/cfg['run_id']/'attempt_01'
    row=dict(run_id=cfg['run_id'],case=cfg['case'],replica='R1',phase='donor',
        directory=str(directory),status='FAILED',step=14219,poisoned=False,error=failures.SUPPORT_ERROR)
    dump(directory/'config.json',cfg);dump(directory/'requested_config.json',cfg)
    dump(directory/'run_state.json',row)
    (directory/'train.jsonl').write_text('{"step":14219}\n')
    checkpoint=directory/'models/stop_0014219.pt';checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'retained synthetic checkpoint')
    dump(directory/'models/latest_resume.json',dict(checkpoint_path=str(checkpoint),
        file_sha256=sha(checkpoint),completed_update=14219,campaign_resume_complete=True,checkpoint_verified=True))
    state=dict(status='FAILED',started_unix=1000,runs=[row],groups=[],timing={},donors={})
    dump(folder/'campaign_state.json',state)
    return folder,cfg,row,state


class FailureRecoveryTests(unittest.TestCase):
    def test_existing_failure_skips_only_its_two_students(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,40*3600);camp.state=state
            with patch.object(plan,'ROOT',root),patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),\
                 patch('panda_sensor_focus.early_stop.run_case',side_effect=complete):
                result=runner.run(camp,catalog='fixture')
            self.assertEqual(result['status'],failures.STATUS)
            self.assertEqual(len(result['failed_runs']),1);self.assertEqual(len(result['blocked_runs']),2)
            self.assertEqual(result['runs'][1]['case'],'QB5_C60LR050')
            self.assertEqual(len(result['runs']),10)
            self.assertEqual(result['runs'][0],row)
            self.assertNotIn(cfg['run_id'],camp.validated)
            self.assertFalse(result['all_registered_completed'])

    def test_new_support_failure_continues_independent_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,40*3600)
            def execute(c,cfg,budget):
                if cfg['case']==row['case']:
                    c.state['runs'].append(deepcopy(row))
                    raise ValueError('SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7')
                complete(c,cfg,budget)
            with patch.object(plan,'ROOT',root),patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),\
                 patch('panda_sensor_focus.early_stop.run_case',side_effect=execute):
                result=runner.run(camp,catalog='fixture')
            self.assertEqual(result['status'],failures.STATUS)
            self.assertEqual([r['case'] for r in result['runs']][1:3],['QB5_C60LR050']*2)
            self.assertEqual(camp.registered,['QB5_S1_A400_E030'])

    def test_unknown_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),\
             patch('panda_sensor_focus.early_stop.run_case',side_effect=ValueError('bad source hash')):
            camp=FakeCampaign(tmp,40*3600)
            with self.assertRaisesRegex(ValueError,'source hash'):runner.run(camp,catalog='fixture')
            self.assertEqual(camp.state['status'],'FAILED')
            self.assertEqual(len(camp.validated),1)

    def test_pending_excludes_failed_and_blocked_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,600);camp.state=state
            with patch.object(plan,'ROOT',root):result=runner.run(camp,catalog='fixture')
            self.assertEqual(result['status'],'BUDGET_COMPLETE_WITH_PENDING')
            self.assertEqual(len(result['pending_groups']),9)
            self.assertEqual(result['pending_groups'][0]['runs'][0][0],'QB5_C60LR050')

    def test_tampered_checkpoint_and_unsafe_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,40*3600);camp.state=state
            with patch.object(plan,'ROOT',root):
                proof=failures.verify(camp,row)
                self.assertEqual(proof['step'],14219)
                unsafe={**row,'poisoned':True}
                with self.assertRaises(ValueError):failures.verify(camp,unsafe)
                checkpoint=Path(row['directory'])/'models/stop_0014219.pt'
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(ValueError):failures.verify(camp,row)

    def test_revision_rejects_method_config_and_removed_code(self):
        old=dict(code={'panda_qb40/runtime.py':'a','panda_qb40/stage1.py':'b'},config={'lr':1})
        new=deepcopy(old);new['code']['panda_qb40/runtime.py']='c'
        self.assertEqual(recovery.validate_revision(old,new),['panda_qb40/runtime.py'])
        for bad in (dict(new,config={'lr':2}),dict(new,code={**new['code'],'panda_qb40/stage1.py':'c'}),
                    dict(new,code={'panda_qb40/stage1.py':'b'})):
            with self.assertRaises(ValueError):recovery.validate_revision(old,bad)

    def test_recovery_preserves_admission_failure_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            old=dict(code={'panda_qb40/runtime.py':'old','panda_qb40/stage1.py':'same'},config={'same':True})
            dump(folder/'admission.json',old)
            original=(folder/'admission.json').read_bytes()
            code={**old['code'],'panda_qb40/runtime.py':'new','panda_qb40/recovery.py':'new'}
            with patch.object(plan,'ROOT',root):
                receipt=recovery.prepare(folder,'s1',code)
                camp=FakeCampaign(folder,40*3600);camp.state=state
                recovery.accept_admission(camp,receipt['expected_admission'])
                self.assertEqual((folder/'admission.json').read_bytes(),original)
                self.assertEqual(state['started_unix'],1000)
                self.assertEqual(state['runs'],[row])
                self.assertEqual(recovery.prepare(folder,'s1',code),receipt)
                camp.state['runs'][0]={**row,'status':'RUNNING'}
                with self.assertRaises(ValueError):recovery.accept_admission(camp,receipt['expected_admission'])

    def test_recovery_refuses_an_existing_active_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            dump(folder/'admission.json',dict(code={'panda_qb40/runtime.py':'old'}))
            state['runs'][0]['status']='RUNNING';dump(folder/'campaign_state.json',state)
            with self.assertRaisesRegex(ValueError,'no active/completed'):
                recovery.prepare(folder,'s1',{'panda_qb40/runtime.py':'new'})

    def test_terminal_counts_failure_and_blocks_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,600);camp.state=state
            with patch.object(plan,'ROOT',root):
                result=runner.run(camp,catalog='fixture')
                dump(folder/'campaign_state.json',result)
                proof=prerequisites.terminal(root,'s1')
                self.assertEqual((proof['failed_count'],proof['blocked_count'],proof['pending_count']),(1,2,9))
                self.assertFalse(proof['full_queue_completed_without_early_stops'])
                result['blocked_runs']={};dump(folder/'campaign_state.json',result)
                with self.assertRaises(ValueError):prerequisites.terminal(root,'s1')

    def test_finished_failed_branch_allows_verified_closeout_of_nine_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,cfg,row,state=failed_fixture(root)
            camp=FakeCampaign(folder,40*3600);camp.state=state
            def finish(c,cfg,budget):
                complete(c,cfg,budget)
                directory=folder/'runs'/cfg['run_id']/'attempt_01'
                c.state['runs'][-1]['directory']=str(directory)
                dump(directory/'config.json',cfg)
            with patch.object(plan,'ROOT',root),patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),\
                 patch('panda_sensor_focus.early_stop.run_case',side_effect=finish),\
                 patch('panda_sensor_refine.orchestration.final_evidence',side_effect=lambda root,cfg:{'run_id':cfg['run_id']}):
                result=runner.run(camp,catalog='fixture');dump(folder/'campaign_state.json',result)
                proof=prerequisites.terminal(root,'s1')
                self.assertEqual(proof['campaign_status'],failures.STATUS)
                self.assertEqual((proof['completed_count'],proof['failed_count'],proof['blocked_count']),(9,1,2))
                self.assertFalse(proof['full_queue_completed_without_early_stops'])
                result['status']='FINISHED_REGISTERED_QUEUE';dump(folder/'campaign_state.json',result)
                with self.assertRaisesRegex(ValueError,'Failed branches'):
                    prerequisites.terminal(root,'s1')


if __name__=='__main__':unittest.main()
