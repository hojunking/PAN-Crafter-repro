import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from panda_sensor5 import queue
from panda_sensor5.wv3 import controlled_class


class TestChainOrder(unittest.TestCase):
    def test_waiting_current_run_does_not_start_dataset(self):
        chain={'stage':'CUTOVER'}
        with patch.object(queue,'ensure_cutover',return_value=False), \
             patch.object(queue.subprocess,'run') as process:
            queue.advance('s1',Path('/tmp'),chain)
        self.assertEqual(chain['stage'],'CUTOVER')
        process.assert_not_called()

    def test_dataset_must_exit_before_wv3_resumes(self):
        chain={'stage':'WAIT_SENSOR5','sensor_container':'sensor'}
        with patch.object(queue,'wait_exit',return_value=False), \
             patch.object(queue,'campaign_complete') as complete:
            queue.advance('s1',Path('/tmp'),chain)
        self.assertEqual(chain['stage'],'WAIT_SENSOR5')
        complete.assert_not_called()

    def test_dataset_complete_routes_tune_then_hard(self):
        chain={'stage':'WAIT_SENSOR5','sensor_container':'sensor'}
        with patch.object(queue,'wait_exit',return_value=True), \
             patch.object(queue,'campaign_complete',return_value=True):
            queue.advance('s1',Path('/tmp'),chain)
        self.assertEqual(chain['stage'],'RESUME_TUNE30')
        with patch.object(queue,'state_of',return_value={'status':'FINISHED_REGISTERED_QUEUE'}), \
             patch.object(queue,'wait_exit',return_value=True), \
             patch.object(queue,'campaign_complete',return_value=True):
            queue.advance('s1',Path('/tmp'),chain)
            self.assertEqual(chain['stage'],'RESUME_HARD36')
            queue.advance('s1',Path('/tmp'),chain)
        self.assertEqual(chain['stage'],'COMPLETED')

    def test_terminal_state_still_waits_for_own_adapter_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'wv3_tune30_resume-all.json').write_text(json.dumps({'container':'resume'}))
            chain={'stage':'RESUME_TUNE30'}
            with patch.object(queue,'wait_exit',return_value=False), \
                 patch.object(queue,'campaign_complete') as complete:
                queue.advance('s1',root,chain)
        self.assertEqual(chain['stage'],'RESUME_TUNE30')
        complete.assert_not_called()

    def test_failed_child_does_not_advance_chain(self):
        chain={'stage':'WAIT_SENSOR5','sensor_container':'sensor'}
        with patch.object(queue,'wait_exit',side_effect=ValueError('failed')):
            with self.assertRaises(ValueError):
                queue.advance('s1',Path('/tmp'),chain)
        self.assertEqual(chain['stage'],'WAIT_SENSOR5')


class TestBoundaryAdapter(unittest.TestCase):
    def setup_controller(self,root):
        cfgs={n:{'run_id':n,'planned_horizon':10} for n in 'ABC'}
        groups=[{'name':n,'optional':False,'runs':[(n,'R1')]} for n in 'ABC']
        class Base:
            def __init__(self,server,**kwargs):
                self.root=root; self.server=server; self.clock=lambda:1000.
                self.stop_requested=False
                self.config={'budget':{'total_seconds':1},'groups':groups}
                self.state={'started_unix':100.,'elapsed_seconds':10.,
                    'groups':[{'admitted':False,'reason':'INSUFFICIENT_30H_BUDGET'}],
                    'runs':[{'run_id':'A','status':'COMPLETED','step':10},
                            {'run_id':'B','status':'STOPPED','step':7}]}
                self.called=[]
            def persist(self):
                self.state['elapsed_seconds']=self.elapsed_seconds()
            def admission(self,catalog):
                self.state['status']='RUNNING'
            def register_donor(self,donor):
                pass
            def completed(self,cfg):
                return next((r for r in self.state['runs'] if r['run_id']==cfg['run_id']
                             and r['status']=='COMPLETED'),None)
            def run_case(self,cfg):
                if self.completed(cfg): return
                self.called.append(cfg['run_id'])
                row=next((r for r in self.state['runs'] if r['run_id']==cfg['run_id']),None)
                if row is None:
                    row={'run_id':cfg['run_id']}; self.state['runs'].append(row)
                row.update(status='COMPLETED',step=10)
        runtime=types.SimpleNamespace(Campaign=Base,BudgetExpired=RuntimeError)
        plan=types.SimpleNamespace(ROOT=root,DONORS=['donor'],
            resolve_case=lambda server,c,r:cfgs[c])
        return runtime,plan

    def test_finishes_only_original_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,plan=self.setup_controller(Path(directory))
            with patch('panda_sensor5.wv3.importlib.import_module',side_effect=[runtime,plan]):
                cls=controlled_class('panda_tune30')
            c=cls('s1',mode='finish-current',target_run='B')
            result=c.run()
        self.assertEqual(c.called,['B'])
        self.assertEqual(result['status'],'PAUSED_FOR_SENSOR5')
        self.assertEqual(result['started_unix'],100.)
        self.assertEqual(result['sensor5_control']['original_budget_seconds'],1)

    def test_resume_runs_every_pending_case_despite_old_budget_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,plan=self.setup_controller(Path(directory))
            with patch('panda_sensor5.wv3.importlib.import_module',side_effect=[runtime,plan]):
                cls=controlled_class('panda_tune30')
            c=cls('s1',mode='resume-all')
            result=c.run()
        self.assertEqual(c.called,['B','C'])
        self.assertEqual(result['status'],'FINISHED_REGISTERED_QUEUE')
        self.assertTrue(all(r['status']=='COMPLETED' for r in result['runs']))
        self.assertEqual(result['groups'][-1]['reason'],'USER_AUTHORIZED_ALL_REMAINING_WV3')

    def test_existing_manual_skip_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'pending_control.json').write_text('{}')
            runtime,plan=self.setup_controller(root)
            with patch('panda_sensor5.wv3.importlib.import_module',side_effect=[runtime,plan]):
                cls=controlled_class('panda_tune30')
            c=cls('s1',mode='resume-all')
            with self.assertRaises(ValueError): c.run()
        self.assertEqual(c.called,[])
