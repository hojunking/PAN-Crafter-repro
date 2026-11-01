"""Boundary admission and explicit pending accounting without GPU work."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from panda_qb40 import plan,runner

class FakeCampaign:
 def __init__(self,root,remaining):
  self.root=Path(root);self.server='s1';self.stop_requested=False;self.config=plan.default_config('s1')
  self.state=dict(runs=[],groups=[],timing={},donors={});self.left=remaining;self.validated=[];self.registered=[]
 def elapsed_seconds(self):return 40*3600-self.left
 def persist(self):pass
 def admission(self,catalog):pass
 def register_external_donor(self):pass
 def completed(self,c):return next((r for r in self.state['runs'] if r['run_id']==c['run_id'] and r['status']=='COMPLETED'),None)
 def register_donor(self,case):self.registered.append(case)
 def check(self):pass
 def validate_preflight(self,cfg):self.validated.append(cfg['run_id'])
 def measured_timing(self,t):return False
 def timing_key(self,c):return c['case']

def complete(camp,cfg,budget):
 camp.state['runs'].append(dict(run_id=cfg['run_id'],case=cfg['case'],replica=cfg['replica'],step=cfg['planned_horizon'],status='COMPLETED'))

class RunnerTests(unittest.TestCase):
 def test_insufficient_budget_retains_every_unstarted_job(self):
  with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case') as execute:
   camp=FakeCampaign(tmp,600)
   result=runner.run(camp,catalog='fixture')
   self.assertEqual(result['status'],'BUDGET_COMPLETE_WITH_PENDING')
   self.assertEqual(len(result['pending_groups']),12)
   execute.assert_not_called();self.assertFalse(camp.validated)
 def test_full_plan_visits_exact_order_and_selects_each_new_donor(self):
  with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=complete):
   camp=FakeCampaign(tmp,40*3600)
   result=runner.run(camp,catalog='fixture')
   self.assertEqual(result['status'],'FINISHED_REGISTERED_QUEUE')
   self.assertEqual([(r['case'],r['replica']) for r in result['runs']],plan.G.RUNS['s1'] and list(plan.G.RUNS['s1']))
   self.assertEqual(camp.registered,['QB5_S1_A200_E100','QB5_S1_A400_E030'])
 def test_completed_stage1_is_selected_again_on_recovery_without_retraining(self):
  with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case') as execute:
   camp=FakeCampaign(tmp,600);cfg=plan.resolve_case('s1','QB5_S1_A200_E100');complete(camp,cfg,None)
   result=runner.run(camp,catalog='fixture')
   execute.assert_not_called();self.assertEqual(camp.registered,['QB5_S1_A200_E100'])
   self.assertEqual(len(result['pending_groups']),11)

if __name__=='__main__':unittest.main()
