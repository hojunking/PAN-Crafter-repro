"""Registry, real optimizer state, and execution controls; never official metrics."""
import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from panda_qb40 import registry as G,plan,reporting as R

class ContractTests(unittest.TestCase):
 def test_plan_is_executable_without_recipe_drift(self):
  from panda_qb40.engine import ScheduleTrainingSpec
  from panda_qb40.contract import validate_loss_pair_config
  from panda_qb40.runtime import Campaign
  keys={}
  for server in G.RUNS:
   self.assertEqual(len(plan.groups(server)),12 if server=='s1' else 19)
   for case,rep in G.RUNS[server]:
    row=G.job(server,case,rep);cfg=plan.resolve_case(server,case,rep)
    R._config(cfg);validate_loss_pair_config(cfg)
    self.assertEqual(cfg['planned_horizon'],row['updates'])
    self.assertEqual(cfg['seed'],row['seed'])
    self.assertEqual(cfg['evaluation_steps'],list(range(1000,row['updates']+1,1000)))
    self.assertEqual(cfg['training_spec']['effective_batch_size'],row['batch'])
    self.assertEqual(cfg['training_spec']['microbatch_size'],12)
    self.assertEqual(cfg['epsilon_radius_rr'],3.)
    self.assertEqual(R.destination(server)['sheet_title'],'QB-ablations')
    self.assertIn('B'+str(row['batch']),R.approach_description(cfg))
    if cfg['phase']=='stage2':
     t=ScheduleTrainingSpec(**cfg['training_spec'])
     self.assertEqual(t.rate(t.total_updates),0.)
     self.assertEqual(t.rate(t.warmup_updates),row['lr_student'])
     self.assertEqual(t.effective_batch_size//t.microbatch_size,row['accumulation'])
     self.assertEqual(cfg['student_architecture']['width'],row['width'])
     keys[row['batch']]=Campaign.timing_key(cfg)
     self.assertEqual(cfg['stage2_spec']['beta'],0.)
    bad=deepcopy(cfg);bad['training_spec']['effective_batch_size']+=12
    with self.assertRaises(ValueError):R._config(bad)
  self.assertNotEqual(keys[48],keys[96])

 def test_no_predecessor_gate(self):
  from panda_qb40.orchestration import predecessor_receipt
  for s in G.RUNS:self.assertFalse(predecessor_receipt('/missing',s)['predecessor_queue_gate'])

 def test_s3_dedicated_source_dispatch(self):
  code='''
from panda_qb_deepen.adapter import install
install()
from panda_qb40 import plan,registry as G,reporting
from panda_s1loss9 import plan as source
assert source.resolve_case('s3','S1_A200_E030')['seed']==261016301
for c,r in G.RUNS['s3']:
 cfg=plan.resolve_case('s3',c,r);reporting._config(cfg)
 assert cfg['dataset']=='QB'
 assert cfg['seed']==G.job('s3',c,r)['seed']
'''
  result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)

 def test_source_paths_preserve_origin(self):
  from panda_qb40.external import receipt_path,verify_provenance
  for s in G.RUNS:
   source=G.SOURCES[s];step=44000 if s=='s1' else 39000
   cfg=dict(server=s,dataset='QB',campaign_id=source['campaign'],case='S1_A200_E030',seed=source['seed'],
    phase='donor',planned_horizon=50000,lambda_abs=2.,lambda_epsilon=.03,run_id='fixture')
   path=receipt_path(s,'/fixture').parents[2]/'runs/fixture/attempt_01/models'/f'step_{step:07d}.pt'
   receipt=dict(server=s,campaign_id=source['campaign'],selected=dict(source_config=cfg,completed_update=step,checkpoint_path=str(path)))
   verify_provenance(receipt,s,'/fixture')
   bad=deepcopy(receipt);bad['selected']['source_config']['seed']+=1
   with self.assertRaises(ValueError):verify_provenance(bad,s,'/fixture')

 def test_empty_projection_target_and_namespace(self):
  from panda_qb40.best_publication import project
  for s in G.RUNS:
   with TemporaryDirectory() as tmp:
    v=project(tmp,s)
    self.assertFalse(v['conflicts'])
    self.assertEqual(v['publication_thresholds']['hqnr_min'],.9315)
    self.assertEqual(v['publication_thresholds']['ergas_max'],3.5)
  self.assertEqual(R.RECORD_PREFIX,'PQ40-')

if __name__=='__main__':unittest.main()
