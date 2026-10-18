"""Controller/file/reporting integration with explicit CPU-only fake training.

These tests do not claim measured model quality or GPU throughput. Scientific
engine, native evaluator and checkpoint tests live in their separate suites.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import torch
from torch import nn

from panda30r import plan, reporting as R, publication as P
from panda30r.evaluation import model_state_hash
from panda30r.runtime import Campaign
from panda30r.tests.test_reporting import evaluation
from experiment_cost.timing import timing_for_config
from ta2.evaluation import file_sha


class TinyState(nn.Module):
    def __init__(self):
        super().__init__()
        self.aligner=nn.Linear(1,1,bias=False)
        self.student=nn.Linear(1,1,bias=False)
        self.teacher=None
        with torch.no_grad():
            self.aligner.weight.zero_(); self.student.weight.zero_()


def controller(root):
    campaign=Campaign.__new__(Campaign)
    campaign.config=plan.default_config('s1')
    campaign.root=Path(root); campaign.server='s1'; campaign.device='cpu'
    campaign.state_path=campaign.root/'campaign_state.json'
    campaign.state={'started_unix':time.time(),'runs':[],'groups':[],'donors':{},
                    'timing':{'stage2_gt':{'update_p90':.1,'eval_seconds':.1}}}
    campaign.initial_elapsed=0.; campaign.tick=time.monotonic(); campaign.stop_requested=False
    campaign.code={'fixture':'c'*64}; campaign.evaluator_pin={'fixture':True}
    campaign.verify_code=lambda:None
    campaign.release=lambda:None
    campaign.calls={'trace':0,'evaluate':0}

    def build(cfg):
        return SimpleNamespace(config=deepcopy(cfg),updates=0,poisoned=False,stage=TinyState()),object()
    def trace(trainer,stream):
        campaign.calls['trace']+=1; trainer.updates+=1
        with torch.no_grad(): trainer.stage.student.weight.fill_(trainer.updates)
        return {'update':trainer.updates},.25
    def evaluate(trainer):
        campaign.calls['evaluate']+=1
        value=evaluation(trainer.config,trainer.updates,.9592,2.1)
        value['elapsed_seconds']=.5
        for key in ('state_before','rr_state','fr_state','state_after'):
            value[key]['model_state_sha256']=model_state_hash(trainer.stage)
        return value
    def save(trainer,stream,path,cfg,ev=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('SYNTHETIC_CHECKPOINT_'+str(trainer.updates)).encode())
        artifact={'checkpoint_path':str(path),'file_sha256':file_sha(path),
            'model_state_sha256':model_state_hash(trainer.stage),
            'state_hash':model_state_hash(trainer.stage),'completed_update':trainer.updates,
            'identity':{} if ev is None else deepcopy(ev['state_before']),
            'weights_saved':True,'checkpoint_verified':True}
        R.atomic_json(path.with_suffix('.json'),{'artifact':artifact,'evaluation':ev,'config':cfg,
                      'code_pin':campaign.code,'evaluator_pin':campaign.evaluator_pin})
        return artifact
    def restore(artifact,trainer,stream,cfg):
        trainer.updates=artifact['completed_update']
        with torch.no_grad(): trainer.stage.student.weight.fill_(trainer.updates)
        return trainer,stream
    campaign.build=build; campaign.trace=trace; campaign.evaluate=evaluate
    campaign._save=save; campaign._restore=restore
    return campaign


class RuntimeAdapterTests(unittest.TestCase):
    def test_completed_evaluation_republication_is_idempotent_after_resume(self):
        with tempfile.TemporaryDirectory() as root:
            campaign=controller(root)
            cfg=plan.resolve_case('s1','M_G')
            cfg.update(planned_horizon=2,evaluation_steps=[2])
            campaign.run_case(cfg)
            self.assertEqual(campaign.calls,{'trace':2,'evaluate':1})
            source=P.scan_campaign(root)
            evaluations=[r for r in source['records'] if r['record_type']=='EVAL_SUMMARY']
            self.assertEqual(len(evaluations),1)
            rid=evaluations[0]['record_id']
            original=(Path(root)/'outbox'/f'{rid}.json').read_bytes()
            # Simulate process death after evaluated checkpoint/outbox but before
            # the controller's terminal status was committed.
            campaign.state['runs'][0]['status']='STOPPED'
            campaign.run_case(cfg)
            self.assertEqual(campaign.calls,{'trace':2,'evaluate':1})
            self.assertEqual((Path(root)/'outbox'/f'{rid}.json').read_bytes(),original)
            self.assertFalse(P.project_campaign(root,'s1')['conflicts'])

    def test_contiguous_local_timing_trace_can_fill_selected_step_train_hours(self):
        with tempfile.TemporaryDirectory() as root:
            campaign=controller(root)
            cfg=plan.resolve_case('s1','M_G')
            cfg.update(planned_horizon=3,evaluation_steps=[3])
            campaign.run_case(cfg)
            directory=Path(campaign.state['runs'][0]['directory'])
            rows=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[1,2,3])
            persisted=json.loads((directory/'config.json').read_text())
            value=timing_for_config(persisted,3,root,history_roots=[],cost_root=root)
            self.assertAlmostEqual(value['sheet_fields']['Train(h)'],.75/3600)
            self.assertAlmostEqual(value['sheet_fields']['Eval(h)'],.5/3600)


if __name__=='__main__':
    unittest.main()
