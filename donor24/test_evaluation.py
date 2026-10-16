from copy import deepcopy
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from directshift48.test_evaluation import SyntheticSources as BaseSources, metrics
from directshift48.test_method import batch
from donor30.test_method import model
from directshift48.training import DirectStage1Trainer
from . import plan, evaluation as E

torch.set_num_threads(1)


class Sources(BaseSources):
    sensor,bands,peak='WV3',8,2047.
    def batch(self,split,rows,*,device='cpu'):
        b=batch(sensor='WV3',batch_size=len(rows),resolution='FR' if split=='fr' else 'RR',
                split={'train':'train','val':'validation','rr':'test','fr':'test'}[split])
        return replace(b,lineage=replace(b.lineage,sample_ids=tuple(f'WV3:{split}:{i}' for i in rows)))


def engine(server):
    stage=model('WV3',lambda_abs=3.,lambda_eps=.003)
    t=DirectStage1Trainer(stage,torch.optim.AdamW(stage.parameters(),lr=1e-4),gradient_log_interval=100)
    t.updates=1;t.config=plan.resolve_case(plan.default_config(server),'A0');t.evaluator_pin={}
    t.config['baseline_resolved']['teacher_architecture']={'bands':8,'width':8,'depth':[1,1,1]}
    return t


class EvaluationTests(unittest.TestCase):
    def test_real_saved_donor_cpu_reload(self):
        from donor30.artifacts import save_candidate, read_selection
        from types import SimpleNamespace
        t=engine('s3');s=Sources();f=metrics(hqnr=.960,ergas=2.03);rr=f['rr']
        def rr8(p,a):
            v=rr(p,a);v['q8']=v.pop('q4');return v
        f['rr']=rr8
        ev=E.official_evaluate(t,s,metric_functions=f,evaluator_revision='TEST')
        with tempfile.TemporaryDirectory() as root:
            result=save_candidate(t,SimpleNamespace(state_dict=lambda:{'test_stream':1}),ev,root,t.config,
                reload_factory=lambda cfg:engine('s3'))
            self.assertTrue(result['artifact']['weights_saved'])
            self.assertTrue(Path(result['artifact']['checkpoint_path']).is_file())
            selection=read_selection(root)
            self.assertEqual(selection['best_hqnr']['state_hash'],E.model_state_hash(t.stage))
            self.assertIsNotNone(selection['best_gate_ergas'])

    def test_both_servers_use_wv3_q8_original_references(self):
        for server in ('s1','s3'):
            t,s=engine(server),Sources()
            with torch.no_grad():t.stage.aligner.fc2.bias.copy_(torch.tensor([.5,-.25]))
            f=metrics(hqnr=.959,ergas=2.1);rr=f['rr']
            seen=[]
            def rr8(raw,arrays):
                self.assertEqual(raw.shape,(8,32,32))
                np.testing.assert_array_equal(arrays['pan'],s.raw('rr',[len(seen)])['pan'][0])
                seen.append(1);v=rr(raw,arrays);v['q8']=v.pop('q4');return v
            f['rr']=rr8
            before=E.model_state_hash(t.stage)
            rec=E.official_evaluate(t,s,metric_functions=f,evaluator_revision='TEST')
            self.assertEqual(E.model_state_hash(t.stage),before)
            self.assertEqual(len(seen),20)
            self.assertTrue(rec['qualification']['metrics_complete'])
            self.assertIn('Q8',rec['rr']['metrics'])
            self.assertFalse(rec['qualification']['joint_pass'])
            self.assertEqual(rec['state_before']['dataset'],'WV3')
            self.assertEqual(rec['state_before']['server'],server)
            bad=deepcopy(rec);bad['fr_state']['completed_update']=2
            q=E.qualify_evaluation(bad,server=server,expected_rr_ids=E.expected_ids(s,'rr'),expected_fr_ids=E.expected_ids(s,'fr'))
            self.assertFalse(q['same_state_verified'])

    def test_no_production_metric_injection_or_stage2(self):
        t=engine('s3')
        with self.assertRaisesRegex(ValueError,'synthetic TEST'):
            E.official_evaluate(t,Sources(),metric_functions=metrics(),evaluator_revision='PROD')
        t.config['phase']='stage2'
        with self.assertRaisesRegex(ValueError,'Stage 1 only'):E.configured_pipeline(t,Sources())


if __name__=='__main__':unittest.main()
