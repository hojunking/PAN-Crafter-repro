"""Controller glue tests: disposable tiny CPU model and offline sheet outbox."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from panda_method.test_stages import batch
from .evaluation import official_evaluation
from .plan import resolve_case
from .reporting import INDEX
from .runtime import Campaign, numeric_chunks, repeat_statistics
from ta2.evaluation import atomic_json
from .plan import CAMPAIGN_ID, PARENT_CAMPAIGN_ID, METHOD_CONTRACT
from .test_evaluation import SyntheticSources, metrics
from .test_plan import baseline
from .test_training import config as engine_config, model
from .training import Trainer


class Stream:
    def __init__(self, branch): self.branch=branch
    def __call__(self): return [batch('RR' if self.branch=='rr' else 'FR')]
    def state(self): return {'synthetic':True}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        b=baseline(); b['teacher_architecture']['bands']=4
        b.update(artifact_reserve_seconds=1,probe_reserve_seconds=1)
        self.c=Campaign(self.temp.name,'s2',b,device='cpu')
        self.c.sources=SyntheticSources(); self.c.evaluator_pin={}; self.c.started_at=time.time()
        self.c.timing={'eval_seconds':1,'update_p90_seconds':.01,'artifact_seconds':1,'probe_seconds':1}
        self.case=resolve_case('s2',baseline=b)
        self.case.update(planned_horizon=6,evaluation_steps=[2,6])

    def test_status_and_metrics_have_exact_numeric_fixed64_types(self):
        self.c.emit(self.case,'PLANNED')
        record=self.c.outbox.pending()[0]
        self.assertEqual(record['row'][INDEX['Repeat']],1)
        self.assertEqual(record['row'][INDEX['Sort_case']],0)
        self.assertEqual(record['row'][INDEX['Readback']],'PENDING')
        self.c.emit({},'BLOCKED_DATA_PROTOCOL',extra={'reason':'synthetic fixture'})
        self.assertEqual(len(self.c.outbox.pending()),2)

    def test_nonpassing_run_emits_all_scenes_and_never_saves_weights(self):
        def evaluate(t,c,s,**kwargs):
            kwargs.update(evaluator_revision='TEST',evaluator_pin={},metric_functions=metrics(hqnr=.9))
            return official_evaluation(t,c,s,**kwargs)
        with patch('teacher48.runtime.build_trainer',side_effect=lambda *_:Trainer(model(),engine_config())), \
             patch('teacher48.runtime.streams',return_value={s:Stream(s) for s in ('rr','fr')}), \
             patch('teacher48.runtime.official_evaluation',side_effect=evaluate):
            run=self.c.run_case(self.case)
        self.assertEqual(run['status'],'COMPLETE',run.get('reason'))
        self.assertEqual(len(run['evaluations']),2)
        self.assertFalse(list(Path(self.temp.name).rglob('*.pt')))
        self.assertFalse(list(Path(self.temp.name).rglob('*.npy')))
        records=self.c.outbox.pending()
        self.assertEqual(sum('SCENE:' in r['identity']['eval_scope'] for r in records),80)
        rows=[r['row'] for r in records if r['identity']['selector']=='HQNR_MAX_OBSERVED_NUMERIC']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][INDEX['RMSE↓']],11.)
        self.assertAlmostEqual(rows[0][INDEX['CC↑']],.99)
        self.assertAlmostEqual(rows[0][INDEX['JQM↑']],.85)
        self.assertFalse(self.c.summarize()['representative_available'])

    def test_failed_evaluation_keeps_completed_scenes_and_error(self):
        calls=[0]
        funcs=metrics(); original=funcs['rr']
        def fail(p,a):
            calls[0]+=1
            if calls[0]==3: raise ValueError('synthetic metric failure')
            return original(p,a)
        funcs['rr']=fail
        def evaluate(t,c,s,**kwargs):
            kwargs.update(evaluator_revision='TEST',evaluator_pin={},metric_functions=funcs)
            return official_evaluation(t,c,s,**kwargs)
        trainer=Trainer(model(),engine_config())
        with patch('teacher48.runtime.official_evaluation',side_effect=evaluate):
            with self.assertRaisesRegex(ValueError,'synthetic metric failure'):
                self.c._evaluate(trainer,self.case,selector='CADENCE',root=self.c.root/'unit')
        records=self.c.outbox.pending()
        self.assertEqual(sum('SCENE:' in r['identity']['eval_scope'] for r in records),2)
        errors=[r for r in records if r['row'][INDEX['Status']]=='EVALUATION_FAILED']
        self.assertEqual(len(errors),1)
        self.assertEqual(errors[0]['row'][INDEX['HQNR↑']],'')
        self.assertTrue((self.c.root/'unit/evaluation_progress.jsonl').is_file())

    def test_large_numeric_diagnostics_split_without_truncation(self):
        obj={'rr':{'rows':[{'n':i,'values':list(range(100))} for i in range(200)]}}
        chunks=list(numeric_chunks(obj))
        self.assertGreater(len(chunks),1)
        self.assertEqual(len(chunks),200)
        self.assertEqual([v['n'] for _,v in chunks],list(range(200)))
        self.assertTrue(all(len(json.dumps(v))<=18000 for _,v in chunks))

    def test_repeat_statistics_include_target_misses_and_incomplete_seeds(self):
        runs=[]
        for seed,h,status in ((1,.93,'COMPLETE'),(2,.91,'COMPLETE'),(3,.92,'FAILED')):
            r=copy.deepcopy(self.case)
            r.update(seed=seed,status=status,completed_update=6 if status=='COMPLETE' else 3,
                evaluations=[{'HQNR':h,'ERGAS':3.4,'SCC':.97,'completed_update':6,'joint_pass':h>.923}])
            runs.append(r)
        stat=repeat_statistics(runs)[0]
        self.assertEqual(stat['attempted_independent_seeds'],3)
        self.assertEqual(stat['complete_seeds'],2)
        self.assertEqual(stat['target_hit_count'],1)
        self.assertAlmostEqual(stat['metrics']['HQNR']['mean'],.92)
        self.assertEqual(len(stat['failed_or_incomplete_seeds']),1)

    def test_existing_attempt_cannot_restart_or_overwrite(self):
        root=self.c.root/'runs'/self.case['run_id']/'attempt_1'
        root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Attempt directory exists'):
            self.c.run_case(self.case)

    def test_resume_keeps_original_clock_and_does_not_reinitialize_completed_runs(self):
        original=time.time()-1234
        clock={'campaign':CAMPAIGN_ID,'server':'s2','dataset':'QB','started_unix':original,
               'method_contract':METHOD_CONTRACT,'parent_campaign_id':PARENT_CAMPAIGN_ID,
               'deadline_unix':original+48*3600}
        atomic_json(self.c.root/'campaign.json',clock)
        with self.assertRaisesRegex(ValueError,'resume-campaign'):
            self.c.launch('not-used')
        with patch.object(self.c,'admission'),patch.object(self.c,'timing_probe'), \
             patch.object(self.c,'_admit_case',return_value=False),patch.object(self.c,'_anchors',return_value=[]):
            self.c.launch('not-used',resume_campaign=True)
        self.assertEqual(self.c.started_at,original)
        self.assertEqual(json.loads((self.c.root/'campaign.json').read_text()),clock)

    def test_failed_run_restarts_as_new_attempt_not_new_seed(self):
        old=copy.deepcopy(self.case); old.update(status='FAILED',attempt=1,evaluations=[],completed_update=3)
        self.c.runs=[old]
        new=copy.deepcopy(self.case)
        with patch.object(self.c,'_estimate',return_value=1000):
            self.assertTrue(self.c._admit_case(new,phase_end=48*3600))
        self.assertEqual(new['attempt'],2)
        self.assertEqual(new['seed'],old['seed'])
        self.assertEqual(new['run_id'],old['run_id'])

    def test_restarting_metadata_publisher_never_reuses_status_result_id(self):
        self.c._sequence=91  # Gaps can remain after failed local serialization.
        first=self.c.emit({},'ADMITTED')
        reopened=Campaign(self.temp.name,'s2',self.c.baseline,device='cpu')
        second=reopened.emit({},'ADMITTED')
        self.assertNotEqual(first,second)
        self.assertEqual(len(reopened.outbox.pending()),2)


if __name__=='__main__': unittest.main()
