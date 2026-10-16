from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import plan, runtime as R
from .test_evaluation import Sources
from donor30.test_runtime import eval_fixture


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.stack=ExitStack()
        self.stack.enter_context(patch.object(R,'evaluator_identity',return_value={}))
        self.stack.enter_context(patch.object(R,'code_pin',return_value={'test':'a'}))
        self.stack.enter_context(patch.object(R,'emit_status'))
        self.stack.enter_context(patch.object(R,'emit_evaluation'))
        self.c=R.Campaign(plan.default_config(),output=self.tmp.name,device='cpu')
        self.c.state['timing']={'donor_update_seconds':.16,'evaluation_seconds':20.,
                               'checkpoint_seconds_estimate':10.,'probe_seconds':10.}

    def tearDown(self):
        self.stack.close();self.tmp.cleanup()

    def test_production_s3_build_is_eight_band_masked_p0(self):
        from donor30.masking import MaskedDonorStage1
        from donor24.evaluation import configured_pipeline
        cfg=plan.resolve_case(plan.default_config('s3'),'B4')
        t=R.build_trainer(cfg,Sources())
        self.assertIsInstance(t.stage,MaskedDonorStage1)
        self.assertEqual(t.stage.teacher.spec.bands,8)
        self.assertEqual(t.stage.teacher.spec.input_layout,'P0')
        self.assertEqual(t.optimizer.param_groups[0]['lr'],5e-6)
        self.assertTrue(configured_pipeline(t,Sources())['loss_mask'])

    def test_seven_cases_reserved_and_no_stage2(self):
        calls=[]
        def run(cfg,**kw):
            calls.append((cfg,kw))
            self.c.state['runs'].append(dict(run_id=cfg['run_id'],attempt=1,status='COMPLETED',phase='donor'))
        winner={'case':'A1'}
        with patch.object(self.c,'admission'),patch.object(self.c,'preflight'),\
             patch.object(self.c,'recover_candidates'),patch.object(self.c,'run_case',side_effect=run),\
             patch.object(R.plan,'choose_completed',return_value=winner),\
             patch.object(self.c,'prepare_local_candidate'):
            result=self.c.run(catalog_path='unused')
        self.assertEqual(len(calls),7)
        self.assertTrue(all(kw.get('reserve_seconds',0)>0 for _,kw in calls[:6]))
        self.assertEqual(calls[-1][0]['case'],'A1')
        self.assertEqual(calls[-1][0]['replica'],'R2')
        self.assertTrue(all(c['phase']=='donor' for c,_ in calls))
        self.assertTrue(result['confirmation_completed'])
        self.assertIn('SHARED_DONOR_PENDING',result['status'])

    def test_budget_skip_without_model_build(self):
        cfg=plan.resolve_case(self.c.config,'A0')
        with patch.object(self.c,'elapsed',return_value=80000),patch.object(R,'build_trainer') as build:
            self.c.run_case(cfg,reserve_seconds=self.c.estimate())
        self.assertFalse(build.called)
        self.assertEqual(self.c.state['runs'][-1]['status'],'SKIPPED_BUDGET')

    def test_restart_deadline_and_attempt_not_overwritten(self):
        started=self.c.state['started_unix'];self.c.persist()
        other=R.Campaign(plan.default_config(),output=self.tmp.name,device='cpu')
        self.assertEqual(other.state['started_unix'],started)
        cfg=plan.resolve_case(self.c.config,'A0')
        d=self.c.root/'runs'/cfg['run_id']/'attempt_01';d.mkdir(parents=True)
        (d/'config.json').write_text(json.dumps(cfg))
        self.c.state['runs'].append(dict(run_id=cfg['run_id'],attempt=1,status='RUNNING',directory=str(d),phase='donor'))
        attempt=self.c._attempt(cfg)
        self.assertEqual(attempt['attempt'],2)
        self.assertTrue((d/'config.json').exists())
        self.assertEqual(self.c.state['runs'][0]['status'],'INTERRUPTED_PROCESS_EXIT')

    def test_small_controller_run_and_failure_are_explicit(self):
        for fail in (False,True):
            with self.subTest(fail=fail),ExitStack() as stack:
                cfg=plan.resolve_case(self.c.config,'A0' if not fail else 'A1')
                cfg.update(planned_horizon=2,evaluation_steps=[1,2])
                stack.enter_context(patch.object(R.plan,'validate_case'))
                trainer=SimpleNamespace(updates=0,poisoned=False,config=cfg)
                stack.enter_context(patch.object(R,'build_trainer',return_value=trainer))
                stack.enter_context(patch.object(R,'RRStream'))
                stack.enter_context(patch.object(self.c,'cpu_trainer_to_gpu',side_effect=lambda t:t))
                stack.enter_context(patch.object(self.c,'verify_code'))
                stack.enter_context(patch.object(self.c,'release'))
                def trace(t,s,d):
                    if fail:raise ValueError('test injected loss failure')
                    t.updates+=1;return .1
                stack.enter_context(patch.object(self.c,'trace',side_effect=trace))
                stack.enter_context(patch.object(self.c,'evaluate',side_effect=lambda t:eval_fixture(t.updates)))
                save=stack.enter_context(patch('donor30.artifacts.save_candidate',return_value=None))
                resume=stack.enter_context(patch('donor30.artifacts.save_resume'))
                self.c.run_case(cfg)
                self.assertEqual(self.c.state['runs'][-1]['status'],'FAILED' if fail else 'COMPLETED')
                self.assertEqual(save.call_count,0 if fail else 2)
                self.assertTrue(resume.called)


if __name__=='__main__':unittest.main()
