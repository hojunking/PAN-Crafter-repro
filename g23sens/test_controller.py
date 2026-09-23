from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g23sens import common as C,controller as R,plan as P


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.server='s4';self.folder=C.camp(self.root,self.server)
        self.bindings=dict(server=self.server,fixture=True)
        C.atomic_json(self.folder/'runtime_bindings.json',self.bindings)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for target in ('g23sens.assets.validate_bindings','g23sens.controller.verify_sources',
                       'g23sens.resources.ensure_space','g23sens.handoff.ensure_gpu_idle',
                       'g23sens.controller._complete','g23sens.controller._queue_results',
                       'g23sens.reporting.build_report'):
            self.stack.enter_context(patch(target))
        self.stack.enter_context(patch('g23sens.controller.build_config',side_effect=lambda c,r,b,a:
            dict(work_dir=str(C.run_dir(c,r,a)),g23sens=dict(case=c,attempt=a))))
        self.stack.enter_context(patch('g23sens.controller.resume_available',
            side_effect=lambda w,c:(w/'last/identity.json').exists()))

    def invoke(self,worker):return R.run(self.root,self.server,activated=True,no_upload=True,worker=worker)

    def state(self):return C.read(self.folder/'state.json')

    def stop(self):C.atomic_json(self.folder/'STOP_AFTER_RUN',dict(operator=True))

    def test_inactive_no_admission(self):
        with self.assertRaises(PermissionError):R.run(self.root,self.server)
        self.assertFalse((self.folder/'state.json').exists())

    def test_stop_after_run_still_evaluates_before_advancing(self):
        calls=[]
        def worker(phase,config,case,resume=False):
            calls.append(phase)
            if phase=='train':self.stop()
            return 0
        self.assertEqual(self.invoke(worker),75)
        self.assertEqual(calls,['train','postrun'])
        self.assertEqual(self.state()['position'],1)
        self.assertEqual(self.state()['runs'][P.make_case('s4',0,'BASE')['run_id']]['status'],'COMPLETE')

    def test_evaluation_retry_does_not_retrain(self):
        calls=[]
        def worker(phase,config,case,resume=False):
            calls.append(phase)
            if phase=='postrun' and calls.count('postrun')==1:return 74
            if phase=='postrun':self.stop()
            return 0
        self.assertEqual(self.invoke(worker),75)
        self.assertEqual(calls,['train','postrun','postrun'])

    def test_evaluation_exhaustion_remains_paused_across_restart(self):
        calls=[]
        def worker(phase,config,case,resume=False):
            calls.append(phase);return 74 if phase=='postrun' else 0
        self.assertEqual(self.invoke(worker),74)
        self.assertEqual(calls,['train','postrun','postrun','postrun'])
        self.assertEqual(self.invoke(worker),3)
        self.assertEqual(calls,['train','postrun','postrun','postrun'])
        self.assertEqual(list(self.state()['runs'].values())[0]['status'],'EVALUATION_FAILED')

    def test_baseline_divergence_pauses_and_restart_does_not_reroll(self):
        calls=[]
        def worker(*args,**kwargs):calls.append(args[0]);return 3
        self.assertEqual(self.invoke(worker),3);self.assertEqual(self.invoke(worker),3)
        self.assertEqual(calls,['train']);self.assertEqual(self.state()['position'],0)

    def test_variant_divergence_advances_without_fake_completion(self):
        calls=[]
        def worker(phase,config,case,resume=False):
            calls.append((case['case_id'],phase))
            if case['case_id']!='BASE':self.stop();return 3
            return 0
        self.assertEqual(self.invoke(worker),75)
        case=P.cycle_cases('s4',0)[1]
        self.assertEqual(self.state()['runs'][case['run_id']]['status'],'DIVERGED')
        self.assertEqual(self.state()['position'],2)
        self.assertEqual(calls,[('BASE','train'),('BASE','postrun'),('AL05','train')])

    def test_safe_resume_uses_same_attempt_and_cursor(self):
        calls=[]
        def pause(phase,config,case,resume=False):
            C.atomic_json(Path(config).parent/'last/identity.json',dict(fake_for_test=True));return 75
        self.assertEqual(self.invoke(pause),75)
        def worker(phase,config,case,resume=False):
            calls.append((phase,resume))
            if phase=='postrun':self.stop()
            return 0
        self.assertEqual(self.invoke(worker),75)
        self.assertEqual(calls,[('train',True),('postrun',False)])
        self.assertEqual(list(self.state()['runs'].values())[0]['attempt'],0)

    def test_missing_resume_keeps_old_attempt_and_bounded_technical_retries(self):
        def worker(phase,config,case,resume=False):return 74
        self.assertEqual(self.invoke(worker),74)
        item=list(self.state()['runs'].values())[0]
        self.assertEqual(item['attempt'],2);self.assertEqual(len(item['failures']),3)
        for attempt in range(3):self.assertTrue((C.run_dir(P.make_case('s4',0,'BASE'),self.root,attempt)/'config.json').exists())

    def test_recipe_integrity_error_not_retried_as_infrastructure(self):
        calls=[]
        def worker(*args,**kwargs):calls.append(1);return 65
        self.assertEqual(self.invoke(worker),65);self.assertEqual(calls,[1])

    def test_no_other_server_state_read_or_wait(self):
        C.atomic_json(C.camp(self.root,'s5')/'state.json',dict(status='BROKEN'))
        self.stop()
        self.assertEqual(self.invoke(lambda *a,**k:self.fail('Unexpected admission')),75)
        self.assertEqual(C.read(C.camp(self.root,'s5')/'state.json'),dict(status='BROKEN'))

    def test_bindings_change_refused(self):
        state=R.initial_state('s4',self.bindings);state['locked_bindings_sha256']='changed'
        C.atomic_json(self.folder/'state.json',state)
        with self.assertRaises(ValueError):self.invoke(lambda *a,**k:0)


if __name__=='__main__':unittest.main()
