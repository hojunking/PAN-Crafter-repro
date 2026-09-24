"""Controller transitions use fake workers and temporary state, never a GPU/job."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from maina_hqnr import controller as ctl
from maina_hqnr.common import atomic_json,read_json,object_sha,sha256,camp,run_dir,RuntimePaused
from maina_hqnr.plan import make_case,CAMPAIGN_ID


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=camp(self.root,'s4')
        self.bindings={'server':'s4','campaign_id':CAMPAIGN_ID,'fixture':'fixed F1'}
        self.source={'files':{'fixture':'original'},'git_release':'fixed'}
        atomic_json(self.folder/'runtime_bindings.json',self.bindings)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for name in ('maina_hqnr.assets.validate_bindings','maina_hqnr.resources.ensure_space',
                     'maina_hqnr.handoff.ensure_gpu_idle','maina_hqnr.controller._complete',
                     'maina_hqnr.controller._queue_results','maina_hqnr.reporting.build_report'):
            self.stack.enter_context(patch(name))
        self.stack.enter_context(patch.object(ctl,'source_identity',return_value=self.source))
        self.stack.enter_context(patch.object(ctl,'build_config',side_effect=self.config))
        self.stack.enter_context(patch('maina_hqnr.common.selected_gpu_uuid',return_value='GPU-fixture'))

    def config(self,case,root,bindings,attempt):
        return {'work_dir':str(run_dir(case,root,attempt)), 'maina_hqnr':{
            'case':case,'attempt':attempt,'source_identity':self.source,
            'binding_sha256':object_sha(bindings)}}

    def state(self,position=0):
        value=ctl.initial_state('s4',self.bindings,self.source);value['position']=position
        if position:
            base=make_case('s4',0,'BASE')
            value['runs'][base['run_id']]={'status':'COMPLETE','attempt':0,'failures':[]}
        atomic_json(self.folder/'state.json',value)
        return value

    def test_explicit_activation_and_protected_servers(self):
        with self.assertRaises(PermissionError): ctl.run(self.root,'s4')
        with self.assertRaises(ValueError): ctl.run(self.root,'s1',activated=True)

    def test_source_change_pauses_before_worker(self):
        state=self.state();state['source_identity']={'different':'source'}
        atomic_json(self.folder/'state.json',state)
        calls=[]
        with self.assertRaises(RuntimePaused):
            ctl.run(self.root,'s4',activated=True,worker=lambda *a,**kw:calls.append(a))
        self.assertEqual(calls,[])
        self.assertEqual(read_json(self.folder/'state.json')['status'],'PAUSED_SOURCE_CHANGED')

    def test_binding_and_active_case_tampering_rejected(self):
        state=self.state();state['locked_bindings_sha256']='bad'
        with self.assertRaises(ValueError): ctl.validate_state(state,'s4',self.bindings)
        state=self.state();state.update(active_run_id=make_case('s4',0,'AL05')['run_id'])
        with self.assertRaises(ValueError): ctl.validate_state(state,'s4',self.bindings)

    def test_base_technical_failure_pauses_without_variants(self):
        calls=[]
        def worker(phase,config,case,resume=False):
            calls.append((phase,case['code'],case['seed']));return 74
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=worker),74)
        state=read_json(self.folder/'state.json')
        self.assertEqual(state['status'],'PAUSED_BASE_TECHNICAL')
        self.assertEqual(calls,[('train','BASE',73101)])
        self.assertEqual(state['position'],0)

    def test_variant_has_at_most_two_automatic_retries_same_seed(self):
        self.state(position=1);calls=[]
        def worker(phase,config,case,resume=False):
            calls.append((phase,Path(config).parent.name,case['seed'],case['run_id']));return 74
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=worker),74)
        self.assertEqual([r[1] for r in calls],['attempt000','attempt001','attempt002'])
        self.assertEqual(len({r[2:] for r in calls}),1)
        state=read_json(self.folder/'state.json')
        self.assertEqual(state['status'],'TECHNICAL_FAILED');self.assertEqual(state['position'],1)
        self.assertTrue(all((run_dir(make_case('s4',0,'AL05'),self.root,i)/'meta/attempt_failure.json').is_file() for i in range(3)))

    def test_safe_pause_is_not_completion_or_retry(self):
        calls=[]
        def worker(phase,config,case,resume=False):calls.append(phase);return 75
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=worker),75)
        state=read_json(self.folder/'state.json')
        self.assertEqual(state['status'],'PAUSED_SAFE')
        self.assertEqual(state['runs'][state['active_run_id']]['failures'],[])
        self.assertEqual(calls,['train'])

    def test_operator_stop_prevents_new_admission(self):
        atomic_json(self.folder/'STOP_AFTER_RUN',{'operator':True});calls=[]
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=lambda *a,**kw:calls.append(a)),75)
        self.assertEqual(calls,[])

    def test_postrun_failure_retries_evaluation_not_training(self):
        calls=[]
        def first(phase,config,case,resume=False):calls.append(phase);return 0 if phase=='train' else 74
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=first),74)
        state=read_json(self.folder/'state.json')
        self.assertEqual(state['status'],'PENDING_EVAL_NOT_COMPARABLE')
        with self.assertRaises(ValueError):ctl.retry(self.root,'s4','train')
        ctl.retry(self.root,'s4','postrun')
        def second(phase,config,case,resume=False):
            calls.append(phase);atomic_json(self.folder/'STOP_AFTER_RUN',{'operator':True});return 0
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=second),75)
        self.assertEqual(calls,['train','postrun','postrun'])
        state=read_json(self.folder/'state.json')
        self.assertEqual(state['runs'][make_case('s4',0,'BASE')['run_id']]['status'],'COMPLETE')

    def test_variant_divergence_recorded_not_reseeded(self):
        self.state(position=1);calls=[]
        def worker(phase,config,case,resume=False):
            calls.append(case['code'])
            return 3 if case['code']=='AL05' else 75
        self.assertEqual(ctl.run(self.root,'s4',activated=True,worker=worker),75)
        state=read_json(self.folder/'state.json')
        self.assertEqual(calls,['AL05','AL15'])
        self.assertEqual(state['runs'][make_case('s4',0,'AL05')['run_id']]['status'],'DIVERGED')

    def test_resume_requires_bound_source_and_fullstate_sha(self):
        case=make_case('s4',0,'BASE');work=run_dir(case,self.root,0);cfg=self.config(case,self.root,self.bindings,0)
        self.assertFalse(ctl.resume_available(work,cfg))
        (work/'last').mkdir(parents=True);state=work/'last/training_state.pt';state.write_bytes(b'fixture')
        ident=dict(full_state=True,config_sha256=object_sha(cfg),source_identity=self.source,
            bindings_sha256=object_sha(self.bindings),training_state_sha256=sha256(state))
        atomic_json(work/'last/identity.json',ident)
        self.assertTrue(ctl.resume_available(work,cfg))
        state.write_bytes(b'changed')
        with self.assertRaises(ctl.ResumeUnusable):ctl.resume_available(work,cfg)
        ident['source_identity']={'other':'source'};atomic_json(work/'last/identity.json',ident)
        with self.assertRaises(ValueError):ctl.resume_available(work,cfg)

    def test_launch_admission_scope_and_preflight(self):
        case=make_case('s4',0,'BASE');config=run_dir(case,self.root,0)/'config.json'
        atomic_json(config,self.config(case,self.root,self.bindings,0))
        state=self.state();state.update(active_run_id=case['run_id'],active_case_spec_sha256=case['case_spec_sha256'],status='ADMITTED')
        state['runs'][case['run_id']]={'attempt':0};atomic_json(self.folder/'state.json',state)
        atomic_json(self.folder/'preflight.json',{'status':'PASSED','source_identity':self.source,
            'bindings_sha256':object_sha(self.bindings),'gpu_uuid':'GPU-fixture'})
        ctl.authorize_train(self.root,case,config)
        with self.assertRaises(PermissionError):ctl.authorize_train(self.root,case,config.parent.parent/'config.json')
        for changed in ({'bindings_sha256':'wrong'},{'gpu_uuid':'GPU-other'}):
            receipt=dict(status='PASSED',source_identity=self.source,bindings_sha256=object_sha(self.bindings),gpu_uuid='GPU-fixture')
            receipt.update(changed);atomic_json(self.folder/'preflight.json',receipt)
            with self.assertRaises(PermissionError):ctl.authorize_train(self.root,case,config)
        atomic_json(self.folder/'preflight.json',{'status':'CPU_TEST_ONLY','source_identity':self.source})
        with self.assertRaises(PermissionError):ctl.authorize_train(self.root,case,config)


if __name__=='__main__':unittest.main()
