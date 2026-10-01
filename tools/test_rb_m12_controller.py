"""Finite real-registry M12 queue tests; all GPU/model/network actions mocked."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_rb_m12 import controller as ctrl
from panda_rb_m12 import preflight
from panda_rb_m12.common import ROOT, atomic_json, read, server_dir
from panda_rb_m12.plan import schedule, training_runs, STEP1, MODES
from panda_rb.deployment import DEFAULT_IMAGE


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = schedule('s1',ROOT); self.runs = training_runs('s1',ROOT)
        self.by_id = {row['run_id']:row for row in self.runs}
        self.training = []; self.native = []; self.curves = []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('panda_rb_m12.common.docker_required'))
        self.stack.enter_context(patch.object(ctrl,'validate_plan',return_value={'status':'PASS'}))
        self.stack.enter_context(patch.object(ctrl,'schedule',return_value=self.queue))
        self.stack.enter_context(patch.object(ctrl,'training_runs',return_value=self.runs))
        self.stack.enter_context(patch.object(ctrl,'run_dir',side_effect=self.wd))
        self.prepare = self.stack.enter_context(patch('panda_rb_m12.preflight.prepare',return_value={'passed':True}))
        self.train = self.stack.enter_context(patch('panda_rb_m12.training.train_run',side_effect=self.do_train))
        self.eval_native = self.stack.enter_context(patch('panda_rb_m12.evaluation.evaluate_native',side_effect=self.do_native))
        self.eval_stress = self.stack.enter_context(patch('panda_rb_m12.evaluation.evaluate_stress',side_effect=self.do_stress))
        self.stack.enter_context(patch('panda_rb_m12.evaluation.validate_artifacts'))
        self.summary = self.stack.enter_context(patch('panda_rb_m12.reporting.summarize',side_effect=self.summarize))

    def wd(self,run_id,*args,**kwargs):
        row = self.by_id[run_id]
        return self.root/'train'/('R'+str(row['repeat']))/row['case_id']

    def do_train(self,run_id,*args,**kwargs):
        self.training.append((run_id,kwargs['resume']))
        atomic_json(self.wd(run_id)/'checkpoints/selection_manifest.json',{'complete':True})
        atomic_json(self.wd(run_id)/'meta/training_status.json',{'status':'TRAINING_COMPLETE'})
        return 0

    def do_native(self,run_id,*args,**kwargs):
        self.native.append(run_id)
        payload = {'payload_sha256':'a'*64}
        atomic_json(self.wd(run_id)/'native/metrics.json',payload)
        return payload

    def do_stress(self,run_id,binding,mode,*args,**kwargs):
        self.curves.append((run_id,mode))
        payload = {'payload_sha256':'b'*64}
        atomic_json(self.wd(run_id)/'stress'/mode/'completion.json',payload)
        return payload

    def summarize(self,*args,**kwargs):
        return dict(completed_students=sum((self.wd(r)/'checkpoints/selection_manifest.json').exists() for r in self.by_id),
                    completed_native=2*sum((self.wd(r)/'native/metrics.json').exists() for r in self.by_id),
                    completed_curves=sum((self.wd(r)/'stress'/m/'completion.json').exists()
                        for r,row in self.by_id.items() if row['case_id'] in STEP1 for m in MODES))

    def run_queue(self,**kw):
        return ctrl.run('s1',root=self.root,binding=self.root/'binding.json',activate=True,**kw)

    def test_real_181_action_schedule_finite_40_80_96_and_phase_order(self):
        self.assertEqual(len(self.queue),181)
        result = self.run_queue()
        self.assertEqual(result['state'],'STOP_FOR_REVIEW')
        self.assertEqual((len(self.training),len(self.native),len(self.curves)),(40,40,96))
        self.assertEqual([r for r,_ in self.training],[a['run_id'] for a in self.queue if a['action']=='TRAIN_FRESH_50K_AND_NATIVE'])
        self.assertTrue(all(not resumed for _,resumed in self.training))
        self.assertTrue(all(self.by_id[r]['case_id'] in STEP1 for r,m in self.curves))
        self.assertEqual(self.summary.call_count,3)
        for phase in ('STEP1_QROUTE','STEP2_FITTING'):
            self.assertTrue((server_dir('s1',self.root)/('phase_report_'+phase+'.json')).exists())
        self.run_queue()
        self.assertEqual(len(self.training),40)  # STOP does not create a new cycle.

    def test_diagnostic_io_debt_does_not_block_second_phase(self):
        failed = [False]
        def stress(*args,**kwargs):
            if not failed[0]: failed[0]=True; raise OSError('diagnostic storage retry')
            return self.do_stress(*args,**kwargs)
        self.eval_stress.side_effect = stress
        result = self.run_queue()
        self.assertEqual(result['diagnostic_debt'],1)
        self.assertEqual(len(self.training),40)
        self.assertEqual(len(self.curves),95)
        self.eval_stress.side_effect = self.do_stress
        result = ctrl.retry_diagnostics('s1',root=self.root,binding=self.root/'binding.json',activate=True)
        self.assertEqual(result['pending_diagnostics'],0)
        self.assertEqual(len(self.training),40)
        self.assertEqual(len(self.curves),96)
        self.assertEqual(read(server_dir('s1',self.root)/'status.json')['diagnostic_debt'],0)

    def test_train_technical_failure_requires_explicit_retry_same_run_resume(self):
        first = next(a['run_id'] for a in self.queue if a['action']=='TRAIN_FRESH_50K_AND_NATIVE')
        def fail(run_id,*args,**kwargs):
            (self.wd(run_id)/'checkpoints/last').mkdir(parents=True)
            raise OSError('simulated technical failure')
        self.train.side_effect = fail
        with self.assertRaises(OSError): self.run_queue()
        self.assertEqual(read(server_dir('s1',self.root)/'status.json')['state'],'TECHNICAL_FAILURE')
        with self.assertRaises(RuntimeError): self.run_queue()
        self.train.side_effect = self.do_train
        self.run_queue(retry_technical=True)
        self.assertEqual(self.training[0],(first,True))
        record = read(self.wd(first)/'meta/controller_failure.json')
        self.assertEqual(record['kind'],'RESOLVED')
        self.assertEqual(record['previous_failure']['kind'],'TECHNICAL_FAILURE')

    def test_numerical_failure_keeps_failed_seed_and_continues(self):
        first = [None]
        def numerical(run_id,*args,**kwargs):
            if first[0] is None: first[0]=run_id; raise FloatingPointError('diverged')
            return self.do_train(run_id,*args,**kwargs)
        self.train.side_effect = numerical
        result = self.run_queue()
        self.assertEqual(result['completed_students'],39)
        self.assertNotIn(first[0],self.native)
        self.assertFalse(any(r==first[0] for r,m in self.curves))
        self.assertEqual(read(self.wd(first[0])/'meta/controller_failure.json')['kind'],'NUMERICAL_FAILURE')

    def test_evaluator_identity_failure_is_not_diagnostic_debt(self):
        self.eval_native.side_effect = ValueError('checkpoint/source mismatch')
        with self.assertRaisesRegex(ValueError,'checkpoint/source'): self.run_queue()
        self.assertEqual(len(self.training),1)
        self.assertEqual(read(server_dir('s1',self.root)/'status.json')['state'],'TECHNICAL_FAILURE')
        self.assertFalse((server_dir('s1',self.root)/'diagnostic_debt.json').exists())

    def test_signal_and_bad_return_code(self):
        self.train.return_value=75; self.train.side_effect=None
        self.assertEqual(self.run_queue()['state'],'PAUSED_SIGNAL')
        self.assertEqual(len(self.native),0)
        self.train.return_value=1
        with self.assertRaisesRegex(RuntimeError,'successful completion'): self.run_queue()

    def test_no_activation_means_no_prepare(self):
        with self.assertRaises(PermissionError): ctrl.run('s1',root=self.root)
        self.prepare.assert_not_called()

    def test_phase_report_integrity_failure_marks_controller_failed(self):
        self.summary.side_effect=ValueError('evidence seal corrupt')
        with self.assertRaisesRegex(ValueError,'evidence seal'): self.run_queue()
        self.assertEqual(len(self.training),24)
        self.assertEqual(read(server_dir('s1',self.root)/'status.json')['state'],'TECHNICAL_FAILURE')

    def test_preflight_failure_is_persisted_without_training(self):
        self.prepare.side_effect=ValueError('frozen F1 identity changed')
        with self.assertRaises(ValueError): self.run_queue()
        self.assertEqual(read(server_dir('s1',self.root)/'status.json')['stage'],'PREFLIGHT')
        self.assertEqual(len(self.training),0)
        self.prepare.side_effect=RuntimeError('WAITING_FOR_GPU_OWNER: 99')
        result=self.run_queue(retry_technical=True)
        self.assertEqual(result['state'],'WAITING_FOR_GPU_OWNER')
        self.assertEqual(len(self.training),0)


class AdmissionTests(unittest.TestCase):
    def test_image_server_and_gpu_owner_are_bound(self):
        env={'PANDA_RB_IMAGE_ID':DEFAULT_IMAGE,'PANDA_RB_SERVER':'s1'}
        with patch.dict(os.environ,env),patch.object(preflight.subprocess,'check_output',return_value=str(os.getpid())):
            self.assertEqual(preflight.resource_admission('s1')['other_compute_pids'],[])
            with self.assertRaises(ValueError): preflight.resource_admission('s3')
        with patch.dict(os.environ,env),patch.object(preflight.subprocess,'check_output',return_value='999999'):
            with self.assertRaisesRegex(RuntimeError,'WAITING_FOR_GPU_OWNER'): preflight.resource_admission('s1')
        with patch.dict(os.environ,dict(env,PANDA_RB_IMAGE_ID='not-pinned')):
            with self.assertRaisesRegex(RuntimeError,'exact pinned'): preflight.resource_admission('s1')

    def test_retry_allocations_do_not_reduce_future_budget(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp); wd=root/'run'; wd.mkdir()
            (wd/'many_retry_states.bin').write_bytes(b'x'*1000)
            estimate={'estimated_per_run_bytes':100,'required_bytes':4000,'runs':40}
            stack.enter_context(patch('panda_rb_m12.training.required_storage_bytes',return_value=estimate))
            stack.enter_context(patch.object(preflight,'training_runs',return_value=[{'run_id':'test','case_id':'QFULL'}]))
            stack.enter_context(patch.object(preflight,'run_dir',return_value=wd))
            stack.enter_context(patch.object(preflight.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)))
            report=preflight.storage_admission('s1',root)
            self.assertEqual(report['estimated_remaining_bytes'],100+5*1024**3)
            self.assertEqual(report['retained_bytes'],1000)
            for rel in ['checkpoints/selection_manifest.json','native/metrics.json']+[
                'stress/'+mode+'/completion.json' for mode in MODES]:
                atomic_json(wd/rel,{'complete':True})
            check=stack.enter_context(patch('panda_rb_m12.evaluation.validate_artifacts'))
            stack.enter_context(patch('panda_rb_m12.evaluation.selected_checkpoints'))
            report=preflight.storage_admission('s1',root)
            self.assertEqual(report['estimated_remaining_bytes'],5*1024**3)
            self.assertEqual(check.call_count,5)


if __name__=='__main__':
    unittest.main()
