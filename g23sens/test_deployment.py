from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g23sens import common as C,handoff as H
from tools.g23sens_docker_start import build_command


class DeploymentTests(unittest.TestCase):
    def finished_original(self,root,server='s4'):
        from pcrepro.plan import Case
        from fh12.common import object_sha as original_sha
        case=Case(server,0,'WV3');wd=root/'work_dir'/case.run_id;records={}
        C.atomic_json(wd/'meta/training_status.json',dict(training_complete=True,actual_updates=50000))
        for selection,label in [('EXACT_50000','exact_50000'),('RR_VAL_ERGAS_MIN','best_val')]:
            checkpoint=wd/'checkpoints'/(label+'_050000_fixture');checkpoint.mkdir(parents=True)
            weight=checkpoint/'model.safetensors';weight.write_bytes(b'offline synthetic opaque checkpoint')
            identity=dict(update=50000,model_sha256=C.sha256(weight),label=label)
            C.atomic_json(checkpoint/'identity.json',identity)
            C.atomic_json(wd/'checkpoints'/(label+'.json'),dict(directory=checkpoint.name,identity_sha256=original_sha(identity)))
            relative='official/evaluations/'+identity['model_sha256']+'/metrics.json'
            metrics=dict(metadata={'checkpoint_sha256':identity['model_sha256']},
                rr={'official_complete':True},fr={'official_complete':True})
            C.atomic_json(wd/relative,metrics);C.atomic_json((wd/relative).parent/'evaluation_cursor.json',dict(complete=True))
            records[selection]=dict(checkpoint_sha256=identity['model_sha256'],checkpoint_identity=identity,
                evaluation_path=relative,evaluation_manifest_sha256=C.sha256(wd/relative),rr=metrics['rr'],fr=metrics['fr'])
        C.atomic_json(wd/'official/summary.json',dict(complete=True,run_id=case.run_id,case=case.to_dict(),
            actual_updates=50000,selections=records))
        return case

    def test_docker_no_finite_lease_and_no_shared_lock(self):
        argv=build_command(frozen='/repo/frozen',server='s4',image_id='sha256:'+'a'*64,
            commit='b'*40,name='fixture',mounts=[('/repo','ro'),('/repo/work_dir','rw')],dlpan='/DLPan')
        self.assertIn('--restart',argv);self.assertEqual(argv[argv.index('--restart')+1],'no')
        self.assertNotIn('--lease-hours',argv);self.assertNotIn('--until-operator-stop',argv)
        self.assertIn('type=bind,src=/repo,dst=/repo,readonly',argv)
        self.assertIn('type=bind,src=/repo/work_dir,dst=/repo/work_dir',argv)
        self.assertNotIn('s5',' '.join(argv))

    def test_docker_protects_s1_s3_and_image_content(self):
        args=dict(frozen='/repo',server='s1',image_id='latest',commit='b'*40,name='fixture',mounts=[],dlpan='/DLPan')
        with self.assertRaises(ValueError):build_command(**args)
        args['server']='s4'
        with self.assertRaises(ValueError):build_command(**args)

    def test_known_repro_current_run_boundary_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'work_dir/_pcrepro/s4';old.mkdir(parents=True)
            with patch.object(H,'inventory',return_value=[dict(pid=123,script='pcrepro_runner.py',server='s4')]):
                result=H.request_boundary(root,'s4',activated=True)
            self.assertFalse(result['signals_sent'])
            self.assertEqual(C.read(old/'control.json')['command'],'STOP_AFTER_CURRENT_RUN')
            self.assertFalse((root/'work_dir/_pcrepro/s5/control.json').exists())

    def test_unknown_owner_never_forced_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(H,'inventory',return_value=[dict(pid=123,script='gfp40_runner.py',server='s4')]):
                with self.assertRaises(C.RuntimePaused):H.request_boundary(directory,'s4',activated=True)
            self.assertFalse((Path(directory)/'work_dir/_gfp40/s4/control.json').exists())

    def test_explicit_start_needed_for_cooperative_write(self):
        with self.assertRaises(PermissionError):H.request_boundary('/tmp/not-created','s4')

    def test_stronger_original_stop_is_preserved_and_request_snapshot_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(H,'inventory',return_value=[]):
            root=Path(directory);old=root/'work_dir/_pcrepro/s4'
            control=dict(command='STOP_NOW_SAFE',operator_request='original manual stop')
            state=dict(server='s4',cycle=2,index=1,active_run=None,costs={'train_h':17.5})
            C.atomic_json(old/'control.json',control);C.atomic_json(old/'status.json',state)
            before=(old/'control.json').read_bytes();first=H.request_boundary(root,'s4',activated=True)
            changed=dict(state,cycle=3);C.atomic_json(old/'status.json',changed)
            self.assertEqual(H.request_boundary(root,'s4',activated=True),first)
            self.assertEqual((old/'control.json').read_bytes(),before)
            self.assertEqual(first['original_state'],state);self.assertEqual(C.read(old/'status.json'),changed)

    def test_process_exit_never_converts_incomplete_original_run_to_completed_boundary(self):
        from pcrepro.plan import Case
        with tempfile.TemporaryDirectory() as directory,patch.object(H,'inventory',return_value=[]),patch.object(H,'gpu_rows',return_value=[]):
            root=Path(directory);old=root/'work_dir/_pcrepro/s4';run=Case('s4',0,'WV3').run_id
            C.atomic_json(old/'status.json',dict(server='s4',active_run=run))
            request=H.request_boundary(root,'s4',activated=True)
            with self.assertRaisesRegex(C.RuntimePaused,'ACTIVE_CURSOR'):H.verify_boundary(root,'s4',request,'GPU-owned')
            C.atomic_json(old/'status.json',dict(server='s4',active_run=None))
            with self.assertRaisesRegex(C.RuntimePaused,'RUN_INCOMPLETE'):H.verify_boundary(root,'s4',request,'GPU-owned')
            self.assertFalse((C.camp(root,'s4')/'handoff/complete.json').exists())

    def test_original_endpoint_and_evaluation_evidence_checked_before_handoff(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(H,'inventory',return_value=[]),patch.object(H,'gpu_rows',return_value=[]):
            root=Path(directory);old=root/'work_dir/_pcrepro/s4';case=self.finished_original(root)
            C.atomic_json(old/'status.json',dict(server='s4',active_run=case.run_id))
            request=H.request_boundary(root,'s4',activated=True)
            C.atomic_json(old/'status.json',dict(server='s4',active_run=None))
            receipt=H.verify_boundary(root,'s4',request,'GPU-owned')
            self.assertEqual(receipt['verified_original_runs'][0]['run_id'],case.run_id)
            path=root/'work_dir'/case.run_id/'checkpoints/exact_50000_050000_fixture/model.safetensors'
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity'):H.verify_boundary(root,'s4',request,'GPU-owned')

    def test_other_lane_on_another_gpu_is_not_a_barrier_or_control_target(self):
        foreign=dict(pid=7,script='gfp40_runner.py',server='s3')
        with tempfile.TemporaryDirectory() as directory,patch.object(H,'inventory',return_value=[foreign]), \
             patch.object(H,'gpu_rows',return_value=[dict(pid=7,gpu_uuid='GPU-other')]):
            root=Path(directory);request=H.request_boundary(root,'s4',activated=True)
            result=H.verify_boundary(root,'s4',request,'GPU-owned')
            self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['protected_other_lane_processes'],[foreign])
            self.assertFalse((root/'work_dir/_pcrepro/s3').exists())
            with self.assertRaisesRegex(C.RuntimePaused,'Selected GPU'):
                H.verify_boundary(root,'s4',request,'GPU-other')

    def test_old_admission_reenabled_is_not_silently_overridden(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(H,'inventory',return_value=[]):
            root=Path(directory);old=root/'work_dir/_pcrepro/s4';old.mkdir(parents=True)
            H.request_boundary(root,'s4',activated=True)
            C.atomic_json(old/'control.json',dict(command='CONTINUE'))
            with self.assertRaisesRegex(C.RuntimePaused,'reenabled'):H.request_boundary(root,'s4',activated=True)
            self.assertEqual(C.read(old/'control.json')['command'],'CONTINUE')

    def test_storage_pause_never_deletes(self):
        from g23sens.resources import ensure_space
        from collections import namedtuple
        usage=namedtuple('Usage','total used free')(100,99,1)
        with patch('shutil.disk_usage',return_value=usage):
            with self.assertRaises(C.RuntimePaused):ensure_space('/tmp',8)


if __name__=='__main__':unittest.main()
