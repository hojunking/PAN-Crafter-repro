"""Offline entrypoint/deployment tests; no GPU, worker, cron or git mutations."""
from contextlib import ExitStack, contextmanager, nullcontext
import importlib.util
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pcrepro.common import ROOT, atomic_json, camp, read
from pcrepro import deployment, preflight
from pcrepro.plan import Case


spec=importlib.util.spec_from_file_location('pcrepro_runtime_entrypoint',ROOT/'tools/pcrepro_runner.py')
CLI=importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


@contextmanager
def offline_cli():
    with ExitStack() as stack:
        signals=stack.enter_context(patch.object(CLI.signal,'signal'))
        stack.enter_context(patch.object(CLI,'locked',side_effect=lambda _:nullcontext()))
        gpu=stack.enter_context(patch('pcrepro.handoff.gpu_processes',return_value=[]))
        stack.enter_context(patch('subprocess.call',side_effect=AssertionError('unmocked child process')))
        stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('unmocked child process')))
        yield signals,gpu


class EntrypointTests(unittest.TestCase):
    def test_preflight_branch_reaches_worker_with_safe_stop_callback(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli() as (signals,_):
            root=Path(temporary)
            def execute(actual_root,server,dataset,bindings,*,stopcheck):
                self.assertEqual((actual_root,server,dataset,bindings),(root,'s3','GF2',Path('binding.json')))
                self.assertFalse(stopcheck())
                callback=next(call.args[1] for call in signals.call_args_list if call.args[0]==signal.SIGTERM)
                callback(signal.SIGTERM,None)
                self.assertTrue(stopcheck())
                return {'fixture':True}
            with patch('pcrepro.preflight.execute',side_effect=execute) as action:
                result=CLI.main(['preflight','--root',str(root),'--server','s3','--dataset','GF2','--bindings','binding.json'])
            self.assertEqual(result,{'fixture':True});action.assert_called_once()

    def test_busy_gpu_rejects_preflight_before_smoke_and_records_reason(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli() as (_,gpu):
            gpu.return_value=[98765]
            with patch('pcrepro.preflight.execute') as action,self.assertRaisesRegex(ValueError,'GPU busy'):
                CLI.main(['preflight','--root',temporary,'--server','s3','--dataset','GF2','--bindings','binding.json'])
            action.assert_not_called()
            self.assertIn('GPU busy',read(camp(temporary,'s3')/'preflight/GF2.failure.json')['reason'])

    def test_preflight_io_error_is_not_silently_converted_to_success(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli(), \
                patch('pcrepro.preflight.execute',side_effect=OSError('fixture disk failure')):
            with self.assertRaisesRegex(OSError,'fixture disk failure'):
                CLI.main(['preflight','--root',temporary,'--server','s3','--dataset','GF2','--bindings','binding.json'])
            self.assertEqual(read(camp(temporary,'s3')/'preflight/GF2.failure.json')['reason'],'OSError: fixture disk failure')

    def test_train_branch_forwards_resume_without_entering_real_training(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli(), \
                patch.object(CLI,'read_config',return_value={'fixture':True}), \
                patch('pcrepro.plan.validate_config',return_value=Case('s3',0,'WV3')) as validate, \
                patch('pcrepro.training.train_run',return_value=75) as train:
            self.assertEqual(CLI.main(['train','--root',temporary,'--server','s3','--config','run.json','--resume']),{'exit_code':75})
            validate.assert_called_once_with({'fixture':True},require_bound=True)
            train.assert_called_once_with(Path('run.json'),Path(temporary),resume=True)

    def test_postrun_branch_passes_interrupt_callback(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli(), \
                patch.object(CLI,'read_config',return_value={}), \
                patch('pcrepro.plan.validate_config',return_value=Case('s3',0,'WV3')), \
                patch('pcrepro.postrun.execute',return_value={'complete':True}) as evaluate:
            self.assertEqual(CLI.main(['postrun','--root',temporary,'--server','s3','--config','run.json']),{'complete':True})
            self.assertEqual(evaluate.call_args.args,(Path('run.json'),Path(temporary)))
            self.assertFalse(evaluate.call_args.kwargs['stopcheck']())

    def test_config_from_another_server_never_enters_train_or_postrun(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli(), \
                patch.object(CLI,'read_config',return_value={}), \
                patch('pcrepro.plan.validate_config',return_value=Case('s4',0,'WV3')), \
                patch('pcrepro.training.train_run') as train,patch('pcrepro.postrun.execute') as evaluate:
            for action in ('train','postrun'):
                with self.subTest(action=action),self.assertRaisesRegex(ValueError,'Config/server differs'):
                    CLI.main([action,'--root',temporary,'--server','s3','--config','run.json'])
            train.assert_not_called();evaluate.assert_not_called()

    def test_unknown_gpu_state_never_enters_train_or_postrun(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli() as (_,gpu), \
                patch.object(CLI,'read_config',return_value={}), \
                patch('pcrepro.plan.validate_config',return_value=Case('s3',0,'WV3')), \
                patch('pcrepro.training.train_run') as train,patch('pcrepro.postrun.execute') as evaluate:
            gpu.return_value=None
            for action in ('train','postrun'):
                with self.subTest(action=action),self.assertRaisesRegex(ValueError,'GPU busy/unavailable'):
                    CLI.main([action,'--root',temporary,'--server','s3','--config','run.json'])
            train.assert_not_called();evaluate.assert_not_called()

    def test_frozen_release_routing_strips_mutable_root_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli(),patch('subprocess.call',return_value=75) as call:
            root=Path(temporary)/'repo';target=root.parent/'repo-runtime-pcrepro-s3-123456789abc'
            atomic_json(camp(root,'s3')/'runtime_release.json',{'path':str(target)})
            result=CLI.main(['status','--root',str(root),'--server','s3'])
            self.assertEqual(result,{'exit_code':75})
            self.assertEqual(call.call_args.args[0][1:],[str(target/'tools/pcrepro_runner.py'),'status','--server','s3'])
            self.assertEqual(call.call_args.kwargs['cwd'],target)

    def test_foreign_release_routing_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary,offline_cli():
            root=Path(temporary)/'repo'
            for target in (root.parent/'unrelated',root.parent/'repo-runtime-pcrepro-s4-123456789abc'):
                atomic_json(camp(root,'s3')/'runtime_release.json',{'path':str(target)})
                with self.subTest(target=target),self.assertRaisesRegex(ValueError,'Unexpected release'):
                    CLI.main(['status','--root',str(root),'--server','s3'])


class DeploymentTests(unittest.TestCase):
    def test_existing_release_requires_matching_hashes_and_commit(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch('pcrepro.deployment.verify_sources'), \
                patch('pcrepro.deployment.subprocess.check_output',side_effect=AssertionError('unexpected git command')), \
                patch('pcrepro.deployment.subprocess.run',side_effect=AssertionError('unexpected worktree mutation')):
            root=Path(temporary)/'repo';target=root.parent/'repo-runtime-pcrepro-s3-123456789abc'
            receipt={'path':str(target),'files':{'model.py':'original'},'git_commit':'commit'}
            atomic_json(camp(root,'s3')/'runtime_release.json',receipt)
            for identity in ({'files':{'model.py':'changed'},'git_release':'commit'},
                             {'files':receipt['files'],'git_release':'other'}):
                with patch('pcrepro.deployment.source_identity',return_value=identity),self.assertRaisesRegex(ValueError,'release changed'):
                    deployment.frozen_checkout(root,'s3')
            with patch('pcrepro.deployment.source_identity',return_value={'files':receipt['files'],'git_release':'commit'}):
                self.assertEqual(deployment.frozen_checkout(root,'s3'),target)

    def test_deployment_never_accepts_s1_s2(self):
        with patch('pcrepro.deployment.subprocess.run',side_effect=AssertionError('unexpected git command')):
            for server in ('s1','s2'):
                with self.subTest(server=server),self.assertRaises(ValueError):deployment.frozen_checkout(ROOT,server)


class PreflightTests(unittest.TestCase):
    def test_disk_guard_measures_shared_output_volume_not_release_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'release';root.mkdir()
            outputs=Path(temporary)/'actual_output_volume';outputs.mkdir()
            (root/'work_dir').symlink_to(outputs,target_is_directory=True)
            with patch('pcrepro.preflight.shutil.disk_usage',return_value=SimpleNamespace(free=19*1024**3)) as usage:
                result=preflight.disk_guard(root)
            usage.assert_called_once_with(outputs)
            self.assertEqual(result['measured_path'],str(outputs));self.assertFalse(result['allowed'])
            self.assertFalse(result['automatic_pruning'])

    def test_disk_guard_accounts_for_next_atomic_save_above_minimum(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch('pcrepro.preflight.shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
            result=preflight.disk_guard(temporary,32*1024**3)
            self.assertEqual(result['required_bytes'],42*1024**3)
            self.assertFalse(result['allowed'])

    def test_stop_before_manifest_hashing_or_cuda_work(self):
        with tempfile.TemporaryDirectory() as root,patch('pcrepro.preflight.verify_sources'), \
                patch('pcrepro.preflight.apply_runtime_policy'),patch('pcrepro.data.validate_manifest') as manifest:
            with self.assertRaises(InterruptedError):
                preflight.execute(root,'s3','GF2',{'datasets':{'GF2':{'fixture':True}}},stopcheck=lambda:True)
            manifest.assert_not_called()

    def test_missing_binding_blocks_only_requested_dataset_without_cuda(self):
        with tempfile.TemporaryDirectory() as root,patch('pcrepro.preflight.verify_sources'), \
                patch('pcrepro.preflight.apply_runtime_policy'),patch('pcrepro.data.validate_manifest') as manifest:
            with self.assertRaisesRegex(ValueError,'BLOCKED_DATASET.*QB'):
                preflight.execute(root,'s3','QB',{'datasets':{'GF2':{'fixture':True}}})
            manifest.assert_not_called()
            self.assertFalse((camp(root,'s3')/'preflight/GF2.json').exists())


if __name__=='__main__':unittest.main()
