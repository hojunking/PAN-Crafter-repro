"""Launcher guard tests. Every external command and operational mutation is mocked."""
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import maina_hqnr_runner as runner
from tools import maina_hqnr_docker_start as docker
from maina_hqnr.common import RuntimePaused

GPU = 'GPU-unit-test-1234'
COMMIT = 'abcde1234567890abcde1234567890abcde1234567'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='maina-launch-test-')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'repo';self.root.mkdir()
        self.work=self.root/'work_dir';self.work.mkdir()
        (self.root/'gspread').mkdir()
        self.dlpan=self.base/'DLPan-Toolbox';self.dlpan.mkdir()
        self.frozen=self.base/f'repo-runtime-maina-hqnr-s4-{COMMIT[:12]}'
        self.asset=self.work/'asset.bin';self.asset.touch()
        self.external=self.base/'external.h5';self.external.touch()
        self.bindings=dict(canonical_bridge_path=str(self.asset),
            resolved_artifacts=dict(teacher_checkpoint=str(self.asset)),
            dataset_manifest=dict(splits=dict(fr=dict(dataroot=str(self.external),lpan_path=str(self.asset)))))

    def command(self,**overrides):
        args=dict(frozen=self.frozen,server='s4',image_id=docker.PINNED_IMAGE,
            commit=COMMIT,name='unit-main-a',mounts=[(str(self.root),'ro'),(str(self.work),'rw')],
            dlpan=self.dlpan,gpu=GPU)
        args.update(overrides)
        return docker.build_command(**args)

    def test_command_pins_image_gpu_server_and_readonly_sources(self):
        command=self.command(no_upload=True,asset_map=self.external)
        self.assertEqual(command[:3],['docker','run','--detach'])
        self.assertEqual(command[command.index('--gpus')+1],'device='+GPU)
        self.assertEqual(command[command.index('--restart')+1],'no')
        self.assertIn(docker.PINNED_IMAGE,command)
        self.assertIn('org.pancrafter.maina_hqnr.server=s4',command)
        self.assertIn('org.pancrafter.maina_hqnr.gpu_uuid='+GPU,command)
        self.assertIn('PANCRAFTER_MAINA_GPU_UUID='+GPU,command)
        self.assertIn('type=bind,src='+str(self.root)+',dst='+str(self.root)+',readonly',command)
        self.assertIn('type=bind,src='+str(self.work)+',dst='+str(self.work),command)
        self.assertIn('--host-cutover-done',command);self.assertIn('--no-upload',command)
        self.assertEqual(command[-2:],['--asset-map',str(self.external)])
        self.assertNotIn('--privileged',command)

    def test_command_rejects_unpinned_image_gpu_list_and_unsafe_mount(self):
        for fields in (dict(image_id='sha256:other'),dict(gpu='0'),dict(gpu='GPU-a,GPU-b'),
                       dict(mounts=[('/tmp/a,b','ro')]),dict(mounts=[('/tmp/safe','invalid')])):
            with self.subTest(fields=fields),self.assertRaises(ValueError):self.command(**fields)

    def test_mounts_cover_external_symlink_targets_and_only_workdir_is_writable(self):
        credential=self.base/'credential.json';credential.touch()
        (self.root/'gspread/account.json').symlink_to(credential)
        mounts=dict(docker.build_mounts(self.root,self.frozen,self.dlpan,self.bindings,self.external))
        self.assertEqual(mounts[str(self.work)],'rw')
        self.assertEqual([path for path,mode in mounts.items() if mode=='rw'],[str(self.work)])
        for path in (self.root,self.frozen,self.external,self.dlpan,credential):
            self.assertEqual(mounts[str(path)],'ro')

    def _docker_context(self,live=None,image=None,verify_error=None):
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(docker,'ROOT',self.root))
        stack.enter_context(patch.dict(os.environ,{'PANCRAFTER_DLPAN':str(self.dlpan)}))
        self.calls=[]
        def output(argv,**kwargs):
            self.calls.append(argv)
            if argv[:3]==['docker','image','inspect']:return image or docker.PINNED_IMAGE
            if argv[0]=='nvidia-smi':return GPU
            if argv[:3]==['git','rev-parse','HEAD']:return COMMIT
            if argv[:2]==['docker','ps']:return 'live-container' if live is not None else ''
            if argv[:2]==['docker','inspect']:return json.dumps(live)
            raise AssertionError('Unmocked external command: '+repr(argv))
        stack.enter_context(patch.object(docker,'output',side_effect=output))
        self.freeze=stack.enter_context(patch('maina_hqnr.deployment.frozen_checkout',side_effect=AssertionError('No worktree permitted')))
        self.wait=stack.enter_context(patch('maina_hqnr.handoff.wait_boundary',side_effect=AssertionError('No cutover permitted')))
        self.lock=stack.enter_context(patch('maina_hqnr.common.locked',side_effect=lambda *_:nullcontext()))
        self.verify=stack.enter_context(patch('maina_hqnr.assets.verify_assets',return_value=self.bindings,side_effect=verify_error))
        self.stdout=io.StringIO();stack.enter_context(redirect_stdout(self.stdout))
        return stack

    def test_dry_run_never_stops_creates_runtime_or_starts_docker(self):
        self._docker_context()
        before=sorted(str(p.relative_to(self.base)) for p in self.base.rglob('*'))
        self.assertEqual(docker.main(['--server','s4','--dry-run']),0)
        self.verify.assert_called_once_with('s4',root=self.root,overrides=None,persist=False)
        self.freeze.assert_not_called();self.wait.assert_not_called();self.lock.assert_not_called()
        self.assertEqual(before,sorted(str(p.relative_to(self.base)) for p in self.base.rglob('*')))
        self.assertFalse(any(row[:2]==['docker','run'] for row in self.calls))
        self.assertIn('--host-cutover-done',self.stdout.getvalue())

    def test_wrong_local_image_fails_before_asset_or_stop_checks(self):
        self._docker_context(image='sha256:wrong')
        with self.assertRaisesRegex(RuntimePaused,'RUNTIME_MISMATCH'):docker.main(['--server','s4'])
        self.verify.assert_not_called();self.wait.assert_not_called();self.freeze.assert_not_called()

    def test_missing_asset_fails_before_old_cutover(self):
        self._docker_context(verify_error=FileNotFoundError('missing F1'))
        with self.assertRaises(FileNotFoundError):docker.main(['--server','s4'])
        self.wait.assert_not_called();self.freeze.assert_not_called();self.lock.assert_not_called()

    def test_same_live_owner_is_idempotent_no_cutover_or_second_launch(self):
        live=[dict(Image=docker.PINNED_IMAGE,Config=dict(Labels={docker.LABEL+'.runtime':str(self.frozen),docker.LABEL+'.gpu_uuid':GPU}))]
        self._docker_context(live=live)
        self.assertEqual(docker.main(['--server','s4']),0)
        self.assertEqual(json.loads(self.stdout.getvalue())['status'],'ALREADY_RUNNING')
        self.assertEqual(json.loads(self.stdout.getvalue())['gpu_uuid'],GPU)
        self.wait.assert_not_called();self.freeze.assert_not_called()
        self.assertFalse(any(row[:2]==['docker','run'] for row in self.calls))

    def test_different_live_release_rejected(self):
        live=[dict(Image=docker.PINNED_IMAGE,Config=dict(Labels={docker.LABEL+'.runtime':'/other/release'}))]
        self._docker_context(live=live)
        with self.assertRaisesRegex(RuntimePaused,'Different local'):docker.main(['--server','s4'])
        self.wait.assert_not_called();self.freeze.assert_not_called()

    def test_different_live_gpu_rejected_without_second_launch(self):
        live=[dict(Image=docker.PINNED_IMAGE,Config=dict(Labels={docker.LABEL+'.runtime':str(self.frozen),docker.LABEL+'.gpu_uuid':'GPU-other-4321'}))]
        self._docker_context(live=live)
        with self.assertRaisesRegex(RuntimePaused,'GPU identity'):docker.main(['--server','s4'])
        self.wait.assert_not_called();self.freeze.assert_not_called()
        self.assertFalse(any(row[:2]==['docker','run'] for row in self.calls))

    def test_missing_live_gpu_label_fails_closed(self):
        live=[dict(Image=docker.PINNED_IMAGE,Config=dict(Labels={docker.LABEL+'.runtime':str(self.frozen)}))]
        self._docker_context(live=live)
        with self.assertRaisesRegex(RuntimePaused,'GPU identity'):docker.main(['--server','s4'])
        self.wait.assert_not_called();self.freeze.assert_not_called()

    def test_multiple_live_owners_rejected(self):
        self._docker_context(live=[{},{}])
        with self.assertRaisesRegex(RuntimePaused,'Multiple active'):docker.main(['--server','s4'])
        self.wait.assert_not_called();self.freeze.assert_not_called()

    def test_both_parsers_protect_s1_s2_s3_before_any_action(self):
        for server in ('s1','s2','s3'):
            for module,args in ((docker,['--server',server]),(runner,['preview','--server',server])):
                with self.subTest(module=module.__name__,server=server),redirect_stderr(io.StringIO()),patch.object(docker,'output') as command:
                    with self.assertRaises(SystemExit) as error:module.main(args)
                    self.assertEqual(error.exception.code,2);command.assert_not_called()
            with self.assertRaises(ValueError):self.command(server=server)

    def test_runner_train_postrun_server_flag_must_match_config_before_lock(self):
        for phase in ('train','postrun'):
            with self.subTest(phase=phase),patch('maina_hqnr.common.read_config',return_value={'maina_hqnr':{'case':{'server':'s5'}}}),patch('maina_hqnr.common.locked') as lock:
                with self.assertRaisesRegex(PermissionError,'must match'):
                    runner.main([phase,'--server','s4','--config',str(self.root/'s5.json')])
                lock.assert_not_called()

    def test_assets_cli_is_readonly_and_does_not_resolve_gpu(self):
        with patch.object(runner,'ROOT',self.root),patch('maina_hqnr.assets.verify_assets',return_value={'verified':True}) as verify,patch('maina_hqnr.common.selected_gpu_uuid') as gpu,redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['assets','--server','s4']),0)
            verify.assert_called_once_with('s4',root=self.root,overrides=None,persist=False)
            gpu.assert_not_called()

    def test_preflight_rejects_nonfrozen_root_without_gpu_or_numerical_work(self):
        for receipt in ({},dict(path='/another/frozen')):
            with self.subTest(receipt=receipt),patch.object(runner,'ROOT',self.root),patch('maina_hqnr.common.read',return_value=receipt),patch('maina_hqnr.common.selected_gpu_uuid') as gpu,patch('maina_hqnr.preflight.run_preflight') as probe:
                with self.assertRaisesRegex(PermissionError,'committed campaign runtime'):
                    runner.main(['preflight','--server','s4'])
                gpu.assert_not_called();probe.assert_not_called()

    def test_preflight_frozen_runtime_checks_gpu_idle_before_probe(self):
        order=[]
        with patch.object(runner,'ROOT',self.root),patch('maina_hqnr.common.read',return_value=dict(path=str(self.root))),patch('maina_hqnr.common.selected_gpu_uuid',return_value=GPU),patch('maina_hqnr.handoff.ensure_gpu_idle',side_effect=lambda gpu:order.append(('idle',gpu))),patch('maina_hqnr.preflight.run_preflight',side_effect=lambda *a,**k:order.append(('probe',a)) or {'status':'PASSED'}),redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['preflight','--server','s4']),0)
        self.assertEqual(order[0],('idle',GPU));self.assertEqual(order[1][0],'probe')

    def test_busy_preflight_gpu_does_not_run_probe(self):
        with patch.object(runner,'ROOT',self.root),patch('maina_hqnr.common.read',return_value=dict(path=str(self.root))),patch('maina_hqnr.common.selected_gpu_uuid',return_value=GPU),patch('maina_hqnr.handoff.ensure_gpu_idle',side_effect=RuntimePaused('busy GPU')),patch('maina_hqnr.preflight.run_preflight') as probe:
            with self.assertRaises(RuntimePaused):runner.main(['preflight','--server','s4'])
            probe.assert_not_called()


if __name__=='__main__':unittest.main()
