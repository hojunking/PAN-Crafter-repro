"""No real git worktree, Docker command, GPU, stop request or network is used."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from maina_hqnr import deployment as dep
from maina_hqnr.common import ROOT,atomic_json,camp

_spec=importlib.util.spec_from_file_location('maina_docker_launcher_test',ROOT/'tools/maina_hqnr_docker_start.py')
launcher=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(launcher)


class DeploymentTests(unittest.TestCase):
    def test_link_never_overwrites_different_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';target=root/'target'
            source.write_bytes(b'original');target.write_bytes(b'other')
            with self.assertRaises(ValueError):dep._link(source,target)
            self.assertEqual(target.read_bytes(),b'other')
            target.unlink();dep._link(source,target)
            self.assertTrue(target.is_symlink());self.assertEqual(target.resolve(),source)

    def test_receipt_source_change_pauses_without_worktree(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';root.mkdir();target=root.parent/'repo-runtime-maina-hqnr-s4-abcdef'
            atomic_json(camp(root,'s4')/'runtime_release.json',dict(path=str(target),files={'a':'old'},git_commit='abcdef'))
            with patch.object(dep,'verify_sources'),patch.object(dep,'source_identity',return_value={'files':{'a':'changed'},'git_release':'abcdef'}),patch.object(dep.subprocess,'run') as run:
                with self.assertRaises(ValueError):dep.frozen_checkout(root,'s4')
                run.assert_not_called()

    def test_untracked_execution_source_blocks_deployment(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source={'files':{'maina_hqnr/new.py':'hash'}}
            with patch.object(dep,'verify_sources'),patch.object(dep,'source_identity',return_value=source),patch.object(dep.subprocess,'check_output',side_effect=['abcdef',b'']),patch.object(dep.subprocess,'run') as run:
                with self.assertRaises(ValueError):dep.frozen_checkout(root,'s4')
                run.assert_not_called()

    def test_dirty_execution_source_blocks_deployment(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source={'files':{'maina_hqnr/new.py':'hash'}}
            outputs=['abcdef',b'maina_hqnr/new.py\0','maina_hqnr/new.py\n']
            with patch.object(dep,'verify_sources'),patch.object(dep,'source_identity',return_value=source),patch.object(dep.subprocess,'check_output',side_effect=outputs),patch.object(dep.subprocess,'run') as run:
                with self.assertRaises(ValueError):dep.frozen_checkout(root,'s4')
                run.assert_not_called()

    def test_pinned_docker_gpu_and_readonly_source(self):
        command=launcher.build_command(frozen='/fixture/runtime',server='s4',image_id=launcher.PINNED_IMAGE,
            commit='abcdef',name='fixture',mounts=[('/fixture/runtime','ro'),('/fixture/work_dir','rw')],
            dlpan='/fixture/DLPan',gpu='GPU-fixture',no_upload=True)
        self.assertIn('device=GPU-fixture',command)
        self.assertIn('type=bind,src=/fixture/runtime,dst=/fixture/runtime,readonly',command)
        self.assertIn('--no-upload',command);self.assertIn('--host-cutover-done',command)
        self.assertEqual(command[command.index('--restart')+1],'no')
        self.assertNotIn('--privileged',command)
        for changes in ({'server':'s1'},{'gpu':'0,1'},{'image_id':'unverified:latest'}):
            kw=dict(frozen='/fixture/runtime',server='s4',image_id=launcher.PINNED_IMAGE,commit='a',
                name='fixture',mounts=[],dlpan='/fixture/DLPan',gpu='GPU-fixture');kw.update(changes)
            with self.assertRaises(ValueError):launcher.build_command(**kw)

    def test_mount_external_data_ro_and_only_work_rw(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'repo';runtime=base/'runtime';external=base/'external'
            for p in (root/'work_dir',runtime,external):p.mkdir(parents=True)
            native=external/'native.h5';native.write_bytes(b'fixture')
            bridge=root/'bridge.json';bridge.write_text('{}')
            bindings={'canonical_bridge_path':str(bridge),'resolved_artifacts':{'teacher':str(native)},
                'dataset_manifest':{'splits':{'rr':{'dataroot':str(native),'lpan_path':str(native)}}}}
            mounts=dict(launcher.build_mounts(root,runtime,external,bindings))
            self.assertEqual(mounts[str(native)],'ro')
            self.assertEqual([path for path,access in mounts.items() if access=='rw'],[str(root/'work_dir')])


if __name__=='__main__':unittest.main()
