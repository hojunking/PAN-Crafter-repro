"""Real temporary Git worktrees and mocked launches; never starts GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pcrepro import s5_runtime as runtime
from pcrepro.common import atomic_json,sha256,read_json
from pcrepro.finite import folder


class S5RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'repo';self.root.mkdir()
        (self.root/'pcrepro').mkdir();(self.root/'data').mkdir();(self.root/'gspread').mkdir()
        (self.root/'pcrepro/stub.py').write_text('VERSION = 1\n')
        (self.root/'data/fixture.h5').write_bytes(b'x')
        (self.root/'gspread/server.txt').write_text('s5\n')
        (self.root/'.gitignore').write_text('data/\nwork_dir/\ngspread/server.txt\n')
        self.dlpan=self.base/'DLPan-Toolbox';wald=self.dlpan/runtime.WALD
        wald.parent.mkdir(parents=True);wald.write_text('# fixture evaluator\n')
        atomic_json(self.root/'pcrepro/s5_assets.json',dict(wald_sha256=sha256(wald),
            data_files={'data/fixture.h5':dict(bytes=1,sha256=sha256(self.root/'data/fixture.h5'))}))
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        self.commit()
        self.files={n:sha256(self.root/n) for n in ('pcrepro/stub.py','pcrepro/s5_assets.json')}
        self.files['external/DLPan/wald_utilities.py']=sha256(wald)
        patches=[mock.patch.object(runtime,'source_identity',return_value=dict(files=self.files)),
            mock.patch.dict(os.environ,PANCRAFTER_DLPAN=str(self.dlpan),CUDA_VISIBLE_DEVICES='0')]
        for patch in patches:patch.start();self.addCleanup(patch.stop)

    def commit(self):
        subprocess.run(['git','add','.'],cwd=self.root,check=True)
        subprocess.run(['git','-c','user.name=Offline Test','-c','user.email=offline@example.invalid',
            '-c','core.hooksPath=/dev/null','commit','-qm','fixture'],cwd=self.root,check=True)

    def test_non_s5_stops_before_deployment(self):
        (self.root/'gspread/server.txt').write_text('s1')
        with self.assertRaisesRegex(ValueError,'s5 routing'):runtime.snapshot(self.root)
        self.assertEqual(list(self.base.glob('repo-runtime-*')),[])

    def test_committed_worktree_shares_assets_and_survives_parent_pull(self):
        receipt=runtime.snapshot(self.root);target=Path(receipt['path'])
        self.assertEqual((target/'data').resolve(),self.root/'data')
        self.assertEqual((target/'work_dir').resolve(),self.root/'work_dir')
        self.assertEqual((target/'gspread/server.txt').read_text().strip(),'s5')
        self.assertEqual(list((target/'gspread').glob('*.json')),[])
        (self.root/'pcrepro/stub.py').write_text('VERSION = 2\n');self.commit()
        self.assertNotEqual(runtime.git(self.root,'rev-parse','HEAD'),receipt['commit'])
        self.assertEqual(runtime.snapshot(self.root),receipt)
        self.assertEqual((target/'pcrepro/stub.py').read_text(),'VERSION = 1\n')

    def test_dirty_source_cannot_be_deployed(self):
        (self.root/'pcrepro/stub.py').write_text('VERSION = 99\n')
        with self.assertRaisesRegex(ValueError,'local edits'):runtime.snapshot(self.root)
        self.assertEqual(list(self.base.glob('repo-runtime-*')),[])

    def test_mutated_pinned_runtime_is_rejected(self):
        receipt=runtime.snapshot(self.root)
        (Path(receipt['path'])/'pcrepro/stub.py').write_text('mutated\n')
        with self.assertRaisesRegex(ValueError,'source changed'):runtime.snapshot(self.root)

    def test_missing_data_and_different_evaluator_fail_before_launch(self):
        (self.root/'data/fixture.h5').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'data files'):runtime.resolve_inputs(self.root)
        (self.root/'data/fixture.h5').write_bytes(b'x')
        (self.dlpan/runtime.WALD).write_text('# different\n')
        with self.assertRaisesRegex(ValueError,'differs'):runtime.resolve_inputs(self.root)

    def test_detached_fixed_launch_is_idempotent(self):
        release=dict(path=str(self.root),python=sys.executable,dlpan=str(self.dlpan),commit='a'*40)
        child=SimpleNamespace(pid=12345,returncode=None,poll=lambda:None)
        with mock.patch.object(runtime,'require_idle'),mock.patch.object(runtime,'snapshot',return_value=release), \
                mock.patch.object(runtime,'alive',side_effect=[False,True]), \
                mock.patch.object(runtime.subprocess,'call',return_value=0) as prepare, \
                mock.patch.object(runtime.subprocess,'Popen',return_value=child) as launch, \
                mock.patch.object(runtime,'process_start',return_value='123'),mock.patch.object(runtime.time,'sleep'):
            result=runtime.start(self.root)
            self.assertEqual(result['state'],'LAUNCHED')
            self.assertEqual(prepare.call_args.args[0][-3:],['prepare','--seeds','2025'])
            self.assertEqual(launch.call_args.args[0][-1],'run')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(runtime.start(self.root)['state'],'ALREADY_RUNNING')
            self.assertEqual(launch.call_count,1)

    def test_failed_prepare_never_starts_training(self):
        release=dict(path=str(self.root),python=sys.executable,dlpan=str(self.dlpan),commit='a'*40)
        with mock.patch.object(runtime,'require_idle'),mock.patch.object(runtime,'snapshot',return_value=release), \
                mock.patch.object(runtime.subprocess,'call',return_value=1), \
                mock.patch.object(runtime.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'Preparation failed'):runtime.start(self.root)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
