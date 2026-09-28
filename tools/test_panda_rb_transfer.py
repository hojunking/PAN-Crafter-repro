import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.panda_rb_transfer import DEFAULT_IMAGE, transfer


class TransferTests(unittest.TestCase):
    def test_export_dry_run_is_cpu_pinned_no_network_no_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'work_dir').mkdir()
            with patch('tools.panda_rb_transfer.git_mounts',return_value=[root/'.git']), \
                 patch('tools.panda_rb_transfer.asset_mounts',return_value=[]), \
                 patch('tools.panda_rb_transfer.subprocess.check_output') as inspect, \
                 patch('tools.panda_rb_transfer.subprocess.run') as run:
                result=transfer('export','work_dir/package',root=root,dry_run=True)
                inspect.assert_not_called();run.assert_not_called()
            cmd=result['command']
            self.assertIn(DEFAULT_IMAGE,cmd);self.assertNotIn('--gpus',cmd)
            self.assertEqual(cmd[cmd.index('--network')+1],'none')
            self.assertIn(f'{root}:{root}:ro',cmd)
            self.assertIn(f'{root/"work_dir"}:{root/"work_dir"}:rw',cmd)
            self.assertIn(f'{root/".git"}:{root/".git"}:ro',cmd)
            self.assertFalse(result['training_started']);self.assertFalse(result['remote_actions'])

    def test_import_explicit_four_data_paths_and_mock_docker_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'work_dir/package';package.mkdir(parents=True)
            (package/'package.json').write_text('{}');mapping={}
            for split in ('train','val','rr','fr'):
                path=root/(split+'.h5');path.write_bytes(b'fixture');mapping[split]=str(path)
            paths=root/'local_data_paths.json';paths.write_text(json.dumps(mapping))
            with patch('tools.panda_rb_transfer.git_mounts',return_value=[]), \
                 patch('tools.panda_rb_transfer.asset_mounts',return_value=[]) as mounts, \
                 patch('tools.panda_rb_transfer.subprocess.check_output',return_value=json.dumps([{'Id':DEFAULT_IMAGE}])) as inspect, \
                 patch('tools.panda_rb_transfer.subprocess.run') as run:
                result=transfer('import',package,root=root,data_paths=paths)
                run.assert_called_once_with(result['command'],check=True)
                inspect.assert_called_once()
                self.assertIn(paths,mounts.call_args.args[2])
            self.assertIn('--data-paths',result['command'])
            self.assertIn(str(package),result['command']);self.assertTrue(result['asset_operation_complete'])

    def test_reject_broad_external_and_symlink_escape_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work_dir';work.mkdir();outside=root/'outside';outside.mkdir()
            (work/'escape').symlink_to(outside,target_is_directory=True)
            for destination in (work,outside,work/'escape/package'):
                with self.assertRaises(ValueError):transfer('export',destination,root=root,dry_run=True)


if __name__=='__main__':unittest.main()
