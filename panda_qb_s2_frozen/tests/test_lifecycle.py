from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from panda_qb_s2_frozen import enqueue as E, registry as G, worker


class LifecycleTests(unittest.TestCase):
    def fixture(self, root, *, running=True):
        (root/'gspread').mkdir()
        (root/'gspread/server.txt').write_text('s2\n')
        (root/'gspread/account.json').write_text('{}')
        identity={'image':'sha256:test','user':'1000:1000','working_dir':'/workspace',
                  'env':['PATH=/bin'],'mounts':[]}
        receipt={'identity':identity,'token':'token','source_sha256':'release','container_id':'container'}
        current={'Id':'container','Image':identity['image'],'Mounts':[],
            'Config':{'Cmd':E.training_args(),'User':identity['user'],'WorkingDir':'/workspace',
                'Env':identity['env'],'Labels':{'panda.campaign':G.CAMPAIGN_ID,
                    'panda.launch-token':'token','panda.source':'release'}},
            'State':{'Running':running,'ExitCode':0 if running else 1}}
        E.save(root/G.OUTPUT/'launcher_s2/launch.json',receipt)
        return receipt,current

    def test_duplicate_launch_is_idempotent_and_modified_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(E,'ROOT',Path(tmp)), \
                patch.object(E,'verify_sources',return_value='release'), patch.object(E.subprocess,'run') as run:
            _,current=self.fixture(Path(tmp))
            with patch.object(E,'inspect',return_value=current):
                self.assertEqual(E.launch(None)['status'],'ALREADY_RUNNING')
            run.assert_not_called()
            bad=deepcopy(current);bad['Config']['Cmd']=['wrong-worker']
            with patch.object(E,'inspect',return_value=bad),self.assertRaises(ValueError):E.launch(None)

    def test_interrupted_launch_adopts_only_its_authenticated_container(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(E,'ROOT',Path(tmp)), \
                patch.object(E,'verify_sources',return_value='release'), patch.object(E.subprocess,'run') as run:
            receipt,current=self.fixture(Path(tmp))
            receipt.pop('container_id')
            E.save(E.control_dir()/'launch_intent.json',receipt)
            (E.control_dir()/'launch.json').unlink()
            with patch.object(E,'inspect',return_value=current):
                self.assertEqual(E.launch(None)['status'],'ALREADY_RUNNING')
            self.assertEqual(E.read(E.control_dir()/'launch.json')['container_id'],'container')
            run.assert_not_called()

    def test_resume_uses_original_container_without_new_cost_or_training_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(E,'ROOT',Path(tmp)), \
                patch.object(E,'verify_sources',return_value='release'), patch.object(E.subprocess,'run') as run, \
                patch.object(E,'wait_for_gpu') as idle:
            _,current=self.fixture(Path(tmp),running=False)
            with patch.object(E,'inspect',return_value=current):
                with self.assertRaisesRegex(ValueError,'--resume'):E.launch(None)
                self.assertEqual(E.launch(None,resume=True)['status'],'RESUMED')
            idle.assert_called_once_with(False)
            run.assert_called_once_with(['docker','start',E.NAME],check=True)

    def test_worker_requires_fresh_verified_final_sheet_readback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(G,'ROOT',Path(tmp)), \
                patch('panda_qb_s2_frozen.release.verify'), patch('panda_qb_s2_frozen.prerequisites.terminal',
                    return_value={'campaign_status':'FINISHED_REGISTERED_QUEUE','completed_count':22}):
            root=Path(tmp);self.fixture(root)
            folder=root/G.OUTPUT/G.CAMPAIGN_ID/'s2/QB'
            upload=folder/'sheet_publish_panda_qb_s2_frozen/terminal_upload.json'
            E.save(upload,{'campaign_status':'FINISHED_REGISTERED_QUEUE','readback_verified':True,'old':True})
            publisher=Mock(pid=1,returncode=None);publisher.poll.return_value=None
            trainer=Mock(pid=2,returncode=0);trainer.poll.return_value=0
            def tick(_):
                self.assertFalse(upload.exists())
                E.save(upload,{'campaign_status':'FINISHED_REGISTERED_QUEUE','readback_verified':True})
            with patch.object(worker.subprocess,'Popen',side_effect=[publisher,trainer]) as spawn, \
                    patch.object(worker.time,'sleep',side_effect=tick):
                self.assertEqual(worker.run(),0)
            self.assertEqual(len(list(upload.parent.glob('terminal_upload_previous_*.json'))),1)
            self.assertTrue(E.read(folder/'worker_status.json')['sheet_verified'])
            self.assertEqual(spawn.call_args_list[0].kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            publisher.terminate.assert_called_once()
            trainer.send_signal.assert_not_called()

    def test_worker_stops_training_if_publisher_process_dies(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(G,'ROOT',Path(tmp)), \
                patch('panda_qb_s2_frozen.release.verify'):
            root=Path(tmp);self.fixture(root)
            publisher=Mock(pid=1,returncode=1);publisher.poll.return_value=1
            trainer=Mock(pid=2,returncode=None);trainer.poll.return_value=None
            with patch.object(worker.subprocess,'Popen',side_effect=[publisher,trainer]), \
                    self.assertRaisesRegex(RuntimeError,'Sheet publisher exited'):
                worker.run()
            trainer.send_signal.assert_called_with(signal.SIGTERM)
            trainer.wait.assert_called_once()
            state=E.read(root/G.OUTPUT/G.CAMPAIGN_ID/'s2/QB/worker_status.json')
            self.assertEqual(state['stage'],'FAILED')


if __name__=='__main__':unittest.main()
