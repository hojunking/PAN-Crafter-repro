"""Host launcher ownership and independent-start checks; no Docker/GPU mutations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_structure import enqueue as E


class EnqueueTests(unittest.TestCase):
    def identity(self):
        return {'source_container': 'original', 'image': 'sha256:original', 'user': '1000:1000',
                'working_dir': '/workspace', 'env': ['B=2', 'A=1'],
                'mounts': [{'Source': '/repo', 'Destination': '/workspace', 'RW': False}]}

    def current(self, server, identity):
        return {'Id': 'owned', 'Image': identity['image'], 'Mounts': deepcopy(identity['mounts']),
            'Config': {'Cmd': E.training_args(server), 'User': identity['user'],
                'WorkingDir': identity['working_dir'], 'Env': list(reversed(identity['env'])),
                'Labels': {'panda.campaign': E.G.CAMPAIGN_ID, 'panda.launch-token': 'token'}},
            'State': {'Running': True}}

    def test_container_ownership_and_server_are_not_interchangeable(self):
        identity = self.identity(); intent = {'identity': identity, 'token': 'token'}
        current = self.current('s3', identity)
        self.assertEqual(E.authenticate_container('s3', current, intent)['Id'], 'owned')
        with self.assertRaises(ValueError): E.authenticate_container('s1', current, intent)
        altered = deepcopy(current); altered['Mounts'][0]['RW'] = True
        with self.assertRaises(ValueError): E.authenticate_container('s3', altered, intent)
        altered = deepcopy(current); altered['Config']['Labels']['panda.launch-token'] = 'foreign'
        with self.assertRaises(ValueError): E.authenticate_container('s3', altered, intent)
        current['State'] = {'Running': False}
        with self.assertRaises(ValueError): E.authenticate_container('s3', current, intent)
        self.assertEqual(E.authenticate_container('s3', current, intent, {'container_id': 'owned'})['Id'], 'owned')

    def test_both_servers_mount_qb_data_read_only_and_only_new_output_writable(self):
        source = {'Image': 'sha256:source', 'Config': {'User': '1000:1000', 'WorkingDir': '/workspace', 'Env': []},
            'Mounts': [
                {'Type': 'bind', 'Source': '/repo', 'Destination': '/workspace', 'RW': False},
                {'Type': 'bind', 'Source': '/old', 'Destination': '/workspace/work_dir/old', 'RW': True}]}
        with tempfile.TemporaryDirectory() as temp, patch.object(E, 'ROOT', Path(temp)), \
             patch('panda_sensor_refine.launch.source_container', return_value=('original', source)), \
             patch('edgeres48.docker_mounts.targets', return_value=[Path('/data/qb-train.h5'), Path('/data/qb-val.h5')]):
            for server in ('s1', 's3'):
                cmd, identity = E.command(server, E.container_name(server), token='token')
                self.assertEqual(cmd[-1], 'sha256:source')
                self.assertFalse(any(m['Destination'].endswith('/old') for m in identity['mounts']))
                self.assertEqual(sum(m['RW'] for m in identity['mounts']), 1)
                self.assertTrue(all(not m['RW'] for m in identity['mounts'] if m['Destination'].startswith('/data/')))
                self.assertEqual(identity['mounts'][-1]['Destination'], '/workspace/' + E.G.OUTPUT)
                reordered = deepcopy(identity); reordered['mounts'].reverse()
                self.assertTrue(E.same_environment(identity, reordered))
                reordered['mounts'][0]['RW'] = not reordered['mounts'][0]['RW']
                self.assertFalse(E.same_environment(identity, reordered))

    def test_gpu_wait_only_retries_occupied_gpu(self):
        statuses = []
        with patch('directshift48.__main__.require_idle_gpu', side_effect=[
                RuntimeError('Active GPU compute workload; current'), None]) as check, patch.object(E.time, 'sleep') as sleep:
            E.wait_for_gpu(True, lambda stage, **extra: statuses.append(stage))
            self.assertEqual(check.call_count, 2); sleep.assert_called_once_with(30)
            self.assertEqual(statuses, ['WAIT_LOCAL_GPU'])
        with patch('directshift48.__main__.require_idle_gpu', side_effect=RuntimeError('CUDA driver failure')):
            with self.assertRaisesRegex(RuntimeError, 'driver'): E.wait_for_gpu(True, lambda *a, **k: None)

    def test_only_unstarted_intents_may_refresh(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(E, 'ROOT', Path(temp)), \
             patch.object(E, 'command', return_value=(['docker'], self.identity())):
            control = E.control_dir('s3'); control.mkdir(parents=True)
            path = control / 'launch_intent.json'
            stale = {'campaign_id': E.G.CAMPAIGN_ID, 'server': 's3', 'registry_digest': 'old',
                     'token': 'old-token', 'identity': self.identity()}
            path.write_text(json.dumps(stale))
            intent = E.launch_intent('s3', None, None, 'new')
            self.assertEqual(intent['registry_digest'], 'new')
            self.assertEqual(len(list((control / 'intent_history').glob('*.json'))), 1)
            with self.assertRaises(ValueError): E.launch_intent('s3', {'Id': 'started'}, None, 'different')
            folder = Path(temp) / E.G.OUTPUT / E.G.CAMPAIGN_ID / 's3/QB'
            folder.mkdir(parents=True)
            (folder / 'campaign_state.json').write_text(json.dumps({'runs': [{'status': 'RUNNING'}]}))
            with self.assertRaises(ValueError): E.launch_intent('s3', None, None, 'different')

    def test_completed_container_and_sheet_readback_without_predecessor_records(self):
        from panda_qb_structure import prerequisites
        with tempfile.TemporaryDirectory() as temp, patch.object(E, 'ROOT', Path(temp)):
            root = Path(temp); control = E.control_dir('s3'); control.mkdir(parents=True)
            identity = self.identity(); intent = {'identity': identity, 'token': 'token'}
            current = self.current('s3', identity)
            current['State'] = {'Running': False, 'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False}
            receipt = {'container_id': 'owned', 'container': E.container_name('s3')}
            (control / 'launch.json').write_text(json.dumps(receipt))
            upload = root / E.G.OUTPUT / E.G.CAMPAIGN_ID / 's3/QB/sheet_publish_panda_sensor_focus/terminal_upload.json'
            upload.parent.mkdir(parents=True)
            upload.write_text(json.dumps({'readback_verified': True, 'campaign_status': 'FINISHED_REGISTERED_QUEUE'}))
            finished = {'campaign_status': 'FINISHED_REGISTERED_QUEUE', 'completed_count': 8, 'early_stopped_count': 0}
            statuses = []
            with patch('panda_sensor_refine.launch.inspect_optional', return_value=current), \
                 patch.object(E, 'launch_intent', return_value=intent), \
                 patch.object(E, 'command', return_value=(['docker'], identity)), \
                 patch.object(E, 'ensure_publisher', return_value=42), \
                 patch.object(prerequisites, 'terminal', return_value=finished), \
                 patch('directshift48.__main__.require_idle_gpu') as gpu:
                self.assertEqual(E.execute('s3', False, lambda stage, **extra: statuses.append((stage, extra))), 0)
                gpu.assert_not_called()
            self.assertEqual(statuses[-1][0], 'COMPLETED')
            self.assertTrue(statuses[-1][1]['sheet_verified'])


if __name__ == '__main__':
    unittest.main()
