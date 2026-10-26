import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_sensor5 import migration


class TestSensorMigration(unittest.TestCase):
    def fixture(self, root):
        directory = migration.old_root(root, 's1')
        run = directory / 'runs/old_run/attempt_01'
        (run / 'models').mkdir(parents=True)
        checkpoint = run / 'models/stop_0000012.pt'
        checkpoint.write_bytes(b'unchanged original checkpoint')
        artifact = {'checkpoint_path': str(checkpoint), 'completed_update': 12,
            'campaign_resume_complete': True, 'checkpoint_verified': True,
            'file_sha256': migration.sha(checkpoint)}
        (run / 'models/latest_resume.json').write_text(json.dumps(artifact))
        status = {'run_id': 'old_run', 'status': 'STOPPED', 'step': 12,
            'directory': str(run), 'poisoned': False}
        (run / 'run_state.json').write_text(json.dumps(status))
        state = {'status': 'STOPPED_SAFE', 'reason': 'STOP_REQUESTED', 'runs': [status]}
        (directory / 'campaign_state.json').write_text(json.dumps(state))
        container = {'Id': 'old-container', 'State': {'Running': False,
            'Status': 'exited', 'ExitCode': 1, 'OOMKilled': False}}
        return directory, run, container

    def test_intentional_cli_exit_one_requires_complete_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, run, container = self.fixture(root)
            before = (directory / 'campaign_state.json').read_bytes()
            proof = migration.verify_old_campaign(root, 's1', container)
            self.assertEqual(proof['stopped_runs'][0]['step'], 12)
            self.assertEqual(proof['container_exit_code'], 1)
            self.assertEqual((directory / 'campaign_state.json').read_bytes(), before)
            (run / 'models/stop_0000012.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'verified complete'):
                migration.verify_old_campaign(root, 's1', container)

    def test_failed_or_poisoned_run_cannot_be_relabelled_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, _, container = self.fixture(root)
            path = directory / 'campaign_state.json'
            state = json.loads(path.read_text())
            state['runs'][0]['poisoned'] = True
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'unsafe run'):
                migration.verify_old_campaign(root, 's1', container)
            state['runs'][0]['poisoned'] = False
            state['status'] = 'FAILED'
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'not safely stopped'):
                migration.verify_old_campaign(root, 's1', container)

    def test_process_filter_never_targets_other_server_wv3_or_new_writer(self):
        root = Path('/repo')
        argv = ['python', '-m', 'panda_sensor5.queue', '--server', 's1']
        self.assertTrue(migration.process_matches(argv, root, root, 's1'))
        self.assertFalse(migration.process_matches(argv, root, root, 's3'))
        argv[2] = 'panda_sensor5.wv3'
        self.assertFalse(migration.process_matches(argv, root, root, 's1'))
        writer = ['python', '-m', 'panda_sensor5.best_publication', '--server', 's1',
            '--root', str(migration.old_root(root, 's1')), '--apply', '--watch']
        self.assertTrue(migration.process_matches(writer, root, root, 's1'))
        writer[6] = '/repo/work_dir/new'
        self.assertFalse(migration.process_matches(writer, root, root, 's1'))

    def test_only_sensor_wait_is_replaced_wv3_proof_remains(self):
        chain = {'stage': 'WAIT_SENSOR5', 'sensor_container': migration.old_container_name('s3'),
            'request': 'original-cutover', 'panda_tune30_resume': {'preserve': True}}
        updated = migration.replacement_chain(chain, 's3', 'new-campaign')
        self.assertEqual(updated['stage'], 'START_SENSOR5')
        self.assertEqual(updated['panda_tune30_resume'], chain['panda_tune30_resume'])
        self.assertEqual(updated['request'], 'original-cutover')
        self.assertEqual(chain['stage'], 'WAIT_SENSOR5')
        with self.assertRaisesRegex(ValueError, 'WV3 continuation already advanced'):
            migration.replacement_chain({'stage': 'RESUME_HARD36'}, 's3', 'new-campaign')

    def test_apply_is_idempotent_and_preserves_old_evidence_and_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, _, container = self.fixture(root)
            (root / 'gspread').mkdir()
            (root / 'gspread/server.txt').write_text('s1')
            directory = root / 'work_dir/panda_sensor5_launchers/s1'
            directory.mkdir(parents=True)
            initial = directory / 'initial_cutover.json'
            initial.write_text('{"original":"WV3 receipt"}')
            chain = {'stage': 'WAIT_SENSOR5', 'sensor_container': migration.old_container_name('s1')}
            (directory / 'chain.json').write_text(json.dumps(chain))
            old_state = (old / 'campaign_state.json').read_bytes()
            def inspect(name):
                return container if name == migration.old_container_name('s1') else None
            with patch('panda_sensor5.contract.CAMPAIGN_ID', 'new-campaign'), \
                    patch.object(migration, 'matching_processes', return_value=[]), \
                    patch.object(migration, 'inspect_optional', side_effect=inspect), \
                    patch.object(migration, 'predecessor_receipt', return_value={}):
                first = migration.migrate('s1', root=root, apply=True)
                receipt = (directory / migration.RECEIPT).read_bytes()
                second = migration.migrate('s1', root=root, apply=True)
            self.assertEqual(first['status'], 'MIGRATED_READY_TO_ENQUEUE')
            self.assertEqual(second['status'], 'ALREADY_MIGRATED')
            self.assertEqual((directory / migration.RECEIPT).read_bytes(), receipt)
            self.assertEqual(json.loads((directory / 'chain_before_rgb_l4_v2.json').read_text()), chain)
            self.assertEqual(initial.read_text(), '{"original":"WV3 receipt"}')
            self.assertEqual((old / 'campaign_state.json').read_bytes(), old_state)


if __name__ == '__main__':
    unittest.main()
