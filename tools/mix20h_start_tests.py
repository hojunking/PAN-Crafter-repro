"""Single-command startup tests: temporary files and mocked child launch only."""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import mix20h_runner as R
from tools import mix20h_launch_config as C
from kdv.mix20h_plan import cases_for


class StartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        R.write_text(self.root/'gspread/server.txt', 's1\n')
        R.write_text(self.root/'work_dir/cases_queue.txt', 'OLD_CASE\n')
        self.clock = 1000.
        self.hashes = {c.run_id: 'fixture_sha' for c in cases_for('s1')}
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(R.time, 'time', side_effect=lambda: self.clock))
        self.stack.enter_context(patch.object(C, 'ensure_configs', return_value={'unchanged': list(self.hashes)}))
        self.stack.enter_context(patch.object(R, 'validate_configs', return_value=self.hashes))
        self.assets = self.stack.enter_context(patch.object(R, 'verify_assets', return_value={'fixture': True}))
        self.stack.enter_context(patch.object(R, 'inventory', side_effect=self.inventory))
        self.launch = self.stack.enter_context(patch.object(R, 'launch_background', return_value={'launch_status': 'RUNNER_SUBMITTED', 'runner_pid': 123}))

    def inventory(self, root, server):
        files = {}
        for rel in R.LEGACY_FILES:
            if (root/rel).is_file():
                files[rel] = dict(content=(root/rel).read_text(), sha256=R.digest(root/rel))
        return dict(processes=[], files=files, new_runs={r: {'status': 'unstarted'} for r in self.hashes})

    def hold(self, reason='old evaluation'):
        R.write_json(self.root/'work_dir/_eval_phase/hold.json', dict(
            protocol_id=R.LEGACY_EVAL_PROTOCOL, server='s1', reason=reason, allow_new_main_training=False))

    def test_one_command_automatic_time_hold_and_launch(self):
        self.hold(); original = (self.root/'work_dir/_eval_phase/hold.json').read_bytes()
        result = R.start_campaign(self.root)
        self.assertEqual(result['clock_policy'], 'local_first_start_20h')
        self.assertEqual(R.timestamp(result['campaign_started_at_utc']), 1000.)
        self.assertEqual(R.timestamp(result['deadline_at_utc']), 73000.)
        self.assertFalse((self.root/'work_dir/_eval_phase/hold.json').exists())
        receipt = R.read_json(self.root/R.CAMP/'hold_takeover.json')
        self.assertEqual((self.root/receipt['archived_path']).read_bytes(), original)
        self.assertFalse(receipt['previous_evaluation_complete'])
        self.assertEqual(R.load_plan(self.root)['activation_state'], 'ready')
        self.launch.assert_called_once_with(self.root, upload=True)

    def test_dry_run_has_no_clock_hold_queue_or_launch_mutations(self):
        self.hold()
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = R.start_campaign(self.root, dry_run=True)
        self.assertEqual(result['status'], 'WOULD_START')
        self.assertTrue(result['legacy_hold_would_be_archived'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse((self.root/R.CAMP).exists())
        self.launch.assert_not_called()

    def test_restart_preserves_clock_and_new_hold(self):
        R.start_campaign(self.root)
        self.clock += 1000; self.hold('new explicitly requested diagnostic')
        R.start_campaign(self.root)
        self.assertEqual(R.timestamp(R.load_plan(self.root)['deadline_at_utc']), 73000.)
        self.assertTrue((self.root/'work_dir/_eval_phase/hold.json').exists())
        self.assertEqual(self.launch.call_count, 2)

    def test_existing_manifest_identifies_server_when_local_file_missing(self):
        R.start_campaign(self.root)
        (self.root/'gspread/server.txt').unlink()
        self.assertEqual(R.start_campaign(self.root)['server'], 's1')

    def test_expired_or_closed_campaign_does_not_reset_or_launch(self):
        R.start_campaign(self.root); self.launch.reset_mock(); self.clock = 74000.
        result = R.start_campaign(self.root)
        self.assertEqual(result['status'], 'BUDGET_CLOSED')
        self.assertEqual(R.timestamp(R.load_plan(self.root)['deadline_at_utc']), 73000.)
        self.launch.assert_not_called()

    def test_shared_clock_is_optional_and_cannot_be_changed_on_restart(self):
        result = R.start_campaign(self.root, start_at=R.iso(2000))
        self.assertEqual(result['clock_policy'], 'shared_explicit_20h')
        with self.assertRaisesRegex(ValueError, 'cannot reset'):
            R.start_campaign(self.root, start_at=R.iso(3000))

    def test_mismatched_assets_preserve_hold_and_do_not_activate(self):
        self.hold(); self.assets.side_effect = ValueError('fixture dataset mismatch')
        with self.assertRaisesRegex(ValueError, 'dataset mismatch'):
            R.start_campaign(self.root)
        self.assertTrue((self.root/'work_dir/_eval_phase/hold.json').exists())
        self.assertFalse((self.root/R.CAMP/'plan_manifest.json').exists())
        self.launch.assert_not_called()

    def test_hold_replaced_during_hashing_is_preserved_for_automatic_wait(self):
        self.hold()
        def replace(root, server):
            self.hold('new hold from another controller')
            return {'fixture': True}
        self.assets.side_effect = replace
        R.start_campaign(self.root)
        self.assertEqual(R.read_json(self.root/R.CAMP/'hold_takeover.json')['status'], 'new_hold_preserved')
        self.assertTrue((self.root/'work_dir/_eval_phase/hold.json').exists())
        self.assertEqual(R.load_plan(self.root)['activation_state'], 'ready')

    def test_interrupted_takeover_recovers_original_time_and_archive(self):
        self.hold(); original = R.write_json
        def crash(path, data):
            if Path(path).name == 'budget.json':
                raise OSError('fixture interruption')
            original(path, data)
        with patch.object(R, 'write_json', side_effect=crash), self.assertRaises(OSError):
            R.start_campaign(self.root)
        self.clock += 300
        R.start_campaign(self.root)
        self.assertEqual(R.timestamp(R.load_plan(self.root)['campaign_started_at_utc']), 1000.)
        self.assertFalse((self.root/'work_dir/_eval_phase/hold.json').exists())

    def test_resource_wait_includes_evaluator_between_gpu_batches(self):
        R.start_campaign(self.root); plan = R.load_plan(self.root); state = {}
        with patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'active_gpu_workers', return_value=[{'script': 'noa_eval.py'}]):
            self.assertFalse(R.wait_for_resources(self.root, plan, state, wait=False))
        self.assertEqual(state['status'], 'WAIT_RESOURCE')

    def test_wait_mode_resumes_without_watchdog(self):
        R.start_campaign(self.root); plan = R.load_plan(self.root); state = {}
        with patch.object(R, 'gpu_processes', side_effect=[['fixture GPU owner'], []]), patch.object(R, 'active_gpu_workers', return_value=[]), patch.object(R.time, 'sleep') as sleep:
            self.assertTrue(R.wait_for_resources(self.root, plan, state, wait=True))
        sleep.assert_called_once_with(15)

    def test_wrong_server_and_unknown_hold_are_not_silently_overridden(self):
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            R.start_campaign(self.root, server='s2')
        R.write_json(self.root/'work_dir/_eval_phase/hold.json', {'reason': 'unknown owner'})
        with self.assertRaisesRegex(ValueError, 'unrecognized'):
            R.start_campaign(self.root)
        self.launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
