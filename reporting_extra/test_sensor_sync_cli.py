"""Offline CLI orchestration tests: no network, Sheet writes, GPU, or sleeps."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


_SPEC = importlib.util.spec_from_file_location(
    'sensor_sheet_sync_cli_test_target',
    Path(__file__).resolve().parents[1] / 'tools/sensor_sheet_sync.py')
CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CLI)


class SensorSyncCliTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path('/tmp/sensor-sheet-cli-read-only-fixture')
        self.run = self.root / 'work_dir' / 'G20_GF2_fixture'
        self.before = Mock(name='worksheet_before_column_migration')
        self.after = Mock(name='worksheet_after_column_migration')
        self.book = Mock(name='spreadsheet')
        self.book.worksheet.side_effect = [self.before, self.after]
        self.ready = self.stack.enter_context(patch.object(CLI, 'ready_runs', return_value=[self.run]))
        self.connect = self.stack.enter_context(patch.object(CLI, 'connect', return_value=self.book))
        self.sync = self.stack.enter_context(patch.object(CLI, 'sync_run', return_value={'status': 'PLANNED'}))
        self.sleep = self.stack.enter_context(patch.object(CLI.time, 'sleep', side_effect=KeyboardInterrupt))
        self.repair = self.stack.enter_context(patch(
            'reporting_extra.sensor_repair_layout.repair_layout',
            return_value={'schema': 'GF2_LAYOUT_REPAIR_v1', 'apply': False,
                          'batch_requests': ['not emitted'], 'final_headers': ['not emitted']}))
        self.output = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def invoke(self, *extra):
        return CLI.main(['--root', str(self.root), '--server', 's1', *extra])

    def lines(self):
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def test_default_dry_run_has_no_apply_or_layout(self):
        self.assertEqual(self.invoke(), 0)
        self.sync.assert_called_once_with(self.run, 's1', self.before, apply=False)
        self.repair.assert_not_called()
        self.sleep.assert_not_called()
        self.assertFalse(any('update' in name or 'add_' in name or 'delete' in name
                             for name, _args, _kwargs in self.book.mock_calls))

    def test_layout_alone_runs_without_ready_runs(self):
        self.ready.return_value = []
        self.assertEqual(self.invoke('--repair-layout'), 0)
        self.repair.assert_called_once_with(self.before, self.root, 's1', apply=False)
        self.sync.assert_not_called()
        self.assertEqual(self.book.worksheet.call_count, 2)
        result = self.lines()[0]
        self.assertFalse(result['apply'])
        self.assertNotIn('batch_requests', result)
        self.assertNotIn('final_headers', result)

    def test_no_ready_and_no_layout_does_not_connect(self):
        self.ready.return_value = []
        self.assertEqual(self.invoke(), 0)
        self.connect.assert_not_called()
        self.repair.assert_not_called()
        self.sync.assert_not_called()

    def test_layout_dry_run_remains_read_only_for_both_operations(self):
        self.assertEqual(self.invoke('--repair-layout'), 0)
        self.repair.assert_called_once_with(self.before, self.root, 's1', apply=False)
        self.sync.assert_called_once_with(self.run, 's1', self.after, apply=False)

    def test_apply_uses_refreshed_geometry_after_column_migration(self):
        self.assertEqual(self.invoke('--repair-layout', '--apply'), 0)
        self.repair.assert_called_once_with(self.before, self.root, 's1', apply=True)
        self.sync.assert_called_once_with(self.run, 's1', self.after, apply=True)
        self.assertEqual(self.book.worksheet.call_count, 2)

    def test_layout_failure_exits_without_metadata_updates(self):
        self.repair.side_effect = ValueError('protected range; fail closed')
        self.assertEqual(self.invoke('--repair-layout', '--apply'), 2)
        self.sync.assert_not_called()
        self.assertEqual(self.lines()[-1]['status'], 'LAYOUT_PENDING')
        self.assertEqual(self.book.worksheet.call_count, 1)

    def test_refresh_failure_does_not_use_stale_worksheet(self):
        self.book.worksheet.side_effect = [self.before, ValueError('refresh failed')]
        self.assertEqual(self.invoke('--repair-layout', '--apply'), 2)
        self.sync.assert_not_called()
        self.assertEqual(self.lines()[-1]['status'], 'LAYOUT_PENDING')

    def test_connection_or_ready_failure_is_nonmutating(self):
        self.connect.side_effect = RuntimeError('unavailable')
        self.assertEqual(self.invoke('--repair-layout', '--apply'), 2)
        self.repair.assert_not_called()
        self.sync.assert_not_called()
        self.assertEqual(self.lines()[-1]['status'], 'REPORTING_PENDING')
        self.connect.reset_mock(side_effect=True)
        self.ready.side_effect = ValueError('invalid local provenance')
        self.assertEqual(self.invoke('--repair-layout', '--apply'), 2)
        self.connect.assert_not_called()
        self.repair.assert_not_called()
        self.sync.assert_not_called()

    def test_watch_repairs_once_and_reuses_refreshed_worksheet(self):
        self.sleep.side_effect = [None, KeyboardInterrupt]
        with self.assertRaises(KeyboardInterrupt):
            self.invoke('--repair-layout', '--apply', '--watch', '60')
        self.assertEqual(self.ready.call_count, 2)
        self.repair.assert_called_once_with(self.before, self.root, 's1', apply=True)
        self.assertEqual(self.sync.call_count, 2)
        self.assertTrue(all(call.args[2] is self.after for call in self.sync.call_args_list))
        self.assertEqual(self.book.worksheet.call_count, 2)
        self.assertTrue(all(call.args == (60.,) for call in self.sleep.call_args_list))

    def test_watch_retries_failed_layout_before_any_metadata_sync(self):
        self.repair.side_effect = [ValueError('blocked'), {'readback_verified': True}]
        self.sleep.side_effect = [None, KeyboardInterrupt]
        with self.assertRaises(KeyboardInterrupt):
            self.invoke('--repair-layout', '--apply', '--watch')
        self.assertEqual(self.repair.call_count, 2)
        self.sync.assert_called_once_with(self.run, 's1', self.after, apply=True)
        self.assertEqual(self.lines()[0]['status'], 'LAYOUT_PENDING')

    def test_short_watch_interval_rejected_before_any_inspection(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke('--watch', '59')
        self.assertEqual(raised.exception.code, 2)
        self.ready.assert_not_called()
        self.connect.assert_not_called()
        self.sync.assert_not_called()


if __name__ == '__main__':
    unittest.main()
