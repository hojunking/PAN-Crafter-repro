"""Offline worker tests: temporary runs, mocked subprocesses; no cron/Sheet writes."""
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from fh12.common import atomic_json, read_json, sha256
from reporting_extra import worker as W


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='extra-worker-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = 's1'
        self.run = 'FH20R1_S1_S_SYNTHETIC'

    def ready(self, run=None, *, official=True, complete=True, updates=50000, stamp=1):
        run = run or self.run
        wd = self.root / 'work_dir' / run
        status = wd / 'official/postrun_status.json'
        atomic_json(status, dict(official_complete=official, actual_updates=updates))
        atomic_json(wd / 'meta/training_status.json', dict(training_complete=complete, actual_updates=updates))
        os.utime(status, ns=(stamp, stamp))
        return wd

    def receipt(self, run=None, **overrides):
        run = run or self.run
        folder = self.root / 'work_dir' / run / 'supplemental_metrics'
        atomic_json(folder / 'report.json', {'run_id': run, 'server_id': self.server,
                                           'campaign_id': 'fixture', 'fixture': True})
        value = dict(schema='PAN_SUPPLEMENTAL_UPLOAD_v1', run_id=run,
                     server_id=self.server, gid=994031662, campaign_id='fixture', row=4,
                     readback_verified=True, report_sha256=sha256(folder / 'report.json'))
        value.update(overrides)
        atomic_json(folder / 'upload_receipt.json', value)
        return value

    def passing_job(self, root, server, run):
        self.assertEqual(Path(root), self.root)
        self.assertEqual(server, self.server)
        self.receipt(run)
        return 0


class AdmissionTests(Fixture):
    def test_only_local_completed_fh_runs(self):
        self.ready(stamp=10)
        old = 'FH12_S1_T_SYNTHETIC'
        self.ready(old, stamp=1)
        for run, values in (
            ('FH20R1_S2_S_REMOTE', {}),
            ('FH20R1_S1_S_TRAINING', {'complete': False}),
            ('FH20R1_S1_S_NOT50K', {'updates': 49999}),
            ('FH20R1_S1_S_UNOFFICIAL', {'official': False}),
            ('ANOTHER_S1_S_RUN', {}),
        ):
            self.ready(run, **values)
        self.assertEqual(W.ready_runs(self.root, self.server), [self.run, old])

    def test_reuse_links_missing_and_malformed_status_are_excluded(self):
        self.ready()
        reused = self.ready('FH20R1_S1_S_REUSED')
        atomic_json(reused / 'meta/reuse_reference.json', {'reused': True})
        incomplete = self.ready('FH12_S1_T_MISSING')
        (incomplete / 'meta/training_status.json').unlink()
        broken = self.ready('FH12_S1_T_BROKEN')
        (broken / 'official/postrun_status.json').write_text('{bad json')
        self.assertEqual(W.ready_runs(self.root, self.server), [self.run])

    def test_state_path_rejects_path_traversal(self):
        for server in ('../s1', 's0', 's6', '/tmp', ''):
            with self.subTest(server=server), self.assertRaises(ValueError):
                W.state_dir(self.root, server)

    def test_process_rejects_bad_or_remote_run_before_creating_outputs(self):
        for run in ('../run', '/tmp/run', 'FH20R1_S1_a/b', 'FH20R1_S1_'):
            with self.subTest(run=run), self.assertRaises(ValueError):
                W.process_run(self.root, run)
        with patch('tools.fh12_runner.detect_server', return_value='s1'):
            with self.assertRaisesRegex(ValueError, 'different server'):
                W.process_run(self.root, 'FH20R1_S2_S_OTHER')
        self.assertFalse((self.root / 'work_dir').exists())

    def test_per_run_lock_prevents_double_processing(self):
        folder = self.root / 'work_dir' / self.run / 'supplemental_metrics'
        folder.mkdir(parents=True)
        with (folder / '.job.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch('tools.fh12_runner.detect_server', return_value='s1'):
                with self.assertRaises(BlockingIOError):
                    W.process_run(self.root, self.run)


class RetryTests(Fixture):
    def test_success_requires_receipt_and_skips_unchanged_job(self):
        wd = self.ready()
        official_before = (wd / 'official/postrun_status.json').read_bytes()
        training_before = (wd / 'meta/training_status.json').read_bytes()
        calls = []
        def job(*args):
            calls.append(args[-1])
            return self.passing_job(*args)
        state = W.tick(self.root, self.server, {}, job=job, now=lambda: 100.)
        self.assertEqual(state['jobs'][self.run]['status'], 'VERIFIED')
        self.assertEqual(state['jobs'][self.run]['attempts'], 1)
        W.tick(self.root, self.server, state, job=job, now=lambda: 10000.)
        self.assertEqual(calls, [self.run])
        self.assertEqual((wd / 'official/postrun_status.json').read_bytes(), official_before)
        self.assertEqual((wd / 'meta/training_status.json').read_bytes(), training_before)
        self.assertEqual(read_json(W.state_dir(self.root, self.server) / 'status.json'), state)

    def test_failure_backoff_then_successful_retry(self):
        self.ready()
        state = W.tick(self.root, self.server, {}, job=lambda *_: 7, now=lambda: 100.)
        row = state['jobs'][self.run]
        self.assertEqual(row['status'], 'RETRY_PENDING')
        self.assertEqual(row['retry_after'], 100. + W.RETRY_SECONDS)
        with patch.object(W, 'run_job', side_effect=AssertionError('must not run')) as job:
            W.tick(self.root, self.server, state, job=job, now=lambda: 101.)
            job.assert_not_called()
        W.tick(self.root, self.server, state, job=self.passing_job, now=lambda: 100. + W.RETRY_SECONDS)
        self.assertEqual(state['jobs'][self.run]['status'], 'VERIFIED')
        self.assertEqual(state['jobs'][self.run]['attempts'], 2)

    def test_exception_isolated_and_next_run_is_processed(self):
        self.ready(stamp=20)
        older = 'FH12_S1_T_OLDER'
        self.ready(older, stamp=1)
        def job(root, server, run):
            if run == self.run:
                raise RuntimeError('synthetic offline failure')
            return self.passing_job(root, server, run)
        state = W.tick(self.root, self.server, {}, job=job, now=lambda: 100.)
        self.assertEqual(state['jobs'][self.run]['status'], 'RETRY_PENDING')
        self.assertEqual(state['jobs'][older]['status'], 'VERIFIED')
        self.assertIsNone(state['active_run'])

    def test_zero_exit_without_receipt_is_not_success(self):
        self.ready()
        state = W.tick(self.root, self.server, {}, job=lambda *_: 0, now=lambda: 100.)
        self.assertEqual(state['jobs'][self.run]['status'], 'RETRY_PENDING')

    def test_invalid_receipts_are_not_verified(self):
        for fields in ({'readback_verified': False}, {'run_id': 'wrong-run'}, {'report_sha256': 'wrong-hash'},
                       {'schema': 'wrong-schema'}, {'server_id': 's2'}, {'gid': 12345},
                       {'campaign_id': 'wrong-campaign'}):
            with self.subTest(fields=fields):
                self.ready()
                def job(*_):
                    self.receipt(**fields)
                    return 0
                state = W.tick(self.root, self.server, {}, job=job, now=lambda: 100.)
                self.assertEqual(state['jobs'][self.run]['status'], 'RETRY_PENDING')

    def test_report_tampering_after_receipt_is_detected(self):
        self.ready()
        def job(*_):
            self.receipt()
            atomic_json(self.root / 'work_dir' / self.run / 'supplemental_metrics/report.json', {'tampered': True})
            return 0
        state = W.tick(self.root, self.server, {}, job=job, now=lambda: 100.)
        self.assertEqual(state['jobs'][self.run]['status'], 'RETRY_PENDING')

    def test_changed_official_report_or_code_invalidates_skip_and_backoff(self):
        wd = self.ready()
        state = W.tick(self.root, self.server, {}, job=self.passing_job, now=lambda: 100.)
        atomic_json(wd / 'official/raw_max.json', {'different_selected_checkpoint': True})
        W.tick(self.root, self.server, state, job=lambda *_: 7, now=lambda: 101.)
        self.assertEqual(state['jobs'][self.run]['attempts'], 2)
        code = self.root / 'reporting_extra/evaluation.py'
        code.parent.mkdir()
        code.write_text('# new reporting implementation\n')
        W.tick(self.root, self.server, state, job=self.passing_job, now=lambda: 102.)
        self.assertEqual(state['jobs'][self.run]['attempts'], 3)
        self.assertEqual(state['jobs'][self.run]['status'], 'VERIFIED')

    def test_interrupted_running_job_is_retried(self):
        self.ready()
        state = {'jobs': {self.run: dict(status='RUNNING', attempts=3,
                                        fingerprint=W.job_fingerprint(self.root, self.run))}}
        W.tick(self.root, self.server, state, job=self.passing_job)
        self.assertEqual(state['jobs'][self.run]['attempts'], 4)
        self.assertEqual(state['jobs'][self.run]['status'], 'VERIFIED')


class LaunchTests(Fixture):
    def test_cpu_environment_does_not_mutate_parent(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '17'}):
            env = W.cpu_environment()
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['OMP_NUM_THREADS'], '2')
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '17')

    def test_run_job_uses_isolated_cpu_subprocess_and_per_run_log(self):
        W.state_dir(self.root, self.server).mkdir(parents=True)
        with patch.object(W.subprocess, 'run', return_value=types.SimpleNamespace(returncode=19)) as run:
            self.assertEqual(W.run_job(self.root, self.server, self.run), 19)
        args, kwargs = run.call_args
        self.assertEqual(args[0][-5:], ['process', '--server', self.server, '--run', self.run])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
        self.assertEqual(kwargs['cwd'], self.root)
        self.assertTrue((W.state_dir(self.root, self.server) / (self.run + '.log')).is_file())

    def test_duplicate_worker_lock_stops_watch_and_reports_running(self):
        folder = W.state_dir(self.root, self.server)
        folder.mkdir(parents=True)
        with (folder / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(W.running(self.root, self.server))
            with patch.object(W, 'tick', side_effect=AssertionError('duplicate worker tick')):
                self.assertEqual(W.watch(self.root, self.server, once=True), {'status': 'ALREADY_RUNNING'})
        self.assertFalse(W.running(self.root, self.server))

    def test_watch_once_does_not_sleep_or_change_real_process_priority(self):
        def fake_tick(root, server, state):
            state.update(seen_server=server)
        with patch.object(W.os, 'nice') as nice, patch.object(W, 'tick', side_effect=fake_tick) as tick, \
                patch.object(W.time, 'sleep', side_effect=AssertionError('once must not sleep')):
            result = W.watch(self.root, self.server, once=True)
        self.assertEqual(result['seen_server'], self.server)
        nice.assert_called_once_with(15)
        tick.assert_called_once()

    def test_unenabled_automatic_start_has_no_launch_or_cron(self):
        with patch.object(W.subprocess, 'Popen') as child, patch.object(W, 'install_cron') as cron:
            self.assertEqual(W.start(self.root, self.server, if_enabled=True), {'status': 'NOT_ENABLED'})
        child.assert_not_called()
        cron.assert_not_called()
        self.assertFalse((W.state_dir(self.root, self.server) / 'enabled.json').exists())

    def test_explicit_start_survives_cron_failure_and_detaches_cpu_worker(self):
        with patch.object(W, 'install_cron', side_effect=RuntimeError('cron unavailable')), \
                patch.object(W.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=12345)) as child:
            result = W.start(self.root, self.server)
        self.assertEqual(result['status'], 'STARTED')
        self.assertEqual(result['restart_warning'], 'cron unavailable')
        self.assertTrue((W.state_dir(self.root, self.server) / 'enabled.json').is_file())
        kwargs = child.call_args.kwargs
        self.assertTrue(kwargs['start_new_session'])
        self.assertTrue(kwargs['close_fds'])
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_existing_worker_does_not_spawn_again(self):
        with patch.object(W, 'running', return_value=True), patch.object(W.subprocess, 'Popen') as child:
            result = W.start(self.root, self.server, cron=False)
        self.assertEqual(result['status'], 'ALREADY_RUNNING')
        child.assert_not_called()


class CronTests(Fixture):
    def fake_cron(self, stdout='', stderr='', returncode=0):
        writes = []
        def runner(args, **kwargs):
            if args == ['crontab', '-l']:
                return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)
            self.assertEqual(args, ['crontab', '-'])
            self.assertTrue(kwargs['check'])
            writes.append(kwargs['input'])
            return types.SimpleNamespace(returncode=0)
        return runner, writes

    def test_install_preserves_training_other_servers_comments_and_environment(self):
        keep = ['SHELL=/bin/bash', '# original notes', '',
                '*/15 * * * * /repo/tools/_watchdog.sh # PANCRAFTER-WATCHDOG',
                '@reboot /repo/start-existing.sh',
                '*/5 * * * * old-s2 # PANCRAFTER-EXTRA-METRICS-s2']
        old = keep + ['*/5 * * * * old-s1 # PANCRAFTER-EXTRA-METRICS-s1']
        runner, writes = self.fake_cron('\n'.join(old) + '\n')
        W.install_cron(self.root, self.server, runner=runner)
        self.assertEqual(writes[0].splitlines()[:-1], keep)
        line = writes[0].splitlines()[-1]
        self.assertIn('start --server s1 --if-enabled', line)
        self.assertTrue(line.endswith('# PANCRAFTER-EXTRA-METRICS-s1'))
        runner2, writes2 = self.fake_cron(writes[0])
        W.install_cron(self.root, self.server, runner=runner2)
        self.assertEqual(writes2, writes)

    def test_missing_crontab_is_allowed_but_read_errors_never_overwrite(self):
        runner, writes = self.fake_cron(stderr='no crontab for fixture-user', returncode=1)
        W.install_cron(self.root, self.server, runner=runner)
        self.assertEqual(len(writes), 1)
        for code, error in ((1, 'permission denied'), (2, 'service unavailable')):
            with self.subTest(code=code):
                runner, writes = self.fake_cron(stderr=error, returncode=code)
                with self.assertRaises(RuntimeError):
                    W.install_cron(self.root, self.server, runner=runner)
                self.assertEqual(writes, [])

    def test_cron_quotes_paths_with_spaces(self):
        root = self.root / 'repo with spaces'
        runner, writes = self.fake_cron()
        W.install_cron(root, self.server, runner=runner)
        self.assertIn("'" + str(root / 'tools/extra_metrics.py') + "'", writes[0])


class SourceScopeTests(Fixture):
    def test_additive_worker_and_watchdog_are_outside_fh12_fh20_identity(self):
        from fh12.common import source_identity as fh12_identity
        from fh20r1.common import source_identity as fh20_identity
        for name in ('fh12/synthetic.py', 'fh20r1/synthetic.py', 'tools/fh20r1_synthetic.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# unchanged inference source\n')
        with patch('fh12.common.subprocess.check_output', return_value='frozen-git-head\n'):
            before = fh12_identity(self.root), fh20_identity(self.root)
            for name in ('reporting_extra/worker.py', 'reporting_extra/evaluation.py',
                         'tools/extra_metrics.py', 'tools/extra_metrics_start.sh', 'tools/_watchdog.sh'):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# additive reporting implementation\n')
            after = fh12_identity(self.root), fh20_identity(self.root)
        self.assertEqual(before, after)
        self.assertNotIn('tools/_watchdog.sh', after[1]['files'])


if __name__ == '__main__':
    unittest.main()
