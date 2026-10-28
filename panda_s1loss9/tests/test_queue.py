import json
import fcntl
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from panda_s1loss9 import queue, orchestration


class PriorityQueueTests(unittest.TestCase):
    def test_old_watcher_lock_waits_for_actual_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            with path.open('a+') as old, path.open('a+') as new:
                fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
                timer = threading.Timer(.03, lambda: fcntl.flock(old, fcntl.LOCK_UN))
                timer.start()
                try:
                    queue.acquire_old_watcher_lock(new, timeout=1.)
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    timer.join()

    def test_lock_timeout_retains_stop_history_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stopped = [{'pid': 123, 'generation': '456', 'argv': ['old-watcher'], 'cwd': tmp}]
            queue.preserve_held_watchers(directory, 's1', stopped)
            with (directory / 'lock').open('a+') as old, (directory / 'lock').open('a+') as new:
                fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'trainer untouched'):
                    queue.acquire_old_watcher_lock(new, timeout=.01)
            self.assertEqual(queue.preserve_held_watchers(directory, 's1', []), stopped)
            value = json.loads((directory / 'held_watchers.json').read_text())
            self.assertEqual(len(value['attempts']), 2)

    def test_only_same_server_same_checkout_queue_process_matches(self):
        argv = ['python', '-m', 'panda_sensor5.queue', '--server', 's1', '--pin', '/tmp/pin']
        self.assertTrue(queue.process_matches(argv, '/tmp/repo', '/tmp/repo', 's1'))
        self.assertFalse(queue.process_matches(argv, '/tmp/other', '/tmp/repo', 's1'))
        self.assertFalse(queue.process_matches(argv, '/tmp/repo', '/tmp/repo', 's3'))
        for module in ('panda_sensor5', 'panda_sensor5.best_publication', 'panda_s1loss9.queue'):
            changed = [module if x == 'panda_sensor5.queue' else x for x in argv]
            self.assertFalse(queue.process_matches(changed, '/tmp/repo', '/tmp/repo', 's1'))

    def test_initial_boundary_requires_both_full_runs(self):
        cfgs = [{'run_id': 'donor', 'planned_horizon': 50000},
                {'run_id': 'base', 'planned_horizon': 30000}]
        with patch.object(orchestration, 'initial_configs', return_value=cfgs):
            state = {'runs': [{'run_id': 'donor', 'status': 'COMPLETED', 'step': 50000}]}
            self.assertFalse(orchestration.initial_complete(state, 's1'))
            state['runs'].append({'run_id': 'base', 'status': 'COMPLETED', 'step': 29999})
            self.assertFalse(orchestration.initial_complete(state, 's1'))
            state['runs'][1]['step'] = 30000
            self.assertTrue(orchestration.initial_complete(state, 's1'))
            state['runs'].append(dict(state['runs'][1]))
            self.assertFalse(orchestration.initial_complete(state, 's1'))

    def test_takeover_rejects_v1_and_already_resumed_wv3(self):
        from panda_sensor5.plan import CAMPAIGN_ID
        from panda_sensor5.launch import sensor_container_name
        chain = {'stage': 'WAIT_SENSOR5', 'sensor_campaign_id': CAMPAIGN_ID,
                 'sensor_container': sensor_container_name('s1')}
        state = {'status': 'RUNNING', 'runs': [{'run_id': 'd', 'status': 'RUNNING', 'step': 45000}]}
        self.assertEqual(queue.validate_takeover('s1', chain, {'Id': 'x'}, state)['observed_step'], 45000)
        for changes in ({'sensor_container': 'panda-sensor5-stage12-qb-s1-v1'},
                        {'stage': 'RESUME_TUNE30'}, {'sensor_campaign_id': 'v1'}):
            with self.assertRaises(ValueError):
                queue.validate_takeover('s1', dict(chain, **changes), {'Id': 'x'}, state)
        state['runs'][0]['poisoned'] = True
        with self.assertRaises(ValueError):
            queue.validate_takeover('s1', chain, {'Id': 'x'}, state)

    def test_loss_grid_precedes_existing_lr_and_wv3(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            chain = {'stage': 'WAIT_INITIAL_PAIR'}
            with patch.object(queue, 'wait_receipt', return_value=True), \
                    patch.object(queue, 'sensor_state', return_value={'status': orchestration.PAUSED}), \
                    patch.object(queue, 'boundary_complete', return_value=True), \
                    patch.object(queue, 'predecessor_receipt', return_value={'proofs': []}):
                queue.advance('s1', directory, chain)
            self.assertEqual(chain['stage'], 'START_LOSS9')
            with patch.object(queue.launch, 'launch') as start:
                queue.advance('s1', directory, chain)
                start.assert_called_once_with('s1', 'loss9')
            self.assertEqual(chain['stage'], 'WAIT_LOSS9')
            with patch.object(queue, 'wait_receipt', return_value=True), \
                    patch.object(queue, 'loss_campaign_complete', return_value=True):
                queue.advance('s1', directory, chain)
            self.assertEqual(chain['stage'], 'RESUME_SENSOR5')
            with patch.object(queue, 'sensor_state', return_value={'status': 'STOPPED_SAFE'}), \
                    patch.object(queue.launch, 'launch') as start:
                queue.advance('s1', directory, chain)
                start.assert_called_once_with('s1', 'resume-sensor5')
            self.assertEqual(chain['stage'], 'WAIT_REMAINING_SENSOR5')
            with patch.object(queue, 'wait_receipt', return_value=True), \
                    patch('panda_sensor5.queue.campaign_complete', return_value=True):
                queue.advance('s1', directory, chain)
            self.assertEqual(chain['stage'], 'RESUME_TUNE30')

    def test_changed_container_is_never_signalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = {'stage': 'TAKEOVER'}
            with patch.object(queue, 'begin_takeover', return_value={
                    'source_container': 'expected', 'container_id': 'original'}), \
                    patch.object(queue, 'inspect_optional', return_value={'Id': 'replacement'}), \
                    patch.object(queue.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    queue.advance('s1', Path(tmp), chain)
                run.assert_not_called()

    def test_initial_pair_adapter_preserves_training_config_and_omits_lr_cases(self):
        from panda_sensor5 import plan, runtime
        configs = {name: {'case': name, 'run_id': name, 'planned_horizon': h}
                   for name, h in [('D', 50000), ('BASE', 30000), ('LR', 30000)]}
        groups = [{'name': n, 'runs': [(n, 'R1')]} for n in configs]
        with tempfile.TemporaryDirectory() as tmp:
            class FakeCampaign:
                def __init__(self, server, **kwargs):
                    self.server, self.root = server, Path(tmp)
                    self.state = {'runs': kwargs.get('initial_runs', []), 'status': 'STOPPED_SAFE'}
                    self.config = {'groups': groups, 'immutable_marker': 'original'}
                    self.events, self.stop_requested = [], False
                def check(self, **kwargs): pass
                def persist(self): pass
                def admission(self, catalog): self.events.append(('admission', self.config.copy()))
                def admit_group(self, group): pass
                def completed(self, cfg):
                    return next((r for r in self.state['runs'] if r['run_id'] == cfg['run_id']
                                 and r['status'] == 'COMPLETED'), None)
                def validate_preflight(self, cfg): self.events.append(('preflight', cfg['case']))
                def run_case(self, cfg):
                    self.events.append(('run', cfg['case']))
                    row = next((r for r in self.state['runs'] if r['run_id'] == cfg['run_id']), None)
                    if row is None:
                        row = {'run_id': cfg['run_id']}
                        self.state['runs'].append(row)
                    row.update(status='COMPLETED', step=cfg['planned_horizon'])
                def register_donor(self, case): self.events.append(('donor', case))
            with patch.object(runtime, 'Campaign', FakeCampaign), \
                    patch.object(plan, 'DONORS', ('D',)), \
                    patch.object(plan, 'resolve_case', side_effect=lambda s, c, r='R1': configs[c]), \
                    patch.object(orchestration, 'initial_configs', return_value=[configs['D'], configs['BASE']]):
                campaign = orchestration.controlled_class()('s1', mode='through-base')
                state = campaign.run()
                initial = [{'run_id': c['run_id'], 'status': 'COMPLETED', 'step': c['planned_horizon']}
                           for c in (configs['D'], configs['BASE'])]
                initial.append({'run_id': 'LR', 'status': 'STOPPED', 'step': 5000})
                later = orchestration.controlled_class()('s1', mode='through-base',
                    finish_run='LR', initial_runs=initial)
                later_state = later.run()
            self.assertEqual(state['status'], orchestration.PAUSED)
            self.assertEqual([e for e in campaign.events if e[0] == 'run'], [('run', 'D'), ('run', 'BASE')])
            self.assertEqual(campaign.config, {'groups': groups, 'immutable_marker': 'original'})
            self.assertFalse(state['s1loss9_control']['sessions'][0]['optimizer_state_reset'])
            self.assertEqual(later_state['status'], orchestration.PAUSED)
            self.assertEqual([e for e in later.events if e[0] == 'run'], [('run', 'LR')])
            self.assertEqual(len(later_state['runs']), 3)


if __name__ == '__main__':
    unittest.main()
