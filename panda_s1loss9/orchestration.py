"""Control-only SENSOR5 boundary adapter; original model/checkpoint pins stay intact."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import traceback

from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'USER_20261026_STAGE1_LOSS_GRID_BEFORE_SENSOR_LR_AND_WV3_NO_DEADLINE'
PAUSED = 'PAUSED_FOR_S1LOSS9'


def adapter_pin():
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}


def initial_configs(server):
    from panda_sensor5 import plan
    sensor = plan.dataset_for_server(server)
    return [plan.resolve_case(server, plan.DONORS[0]),
            plan.resolve_case(server, sensor + '_BASE')]


def initial_complete(state, server):
    """A boundary requires both the full fresh donor and full BASE Student."""
    if state is None:
        return False
    for cfg in initial_configs(server):
        rows = [r for r in state.get('runs', []) if r['run_id'] == cfg['run_id']]
        if (len(rows) != 1 or rows[0].get('status') != 'COMPLETED'
                or rows[0].get('step') != cfg['planned_horizon']):
            return False
    return True


def predecessor_receipt(root, server):
    """Hash the same completed initial SENSOR5 evidence on every admission."""
    from panda_sensor5.transition import state_of, completed_run
    from panda_sensor5 import plan
    state = state_of(root, 'panda_sensor5', server)
    if not initial_complete(state, server):
        raise ValueError('Complete RGB/L4 SENSOR5 Stage1 and BASE required before loss tuning')
    return {'schema': 'S1LOSS9_COMPLETED_SENSOR5_INITIAL_PAIR_V1',
            'server': server, 'campaign_id': plan.CAMPAIGN_ID, 'policy': POLICY,
            'proofs': [completed_run(root, 'panda_sensor5', server, cfg['run_id'])
                       for cfg in initial_configs(server)],
            'successor_skips': [],
            'remaining_sensor5_and_wv3_deferred_not_cancelled': True}


def controlled_class():
    from panda_sensor5 import runtime, plan

    class ControlledCampaign(runtime.Campaign):
        def __init__(self, server, *, mode, finish_run=None, **kwargs):
            if mode not in ('through-base', 'resume-all'):
                raise ValueError('Unknown SENSOR5 boundary adapter mode')
            self.mode = mode
            self.finish_run = finish_run
            if finish_run and mode != 'through-base':
                raise ValueError('Only initial-pair boundary can include a current later run')
            super().__init__(server, **kwargs)
            registered = {plan.resolve_case(server, c, r)['run_id']
                          for g in self.config['groups'] for c, r in g['runs']}
            if finish_run and (finish_run not in registered or not any(
                    r['run_id'] == finish_run for r in self.state['runs'])):
                raise ValueError('Additional boundary run must be an already-started registered run')
            self._adapter_pin = adapter_pin()
            control = self.state.setdefault('s1loss9_control', {})
            if control.get('adapter_pin', self._adapter_pin) != self._adapter_pin:
                raise ValueError('SENSOR5 control adapter changed after registration')
            control.update(policy=POLICY, adapter_pin=self._adapter_pin)
            control.setdefault('sessions', []).append({'mode': mode, 'finish_run': finish_run, 'opened_at': utc_now(),
                'original_training_config_unchanged': True, 'optimizer_state_reset': False})

        def check(self, **kwargs):
            super().check(**kwargs)
            if adapter_pin() != self._adapter_pin:
                raise RuntimeError('SENSOR5 boundary adapter changed while running')

        def run(self, *, catalog=ROOT / 'ablr2/sensor_sources.json'):
            with (self.root / '.controller.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                old = {sig: signal.signal(sig, lambda *a: setattr(self, 'stop_requested', True))
                       for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    if (self.root / 'pending_control.json').exists():
                        raise ValueError('Unexpected manual SENSOR5 skip requires inspection')
                    self.persist()
                    # Original implementation authenticates the unchanged source,
                    # config, environment, data and original WV3 predecessor.
                    self.admission(catalog)
                    allowed = {c['run_id'] for c in initial_configs(self.server)}
                    if self.finish_run:
                        allowed.add(self.finish_run)
                    expected = []
                    for group in self.config['groups']:
                        configs = [plan.resolve_case(self.server, c, r) for c, r in group['runs']]
                        configs = [c for c in configs if self.mode == 'resume-all' or c['run_id'] in allowed]
                        if not configs:
                            continue
                        self.admit_group(group)
                        for cfg in configs:
                            self.check()
                            if not self.completed(cfg):
                                self.validate_preflight(cfg)
                                self.run_case(cfg)
                            if cfg['case'] in plan.DONORS:
                                self.register_donor(cfg['case'])
                            expected.append(cfg)
                    if not expected or not all(self.completed(c) for c in expected):
                        raise RuntimeError('Requested SENSOR5 boundary is incomplete')
                    self.state['status'] = PAUSED if self.mode == 'through-base' else 'FINISHED_REGISTERED_QUEUE'
                    self.state.pop('reason', None)
                except runtime.BudgetExpired as exc:
                    self.state.update(status='STOPPED_SAFE', reason=str(exc))
                except BaseException as exc:
                    self.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
                    raise
                finally:
                    self.state['s1loss9_control']['sessions'][-1].update(
                        closed_at=utc_now(), resulting_status=self.state.get('status'))
                    self.persist()
                    for sig, handler in old.items():
                        signal.signal(sig, handler)
            return self.state
    return ControlledCampaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--mode', choices=('through-base', 'resume-all'), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--finish-run')
    args = parser.parse_args(argv)
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    campaign = controlled_class()(args.server, mode=args.mode, finish_run=args.finish_run, device=args.device)
    state = campaign.run()
    print(json.dumps({'root': str(campaign.root), 'status': state['status']}), flush=True)
    return int(state['status'] != (PAUSED if args.mode == 'through-base' else 'FINISHED_REGISTERED_QUEUE'))


if __name__ == '__main__':
    raise SystemExit(main())
