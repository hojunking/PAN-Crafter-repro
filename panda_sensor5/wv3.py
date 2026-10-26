"""Control-only adapter for exact WV3 run boundaries and deferred queue completion.

The registered WV3 compute modules/configs/checkpoint pins are unchanged.  This
adapter has a separate immutable pin and changes only queue order and the user's
explicit extension of the old wall-clock budget. Never reset a training state.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

from directshift48.reporting import atomic_json, utc_now

POLICY = 'USER_20261026_SENSOR5_THEN_ALL_REMAINING_WV3_NO_DEADLINE'


def adapter_pin():
    path = Path(__file__).resolve()
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def active_clock(state, now):
    """Keep the original start and durable elapsed; exclude the inactive gap."""
    control = state.get('sensor5_control', {})
    last = control.get('last_closed_unix')
    if last is None:
        last = state['started_unix'] + state.get('elapsed_seconds', 0.)
    if last > now + 1:
        raise ValueError('WV3 pause accounting is in the future')
    return {'active_seconds_at_open':float(state.get('elapsed_seconds', 0.)),
            'excluded_gap_seconds':max(0.,now-last), 'opened_unix':now}


def controlled_class(package):
    if package not in ('panda_tune30', 'panda_hard36'):
        raise ValueError('Unsupported registered WV3 campaign')
    runtime = importlib.import_module(package + '.runtime')
    plan = importlib.import_module(package + '.plan')

    class ControlledCampaign(runtime.Campaign):
        def __init__(self, server, *, mode, target_run=None, **kwargs):
            self.mode, self.target_run = mode, target_run
            if mode not in ('finish-current', 'resume-all'):
                raise ValueError('Unknown WV3 control mode')
            if (mode == 'finish-current') != bool(target_run):
                raise ValueError('finish-current requires exactly one target run')
            super().__init__(server, **kwargs)
            if mode == 'finish-current' and not any(
                    r['run_id'] == target_run for r in self.state['runs']):
                raise ValueError('Current-run handoff must resume an existing run')
            now = self.clock()
            clock = active_clock(self.state, now)
            self._active_at_open = clock['active_seconds_at_open']
            self._active_opened = time.monotonic()
            self._adapter_pin = adapter_pin()
            control = self.state.setdefault('sensor5_control', {})
            previous_pin = control.get('adapter_pin')
            if previous_pin is not None and previous_pin != self._adapter_pin:
                raise ValueError('WV3 orchestration adapter changed after registration')
            control.update(policy=POLICY, adapter_pin=self._adapter_pin,
                original_started_unix=self.state['started_unix'],
                original_budget_seconds=self.config['budget']['total_seconds'],
                excluded_wall_seconds=control.get('excluded_wall_seconds',0.)+
                    clock['excluded_gap_seconds'])
            control.setdefault('sessions',[]).append(dict(clock, mode=mode,
                target_run=target_run, original_training_config_unchanged=True))

        def elapsed_seconds(self):
            if not hasattr(self,'_active_at_open'):
                return super().elapsed_seconds()
            return self._active_at_open + time.monotonic()-self._active_opened

        def check(self, **_):
            if self.stop_requested:
                raise runtime.BudgetExpired('STOP_REQUESTED')
            if shutil.disk_usage(self.root).free < 2*1024**3:
                raise runtime.BudgetExpired('DISK_CLOSEOUT_2GiB_RESERVE')
            if adapter_pin() != self._adapter_pin:
                raise RuntimeError('WV3 orchestration adapter changed while running')
            # The user explicitly requested every deferred run after SENSOR5.
            # Old 30h/36h caps are retained in provenance, not used to skip work.

        def run(self, *, catalog=plan.ROOT/'ablr2/sensor_sources.json'):
            with (self.root/'.controller.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                old={s:signal.signal(s,lambda *a:setattr(self,'stop_requested',True))
                     for s in (signal.SIGINT,signal.SIGTERM)}
                try:
                    if (self.root/'pending_control.json').exists():
                        raise ValueError('Existing manual WV3 skip control requires inspection')
                    self.persist()
                    self.admission(catalog)
                    self.register_donor(plan.DONORS[0])
                    expected=[plan.resolve_case(self.server,c,r)
                        for group in self.config['groups'] for c,r in group['runs']]
                    if self.mode == 'finish-current':
                        targets=[c for c in expected if c['run_id']==self.target_run]
                        if len(targets)!=1:
                            raise ValueError('Cutover target not registered exactly once')
                        self.check()
                        self.run_case(targets[0])
                        if not self.completed(targets[0]):
                            raise RuntimeError('Current run did not finish its full horizon')
                        self.state['status']='PAUSED_FOR_SENSOR5'
                    else:
                        for group in self.config['groups']:
                            pending=[plan.resolve_case(self.server,c,r) for c,r in group['runs']]
                            pending=[c for c in pending if not self.completed(c)]
                            if pending:
                                self.state['groups'].append({'group':group['name'],
                                    'at':utc_now(),'admitted':True,'optional':group['optional'],
                                    'reason':'USER_AUTHORIZED_ALL_REMAINING_WV3',
                                    'budget_policy':POLICY,
                                    'pending':[c['run_id'] for c in pending]})
                                self.persist()
                            for cfg in pending:
                                self.check()
                                self.run_case(cfg)
                        if not all(self.completed(c) for c in expected):
                            raise RuntimeError('WV3 registered queue is not complete')
                        self.state['status']='FINISHED_REGISTERED_QUEUE'
                    self.state.pop('reason',None)
                except runtime.BudgetExpired as exc:
                    self.state.update(status='STOPPED_SAFE',reason=str(exc))
                except BaseException as exc:
                    self.state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
                    raise
                finally:
                    control=self.state['sensor5_control']
                    control['last_closed_unix']=self.clock()
                    control['sessions'][-1].update(closed_unix=control['last_closed_unix'],
                        resulting_status=self.state.get('status'))
                    self.persist()
                    for sig,handler in old.items():
                        signal.signal(sig,handler)
                return self.state
    return ControlledCampaign


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--package',choices=('panda_tune30','panda_hard36'),required=True)
    parser.add_argument('--mode',choices=('finish-current','resume-all'),required=True)
    parser.add_argument('--target-run')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    plan=importlib.import_module(args.package+'.plan')
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Local server marker mismatch')
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    campaign=controlled_class(args.package)(args.server,mode=args.mode,
        target_run=args.target_run,device=args.device,output='work_dir/'+args.package)
    state=campaign.run()
    print(json.dumps({'status':state['status'],'root':str(campaign.root)}))
    wanted='PAUSED_FOR_SENSOR5' if args.mode=='finish-current' else 'FINISHED_REGISTERED_QUEUE'
    return int(state['status']!=wanted)


if __name__=='__main__':
    raise SystemExit(main())
