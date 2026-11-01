"""Finish one existing QB40 run with its original code/config/checkpoint pins."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import traceback

from directshift48.reporting import atomic_json, utc_now

PAUSED = 'PAUSED_FOR_RRE_PRIORITY'


def adapter_pin():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def target_config(server, run_id, state):
    from panda_qb40 import plan
    configs = [plan.resolve_case(server, *g['runs'][0]) for g in plan.groups(server)]
    matches = [cfg for cfg in configs if cfg['run_id'] == run_id]
    observed = [row for row in state.get('runs', []) if row['run_id'] == run_id]
    if len(matches) != 1 or len(observed) != 1:
        raise ValueError('Finish-current requires exactly one already-started registered run')
    if any(row['run_id'] != run_id and row['status'] in ('RUNNING', 'STOPPED')
           for row in state['runs']):
        raise ValueError('Another QB40 run is active')
    return matches[0]


def finish(camp, run_id, catalog):
    from panda_qb40 import failures, plan
    from panda_qb40.runtime import BudgetExpired
    from panda_sensor_focus import early_stop
    from panda_sensor_refine.orchestration import final_evidence
    cfg = target_config(camp.server, run_id, camp.state)
    pin = adapter_pin()
    path = camp.root / 'rre_priority_boundary.json'
    with (camp.root / '.controller.lock').open('a+') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handlers = {s: signal.signal(s, lambda *a: setattr(camp, 'stop_requested', True))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            camp.admission(catalog)
            failures.synchronize(camp)
            if not camp.completed(cfg) and not failures.held(camp, cfg):
                if not early_stop.finish_stopped(camp, cfg):
                    if cfg['donor_version'] == 'S1_A200_E030' and 'S1_A200_E030' not in camp.state['donors']:
                        camp.register_external_donor()
                    camp.validate_preflight(cfg)
                    try:
                        early_stop.run_case(camp, cfg, BudgetExpired)
                    except ValueError as exc:
                        if cfg['phase'] != 'donor' or repr(exc) != failures.SUPPORT_ERROR:
                            raise
                        failures.synchronize(camp)
                        if cfg['run_id'] not in camp.state.get('failed_runs', {}):
                            raise
            if adapter_pin() != pin:
                raise ValueError('Finish-current adapter changed during execution')
            row = next(r for r in camp.state['runs'] if r['run_id'] == run_id)
            if camp.completed(cfg):
                if cfg['phase'] == 'donor':
                    camp.register_donor(cfg['case'])
                proof = final_evidence(camp.root, cfg)
            elif row['status'] == early_stop.STATUS:
                proof = early_stop.verify(row, cfg, plan.ROOT)
            elif run_id in camp.state.get('failed_runs', {}):
                proof = failures.verify(camp, row)
            else:
                raise ValueError('Current case has no complete/early-stop/known-failure proof')
            started = {(r['case'], r['replica']) for r in camp.state['runs']}
            unstarted = [list(g['runs'][0]) for g in camp.config['groups']
                         if tuple(g['runs'][0]) not in started]
            receipt = dict(schema='QB_RRE_FINISH_CURRENT_V1', at=utc_now(), server=camp.server,
                target_run=run_id, result_status=row['status'], step=row['step'], proof=proof,
                adapter_sha256=pin, training_config_unchanged=True, checkpoint_pins_unchanged=True,
                original_started_unix=camp.state['started_unix'], unstarted_superseded=unstarted)
            atomic_json(path, receipt)
            camp.state.update(status=PAUSED, all_registered_completed=False,
                              rre_priority_boundary=str(path), superseded_unstarted=unstarted)
            camp.state.pop('reason', None)
        except BudgetExpired as exc:
            camp.state.update(status='STOPPED_SAFE', reason=str(exc))
        except BaseException as exc:
            camp.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return camp.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    parser.add_argument('--target-run', required=True)
    args = parser.parse_args(argv)
    if args.server == 's3':
        from panda_qb_deepen.adapter import install
        install()
    from panda_qb40 import plan, runtime
    from directshift48.__main__ import require_idle_gpu
    if (plan.ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker differs')
    require_idle_gpu('cuda')
    camp = runtime.Campaign(args.server, output=str(plan.ROOT / 'work_dir/panda_qb40'))
    result = finish(camp, args.target_run, plan.ROOT / 'ablr2/sensor_sources.json')
    print(json.dumps(dict(status=result['status'], target_run=args.target_run)), flush=True)
    return int(result['status'] != PAUSED)


if __name__ == '__main__':
    raise SystemExit(main())
