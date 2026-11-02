"""Run the finite donor-priority queue; isolate authenticated failed branches."""
import fcntl
import signal
import traceback
from copy import deepcopy
from directshift48.reporting import utc_now
from . import registry as G

TERMINAL=('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_EXPLICIT_EARLY_STOPS',
          'BUDGET_COMPLETE_WITH_PENDING','FINISHED_WITH_FAILED_BRANCHES')

def estimate_seconds(campaign,cfg):
    row=G.job(campaign.server,cfg['case'],cfg['replica'])
    estimate=row['estimate_hours']*3600
    # Full official eval/save costs, measured on the same graph AND batch.
    timing=campaign.state['timing'].get(campaign.timing_key(cfg),{})
    if campaign.measured_timing(timing):
        estimate=max(estimate,(timing['update_p90']*cfg['planned_horizon']+
                     (timing['eval_seconds']+timing.get('save_seconds',0))*len(cfg['evaluation_steps']))*1.08)
    # A resumed run only needs its remaining portion; calibration and closeout
    # are conservatively reserved independently below.
    previous=next((r for r in campaign.state['runs'] if r['run_id']==cfg['run_id']),None)
    if previous:
        estimate *= max(0.,1-previous['step']/cfg['planned_horizon'])
    return estimate+(600 if cfg['phase']=='donor' else 0)

def run(camp,*,catalog):
    from . import plan
    from .runtime import BudgetExpired
    from . import failures
    from panda_sensor_focus import early_stop
    with (camp.root/'.controller.lock').open('a+') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handlers={sig:signal.signal(sig,lambda *a:setattr(camp,'stop_requested',True)) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            camp.persist();camp.admission(catalog)
            failures.synchronize(camp)
            camp.persist()
            pending=[]
            for index,group in enumerate(camp.config['groups']):
                case,replica=group['runs'][0];cfg=plan.resolve_case(camp.server,case,replica)
                if failures.held(camp,cfg): continue
                if camp.completed(cfg):
                    if case in plan.DONORS: camp.register_donor(case)
                    continue
                if early_stop.finish_stopped(camp,cfg): continue
                needed=estimate_seconds(camp,cfg)
                left=None
                admitted=True
                camp.state['groups'].append(dict(group=group['name'],at=utc_now(),admitted=admitted,
                    predicted_seconds=needed,remaining_seconds=left,budget_policy='REGISTERED_QUEUE_NO_DEADLINE'))
                camp.persist()
                if not admitted:
                    pending=[deepcopy(g) for g in camp.config['groups'][index:]
                             if not failures.held(camp,plan.resolve_case(camp.server,*g['runs'][0]))]
                    break
                camp.check()
                camp.validate_preflight(cfg)
                try:
                    early_stop.run_case(camp,cfg,BudgetExpired)
                except ValueError as exc:
                    if cfg['phase']!='donor' or repr(exc)!=failures.SUPPORT_ERROR: raise
                    # Only a saved, nonpoisoned pre-optimizer support failure is
                    # isolated. Unknown errors and failed saves still abort.
                    failures.synchronize(camp)
                    if cfg['run_id'] not in camp.state['failed_runs']: raise
                    camp.persist()
                    print(f"[QB_S2 HOLD] {case}/{replica}: Stage1 failed; dependent Students blocked",flush=True)
                    continue
                if case in plan.DONORS: camp.register_donor(case)
            camp.state['pending_groups']=pending
            if pending:
                camp.state.update(status='BUDGET_COMPLETE_WITH_PENDING',all_registered_completed=False)
            else:
                expected=[plan.resolve_case(camp.server,*g['runs'][0]) for g in camp.config['groups']]
                stopped=camp.state.get('early_stopped_runs',{})
                failed=camp.state.get('failed_runs',{})
                if not all(camp.completed(c) or c['run_id'] in stopped or failures.held(camp,c) for c in expected):
                    raise RuntimeError('Incomplete QB_S2 run accounting')
                camp.state.update(status=failures.STATUS if failed else
                                  'FINISHED_WITH_EXPLICIT_EARLY_STOPS' if stopped else 'FINISHED_REGISTERED_QUEUE',
                                  all_registered_completed=not stopped and not failed)
        except BudgetExpired as exc:
            camp.state.update(status='STOPPED_SAFE',reason=str(exc))
        except Exception as exc:
            camp.state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            for sig,handler in handlers.items():signal.signal(sig,handler)
        return camp.state
