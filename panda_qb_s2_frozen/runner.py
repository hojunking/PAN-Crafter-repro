"""Run 22 fresh Students, with exact resume and the existing QB early stop."""
import fcntl
import signal
import traceback
from directshift48.reporting import utc_now
from panda_sensor_focus import early_stop
from . import plan

TERMINAL=('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_EXPLICIT_EARLY_STOPS')

def run(camp,*,catalog):
    from .runtime import BudgetExpired
    with (camp.root/'.controller.lock').open('a+') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handlers={sig:signal.signal(sig,lambda *a:setattr(camp,'stop_requested',True))
                  for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            camp.persist();camp.admission(catalog)
            for group in camp.config['groups']:
                cfg=plan.resolve_case(camp.server,*group['runs'][0])
                if cfg['phase']!='stage2': raise ValueError('Stage1 execution forbidden')
                if camp.completed(cfg) or early_stop.finish_stopped(camp,cfg): continue
                camp.state['groups'].append(dict(group=group['name'],at=utc_now(),admitted=True,
                    budget_policy='REGISTERED_QUEUE_NO_DEADLINE'))
                camp.persist();camp.check();camp.validate_preflight(cfg)
                early_stop.run_case(camp,cfg,BudgetExpired)
            expected=[plan.resolve_case(camp.server,*g['runs'][0]) for g in camp.config['groups']]
            stopped=camp.state.get('early_stopped_runs',{})
            if not all(camp.completed(c) or c['run_id'] in stopped for c in expected):
                raise RuntimeError('Incomplete frozen Stage2 accounting')
            camp.state.update(status=TERMINAL[1] if stopped else TERMINAL[0],
                              all_registered_completed=not stopped,pending_groups=[])
        except BudgetExpired as exc:
            camp.state.update(status='STOPPED_SAFE',reason=str(exc))
        except Exception as exc:
            camp.state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            for sig,handler in handlers.items(): signal.signal(sig,handler)
        return camp.state
