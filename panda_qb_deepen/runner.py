"""Original paired runtime ordering plus the already-authorized QB Student gate."""
import fcntl
import signal
import traceback


def run(self, *, catalog=None):
    from panda_s1loss9 import plan
    from panda_s1loss9.runtime import BudgetExpired
    from panda_sensor_focus import early_stop
    catalog = catalog or plan.ROOT / 'ablr2/sensor_sources.json'
    with (self.root / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handlers = {sig: signal.signal(sig, lambda *a: setattr(self, 'stop_requested', True))
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.persist(); self.admission(catalog)
            for group in self.config['groups']:
                self.admit_group(group)
                for case, replica in group['runs']:
                    self.check(); cfg = plan.resolve_case(self.server, case, replica)
                    if not self.completed(cfg) and not early_stop.finish_stopped(self, cfg):
                        self.validate_preflight(cfg)
                        early_stop.run_case(self, cfg, BudgetExpired)
                    if case in plan.DONORS: self.register_donor(case)
            expected = [plan.resolve_case(self.server, c, r) for g in self.config['groups'] for c, r in g['runs']]
            stopped = self.state.get('early_stopped_runs', {})
            if not all(self.completed(cfg) or cfg['run_id'] in stopped for cfg in expected):
                raise RuntimeError('Incomplete independent-Stage1 QB queue')
            self.state['status'] = 'FINISHED_WITH_EXPLICIT_EARLY_STOPS' if stopped else 'FINISHED_REGISTERED_QUEUE'
        except BudgetExpired as exc:
            self.state.update(status='STOPPED_SAFE', reason=str(exc))
        except Exception as exc:
            self.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            self.persist()
            for sig, handler in handlers.items(): signal.signal(sig, handler)
        return self.state
