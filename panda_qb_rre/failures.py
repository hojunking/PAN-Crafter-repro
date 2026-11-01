"""Hold authenticated Stage1 support failures and only their dependent Students."""
from . import plan
from panda_sensor_refine.orchestration import SUPPORT_ERROR, support_failure

STATUS = 'FINISHED_WITH_FAILED_BRANCHES'


def verify(camp, row):
    cfg = plan.resolve_case(camp.server, row['case'], row['replica'])
    if row['run_id'] != cfg['run_id'] or not 0 < row['step'] < cfg['planned_horizon']:
        raise ValueError('Failed branch identity or step differs')
    proof = support_failure(row, cfg, plan.ROOT)
    from panda_sensor_refine.orchestration import local, read
    directory = local(row['directory'], plan.ROOT).resolve()
    expected = camp.root / 'runs' / cfg['run_id'] / 'attempt_01'
    if (directory != expected.resolve() or
            plan.canonical_sha256(read(directory/'requested_config.json')) != plan.canonical_sha256(cfg)):
        raise ValueError('Failed branch request/directory differs')
    return proof


def synchronize(camp):
    """Reauthenticate held evidence on restart; never rewrite the failed run."""
    failed = {}
    for row in camp.state['runs']:
        if row['status'] == 'FAILED':
            failed[row['run_id']] = verify(camp, row)
    old = camp.state.get('failed_runs', {})
    if any(failed.get(key) != value for key, value in old.items()):
        raise ValueError('Held Stage1 failure evidence changed')
    failed_cases = {p['failed_record']['case']: key for key, p in failed.items()}
    blocked = {}
    for group in camp.config['groups']:
        cfg = plan.resolve_case(camp.server, *group['runs'][0])
        source = failed_cases.get(cfg['donor_version'])
        if cfg['phase'] == 'stage2' and source:
            if any(r['run_id'] == cfg['run_id'] for r in camp.state['runs']):
                raise ValueError('A Student was started with a failed Stage1')
            blocked[cfg['run_id']] = dict(case=cfg['case'], replica=cfg['replica'],
                donor_version=cfg['donor_version'], failed_run_id=source,
                status='BLOCKED_FAILED_STAGE1', reason=SUPPORT_ERROR)
    if any(blocked.get(key) != value for key, value in camp.state.get('blocked_runs', {}).items()):
        raise ValueError('Held dependent Student evidence changed')
    camp.state.update(failed_runs=failed, blocked_runs=blocked)


def held(camp, cfg):
    return (cfg['run_id'] in camp.state.get('failed_runs', {}) or
            cfg['run_id'] in camp.state.get('blocked_runs', {}))
