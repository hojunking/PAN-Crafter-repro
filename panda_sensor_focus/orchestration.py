"""Authenticated accounting of retired sensor queues; no training-code mutation."""
import importlib
from pathlib import Path

from directshift48.reporting import atomic_json, utc_now
from panda_sensor_refine.orchestration import read, sha, local, authenticate_stop, final_evidence

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'USER_20261029_SENSOR_FIXED_DONOR_FOCUS_KEEP_WV3'
SENSOR_PACKAGES = ('panda_sensor5', 'panda_s1loss9', 'panda_sensor_refine')
WV3_PACKAGES = ('panda_tune30', 'panda_hard36')


def launcher_root(root, server):
    return Path(root) / 'work_dir/panda_sensor_focus_launchers' / server


def registered(package, server):
    plan = importlib.import_module(package + '.plan')
    return [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]


def campaign_root(root, package, server):
    plan = importlib.import_module(package + '.plan')
    dataset = plan.dataset_for_server(server) if package in SENSOR_PACKAGES else 'WV3'
    return Path(root) / 'work_dir' / package / plan.CAMPAIGN_ID / server / dataset


def state_of(root, package, server):
    path = campaign_root(root, package, server) / 'campaign_state.json'
    return read(path) if path.exists() else None


def boundary_configs(package, server, state):
    """Keep an actually started run and, for an active donor, its Student(s)."""
    cfgs = registered(package, server)
    by_id = {c['run_id']: c for c in cfgs}
    rows = state.get('runs', []) if state else []
    if len({r['run_id'] for r in rows}) != len(rows) or any(r['run_id'] not in by_id for r in rows):
        raise ValueError('Unknown/duplicate previous run identity')
    active = [r for r in rows if r['status'] in ('RUNNING', 'STOPPED')]
    if len(active) > 1 or any(r.get('poisoned') or r.get('stop_checkpoint_error') for r in active):
        raise ValueError('Unsafe or ambiguous previous active run')
    if not active:
        return []
    cfg = by_id[active[0]['run_id']]
    selected = [cfg]
    if cfg['phase'] == 'donor' and package in SENSOR_PACKAGES:
        existing = {r['run_id'] for r in rows}
        children = [c for c in cfgs if c['phase'] == 'stage2' and c.get('donor_version') == cfg['case']]
        if children and children[0]['run_id'] not in existing:
            selected.append(children[0])
    return selected


def retirement_accounting(root, package, server, state):
    """No deletion and no fabricated completion for unstarted registrations."""
    cfgs = registered(package, server)
    rows = state.get('runs', []) if state else []
    by_id = {r['run_id']: r for r in rows}
    known = {c['run_id'] for c in cfgs}
    if len(by_id) != len(rows) or not set(by_id) <= known:
        raise ValueError('Previous run registry differs')
    result = []
    for cfg in cfgs:
        row = by_id.get(cfg['run_id'])
        item = {k: cfg[k] for k in ('run_id', 'case', 'replica', 'phase', 'planned_horizon')}
        if row is None:
            boundary = (state or {}).get('sensor_focus_boundary', {}).get('outcomes', {}).get(cfg['run_id'], {})
            held = (state or {}).get('held_runs', {}).get(cfg['run_id']) or boundary.get('status') == 'HELD_FAILED_DONOR'
            failed = ((state or {}).get('isolated_support_failures', {}).get(cfg['run_id'])
                      or (state or {}).get('sensor_refine_control', {}).get('preflight_failures', {}).get(cfg['run_id'])
                      or boundary.get('status') == 'FAILED_PREFLIGHT_SUPPORT')
            item.update(status='PRESERVED_HELD' if held else 'PRESERVED_PREFLIGHT_FAILURE' if failed else 'EXCLUDED_UNSTARTED',
                        reason='RETAIN_ORIGINAL_FAILURE_OR_HOLD' if held or failed else 'USER_REPLACED_PENDING_WITH_FIXED_DONOR_FOCUS', step=0)
        elif row['status'] == 'COMPLETED':
            if row['step'] != cfg['planned_horizon']:
                raise ValueError('Completed previous run lacks its full horizon')
            item.update(status='PRESERVED_COMPLETED', step=row['step'],
                        evidence=final_evidence(campaign_root(root, package, server), cfg))
        elif row['status'] == 'EARLY_STOPPED_HQNR':
            from panda_sensor_refine.early_stop import verify
            item.update(status='PRESERVED_EARLY_STOP', step=row['step'], evidence=verify(row, cfg, root=root))
        elif row['status'] == 'FAILED':
            from .boundary import row_support_message
            if row_support_message(row) is None or row.get('poisoned') is not False:
                raise ValueError('Unclassified previous failure; inspect before retirement')
            item.update(status='PRESERVED_SUPPORT_FAILURE', step=row['step'], evidence=authenticate_stop(row, cfg, root=root))
        else:
            raise ValueError('Started previous run remains incomplete: ' + cfg['run_id'])
        result.append(item)
    return result


def write_predecessor(root, server):
    from .queue import controller_pin
    from donor30.plan import canonical_sha256
    directory = launcher_root(root, server)
    result = {'schema': 'SENSOR_FOCUS_PREDECESSOR_V1', 'policy': POLICY, 'server': server,
              'controller_pin_sha256': canonical_sha256(controller_pin()),
              'sensor_queues': {}, 'wv3_unchanged': True}
    for package in SENSOR_PACKAGES:
        state = state_of(root, package, server)
        path = campaign_root(root, package, server) / 'campaign_state.json'
        result['sensor_queues'][package] = {
            'state_path': str(path.relative_to(root)), 'state_sha256': sha(path) if path.exists() else None,
            'accounting': retirement_accounting(root, package, server, state)}
    path = directory / 'predecessor_complete.json'
    if path.exists():
        if read(path) != result:
            raise ValueError('Immutable predecessor accounting changed')
    else:
        atomic_json(path, result)
    return result


def predecessor_receipt(root, server):
    from .queue import controller_pin
    from donor30.plan import canonical_sha256
    path = launcher_root(root, server) / 'predecessor_complete.json'
    value = read(path)
    if (value.get('schema') != 'SENSOR_FOCUS_PREDECESSOR_V1' or value.get('policy') != POLICY
            or value.get('server') != server or not value.get('wv3_unchanged')
            or value.get('controller_pin_sha256') != canonical_sha256(controller_pin())
            or set(value.get('sensor_queues', {})) != set(SENSOR_PACKAGES)):
        raise ValueError('Unverified fixed-donor predecessor receipt')
    for record in value['sensor_queues'].values():
        state_path = Path(root) / record['state_path']
        if (sha(state_path) if state_path.exists() else None) != record['state_sha256']:
            raise ValueError('Retired sensor state changed after handoff')
    return value
