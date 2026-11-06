"""Portable, read-only s2/s3 handoff inventory. Never creates a deletion manifest."""
import argparse
import json
import math
from pathlib import Path

from .ledger import ROOT, read, sha, write, now

TARGETS = {
    'WV3': {'HQNR_min': .9585, 'ERGAS_max': 2.040},
    'QB': {'HQNR_min': .9315, 'ERGAS_max': 3.5},
    'GF2': {'HQNR_min': .9645, 'ERGAS_max': .522, 'PSNR_min': 45.08},
}
ABLATIONS = {'panda_wv3_ablation', 'panda_wv3_ablation_rgb'}
MAIN = {
    'panda_qb_s2', 'panda_qb_s2_frozen', 'panda_qb_refit',
    'panda_qb_structure', 'panda_qb40', 'panda_qb_rre',
    'panda_sensor5', 'panda_sensor_focus', 'panda_sensor_refine',
    'panda_s1loss9', 'panda_tune30', 'panda_wv3_main_rgb', 'panda_wv3_w128',
}
TERMINAL = {'COMPLETED', 'EARLY_STOPPED_HQNR'}


def joint_pass(dataset, metrics):
    """Unrounded metrics from ONE checkpoint; unavailable metrics stop judgment."""
    target = TARGETS[dataset]
    for key in ['HQNR', 'ERGAS'] + (['PSNR'] if 'PSNR_min' in target else []):
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite metric: ' + key)
    if not 0 <= metrics['HQNR'] <= 1 or metrics['ERGAS'] < 0:
        raise ValueError('Metric outside physical range')
    return (metrics['HQNR'] >= target['HQNR_min']
            and metrics['ERGAS'] <= target['ERGAS_max']
            and ('PSNR_min' not in target or metrics['PSNR'] >= target['PSNR_min']))


def category(family):
    # A Sheet tab named "ablations" does not make a tuning campaign an ablation.
    return 'COMPONENT_ABLATION' if family in ABLATIONS else 'MAIN_TUNING' if family in MAIN else 'REVIEW_UNCLASSIFIED'


def rule(family, phase, status, horizon):
    if status not in TERMINAL:
        return {'action': 'KEEP_INCOMPLETE'}
    kind = category(family)
    if kind == 'COMPONENT_ABLATION':
        if horizon not in (30000, 50000):
            return {'action': 'REVIEW_UNSPECIFIED_HORIZON'}
        return {'action': 'REVIEW_EARLY_CHECKPOINTS',
                'cutoff_inclusive': 20000 if horizon == 30000 else 30000,
                'protect_validation_selected': phase in ('stage1', 'donor')}
    if phase in ('stage1', 'donor'):
        return {'action': 'KEEP_STAGE1_DONOR'}
    if kind == 'MAIN_TUNING' and phase == 'stage2':
        return {'action': 'REVIEW_NO_JOINT_PASS_RUNS'}
    return {'action': 'REVIEW_UNCLASSIFIED'}


def s2_pins(root):
    """Includes two Stage2 baseline files required by verify_baselines()."""
    plan = read(root / 'config/panda_qb_s2_frozen/plan.json')
    result = []
    for alias, pin in plan['donors'].items():
        entries = [(pin['checkpoint_path'], pin['checkpoint_sha256'], 'FROZEN_STAGE1_DONOR'),
                   (pin['baseline_R1']['artifact']['checkpoint_path'],
                    pin['baseline_R1']['artifact']['file_sha256'], 'REQUIRED_STAGE2_BASELINE')]
        for raw, expected, role in entries:
            path = root / Path(raw).relative_to('/workspace')
            regular = path.is_file() and not any(p.is_symlink() for p in [path, *path.parents])
            actual = sha(path) if regular else None
            result.append({'alias': alias, 'role': role, 'path': str(path.relative_to(root)),
                           'expected_sha256': expected, 'actual_sha256': actual,
                           'status': 'VERIFIED' if actual == expected else 'MISSING_OR_CHANGED'})
    return result


def inventory(server, root=ROOT):
    root = Path(root).resolve()
    campaigns, errors = [], []
    # Only the named server, only normal campaign layouts, no remote access.
    for state_path in sorted((root / 'work_dir').glob(f'*/*/{server}/*/campaign_state.json')):
        if any(p.is_symlink() for p in [state_path, *state_path.parents]):
            errors.append({'path': str(state_path), 'error': 'SYMLINK_NOT_SCANNED'})
            continue
        family = state_path.relative_to(root / 'work_dir').parts[0]
        try:
            state = read(state_path)
            rows = []
            for run in state.get('runs', []):
                run_id = run['run_id']
                if Path(run_id).name != run_id or run_id in ('.', '..'):
                    raise ValueError('Invalid run ID')
                directory = state_path.parent / 'runs' / run_id
                configs = sorted(directory.glob('attempt_*/config.json'))
                if (directory / 'config.json').is_file():
                    configs.append(directory / 'config.json')
                if not configs:
                    rows.append({'run_id': run_id, 'status': run.get('status'),
                                 'action': 'NO_LOCAL_CONFIG_OR_NOT_STARTED'})
                for cfg_path in configs:
                    if any(p.is_symlink() for p in [cfg_path, *cfg_path.parents]):
                        raise ValueError('Symlink config')
                    cfg = read(cfg_path)
                    if cfg['run_id'] != run_id or cfg['server'] != server:
                        raise ValueError('Run/server config identity mismatch')
                    phase = cfg.get('phase', cfg.get('stage'))
                    horizon = cfg.get('planned_horizon', cfg.get('horizon'))
                    status = run.get('status')
                    # Older attempts must not inherit a later attempt's completion.
                    run_state = cfg_path.parent / 'run_state.json'
                    if run_state.exists():
                        status = read(run_state).get('status', status)
                    elif len(configs) > 1:
                        status = 'ATTEMPT_STATUS_UNVERIFIED'
                    rows.append({'run_id': run_id, 'phase': phase, 'status': status,
                                 'horizon': horizon, 'directory': str(cfg_path.parent.relative_to(root)),
                                 **rule(family, phase, status, horizon)})
            campaigns.append({'family': family, 'kind': category(family),
                              'root': str(state_path.parent.relative_to(root)),
                              'status': state.get('status'), 'runs': rows})
        except (KeyError, ValueError, OSError, TypeError) as exc:
            errors.append({'path': str(state_path.relative_to(root)), 'error': str(exc)})
    pins = s2_pins(root) if server == 's2' else []
    return {'schema': 'PANDA_REMOTE_CLEANUP_HANDOFF_V1', 'at': now(), 'server': server,
            'mode': 'READ_ONLY_SCOPE_AND_DEPENDENCY_INVENTORY', 'deletion_ready': False,
            'delete_paths': [], 'targets': TARGETS, 'campaigns': campaigns,
            'pinned_dependencies': pins, 'errors': errors,
            'next': 'Follow results_log/2026-11-06_s2-s3-weight-cleanup-handoff.md; '
                    'audit local evaluations, dependency closure, file hashes and publication before unlink.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=['s2', 's3'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f'work_dir/_operator_storage/20261106_{args.server}_cleanup/inventory.json'
    output = output.absolute()
    allowed = ROOT / 'work_dir/_operator_storage'
    if allowed not in output.parents or '..' in output.parts or output.suffix != '.json':
        parser.error('Output must be a new JSON in work_dir/_operator_storage')
    if output.exists() or any(p.is_symlink() for p in [output, *output.parents]):
        parser.error('Refuse to overwrite an existing report or traverse a symlink')
    report = inventory(args.server)
    write(output, report)
    print(json.dumps({'report': str(output), 'campaigns': len(report['campaigns']),
                      'errors': report['errors'], 'pinned_dependencies': report['pinned_dependencies'],
                      'deletion_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
