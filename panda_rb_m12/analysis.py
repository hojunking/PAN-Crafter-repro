"""Preregistered Student-block statistics; no scene-level pseudoreplication."""
from __future__ import annotations

import math
import statistics

from reporting_bridge.rb_m12_contract import CASES, SELECTIONS, SERVERS, cohort_id, validate_case, sha

LOWER = {'ergas', 'sam', 'rmse', 'd_s', 'd_lambda', 'edge_error_dn', 'relative_response_l1_sum'}
PRIMARY = {
    'Q_ASSOCIATION': {'QFULL': 1, 'QSHUF': -1},
    'Q_VS_E': {'QFULL': 1, 'QESUR': -1},
    'HARD_ALLOCATION': {'QFULL': 1, 'HSPMEAN': -1},
    'ADVANTAGE_ALLOCATION': {'QFULL': 1, 'ADVMEAN': -1},
}
SECONDARY = {
    'Q_VS_MEAN': {'QFULL': 1, 'QMEAN': -1},
    'EDGE_MAIN_EFFECT': {'QEDGE': .5, 'QMEAN': -.5, 'QFULL': .5, 'QALIGN': -.5},
    'ALIGNER_MAIN_EFFECT': {'QALIGN': .5, 'QMEAN': -.5, 'QFULL': .5, 'QEDGE': -.5},
    'GEOMETRY_INTERACTION': {'QFULL': 1, 'QEDGE': -1, 'QALIGN': -1, 'QMEAN': 1},
    'HARD_PRESENCE': {'QFULL': 1, 'H0': -1},
    'HARD_MASS': {'HSPMEAN': 1, 'H0': -1},
    'ADVANTAGE_PRESENCE': {'QFULL': 1, 'NOADV': -1},
    'ADVANTAGE_MASS': {'ADVMEAN': 1, 'NOADV': -1},
}


def stats(values):
    values = list(values)
    if not all(type(x) in (int, float) and math.isfinite(x) for x in values):
        raise ValueError('Nonfinite summary value: preserve failure instead of silently dropping')
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None, ddof=1)


def paired_statistics(per_seed, metric, *, expected_n=12):
    """Input has exactly one finite paired difference per independent Student block."""
    import numpy as np
    from scipy.stats import t
    values = [r['delta'] for r in per_seed]
    result = dict(stats(values), expected_n=expected_n, per_seed=per_seed,
                  ci_method='two-sided paired Student t, df=n-1; independent blocks and approximately normal paired differences',
                  bootstrap_method='within-server paired Student-block percentile bootstrap; 10000 resamples; seed20261001',
                  ci95=None, bootstrap_ci95=None, p_two_sided=None,
                  improved=sum(v < 0 if metric in LOWER else v > 0 for v in values),
                  tied=sum(v == 0 for v in values),
                  worsened=sum(v > 0 if metric in LOWER else v < 0 for v in values))
    if len(values) < 2:
        return result
    mean, sd, n = result['mean'], result['sample_std'], len(values)
    margin = float(t.ppf(.975, n - 1))*sd/math.sqrt(n)
    result['ci95'] = [mean - margin, mean + margin]
    result['p_two_sided'] = (1. if mean == 0 else 0.) if sd == 0 else float(2*t.sf(abs(mean)/(sd/math.sqrt(n)), n - 1))
    rng = np.random.default_rng(20261001)
    total = np.zeros(10000, dtype=np.float64)
    for server in SERVERS:
        group = np.asarray([r['delta'] for r in per_seed if r['server'] == server], dtype=np.float64)
        if len(group):
            total += rng.choice(group, size=(10000, len(group)), replace=True).sum(axis=1)
    result['bootstrap_ci95'] = np.quantile(total/n, [.025, .975]).tolist()
    result['bootstrap_stratum_sizes'] = {s: sum(r['server'] == s for r in per_seed) for s in SERVERS}
    return result


def holm_family(tests):
    """Always eight preregistered endpoints; unknown tests occupy conservative p=1 slots."""
    if len(tests) != 8:
        raise ValueError('M12 primary family has exactly eight tests')
    ordered = sorted(enumerate(tests), key=lambda p: 1. if p[1]['p_two_sided'] is None else p[1]['p_two_sided'])
    running = 0.
    for rank, (index, item) in enumerate(ordered):
        p = item['p_two_sided']
        running = max(running, min(1., (8 - rank)*(1. if p is None else p)))
        tests[index]['p_holm_primary8'] = None if p is None else running
    return tests


def analyze(rows, servers=SERVERS):
    """Never pool old cohorts, aliases, or intentionally different case configs as new seeds."""
    if not servers or len(set(servers)) != len(servers) or not set(servers) <= set(SERVERS):
        raise ValueError('Only unique registered M12 server cohorts')
    index = {}; cohorts = set(); block_identity = {}
    for row in rows:
        validate_case(row)
        if row['server'] not in servers or row['selection_id'] not in SELECTIONS:
            raise ValueError('Observation outside requested server/selection cohort')
        key = (row['server'], row['repeat'], row['seed'], row['case_id'], row['selection_id'])
        if key in index:
            raise ValueError('Duplicate Student/selector observation')
        index[key] = row
        cohorts.add(cohort_id(row['provenance']))
        block = (row['server'], row['repeat'], row['seed'])
        pair = tuple(sha(row['provenance'].get(k)) for k in ('initialization_sha256', 'consumed_stream_sha256'))
        if block in block_identity and block_identity[block] != pair:
            raise ValueError('Paired cases did not share the same initial states and actual sample/view stream')
        block_identity[block] = pair
    if len(cohorts) > 1:
        raise ValueError('Cannot pool different source/F1/runtime/evaluator/data/raw-q releases')
    native = []; contrasts = []; paired = []; primary_tests = []
    expected = len(servers)*4
    for selector in SELECTIONS:
        selected = [r for r in rows if r['selection_id'] == selector]
        keys = sorted({k for r in selected for k in r['metrics']})
        for case in CASES:
            subset = [r for r in selected if r['case_id'] == case]
            for server in ('ALL', *servers):
                group = subset if server == 'ALL' else [r for r in subset if r['server'] == server]
                native.append(dict(selection_id=selector, case_id=case, server=server,
                                   expected_n=expected if server == 'ALL' else 4,
                                   metrics={k: stats(r['metrics'][k] for r in group if k in r['metrics']) for k in keys}))
        definitions = {**PRIMARY, **SECONDARY, **{f'{c}_MINUS_QFULL': {c: 1, 'QFULL': -1} for c in CASES if c != 'QFULL'}}
        blocks = {}
        for row in selected:
            blocks.setdefault((row['server'], row['repeat'], row['seed']), {})[row['case_id']] = row
        for name, weights in definitions.items():
            # Primary endpoint slots exist even before any run has completed.
            metrics = sorted(set(keys) | ({'hqnr', 'ergas'} if name in PRIMARY else set()))
            per_metric = {}
            for metric in metrics:
                differences = []
                for block, by_case in sorted(blocks.items()):
                    if not all(c in by_case and metric in by_case[c]['metrics'] for c in weights):
                        continue
                    differences.append(dict(server=block[0], repeat=block[1], seed=block[2],
                                            delta=sum(w*by_case[c]['metrics'][metric] for c, w in weights.items())))
                summary = paired_statistics(differences, metric, expected_n=expected)
                # Only EXACT and HQNR/ERGAS belong to the confirmatory family.
                if selector == SELECTIONS[0] and name in PRIMARY and metric in ('hqnr', 'ergas'):
                    summary['primary_family'] = True
                    primary_tests.append(dict(contrast=name, metric=metric, **summary))
                else:
                    summary.pop('p_two_sided')
                per_metric[metric] = summary
            item = dict(selection_id=selector, contrast=name, weights=weights,
                        classification='primary' if selector == SELECTIONS[0] and name in PRIMARY else 'secondary_effect_size',
                        metrics=per_metric)
            (paired if name.endswith('_MINUS_QFULL') else contrasts).append(item)
    holm_family(primary_tests)
    return dict(native_summary=native, paired_differences=paired, contrasts=contrasts,
                primary_endpoint_tests=primary_tests,
                statistical_unit='new M12 Student seed, fixed F1; neither scene nor direction nor selector alias',
                server_effect_identifiable=False, server_note='Server and seed are confounded; do not estimate a pure hardware effect',
                cohort_id=next(iter(cohorts), None), equivalence_claimed=False)


def radius_summary(curves, servers=SERVERS):
    """Only complete scene/direction coverage contributes to clean Student means."""
    keys = ('ergas', 'psnr', 'sam', 'edge_error_dn', 'relative_response_l1_sum', 'coverage_all_pan_paths')
    output = []
    for case in CASES[:6]:
        for mode in ('A_ON', 'A_ZERO_INFERENCE_ONLY', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE'):
            group = [c for c in curves if c['case_id'] == case and c['mode'] == mode]
            for radius in (0., .25, .5, 1., 2., 3., 4.):
                clean = {k: [] for k in keys}; flagged = {k: [] for k in keys}; valid = 0; failures = 0
                for curve in group:
                    points = [p for p in curve['points'] if p['radius_hr'] == radius]
                    ndirections = 1 if radius == 0 else 8
                    if len(points) != ndirections:
                        raise ValueError('Missing preregistered directions: cannot silently average a subset')
                    is_clean = all(p.get('n_valid') == 20 and p.get('clean') is True and p.get('n_scenes') == 20 for p in points)
                    valid += int(is_clean)
                    failures += sum(20 - int(p.get('n_valid', 0)) for p in points)
                    for key in keys:
                        values = [p.get(key) for p in points]
                        if any(v is None for v in values):
                            continue
                        average = statistics.mean(values)
                        flagged[key].append(average)
                        if is_clean:
                            clean[key].append(average)
                if group:
                    output.append(dict(case_id=case, mode=mode, radius_hr=radius,
                                       n_students=len(group), expected_n=len(servers)*4,
                                       n_clean_students=valid, n_flagged_students=len(group)-valid,
                                       invalid_scene_directions=failures,
                                       clean={k: stats(v) for k, v in clean.items()},
                                       flagged_all={k: stats(v) for k, v in flagged.items()}))
    return output


def paired_modes(curves, servers=SERVERS):
    """A_ON - fixed/zero/inverse within the same Student; modes do not enlarge n."""
    metrics = ('ergas', 'psnr', 'sam', 'edge_error_dn', 'relative_response_l1_sum')
    blocks = {}
    for curve in curves:
        key = (curve['case_id'], curve['server'], curve['repeat'], curve['seed'])
        by_mode = blocks.setdefault(key, {})
        if curve['mode'] in by_mode:
            raise ValueError('Duplicate Student/mode curve')
        by_mode[curve['mode']] = curve
    output = []
    for case in CASES[:6]:
        for comparison in ('A_ZERO_INFERENCE_ONLY', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE'):
            for radius in (0., .25, .5, 1., 2., 3., 4.):
                clean = {m: [] for m in metrics}; flagged = {m: [] for m in metrics}; pairs = 0; clean_pairs = 0
                for (block_case, server, repeat, seed), modes in blocks.items():
                    if block_case != case or 'A_ON' not in modes or comparison not in modes:
                        continue
                    on, other = modes['A_ON'], modes[comparison]
                    if on['checkpoint_sha256'] != other['checkpoint_sha256']:
                        raise ValueError('Paired inference modes used different checkpoint bytes')
                    left = [p for p in on['points'] if p['radius_hr'] == radius]
                    right = [p for p in other['points'] if p['radius_hr'] == radius]
                    if len(left) != (1 if radius == 0 else 8) or [p['id'] for p in left] != [p['id'] for p in right]:
                        raise ValueError('Paired mode direction grid differs')
                    pairs += 1
                    valid = all(p.get('clean') is True and p.get('n_valid') == 20 for p in (*left, *right))
                    clean_pairs += int(valid)
                    for metric in metrics:
                        if any(p.get(metric) is None for p in (*left, *right)):
                            continue
                        delta = statistics.mean(p[metric]-q[metric] for p, q in zip(left, right))
                        row = dict(server=server, repeat=repeat, seed=seed, delta=delta)
                        flagged[metric].append(row)
                        if valid:
                            clean[metric].append(row)
                if pairs:
                    summaries = {}
                    for metric in metrics:
                        ci = paired_statistics(clean[metric], metric, expected_n=len(servers)*4)
                        ci.pop('p_two_sided')  # No unregistered expansion of the primary significance family.
                        summaries[metric] = dict(clean=ci, flagged_all=stats(r['delta'] for r in flagged[metric]))
                    output.append(dict(case_id=case, contrast='A_ON_MINUS_' + comparison, radius_hr=radius,
                                       n_paired_students=pairs, n_clean_students=clean_pairs,
                                       n_flagged_students=pairs-clean_pairs, metrics=summaries,
                                       interpretation='Correction dependence/response in same Student; inverse is not an absolute alignment oracle'))
    return output
