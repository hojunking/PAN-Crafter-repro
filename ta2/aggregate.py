"""Three independent seeds and paired contrasts, not scene/checkpoint pseudo-N."""
from __future__ import annotations

import numpy as np


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and np.isfinite(value)


def seed_aggregate(records, *, metric, expected_seeds, scope_keys=('dataset', 'case_id', 'selector', 'source_revision')):
    """Input record: seed,status,metric and scope keys. Retries are not replicas.

    Caller resolves which attempt is canonical, rather than this routine
    silently choosing the best successful retry of the same seed.
    """
    expected = list(expected_seeds)
    if len(expected) != 3 or len(set(expected)) != 3:
        raise ValueError('Exactly three preregistered independent seeds required')
    by_seed = {}
    scopes = set()
    for row in records:
        if row.get('record_type') in ('ORACLE', 'COUNTERFACTUAL'):
            raise ValueError('Interventions and oracle cannot enter native seed aggregate')
        seed = row['seed']
        if seed not in expected or seed in by_seed:
            raise ValueError('Unexpected or duplicated seed; retries must be resolved explicitly')
        scopes.add(tuple(row.get(k) for k in scope_keys))
        if any(k not in row for k in scope_keys):
            raise ValueError('Missing aggregation scope identity')
        if row.get('status') == 'COMPLETE' and not _number(row.get(metric)):
            raise ValueError('Completed seed lacks a finite measured metric')
        by_seed[seed] = row
    if len(scopes) > 1:
        raise ValueError('Cannot aggregate across dataset/case/selector/source revision')
    successful = [s for s in expected if s in by_seed and by_seed[s]['status'] == 'COMPLETE' and _number(by_seed[s].get(metric))]
    failed = [s for s in expected if s in by_seed and by_seed[s]['status'] in ('FAILED', 'NUMERICAL_FAILURE', 'BLOCKED')]
    values = np.asarray([by_seed[s][metric] for s in successful], dtype=np.float64)
    return dict(record_type='SEED_AGG', metric=metric, seed_values={str(s):float(by_seed[s][metric]) if s in successful else None for s in expected},
                seed_status={str(s):by_seed[s]['status'] if s in by_seed else 'NOT_ATTEMPTED' for s in expected},
                n_expected=3, n_success=len(successful), n_attempted=len(by_seed), n_failed=len(failed),
                failure_fraction=len(failed)/len(by_seed) if by_seed else None,
                mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values)>1 else None, std_ddof=1,
                median=float(np.median(values)) if len(values) else None,
                minimum=float(values.min()) if len(values) else None, maximum=float(values.max()) if len(values) else None,
                status='COMPLETE_3_SEEDS' if len(successful)==3 else 'INCOMPLETE_3_SEEDS',
                sampling_unit='independent_training_seed', geographic_independence='unverified',
                scope=dict(zip(scope_keys, next(iter(scopes)))) if scopes else None)


def paired_difference(treatment, control, *, metric, expected_seeds):
    if len(expected_seeds) != 3 or len(set(expected_seeds)) != 3:
        raise ValueError('Paired contrast requires exactly three preregistered seeds')
    by_t = {r['seed']:r for r in treatment}; by_c = {r['seed']:r for r in control}
    if len(by_t) != len(treatment) or len(by_c) != len(control):
        raise ValueError('Duplicate training seed/attempt in paired contrast')
    if (set(by_t) | set(by_c)) - set(expected_seeds):
        raise ValueError('Unregistered seed in paired contrast')
    rows = []
    pairs = []
    for seed in expected_seeds:
        t, c = by_t.get(seed), by_c.get(seed)
        if t is None or c is None or t.get('status') != 'COMPLETE' or c.get('status') != 'COMPLETE':
            rows.append(dict(seed=seed, status='PAIR_INCOMPLETE', difference=None)); continue
        for key in ('dataset', 'replica', 'initial_u_sha256', 'native_stream_sha256', 'source_revision', 'selector'):
            if key not in t or key not in c or t[key] != c[key]:
                raise ValueError('Unpaired contrast identity: ' + key)
        if ('initial_a_sha256' not in t or 'initial_a_sha256' not in c
                or (t['initial_a_sha256'] is not None and c['initial_a_sha256'] is not None
                    and t['initial_a_sha256'] != c['initial_a_sha256'])):
            raise ValueError('Unpaired aligner initialization among A-present cases')
        if t.get('record_type') in ('ORACLE', 'COUNTERFACTUAL') or c.get('record_type') in ('ORACLE', 'COUNTERFACTUAL'):
            raise ValueError('Oracle/intervention is not a native paired training contrast')
        if not _number(t.get(metric)) or not _number(c.get(metric)):
            rows.append(dict(seed=seed, status='PAIR_NONFINITE', difference=None)); continue
        delta = float(t[metric]-c[metric]); pairs.append(delta)
        rows.append(dict(seed=seed, replica=t['replica'], status='COMPLETE', treatment=t['case_id'], control=c['case_id'],
                         difference=delta, treatment_value=float(t[metric]), control_value=float(c[metric])))
    return dict(record_type='PAIRED', metric=metric, direction='treatment_minus_control_no_sign_flip', rows=rows,
                n_pairs=len(pairs), n_expected=len(expected_seeds), mean=float(np.mean(pairs)) if pairs else None,
                std=float(np.std(pairs, ddof=1)) if len(pairs)>1 else None, std_ddof=1,
                sampling_unit='paired_independent_training_seed', status='COMPLETE' if len(pairs)==3 else 'INCOMPLETE')


def scene_paired_interval(treatment, control, *, metric, seed=261006900, repeats=2000, parent_ids=None):
    """Separate scene-cluster CI; never substitutes for three-seed variation."""
    t = {r['scene_index']:r for r in treatment}; c = {r['scene_index']:r for r in control}
    if len(t)!=len(treatment) or len(c)!=len(control) or set(t)!=set(c) or not t:
        raise ValueError('Scene paired coverage mismatch')
    ids = sorted(t)
    differences = np.asarray([t[i][metric]-c[i][metric] for i in ids], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite scene paired values')
    parents = [parent_ids[i] for i in ids] if parent_ids is not None else ids
    unique = sorted(set(parents))
    clusters = [differences[np.asarray([p==u for p in parents])] for u in unique]
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = []
    if len(clusters)>1:
        chosen = rng.integers(len(clusters), size=(repeats,len(clusters)))
        totals=np.asarray([c.sum() for c in clusters]);sizes=np.asarray([len(c) for c in clusters])
        estimates=totals[chosen].sum(1)/sizes[chosen].sum(1)
    order = np.argsort(differences)
    return dict(mean=float(differences.mean()), ci95=np.quantile(estimates,[.025,.975]).tolist() if len(estimates) else None,
                n_scenes=len(ids), n_clusters=len(clusters), n_training_seeds=1,
                scope='parent_cluster' if parent_ids is not None else 'scene_cluster',
                geographic_independence='provided_parent_registry' if parent_ids is not None else 'unverified',
                scene_differences=[dict(scene_index=i, difference=float(d)) for i,d in zip(ids,differences)],
                worst=dict(scene_index=ids[int(order[0])],difference=float(differences[order[0]])),
                best=dict(scene_index=ids[int(order[-1])],difference=float(differences[order[-1]])),
                median=float(np.median(differences)), bootstrap_repeats=repeats,
                note='numeric ordering; favorable direction depends on metric; CI is not seed STD')
