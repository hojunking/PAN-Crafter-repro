"""Balanced, local-BASE Table A reports; historical-seed anchors stay separate."""
from pathlib import Path
import math
import statistics

from maina_hqnr.common import ROOT, atomic_json, camp, object_sha, read_config
from maina_hqnr.postrun import SELECTIONS

METRICS = ('hqnr', 'd_lambda', 'd_s', 'ergas', 'sam', 'psnr', 'scc', 'ssim', 'q8', 'rmse', 'cc', 'jqm')
PAIR_KEYS = ('runtime_content_sha256', 'teacher_checkpoint_sha256', 'initial_U_sha256',
             'initial_A_sha256', 'stream_sha256', 'binding_sha256', 'data_sha256', 'evaluator_sha256')
AXES = {'alpha': ('AL05', 'BASE', 'AL15'), 'beta': ('BE005', 'BASE', 'BE020'),
        'lambda_E': ('ED0006', 'BASE', 'ED006')}


def metrics_for(summary, selection):
    selected = summary['selections'][selection]
    return {key: selected['fr' if key in ('hqnr', 'd_lambda', 'd_s', 'jqm') else 'rr'][key] for key in METRICS}


def paired_deltas(summary, baseline, selection):
    if not summary.get('complete') or not baseline.get('complete') or selection not in SELECTIONS:
        raise ValueError('Only completed same-selector evaluations can be paired')
    case, base = summary['case'], baseline['case']
    if (base['case_id'] != 'BASE' or base['run_id'] != case['local_baseline_run_id']
            or any(case[key] != base[key] for key in ('server', 'cycle', 'seed'))):
        raise ValueError('BASE must be from the same server/cycle/seed')
    if any(not summary['provenance'].get(key) or summary['provenance'][key] != baseline['provenance'].get(key)
           for key in PAIR_KEYS):
        raise ValueError('Paired source/data/F1/initialization/stream/evaluator differs')
    current, original = metrics_for(summary, selection), metrics_for(baseline, selection)
    return dict(values={key: current[key]-original[key] for key in METRICS},
                baseline_run_id=base['run_id'], baseline_attempt=baseline['attempt'],
                baseline_selection=selection, baseline_summary_sha256=object_sha(baseline),
                scope='same server/cycle/selector, paired Student seed, single fixed F1')


def _stats(values):
    if not values:
        return dict(n=0, mean=None, sd=None, raw=[])
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite aggregate input')
    return dict(n=len(values), mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None, raw=values)


def summarize(summaries, server=None):
    if server is not None and server not in ('s4', 's5'):
        raise ValueError('Only MAIN-A s4/s5 may be reported')
    by_attempt = {}
    for summary in summaries:
        case = summary['case']
        if case['server'] not in ('s4', 's5') or (server is not None and case['server'] != server):
            raise ValueError('Foreign server supplied to local report')
        key = (case['run_id'], summary['attempt'])
        if key in by_attempt and by_attempt[key] != summary:
            raise ValueError('Conflicting evidence for the same attempt')
        by_attempt[key] = summary
    complete = {}; failures = []; raw = []; blocks = {}
    for summary in by_attempt.values():
        case = summary['case']; block = (case['server'], case['cycle'])
        blocks.setdefault(block, {})
        if not summary.get('complete'):
            failures.append(dict(run_id=case['run_id'], attempt=summary['attempt'], server=case['server'],
                cycle=case['cycle'], seed=case['seed'], case_id=case['case_id'], status=summary['status'],
                reason=summary.get('reason', ''), actual_updates=summary.get('actual_updates')))
            continue
        if case['run_id'] in complete:
            raise ValueError('Two successful attempts cannot count as two independent Student seeds')
        complete[case['run_id']] = summary
        blocks[block][case['case_id']] = summary
    for summary in complete.values():
        case = summary['case']; baseline = complete.get(case['local_baseline_run_id'])
        for selection in SELECTIONS:
            row = dict(run_id=case['run_id'], attempt=summary['attempt'], server=case['server'],
                cycle=case['cycle'], seed=case['seed'], case_id=case['case_id'], selection=selection,
                anchor=case['server'] == 's4' and case['cycle'] == 0,
                raw=metrics_for(summary, selection), status='COMPLETE',
                selection_alias_of=summary['selections'][selection].get('alias_of'))
            try:
                if baseline is None:
                    raise ValueError('Local completed BASE not yet available')
                row['paired_delta'] = paired_deltas(summary, baseline, selection)
                row['pair_status'] = 'COMPLETE'
            except ValueError as exc:
                row.update(pair_status='NOT_COMPARABLE', pair_reason=str(exc), paired_delta=None)
            raw.append(row)
    balanced = []; incomplete = []; fresh_complete = []
    for block, runs in sorted(blocks.items()):
        anchor = block == ('s4', 0)
        if len(runs) == 7:
            try:
                for run in runs.values():
                    paired_deltas(run, runs['BASE'], 'HQNR_MAX50')
                if not anchor:
                    fresh_complete.append(dict(server=block[0], cycle=block[1], seed=runs['BASE']['case']['seed']))
            except ValueError:
                pass
        for axis, codes in AXES.items():
            for selection in SELECTIONS:
                if anchor:
                    continue
                try:
                    if any(code not in runs for code in codes):
                        raise ValueError('Incomplete low/default/high block')
                    pairs = {code: paired_deltas(runs[code], runs['BASE'], selection) for code in codes}
                    identity = {key: runs['BASE']['provenance'][key] for key in
                                ('runtime_content_sha256', 'teacher_checkpoint_sha256', 'data_sha256', 'evaluator_sha256')}
                    balanced.append(dict(axis=axis, selection=selection, server=block[0], cycle=block[1],
                        seed=runs['BASE']['case']['seed'], protocol_group=object_sha(identity),
                        source_identity=identity,
                        values={code: metrics_for(runs[code], selection) for code in codes},
                        paired_delta={code: pairs[code]['values'] for code in codes}))
                except ValueError as exc:
                    incomplete.append(dict(axis=axis, selection=selection, server=block[0], cycle=block[1],
                                           status='INCOMPLETE_NOT_COMPARABLE', reason=str(exc)))
    aggregate = []
    groups = sorted({(r['axis'], r['selection'], r['protocol_group']) for r in balanced})
    for axis, selection, protocol_group in groups:
        rows = [r for r in balanced if (r['axis'], r['selection'], r['protocol_group']) == (axis, selection, protocol_group)]
        seeds = [r['seed'] for r in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate fresh seed would inflate sensitivity evidence')
        aggregate.append(dict(axis=axis, selection=selection, protocol_group=protocol_group,
            n_paired_student_seeds=len(rows), balanced_seed_set=seeds,
            values={code: {key: _stats([r['values'][code][key] for r in rows]) for key in METRICS} for code in AXES[axis]},
            paired_delta={code: {key: _stats([r['paired_delta'][code][key] for r in rows]) for key in METRICS} for code in AXES[axis]}))
    counts = {}
    for summary in by_attempt.values():
        counts[summary['status']] = counts.get(summary['status'], 0)+1
    return dict(schema='MAINA_BALANCED_TABLE_A_v1', server=server, raw=raw, failures=failures,
        anchors=[row for row in raw if row['anchor']], aggregate=aggregate, incomplete_blocks=incomplete,
        fresh_complete_blocks=fresh_complete, status_counts=counts,
        initial_three_seed_summary_ready=len(fresh_complete) >= 3,
        six_seed_summary_ready=len(fresh_complete) >= 6, early_stop=False,
        default_reused_across_axes=True, default_is_three_independent_observations=False,
        aliases_are_independent_runs=False, teacher_seed_generalization=False)


def build_report(root=ROOT, server='s4'):
    from maina_hqnr.upload import status_summary
    from maina_hqnr.postrun import verify_summary_for_upload
    folders = {path.parent for path in (camp(root, server)/'runs').glob('*/attempt*/config.json')}
    folders.update(path.parent.parent for path in (camp(root, server)/'runs').glob('*/attempt*/meta/config.resolved.yaml'))
    summaries = []
    for folder in sorted(folders):
        if not (folder/'meta/training_status.json').exists() and not (folder/'meta/attempt_failure.json').exists():
            continue
        summary = status_summary(folder, root)
        if summary.get('complete'):
            summary = verify_summary_for_upload(folder, root)
        summaries.append(summary)
    report = summarize(summaries, server)
    atomic_json(camp(root, server)/'reporting/paired_results.json', report)
    return report
