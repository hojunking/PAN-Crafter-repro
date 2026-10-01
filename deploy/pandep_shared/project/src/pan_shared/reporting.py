"""Exposure-matched reports from immutable local observations, never Sheet inputs."""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path

from .common import CAMPAIGN, atomic_json, canonical_sha
from .sheets import result_id

METRICS = ('hqnr', 'd_s', 'd_lambda', 'jqm', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'rmse', 'cc')


def stats(values):
    values = [v for v in values if type(v) in (float, int) and math.isfinite(v)]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def _metric(row, metric):
    return row['fr'].get(metric, row['rr'].get(metric))


def _cohort(row):
    identity = row.get('notes', {}).get('identity', {})
    keys = ('source_sha256', 'environment_sha256', 'data_sha256', 'protocol_sha256')
    if any(not identity.get(k) for k in keys) or not row.get('evaluator_sha256'):
        return None
    return canonical_sha([row['evaluator_sha256'], *[identity[k] for k in keys]])


def _unique(observations):
    ids = {}
    for row in observations:
        if row.get('campaign_id', CAMPAIGN) != CAMPAIGN:
            continue
        if row.get('payload_sha256') != canonical_sha({k: v for k, v in row.items() if k != 'payload_sha256'}):
            raise ValueError('Observation payload seal missing or mismatched')
        key = result_id(row)
        if key in ids and canonical_sha(ids[key]) != canonical_sha(row):
            raise ValueError('Conflicting immutable observation payload for '+key)
        ids[key] = row
    return list(ids.values())


def _select(rows, run, sensor, stage, secondary=False, auxiliary=False):
    endpoint = (75000 if auxiliary else run['exposure_stage_updates']*stage)
    selector = ('BEST_JOINT_VAL' if run['mode'] == 'SHARED' else 'BEST_SENSOR_VAL') + f'_TO_B{endpoint//50000:04d}' if secondary else 'EXACT'
    candidates = [r for r in rows if r['run_id'] == run['run_id'] and r['sensor'] == sensor
                  and r['selector_scope'] == selector
                  and (r['completed_step'] == endpoint if secondary else r['selected_step'] == endpoint)]
    if len(candidates) > 1:
        raise ValueError('Multiple model/evaluator observations for the same registered analysis scope')
    if not candidates:
        return None, 'NOT_SUPPLIED'
    row = candidates[0]
    for field in ('case_id', 'repeat', 'seed', 'attempt', 'mode', 'config_sha256', 'width', 'depth', 'train_fraction'):
        if row[field] != run[field]:
            raise ValueError('Observation registry mismatch: '+field)
    if row.get('protocol_id') != 'PANDEP_NATIVE_RR20_FR20_v1':
        raise ValueError('Unregistered observation evaluation protocol')
    if row['rr'].get('n_scenes') != 20 or row['fr'].get('n_scenes') != 20:
        return None, 'INCOMPLETE_RR20_FR20'
    if _cohort(row) is None:
        return None, 'MISSING_NUMERICAL_PROVENANCE'
    return row, None


def _panel(rows, registry, stage, secondary=False):
    groups, selected = {}, {}
    for run in registry['runs']:
        for sensor in run['sensors']:
            row, reason = _select(rows, run, sensor, stage, secondary)
            group = groups.setdefault((run['case_id'], sensor), {'rows': [], 'missing': [], 'expected_repeats': []})
            group['expected_repeats'].append(run['repeat'])
            if row:
                group['rows'].append(row)
                selected[(run['case_id'], sensor, run['repeat'])] = row
            else:
                group['missing'].append({'run_id': run['run_id'], 'repeat': run['repeat'], 'reason': reason})
    panels = []
    checkpoint_by_run = {}
    for row in selected.values():
        previous = checkpoint_by_run.setdefault(row['run_id'], row['checkpoint_sha256'])
        if previous != row['checkpoint_sha256']:
            raise ValueError('Shared sensors may not combine different checkpoint bytes')
    for (case, sensor), group in sorted(groups.items()):
        cohorts = {}
        for row in group['rows']:
            cohorts.setdefault(_cohort(row), []).append(row)
        # A changed evaluator/environment/source is never silently pooled into one mean.
        for cohort, members in sorted(cohorts.items()) if cohorts else [(None, [])]:
            missing = group['missing'] + [{'repeat': r['repeat'], 'run_id': r['run_id'], 'reason': 'OTHER_COHORT'}
                                          for r in group['rows'] if r not in members]
            panels.append({'case_id': case, 'sensor': sensor, 'cohort_id': cohort,
                'expected_n': len(group['expected_repeats']), 'n_students': len(members), 'missing': missing,
                'metrics': {m: stats([_metric(r, m) for r in members]) for m in METRICS+('q8' if sensor == 'WV3' else 'q4',)},
                'repeats': [r['repeat'] for r in members],
                'observations': [{'result_id': result_id(r), 'run_id': r['run_id'], 'repeat': r['repeat'],
                    'seed': r['seed'], 'checkpoint_sha256': r['checkpoint_sha256'], 'selected_step': r['selected_step'],
                    'selector_scope': r['selector_scope'], 'candidate_count': r.get('notes', {}).get('candidate_count'),
                    'selected_through_global_step': r.get('notes', {}).get('selected_through_global_step'),
                    'exposure': r.get('notes', {}).get('selected_exposures', {}).get(sensor),
                    'cost': r.get('cost', {})} for r in members]})
    return panels, selected


def _pairs(selected):
    results = []
    for sensor, single in [('WV3', 'S01'), ('GF2', 'S02'), ('QB', 'S03')]:
        for case, baseline in [('C00', single), ('C11', 'C00'), ('C01', 'C00')]:
            values, missing = [], []
            for repeat in (1, 2, 3):
                left, right = selected.get((case, sensor, repeat)), selected.get((baseline, sensor, repeat))
                reason = None
                if not left or not right: reason = 'PAIR_NOT_SUPPLIED'
                elif left['seed'] != right['seed'] or _cohort(left) != _cohort(right): reason = 'PAIR_SEED_OR_COHORT_MISMATCH'
                if reason:
                    missing.append({'repeat': repeat, 'reason': reason}); continue
                metrics = {}
                for metric in METRICS+('q8' if sensor == 'WV3' else 'q4',):
                    a, b = _metric(left, metric), _metric(right, metric)
                    metrics[metric] = a-b if a is not None and b is not None else None
                values.append({'repeat': repeat, 'seed': left['seed'], 'left_result_id': result_id(left),
                    'right_result_id': result_id(right), 'metrics': metrics})
            results.append({'case_id': case, 'baseline_case_id': baseline, 'sensor': sensor,
                'direction': 'case_minus_baseline', 'expected_n': 3, 'n_pairs': len(values), 'missing': missing,
                'metrics': {m: stats([v['metrics'][m] for v in values]) for m in METRICS+('q8' if sensor == 'WV3' else 'q4',)},
                'members': values})
    return results


def summarize_stage(observations, stage, registry):
    if type(stage) is not int or stage < 1:
        raise ValueError('Positive exposure stage required')
    rows = _unique(observations)
    primary, chosen = _panel(rows, registry, stage)
    secondary, best = _panel(rows, registry, stage, secondary=True)
    auxiliary = []
    if stage == 1:
        for run in registry['runs']:
            if run['case_id'] != 'C01': continue
            for sensor in run['sensors']:
                half, reason = _select(rows, run, sensor, 1, auxiliary=True)
                full = chosen.get(('C00', sensor, run['repeat']))
                valid = half is not None and full is not None and _cohort(half) == _cohort(full)
                auxiliary.append({'sensor': sensor, 'repeat': run['repeat'], 'kind': 'APPROXIMATE_EQUAL_EFFECTIVE_EPOCH',
                    'not_equal_updates_or_compute': True, 'status': 'MEASURED' if valid else 'NOT_SUPPLIED_OR_COHORT_MISMATCH',
                    'half_result_id': result_id(half) if half else None, 'full_result_id': result_id(full) if full else None,
                    'half_exposure': half.get('notes', {}).get('selected_exposures', {}).get(sensor) if half else None,
                    'full_exposure': full.get('notes', {}).get('selected_exposures', {}).get(sensor) if full else None,
                    'metrics_half_minus_full': {m: _metric(half, m)-_metric(full, m) for m in ('hqnr', 'ergas', 'scc')} if valid else {}})
    by_run = {}
    for row in list(chosen.values())+list(best.values()):
        old = by_run.get(row['run_id'])
        if old is None or (row['completed_step'], row.get('cost', {}).get('wall_hours') or 0) > (old['completed_step'], old.get('cost', {}).get('wall_hours') or 0):
            by_run[row['run_id']] = row
    return {'schema': 'PANDEP_MATCHED_EXPOSURE_REPORT_v1', 'campaign_id': CAMPAIGN, 'exposure_stage': stage,
            'primary': primary, 'primary_paired_differences': _pairs(chosen),
            'secondary_validation_selected': secondary, 'secondary_paired_differences': _pairs(best),
            'auxiliary_half75k_full150k': auxiliary,
            'run_costs': [{'run_id': r['run_id'], 'source_result_id': result_id(r),
                           'completed_step': r['completed_step'], 'cost': r.get('cost', {})} for r in by_run.values()],
            'cost_aggregation': 'one latest cumulative observation per Run_ID; shared sensor rows never summed',
            'statistical_unit': 'registered logical run/repeat, not checkpoint/sensor/scene',
            'limitations': ['BEST search scopes differ between shared and single; secondary is not matched exposure',
                'Missing repeats stay in denominator; mixed source/evaluator/environment cohorts not pooled',
                'Train/test geographic independence not established; no private data used',
                'Latency is server-only, not laptop measurement; unmeasured costs remain null',
                'Cause labels are candidates, never automatic recipe or queue decisions']}


def _curves(work, stage, registry):
    result = []
    for run in registry['runs']:
        endpoint = stage*run['exposure_stage_updates']
        entries = {'run_id': run['run_id'], 'through_step': endpoint, 'logs': {}}
        for name in ('train_steps.jsonl', 'probes.jsonl', 'validation.jsonl'):
            path = work/'runs'/run['run_id']/name
            records = []
            if path.exists():
                for line in path.read_text().splitlines():
                    value = json.loads(line)
                    if value['step'] <= endpoint: records.append(value)
            # Keep raw files authoritative; report first/latest and all full validation values.
            entries['logs'][name] = {'count': len(records), 'first': records[0] if records else None,
                'latest': records[-1] if records else None,
                'validation_curve': records if name == 'validation.jsonl' else None}
        entries['interpretation'] = 'INSUFFICIENT_TREND_EVIDENCE' if entries['logs']['validation.jsonl']['count'] < 2 else 'TRAIN_VALIDATION_TRENDS_REQUIRE_PAIRED_INTERPRETATION'
        result.append(entries)
    return result


def _format(value):
    return '—' if value is None else f'{value:.6g}'


def _markdown(report):
    lines = [f"# Exposure stage k={report['exposure_stage']}", '',
             'Generated local report. Exact matched exposure is primary; validation-selected results are separate.', '',
             '## Primary EXACT matched-exposure results', '',
             '| Case | Sensor | n / expected | HQNR mean | sample SD | ERGAS mean | sample SD | SCC mean |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for group in report['primary']:
        h, e, s = (group['metrics'][k] for k in ('hqnr', 'ergas', 'scc'))
        lines.append(f"| {group['case_id']} | {group['sensor']} | {group['n_students']}/{group['expected_n']} | {_format(h['mean'])} | {_format(h['sample_sd'])} | {_format(e['mean'])} | {_format(e['sample_sd'])} | {_format(s['mean'])} |")
    lines += ['', '## Paired differences (case minus baseline)', '',
              '| Pair | Sensor | n / 3 | ΔHQNR | ΔERGAS | ΔSCC |', '| --- | --- | --- | --- | --- | --- |']
    for group in report['primary_paired_differences']:
        lines.append(f"| {group['case_id']} − {group['baseline_case_id']} | {group['sensor']} | {group['n_pairs']}/3 | "+
                     ' | '.join(_format(group['metrics'][k]['mean']) for k in ('hqnr', 'ergas', 'scc'))+' |')
    lines += ['', '## Secondary validation-selected results', '',
              'Whole shared checkpoint selected by macro-sensor validation L1; no per-sensor best mixing.', '',
              '| Case | Sensor | n / expected | HQNR mean | ERGAS mean |', '| --- | --- | --- | --- | --- |']
    for group in report['secondary_validation_selected']:
        lines.append(f"| {group['case_id']} | {group['sensor']} | {group['n_students']}/{group['expected_n']} | {_format(group['metrics']['hqnr']['mean'])} | {_format(group['metrics']['ergas']['mean'])} |")
    lines += ['', '## Exposure, costs, missing evidence and interpretation', '',
              'Per-run samples, distinct IDs, exact effective epochs, selected steps, provenance, missing reasons,',
              'paired seed values, cumulative costs and train/probe/full-validation curves are in the adjacent JSON.',
              'C01@75K versus C00@150K is approximate equal effective epochs, not equal updates/compute.',
              'Costs repeat across shared sensor rows: do not sum them. No laptop or tile-release claim.', '',
              'Interpret UNDERTRAINING / CAPACITY_OR_OPTIMIZATION / OVERFIT / CROSS_SENSOR_INTERFERENCE only',
              'as candidate explanations supported by the paired train-validation curves; do not auto-tune the campaign.', '',
              '### Limitations', '']
    lines.extend('- '+text for text in report['limitations'])
    lines += ['', 'Source observations and ownership hash: '+report['source_observations_sha256'], '']
    return '\n'.join(lines)


def write_stage_report(work_root, stage, registry):
    work = Path(work_root).resolve()
    all_rows = [json.loads(path.read_text()) for path in sorted((work/'observations').glob('*.json'))]
    endpoints = {r['run_id']: stage*r['exposure_stage_updates'] for r in registry['runs']}
    rows = [r for r in all_rows if r['run_id'] in endpoints and r['completed_step'] <= endpoints[r['run_id']]]
    report = summarize_stage(rows, stage, registry)
    report['training_validation_curves'] = _curves(work, stage, registry)
    report['source_observations_sha256'] = canonical_sha({result_id(r): r['payload_sha256'] for r in _unique(rows)})
    directory = work/'reports'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'exposure_k{stage:04d}.json'
    markdown = directory/f'exposure_k{stage:04d}.md'
    text = _markdown(report)
    if path.exists() and json.loads(path.read_text()) != report:
        raise ValueError('Existing completed-stage report differs; keep immutable report and investigate')
    if markdown.exists() and markdown.read_text() != text:
        raise ValueError('Existing stage report text differs; refusing to overwrite user edits')
    if not path.exists(): atomic_json(path, report)
    if not markdown.exists():
        fd, temporary = tempfile.mkstemp(prefix='.report-', dir=directory)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, markdown)
    return {'json': str(path), 'markdown': str(markdown), 'report': report}
