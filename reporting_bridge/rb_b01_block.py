"""Pure, wide B01 display for an owned block in existing ``ablations``.

No worksheet/client access occurs here. Coordinates are relative to the owned
block; the placement writer must discover/protect existing content and add the
absolute range to the first-cell ownership note. Literal values are refreshed
from verified per-Student sources, never independently recomputed image metrics.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

from .rb_b01_contract import (
    CAMPAIGN_ID, CASES, MODES, RADII, SELECTIONS, SERVERS, SPREADSHEET_ID,
    STRESS_METRICS, ContractError, assert_no_conflicts, build_native_row,
    cohort_id, compact_json, finite_number, native_result_id, registry,
    source_row, stress_record_id, stress_source_row, validate_case,
)
from .rb_b01_summary import summarize

OWNER = 'B01_EXISTING_SHEETS_APPEND_ONLY_v2'
SCHEMA = 'B01_ABLATIONS_WIDE_BLOCK_v2'
TITLE = 'Rebuttal B01 — RB01 q reliability / RB02 controlled shifts'
COLUMN_COUNT = 31
SECTION_IDS = ('status', 'rb01_primary', 'rb01_paired', 'rb01_secondary',
               'rb01_individual', 'rb02_radius', 'rb02_paired', 'provenance')
PRIMARY_METRICS = ('hqnr', 'd_s', 'd_lambda', 'ergas', 'psnr', 'sam')
OTHER_METRICS = ('scc', 'ssim', 'q8', 'jqm', 'rmse', 'cc')
ALL_NATIVE_METRICS = PRIMARY_METRICS + OTHER_METRICS
LABELS = {'hqnr': 'HQNR↑', 'd_s': 'D_s↓', 'd_lambda': 'D_lambda↓', 'ergas': 'ERGAS↓',
          'psnr': 'PSNR↑', 'sam': 'SAM↓', 'scc': 'SCC↑', 'ssim': 'SSIM↑', 'q8': 'Q8↑',
          'jqm': 'JQM↑', 'rmse': 'RMSE↓', 'cc': 'CC↑', 'edge_error_dn': 'Edge DN↓',
          'relative_response_l1_sum': 'Response L1', 'coverage_all_pan_paths': 'Coverage'}
NATIVE_GIDS = {'s1': 269300000, 's3': 269300001, 's5': 269300002}
STRESS_GID = 269300003
STATUS_GID = 269300004


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _url(gid, cell_range=None):
    link = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={gid}'
    return link if cell_range is None else link + '&range=' + cell_range


def _cell(value, *, note=None, heading=False):
    if value is None:
        value = ''
    if type(value) in (int, float):
        finite_number(value, 'block cell')
        entered = {'numberValue': value}
        fmt = {'numberFormat': {'type': 'NUMBER', 'pattern': '0' if type(value) is int else '0.0000'}}
    elif type(value) is bool:
        entered, fmt = {'boolValue': value}, {}
    elif isinstance(value, str):
        entered, fmt = {'stringValue': value}, {}
    else:
        raise ContractError('Unsupported display cell; use notes for structured provenance')
    result = {'userEnteredValue': entered}
    if heading:
        fmt.update(textFormat={'bold': True}, wrapStrategy='WRAP')
    if fmt:
        result['userEnteredFormat'] = fmt
    if note is not None:
        result['note'] = note if isinstance(note, str) else compact_json(note)
    return result


def _empty_stat():
    return dict(n=0, mean=None, sample_std=None, n_students=0, expected_n=6,
                missing_n=6, failed_n=0, flagged_n=0, full_six=False,
                improved=0, ties=0, member_ids=[], member_pairs=[], student_values=[])


def _equivalent(actual, expected):
    """Account only for JSON/stdlib float roundoff, not missing data or counts."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _equivalent(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equivalent(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected) <= 1e-12*max(1,abs(expected))
    return type(actual) is type(expected) and actual == expected


def render_block(prepared, delivery=None):
    """Return 31-column Google RowData, exact expected values and section bounds.

    All statistics are recalculated from existing canonical scalar evidence and
    checked against the prepared summary. This is NOT model/metric evaluation.
    Missing Student slots are reserved, with empty metrics rather than zero.
    One-cohort capacity/coordinates stay fixed as s3/s5 evidence later arrives.
    Additional incompatible cohorts expand separate panels, never pooled six.
    """
    if 'payload_sha256' in prepared:
        source_hash = prepared['payload_sha256']
        if _digest({k: v for k, v in prepared.items() if k != 'payload_sha256'}) != source_hash:
            raise ContractError('Prepared display source hash mismatch')
    else:
        source_hash = _digest(prepared)
    native = assert_no_conflicts(prepared.get('native_records', []))
    stress = assert_no_conflicts(prepared.get('stress_records', []))
    statuses = prepared.get('statuses', [])
    recomputed = summarize(native, stress, statuses)
    if 'summary' in prepared and not _equivalent(prepared['summary'], recomputed):
        raise ContractError('Prepared summary differs from source-backed Student statistics')
    summary = prepared.get('summary', recomputed)
    delivery = dict(delivery or {})
    if set(delivery) - {'source_readback_verified', 'view_readback_verified', 'checked_at_utc'}:
        raise ContractError('Unknown block delivery overlay field')
    for key in ('source_readback_verified', 'view_readback_verified'):
        if key in delivery and type(delivery[key]) is not bool:
            raise ContractError('Delivery verification flags must be actual booleans')
    source_readback_verified = delivery.get('source_readback_verified', False)
    status_by_id = {record['run_id']: record for record in statuses}
    native_by_slot = {(record['run_id'], record['selection_id']): record for record in native}
    cohorts = sorted(summary['cohorts']) or [None]
    updated = prepared.get('prepared_at_utc')
    records_by_cohort = {}
    points_by_cohort = {}
    for cohort in cohorts:
        records_by_cohort[cohort] = [row for row in native if cohort_id(row['provenance']) == cohort]
        points_by_cohort[cohort] = [row for row in stress if cohort_id(row['provenance']) == cohort]
    stat_lookup = {(item['cohort_id'], item['selection_id'], item['case_id'], item['metric']): item
                   for item in summary['native']}
    paired_lookup = {(item['cohort_id'], item['selection_id'], item['case_id'], item['metric']): item
                     for item in summary['paired']}
    radius_lookup = {(item['cohort_id'], item['case_id'], item['mode'], item['radius_hr'], item['metric']): item
                     for item in summary['stress_radius']}
    rows, expected, sections = [], [], []
    panel_ranges = []
    native_slots = []

    def add(values=(), *, notes=None, heading=False):
        if len(values) > COLUMN_COUNT:
            raise ContractError('Wide block exceeds existing A:AE column capacity')
        normalized = ['' if value is None else value for value in values]
        normalized += [''] * (COLUMN_COUNT-len(normalized))
        cell_values = [_cell(value, note=(notes or {}).get(i), heading=heading)
                       for i, value in enumerate(normalized)]
        if len(values) == 1 and isinstance(values[0], str):
            cell_values[0].setdefault('userEnteredFormat', {})['wrapStrategy'] = 'OVERFLOW_CELL'
        rows.append({'values': cell_values})
        expected.append(normalized)
        return len(rows)

    def section(identifier, title, explanation):
        if sections:
            sections[-1]['end_row'] = len(rows)
        line = add([title], notes={0: dict(campaign_id=CAMPAIGN_ID,
                   schema=SCHEMA, section_id=identifier, source_hash=source_hash)}, heading=True)
        sections.append(dict(id=identifier, title=title, start_row=line))
        add([explanation])

    def cohort_label(cohort):
        return cohort[:12] if cohort else '자료 미수집 / cohort 미확정'

    def proof(cohort, **extra):
        return dict(schema=SCHEMA, source_hash=source_hash, cohort_id=cohort,
                    cohort_identity=summary['cohorts'].get(cohort), **extra)

    def group_records(cohort, selector, case):
        return [row for row in records_by_cohort[cohort]
                if row['selection_id'] == selector and row['case_id'] == case]

    def native_links(records):
        return [dict(result_id=native_result_id(row), run_id=row['run_id'],
                     checkpoint_sha256=row['checkpoint_sha256'], alias_of=row.get('alias_of'),
                     link=_url(NATIVE_GIDS[row['server']],
                               f"A{source_row(row['repeat'],row['case_id'],row['selection_id'])}:BL{source_row(row['repeat'],row['case_id'],row['selection_id'])}"))
                for row in records]

    def native_panel(selector):
        for cohort in cohorts:
            for metrics in (PRIMARY_METRICS, OTHER_METRICS):
                start = add([f'{selector} / cohort {cohort_label(cohort)} / ' + ', '.join(LABELS[key] for key in metrics)],
                            notes={0: proof(cohort)}, heading=True)
                header = ['Case', 'Cohort', 'Selector', 'Collected Students', 'Expected Students']
                header += [label for key in metrics for label in (LABELS[key]+' mean', 'Sample SD', 'n')]
                header += ['Missing Students', 'Metric missing max', 'Alias observations', 'Full-six clean',
                           'Source', 'Interpretation', 'Source SHA', 'Updated UTC']
                add(header, heading=True)
                for case in CASES:
                    items = [stat_lookup.get((cohort, selector, case, metric), _empty_stat()) for metric in metrics]
                    measured = group_records(cohort, selector, case)
                    n_students = len(measured)
                    values = [case, cohort_label(cohort), selector, n_students, 6]
                    values += [value for item in items for value in (item['mean'], item['sample_std'], item['n'])]
                    values += [6-n_students, max(item['failed_n'] for item in items),
                               sum(bool(row.get('alias_of')) for row in measured),
                               all(item['full_six'] for item in items), 'native source slots' if measured else '미수집',
                               'Student sample SD; n<2 SD empty', source_hash[:12], updated]
                    add(values, notes={0: proof(cohort, selections=native_links(measured)),
                                       1: cohort or 'No verified cohort supplied',
                                       24: 'Metric missing is per-metric unavailable data, not automatically a training/evaluation failure.',
                                       29: source_hash})
                panel_ranges.append(dict(section_id=sections[-1]['id'], cohort_id=cohort,
                                         metrics=list(metrics), start_row=start, end_row=len(rows)))

    def paired_panel():
        for cohort in cohorts:
            for metrics in (PRIMARY_METRICS, OTHER_METRICS):
                start = add([f'EXACT_50000: case − QFULL / cohort {cohort_label(cohort)}'],
                            notes={0: proof(cohort)}, heading=True)
                header = ['Case', 'Cohort', 'Reference / Selector', 'Expected pairs']
                header += [label for key in metrics for label in (LABELS[key]+' Δ mean', 'Sample SD', 'Paired n', 'Improved')]
                header += ['Missing pairs', 'Source', 'Interpretation']
                add(header, heading=True)
                for case in CASES[1:]:
                    items = [paired_lookup.get((cohort, SELECTIONS[0], case, metric), _empty_stat()) for metric in metrics]
                    values = [case, cohort_label(cohort), 'QFULL / EXACT_50000', 6]
                    values += [value for item in items for value in (item['mean'], item['sample_std'], item['n'], item['improved'])]
                    values += [max(item['missing_n'] for item in items), 'matched native slots',
                               'same server/repeat/seed/cohort; case−QFULL']
                    evidence = {metric: dict(member_pairs=item['member_pairs'], ties=item['ties'],
                                failed_n=item['failed_n']) for metric, item in zip(metrics, items)}
                    add(values, notes={0: proof(cohort, metric_evidence=evidence),
                                       1: cohort or 'No verified cohort supplied',
                                       30: 'Improved direction follows the metric: lower ERGAS/SAM/RMSE/D_s/D_lambda; higher HQNR/SCC/PSNR/SSIM/Q8/JQM/CC. Ties are preserved in the Case note.'})
                panel_ranges.append(dict(section_id='rb01_paired', cohort_id=cohort,
                                         metrics=list(metrics), start_row=start, end_row=len(rows)))

    def radius_flags(points, mode, radius):
        wanted_modes = MODES if mode == 'A_ON_MINUS_A_ZERO' else (mode,)
        chosen = [point for point in points if point['mode'] in wanted_modes]
        if mode == 'A_ON_MINUS_A_ZERO':
            registered_modes = {}
            for point in chosen:
                registered_modes.setdefault(point['run_id'], set()).add(point['mode'])
            paired = {run for run, modes in registered_modes.items() if modes == set(MODES)}
            chosen = [point for point in chosen if point['run_id'] in paired]
        at_radius = [point for point in chosen if point['radius_hr'] == radius]
        curves = {(point['run_id'], point['mode']): point for point in chosen}
        if not chosen:
            return dict(radius_numerical_failures=None, geometry_flag_students=None,
                        curve_invalid_geometry=None, curve_numerical_failures=None, curves=0,
                        point_ids=[], first_link=None)
        flagged = {point['run_id'] for point in at_radius
                   if point.get('coverage_all_pan_paths') is not None and point['coverage_all_pan_paths'] < 1.}
        first = at_radius[0] if at_radius else None
        return dict(radius_numerical_failures=sum(point['n_failures'] for point in at_radius),
                    geometry_flag_students=len(flagged),
                    curve_invalid_geometry=sum(point['curve_invalid_geometry'] for point in curves.values()),
                    curve_numerical_failures=sum(point['curve_numerical_failures'] for point in curves.values()),
                    curves=len(curves), point_ids=[stress_record_id(point) for point in at_radius],
                    first_link=None if first is None else _url(STRESS_GID, f'A{stress_source_row(first)}:AQ{stress_source_row(first)}'))

    def stress_panel(paired=False):
        modes = ('A_ON_MINUS_A_ZERO',) if paired else MODES
        for cohort in cohorts:
            start = add([f'RB02 fixed192 / cohort {cohort_label(cohort)}'], notes={0: proof(cohort)}, heading=True)
            header = ['Case', 'Mode', 'Radius HR', 'Cohort', 'Collected Students']
            header += [label for metric in STRESS_METRICS for label in (LABELS[metric]+' mean', 'Sample SD', 'n')]
            header += ['Expected Students', 'Missing Students', 'Failed Students max', 'Numerical failures / radius',
                       'Geometry-flag Students / radius', 'Invalid geometry / FULL curves', 'Source', 'Interpretation']
            add(header, heading=True)
            for case in CASES:
                case_points = [point for point in points_by_cohort[cohort] if point['case_id'] == case]
                for mode in modes:
                    for radius in RADII:
                        items = [radius_lookup.get((cohort, case, mode, radius, metric), _empty_stat()) for metric in STRESS_METRICS]
                        flags = radius_flags(case_points, mode, radius)
                        values = [case, mode, radius, cohort_label(cohort), items[0]['n_students']]
                        values += [value for item in items for value in (item['mean'], item['sample_std'], item['n'])]
                        values += [6, items[0]['missing_n'], max(item['failed_n'] for item in items),
                                   flags['radius_numerical_failures'], flags['geometry_flag_students'],
                                   flags['curve_invalid_geometry'], flags['first_link'],
                                   'paired inference modes; NOT separate models' if paired else 'directions → Student mean → sample SD']
                        add(values, notes={0: proof(cohort, case_id=case, mode=mode, radius_hr=radius,
                            direction_count=1 if radius == 0 else 8, flags=flags,
                            metric_student_values={metric:[dict(run_id=student['run_id'], value=student['value'])
                                for student in item['student_values']] for metric,item in zip(STRESS_METRICS,items)}),
                            3: cohort or 'No verified cohort supplied',
                            28: 'Curve-level invalid geometry is summed once per Student/mode across all 49 shifts. It is repeated here for visibility, NOT a radius-level count and must NOT be summed down this column.',
                            30: 'One invalid/missing direction excludes that Student from that radius statistic and remains visible in failed n; source scalars are retained. Geometry flags use measured coverage<1; numerical failures use original point n_failures. Stress ROI fixed192 differs from native RR20:-21.'})
            panel_ranges.append(dict(section_id=sections[-1]['id'], cohort_id=cohort,
                                     metrics=list(STRESS_METRICS), start_row=start, end_row=len(rows)))

    add([TITLE], notes={0: dict(owner=OWNER, campaign_id=CAMPAIGN_ID, section_id='block',
        schema=SCHEMA, source_hash=source_hash, relative_range='A1:AE_PENDING')}, heading=True)
    add(['Verified scalar evidence only; no new training/inference/metric evaluation.',
         'Prepared UTC', updated, 'Source SHA256', source_hash, 'Cohorts', len(summary['cohorts'])])
    add(['Mean, sample SD and n are separate numeric cells. Scene/direction/mode/selection aliases are not independent seeds.'])
    add()

    section('status', '1. 상태 / verified collection and delivery',
            '미수집·자료 미제공은 미실행·학습 실패를 뜻하지 않습니다. State-ledger readback does not prove metric evidence exists.')
    add(['Server', 'Students collected', 'Students expected', 'Native observations', 'Native expected',
         'Curves collected', 'Curves expected', 'Points collected', 'Points expected',
         'Students missing evidence', 'Curves missing evidence', 'Numerical failures', 'Invalid geometry',
         'Evidence errors reported', 'Evidence state', 'Source delivery state',
         'Native source readback pending', 'Stress source readback pending', 'Ledger rows readback verified',
         'Ledger rows expected', 'Last checked UTC', 'Status source', 'Notes'], heading=True)
    count_rows = []
    for server in SERVERS:
        server_native = [row for row in native if row['server'] == server]
        server_stress = [point for point in stress if point['server'] == server]
        server_status = [row for row in statuses if row['server'] == server]
        first = next(item for item in summary['server_status'] if item['server'] == server)
        ledger_delivery = Counter(row.get('upload_status', 'UNKNOWN') for row in server_status)
        evidence_errors = sum(bool(row.get('errors') or row.get('verification_status') == 'EVIDENCE_INVALID') for row in server_status)
        state = ('REMOTE_EVIDENCE_NOT_SUPPLIED' if not server_native and not server_stress
                 and (not server_status or all(row.get('verification_status') == 'REMOTE_EVIDENCE_NOT_SUPPLIED' for row in server_status))
                 else 'VERIFIED_AVAILABLE_EVIDENCE' if server_native or server_stress else 'NO_VERIFIED_EVIDENCE')
        times = [row['last_checked_at_utc'] for row in server_status if row.get('last_checked_at_utc')]
        values = [server, first['verified_students'], 8, len(server_native), 16, first['verified_curves'], 16,
                  len(server_stress), 784, first['missing_students'], first['missing_curves'],
                  first['numerical_failures'] if first['verified_curves'] else None,
                  first['invalid_geometry'] if first['verified_curves'] else None, evidence_errors, state,
                  ('SOURCE_READBACK_VERIFIED' if source_readback_verified and (server_native or server_stress)
                   else 'NO_MEASURED_SOURCE' if not server_native and not server_stress
                   else '; '.join(f'{key}:{value}' for key,value in sorted(ledger_delivery.items())) or 'UNKNOWN'),
                  0 if source_readback_verified else sum(row.get('readback') != 'READBACK_VERIFIED' for row in server_native),
                  0 if source_readback_verified else sum(row.get('readback') != 'READBACK_VERIFIED' for row in server_stress),
                  ledger_delivery.get('READBACK_VERIFIED', 0), 8, max(times) if times else None,
                  _url(STATUS_GID), 'readback counts describe transport, not experiment completion']
        add(values, notes={0: proof(None, server=server, status_records=server_status)})
        count_rows.append(values)
    totals = ['ALL', len([row for row in native if row['selection_id'] == SELECTIONS[0]]), 24,
              len(native), 48, len({(point['run_id'],point['mode']) for point in stress}), 48,
              len(stress), 2352, 24-len([row for row in native if row['selection_id'] == SELECTIONS[0]]),
              48-len({(point['run_id'],point['mode']) for point in stress}),
              sum(row[11] or 0 for row in count_rows) if stress else None,
              sum(row[12] or 0 for row in count_rows) if stress else None,
              sum(row[13] for row in count_rows), 'PARTIAL' if len(native) < 48 or len(stress) < 2352 else 'ALL_EVIDENCE_COLLECTED',
              'see original-server rows', sum(row[16] for row in count_rows), sum(row[17] for row in count_rows),
              sum(row[18] for row in count_rows), 24, updated, _url(STATUS_GID),
              'Collected ≠ all numerical/geometry checks passed; cohort-pooled n is never inferred from total rows.']
    add(totals)
    add()

    section('rb01_primary', '2. RB01 primary / EXACT_50000',
            'Case-wise Student means; maximum six independent Students per case, separated by compatible source/runtime/F1/data/map/evaluator cohort.')
    native_panel(SELECTIONS[0])
    add()

    section('rb01_paired', '3. RB01 paired / EXACT_50000 case − QFULL',
            'Pairs require identical server/repeat/seed/cohort. Improved is metric-direction aware; tie counts and exact memberships are retained in Case notes.')
    paired_panel()
    add()

    section('rb01_secondary', '4. RB01 secondary / RR_VAL_ERGAS_MIN',
            'Validation-selected observations are separate from the primary analysis. Same-SHA alias observations do not create additional Students.')
    native_panel(SELECTIONS[1])
    add()

    section('rb01_individual', '5. RB01 individual / fixed 24 Student × 2 selector slots',
            'Registry identifiers are shown even when files were not supplied; blank metric/update/SHA fields are not zeros or measured endpoints. Full identities and evidence in notes.')
    add(['Case', 'Server', 'Repeat', 'Seed', 'Selector', 'Selected step', 'Observation state',
         *[LABELS[key] for key in ALL_NATIVE_METRICS], 'Checkpoint SHA (short)', 'Run ID', 'Alias of',
         'Cohort', 'Native source', 'Result ID', 'Train(h)', 'Eval(h) run total', 'Completed UTC',
         'JQM variant', 'Review', 'Source SHA'], heading=True)
    for case in registry():
        for selector in SELECTIONS:
            record = native_by_slot.get((case['run_id'], selector))
            status = status_by_id.get(case['run_id'], {})
            number = source_row(case['repeat'], case['case_id'], selector)
            link = _url(NATIVE_GIDS[case['server']], f'A{number}:BL{number}')
            values = [case['case_id'], case['server'], case['repeat'], case['seed'], selector,
                      record['update'] if record else None,
                      'NATIVE_EVAL_COMPLETE' if record else status.get('native_status', 'FILES_NOT_PROVIDED')]
            values += [record['metrics'].get(key) if record else None for key in ALL_NATIVE_METRICS]
            values += [record['checkpoint_sha256'][:12] if record else None, case['run_id'],
                       record.get('alias_of') if record else None,
                       cohort_label(cohort_id(record['provenance'])) if record else None,
                       link, native_result_id(record) if record else None,
                       record.get('costs',{}).get('train_hours') if record else None,
                       record.get('costs',{}).get('eval_hours') if record else None,
                       record.get('completed_at_utc') if record else None,
                       record.get('jqm_variant') if record else None,
                       record.get('review') if record else '미수집/자료 미제공 — 실행 상태 판단 금지', source_hash[:12]]
            line = add(values, notes={0: proof(cohort_id(record['provenance']) if record else None,
                             registered_slot=case, selection_id=selector,
                             native_record=record, status_record=status),
                             19: record['checkpoint_sha256'] if record else 'No verified checkpoint supplied',
                             20: case['run_id'], 23: link, 30: source_hash})
            native_slots.append(dict(run_id=case['run_id'], selection_id=selector,
                                     relative_row=line, source_row=number, measured=record is not None))
    add()

    section('rb02_radius', '6. RB02 radius summary / fixed192 ROI',
            'First average registered directions within each Student, then Student mean/sample SD. Native RR20:-21 ERGAS is a different evaluation scope.')
    stress_panel()
    add()

    section('rb02_paired', '7. RB02 paired / A_ON − A_ZERO_INFERENCE_ONLY',
            'Paired correction-at-inference contrast using the identical exact50K model. A_ZERO is NOT an independently trained no-align model.')
    stress_panel(paired=True)
    add()

    section('provenance', '8. 출처·한계 / provenance and limitations',
            'Literal summary cells are regenerated automatically after verified source upsert. Full source values and deterministic observation IDs remain in the existing hidden source tabs.')
    add(['Item', 'Value', 'Source / scope'], heading=True)
    add(['Campaign', CAMPAIGN_ID, '24 planned Students / 48 native selectors / 48 curves / 2352 shift points'])
    add(['Prepared source SHA256', source_hash, 'Immutable prepared payload digest'], notes={1: source_hash})
    add(['Prepared UTC', updated, 'Not the original experiment completion time'])
    add(['Verified collected scope', ', '.join(server for server in SERVERS if any(row['server']==server for row in native)) or 'none',
         'Missing server packages indicate evidence not supplied, not failed/unrun experiments.'])
    add(['Native scope', 'RR20:-21; FR_FULL512; A_ON', 'RR and FR each 20 scenes; alias selector rows never inflate Student n.'])
    add(['Stress scope', '32:-32 / fixed192; exact50K; 49 shifts × 20 scenes', _url(STRESS_GID)])
    add(['Statistics', 'Student sample SD (ddof=1); n<2 SD blank', 'No pooled six across incompatible cohorts; server and seed effects are confounded.'])
    add(['Numerical / geometry flags', f"numerical={totals[11] if totals[11] is not None else 'unmeasured'}; geometry={totals[12] if totals[12] is not None else 'unmeasured'}",
         'Geometry totals refer to original curve observations. Radius tables explicitly separate radius numerical failures and full-curve geometry totals.'])
    add(['Delivery state', 'SOURCE_READBACK_VERIFIED' if source_readback_verified else 'SOURCE_READBACK_NOT_CONFIRMED',
         'Explicit writer overlay only after source readback; final block-view readback is recorded in the external receipt, not assumed here.'],
        notes={1: dict(delivery_overlay=delivery, immutable_source_hash=source_hash,
                       timestamp_scope='evidence prepared/check reference; actual execution time remains in receipt')})
    for server in SERVERS:
        add([f'Native source / {server}', _url(NATIVE_GIDS[server]), 'Existing source tab; no new worksheet created'])
    add(['Stress source', _url(STRESS_GID), 'All detailed shift points remain here; no new public point table'])
    add(['Status ledger', _url(STATUS_GID), 'Registry slot state / source verification / delivery are separate'])
    for cohort in cohorts:
        provenance = summary['cohorts'].get(cohort)
        add(['Cohort', cohort or 'UNCONFIRMED — no verified evidence',
             'Full numerical/runtime/F1/data/raw-map/evaluator identity is in note'], notes={1: provenance or 'Unknown cohort is not pooled'})
    sections[-1]['end_row'] = len(rows)
    rows[0]['values'][0]['note'] = compact_json(dict(owner=OWNER, campaign_id=CAMPAIGN_ID,
        section_id='block', schema=SCHEMA, source_hash=source_hash,
        relative_range=f'A1:AE{len(rows)}', section_ids=list(SECTION_IDS)))
    if tuple(section['id'] for section in sections) != SECTION_IDS:
        raise ContractError('Block section ordering differs from amendment')
    return dict(schema=SCHEMA, owner=OWNER, campaign_id=CAMPAIGN_ID, source_hash=source_hash,
                rows=rows, expected=expected, sections=sections, panels=panel_ranges,
                native_slots=native_slots, row_count=len(rows), column_count=COLUMN_COUNT,
                cohort_count=len(summary['cohorts']), new_sheet_count=0,
                statistics_source='verified canonical Student evidence; literal scalars refreshed on source upsert',
                original_metrics_recomputed=False)
