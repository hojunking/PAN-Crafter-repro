"""Student-level B01 statistics, independent of trainer and Google clients."""
from __future__ import annotations

from collections import defaultdict
import statistics

from .rb_b01_contract import (
    CAMPAIGN_ID, CASES, MODES, NATIVE_METRICS, RADII, SELECTIONS, SERVERS,
    STRESS_METRICS, ContractError, assert_no_conflicts, cohort_id, finite_number,
    native_result_id, registry, stress_record_id, validate_case,
)

LOWER = frozenset(('ergas', 'sam', 'rmse', 'd_lambda', 'd_s', 'edge_error_dn'))


def stats(values):
    """None is missing, never zero. Finite values only; Student sample SD."""
    values = list(values)
    for value in values:
        finite_number(value, 'summary observation')
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def _cohort(record):
    actual = cohort_id(record['provenance'])
    if record.get('cohort_id', actual) != actual:
        raise ContractError('Declared cohort differs from evidence identity')
    return actual


def _block(record):
    return record['server'], record['repeat'], record['seed']


def _panel(values, *, n_students, flagged_n=0, expected_n=6):
    good = [value for value in values if value is not None]
    measured = stats(good)
    return dict(measured, n_students=n_students, expected_n=expected_n,
                missing_n=expected_n - n_students, failed_n=len(values) - len(good),
                flagged_n=flagged_n,
                full_six=expected_n == 6 and len(good) == 6 and flagged_n == 0)


def _point_value(point, metric):
    value = finite_number(point.get(metric), metric, optional=True)
    # Keep originals in the source sheet, but do not advertise invalid geometry
    # as clean Student replication. Curve-level failure totals are NOT assigned
    # as per-point failures; measured point coverage determines this gate.
    coverage = finite_number(point.get('coverage_all_pan_paths'), 'point coverage', optional=True)
    if point['n_failures'] or coverage is None or coverage < 1.:
        return None
    return value


def summarize(native_records, stress_records, statuses=()):
    native_records = assert_no_conflicts(native_records)
    stress_records = assert_no_conflicts(stress_records)
    natives = defaultdict(list)
    stresses = defaultdict(list)
    cohort_provenance = {}
    for record in (*native_records, *stress_records):
        validate_case(record)
        cohort = _cohort(record)
        provenance = record['provenance']
        cohort_provenance[cohort] = {key: provenance[key] for key in (
            'numerical_source_sha256', 'evaluator_sha256', 'binding_common_sha256',
            'data_content_identity_sha256', 'teacher_sha256', 'runtime_policy_sha256',
            'raw_map_identity_sha256')}
        (stresses if 'shift_id' in record else natives)[cohort].append(record)
    native_panels, paired_panels, point_panels, radius_panels = [], [], [], []
    for cohort in sorted(cohort_provenance):
        rows = natives[cohort]
        for selector in SELECTIONS:
            selected = [row for row in rows if row['selection_id'] == selector]
            baseline = {_block(row): row for row in selected if row['case_id'] == 'QFULL'}
            for case in CASES:
                group = [row for row in selected if row['case_id'] == case]
                for metric in NATIVE_METRICS:
                    values = [finite_number(row['metrics'].get(metric), metric, optional=True) for row in group]
                    native_panels.append(dict(panel='native', cohort_id=cohort, selection_id=selector,
                        case_id=case, metric=metric, **_panel(values, n_students=len(group)),
                        member_ids=[native_result_id(row) for row, value in zip(group, values) if value is not None]))
                    if case == 'QFULL':
                        continue
                    pairs = [(row, baseline[_block(row)]) for row in group if _block(row) in baseline]
                    differences = []
                    member_pairs = []
                    for left, right in pairs:
                        lv, rv = left['metrics'].get(metric), right['metrics'].get(metric)
                        difference = None if lv is None or rv is None else finite_number(lv, metric) - finite_number(rv, metric)
                        differences.append(difference)
                        if difference is not None:
                            member_pairs.append(dict(left=native_result_id(left), right=native_result_id(right)))
                    good = [value for value in differences if value is not None]
                    paired_panels.append(dict(panel='paired', cohort_id=cohort, selection_id=selector,
                        case_id=case, reference='QFULL', metric=metric,
                        **_panel(differences, n_students=len(pairs)), member_pairs=member_pairs,
                        improved=sum(value < 0 if metric in LOWER else value > 0 for value in good),
                        ties=sum(value == 0 for value in good), difference='case minus QFULL; matched server/repeat/seed'))

        curve_map = defaultdict(list)
        for point in stresses[cohort]:
            curve_map[(point['run_id'], point['mode'])].append(point)
        for (run_id, mode), points in curve_map.items():
            if len(points) != 49 or {point['shift_id'] for point in points} != {f'D{i:03d}' for i in range(49)}:
                raise ContractError('Summary refuses incomplete/duplicated 49-point curve')
            identities = {(point['checkpoint_sha256'], point['grid_sha256'], point['payload_sha256']) for point in points}
            if len(identities) != 1:
                raise ContractError('Curve identity changed within shift grid')
            exact = [row for row in rows if row['run_id'] == run_id and row['selection_id'] == SELECTIONS[0]]
            if len(exact) != 1 or exact[0]['checkpoint_sha256'] != points[0]['checkpoint_sha256']:
                raise ContractError('Stress summary lacks matching cohort/native exact SHA')

        for case in CASES:
            for mode in (*MODES, 'A_ON_MINUS_A_ZERO'):
                students = []
                for reg in registry():
                    if reg['case_id'] != case:
                        continue
                    run_id = reg['run_id']
                    if mode == 'A_ON_MINUS_A_ZERO':
                        on, zero = curve_map.get((run_id, MODES[0])), curve_map.get((run_id, MODES[1]))
                        if not on or not zero:
                            continue
                        if on[0]['grid_sha256'] != zero[0]['grid_sha256']:
                            raise ContractError('Paired mode grid identities differ')
                        students.append((run_id, {p['shift_id']: p for p in on}, {p['shift_id']: p for p in zero}))
                    else:
                        points = curve_map.get((run_id, mode))
                        if points:
                            students.append((run_id, {p['shift_id']: p for p in points}, None))
                if not students:
                    continue
                for metric in STRESS_METRICS:
                    per_point = {}
                    for i in range(49):
                        shift_id = f'D{i:03d}'
                        observed = []
                        for run_id, left, right in students:
                            lv = _point_value(left[shift_id], metric)
                            rv = None if right is None else _point_value(right[shift_id], metric)
                            value = lv if right is None else (None if lv is None or rv is None else lv-rv)
                            membership = dict(member_ids=[stress_record_id(left[shift_id])]) if right is None else dict(
                                member_pairs=[dict(left=stress_record_id(left[shift_id]), right=stress_record_id(right[shift_id]))])
                            observed.append(dict(run_id=run_id, value=value, **membership))
                        per_point[shift_id] = observed
                        radius = students[0][1][shift_id]['radius_hr']
                        point_panels.append(dict(panel='stress_point', cohort_id=cohort, case_id=case,
                            mode=mode, shift_id=shift_id, radius_hr=radius, metric=metric,
                            **_panel([item['value'] for item in observed], n_students=len(students)),
                            student_values=observed))
                    for radius in RADII:
                        shift_ids = [key for key, point in students[0][1].items() if point['radius_hr'] == radius]
                        if len(shift_ids) != (1 if radius == 0 else 8):
                            raise ContractError('Radius summary lacks registered direction count')
                        observed = []
                        for index, (run_id, _, _) in enumerate(students):
                            directions = [per_point[key][index] for key in shift_ids]
                            values = [item['value'] for item in directions]
                            value = None if any(x is None for x in values) else statistics.mean(values)
                            membership_key = 'member_ids' if mode in MODES else 'member_pairs'
                            observed.append(dict(run_id=run_id, value=value, **{membership_key: [member
                                for item in directions for member in item[membership_key]]}))
                        radius_panels.append(dict(panel='stress_radius', cohort_id=cohort, case_id=case,
                            mode=mode, radius_hr=radius, metric=metric,
                            **_panel([item['value'] for item in observed], n_students=len(students)),
                            student_values=observed, direction_count=len(shift_ids),
                            interpretation='direction mean within Student, then Student sample SD; invalid points produce explicit failed_n'))

    primary = [record for record in native_records if record['selection_id'] == SELECTIONS[0]]
    curves = {(row['run_id'], row['mode']): row for row in stress_records}
    status_by_id = {}
    for status in statuses:
        validate_case(status)
        if status['run_id'] in status_by_id:
            raise ContractError('Duplicated status ledger slot')
        status_by_id[status['run_id']] = status
    server_status = []
    for server in SERVERS:
        server_students = [row for row in primary if row['server'] == server]
        server_curves = [row for row in curves.values() if row['server'] == server]
        server_status.append(dict(server=server, registered_students=8, expected_native_observations=16,
            verified_students=len(server_students), verified_native_observations=sum(row['server'] == server for row in native_records),
            expected_curves=16, verified_curves=len(server_curves),
            missing_students=8-len(server_students), missing_curves=16-len(server_curves),
            numerical_failures=sum(row['curve_numerical_failures'] for row in server_curves),
            invalid_geometry=sum(row['curve_invalid_geometry'] for row in server_curves)))
    return dict(schema='RB_B01_STUDENT_SUMMARY_v1', campaign_id=CAMPAIGN_ID,
                statistical_unit='registered Student; not selector alias, scene, direction, or inference mode',
                variance='sample SD ddof=1; n=1 is null', server_effect_identifiable=False,
                server_note='server and seed cohorts are confounded; no server-effect claim',
                counts=dict(registered_students=24, native_observations=len(native_records),
                            verified_students=len(primary), curves=len(curves), stress_points=len(stress_records),
                            cohort_count=len(cohort_provenance)),
                cohorts=cohort_provenance, server_status=server_status, native=native_panels,
                paired=paired_panels, stress_points=point_panels, stress_radius=radius_panels)
