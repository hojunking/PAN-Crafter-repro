"""TA2 append-only analysis152 outbox and explicitly invoked Sheets publisher.

No import, constructor of Outbox, or record creation contacts Google. Existing
rows, headers, other tabs and workbook layout are never rewritten. Publishing
uses atomic values.append (RAW/INSERT_ROWS), not a first-empty-row update. Only
the new record's own transport-marker cells and background are then updated.

One local publisher lock per server namespace is required. Row IDs include the
server; distinct servers cannot own the same ID. Sheets has no conditional
append-by-ID primitive: an accidental duplicate same-lane remote writer is
detected on readback and reported, never silently deleted. Capacity or network
failure leaves all unverified records in the outbox; partial upload is not
campaign completion, and no overflow worksheet is automatically created.
"""
from __future__ import annotations

import csv
import fcntl
import io
import hashlib
import math
import numbers
import os
from pathlib import Path
import re
import tempfile

from .common import BUNDLE, CAMPAIGN, ROOT, atomic_json, digest, immutable_json, load_json, now

SPREADSHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'
SHEET = 'analysis'
SHEET_ID = 261006100
HEADER_ROW = 7
SCHEMA = 'AAX_ANALYSIS152_v2'
COLUMNS = tuple(load_json(BUNDLE / 'analysis_columns_v2.json'))
if len(COLUMNS) != 152 or len(set(COLUMNS)) != 152:
    raise ValueError('TA2 requires exactly the supplied 152-column schema')
INDEX = {name: i for i, name in enumerate(COLUMNS)}
RECORD_TYPES = {'PLAN_V2', 'STATUS', 'ASSET', 'UNIT', 'SUMMARY', 'RESPONSE', 'CURVE',
                'GRAD', 'PAIRED', 'SEED_AGG', 'ORACLE', 'COUNTERFACTUAL'}
MUTABLE = {'Readback_status', 'Upload_receipt_SHA'}
VOLATILE = MUTABLE | {'Created_UTC'}
NUMERIC = set('''Step N_units HQNR D_s D_lambda JQM ERGAS SCC SAM PSNR SSIM Q4_Q8 RMSE CC
Infer_ms Mem_MB Train_h Eval_h Gain Gain_y Gain_x Offset_MAE_px Offset_EPE_px Joint_shift_EPE_px
Native_c_dy Native_c_dx Mean_norm_c Std_c Residual_proxy_px Delta_HQNR Delta_ERGAS_pct
Delta_offset_MAE Delta_proxy Grad_ratio Grad_cos Estimate CI_low CI_high N_scenes N_patches
Epsilon_radius_px Lambda_epsilon Offset_every Lambda_shiftrec Lambda_scale A_peak_lr U_peak_lr
Grad_rec_norm Grad_off_raw_norm Grad_off_weighted_norm Gain_num Gain_den Attempt Gain_train64
Gain_RR256 Gain_FR512 Low_texture_fraction Valid_fraction Train_seed Probe_seed Expected_count
Completed_count Scale_factor Replica Epsilon_effective_count Student_A_retention Lambda_struct
Struct_loss Grad_struct_raw_norm Grad_struct_weighted_norm Grad_cos_rec_struct Grad_cos_eps_struct
Shift_confidence N_train_seeds Seed_std Delta_gain Delta_proxy_GT Delta_proxy_LMS Learned_time_ms
Registration_ms Pipeline_ms Params_M FLOPs_G Effective_updates Expected_repeats Completed_repeats
Phase_dx Phase_dy Phase_MAE Failure_fraction'''.split())
BOOLEAN = {'Test_aware', 'A_receives_rec'}
CONTEXT = {
    'case_id': 'Case_ID', 'dataset': 'Dataset', 'server': 'Server', 'seed': 'Train_seed',
    'replica': 'Replica', 'attempt': 'Attempt', 'run_id': 'Run_ID', 'control_id': 'Control_ID',
    'label': 'Experiment', 'group': 'Cohort', 'queue_block': 'Stage', 'completed_step': 'Step',
    'step': 'Step', 'checkpoint_sha256': 'Checkpoint_SHA', 'source_sha256': 'Source_SHA',
    'source_revision': 'Source_SHA', 'config_sha256': 'Config_SHA', 'data_sha256': 'Data_SHA',
    'initial_u_sha256': 'Initial_U_SHA', 'initial_a_sha256': 'Initial_A_SHA',
    'native_stream_sha256': 'Stream_SHA', 'evaluator_sha256': 'Evaluator_SHA',
    'aligner_policy': 'Aligner_policy', 'a_reference': 'A_reference', 'u_reference': 'U_reference',
    'struct_target': 'Struct_target', 'struct_descriptor': 'Struct_descriptor',
    'struct_branch': 'Struct_branch', 'lambda_struct': 'Lambda_struct',
    'lambda_epsilon': 'Lambda_epsilon', 'lambda_scale': 'Lambda_scale',
    'epsilon_every': 'Offset_every', 'a_receives_rec': 'A_receives_rec',
    'lr_a': 'A_peak_lr', 'lr_u': 'U_peak_lr', 'selector': 'Selection',
    'status': 'Status', 'lms_sha256': 'LMS_SHA', 'lms_operator_sha256': 'LMS_operator_SHA',
    'offline_shift_sha256': 'Offline_shift_SHA', 'raw_results_uri': 'Raw_results_URI'}
METRICS = {'hqnr': 'HQNR', 'd_s': 'D_s', 'd_lambda': 'D_lambda', 'jqm': 'JQM', 'ergas': 'ERGAS',
           'scc': 'SCC', 'sam': 'SAM', 'psnr': 'PSNR', 'ssim': 'SSIM', 'q4': 'Q4_Q8',
           'q8': 'Q4_Q8', 'rmse': 'RMSE', 'cc': 'CC'}


class PublicationConflict(RuntimeError):
    pass


class CapacityBlocked(RuntimeError):
    pass


def column_letter(index):
    value, result = index + 1, ''
    while value:
        value, rem = divmod(value - 1, 26)
        result = chr(65 + rem) + result
    return result


def scalar(value, column):
    if value is None or (isinstance(value, str) and value == ''):
        return ''
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            raise ValueError('Vector is not a scalar sheet measurement: ' + column)
    if column in BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError('Boolean column must contain bool/None: ' + column)
        return value
    if column in NUMERIC:
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError('Numeric column must contain finite number/None: ' + column)
        if isinstance(value, numbers.Integral):
            if abs(value) > 2**53:
                raise ValueError('Integer cannot be stored exactly in Sheets')
            return int(value)
        return float(value)
    if not isinstance(value, str):
        raise ValueError('Text column must contain string/None: ' + column)
    if len(value) > 49000:
        raise ValueError('Sheet cell exceeds safe text size')
    return ' '.join(value.split()) if column == 'Experiment' else value


def scientific_payload(record):
    return {key: record[key] for key in COLUMNS if key not in VOLATILE}


def validate_record(record):
    if set(record) != set(COLUMNS):
        raise ValueError('Record must have exactly all 152 named fields')
    row = {key: scalar(record[key], key) for key in COLUMNS}
    if row['Campaign_ID'] != CAMPAIGN or row['Schema_version'] != SCHEMA or row['Role'] != 'T':
        raise ValueError('Wrong campaign/schema/role; never reuse old sheet rows')
    if row['Record_type'] not in RECORD_TYPES:
        raise ValueError('Unregistered record type')
    lanes = {'s1': 'WV3', 's3': 'QB', 's5': 'GF2'}
    if lanes.get(row['Server']) != row['Dataset']:
        raise ValueError('Only assigned server/dataset lanes are writable')
    if not row['Record_ID'] or not row['Case_ID'] or not row['Status']:
        raise ValueError('Record requires identity and explicit measured/pending status')
    if row['Record_type'] == 'PLAN_V2' and any(row[k] != '' for k in METRICS.values()):
        raise ValueError('Plans may not contain invented measured quality')
    if row['Record_type'] == 'SUMMARY' and row['Selection'] == 'HQNR_MAX50':
        if row['Expected_count'] != 50 or row['Completed_count'] != 50:
            raise ValueError('HQNR selection summary requires 50/50 complete candidates')
        if row['Test_aware'] is not True or row['Selection_split'] != 'FR20':
            raise ValueError('HQNR selector is test-aware FR20, not independent validation')
    return row


def make_record(context, record_type, scope, values=None, **fields):
    """Build immutable scientific row; ``scope`` distinguishes each measurement.

    Accepts run registry/resolved config keys or already-mapped schema fields.
    Unknown explicit output fields fail, avoiding silent column-name typos.
    None means blank, never fabricated zero. Retry timestamps do not alter
    identity/scientific content; Outbox retains the first creation timestamp.
    """
    if not isinstance(scope, str) or not scope:
        raise ValueError('An explicit measurement scope is mandatory')
    record = {key: '' for key in COLUMNS}
    for key, value in context.items():
        target = key if key in INDEX else CONTEXT.get(key)
        if target:
            record[target] = value
    for key, value in {**(values or {}), **fields}.items():
        if key not in INDEX:
            raise ValueError('Unknown analysis column: ' + key)
        record[key] = value
    record.update(Campaign_ID=CAMPAIGN, Schema_version=SCHEMA, Record_type=record_type,
                  Role='T', Model_scope='TEACHER_ONLY', Created_UTC=record['Created_UTC'] or now(),
                  Readback_status='PENDING', Upload_receipt_SHA='')
    record['Attempt'] = record['Attempt'] or 1
    record['Status'] = record['Status'] or 'MEASURED'
    if record_type != 'PLAN_V2' and record['Status'] == 'PLANNED_NOT_RUN':
        record['Status'] = 'MEASURED'
    keys = ('Campaign_ID', 'Dataset', 'Server', 'Case_ID', 'Replica', 'Attempt',
            'Run_ID', 'Checkpoint_SHA', 'Record_type', 'Split_probe', 'Step', 'Selection', 'Metric_name')
    record['Record_ID'] = 'TA2:' + digest([[(key, record[key]) for key in keys], scope])
    return validate_record(record)


def summary_records(context, evaluation, selector, selection=None):
    if not evaluation.get('complete') or not evaluation.get('rr_fr_same_checkpoint'):
        raise ValueError('Summary requires complete native evaluation from one checkpoint')
    if selector not in ('HQNR_MAX50', 'EXACT_FINAL', 'PROGRESS', 'FIXED_PROGRESS'):
        raise ValueError('Unknown selection scope')
    if selector == 'HQNR_MAX50':
        if (not selection or selection.get('status') != 'HQNR_SELECTION_COMPLETE'
                or selection['checkpoint_sha256'] != evaluation['checkpoint_sha256']
                or selection.get('candidates_complete') != 50 or selection.get('candidates_expected') != 50):
            raise ValueError('HQNR summary requires matching completed ledger selection')
    values = dict(Selection=selector, Step=evaluation['completed_step'],
                  Checkpoint_SHA=evaluation['checkpoint_sha256'], Split_probe='RR20+FR20',
                  N_scenes=20, Status='MEASURED', Selection_split='FR20' if selector == 'HQNR_MAX50' else 'FIXED_PROGRESS',
                  Test_aware=selector == 'HQNR_MAX50', Expected_count=50 if selector == 'HQNR_MAX50' else None,
                  Completed_count=50 if selector == 'HQNR_MAX50' else None,
                  Eval_h=None if evaluation.get('elapsed_seconds') is None else evaluation['elapsed_seconds'] / 3600,
                  Reason='FR original PAN/native LMS full support; RR separate20; no physical displacement GT')
    for split in ('rr', 'fr'):
        if evaluation[split].get('n_scenes') != 20:
            raise ValueError('Summary requires all20 scenes per official split')
        for key, column in METRICS.items():
            if key in evaluation[split]:
                values[column] = evaluation[split][key]
    return [make_record(context, 'SUMMARY' if selector in ('HQNR_MAX50', 'EXACT_FINAL') else 'CURVE', selector, values)]


def status_record(context, status, step, error=None):
    if status == 'RUNNING' and step < 1:
        raise ValueError('RUNNING requires an actual completed optimizer update')
    return make_record(context, 'STATUS', f'status/{step}/{status}',
        Status=status, Step=step, Effective_updates=step, Reason='' if error is None else str(error),
        Evidence_kind='actual_run_status_not_plan')


def asset_records(context, manifest, smoke_receipt=None):
    """All-sample parity numbers and phase summaries, plus smoke-only evidence."""
    output = []
    if manifest.get('status') != 'PASS':
        return [make_record(context, 'ASSET', 'dataset/blocked', Status=manifest.get('status', 'BLOCKED'),
                            Reason='Native LMS lineage is not verified; no bicubic substitution')]
    for split, item in manifest['splits'].items():
        base = dict(Split_probe=split, Data_SHA=item['sha256'], LMS_SHA=item['sha256'],
                    LMS_operator_SHA=digest(manifest['identity']['operator']), N_units=item['count'],
                    Status='PASS', Evidence_kind='all_samples_native_LMS_audit',
                    Reason='Stored LMS regeneration parity; FR alignment to GT is not certified')
        for name, value in _numeric_leaves(item['lms_parity']):
            output.append(make_record(context, 'ASSET', f'asset/{split}/lms_parity/{name}',
                dict(base, Metric_name='lms_parity/' + name, Estimate=value)))
        for reference, row in item.get('phase_diagnostics', {}).get('summary', {}).items():
            shift = row.get('mean_dy_dx')
            values = dict(base, Reference_kind=reference,
                          Phase_dy=None if shift is None else shift[0],
                          Phase_dx=None if shift is None else shift[1],
                          Phase_MAE=row.get('mean_absolute_component_pixels'))
            for name, value in _numeric_leaves(row):
                output.append(make_record(context, 'ASSET', f'asset/{split}/phase/{reference}/{name}',
                    dict(values, Metric_name='phase/' + name, Estimate=value)))
        for field, stats in item.get('statistics', {}).items():
            for name, value in _numeric_leaves(stats):
                output.append(make_record(context, 'ASSET', f'asset/{split}/range/{field}/{name}',
                    dict(base, Metric_name=f'data_range/{field}/{name}', Estimate=value)))
    if smoke_receipt is not None:
        if smoke_receipt.get('smoke_only') is not True or smoke_receipt.get('formal_optimizer_updates') != 0:
            raise ValueError('UNIT smoke receipt must not masquerade as registered training')
        base = dict(Status=smoke_receipt['status'], Evidence_kind='SMOKE_ONLY_NOT_FORMAL_RESULT',
                    Reason='Disposable fresh CUDA/source smoke; not an HQNR candidate or registered seed result')
        output.append(make_record(context, 'UNIT', 'smoke/case_count',
            dict(base, Metric_name='smoke_case_count', Estimate=len(smoke_receipt.get('cases', [])))))
        for row in smoke_receipt.get('cases', []):
            for name in ('peak_cuda_bytes',):
                if name in row:
                    output.append(make_record(context, 'UNIT', f"smoke/{row['case_id']}/{name}",
                        dict(base, Case_ID=row['case_id'], Metric_name=name, Estimate=row[name])))
            for name, value in _numeric_leaves(row.get('metric_smoke')):
                if name:
                    output.append(make_record(context, 'UNIT', f"smoke/{row['case_id']}/metric/{name}",
                        dict(base, Case_ID=row['case_id'], Metric_name='smoke/' + name, Estimate=value)))
            output.append(make_record(context, 'UNIT', f"smoke/{row['case_id']}/updates",
                dict(base, Case_ID=row['case_id'], Metric_name='disposable_smoke_updates',
                     Estimate=len(row.get('logs', [])))))
    return output


def publication_capacity_estimate(run_count=468, *, observed_rows_per_selector=None, workbook_allocated_cells=None):
    """Transparent planning estimate, never an assertion of already uploaded rows.

    Standard frozen4scene probes currently render ~1160 rows/selector; offline
    controls add HOLD_NATIVE_CACHE responses. Actual pending CSVs remain the
    authoritative required count. This is deliberately a lower-bound estimate
    excluding asset/status/paired/seed-aggregate and extra offline records.
    """
    selector_rows = 1160 if observed_rows_per_selector is None else int(observed_rows_per_selector)
    per_run = 50 + 2 + 3 * (6 * 14) + 2 * selector_rows
    rows = int(run_count) * per_run
    allocated = 0 if workbook_allocated_cells is None else int(workbook_allocated_cells)
    return dict(status='REQUIRES_PUBLICATION_PARTITION_DECISION' if rows*152+allocated > 10_000_000 else 'ESTIMATE_FITS',
                registered_runs=int(run_count), assumed_rows_per_selector=selector_rows,
                rows_per_run_lower_estimate=per_run, total_rows_lower_estimate=rows,
                added_cells_lower_estimate=rows*152, workbook_allocated_cells=workbook_allocated_cells,
                workbook_cell_contract_limit=10_000_000,
                remaining_row_capacity=None if workbook_allocated_cells is None else max(0,(10_000_000-allocated)//152),
                precision='planning estimate, not measured upload receipt',
                omissions='asset/status/paired/seed aggregates and extra offline responses',
                automatic_new_tabs=False, required_numeric_records_omitted=False,
                fallback='retain immutable JSON+CSV local outbox; mark partial/pending publication')


def response_records(context, probe):
    """Summary plus numeric axis numerator/denominator/MAE rows (not raw pixels)."""
    output = []
    for index, row in enumerate(probe.get('summary', probe.get('response_summary', [])) + probe.get('size_summary', [])):
        scope = f"response/{row['split']}/{row['probe']}/{row['mode']}/{row['measurement_domain']}/{row['support_scope']}/{row['radius']}"
        interval = row.get('gain_ci95')
        fields = dict(Split_probe=f"{row['split']}/{row['probe']}/{row['mode']}/{row['support_scope']}",
                      Measurement_domain=row['measurement_domain'], Epsilon_radius_px=row['radius'],
                      Gain=row.get('gain'), Gain_y=row.get('gain_y'), Gain_x=row.get('gain_x'),
                      Gain_num=row['numerator'], Gain_den=row['denominator'],
                      Offset_MAE_px=row['component_mae'], Offset_EPE_px=row['epe'], N_units=row['n'],
                      N_scenes=row.get('n_clusters'), CI_low=None if interval is None else interval[0],
                      CI_high=None if interval is None else interval[1],
                      Reason='OOD separate; scene-cluster CI is not training-seed STD; physical GT unavailable')
        output.append(make_record(context, 'RESPONSE', scope, fields))
        for axis in ('y', 'x'):
            if 'numerator_' + axis in row:
                output.append(make_record(context, 'RESPONSE', scope + '/axis/' + axis,
                    dict(fields, Direction=axis, Gain=row.get('gain_' + axis),
                         Gain_num=row['numerator_' + axis], Gain_den=row['denominator_' + axis],
                         Offset_MAE_px=row['mae_' + axis])))
    output.extend(diagnostic_records(context, probe))
    return output


def _numeric_leaves(value, prefix=''):
    """Flatten small summaries only, not response/pixel vectors."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_leaves(child, (prefix + '/' if prefix else '') + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, prefix + '/' + str(index))
    elif value is None or (isinstance(value, numbers.Real) and not isinstance(value, bool)):
        yield prefix, value


def diagnostic_records(context, probe):
    """Map P02--P08 measured summaries/interventions to real scalar columns.

    Large offset vectors remain in immutable diagnostics JSON; summary, scene
    interventions and distribution coverage are numeric in the central sheet.
    """
    output = []
    for row in probe.get('geometry_summary', []):
        scope = f"geometry/{row['split']}/{row['target']}/{row['descriptor']}"
        fields = dict(Split_probe=row['split'], Reference_kind=row['target'],
                      Shift_estimator=row['descriptor'], N_units=row['n_total'],
                      Valid_fraction=row['n_valid']/row['n_total'] if row['n_total'] else None,
                      Residual_proxy_px=row['after_epe_mean'],
                      Delta_proxy=None if row['before_epe_mean'] is None or row['after_epe_mean'] is None
                      else row['after_epe_mean']-row['before_epe_mean'],
                      Reason='Proxy reestimated after actual warp; not physical displacement GT')
        for name in ('before_epe_mean', 'after_epe_mean', 'boundary_hit_count', 'low_texture_band_count', 'n_valid', 'invalid_fraction'):
            output.append(make_record(context, 'RESPONSE', scope + '/' + name,
                dict(fields, Metric_name=name, Estimate=row.get(name))))
    for split, row in probe.get('correction_summary', {}).items():
        base = dict(Split_probe=split, N_units=row['n'], Native_c_dy=row['mean_vector'][0],
                    Native_c_dx=row['mean_vector'][1], Mean_norm_c=row['mean_norm'])
        for name, value in _numeric_leaves(row):
            output.append(make_record(context, 'RESPONSE', f'correction/{split}/{name}',
                dict(base, Metric_name=name, Estimate=value)))
    for report in probe.get('ms_dependence', []):
        groups = {}
        for row in report['rows']:
            eps = row.get('epsilon')
            radius = math.hypot(*eps) if eps is not None else None
            key = row['mode'], row.get('support_scope', 'normal'), radius
            groups.setdefault(key, []).append(row)
        for (mode, support, radius), rows in groups.items():
            values = [r['residual_epe'] for r in rows if r.get('residual_epe') is not None]
            norms = [math.hypot(*r['delta']) for r in rows]
            fields = dict(Split_probe=f"{report['split']}/MS_DEPENDENCE/{mode}/{support}",
                          Epsilon_radius_px=radius, N_units=len(rows),
                          Offset_EPE_px=sum(values)/len(values) if values else None,
                          Low_texture_fraction=sum(bool(r.get('low_texture')) for r in rows)/len(rows),
                          Mean_norm_c=sum(norms)/len(norms),
                          Reason='Diagnostic LMS intervention only; native reconstruction unchanged')
            for name, value in (('residual_epe_mean', fields['Offset_EPE_px']),
                                ('residual_epe_min', min(values) if values else None),
                                ('residual_epe_max', max(values) if values else None),
                                ('delta_norm_mean', fields['Mean_norm_c'])):
                output.append(make_record(context, 'COUNTERFACTUAL', f"ms_dependence/{report['split']}/{mode}/{support}/{radius}/{name}",
                    dict(fields, Metric_name=name, Estimate=value)))
    for report in probe.get('counterfactual', []):
        for row in report['rows']:
            fields = dict(Split_probe=report['split'], Scene_ID=str(row['scene_index']),
                          Reference_kind=report['policy'], Native_c_dy=row['correction'][0],
                          Native_c_dx=row['correction'][1],
                          Reason='Same frozen U inference intervention; not retrained or independent seed')
            fields.update({column:row[key] for key,column in METRICS.items() if key in row})
            output.append(make_record(context, 'COUNTERFACTUAL',
                f"intervention/{report['split']}/{report['policy']}/{row['scene_index']}", fields))
    for row in probe.get('cost') or []:
        fields = dict(Split_probe=row.get('split', ''), Registration_ms=row['registration_ms'],
                      Infer_ms=row['warp_u_ms'], Pipeline_ms=row['pipeline_ms'], Params_M=row.get('params_M'),
                      Expected_count=row['repeats'], Completed_count=row['repeats'],
                      Reason=f"{row['hardware']}; {row['precision']}; batch1; uncached registration; no FLOPs estimate")
        for key in ('pipeline_std_ms', 'warmup', 'repeats', 'training_registration_precompute_seconds'):
            if key in row:
                output.append(make_record(context, 'RESPONSE', f"cost/{row.get('split','')}/{key}",
                    dict(fields, Metric_name=key, Estimate=row[key])))
    for key in ('known_shift_positive_control', 'rr_gt_oracle'):
        for index, row in enumerate(probe.get(key, [])):
            fields = dict(Split_probe=row.get('split', 'train'), Scene_ID=str(row.get('scene_index', index)),
                          Offset_EPE_px=row.get('epe'), Reference_kind=row.get('scope', key),
                          Reason='Diagnostic oracle/known-shift only; excluded from selector and seed aggregate')
            fields.update({column:row[name] for name,column in METRICS.items() if name in row})
            output.append(make_record(context, 'ORACLE', f"{key}/{row.get('scene_index',index)}", fields))
    return output


def gradient_records(context, rows):
    if isinstance(rows, dict):
        report = rows
        rows = report.get('summary', report.get('rows', []))
        context = dict(context, Probe_seed=report.get('seed'), Stream_SHA=report.get('stream_sha256', ''),
                       N_units=report.get('fixed_probe_batches'))
        if not rows:
            return [make_record(context, 'GRAD', 'gradient/not_applicable',
                Status=report['status'], Reason=report.get('reason', ''), Grad_scope='whole')]
    output = []
    for row in rows:
        group = row['group']; raw = row.get('raw_norm', {}); weighted = row.get('weighted_norm', {})
        cosine = row.get('cosine', {})
        values = dict(Grad_scope=group, Grad_rec_norm=raw.get('rec'),
                      Grad_off_raw_norm=raw.get('eps'), Grad_off_weighted_norm=weighted.get('eps'),
                      Grad_struct_raw_norm=raw.get('struct'), Grad_struct_weighted_norm=weighted.get('struct'),
                      Grad_cos=cosine.get('rec_eps'), Grad_cos_rec_struct=cosine.get('rec_struct'),
                      Grad_cos_eps_struct=cosine.get('eps_struct'),
                      Status=row.get('status', 'MEASURED'), Reason=row.get('reason', row.get('zero_cosine_policy', '')))
        output.append(make_record(context, 'GRAD', 'gradient/' + group, values))
        measurements = {'total_weighted_norm': row.get('total_weighted_norm'),
                        'actual_parameter_update_norm': row.get('actual_parameter_update_norm')}
        measurements.update({'ratio_to_rec/' + key: value for key, value in row.get('ratio_to_rec', {}).items()})
        measurements.update({'raw_norm/' + key: value for key, value in raw.items()})
        measurements.update({'weighted_norm/' + key: value for key, value in weighted.items()})
        measurements.update({'cosine_valid_counts/' + key: value for key, value in row.get('cosine_valid_counts', {}).items()})
        for key, value in measurements.items():
            output.append(make_record(context, 'GRAD', f'gradient/{group}/{key}',
                dict(values, Metric_name=key, Estimate=value,
                     Reason=values['Reason'] if value is not None else 'N/A: undefined zero-gradient ratio or no actual optimizer snapshot')))
    return output


def aggregate_records(context, aggregate):
    record_type, metric = aggregate['record_type'], aggregate['metric']
    if record_type not in ('SEED_AGG', 'PAIRED'):
        raise ValueError('Not a seed/paired aggregate')
    context = dict(context, Replica=None, Train_seed=None, Selection=context.get('Selection', context.get('selector')))
    base = dict(Status=aggregate['status'], Metric_name=metric,
                N_train_seeds=aggregate.get('n_success', aggregate.get('n_pairs')),
                Expected_repeats=3, Completed_repeats=aggregate.get('n_success', aggregate.get('n_pairs')),
                Failure_fraction=aggregate.get('failure_fraction'), Repeat_status=aggregate['status'],
                Paired_scope=aggregate.get('sampling_unit', ''), Seed_std=aggregate.get('std'),
                Reason='ddof=1 training-seed variation; incomplete repeats are not a completed3seed result')
    result = []
    for key in ('mean', 'std', 'median', 'minimum', 'maximum', 'n_attempted', 'n_failed'):
        if key in aggregate:
            result.append(make_record(context, record_type, f'aggregate/{metric}/{key}',
                dict(base, Metric_name=f'{metric}/{key}', Estimate=aggregate[key])))
    if record_type == 'SEED_AGG':
        for seed, value in aggregate['seed_values'].items():
            result.append(make_record(context, record_type, f'aggregate/{metric}/seed/{seed}',
                dict(base, Train_seed=int(seed), Estimate=value, Status=aggregate['seed_status'][seed])))
    else:
        for row in aggregate['rows']:
            result.append(make_record(context, record_type, f'aggregate/{metric}/seed/{row["seed"]}',
                dict(base, Train_seed=row['seed'], Replica=row.get('replica'), Estimate=row['difference'],
                     Control_ID=row.get('control', context.get('control_id')), Status=row['status'])))
    return result


def _pad(values):
    if len(values) > 152:
        raise PublicationConflict('Remote row extends beyond152column contract')
    return list(values) + [''] * (152 - len(values))


def _matches(actual, record):
    for key, value in zip(COLUMNS, _pad(actual)):
        if key in VOLATILE:
            continue
        expected = record[key]
        if isinstance(expected, bool):
            if type(value) is not bool or value != expected:
                return False
        elif isinstance(expected, numbers.Real):
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=2e-14, abs_tol=1e-15)):
                return False
        elif value != expected:
            return False
    return True


class AnalysisSheet:
    """Adapter accepts an existing gspread Spreadsheet; tests use a fake book."""
    def __init__(self, book, *, spreadsheet_id=SPREADSHEET_ID, max_cells=10_000_000):
        self.book = book
        self.spreadsheet_id = spreadsheet_id
        self.max_cells = int(max_cells)
        if not spreadsheet_id or book.id != spreadsheet_id:
            raise PublicationConflict('Wrong spreadsheet: no fallback/create')

    def values(self, area):
        return self.book.values_get(f"'{SHEET}'!{area}", params={'valueRenderOption': 'UNFORMATTED_VALUE'}).get('values', [])

    def preflight(self, extra_rows=1):
        metadata = self.book.fetch_sheet_metadata(params={'fields': 'spreadsheetId,sheets(properties,merges,tables)'})
        if metadata.get('spreadsheetId', self.spreadsheet_id) != self.spreadsheet_id:
            raise PublicationConflict('Spreadsheet metadata identity mismatch')
        matches = [s for s in metadata.get('sheets', []) if s['properties'].get('sheetId') == SHEET_ID]
        if len(matches) != 1 or matches[0]['properties'].get('title') != SHEET:
            raise PublicationConflict('analysis sheet ID/title mismatch')
        props = matches[0]['properties']['gridProperties']
        if props['columnCount'] != 152 or self.values('A7:EV7') != [list(COLUMNS)]:
            raise PublicationConflict('Exact152headers required; never modify/upgrade schema automatically')
        total = sum(s['properties'].get('gridProperties', {}).get('rowCount', 0)
                    * s['properties'].get('gridProperties', {}).get('columnCount', 0) for s in metadata['sheets'])
        if total + extra_rows * 152 > self.max_cells:
            error = CapacityBlocked('SHEET_CAPACITY_BLOCKED: preserve pending local rows; do not create overflow tab')
            error.safe_to_rollover = extra_rows > 0
            raise error
        if matches[0].get('tables'):
            raise PublicationConflict('Native table appeared: explicit table-aware revision required')
        return dict(rows=int(props['rowCount']), columns=152, allocated_cells=total,
                    conservatively_available_rows=max(0, (self.max_cells-total)//152))

    def _native(self, row):
        metadata = self.book.fetch_sheet_metadata(params={'includeGridData': True,
            'ranges': [f"'{SHEET}'!A{row}:EV{row}"],
            'fields': 'sheets(properties(sheetId,title),data(startRow,startColumn,rowData(values(userEnteredValue,effectiveValue,dataValidation,chipRuns))))'})
        sheets = [s for s in metadata.get('sheets', []) if s['properties'].get('sheetId') == SHEET_ID]
        if len(sheets) != 1 or sheets[0]['properties'].get('title') != SHEET:
            raise PublicationConflict('Native metadata read targeted different sheet')
        data = sheets[0].get('data', [{}])[0].get('rowData', [{}])
        cells = data[0].get('values', []) if data else []
        if any(c.get('dataValidation') or c.get('chipRuns') or c.get('userEnteredValue', {}).get('formulaValue') for c in cells):
            raise PublicationConflict('Validation/chips/formulas require reviewed native-row workflow')
        return cells

    def locate(self, record_id, capacity):
        col = column_letter(INDEX['Record_ID'])
        found = []
        # IDs and leading record type are bounded single-column scans, not a
        # multi-million-cell whole-grid download. Include legacy rows in edge.
        last = HEADER_ROW
        for start in range(HEADER_ROW + 1, capacity['rows'] + 1, 5000):
            stop = min(capacity['rows'], start + 4999)
            ids = self.values(f'{col}{start}:{col}{stop}')
            leading = self.values(f'A{start}:A{stop}')
            for i, row in enumerate(leading):
                if row and row[0] != '':
                    last = max(last, start + i)
            for i, row in enumerate(ids):
                if row and row[0] == record_id:
                    found.append(start + i)
        if len(found) > 1:
            raise PublicationConflict('Duplicate Record_ID detected; no remote deletion or overwrite')
        return (found[0] if found else None), last

    def publish(self, record):
        record = validate_record(record)
        capacity = self.preflight(extra_rows=0)
        existing, last = self.locate(record['Record_ID'], capacity)
        appended = existing is None
        if appended:
            self.preflight(extra_rows=1)
            self._native(HEADER_ROW)
            self._native(last)
            if last + 1 <= capacity['rows']:
                self._native(last + 1)
            # The server-side append resolves an occupied tail atomically. We
            # never call values.update at a racy computed empty-row address.
            anchor = min(last + 1, capacity['rows'])
            response = self.book.values_append(f"'{SHEET}'!A{anchor}:EV{anchor}",
                params={'valueInputOption': 'RAW', 'insertDataOption': 'INSERT_ROWS',
                        'includeValuesInResponse': False}, body={'values': [[record[k] for k in COLUMNS]]})
            span = response.get('updates', {}).get('updatedRange', '')
            match = re.fullmatch(r"(?:'analysis'|analysis)!A(\d+):EV(\d+)", span)
            if not match or match[1] != match[2] or int(match[1]) <= last:
                raise PublicationConflict('Unexpected atomic append range; preserve receipt and investigate')
            existing = int(match[1])
        actual = self.values(f'A{existing}:EV{existing}')
        if len(actual) != 1 or not _matches(actual[0], record):
            raise PublicationConflict('Same Record_ID has different scientific values; never overwrite')
        self._native(existing)
        duplicate, _ = self.locate(record['Record_ID'], self.preflight(extra_rows=0))
        if duplicate != existing:
            raise PublicationConflict('Record moved/duplicated during readback')
        receipt = dict(schema='TA2_ANALYSIS_UPLOAD_RECEIPT_v1', spreadsheet_id=self.spreadsheet_id,
                       sheet_id=SHEET_ID, row=existing, record_id=record['Record_ID'],
                       scientific_sha256=digest(scientific_payload(record)),
                       status='VALUES_READBACK_VERIFIED', precision='RAW numeric; rtol2e-14 atol1e-15',
                       assessment='transport verification only; not GPU/metric validation')
        receipt_sha = digest(receipt)
        updates = []
        for key, value in (('Readback_status', 'VERIFIED'), ('Upload_receipt_SHA', receipt_sha)):
            updates.append({'range': f"'{SHEET}'!{column_letter(INDEX[key])}{existing}", 'values': [[value]]})
        self.book.values_batch_update({'valueInputOption': 'RAW', 'data': updates})
        colors = {'SUMMARY': (.83, .93, .85), 'GRAD': (.90, .86, .96), 'RESPONSE': (.90, .86, .96),
                  'PAIRED': (.82, .94, .93), 'PLAN_V2': (.82, .94, .93)}
        color = ((1., .84, .84) if 'FAIL' in record['Status'] else
                 (1., .96, .77) if any(s in record['Status'] for s in ('PENDING', 'INCOMPLETE', 'BLOCKED'))
                 else colors.get(record['Record_type']))
        if color is not None:
            self.book.batch_update({'requests': [{'repeatCell': {
                'range': {'sheetId': SHEET_ID, 'startRowIndex': existing-1, 'endRowIndex': existing,
                          'startColumnIndex': 0, 'endColumnIndex': 152},
                'cell': {'userEnteredFormat': {'backgroundColor': dict(zip(('red','green','blue'), color))}},
                'fields': 'userEnteredFormat.backgroundColor'}}]})
        final = _pad(self.values(f'A{existing}:EV{existing}')[0])
        if (not _matches(final, record) or final[INDEX['Readback_status']] != 'VERIFIED'
                or final[INDEX['Upload_receipt_SHA']] != receipt_sha):
            raise PublicationConflict('Own record transport-marker readback failed')
        return dict(receipt, receipt_sha256=receipt_sha, appended=appended,
                    visual_render_verified=False, native_structure_readback_verified=True)

    def _values_many(self, areas):
        if not areas:
            return []
        if hasattr(self.book, 'values_batch_get'):
            result = self.book.values_batch_get([f"'{SHEET}'!{area}" for area in areas],
                params={'valueRenderOption': 'UNFORMATTED_VALUE'})
            blocks = result.get('valueRanges', [])
            if len(blocks) != len(areas):
                raise PublicationConflict('Batch values readback omitted a requested range')
            return [block.get('values', []) for block in blocks]
        return [self.values(area) for area in areas]

    @staticmethod
    def _row_spans(rows):
        spans = []
        for row in sorted(set(rows)):
            if spans and row == spans[-1][1] + 1:
                spans[-1][1] = row
            else:
                spans.append([row, row])
        return spans

    def _read_rows(self, rows):
        spans = self._row_spans(rows)
        blocks = self._values_many([f'A{first}:EV{last}' for first, last in spans])
        result = {}
        for (first, last), block in zip(spans, blocks):
            for row in range(first, last + 1):
                result[row] = block[row-first] if row-first < len(block) else []
        return result

    def _native_many(self, rows):
        spans = self._row_spans(rows)
        if not spans:
            return
        metadata = self.book.fetch_sheet_metadata(params={'includeGridData': True,
            'ranges': [f"'{SHEET}'!A{first}:EV{last}" for first, last in spans],
            'fields': 'sheets(properties(sheetId,title),data(startRow,startColumn,rowData(values(userEnteredValue,effectiveValue,dataValidation,chipRuns))))'})
        sheets = [s for s in metadata.get('sheets', []) if s['properties'].get('sheetId') == SHEET_ID]
        if len(sheets) != 1 or sheets[0]['properties'].get('title') != SHEET:
            raise PublicationConflict('Native batch metadata targeted different sheet')
        for block in sheets[0].get('data', []):
            for row in block.get('rowData', []):
                if any(cell.get('dataValidation') or cell.get('chipRuns') or
                       cell.get('userEnteredValue', {}).get('formulaValue') for cell in row.get('values', [])):
                    raise PublicationConflict('Validation/chips/formulas in batch require reviewed native-row workflow')

    def _locate_many(self, record_ids, capacity):
        """One bounded multi-range request, not two full scans per record."""
        col = column_letter(INDEX['Record_ID'])
        starts = list(range(HEADER_ROW + 1, capacity['rows'] + 1, 5000))
        areas = []
        for start in starts:
            stop = min(capacity['rows'], start + 4999)
            areas.extend((f'{col}{start}:{col}{stop}', f'A{start}:A{stop}'))
        blocks = self._values_many(areas)
        wanted = set(record_ids); found = {}; last = HEADER_ROW
        for j, start in enumerate(starts):
            ids, leading = blocks[2*j:2*j+2]
            for i, row in enumerate(leading):
                if row and row[0] != '':
                    last = max(last, start+i)
            for i, row in enumerate(ids):
                if row and row[0] in wanted:
                    if row[0] in found:
                        raise PublicationConflict('Duplicate Record_ID detected in publication batch')
                    found[row[0]] = start+i
        return found, last

    def publish_many(self, records):
        """Atomic multi-row append + bounded native/readback guards, max100.

        The method returns all receipts or raises. An ambiguous successful append
        is recovered by the next full-ID read; no blind reappend or overwrites.
        Existing exact rows are immutable. Capacity errors identify only IDs
        proven absent so a router cannot redirect an already appended record.
        """
        records = [validate_record(record) for record in records]
        if not records:
            return []
        if len(records) > 100:
            raise ValueError('Sheet publication batch is bounded to100 records')
        if len(records) == 1:
            return [self.publish(records[0])]
        ids = [record['Record_ID'] for record in records]
        if len(set(ids)) != len(ids):
            raise PublicationConflict('Duplicate input IDs in publication batch')
        capacity = self.preflight(extra_rows=0)
        positions, last = self._locate_many(ids, capacity)
        prior = self._read_rows(positions.values())
        for record in records:
            if record['Record_ID'] in positions and not _matches(prior[positions[record['Record_ID']]], record):
                raise PublicationConflict('Existing batch Record_ID has different scientific values; never overwrite')
        missing = [record for record in records if record['Record_ID'] not in positions]
        if missing:
            try:
                self.preflight(extra_rows=len(missing))
            except CapacityBlocked as exc:
                exc.missing_record_ids = [record['Record_ID'] for record in missing]
                raise
            self._native_many([HEADER_ROW, last] + ([last+1] if last+1 <= capacity['rows'] else []))
            anchor = min(last+1, capacity['rows'])
            response = self.book.values_append(f"'{SHEET}'!A{anchor}:EV{anchor}",
                params={'valueInputOption': 'RAW', 'insertDataOption': 'INSERT_ROWS',
                        'includeValuesInResponse': False},
                body={'values': [[record[key] for key in COLUMNS] for record in missing]})
            span = response.get('updates', {}).get('updatedRange', '')
            match = re.fullmatch(r"(?:'analysis'|analysis)!A(\d+):EV(\d+)", span)
            if not match or int(match[1]) <= last or int(match[2])-int(match[1])+1 != len(missing):
                raise PublicationConflict('Unexpected atomic batch append range; recover by exact IDs')
            positions.update({record['Record_ID']: int(match[1])+i for i, record in enumerate(missing)})
        actual = self._read_rows(positions.values())
        for record in records:
            if not _matches(actual[positions[record['Record_ID']]], record):
                raise PublicationConflict('Batch scientific values readback mismatch; no receipt issued')
        self._native_many(positions.values())
        duplicates, _ = self._locate_many(ids, self.preflight(extra_rows=0))
        if duplicates != positions:
            raise PublicationConflict('Record moved or duplicated during batch readback')
        receipts = []; markers = []; formats = []
        appended_ids = {record['Record_ID'] for record in missing}
        colors = {'SUMMARY': (.83,.93,.85), 'GRAD': (.90,.86,.96), 'RESPONSE': (.90,.86,.96),
                  'PAIRED': (.82,.94,.93), 'PLAN_V2': (.82,.94,.93)}
        for record in records:
            row = positions[record['Record_ID']]
            receipt = dict(schema='TA2_ANALYSIS_UPLOAD_RECEIPT_v1', spreadsheet_id=self.spreadsheet_id,
                sheet_id=SHEET_ID, row=row, record_id=record['Record_ID'],
                scientific_sha256=digest(scientific_payload(record)), status='VALUES_READBACK_VERIFIED',
                precision='RAW numeric; rtol2e-14 atol1e-15',
                assessment='transport verification only; not GPU/metric validation')
            receipt_sha = digest(receipt)
            for key, value in (('Readback_status','VERIFIED'), ('Upload_receipt_SHA',receipt_sha)):
                markers.append({'range': f"'{SHEET}'!{column_letter(INDEX[key])}{row}", 'values': [[value]]})
            color = ((1.,.84,.84) if 'FAIL' in record['Status'] else
                (1.,.96,.77) if any(s in record['Status'] for s in ('PENDING','INCOMPLETE','BLOCKED'))
                else colors.get(record['Record_type']))
            if color is not None:
                formats.append({'repeatCell': {'range': {'sheetId': SHEET_ID,
                    'startRowIndex':row-1,'endRowIndex':row,'startColumnIndex':0,'endColumnIndex':152},
                    'cell': {'userEnteredFormat': {'backgroundColor': dict(zip(('red','green','blue'),color))}},
                    'fields':'userEnteredFormat.backgroundColor'}})
            receipts.append(dict(receipt, receipt_sha256=receipt_sha, appended=record['Record_ID'] in appended_ids,
                visual_render_verified=False, native_structure_readback_verified=True))
        self.book.values_batch_update({'valueInputOption':'RAW','data':markers})
        if formats:
            self.book.batch_update({'requests':formats})
        final = self._read_rows(positions.values())
        for record, receipt in zip(records, receipts):
            row = _pad(final[receipt['row']])
            if (not _matches(row, record) or row[INDEX['Readback_status']] != 'VERIFIED'
                    or row[INDEX['Upload_receipt_SHA']] != receipt['receipt_sha256']):
                raise PublicationConflict('Batch transport-marker readback failed')
        return receipts


def connect(credentials=None):
    """Only explicit caller invocation opens credentials and makes network calls."""
    import gspread
    from .quota import LimitedHTTPClient
    client = gspread.service_account(filename=str(credentials or ROOT / 'gspread/account.json'),
                                    http_client=LimitedHTTPClient)
    return AnalysisSheet(client.open_by_key(SPREADSHEET_ID))


class Outbox:
    """Durable indexed producer/consumer queue, not a scan of historical rows.

    A short SQLite FULL-synchronous transaction stores an enqueue intent before
    either immutable payload file. Pending hardlinks are repairable projections
    of that index, not the only durable queue. Original JSON/CSV files and sealed
    delivery receipts are retained permanently. HTTP calls never hold the
    producer lock. Each operation opens its own SQLite connection (thread-safe
    use by the training producer and background publishing worker).
    """
    def __init__(self, directory):
        self.root = Path(directory) / 'analysis_outbox'
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = self.root / 'pending'
        self.pending.mkdir(exist_ok=True)
        self.database = self.root / 'queue_index.sqlite3'
        self._initialize()

    def _db(self):
        import sqlite3
        connection = sqlite3.connect(str(self.database), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    @staticmethod
    def _priority(record):
        kind = record['Record_type']
        if kind == 'SUMMARY':
            return 0
        if kind == 'STATUS':
            return 1
        if kind in ('SEED_AGG', 'PAIRED'):
            return 2 if record['Metric_name'].endswith('/mean') else 3
        return 4 if kind in ('ASSET', 'UNIT') else 5

    @staticmethod
    def _csv(record):
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer)
        writer.writerow(COLUMNS)
        writer.writerow([record[key] for key in COLUMNS])
        return buffer.getvalue().encode('utf-8')

    @staticmethod
    def _validate_entry(entry):
        record = validate_record(entry['record'])
        if entry['scientific_sha256'] != digest(scientific_payload(record)):
            raise PublicationConflict('Outbox scientific payload modified')
        if hashlib.sha256(Outbox._csv(record)).hexdigest() != entry['csv_sha256']:
            raise PublicationConflict('Outbox CSV identity differs from immutable record')
        return entry

    @staticmethod
    def _receipt_body(receipt):
        return {key:value for key,value in receipt.items() if key != 'payload_sha256'}

    def _validate_receipt(self, receipt, entry):
        if receipt.get('schema') == 'TA2_OUTBOX_RECEIPT_v2':
            if receipt.get('payload_sha256') != digest(self._receipt_body(receipt)):
                raise PublicationConflict('Sealed outbox receipt checksum mismatch')
            if receipt.get('csv_sha256') != entry['csv_sha256']:
                raise PublicationConflict('Outbox receipt refers to different CSV')
        else:
            # One-time import of the previous writer's already-sealed transport
            # receipts. Missing/unverifiable old receipts are never trusted.
            base = {key:value for key,value in receipt.items() if key not in (
                'receipt_sha256', 'appended', 'visual_render_verified', 'native_structure_readback_verified')}
            if receipt.get('receipt_sha256') != digest(base):
                raise PublicationConflict('Legacy receipt checksum missing or invalid')
        if (receipt.get('scientific_sha256') != entry['scientific_sha256']
                or receipt.get('record_id') != entry['record']['Record_ID']):
            raise PublicationConflict('Outbox receipt refers to different measurement')
        return receipt

    def _sealed_receipt(self, delivered, entry):
        if delivered.get('status') != 'VALUES_READBACK_VERIFIED':
            raise PublicationConflict('Publisher has not completed required destination readback')
        if (delivered.get('scientific_sha256') != entry['scientific_sha256']
                or delivered.get('record_id') != entry['record']['Record_ID']):
            raise PublicationConflict('Publisher returned receipt for different measurement')
        body = dict(schema='TA2_OUTBOX_RECEIPT_v2', record_id=entry['record']['Record_ID'],
                    scientific_sha256=entry['scientific_sha256'], csv_sha256=entry['csv_sha256'],
                    delivery=delivered)
        return dict(body, payload_sha256=digest(body))

    def _insert(self, db, name, entry, *, state='pending'):
        import json
        record = entry['record']
        result = db.execute('''INSERT OR IGNORE INTO records
            (filename, record_id, scientific_sha, csv_sha, state, priority, envelope)
            VALUES (?,?,?,?,?,?,?)''', (name, record['Record_ID'], entry['scientific_sha256'],
            entry['csv_sha256'], state, self._priority(record),
            json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)))
        if result.rowcount:
            db.execute("UPDATE counters SET value=value+1 WHERE name='total'")
            if state == 'verified':
                db.execute("UPDATE counters SET value=value+1 WHERE name='verified'")

    def _initialize(self):
        """At most one full legacy-directory pass. New intents never need one."""
        with (self.root / '.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = self._db()
            try:
                db.execute('PRAGMA journal_mode=WAL')
                db.executescript('''
                    CREATE TABLE IF NOT EXISTS records (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL, record_id TEXT UNIQUE NOT NULL,
                        scientific_sha TEXT NOT NULL, csv_sha TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('pending','verified')),
                        priority INTEGER NOT NULL, envelope TEXT,
                        failures INTEGER NOT NULL DEFAULT 0,
                        next_attempt REAL NOT NULL DEFAULT 0,
                        last_error TEXT, cleanup_needed INTEGER NOT NULL DEFAULT 0);
                    CREATE INDEX IF NOT EXISTS pending_priority
                        ON records(state,priority,sequence);
                    CREATE INDEX IF NOT EXISTS pending_due
                        ON records(state,priority,next_attempt,sequence);
                    CREATE INDEX IF NOT EXISTS receipt_cleanup
                        ON records(cleanup_needed,sequence);
                    CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY,value INTEGER NOT NULL);
                    INSERT OR IGNORE INTO counters VALUES ('total',0);
                    INSERT OR IGNORE INTO counters VALUES ('verified',0);
                    INSERT OR IGNORE INTO counters VALUES ('migration_complete',0);
                ''')
                if db.execute("SELECT value FROM counters WHERE name='migration_complete'").fetchone()[0]:
                    return
                # Streaming enumeration once; no sort/materialization of all
                # historical records and no repeat scan on each publish call.
                for path in self.root.glob('*.json'):
                    if not re.fullmatch(r'[0-9a-f]{64}\.json', path.name):
                        continue
                    entry = self._validate_entry(load_json(path))
                    if path.name != digest(entry['record']['Record_ID']) + '.json':
                        raise PublicationConflict('Legacy outbox filename/record identity differs')
                    receipt = self.root / 'receipts' / path.name
                    state = 'pending'
                    if receipt.exists():
                        self._validate_receipt(load_json(receipt), entry)
                        state = 'verified'
                    self._insert(db, path.name, entry, state=state)
                    db.commit()
                    self._materialize(path.name, entry, pending=state == 'pending')
                    db.execute('UPDATE records SET envelope=NULL WHERE filename=?', (path.name,))
                    db.commit()
                db.execute("UPDATE counters SET value=1 WHERE name='migration_complete'")
                db.commit()
            finally:
                db.close()

    def _materialize(self, name, entry, *, pending=True):
        """Idempotent recovery from JSON/CSV/pending-link partial commits."""
        self._validate_entry(entry)
        path = self.root / name
        immutable_json(path, entry)
        raw_csv = self._csv(entry['record'])
        csv_path = path.with_suffix('.csv')
        if csv_path.exists():
            if csv_path.read_bytes() != raw_csv:
                raise PublicationConflict('Existing immutable outbox CSV differs')
        else:
            fd, temporary = tempfile.mkstemp(prefix='.csv-', dir=self.root)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw_csv); stream.flush(); os.fsync(stream.fileno())
                os.link(temporary, csv_path)
            finally:
                os.unlink(temporary)
        marker = self.pending / name
        if pending:
            if marker.exists():
                if not os.path.samefile(marker, path):
                    raise PublicationConflict('Pending marker points to a foreign payload')
            else:
                os.link(path, marker)
        directory_fd = os.open(str(self.root), os.O_RDONLY | os.O_DIRECTORY)
        pending_fd = os.open(str(self.pending), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd); os.fsync(pending_fd)
        finally:
            os.close(directory_fd); os.close(pending_fd)

    def _load_indexed(self, row):
        import json
        entry = json.loads(row['envelope']) if row['envelope'] is not None else load_json(self.root / row['filename'])
        self._validate_entry(entry)
        if (entry['scientific_sha256'] != row['scientific_sha'] or entry['csv_sha256'] != row['csv_sha']
                or entry['record']['Record_ID'] != row['record_id']):
            raise PublicationConflict('Indexed identity differs from immutable payload')
        return entry

    def _legacy_csv_record(self, path, incoming):
        """Recover old CSV-before-JSON crashes without replacing Created_UTC."""
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))
        if len(rows) != 2 or tuple(rows[0]) != COLUMNS or len(rows[1]) != 152:
            raise PublicationConflict('Orphan legacy CSV has invalid schema')
        values = dict(zip(COLUMNS, rows[1]))
        for key, value in list(values.items()):
            if value == '':
                continue
            if key in BOOLEAN:
                if value not in ('True', 'False'):
                    raise PublicationConflict('Orphan legacy CSV boolean is invalid')
                values[key] = value == 'True'
            elif key in NUMERIC:
                values[key] = int(value) if re.fullmatch(r'-?\d+', value) else float(value)
        values = validate_record(values)
        if scientific_payload(values) != scientific_payload(incoming):
            raise PublicationConflict('Orphan legacy CSV scientific values differ from producer retry')
        return values

    def enqueue(self, record):
        record = validate_record(record)
        path = self.root / (digest(record['Record_ID']) + '.json')
        with (self.root / '.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = self._db()
            try:
                row = db.execute('SELECT * FROM records WHERE filename=?', (path.name,)).fetchone()
                if row is None:
                    if path.exists():
                        entry = self._validate_entry(load_json(path))
                        if scientific_payload(entry['record']) != scientific_payload(record):
                            raise PublicationConflict('Local Record_ID changed scientific values')
                    else:
                        if path.with_suffix('.csv').exists():
                            record = self._legacy_csv_record(path.with_suffix('.csv'), record)
                        entry = dict(record=record, scientific_sha256=digest(scientific_payload(record)),
                                     csv_sha256=hashlib.sha256(self._csv(record)).hexdigest())
                    self._insert(db, path.name, entry)
                    db.commit()  # Durable enqueue intent BEFORE any payload file.
                    row = db.execute('SELECT * FROM records WHERE filename=?', (path.name,)).fetchone()
                entry = self._load_indexed(row)
                if scientific_payload(entry['record']) != scientific_payload(record):
                    raise PublicationConflict('Local Record_ID changed scientific values')
                self._materialize(path.name, entry, pending=row['state'] == 'pending')
                if row['state'] == 'verified':
                    self._validate_receipt(load_json(self.root / 'receipts' / path.name), entry)
                db.execute('UPDATE records SET envelope=NULL WHERE filename=?', (path.name,))
                db.commit()  # Canonical files are now durable; free temporary DB copy.
            finally:
                db.close()
        return path

    def enqueue_many(self, records):
        return [self.enqueue(record) for record in records]

    def _cleanup_verified(self, max_records):
        db = self._db()
        try:
            rows = db.execute('SELECT * FROM records WHERE cleanup_needed=1 ORDER BY sequence LIMIT ?',
                              (max_records,)).fetchall()
            for row in rows:
                entry = self._load_indexed(row)
                self._validate_receipt(load_json(self.root / 'receipts' / row['filename']), entry)
                marker = self.pending / row['filename']
                if marker.exists():
                    if not os.path.samefile(marker, self.root / row['filename']):
                        raise PublicationConflict('Verified pending marker points to foreign payload')
                    marker.unlink()  # Remove only queue hardlink; original JSON/CSV remain.
                db.execute('UPDATE records SET cleanup_needed=0 WHERE filename=?', (row['filename'],))
            db.commit()
        finally:
            db.close()

    def _commit_verified(self, row, entry, receipt):
        self._validate_receipt(receipt, entry)
        receipt_path = self.root / 'receipts' / row['filename']
        immutable_json(receipt_path, receipt)
        receipt_fd = os.open(str(receipt_path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(receipt_fd)
        finally:
            os.close(receipt_fd)
        # A crash before this transaction is recovered by reading the sealed
        # receipt while still pending, without calling Sheets a second time.
        db = self._db()
        try:
            changed = db.execute("UPDATE records SET state='verified', cleanup_needed=1, envelope=NULL WHERE filename=? AND state='pending'",
                                 (row['filename'],)).rowcount
            if changed:
                db.execute("UPDATE counters SET value=value+1 WHERE name='verified'")
            db.commit()
        finally:
            db.close()

    def counts(self):
        db = self._db()
        try:
            values = dict(db.execute("SELECT name,value FROM counters WHERE name IN ('total','verified')").fetchall())
            return dict(total=values['total'], verified=values['verified'], pending=values['total']-values['verified'])
        finally:
            db.close()

    def publish(self, adapter=None, max_records=100):
        import time
        if not isinstance(max_records, int) or max_records < 1:
            raise ValueError('Positive bounded publication count required')
        errors, attempted, remote_batches = [], 0, 0
        with (self.root / '.publish.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._cleanup_verified(max_records)
            db = self._db()
            try:
                rows = []
                checked_time = time.time()
                # Six indexed due-time range reads avoid scanning an arbitrarily
                # large deferred queue when every destination is in backoff.
                for priority in range(6):
                    rows.extend(db.execute("SELECT * FROM records WHERE state='pending' AND priority=? AND next_attempt<=? ORDER BY next_attempt,sequence LIMIT ?",
                        (priority, checked_time, max_records-len(rows))).fetchall())
                    if len(rows) >= max_records:
                        break
            finally:
                db.close()
            def failed(group, exc):
                # Backoff only still-pending rows. A local disk failure after
                # remote batch success may already have durably committed some.
                db = self._db()
                try:
                    for row in group:
                        errors.append(dict(record_path=str(self.root / row['filename']),
                                           error_type=type(exc).__name__, reason=str(exc)))
                        delay = min(3600., 10. * 2 ** min(int(row['failures']), 8))
                        db.execute("UPDATE records SET failures=failures+1,next_attempt=?,last_error=? WHERE filename=? AND state='pending'",
                                   (time.time()+delay, str(exc), row['filename']))
                    db.commit()
                finally:
                    db.close()

            prepared = []
            for row in rows:
                path = self.root / row['filename']
                receipt_path = self.root / 'receipts' / row['filename']
                attempted += 1
                try:
                    entry = self._load_indexed(row)
                    # Only this pending record is checked/repaired, not every
                    # historical verified JSON, CSV and receipt on each batch.
                    with (self.root / '.lock').open('a+') as producer_lock:
                        fcntl.flock(producer_lock, fcntl.LOCK_EX)
                        self._materialize(row['filename'], entry)
                    if receipt_path.exists():
                        receipt = self._validate_receipt(load_json(receipt_path), entry)
                        self._commit_verified(row, entry, receipt)
                    else:
                        prepared.append((row, entry))
                except Exception as exc:
                    failed([row], exc)
            for priority in range(6):
                same_priority = [(row, entry) for row, entry in prepared if row['priority'] == priority]
                for start in range(0, len(same_priority), 100):
                    group = same_priority[start:start+100]
                    if not group:
                        continue
                    try:
                        if adapter is None:
                            adapter = connect()
                    except Exception as exc:
                        failed([row for row, _ in group], exc)
                        continue
                    if callable(getattr(adapter, 'publish_many', None)):
                        try:
                            remote_batches += 1
                            delivered = adapter.publish_many([entry['record'] for _, entry in group])
                            if not isinstance(delivered, (list, tuple)) or len(delivered) != len(group):
                                raise PublicationConflict('Batch publisher omitted or added receipts')
                            # Validate the ENTIRE returned batch before marking
                            # any record verified. A later remote group failure
                            # must raise and preserve the whole local batch.
                            receipts = [self._sealed_receipt(receipt, entry)
                                        for (_, entry), receipt in zip(group, delivered)]
                            for (row, entry), receipt in zip(group, receipts):
                                self._commit_verified(row, entry, receipt)
                        except Exception as exc:
                            failed([row for row, _ in group], exc)
                    else:
                        # Compatibility for explicit single-record adapters.
                        for row, entry in group:
                            try:
                                remote_batches += 1
                                receipt = self._sealed_receipt(adapter.publish(entry['record']), entry)
                                self._commit_verified(row, entry, receipt)
                            except Exception as exc:
                                failed([row], exc)
            self._cleanup_verified(max_records)
            counts = self.counts()
            state = dict(status='UPLOAD_VERIFIED' if counts['pending'] == 0 and not errors else 'PARTIAL_OR_PENDING_PUBLICATION',
                         verified_records=counts['verified'], pending_records=counts['pending'], errors=errors,
                         total_records=counts['total'], attempted_this_batch=attempted,
                         remote_batches_this_call=remote_batches,
                         checked_utc=now(), experiment_complete=False,
                         index='durable_sqlite_priority_queue; cached_verified_count; bounded_pending_read',
                         reason='Upload status is separate from registered experiment completion; cached receipts are not a fresh remote re-audit')
            atomic_json(self.root / 'publication_status.json', state)
            return state
