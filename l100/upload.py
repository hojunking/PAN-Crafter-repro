"""Verified complete-grid LOCAL-T rows; no legacy Exact50K column overload.

Selection and persistence retain full precision. Four decimals are display only.
Live access requires explicit activation and writes only the local server tab.
"""
import math
from pathlib import Path

import yaml

from fh12.upload import a1, label_map, legacy_constants, same_cell
from qg40.sheet_helpers import fetch_controls, controlled_reason
from reporting_extra.sensor_sheet import (metadata_values, augment_notes,
                                          display_formats, merge_notes)
from reporting_extra.sensor_layout import row_formats
from reporting_extra.sensor_backfill import write_locks
from l100.common import (ROOT, atomic_json, object_sha, read_json, source_identity,
                          utcnow, apply_runtime_policy)
from l100.plan import CAMPAIGN_ID, SHEET_TABS, case_from_config, sensor_spec
from l100.postrun import select_records, selections, validate_grid, verify_fullstate

RR_LABELS = {'ERGAS↓': 'ergas', 'SAM↓': 'sam', 'PSNR↑': 'psnr', 'SSIM↑': 'ssim',
             'SCC↑': 'scc', 'Q4↑': 'q4', 'RMSE↓': 'rmse', 'CC↑': 'cc'}
FR_LABELS = {'HQNR(raw)↑': 'hqnr', 'D_lambda↓': 'd_lambda', 'D_s↓': 'd_s', 'JQM↑': 'jqm'}
PREFIXES = ('RAW_MAX', 'TARGET', 'EXACT_FINAL', 'RR_VAL_SELECTED', 'E_MIN_DIAG', 'MID50_OF100')


def selection_values(prefix, report):
    if prefix not in PREFIXES or prefix != report['selection_id']:
        raise ValueError('Only registered L100 selector prefixes may be written')
    empty = prefix == 'TARGET' and report.get('target_status') == 'no_eligible'
    values = {prefix + ' step': '' if empty else report['step'],
              prefix + ' checkpoint SHA256': '' if empty else report['checkpoint_sha256'],
              prefix + ' official': report['official_complete'],
              prefix + ' horizon updates': report['horizon_updates'],
              prefix + ' test-aware': report['test_aware'], prefix + ' primary': report['primary']}
    for domain, labels in (('rr', RR_LABELS), ('fr', FR_LABELS)):
        for label, key in labels.items():
            values[prefix + ' ' + label] = '' if empty else report[domain].get(key, '')
    for key in ('variant', 'status', 'reason'):
        values[prefix + ' JQM ' + key] = '' if empty else report['fr'].get('jqm_' + key, '')
    for field in ('signed_mean', 'abs_mean', 'positive_fraction', 'reconstruction_max_abs_error'):
        values[prefix + ' signed Ds ' + field] = '' if empty else report['fr']['signed_ds'][field]
    values[prefix + ' signed Ds SHA256'] = '' if empty else object_sha(report['fr']['signed_ds'])
    values[prefix + ' joint pass'] = '' if empty else report['fr']['hqnr'] > .964 and report['rr']['ergas'] < .552
    values[prefix + ' strong joint pass'] = '' if empty else report['fr']['hqnr'] > .964 and report['rr']['ergas'] < .522
    return {key: '' if value is None else value for key, value in values.items()}


def metric_formats(labels, row):
    metrics = set(RR_LABELS) | set(FR_LABELS) | {'HQNR↑', 'RR_VAL_SELECTED val ERGAS',
        'signed Ds signed_mean', 'signed Ds abs_mean', 'signed Ds positive_fraction',
        'signed Ds reconstruction_max_abs_error'}
    return display_formats(labels, row, metrics, PREFIXES)


def row_values(run, root=ROOT):
    """Prove selectors, exact endpoint and local calibrated reference before I/O."""
    root, wd = Path(root), Path(root) / 'work_dir' / run
    apply_runtime_policy(root)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    case = case_from_config(cfg)
    if case.run_id != run or case.server_id not in SHEET_TABS or case.teacher_owner != case.server_id:
        raise ValueError('LOCAL-T run/server/reference identity differs')
    status = read_json(wd / 'official/postrun_status.json')
    grid = read_json(wd / 'official/raw_grid.json')
    training = read_json(wd / 'meta/training_status.json')
    if (status.get('official_complete') is not True or status.get('actual_updates') != case.updates
            or status.get('horizon_updates') != case.updates or status.get('n_evaluated') != 50
            or training.get('training_complete') is not True or training.get('actual_updates') != case.updates
            or grid.get('complete') is not True or status.get('config_sha256') != object_sha(cfg)
            or status.get('source_identity') != grid.get('source_identity')
            or grid.get('source_identity') != source_identity(root)
            or status.get('data_sha256') != grid.get('data_sha256')):
        raise ValueError('Upload requires completed horizon and all 50 official candidates')
    data = validate_grid(run, cfg, grid, root)
    verify_fullstate(run, cfg, grid, root)
    selected, reports = select_records(grid['records'], case.updates), {}
    for key, label in selections(case.updates):
        filename = 'target_selection' if key == 'target' else key
        doc = read_json(wd / 'official' / (filename + '.json'))
        context = dict(selection_id=label, official_complete=True, normal_same_step_A_U=True,
            run_id=run, campaign_id=CAMPAIGN_ID, sensor='GF2', server_id=case.server_id,
            role=case.role, profile=case.profile, horizon_updates=case.updates,
            teacher_reference_id=case.reference_id, teacher_owner=case.server_id,
            teacher_seed=case.teacher_seed, teacher_updates=case.teacher_updates,
            pair_baseline_run_id=case.pair_baseline_run_id, n_evaluated=50,
            test_aware=key in ('raw_max', 'target', 'e_min_diag'),
            primary=key in ('exact_final', 'rr_val_selected'), independent_test=False)
        context.update({k: grid[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')})
        if any(k not in doc or doc[k] != v for k, v in context.items()):
            raise ValueError(f'Invalid {label} report context or horizon')
        record = selected[key]
        if record is None:
            if (doc.get('selection', 'missing') is not None or doc.get('target_status') != 'no_eligible'
                    or doc.get('n_eligible') != 0 or doc.get('step') is not None):
                raise ValueError('No-eligible TARGET contains fabricated metrics')
        elif (doc.get('step') != record['update'] or doc.get('checkpoint_sha256') != record['checkpoint_identity']['model_sha256']
                or any(doc.get(k) != record[k] for k in ('rr', 'fr', 'val_ergas', 'checkpoint_identity'))
                or doc.get('eval_mode') != 'A_ON' or doc.get('precision') != 'fp32'
                or doc.get('fresh_horizon_endpoint') is not (key == 'exact_final')
                or doc.get('is_fresh50k') is not (key == 'exact_final' and case.updates == 50000)):
            raise ValueError('Selected metrics/checkpoint/fresh-horizon meaning differs from raw grid')
        reports[key] = doc
    spec, target = sensor_spec(), reports['target']
    target_fields = dict(n_eligible=selected['n_eligible'], joint_pass=selected['joint_pass'],
        strong_joint_pass=selected['strong_joint_pass'], threshold_comparison='>',
        hqnr_threshold=spec.hqnr_threshold, ergas_goal=spec.ergas_goal,
        ergas_strong_goal=spec.ergas_strong_goal, target_status=selected['target_status'],
        selector_order=['ergas', '-scc', '-psnr', 'step'])
    if any(target.get(k) != v for k, v in target_fields.items()):
        raise ValueError('TARGET eligibility differs from the strict full-precision selector')
    cost = read_json(wd / 'official/profile.json')
    if (cost.get('config_sha256') != object_sha(cfg) or cost.get('source_identity') != grid['source_identity']
            or cost.get('num_bands') != 4 or cost.get('sensor') != 'GF2'
            or any(not math.isfinite(float(cost[k])) for k in ('params_m', 'flops_g', 'infer_ms'))):
        raise ValueError('Measured C4 profile identity or finite measurements missing')
    from l100.references import reference_path, validate_reference
    field = cfg['l100']
    path = root / field['reference_manifest'] if field.get('reference_manifest') else reference_path(case.reference_id, case.server_id, root)
    ref, _q, _cfg, _paths = validate_reference(case.reference_id, case.server_id, root,
                                             manifest_path=path, dataset_manifest=data)
    expected_ref = dict(owner_server=case.server_id, producer_server=case.server_id,
                        teacher_seed=case.teacher_seed, teacher_update=case.teacher_updates)
    if any(ref.get(k) != v for k, v in expected_ref.items()):
        raise ValueError('Reference is not the registered local Teacher endpoint')
    if case.role == 'T':
        if ref['teacher_checkpoint_sha256'] != reports['exact_final']['checkpoint_sha256']:
            raise ValueError('Teacher calibration does not reference this exact endpoint')
    elif any(r['checkpoint_identity'].get('reference_sha256') != object_sha(ref) for r in grid['records']):
        raise ValueError('Student checkpoints and local Teacher calibration identity differ')
    main = reports['exact_final']
    values = {'Run': run, '캠페인': 'L100', 'L100 sensor': 'GF2', 'L100 campaign': CAMPAIGN_ID,
        'L100 run id': run, 'L100 case id': case.case_id, 'L100 server': case.server_id,
        'L100 role': case.role, 'L100 profile': case.profile, 'L100 tier': case.tier,
        'L100 seed': case.seed, 'L100 horizon updates': case.updates, 'L100 actual updates': case.updates,
        'L100 Teacher seed': case.teacher_seed, 'L100 Student seed': case.student_seed,
        'L100 Teacher owner': case.teacher_owner, 'L100 Teacher updates': case.teacher_updates,
        'L100 reference id': case.reference_id, 'L100 Teacher SHA256': ref['teacher_checkpoint_sha256'],
        'L100 reference SHA256': object_sha(ref), 'L100 calibration ID': ref['calibration_id'],
        'L100 tau_R': ref['tau_R'], 'L100 q_ref': ref['q_ref'], 'L100 q cache SHA256': ref['q_cache_sha256'],
        'L100 paired local baseline': case.pair_baseline_run_id, 'L100 alpha': case.alpha,
        'L100 beta': case.beta, 'L100 lambda edge': case.lambda_edge,
        'L100 C': 4, 'L100 maxDN': 1023, 'L100 band order': ','.join(data['band_order']),
        'L100 MTF': data['mtf_sensor'], 'L100 LP phase': data['recipe']['phase_id'],
        'L100 data SHA256': grid['data_sha256'], 'L100 source SHA256': grid['source_identity']['content_sha256'],
        'L100 numerical revision': grid['source_identity'].get('numeric_method_revision', field['method_revision']),
        'L100 n evaluated': 50, 'L100 status': 'OFFICIAL_EVAL_COMPLETE',
        'L100 upload status': 'READBACK_PENDING', 'L100 primary selections': 'EXACT_FINAL and RR_VAL_SELECTED',
        'L100 selected step': main['step'], 'L100 A/U checkpoint SHA256': main['checkpoint_sha256'],
        'L100 cost scope': cost['scope'], 'L100 FLOPs convention': cost['flops_convention'],
        'Params(M)': cost['params_m'], 'FLOPs(G)': cost['flops_g'], 'Infer(ms)': cost['infer_ms'],
        'Mem(MB)': cost.get('mem_mb'), 'L100 TARGET status': selected['target_status'],
        'L100 TARGET n eligible': selected['n_eligible'], 'L100 TARGET joint pass': selected['joint_pass'],
        'L100 TARGET strong joint pass': selected['strong_joint_pass'],
        'RR_VAL_SELECTED val ERGAS': reports['rr_val_selected']['val_ergas'],
        'Notes': 'Main=EXACT_FINAL at declared horizon; co-primary=RR_VAL_SELECTED. '
            'MID50_OF100 is a 100K-schedule midpoint, never a fresh50K endpoint. '
            'RAW_MAX/TARGET/E_MIN_DIAG are test-aware. Same-step A/U; raw-original native PAN '
            'mean-per-scene HQNR; signed Ds uses actual Qhigh/Qlow. JQM=SRF-substitute.'}
    presentation_training = dict(training)
    if training.get('training_completed_at_utc'):
        presentation_training['updated_at_utc'] = training['training_completed_at_utc']
    elif training.get('recovered_from_exact_final') or training.get('recovered_from_exact_endpoint'):
        # Recovery is not a training-completion event; never display its clock.
        presentation_training['updated_at_utc'] = training.get('training_completed_at_utc')
    metadata = metadata_values(wd, cfg, case, presentation_training, status, 'EXACT_FINAL')
    values.update(metadata)
    values['Notes'] = augment_notes(metadata, values['Notes'])
    for label, key in RR_LABELS.items():
        values[label] = main['rr'][key]
    for label, key in FR_LABELS.items():
        values['HQNR↑' if key == 'hqnr' else label] = main['fr'].get(key)
    for key, label in selections(case.updates):
        values.update(selection_values(label, reports[key]))
    for split, item in data['splits'].items():
        values[f'L100 {split} data SHA256'] = item['sha256']
        values[f'L100 {split} LP SHA256'] = item['lpan_sha256']
    if any('Exact50K' in label or 'EXACT50K' in label for label in values):
        raise AssertionError('L100 must not populate legacy Exact50K columns')
    return {key: '' if value is None else value for key, value in values.items()}


def plan_upsert(table, headers, values, *, header_row=3):
    values, labels = dict(values), label_map(headers)
    if 'Run' not in labels:
        raise ValueError('Existing tab has no semantic Run header')
    key_names = ('L100 sensor', 'L100 campaign', 'L100 run id')
    key = tuple(values[k] for k in key_names)
    if key[:2] != ('GF2', CAMPAIGN_ID) or values['Run'] != key[2]:
        raise ValueError('Invalid LOCAL-T compound key')
    def cell(row, label):
        col = labels.get(label, 0)
        return row[col - 1] if 0 < col <= len(row) else ''
    matches = []
    for number, row in enumerate(table[header_row:], header_row + 1):
        identity = tuple(cell(row, k) for k in key_names)
        if identity == key:
            if cell(row, 'Run') != values['Run']:
                raise ValueError('Conflicting compound key and Run cell')
            matches.append(number)
        elif cell(row, 'Run') == values['Run']:
            raise ValueError('Historical run alias is not an empty LOCAL-T row')
    if len(matches) > 1:
        raise ValueError('Duplicate LOCAL-T compound key')
    rownum = matches[0] if matches else max(len(table) + 1, header_row + 1)
    if 'Notes' in values:
        old = cell(table[rownum - 1], 'Notes') if rownum <= len(table) else ''
        values['Notes'] = merge_notes(values['Notes'], old)
    missing = [label for label in values if label not in labels]
    first = max(len(headers), max((len(r) for r in table[:header_row]), default=0)) + 1
    for col, label in enumerate(missing, first):
        labels[label] = col
    edits = [dict(range=a1(header_row, labels[label]), values=[[label]]) for label in missing]
    edits += [dict(range=a1(rownum, labels[label]), values=[[value]]) for label, value in values.items()]
    owned = {label: labels[label] for label in values}
    formats = metric_formats(owned, rownum)
    formats += [entry for entry in row_formats(owned, rownum) if ':' not in entry['range']]
    return dict(row=rownum, labels=labels, missing_headers=missing, values=values, edits=edits, formats=formats)


def apply_upsert(ws, values, *, header_row=3):
    table, headers = ws.get_all_values(), ws.row_values(header_row)
    plan = plan_upsert(table, headers, values, header_row=header_row)
    if 'L100 upload status' not in plan['values']:
        raise ValueError('Missing owned upload-status payload')
    controls = fetch_controls(ws, [header_row, plan['row']])
    touched = {a1(header_row, plan['labels'][label]): (header_row, plan['labels'][label])
               for label in plan['missing_headers']}
    touched.update({a1(plan['row'], plan['labels'][label]): (plan['row'], plan['labels'][label])
                    for label in plan['values']})
    if any(entry['range'] not in touched for entry in (*plan['edits'], *plan['formats'])):
        raise ValueError('Sheet edit/format extends outside the owned cells')
    for address, (row, col) in touched.items():
        reason = controlled_reason(controls, row, col)
        if reason:
            raise ValueError(f'Refusing controlled Sheet cell {address}: {reason}')
    if ws.get_all_values() != table or ws.row_values(header_row) != headers:
        raise ValueError('Sheet changed during upload planning; retry with fresh evidence')
    needed = max(plan['labels'].values())
    if needed > ws.col_count:
        ws.add_cols(needed - ws.col_count)
    if plan['row'] > ws.row_count:
        ws.add_rows(plan['row'] - ws.row_count)
    ws.batch_update(plan['edits'], value_input_option='RAW')
    if plan['formats']:
        ws.batch_format(plan['formats'])
    observed = ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')
    for label, want in plan['values'].items():
        col = plan['labels'][label]
        if not same_cell(observed[col - 1] if col <= len(observed) else '', want):
            raise ValueError('Sheet readback mismatch: ' + label)
    status_col = plan['labels']['L100 upload status']
    ws.batch_update([dict(range=a1(plan['row'], status_col), values=[['READBACK_VERIFIED']])], value_input_option='RAW')
    if ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')[status_col - 1] != 'READBACK_VERIFIED':
        raise ValueError('Sheet upload-status readback mismatch')
    return dict(row=plan['row'], gid=int(ws.id), worksheet=ws.title, readback_verified=True,
                payload_sha256=object_sha(plan['values']))


def upload_run(run, root=ROOT, *, activated=False, worksheet=None, header_row=3):
    if not activated:
        raise PermissionError('LOCAL-T Sheet upload requires explicit --upload activation')
    root = Path(root)
    cfg = yaml.safe_load((root / 'work_dir' / run / 'meta/config.resolved.yaml').read_text())
    case = case_from_config(cfg)
    values = row_values(run, root)
    if worksheet is None:
        import gspread
        constants = legacy_constants()
        book = gspread.service_account(filename=str(root / 'gspread' / Path(constants.CRED).name)).open(constants.SHEET)
        worksheet = book.worksheet(SHEET_TABS[case.server_id])
    if worksheet.title != SHEET_TABS[case.server_id]:
        raise ValueError('Refusing upload to another server/sensor tab')
    with write_locks(root):
        receipt = apply_upsert(worksheet, values, header_row=header_row)
        receipt.update(campaign_id=CAMPAIGN_ID, run_id=run, sensor='GF2', server=case.server_id,
                       horizon_updates=case.updates, uploaded_at_utc=utcnow())
        atomic_json(root / 'work_dir' / run / 'official/upload_receipt.json', receipt)
    return receipt
