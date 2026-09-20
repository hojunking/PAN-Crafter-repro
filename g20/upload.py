"""Official-JSON-only G20 upsert into existing sensor tabs; explicit activation.

Metrics retain their numeric precision. Four decimals are a Sheet display
format, never a rounding step before selection, persistence or upload.
"""
from pathlib import Path
import json

import yaml

from fh12.upload import a1, column, label_map, legacy_constants, same_cell
from qg40.sheet_helpers import fetch_controls, controlled_reason
from g20.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow, locked, camp
from g20.plan import CAMPAIGN_ID, SHEET_TABS, case_from_config, sensor_spec
from g20.postrun import SELECTIONS, report_selection, select_records, validate_grid

RR_LABELS = {'ERGAS↓': 'ergas', 'SAM↓': 'sam', 'PSNR↑': 'psnr', 'SSIM↑': 'ssim',
             'SCC↑': 'scc', 'Q4↑': 'q4', 'RMSE↓': 'rmse', 'CC↑': 'cc'}
FR_LABELS = {'HQNR(raw)↑': 'hqnr', 'D_lambda↓': 'd_lambda', 'D_s↓': 'd_s', 'JQM↑': 'jqm'}
PREFIXES = ('RAW_MAX', 'Target', 'Exact50K', 'RR_VAL_SELECTED', 'E_MIN_DIAG50')


def selection_values(prefix, report):
    empty = report.get('selection_id') == 'TARGET' and report.get('target_status') == 'no_eligible'
    values = {prefix + ' step': '' if empty else report['step'],
              prefix + ' checkpoint SHA256': '' if empty else report['checkpoint_sha256'],
              prefix + ' official': report['official_complete']}
    for domain, labels in (('rr', RR_LABELS), ('fr', FR_LABELS)):
        for label, key in labels.items():
            values[prefix + ' ' + label] = '' if empty else report[domain].get(key, '')
    values[prefix + ' JQM variant'] = '' if empty else report['fr'].get('jqm_variant', '')
    values[prefix + ' JQM status'] = '' if empty else report['fr'].get('jqm_status', 'not_measured')
    values[prefix + ' JQM reason'] = '' if empty else report['fr'].get('jqm_reason', '')
    values[prefix + ' test-aware'] = report['test_aware']
    values[prefix + ' primary'] = report['primary']
    for field in ('signed_mean', 'abs_mean', 'positive_fraction', 'reconstruction_max_abs_error'):
        values[prefix + ' signed Ds ' + field] = '' if empty else report['fr']['signed_ds'][field]
    values[prefix + ' signed Ds SHA256'] = '' if empty else object_sha(report['fr']['signed_ds'])
    values[prefix + ' joint pass'] = '' if empty else report['fr']['hqnr'] > .964 and report['rr']['ergas'] < .552
    values[prefix + ' strong joint pass'] = '' if empty else report['fr']['hqnr'] > .964 and report['rr']['ergas'] < .522
    return {k: '' if v is None else v for k, v in values.items()}


def metric_formats(labels, row):
    metric_names = set(RR_LABELS) | set(FR_LABELS) | {'HQNR↑', 'RR_VAL_SELECTED val ERGAS'}
    def is_metric(label):
        return label in metric_names or any(label.startswith(p + ' ') and
            (label[len(p) + 1:] in metric_names or label[len(p) + 1:] in
             {'signed Ds signed_mean', 'signed Ds abs_mean', 'signed Ds positive_fraction', 'signed Ds reconstruction_max_abs_error'}) for p in PREFIXES)
    return [dict(range=a1(row, col), format={'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}})
            for label, col in labels.items() if is_metric(label)]


def validate_target_metadata(target, selected, sensor):
    spec = sensor_spec(sensor)
    expected = dict(n_eligible=selected['n_eligible'], joint_pass=selected['joint_pass'],
                    strong_joint_pass=selected['strong_joint_pass'], threshold_comparison='>',
                    hqnr_threshold=spec.hqnr_threshold, ergas_goal=spec.ergas_goal,
                    ergas_strong_goal=spec.ergas_strong_goal, n_evaluated=50,
                    target_status=selected['target_status'],
                    selector_order=['ergas', '-scc', '-psnr', 'step'])
    if any(key not in target or target[key] != value for key, value in expected.items()):
        raise ValueError('Target eligibility/strict sensor threshold differs from the full-precision grid')


def row_values(run, root=ROOT):
    """Recompute selectors and prove same-checkpoint metrics before emitting cells."""
    root = Path(root)
    wd = root / 'work_dir' / run
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    case = case_from_config(cfg)
    if case.run_id != run:
        raise ValueError('Resolved case and run differ')
    status = read_json(wd / 'official/postrun_status.json')
    grid = read_json(wd / 'official/raw_grid.json')
    training = read_json(wd / 'meta/training_status.json')
    if (not status.get('official_complete') or status.get('actual_updates') != 50000
            or not training.get('training_complete') or training.get('actual_updates') != 50000
            or not grid.get('complete') or status.get('config_sha256') != object_sha(cfg)
            or status.get('source_identity') != grid.get('source_identity')):
        raise ValueError('Upload requires exact50K and complete official grid provenance')
    data = validate_grid(run, cfg, grid, root)
    selected = select_records(grid['records'], case.sensor)
    reports = {}
    for key, label in SELECTIONS:
        name = 'target_selection' if key == 'target' else key
        doc = read_json(wd / 'official' / f'{name}.json')
        if (doc.get('selection_id') != label or not doc.get('official_complete')
                or doc.get('normal_same_step_A_U') is not True
                or doc.get('run_id') != run or doc.get('campaign_id') != CAMPAIGN_ID
                or doc.get('sensor') != case.sensor or doc.get('n_evaluated') != 50
                or doc.get('test_aware') is not (key in ('raw_max', 'target', 'e_min_diag'))
                or doc.get('primary') is not (key in ('exact50k', 'rr_val_selected'))
                or any(doc.get(k) != grid.get(k) for k in ('config_sha256', 'data_sha256', 'source_identity'))):
            raise ValueError(f'Invalid {label} report context')
        record = selected[key]
        if record is None:
            if (doc.get('selection', 'missing') is not None or doc.get('target_status') != 'no_eligible'
                    or doc.get('n_eligible') != 0 or doc.get('step') is not None):
                raise ValueError('No-eligible Target contains fabricated checkpoint or metrics')
        elif (doc.get('step') != record['update']
              or doc.get('checkpoint_sha256') != record['checkpoint_identity']['model_sha256']
              or any(doc.get(k) != record.get(k) for k in ('rr', 'fr', 'val_ergas', 'checkpoint_identity'))):
            raise ValueError('Selected metrics differ from the complete raw grid')
        reports[key] = doc
    target = reports['target']
    validate_target_metadata(target, selected, case.sensor)
    cost = read_json(wd / 'official/profile.json')
    if (cost.get('config_sha256') != object_sha(cfg) or cost.get('source_identity') != grid['source_identity']
            or cost.get('num_bands') != 4 or cost.get('sensor') != case.sensor):
        raise ValueError('Measured C4 run-profile identity mismatch')
    field = cfg['g20']
    refpath = root / field['reference_manifest'] if field.get('reference_manifest') else None
    if refpath is None and case.role == 'T':
        from g20.references import reference_path
        own_reference = reference_path(case.reference_id, case.server_id, root)
        if own_reference.is_file():
            refpath = own_reference
    ref = read_json(refpath) if refpath and refpath.is_file() else {}
    if ref.get('schema') == 'G20_REFERENCE_BRIDGE_v1':
        ref = ref['origin_reference']
    if ref or case.role == 'S':
        from g20.references import validate_reference
        ref, _q, _cfg, _paths = validate_reference(field['reference_id'], case.server_id, root,
                                                  manifest_path=refpath, dataset_manifest=data)
    if not ref:
        raise ValueError('Measured Teacher calibration/reference must precede official upload')
    if case.role == 'T' and ref and ref['teacher_checkpoint_sha256'] != reports['exact50k']['checkpoint_sha256']:
        raise ValueError('Teacher calibration does not reference this exact50K')
    if case.role == 'S':
        if not ref or not ref.get('tau_R') or not ref.get('q_ref'):
            raise ValueError('Student measured calibration provenance missing')
        for record in grid['records']:
            if record['checkpoint_identity'].get('reference_sha256') != object_sha(ref):
                raise ValueError('Student checkpoint and Teacher calibration identity differ')
        if case.reference_sha256 and case.reference_sha256 != object_sha(ref):
            raise ValueError('Resolved conditional reference differs from its locked contract')
    branch_evidence = {}
    for name in ('screen_selection.json', 'confirmation_result.json', 'transfer_lock.json'):
        evidence_path = camp(root, case.server_id) / name
        if evidence_path.is_file():
            branch_evidence[name] = dict(path=str(evidence_path), sha256=sha256(evidence_path))
    if case.tier != 'CORE':
        name = 'screen_selection.json' if case.tier == 'CONDITIONAL_CONFIRM' else 'transfer_lock.json'
        branch_path = camp(root, case.server_id) / name
        if not branch_path.is_file() or object_sha(read_json(branch_path)) != case.resolution_sha256:
            raise ValueError('Conditional Student lacks its exact immutable resolution receipt')
    main = reports['exact50k']
    values = {'Run': run, '캠페인': 'G20', 'G20 sensor': case.sensor, 'G20 campaign': CAMPAIGN_ID,
              'G20 run id': run, 'G20 role': case.role, 'G20 server': case.server_id,
              'G20 C': 4, 'G20 maxDN': data['max_pixel'], 'G20 band order': ','.join(data['band_order']),
              'G20 MTF': data['mtf_sensor'], 'G20 Teacher alias': case.teacher_alias,
              'G20 Teacher seed': case.teacher_seed, 'G20 Student seed': case.student_seed,
              'G20 Teacher SHA256': ref.get('teacher_checkpoint_sha256', ref.get('teacher_model_sha256',
                                               reports['exact50k']['checkpoint_sha256'] if case.role == 'T' else field.get('teacher_sha256'))),
              'G20 calibration ID': ref.get('calibration_id', object_sha(ref) if ref else ''),
              'G20 tau_R': ref.get('tau_R'), 'G20 q_ref': ref.get('q_ref'),
              'G20 q cache SHA256': ref.get('q_cache_sha256', field.get('q_cache_sha256')),
              'G20 LP phase': data['recipe']['phase_id'], 'G20 data SHA256': grid['data_sha256'],
              'G20 numerical revision': grid['source_identity'].get('numeric_method_revision', field.get('method_revision')),
              'G20 source SHA256': grid['source_identity'].get('content_sha256'),
              'G20 case id': case.case_id, 'G20 tier': case.tier,
              'G20 profile': case.profile, 'G20 reference id': case.reference_id,
              'G20 alpha': case.alpha, 'G20 beta': case.beta, 'G20 lambda edge': case.lambda_edge,
              'G20 A peak LR': case.student_a_peak_lr, 'G20 lambda con': case.lambda_con,
              'G20 branch condition': case.resolution_sha256,
              'G20 branch evidence': json.dumps(branch_evidence, sort_keys=True),
              'G20 n evaluated': 50, 'G20 actual updates': 50000, 'G20 status': 'OFFICIAL_EVAL_COMPLETE',
              'G20 upload status': 'READBACK_PENDING', 'G20 cost scope': cost['scope'],
              'G20 FLOPs convention': cost['flops_convention'],
              'Params(M)': cost['params_m'], 'FLOPs(G)': cost['flops_g'], 'Infer(ms)': cost['infer_ms'],
              'Mem(MB)': cost.get('mem_mb'), 'Train(h)': training.get('training_seconds', 0) / 3600,
              'Target status': selected['target_status'], 'Target n eligible': selected['n_eligible'],
              'Target joint pass': selected['joint_pass'], 'Target strong joint pass': selected['strong_joint_pass'],
              'Target selector': f'G20_{case.sensor}_H_STRICT_E_SCC_PSNR_STEP_v1',
              'RR_VAL_SELECTED val ERGAS': reports['rr_val_selected']['val_ergas'],
              'E_MIN_DIAG50 independent test': False,
              'G20 primary selections': 'Exact50K and RR_VAL_SELECTED',
              'G20 selected step': main['step'], 'G20 A/U checkpoint SHA256': main['checkpoint_sha256'],
              'Notes': 'Main=Exact50K; co-primary=RR_VAL_SELECTED; raw-original mean-per-scene HQNR, same-step A/U. RAW_MAX, TARGET and E_MIN are test-aware. Signed Ds uses actual native Qhigh/Qlow, not NCC. JQM is SRF-substitute; unmeasured values remain empty.'}
    for label, key in RR_LABELS.items():
        values[label] = main['rr'][key]
    for label, key in FR_LABELS.items():
        values['HQNR↑' if key == 'hqnr' else label] = main['fr'].get(key)
    for prefix, (key, _) in zip(PREFIXES, SELECTIONS):
        values.update(selection_values(prefix, reports[key]))
    for split, item in data['splits'].items():
        values[f'G20 {split} data SHA256'] = item['sha256']
        values[f'G20 {split} LP SHA256'] = item['lpan_sha256']
    return {k: '' if v is None else v for k, v in values.items()}


def plan_upsert(table, headers, values, *, header_row=3, alias_crosswalk=None):
    """Pure, reviewable cell plan. No historic/benchmark row is used as a blank."""
    labels = label_map(headers)
    if 'Run' not in labels:
        raise ValueError('Existing tab has no semantic Run header')
    key = tuple(values[k] for k in ('G20 sensor', 'G20 campaign', 'G20 run id'))
    matches, legacy = [], []
    def cell(row, label):
        col = labels.get(label, 0)
        return row[col - 1] if 0 < col <= len(row) else ''
    for number, row in enumerate(table[header_row:], header_row + 1):
        identity = tuple(cell(row, k) for k in ('G20 sensor', 'G20 campaign', 'G20 run id'))
        if identity == key:
            if cell(row, 'Run') != values['Run']:
                raise ValueError('Conflicting compound key and Run cell')
            matches.append(number)
        elif cell(row, 'Run') == values['Run']:
            legacy.append(number)
    if len(matches) > 1 or (matches and legacy):
        raise ValueError('Duplicate compound key or historical run alias')
    if legacy:
        if len(legacy) != 1 or (alias_crosswalk or {}).get(values['Run']) != dict(row=legacy[0], sensor=key[0], campaign=key[1], run_id=key[2]):
            raise ValueError('Existing QGBASE/run alias requires explicit validated crosswalk; refusing duplicate row')
    rownum = (matches or legacy or [max(len(table) + 1, header_row + 1)])[0]
    missing = [label for label in values if label not in labels]
    # Respect occupied group-header extent too, even where the label row is blank.
    first = max(len(headers), max((len(r) for r in table[:header_row]), default=0)) + 1
    for col, label in enumerate(missing, first):
        labels[label] = col
    edits = [dict(range=a1(header_row, labels[label]), values=[[label]]) for label in missing]
    edits += [dict(range=a1(rownum, labels[label]), values=[[value]]) for label, value in values.items()]
    return dict(row=rownum, labels=labels, missing_headers=missing, edits=edits, formats=metric_formats(labels, rownum))


def apply_upsert(ws, values, *, header_row=3, alias_crosswalk=None):
    table = ws.get_all_values()
    headers = ws.row_values(header_row)
    plan = plan_upsert(table, headers, values, header_row=header_row, alias_crosswalk=alias_crosswalk)
    needed_cols = max(plan['labels'].values())
    # Structural reads precede mutation, including existing formula/dropdown cells.
    controls = fetch_controls(ws, [header_row, plan['row']])
    for edit in plan['edits']:
        row = header_row if edit['range'] in {a1(header_row, plan['labels'][k]) for k in plan['missing_headers']} else plan['row']
        label = next(k for k, col in plan['labels'].items() if a1(row, col) == edit['range'])
        reason = controlled_reason(controls, row, plan['labels'][label])
        if reason:
            raise ValueError(f'Refusing to overwrite controlled Sheet cell {edit["range"]}: {reason}')
    if needed_cols > ws.col_count:
        ws.add_cols(needed_cols - ws.col_count)
    if plan['row'] > ws.row_count:
        ws.add_rows(plan['row'] - ws.row_count)
    ws.batch_update(plan['edits'], value_input_option='RAW')
    if plan['formats']:
        ws.batch_format(plan['formats'])
    observed = ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')
    for label, want in values.items():
        col = plan['labels'][label]
        if not same_cell(observed[col - 1] if col <= len(observed) else '', want):
            raise ValueError(f'Sheet readback mismatch: {label}')
    status_col = plan['labels']['G20 upload status']
    ws.batch_update([dict(range=a1(plan['row'], status_col), values=[['READBACK_VERIFIED']])], value_input_option='RAW')
    if ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')[status_col - 1] != 'READBACK_VERIFIED':
        raise ValueError('Sheet upload-status readback mismatch')
    return dict(row=plan['row'], gid=int(ws.id), worksheet=ws.title, readback_verified=True)


def upload_run(run, root=ROOT, *, activated=False, worksheet=None, header_row=None, alias_crosswalk=None):
    if not activated:
        raise PermissionError('G20 Sheet upload requires explicit activation (--upload)')
    root = Path(root)
    cfg = yaml.safe_load((root / 'work_dir' / run / 'meta/config.resolved.yaml').read_text())
    case = case_from_config(cfg)
    values = row_values(run, root)
    if worksheet is None:
        # Import/connect only after activation and complete local artifact verification.
        import gspread
        constants = legacy_constants()
        book = gspread.service_account(filename=str(root / 'gspread' / Path(constants.CRED).name)).open(constants.SHEET)
        header_row = constants.ORIGIN_ROW + 1 if header_row is None else header_row
        worksheet = open_campaign_worksheet(book, case.server_id, header_row=header_row, activated=activated)
    if worksheet.title != SHEET_TABS[case.server_id]:
        raise ValueError('Refusing upload to a different sensor/server tab')
    with locked(root / 'work_dir/.g20_sheet_write.lock'):
        receipt = apply_upsert(worksheet, values, header_row=3 if header_row is None else header_row,
                               alias_crosswalk=alias_crosswalk)
        receipt.update(campaign_id=CAMPAIGN_ID, run_id=run, sensor=case.sensor, server=case.server_id,
                       payload_sha256=object_sha(values), uploaded_at_utc=utcnow())
        atomic_json(root / 'work_dir' / run / 'official/upload_receipt.json', receipt)
    return receipt


def open_campaign_worksheet(book, server, *, header_row=3, activated=False):
    """Only an explicit upload may create the plan-authorized missing GF2-s4 tab."""
    if not activated:
        raise PermissionError('Live worksheet access requires explicit upload activation')
    import gspread
    title = SHEET_TABS[server]
    try:
        return book.worksheet(title)
    except gspread.WorksheetNotFound:
        if server != 's4' or title != 'GF2-s4':
            raise ValueError('An expected existing GF2 tab is missing; no implicit replacement')
        worksheet = book.add_worksheet(title=title, rows=1000, cols=26)
        worksheet.batch_update([dict(range=a1(header_row, 1), values=[['Run']])], value_input_option='RAW')
        if worksheet.row_values(header_row) != ['Run']:
            raise ValueError('New GF2-s4 semantic header readback failed')
        return worksheet
