"""Append-only metric supplement for existing FH12/FH20R1 Sheet rows.

This module never selects checkpoints, changes official results or touches the
training campaign's source fingerprint. Raw float precision is retained; only
the display number format is shortened. No Sheet/tab/experiment row is created.
"""
import fcntl
import math
from pathlib import Path

from fh12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow
from fh12.upload import GIDS, TABS, a1, column, label_map, legacy_constants, same_cell

SCHEMA = 'PAN_SUPPLEMENTAL_METRICS_v1'
RECEIPT_SCHEMA = 'PAN_SUPPLEMENTAL_UPLOAD_v1'
SELECTIONS = {
    'raw_max': ('RAW_MAX', 'RAW_MAX'),
    'target_selection': ('Target', 'TARGET'),
    'exact50k': ('Exact50K', 'EXACT50K'),
    'rr_val_selected': ('RR_VAL_SELECTED', 'RR_VAL_SELECTED'),
    'e_min_diag': ('E_MIN_DIAG50', 'E_MIN_DIAG50'),
}
EXTRA_LABELS = {'RMSE↓': 'rmse', 'CC↑': 'cc', 'JQM↑': 'jqm'}
METRIC_LABELS = frozenset({
    'ERGAS↓', 'SAM↓', 'PSNR↑', 'SSIM↑', 'SCC↑', 'Q2n↑', 'Q4↑', 'Q8↑',
    'RMSE↓', 'CC↑', 'JQM↑', 'HQNR↑', 'HQNR(raw)↑', 'D_lambda↓', 'D_s↓',
})


class BaseUploadPending(ValueError):
    """A verified ordinary campaign row must be uploaded before its supplement."""


class BaseRowMissing(BaseUploadPending):
    """No campaign row and no potentially conflicting row with this Run exist."""


def _validated_report(run, root):
    from reporting_extra.evaluation import validate_report
    report = validate_report(run, root=root)
    path = Path(root) / 'work_dir' / run / 'supplemental_metrics/report.json'
    if read_json(path) != report:
        raise ValueError('Supplemental report changed during verification')
    return report, sha256(path)


def campaign_prefix(report):
    campaign = report.get('campaign_id', '')
    if campaign.startswith('WV3_FH20R1_'):
        return 'FH20R1'
    if campaign.startswith('WV3_FH12_'):
        return 'FH12'
    raise ValueError('Unsupported supplement campaign')


def values_from_report(report, report_sha256):
    """Build a narrow write set; the evaluator proves all underlying identities."""
    if report.get('schema') != SCHEMA or report.get('server_id') not in GIDS:
        raise ValueError('Invalid supplemental report schema/server')
    campaign_prefix(report)
    selections = report.get('selections', {})
    if set(selections) != set(SELECTIONS):
        raise ValueError('All five official selections must be certified')
    values = {}
    for name, (prefix, selection_id) in SELECTIONS.items():
        item = selections[name]
        if item.get('selection_id') != selection_id:
            raise ValueError(f'Supplement selector mismatch: {name}')
        empty = name == 'target_selection' and item.get('status') == 'no_eligible'
        if empty:
            if any(item.get(k) is not None for k in ('step', 'checkpoint_sha256', 'metrics')):
                raise ValueError('Malformed supplemental no-eligible target')
        elif item.get('status') != 'complete' or not item.get('checkpoint_sha256') or not item.get('step'):
            raise ValueError(f'Incomplete supplemental selection: {name}')
        if not item.get('official_report_sha256'):
            raise ValueError('Missing original report fingerprint')
        for label, metric in EXTRA_LABELS.items():
            value = '' if empty else item['metrics'][metric]
            if not empty and (isinstance(value, bool) or not math.isfinite(float(value))):
                raise ValueError(f'Invalid supplemental metric: {name}.{metric}')
            values[prefix + ' ' + label] = value
            if name == 'raw_max':
                values[label] = value
    if selections['exact50k']['step'] != 50000:
        raise ValueError('Exact50K supplemental checkpoint is not update 50000')
    values.update({
        'Extra metrics schema': SCHEMA,
        'Extra metrics report SHA256': report_sha256,
        'Extra metrics evaluator SHA256': object_sha(report['evaluator_identity']),
        'Extra metrics config SHA256': report['config_sha256'],
        'Extra metrics data SHA256': report['data_sha256'],
        'Extra metrics status': 'READBACK_PENDING',
        'Extra metrics protocol': 'RMSE/CC: RR20; JQM: FR20; each official same-step checkpoint; RAW_MAX main',
        'Extra metrics JQM variant': report['jqm_variant'],
        'Extra metrics inference devices': ','.join(report['inference_devices']),
        'Extra metrics inference note': 'Separate FP32 re-inference; original core9 and selectors retained; CPU/CUDA not claimed bit-identical',
    })
    for name, (prefix, _) in SELECTIONS.items():
        values[prefix + ' extra official report SHA256'] = selections[name]['official_report_sha256']
    return values


def is_metric_header(label):
    """Whitelist quality metrics only; never IDs, steps, seeds, LR, NOA or V64."""
    if label in METRIC_LABELS or label == 'RR_VAL_SELECTED val ERGAS':
        return True
    for prefix, _ in SELECTIONS.values():
        if label.startswith(prefix + ' '):
            return label[len(prefix) + 1:] in METRIC_LABELS
    return False


def metric_formats(labels, last_row, header_row=3, first_row=None):
    if last_row <= header_row:
        return []
    first_row = header_row + 1 if first_row is None else first_row
    return [dict(range=f'{column(col)}{first_row}:{column(col)}{last_row}',
                 format={'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}})
            for label, col in labels.items() if is_metric_header(label)]


def _cell(row, labels, label):
    col = labels.get(label, 0)
    return row[col - 1] if 0 < col <= len(row) else ''


def fetch_controls(ws, rows):
    """Read formulas/controls, not formatted text, before changing any cells.

    Google Sheets CellData and table-column metadata are needed to distinguish
    ordinary numeric cells from formulas, dropdowns, chips and typed columns.
    A failed/unsupported structural read is an error, never permission to write.
    """
    title = ws.title.replace("'", "''")
    fields = ('namedRanges(namedRangeId,range),sheets(properties(sheetId),merges,'
              'protectedRanges(range,namedRangeId,unprotectedRanges),'
              'tables(range,columnProperties(columnIndex,columnType,dataValidationRule)),'
              'data(startRow,startColumn,rowData(values(userEnteredValue(formulaValue),'
              'dataValidation,chipRuns,dataSourceFormula,dataSourceTable))))')
    meta = ws.spreadsheet.fetch_sheet_metadata(params={
        'includeGridData': True,
        'ranges': [f"'{title}'!A{row}:{column(ws.col_count)}{row}" for row in sorted(set(rows))],
        'fields': fields,
    })
    sheets = [sheet for sheet in meta.get('sheets', []) if sheet.get('properties', {}).get('sheetId') == int(ws.id)]
    if len(sheets) != 1:
        raise ValueError('Cannot verify target Sheet controls')
    return dict(sheet=sheets[0], named_ranges={v['namedRangeId']: v['range'] for v in meta.get('namedRanges', [])})


def _covers(grid_range, sheet_id, row, col):
    return (grid_range.get('sheetId', sheet_id) == sheet_id and
            grid_range.get('startRowIndex', 0) <= row - 1 < grid_range.get('endRowIndex', float('inf')) and
            grid_range.get('startColumnIndex', 0) <= col - 1 < grid_range.get('endColumnIndex', float('inf')))


def controlled_reason(controls, row, col):
    sheet = controls['sheet']; sid = sheet['properties']['sheetId']
    for protected in sheet.get('protectedRanges', []):
        region = protected.get('range')
        if region is None:
            region = controls['named_ranges'].get(protected.get('namedRangeId'))
        if region is None:
            raise ValueError('Unresolved protected range; refusing unsafe write')
        if (_covers(region, sid, row, col) and not
                any(_covers(r, sid, row, col) for r in protected.get('unprotectedRanges', []))):
            return 'protected range'
    if any(_covers(region, sid, row, col) for region in sheet.get('merges', [])):
        return 'merged cell'
    for table in sheet.get('tables', []):
        if not _covers(table['range'], sid, row, col):
            continue
        relative_col = col - 1 - table['range'].get('startColumnIndex', 0)
        for prop in table.get('columnProperties', []):
            if prop.get('columnIndex') == relative_col and (prop.get('dataValidationRule') or
                    prop.get('columnType', 'COLUMN_TYPE_UNSPECIFIED') != 'COLUMN_TYPE_UNSPECIFIED'):
                return 'typed table column'
    for grid in sheet.get('data', []):
        ri, ci = row - 1 - grid.get('startRow', 0), col - 1 - grid.get('startColumn', 0)
        rows = grid.get('rowData', [])
        if 0 <= ri < len(rows):
            cells = rows[ri].get('values', [])
            if 0 <= ci < len(cells):
                cell = cells[ci]
                if 'formulaValue' in cell.get('userEnteredValue', {}):
                    return 'formula'
                for key in ('dataValidation', 'chipRuns', 'dataSourceFormula', 'dataSourceTable'):
                    if cell.get(key):
                        return key
    return None


def find_verified_row(table, labels, report, header_row=3):
    prefix = campaign_prefix(report)
    required = ('Run', prefix + ' campaign', prefix + ' run id')
    if any(key not in labels for key in required):
        raise BaseUploadPending('Ordinary campaign compound-key headers are not present')
    matches = []
    for number, row in enumerate(table[header_row:], header_row + 1):
        if (_cell(row, labels, prefix + ' campaign') == report['campaign_id'] and
                _cell(row, labels, prefix + ' run id') == report['run_id']):
            if _cell(row, labels, 'Run') != report['run_id']:
                raise ValueError('Sheet compound key disagrees with Run')
            matches.append((number, row))
    if not matches:
        if any(_cell(row, labels, 'Run') == report['run_id'] for row in table[header_row:]):
            raise ValueError('Existing Run row lacks matching campaign provenance; refusing base-row recovery')
        raise BaseRowMissing('Ordinary campaign row has not been uploaded yet')
    if len(matches) != 1:
        raise ValueError('Duplicate Sheet campaign/run rows; refusing ambiguous update')
    number, row = matches[0]
    for name, (prefix, _) in SELECTIONS.items():
        item = report['selections'][name]
        for suffix, key in ((' step', 'step'), (' checkpoint SHA256', 'checkpoint_sha256')):
            label = prefix + suffix
            if label not in labels:
                raise BaseUploadPending('Original selected-checkpoint provenance is missing')
            expected = item[key] if item[key] is not None else ''
            if not same_cell(_cell(row, labels, label), expected):
                raise ValueError(f'Sheet and supplemental checkpoint differ: {label}')
    return number


def write_supplement(ws, report, report_sha256, header_row=3):
    """Narrow update + readback, injectable worksheet for deterministic tests."""
    server = report['server_id']
    if int(ws.id) != GIDS[server]:
        raise ValueError('Supplement worksheet GID does not match the local server')
    values = values_from_report(report, report_sha256)
    headers = ws.row_values(header_row)
    labels = label_map(headers)
    table = ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
    rownum = find_verified_row(table, labels, report, header_row)
    missing = [label for label in values if label not in labels]
    # Protect far-right human cells even if their header is intentionally blank.
    first = max([len(headers), len(ws.row_values(header_row - 1))] + [len(row) for row in table]) + 1
    last = first + len(missing) - 1
    edits = []
    targets = []
    for col, label in enumerate(missing, first):
        labels[label] = col
        edits.extend([
            {'range': a1(header_row - 1, col), 'values': [['Supplemental metrics']]},
            {'range': a1(header_row, col), 'values': [[label]]},
        ])
        targets.extend([(header_row - 1, col), (header_row, col)])
    edits.extend({'range': a1(rownum, labels[label]), 'values': [[value]]}
                 for label, value in values.items())
    targets.extend((rownum, labels[label]) for label in values)
    controls = fetch_controls(ws, (header_row - 1, header_row, rownum))
    for row, col in targets:
        reason = controlled_reason(controls, row, col)
        if reason:
            raise ValueError(f'Refusing to overwrite {reason} at {a1(row, col)}')
    if last > ws.col_count:
        ws.add_cols(last - ws.col_count)
    ws.batch_update(edits, value_input_option='RAW')
    safe_metric_labels = {label: col for label, col in labels.items()
                          if not controlled_reason(controls, rownum, col)}
    formats = metric_formats(safe_metric_labels, rownum, header_row, first_row=rownum)
    if formats:
        ws.batch_format(formats)
    observed = ws.row_values(rownum, value_render_option='UNFORMATTED_VALUE')
    for label, expected in values.items():
        if not same_cell(_cell(observed, labels, label), expected):
            raise ValueError(f'Supplement readback mismatch at row {rownum}, {label}')
    # Recheck base provenance after the write; a human may have edited/sorted it.
    if find_verified_row([[]] * header_row + [observed], labels, report, header_row) != header_row + 1:
        raise ValueError('Sheet provenance changed during supplemental upload')
    status_col = labels['Extra metrics status']
    ws.batch_update([{'range': a1(rownum, status_col), 'values': [['VERIFIED']]}], value_input_option='RAW')
    verified = ws.row_values(rownum, value_render_option='UNFORMATTED_VALUE')
    if _cell(verified, labels, 'Extra metrics status') != 'VERIFIED':
        raise ValueError('Supplement status readback failed')
    return dict(schema=RECEIPT_SCHEMA, campaign_id=report['campaign_id'], run_id=report['run_id'],
                server_id=server, gid=ws.id, worksheet=TABS[server], row=rownum,
                report_sha256=report_sha256, evaluator_identity_sha256=object_sha(report['evaluator_identity']),
                jqm_variant=report['jqm_variant'], inference_devices=report['inference_devices'],
                selections={name: {key: item[key] for key in ('step', 'checkpoint_sha256', 'official_report_sha256')}
                            for name, item in report['selections'].items()},
                readback_verified=True, display_number_format='0.0000',
                formatted_metric_columns=len(formats), uploaded_at_utc=utcnow())


def upload_run(run, root=ROOT):
    root = Path(root)
    report, fingerprint = _validated_report(run, root)
    from tools.fh12_runner import detect_server
    local = detect_server(root)
    if local != report['server_id']:
        raise ValueError('Refusing a supplemental upload into another server tab')
    gu = legacy_constants()
    import gspread
    lockpath = root / 'work_dir/.gspread_write.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        book = gspread.service_account(filename=str(root / 'gspread' / Path(gu.CRED).name)).open(gu.SHEET)
        ws = book.worksheet(TABS[local])
        receipt = write_supplement(ws, report, fingerprint, header_row=gu.ORIGIN_ROW + 1)
        atomic_json(root / 'work_dir' / run / 'supplemental_metrics/upload_receipt.json', receipt)
    return receipt


def upload_with_base_recovery(run, root=ROOT):
    """Recover only a proven absent base row through its original full validator.

    An existing ambiguous/incomplete row, missing headers, verification failure
    or upload failure is never authorization to append or overwrite a base row.
    A concurrent original uploader is safe: it shares the same write lock and
    performs its own compound-key upsert. One recovery attempt is sufficient.
    """
    try:
        return upload_run(run, root=root)
    except BaseRowMissing:
        report, _ = _validated_report(run, root)
        if campaign_prefix(report) == 'FH20R1':
            from fh20r1.upload import upload_run as original_upload
        else:
            from fh12.upload import upload_run as original_upload
        original_upload(run, root=root)
        return upload_run(run, root=root)
