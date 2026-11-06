"""Promote raw-evidence metrics and costs to four decimal places, accepting 3dp rows."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

from directshift48.reporting import INDEX, HEADERS, atomic_json, canonical_json

POLICY = 'WV3_MAIN_COST_4DP_V1'
LEGACY = {'rgb': 'WV3_MAIN_RGB_REPORTING_COST_3DP_V1', 'w128': 'WV3_W128_COST_3DP_V1'}


def raw_numbers(note):
    reporting = note['reporting']
    values = dict(reporting['raw_metrics'])
    values.update(reporting.get('cost_rr', {}))
    values.update({k: v for k, v in reporting.get('timing', {}).items()
                   if k in ('Train(h)', 'Eval(h)', 'Wall(h)')})
    return values


def promote(value, kind):
    value = deepcopy(value)
    for row in value['rows']:
        note = json.loads(row[INDEX['Original description / notes']])
        report = note['reporting']
        if report['policy'] != LEGACY[kind] or report['decimal_places'] != 3:
            raise ValueError('Unexpected source precision policy')
        raw = raw_numbers(note)
        for i in range(3, 25):
            if type(row[i]) in (int, float):
                if HEADERS[i] not in raw or type(raw[HEADERS[i]]) not in (int, float):
                    raise ValueError('Cannot restore unrounded evidence: ' + HEADERS[i])
                row[i] = round(raw[HEADERS[i]], 4)
        report.update(policy=POLICY, source_reporting_policy=LEGACY[kind], decimal_places=4,
                      selection_uses_unrounded_evidence=True)
        row[INDEX['Original description / notes']] = canonical_json(note)
    value['notes'] = [r[INDEX['Original description / notes']] for r in value['rows']]
    from donor30.plan import canonical_sha256
    value['view_hash'] = canonical_sha256(value['rows'])
    return value


def verified_legacy_row(row, kind):
    """Validate the 4dp representation, then reuse the immutable-evidence verifier."""
    note = json.loads(row[INDEX['Original description / notes']])
    report = note.get('reporting', {})
    if report.get('policy') != POLICY:
        return row  # Original verifier must authenticate legacy/full-precision rows.
    if report.get('decimal_places') != 4 or report.get('source_reporting_policy') != LEGACY[kind]:
        raise ValueError('Unknown four-decimal provenance')
    restored = list(row)
    raw = raw_numbers(note)
    for i in range(3, 25):
        if type(row[i]) in (int, float):
            if HEADERS[i] not in raw or row[i] != round(raw[HEADERS[i]], 4):
                raise ValueError('Four-decimal cell differs from raw evidence')
            restored[i] = round(raw[HEADERS[i]], 3)
    report.update(policy=LEGACY[kind], decimal_places=3)
    report.pop('source_reporting_policy')
    restored[INDEX['Original description / notes']] = canonical_json(note)
    return restored


def install(kind):
    module_name = 'panda_wv3_w128.publication' if kind == 'w128' else 'panda_wv3_main_reporting.publication'
    module = importlib.import_module(module_name)
    if getattr(module, '_reporting4_installed', False):
        return module
    base_project, base_verify = module.project, module.verify_owned
    module.project = lambda root, server: promote(base_project(root, server), kind)
    module.verify_owned = lambda row, value, server: base_verify(verified_legacy_row(row, kind), value, server)

    def publish(root, server, *, apply=False, credential=None):
        from sheet_publish_v2.transport import GoogleSheets
        transport = importlib.import_module('panda_wv3_w128.transport' if kind == 'w128'
                                            else 'panda_wv3_main_rgb.transport')
        base = module if kind == 'w128' else module.original
        adapter = GoogleSheets(credential or module.plan.ROOT / 'gspread/account.json')
        root = Path(root)
        value = module.project(root, server)
        @contextmanager
        def verified():
            if kind == 'w128':
                yield
            else:
                with module.verified_rounding():
                    yield
        with verified():
            receipt = transport.reconcile_rows(adapter, value, root / 'reporting4/publication', server, apply=apply)
        if apply and receipt['sheet_rows']:
            requests = [{'repeatCell': {'range': {'sheetId': base.destination()['sheet_id'],
                'startRowIndex': n - 1, 'endRowIndex': n, 'startColumnIndex': 3, 'endColumnIndex': 25},
                'cell': {'userEnteredFormat': {'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}}},
                'fields': 'userEnteredFormat.numberFormat'}} for n in receipt['sheet_rows']]
            adapter._request('POST', adapter.BASE + base.destination()['spreadsheet_id'] + ':batchUpdate',
                             json={'requests': requests})
        atomic_json(root / 'reporting4/last_poll.json', receipt)
        atomic_json(root / 'publication/last_poll.json', receipt)
        return receipt

    module.publish = publish
    module._reporting4_installed = True
    return module
