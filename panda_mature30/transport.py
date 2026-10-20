"""Value-table append with recoverable provenance notes and row-scoped updates.

Do not use appendCells: explicitly written empty strings in cleared historical
rows can count as sheet data even though the values API returns them as empty.
The values endpoint allocates rows on Google, so different servers do not race
by guessing the next free row. No row insertion, sorting or deletion is used.
"""
import os
from pathlib import Path
import time
from urllib.parse import quote

from . import publication as P
from sheet_publish_v2.transport import _validate_snapshot

TRANSPORT = 'PANDA_MATURE30_LOGICAL_TABLE_APPEND_V1'


def prepare(snapshot, value, server):
    if value['conflicts']:
        raise ValueError('Local evidence/cost conflicts; cloud untouched')
    if len(value['rows']) != len(value['notes']) or any(len(r) != 64 for r in value['rows']):
        raise ValueError('Fixed64 rows/notes mismatch')
    _validate_snapshot(snapshot, value)
    requests = P.row_requests(snapshot, value, server)
    additions = [row for req in requests for row in req.get('appendCells', {}).get('rows', [])]
    if additions:
        # A5:BL searches ALL data tables, including a tail after a blank gap.
        # Every nonempty row must anchor at A; otherwise table detection could
        # start at another column and shift the fixed64 payload sideways.
        for row in snapshot['rows']:
            if any(v != '' for v in row) and row[0] == '':
                raise ValueError('Unanchored data row: unsafe logical-table append')
        # Also check merges/protection immediately below the current extent.
        extended = dict(value, rows=[None] * (len(snapshot['rows']) + len(additions)))
        _validate_snapshot(snapshot, extended)
    return requests, additions


def reconcile_rows(adapter, value, state, server, *, apply=False):
    if value['conflicts']:
        raise ValueError('Local evidence/cost conflicts; cloud untouched')
    snapshot = adapter.snapshot(value['destination'])
    requests, additions = prepare(snapshot, value, server)
    if not apply:
        return {'action': 'DRY_RUN', 'transport': TRANSPORT,
                'new_rows': len(additions), 'request_count': len(requests),
                'readback_verified': False}
    state = Path(state)
    stamp = str(time.time_ns())
    if requests:
        P.R.atomic_json(state / ('cloud_backup_' + stamp + '.json'), snapshot)
        P.R.atomic_json(state / ('append_intent_' + stamp + '.json'), {
            'transport': TRANSPORT, 'destination': value['destination'],
            'rows': value['rows'], 'notes': value['notes']})
    dest = value['destination']
    url = adapter.BASE + dest['spreadsheet_id']
    append_range = None
    if additions:
        # RAW preserves numeric/bool types and does not interpret strings as
        # formulas. Full-width rows contain stable Result_ID + JSON provenance.
        rows = [[next(iter(c['userEnteredValue'].values())) for c in r['values']]
                for r in additions]
        table = "'%s'!A5:BL" % dest['sheet_title'].replace("'", "''")
        response = adapter._request('POST', url + '/values/' + quote(table, safe='') + ':append',
            params={'valueInputOption': 'RAW', 'insertDataOption': 'OVERWRITE'},
            json={'majorDimension': 'ROWS', 'values': rows})
        P.R.atomic_json(state / ('append_response_' + stamp + '.json'), response)
        append_range = response.get('updates', {}).get('updatedRange')
        # Never retry POST here. If the response was lost after commit, the next
        # poll finds the stable IDs and only repairs the missing A-cell notes.
        snapshot = adapter.snapshot(dest)
        requests, remaining = prepare(snapshot, value, server)
        if remaining:
            raise ValueError('Appended IDs missing from readback; inspect intent before retry')
    if any('appendCells' in req for req in requests):
        raise AssertionError('Legacy appendCells must never reach Google')
    if requests:
        # Own rows only, re-resolved by Result_ID after any append. Concurrent
        # other-server appends do not shift existing row indices.
        adapter._request('POST', url + ':batchUpdate', json={'requests': requests})
    fresh = adapter.snapshot(dest)
    pending, _ = prepare(fresh, value, server)
    if pending:
        raise ValueError('Own-row readback differs; retry by fresh snapshot, not blind append')
    ids = {r[P.INDEX['Result_ID']] for r in value['rows']}
    positions = [i + 6 for i, r in enumerate(fresh['rows']) if r[P.INDEX['Result_ID']] in ids]
    return {'action': 'OWN_ROWS_VERIFIED', 'transport': TRANSPORT,
            'publisher_pid': os.getpid(), 'at': P.R.utc_now(),
            'readback_verified': True, 'row_count': len(value['rows']),
            'sheet_rows': positions, 'append_range': append_range,
            'foreign_rows_untouched': True}

