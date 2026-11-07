"""Prepare (never execute) the bounded 2026-11-08 prelaunch Sheet restoration.

This incident-specific utility accepts only the pinned original cloud backup.
The caller applies the reviewed request through Sheets, then reruns --verify.
No training/evaluation/foreign rows outside A381:BL411 are changed by its payload.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from directshift48.reporting import atomic_json
from sheet_publish_v2.transport import GoogleSheets, _matrix_equal
from ta2.evaluation import file_sha
from . import plan
from .publication import INDEX, destination
from .transport import _validate_snapshot

BACKUP_NAME = 'cloud_backup_1794067267070011368.json'
BACKUP_SHA256 = 'f3dbb4bd25cfec993e27400c2911685c393b93843f35424a28dc6397cf56091b'
FIRST, LAST = 381, 411


def _notes(snapshot, offset):
    cells = snapshot.get('cells', [])
    row = cells[offset] if offset < len(cells) else []
    return [row[i].get('note', '') if i < len(row) else '' for i in range(64)]


def build_request(backup, current, known_rows, first=FIRST, last=LAST):
    """Reject unrelated edits and restore all value/note cells in the proven span."""
    requests, changed, originals = [], [], []
    for number in range(first, last + 1):
        offset = number - 6
        before, actual = backup['rows'][offset], current['rows'][offset]
        if _matrix_equal([before], [actual]) and _notes(backup, offset) == _notes(current, offset):
            continue
        if (actual[INDEX['Campaign_ID']] != plan.CAMPAIGN_ID
                or actual[INDEX['Server']] != 's1'
                or not any(_matrix_equal([actual], [row]) for row in known_rows)):
            raise ValueError(f'Row {number} is not an exact known incident write; no request produced')
        # The failed values.append touched values only. Unexpected note changes
        # may be someone else's work and must not be replaced automatically.
        if _notes(current, offset) != _notes(backup, offset):
            raise ValueError(f'Row {number} notes changed outside the known values-only incident')
        old_cells = backup.get('cells', [])[offset]
        values = [{k: deepcopy(old_cells[i][k]) for k in ('userEnteredValue', 'note')
                   if k in old_cells[i]} if i < len(old_cells) else {} for i in range(64)]
        requests.append({'updateCells': {'range': {'sheetId': destination()['sheet_id'],
            'startRowIndex': number - 1, 'endRowIndex': number,
            'startColumnIndex': 0, 'endColumnIndex': 64},
            'rows': [{'values': values}], 'fields': 'userEnteredValue,note'}})
        changed.append(number)
        if any(v != '' for v in before):
            originals.append({'row': number, 'experiment': before[0],
                              'result_id': before[INDEX['Result_ID']]})
    return {'requests': requests}, {'changed_rows': changed, 'original_rows_restored': originals,
        'restored_empty_rows': [n for n in changed if n not in {r['row'] for r in originals}],
        'owned_columns': 'A:BL', 'outside_rows_untouched': True, 'extra_columns_untouched': True,
        'fields': 'userEnteredValue,note', 'physical_rows_inserted_or_deleted': False}


def incident_evidence(directory):
    directory = Path(directory)
    path = directory / BACKUP_NAME
    if file_sha(path) != BACKUP_SHA256:
        raise ValueError('The original prelaunch backup is not the pinned file')
    backup = json.loads(path.read_text()); known_rows = []; writes = []
    affected = set()
    for response_path in sorted(directory.glob('append_response_*.json')):
        response = json.loads(response_path.read_text())
        # Later repaired transport uses batchUpdate and no updates range.
        if 'updates' not in response:
            continue
        stamp = response_path.stem.removeprefix('append_response_')
        intent_path = directory / f'append_intent_{stamp}.json'
        intent = json.loads(intent_path.read_text())
        if intent['destination'] != destination():
            raise ValueError('Incident destination mismatch')
        update = response['updates']
        match = re.fullmatch(r"'WV3-ablations'!A(\d+):BL(\d+)", update.get('updatedRange', ''))
        if (response.get('spreadsheetId') != destination()['spreadsheet_id'] or not match
                or update['updatedColumns'] != 64
                or update['updatedRows'] != int(match[2]) - int(match[1]) + 1
                or update['updatedCells'] != update['updatedRows'] * 64):
            raise ValueError('Unknown write response; no incident restoration inferred')
        span = range(int(match[1]), int(match[2]) + 1)
        if min(span) < FIRST or max(span) > LAST:
            raise ValueError('Incident write extends beyond reviewed restoration rectangle')
        affected.update(span); known_rows.extend(intent['rows'])
        writes.append({'response': str(response_path), 'response_sha256': file_sha(response_path),
                       'intent': str(intent_path), 'intent_sha256': file_sha(intent_path),
                       'range': update['updatedRange']})
    if affected != set(range(FIRST, LAST + 1)):
        raise ValueError('Incident write coverage differs from the reviewed 31 rows')
    return backup, known_rows, writes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Full read-only GoogleSheets snapshot JSON')
    parser.add_argument('--verify', action='store_true', help='Read and check the completed restoration only')
    args = parser.parse_args()
    directory = plan.campaign_root('s1') / 'publication'
    backup, known, writes = incident_evidence(directory)
    current = (json.loads(args.snapshot.read_text()) if args.snapshot else
               GoogleSheets(plan.ROOT / 'gspread/account.json').snapshot(destination()))
    _validate_snapshot(current, {'destination': destination(), 'rows': []})
    output = directory / 'repair_20261108'
    atomic_json(output / ('after_snapshot.json' if args.verify else 'before_snapshot.json'), current)
    request, summary = build_request(backup, current, known)
    summary.update(backup=str(directory / BACKUP_NAME), backup_sha256=BACKUP_SHA256,
                   incident_writes=writes, request_sha256=plan.digest(request),
                   destination=destination(), dry_run=True)
    if args.verify:
        if request['requests']:
            raise ValueError('Original values/notes have not yet been fully restored')
        summary.update(restoration_verified=True)
        atomic_json(output / 'verified.json', summary)
    else:
        atomic_json(output / 'requests.json', request)
        atomic_json(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'incident_writes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
