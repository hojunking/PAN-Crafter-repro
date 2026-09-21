"""Own-server-only, fail-closed GF2 column layout repair without metric writes."""
import contextlib
import copy
import datetime as dt
from pathlib import Path

from reporting_extra.sensor_backfill import (TABS, _atomic_json, object_sha,
    read_json, sha256, utcnow, write_locks)
from reporting_extra.sensor_layout import COMMON_HEADERS, initialize_requests, migrate_front_requests


SCHEMA = 'GF2_LAYOUT_REPAIR_v1'
MAX_CELLS = 2_000_000
_GROUPS = {'GF2', 'GF2 실험', 'RR', 'FR', 'RR · 20 scenes', 'FR · 20 scenes',
           '효율', '기록 / 설정', 'Reduced Resolution', 'Full Resolution',
           'Reduced-Resolution', 'Full-Resolution', 'Efficiency', 'Cost',
           'FR·paper mat20', '본 결과(A_ON) RR', '본 결과(A_ON) FR·paper mat20'}
_CELL_KEYS = ('userEnteredValue', 'note', 'textFormatRuns')


def _column(index):
    out = ''
    while index:
        index, digit = divmod(index - 1, 26)
        out = chr(65 + digit) + out
    return out


def _read(ws):
    rows, cols = int(ws.row_count), int(ws.col_count)
    if rows < 3 or cols < 1 or rows * cols > MAX_CELLS:
        raise ValueError('Target Sheet geometry is absent or exceeds the bounded layout audit')
    title = ws.title.replace("'", "''")
    meta = ws.spreadsheet.fetch_sheet_metadata(params={
        'includeGridData': True, 'ranges': [f"'{title}'!A1:{_column(cols)}{rows}"],
        'fields': 'sheets(properties,merges,protectedRanges,tables,data)'})
    sheets = [s for s in meta.get('sheets', []) if s.get('properties', {}).get('sheetId') == int(ws.id)]
    if len(sheets) != 1:
        raise ValueError('Cannot inspect the exact target worksheet')
    sheet = sheets[0]
    props = sheet['properties']
    if (props.get('title') != ws.title or props.get('gridProperties', {}).get('rowCount') != rows
            or props.get('gridProperties', {}).get('columnCount') != cols):
        raise ValueError('Worksheet title or geometry changed during inspection')
    return sheet


def _cells(sheet):
    cells = {}
    for grid in sheet.get('data', []):
        for ri, row in enumerate(grid.get('rowData', []), grid.get('startRow', 0)):
            for ci, cell in enumerate(row.get('values', []), grid.get('startColumn', 0)):
                if cell:
                    if (ri, ci) in cells:
                        raise ValueError('Overlapping Sheet grid-data blocks')
                    cells[ri, ci] = cell
    return cells


def _known_merge(region, sid):
    return (region.get('sheetId') == sid and region.get('startRowIndex') == 1
            and region.get('endRowIndex') == 2
            and (region.get('startColumnIndex'), region.get('endColumnIndex'))
            in {(3, 11), (11, 16), (16, 18), (16, 21)})


def _inspect(sheet):
    if sheet.get('protectedRanges') or sheet.get('tables'):
        raise ValueError('Layout repair refuses protected ranges or native tables')
    sid = sheet['properties']['sheetId']
    merges = sheet.get('merges', [])
    if any(not _known_merge(region, sid) for region in merges):
        raise ValueError('Layout repair refuses unknown or non-group merged ranges')
    cells = _cells(sheet)
    for location, cell in cells.items():
        if 'formulaValue' in cell.get('userEnteredValue', {}):
            raise ValueError('Layout repair refuses formula cells, including indirect references')
        if any(cell.get(key) for key in ('dataValidation', 'chipRuns', 'dataSourceFormula', 'dataSourceTable')):
            raise ValueError('Layout repair refuses validated, chip, or data-source cells')
        if location[0] == 1:
            value = cell.get('userEnteredValue', {})
            text = value.get('stringValue', '')
            if value and (set(value) != {'stringValue'} or (text and text not in _GROUPS)):
                raise ValueError('Unowned row-2 content cannot be replaced by group headings')
            if any(cell.get(key) for key in ('note', 'textFormatRuns')):
                raise ValueError('Annotated row-2 content cannot be replaced')
        if location[0] == 2 and cell.get('textFormatRuns'):
            raise ValueError('Rich-text header requires explicit handling before layout repair')
    count = sheet['properties']['gridProperties']['columnCount']
    headers = []
    for col in range(count):
        value = cells.get((2, col), {}).get('userEnteredValue', {})
        if value and set(value) != {'stringValue'}:
            raise ValueError('Header labels must be literal strings')
        headers.append(value.get('stringValue', ''))
    if 'Run' not in headers:
        raise ValueError('Existing GF2 worksheet must have its semantic Run header')
    return headers, cells


def _mapping(size, requests):
    """Map every old physical column, including unlabeled data, to its new index."""
    order = list(range(size))
    for item in requests:
        if 'appendDimension' in item:
            order += [None] * item['appendDimension']['length']
        elif 'insertDimension' in item:
            request = item['insertDimension']['range']
            order[request['startIndex']:request['startIndex']] = [None] * (request['endIndex'] - request['startIndex'])
        elif 'moveDimension' in item:
            request = item['moveDimension']
            start, end, dest = request['source']['startIndex'], request['source']['endIndex'], request['destinationIndex']
            moving = order[start:end]
            del order[start:end]
            dest = dest if dest < start else dest - (end - start)
            order[dest:dest] = moving
    return {old: new for new, old in enumerate(order) if old is not None}, len(order)


def _semantic(cells):
    return {location: {key: cell[key] for key in _CELL_KEYS if key in cell}
            for location, cell in cells.items()
            if location[0] != 1 and any(key in cell for key in _CELL_KEYS)}


def repair_layout(ws, root, server, apply=False):
    """Plan/apply only the supplied server's GF2 tab; preserve all historic data.

    A whole-tab bounded control scan and fresh optimistic snapshot precede one
    atomic native request batch. Only known group headings are replaceable.
    Originals, legacy Date values, Notes, metrics, and provenance move together.
    Separate local audit receipts never modify any original upload/run receipt.
    """
    if server not in TABS or ws.title != TABS[server]:
        raise ValueError('Layout repair is restricted to the explicitly selected server GF2 tab')
    root = Path(root).resolve()
    with write_locks(root) if apply else contextlib.nullcontext():
        before = _read(ws)
        headers, cells = _inspect(before)
        sid = int(ws.id)
        requests = [{'unmergeCells': {'range': region}} for region in before.get('merges', [])]
        old_count = len(headers)
        if len(headers) < 30:
            count = 30 - len(headers)
            requests.append({'appendDimension': {'sheetId': sid, 'dimension': 'COLUMNS', 'length': count}})
            headers += [''] * count
        final_headers, migration = migrate_front_requests(sid, headers)
        requests.extend(migration)
        column_map, final_count = _mapping(old_count, requests)
        # Old owned group labels must not migrate into provenance-column headings.
        requests.append({'repeatCell': {
            'range': {'sheetId': sid, 'startRowIndex': 1, 'endRowIndex': 2,
                      'startColumnIndex': 0, 'endColumnIndex': final_count},
            'cell': {}, 'fields': 'userEnteredValue'}})
        # Migration already establishes row 3; avoid redundant header-value writes
        # that could erase a user's existing header rich-text runs.
        requests.extend(request for request in initialize_requests(sid)
                        if request.get('updateCells', {}).get('start', {}).get('rowIndex') != 2)
        expected = {(row, column_map[col]): payload for (row, col), payload in _semantic(cells).items()}
        for col, label in enumerate(final_headers):
            location = (2, col)
            payload = expected.setdefault(location, {})
            payload['userEnteredValue'] = {'stringValue': label}
        summary = dict(schema=SCHEMA, server=server, worksheet=ws.title, gid=sid,
                       apply=bool(apply), old_columns=old_count, new_columns=final_count,
                       moved_columns=sum(k != v for k, v in column_map.items()), requests=len(requests))
        if not apply:
            return dict(summary, final_headers=final_headers, batch_requests=requests)
        store = root / 'work_dir/_sensor_sheet' / server / 'layout'
        latest = read_json(store / 'latest.json') if (store / 'latest.json').exists() else {}
        source = {name: sha256(Path(__file__).parent / name)
                  for name in ('sensor_repair_layout.py', 'sensor_layout.py')}
        if (latest.get('readback_verified') is True and latest.get('after_sha256') == object_sha(before)
                and latest.get('reporting_sources') == source):
            return dict(summary, unchanged=True, readback_verified=True)
        if _read(ws) != before:
            raise ValueError('Worksheet changed after layout planning; no write performed')
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        snapshot = store / (stamp + '.before.json')
        _atomic_json(snapshot, dict(schema=SCHEMA, at_utc=utcnow(), worksheet=before,
                                    requests=requests, reporting_sources=source))
        ws.spreadsheet.batch_update({'requests': requests})
        # Refresh worksheet properties rather than trusting a stale client cache.
        ws = ws.spreadsheet.get_worksheet_by_id(sid)
        after = _read(ws)
        after_headers, after_cells = _inspect(after)
        actual = _semantic(after_cells)
        # Empty literal strings may be omitted by the API; normalize both ways.
        def normalize(values):
            normalized = {}
            for location, cell in values.items():
                item = copy.deepcopy(cell)
                if item.get('userEnteredValue') == {'stringValue': ''}:
                    item.pop('userEnteredValue')
                if item:
                    normalized[location] = item
            return normalized
        if after_headers != final_headers or normalize(actual) != normalize(expected):
            raise ValueError('Layout readback changed historical cell values, Notes, or provenance')
        grid = after['properties']['gridProperties']
        if grid.get('frozenRowCount') != 3 or grid.get('frozenColumnCount') != 3 or after.get('merges'):
            raise ValueError('Layout readback did not establish the common unmerged frozen header')
        receipt = dict(summary, at_utc=utcnow(), readback_verified=True,
                       before_sha256=object_sha(before), after_sha256=object_sha(after),
                       snapshot=str(snapshot), snapshot_sha256=sha256(snapshot), reporting_sources=source)
        _atomic_json(store / (stamp + '.receipt.json'), receipt)
        _atomic_json(store / 'latest.json', receipt)
        return receipt
