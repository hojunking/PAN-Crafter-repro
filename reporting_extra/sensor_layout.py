"""Pure Google Sheets request builders for the common GF2 experiment layout.

Nothing in this module accesses Sheets or imports a training/campaign package.
Callers own authorization, current-sheet inspection, and execution. In particular,
column migration requires the caller to establish that there are no merged ranges
spanning the columns to be moved (or explicitly handle only its own merged ranges).
The builders never merge, hide, clear, or delete cells/columns.
"""
from __future__ import annotations

from collections.abc import Mapping


COMMON_HEADERS = (
    '', 'Run', '캠페인', 'ERGAS↓', 'SAM↓', 'PSNR↑', 'SSIM↑', 'SCC↑', 'Q4↑',
    'RMSE↓', 'CC↑', 'D_lambda↓', 'D_s↓', 'HQNR↑', 'HQNR(V64)↑', 'JQM↑',
    'Params(M)', 'FLOPs(G)', 'Infer(ms)', 'Mem(MB)', 'Train(h)', 'Date', 'Notes',
    'Seed', 'Model', 'Input', 'Selection', 'Wall(h)', 'Train time scope',
)

_ORANGE = {'red': 0.9373, 'green': 0.4902, 'blue': 0.0}
_PALE_ORANGE = {'red': 0.9882, 'green': 0.8941, 'blue': 0.8392}
_WHITE = {'red': 1.0, 'green': 1.0, 'blue': 1.0}


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _headers(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('Headers must be a sequence of strings, not one string')
    labels = list(values)
    if any(not isinstance(label, str) for label in labels):
        raise ValueError('Every header must be a string')
    nonempty = [label for label in labels if label]
    if len(set(nonempty)) != len(nonempty):
        raise ValueError('Nonempty headers must be unique before column migration')
    return labels


def _column(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _write_headers(sheet_id, row, start, labels):
    return {'updateCells': {
        'start': {'sheetId': sheet_id, 'rowIndex': row - 1, 'columnIndex': start},
        'rows': [{'values': [{'userEnteredValue': {'stringValue': label}}
                             for label in labels]}],
        'fields': 'userEnteredValue',
    }}


def initialize_requests(sheet_id, header_row=3):
    """Return native orange formatting/headers for a NEW or owned GF2 sheet.

    This writes the common prefix on ``header_row`` and its group labels on the
    preceding row. It must not be used as a blanket migration of an existing
    user-owned header row; use ``migrate_front_requests`` first where appropriate.
    The caller must provide at least 29 columns and ``header_row`` rows.
    """
    _positive_integer(sheet_id, 'sheet_id', minimum=0)
    _positive_integer(header_row, 'header_row', minimum=2)
    groups = [''] * len(COMMON_HEADERS)
    for label, group in [('Run', 'GF2 실험'), ('ERGAS↓', 'RR · 20 scenes'),
                         ('D_lambda↓', 'FR · 20 scenes'), ('Params(M)', '효율'),
                         ('Date', '기록 / 설정')]:
        groups[COMMON_HEADERS.index(label)] = group
    header_range = {'sheetId': sheet_id, 'startRowIndex': header_row - 1,
                    'endRowIndex': header_row, 'startColumnIndex': 0,
                    'endColumnIndex': len(COMMON_HEADERS)}
    group_range = dict(header_range, startRowIndex=header_row - 2,
                       endRowIndex=header_row - 1)
    requests = [
        _write_headers(sheet_id, header_row, 0, COMMON_HEADERS),
        _write_headers(sheet_id, header_row - 1, 0, groups),
        {'updateSheetProperties': {
            'properties': {'sheetId': sheet_id,
                           'gridProperties': {'frozenRowCount': header_row,
                                              'frozenColumnCount': 3}},
            'fields': 'gridProperties.frozenRowCount,gridProperties.frozenColumnCount'}},
        {'repeatCell': {'range': header_range, 'cell': {'userEnteredFormat': {
            'backgroundColor': dict(_ORANGE), 'horizontalAlignment': 'CENTER',
            'verticalAlignment': 'MIDDLE', 'wrapStrategy': 'CLIP',
            'textFormat': {'bold': True, 'foregroundColor': dict(_WHITE)}}},
            'fields': 'userEnteredFormat.backgroundColor,userEnteredFormat.horizontalAlignment,'
                      'userEnteredFormat.verticalAlignment,userEnteredFormat.wrapStrategy,'
                      'userEnteredFormat.textFormat.bold,userEnteredFormat.textFormat.foregroundColor'}},
        {'repeatCell': {'range': group_range, 'cell': {'userEnteredFormat': {
            'backgroundColor': dict(_PALE_ORANGE), 'horizontalAlignment': 'LEFT',
            'verticalAlignment': 'MIDDLE', 'wrapStrategy': 'CLIP',
            'textFormat': {'bold': True}}},
            'fields': 'userEnteredFormat.backgroundColor,userEnteredFormat.horizontalAlignment,'
                      'userEnteredFormat.verticalAlignment,userEnteredFormat.wrapStrategy,'
                      'userEnteredFormat.textFormat.bold'}},
        {'updateDimensionProperties': {
            'range': {'sheetId': sheet_id, 'dimension': 'COLUMNS',
                      'startIndex': 0, 'endIndex': len(COMMON_HEADERS)},
            'properties': {'pixelSize': 90}, 'fields': 'pixelSize'}},
    ]
    for label, width in [('Run', 400), ('Notes', 500), ('Date', 110)]:
        index = COMMON_HEADERS.index(label)
        requests.append({'updateDimensionProperties': {
            'range': {'sheetId': sheet_id, 'dimension': 'COLUMNS',
                      'startIndex': index, 'endIndex': index + 1},
            'properties': {'pixelSize': width}, 'fields': 'pixelSize'}})
    return requests


def row_formats(labels, row):
    """Return gspread ``batch_format`` entries without changing cell values.

    ``labels`` may be a mapping of header label to its actual one-based column
    number, including sparse or insertion-order-independent mappings, or an
    ordered full header sequence. Apply to a data row, not the header.
    Number formats, colors, and row heights
    are intentionally untouched; long Run/Notes values are clipped visually,
    never truncated in storage.
    """
    _positive_integer(row, 'row')
    if isinstance(labels, Mapping):
        if any(not isinstance(label, str) for label in labels):
            raise ValueError('Every header label must be a string')
        positions = dict(labels)
        for column in positions.values():
            _positive_integer(column, 'column')
        if len(set(positions.values())) != len(positions):
            raise ValueError('Header labels must have distinct column positions')
        extent = max(positions.values(), default=0)
    else:
        ordered = _headers(labels)
        positions = {label: index for index, label in enumerate(ordered, 1) if label}
        extent = len(ordered)
    if not extent:
        return []
    result = [{'range': f'A{row}:{_column(extent)}{row}',
               'format': {'verticalAlignment': 'MIDDLE'}}]
    for label in ('Run', 'Notes'):
        if label in positions:
            column = _column(positions[label])
            result.append({'range': f'{column}{row}',
                           'format': {'wrapStrategy': 'CLIP', 'verticalAlignment': 'MIDDLE'}})
    if 'Date' in positions:
        column = _column(positions['Date'])
        result.append({'range': f'{column}{row}',
                       'format': {'horizontalAlignment': 'CENTER', 'verticalAlignment': 'MIDDLE'}})
    return result


def migrate_front_requests(sheet_id, headers, *, header_row=3):
    """Return ``(final_headers, requests)`` without executing a migration.

    The common 29-column prefix is established with native ``moveDimension``;
    missing headers are inserted and written only in their new empty columns.
    All existing data/formula cells move with their columns, and extra columns
    retain their relative order. Existing unnamed columns are preserved; if A is
    named, a fresh blank A is inserted rather than repurposing an unnamed column.
    Sheets owns formula-reference updates; INDIRECT/string references cannot be
    made semantically invariant by any automatic column migration.

    Preconditions: headers describe existing columns; no spanning merges unless
    the caller explicitly handles its own ranges; sufficient physical columns
    for insert ranges (at least ``max(len(headers), 29) + 1`` is sufficient).
    Header values beyond the common prefix are neither rewritten nor removed.
    Repeating this function on its returned headers produces no requests.
    """
    _positive_integer(sheet_id, 'sheet_id', minimum=0)
    _positive_integer(header_row, 'header_row')
    current = _headers(headers)
    requests = []
    for target, label in enumerate(COMMON_HEADERS):
        if target == 0:
            found = 0 if current and current[0] == '' else None
        else:
            found = current.index(label) if label in current else None
        if found is None:
            requests.append({'insertDimension': {
                'range': {'sheetId': sheet_id, 'dimension': 'COLUMNS',
                          'startIndex': target, 'endIndex': target + 1},
                'inheritFromBefore': target > 0}})
            requests.append(_write_headers(sheet_id, header_row, target, [label]))
            current.insert(target, label)
        elif found != target:
            if found < target:
                raise ValueError('Common-prefix migration encountered an inconsistent header order')
            requests.append({'moveDimension': {
                'source': {'sheetId': sheet_id, 'dimension': 'COLUMNS',
                           'startIndex': found, 'endIndex': found + 1},
                'destinationIndex': target}})
            current.insert(target, current.pop(found))
    return current, requests
