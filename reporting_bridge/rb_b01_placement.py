"""Pure append-only placement planning for an existing ablations worksheet.

No network calls, cell writes, worksheet creation, or research imports. The
caller must supply a complete bounded grid read, its explicit read bounds, and
fresh workbook metadata. Every coordinate returned here is 1-based/inclusive;
Google GridRange objects retain their native 0-based/exclusive convention.
"""
from __future__ import annotations

import hashlib
import json
import re

OWNER = 'B01_EXISTING_SHEETS_APPEND_ONLY_v2'
CAMPAIGN = 'PANDA_REBUTTAL_B01_WV3_S135_20260928_v1'
DEFAULT_SCHEMA = 'B01_ABLATIONS_BLOCK_v2'
TITLE = 'ablations'
SHEET_ID = 1697831403
BLOCK_TITLE = 'Rebuttal B01 — RB01 q reliability / RB02 controlled shifts'
CELL_CONTENT_FIELDS = ('userEnteredValue', 'effectiveValue', 'note', 'dataValidation',
                       'chipRuns', 'textFormatRuns', 'pivotTable', 'dataSourceTable',
                       'dataSourceFormula', 'dataSourcePivotTable', 'hyperlink')
ARRAY_FUNCTIONS = frozenset(('ARRAYFORMULA', 'FILTER', 'SORT', 'SORTN', 'UNIQUE',
    'QUERY', 'SEQUENCE', 'TRANSPOSE', 'VSTACK', 'HSTACK', 'WRAPROWS', 'WRAPCOLS',
    'TOCOL', 'TOROW', 'MAKEARRAY', 'SPLIT', 'FLATTEN', 'IMPORTRANGE', 'IMPORTDATA',
    'IMPORTXML', 'IMPORTHTML', 'GOOGLEFINANCE', 'CHOOSECOLS', 'CHOOSEROWS',
    'EXPAND', 'TAKE', 'DROP', 'MAP', 'BYROW', 'BYCOL', 'SCAN'))


class PlacementError(ValueError):
    """Fail closed with an actionable conflict; never move user rows."""


def _require(condition, message):
    if not condition:
        raise PlacementError(message)


def _integer(value, label, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             'INVALID_RANGE: ' + label)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[a-f0-9]{64}', value))


def _column(index):
    value = ''
    while index:
        index, part = divmod(index - 1, 26)
        value = chr(65 + part) + value
    return value


def _column_number(label):
    result = 0
    for char in label:
        result = 26 * result + ord(char) - 64
    return result


def _a1(r0, r1, c0, c1):
    return f"'{TITLE}'!{_column(c0 + 1)}{r0 + 1}:{_column(c1)}{r1}"


def _parse_a1(value):
    match = re.fullmatch(r"(?:'ablations'|ablations)!([A-Z]+)([1-9][0-9]*):([A-Z]+)([1-9][0-9]*)", value or '')
    _require(match is not None, 'OWNER_MARKER_INVALID: bounded ablations A1 range required')
    c0, r0, c1, r1 = match.groups()
    region = dict(sheetId=SHEET_ID, startRowIndex=int(r0) - 1, endRowIndex=int(r1),
                  startColumnIndex=_column_number(c0) - 1, endColumnIndex=_column_number(c1))
    _require(region['startRowIndex'] < region['endRowIndex'] and region['startColumnIndex'] < region['endColumnIndex'],
             'OWNER_MARKER_INVALID: empty or reversed range')
    return region


def _intersects(a, b):
    return (a['startRowIndex'] < b['endRowIndex'] and b['startRowIndex'] < a['endRowIndex']
            and a['startColumnIndex'] < b['endColumnIndex'] and b['startColumnIndex'] < a['endColumnIndex'])


def _contains(outer, inner):
    return (outer['startRowIndex'] <= inner['startRowIndex'] and inner['endRowIndex'] <= outer['endRowIndex']
            and outer['startColumnIndex'] <= inner['startColumnIndex'] and inner['endColumnIndex'] <= outer['endColumnIndex'])


def _cell_region(row, col):
    return dict(sheetId=SHEET_ID, startRowIndex=row, endRowIndex=row + 1,
                startColumnIndex=col, endColumnIndex=col + 1)


def _split_arguments(value, separator=','):
    """Tokenize only enough formula syntax to prove a conservative size bound."""
    result, start, depth, brace, quote = [], 0, 0, 0, None
    i = 0
    while i < len(value):
        char = value[i]
        if quote:
            if char == quote:
                if i + 1 < len(value) and value[i + 1] == quote:
                    i += 2; continue
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == '{':
            brace += 1
        elif char == '}':
            brace -= 1
        elif char == separator and depth == 0 and brace == 0:
            result.append(value[start:i].strip()); start = i + 1
        _require(depth >= 0 and brace >= 0, 'FORMULA_EXTENT_REQUIRED: malformed formula')
        i += 1
    _require(quote is None and depth == 0 and brace == 0, 'FORMULA_EXTENT_REQUIRED: unbalanced formula')
    result.append(value[start:].strip())
    return result


def _shape_bound(expression):
    """Return a proven (rows, columns) bound or None, never a guessed spill."""
    value = expression.strip().lstrip('=').strip()
    # A bare bounded A1 source is sufficient to prove CHOOSECOLS row extent.
    match = re.fullmatch(r"(?:(?:'(?:[^']|'')+'|[A-Za-z_][A-Za-z_0-9 .]*)!)?\$?([A-Z]+)\$?([1-9][0-9]*)(?::\$?([A-Z]+)\$?([1-9][0-9]*))?", value)
    if match:
        ca, ra, cb, rb = match.groups(); cb, rb = cb or ca, rb or ra
        height, width = int(rb) - int(ra) + 1, _column_number(cb) - _column_number(ca) + 1
        return (height, width) if height > 0 and width > 0 else None
    if not value or re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|TRUE|FALSE', value, re.I):
        return 1, 1
    if re.fullmatch(r'"(?:[^"]|"")*"', value):
        return 1, 1
    if value.startswith('{') and value.endswith('}'):
        literal_rows = _split_arguments(value[1:-1], ';')
        shapes = [[_shape_bound(item) for item in _split_arguments(row)] for row in literal_rows]
        if any(any(shape is None for shape in row) for row in shapes):
            return None
        # Stacking bounded array-valued children also has a safe finite bound.
        height = sum(max(shape[0] for shape in row) for row in shapes)
        width = max(sum(shape[1] for shape in row) for row in shapes) if all(shape[1] is not None for row in shapes for shape in row) else None
        return height, width
    call = re.fullmatch(r'([A-Za-z_][A-Za-z_.0-9]*)\((.*)\)', value, re.S)
    if not call:
        return None
    name, arguments = call.groups(); name = name.upper().removeprefix('_XLFN.')
    args = _split_arguments(arguments)
    shapes = [_shape_bound(arg) for arg in args]
    if name in ('CHOOSECOLS', 'CHOOSEROWS'):
        if not shapes or shapes[0] is None:
            return None
        # Each selector must be a literal scalar; array indices may duplicate
        # arbitrary rows/columns and cannot use the scalar count as a bound.
        if len(args) < 2 or not all(re.fullmatch(r'-?[1-9][0-9]*', x) for x in args[1:]):
            return None
        return (shapes[0][0], len(args) - 1) if name == 'CHOOSECOLS' else (len(args) - 1, shapes[0][1])
    if name in ('FILTER', 'SORT', 'SORTN', 'UNIQUE', 'ARRAYFORMULA', 'TAKE', 'DROP'):
        return shapes[0] if shapes else None
    if name in ('IFERROR', 'IFNA', 'IF'):
        relevant = shapes[1:] if name == 'IF' else shapes
        # IF has array-valued conditions as well as result expressions.
        if name == 'IF': relevant = shapes
        if relevant and all(shape is not None for shape in relevant):
            return max(x[0] for x in relevant), max(x[1] for x in relevant) if all(x[1] is not None for x in relevant) else None
        return None
    if name in ('SEQUENCE', 'MAKEARRAY'):
        if not args or not re.fullmatch(r'[1-9][0-9]*', args[0]):
            return None
        cols = 1 if len(args) < 2 else int(args[1]) if re.fullmatch(r'[1-9][0-9]*', args[1]) else None
        return (int(args[0]), cols) if cols is not None else None
    if name == 'TRANSPOSE' and shapes and shapes[0] is not None and shapes[0][1] is not None:
        return shapes[0][1], shapes[0][0]
    if name == 'SPLIT' and shapes and shapes[0] is not None and shapes[0][0] == 1:
        # Width depends on cell contents and is unknown; TRANSPOSE must not
        # treat an invented finite width as a proven vertical bound.
        return 1, None
    if name in ('VSTACK', 'HSTACK') and shapes and all(x is not None for x in shapes):
        known_widths = all(x[1] is not None for x in shapes)
        return ((sum(x[0] for x in shapes), max(x[1] for x in shapes) if known_widths else None) if name == 'VSTACK'
                else (max(x[0] for x in shapes), sum(x[1] for x in shapes) if known_widths else None))
    return None


def _array_formula(formula):
    # Ignore quoted prose and quoted sheet names when detecting function names.
    clean = re.sub(r'"(?:[^"]|"")*"', '""', formula)
    names = {m.upper() for m in re.findall(r'\b([A-Za-z_][A-Za-z_.0-9]*)\s*\(', clean)}
    return bool(names & ARRAY_FUNCTIONS) or '{' in clean


def _inputs(grid, metadata):
    sheets = [s for s in metadata.get('sheets', []) if s.get('properties', {}).get('title') == TITLE]
    _require(len(sheets) == 1 and sheets[0]['properties'].get('sheetId') == SHEET_ID,
             'TARGET_MISSING_OR_SCHEMA_CONFLICT: ablations title/ID')
    sheet = sheets[0]; shape = sheet['properties']['gridProperties']
    rows = _integer(shape['rowCount'], 'sheet rowCount', 1)
    cols = _integer(shape['columnCount'], 'sheet columnCount', 1)
    bounds = grid.get('_read_bounds')
    expected = dict(sheetId=SHEET_ID, startRowIndex=0, endRowIndex=rows,
                    startColumnIndex=0, endColumnIndex=cols)
    _require(bounds == expected, 'FULL_GRID_READ_REQUIRED: all existing ablations rows AND columns with explicit _read_bounds')
    target = [s for s in grid.get('sheets', []) if s.get('properties', {}).get('sheetId') == SHEET_ID]
    _require(len(target) == 1, 'FULL_GRID_READ_REQUIRED: missing/duplicate target grid')
    cells = {}
    for block in target[0].get('data', []):
        r0, c0 = block.get('startRow', 0), block.get('startColumn', 0)
        for ri, row in enumerate(block.get('rowData', []), r0):
            for ci, cell in enumerate(row.get('values', []), c0):
                _require(0 <= ri < rows and 0 <= ci < cols, 'FULL_GRID_READ_REQUIRED: returned cell outside read bounds')
                if not cell:
                    continue
                _require((ri, ci) not in cells or cells[(ri, ci)] == cell, 'CONFLICTING_GRID_BLOCKS: duplicate cell')
                cells[(ri, ci)] = cell
    return sheet, target[0], cells, rows, cols


def _normal_range(value, rows, cols, *, label):
    _require(value.get('sheetId') == SHEET_ID, 'INVALID_RANGE: wrong sheet ID in ' + label)
    r0 = _integer(value.get('startRowIndex', 0), label + ' start row')
    r1 = _integer(value.get('endRowIndex', rows), label + ' end row', 1)
    c0 = _integer(value.get('startColumnIndex', 0), label + ' start column')
    c1 = _integer(value.get('endColumnIndex', cols), label + ' end column', 1)
    _require(r0 < r1 and c0 < c1, 'INVALID_RANGE: empty ' + label)
    return dict(sheetId=SHEET_ID, startRowIndex=r0, endRowIndex=r1,
                startColumnIndex=c0, endColumnIndex=c1)


def _reservations(sheet, grid_sheet, metadata, rows, cols):
    items = []
    for kind, sources in (
            ('merge', [dict(range=x) for x in sheet.get('merges', grid_sheet.get('merges', []))]),
            ('protected', sheet.get('protectedRanges', [])),
            ('table', sheet.get('tables', grid_sheet.get('tables', []))),
            ('basic_filter', [sheet.get('basicFilter', grid_sheet.get('basicFilter'))] if sheet.get('basicFilter', grid_sheet.get('basicFilter')) else []),
            ('filter_view', sheet.get('filterViews', grid_sheet.get('filterViews', []))),
            ('named', metadata.get('namedRanges', [])),
            ('reserved', metadata.get('reservedRanges', [])),
            ('array_spill', metadata.get('arraySpillRanges', []))):
        for source in sources:
            value = source.get('range', source)
            if value.get('sheetId') != SHEET_ID:
                continue
            region = _normal_range(value, rows, cols, label=kind)
            items.append(dict(kind=kind, range=region, source=source,
                              unbounded_rows='endRowIndex' not in value))
    return items


def _owner_marker(note, row, col, rows, cols):
    if not isinstance(note, str) or OWNER not in note:
        return None
    try:
        value = json.loads(note)
    except (ValueError, TypeError) as exc:
        raise PlacementError('OWNER_MARKER_INVALID: non-JSON owner note') from exc
    _require(isinstance(value, dict) and value.get('owner') == OWNER, 'OWNER_MARKER_INVALID: owner field')
    for key in ('campaign_id', 'section_id', 'schema', 'range'):
        _require(isinstance(value.get(key), str) and bool(value[key]), 'OWNER_MARKER_INVALID: missing ' + key)
    _require(_valid_sha(value.get('source_hash')), 'OWNER_MARKER_INVALID: source hash')
    sequence = _integer(value.get('sequence', 1), 'owner sequence', 1)
    region = _parse_a1(value['range'])
    _require((row, col) == (region['startRowIndex'], region['startColumnIndex']), 'OWNER_MARKER_INVALID: marker not at range start')
    _require(region['endRowIndex'] <= rows and region['endColumnIndex'] <= cols,
             'OWNER_MARKER_INVALID: range beyond current grid')
    return dict(marker=value, range=region, sequence=sequence, marker_row=row + 1, marker_col=col + 1)


def discover(grid, metadata):
    """Inventory all cells, explicit reservations and provable array extents."""
    sheet, grid_sheet, cells, rows, cols = _inputs(grid, metadata)
    reserved = _reservations(sheet, grid_sheet, metadata, rows, cols)
    markers, occupied, formulas, unresolved = [], [], [], []
    last_used = 0
    for (row, col), cell in sorted(cells.items()):
        if any(cell.get(key) not in (None, '', {}, []) for key in CELL_CONTENT_FIELDS):
            occupied.append(dict(row=row + 1, column=col + 1,
                                 fields=[key for key in CELL_CONTENT_FIELDS if cell.get(key) not in (None, '', {}, [])]))
            last_used = max(last_used, row + 1)
        marker = _owner_marker(cell.get('note'), row, col, rows, cols)
        displayed = [cell.get(key, {}).get('stringValue') for key in ('userEnteredValue', 'effectiveValue')]
        if any(isinstance(value, str) and value.startswith(BLOCK_TITLE) for value in displayed):
            _require(marker is not None, 'SCHEMA_CONFLICT: B01 title exists without its valid owner marker')
        if marker:
            markers.append(marker)
            last_used = max(last_used, marker['range']['endRowIndex'])
        formula = cell.get('userEnteredValue', {}).get('formulaValue')
        dynamic_table = next((key for key in ('pivotTable', 'dataSourceTable', 'dataSourcePivotTable') if cell.get(key)), None)
        if (formula and _array_formula(formula)) or dynamic_table:
            anchor = _cell_region(row, col)
            declared = [item for item in reserved if item['kind'] in ('reserved', 'array_spill')
                        and _contains(item['range'], anchor)]
            shape = _shape_bound(formula) if formula else None
            if declared:
                end = max(item['range']['endRowIndex'] for item in declared)
                last_used = max(last_used, end)
                formulas.append(dict(anchor=_a1(row, row + 1, col, col + 1), height_bound=end - row,
                                     proof='explicit reservation'))
            elif shape is not None:
                end = row + shape[0]
                last_used = max(last_used, end)
                formulas.append(dict(anchor=_a1(row, row + 1, col, col + 1), height_bound=shape[0],
                                     proof='bounded formula structure'))
            else:
                unresolved.append(dict(anchor=_a1(row, row + 1, col, col + 1), formula=formula, dynamic_table=dynamic_table))
    for item in reserved:
        last_used = max(last_used, item['range']['endRowIndex'])
    return dict(sheet_id=SHEET_ID, title=TITLE, sheet_rows=rows, sheet_columns=cols,
                last_used_or_reserved_row=last_used, occupied_cells=occupied, reservations=reserved,
                owner_markers=markers, array_formula_extents=formulas, unresolved_array_formulas=unresolved,
                unbounded_reservations=[r for r in reserved if r['unbounded_rows']])


def outside_owned_snapshot(grid, metadata, owned_range=None):
    """Normalize everything outside ONLY the currently writable owned block.

    Prior continuation blocks are intentionally not excluded. Structural items
    crossing the writable boundary remain in the snapshot and are not editable.
    All populated CellData fields (including formatting) are retained; trailing
    absent cells/rows are ignored. Row-capacity growth itself is permitted.
    """
    sheet, grid_sheet, cells, rows, cols = _inputs(grid, metadata)
    owned = _parse_a1(owned_range) if isinstance(owned_range, str) else owned_range
    outside = {f'{r + 1}:{c + 1}': cell for (r, c), cell in sorted(cells.items())
               if owned is None or not _contains(owned, _cell_region(r, c))}
    reservations = _reservations(sheet, grid_sheet, metadata, rows, cols)
    protected = [item for item in reservations if owned is None or not _contains(owned, item['range'])]
    # Preserve row/column metadata and layout fields supplied by the adapter,
    # but not the extendable total rowCount.
    properties = dict(sheet['properties'])
    properties['gridProperties'] = {k: v for k, v in properties.get('gridProperties', {}).items() if k != 'rowCount'}
    result = dict(sheet_id=SHEET_ID, properties=properties, cells=outside, reservations=protected,
                  excluded_owned_range=owned_range,
                  row_metadata=[dict(startRow=b.get('startRow', 0), rowMetadata=b['rowMetadata'])
                                for b in grid_sheet.get('data', []) if b.get('rowMetadata')],
                  column_metadata=[dict(startColumn=b.get('startColumn', 0), columnMetadata=b['columnMetadata'])
                                   for b in grid_sheet.get('data', []) if b.get('columnMetadata')])
    result['snapshot_sha256'] = _digest(result)
    return result


def _chain(markers, campaign_id, section_id, schema):
    selected = [item for item in markers if item['marker']['section_id'] == section_id]
    for item in selected:
        _require(item['marker']['campaign_id'] == campaign_id, 'OWNER_CONFLICT: same section belongs to a different campaign')
        _require(item['marker']['schema'] == schema, 'SCHEMA_CONFLICT: existing owned block schema')
    selected.sort(key=lambda item: item['sequence'])
    if not selected:
        return selected
    _require([item['sequence'] for item in selected] == list(range(1, len(selected) + 1)),
             'OWNER_CONFLICT: duplicate primary, missing or duplicate continuation sequence')
    primary = selected[0]['marker']
    _require(not primary.get('continuation_of') and not primary.get('previous_range'),
             'OWNER_CONFLICT: primary claims continuation')
    for previous, current in zip(selected, selected[1:]):
        marker = current['marker']
        _require(marker.get('continuation_of') == primary['range']
                 and marker.get('previous_range') == previous['marker']['range'],
                 'OWNER_CONFLICT: broken continuation chain')
        _require(current['range']['startRowIndex'] >= previous['range']['endRowIndex'] + 2,
                 'OWNER_CONFLICT: continuation overlaps prior block or separator')
    return selected


def plan_placement(grid, metadata, row_count, width=31, *, source_hash,
                   campaign_id=CAMPAIGN, section_id='B01', schema=DEFAULT_SCHEMA):
    """Plan an append, reuse, or full continuation without changing any cells.

    ``row_count`` is the required RESERVED BLOCK HEIGHT (not sheet rowCount).
    ``width`` is the complete owned block width. Reuse never expands in place;
    insufficient space produces a full continuation below all current content.
    """
    height = _integer(row_count, 'requested block height', 1)
    width = _integer(width, 'requested block width', 1)
    _require(_valid_sha(source_hash), 'INVALID_SOURCE_HASH: full SHA256 required')
    inventory = discover(grid, metadata)
    _require(width <= inventory['sheet_columns'], 'TARGET_COLUMNS_INSUFFICIENT: column expansion is not authorized')
    _require(not inventory['unresolved_array_formulas'], 'ARRAY_SPILL_EXTENT_REQUIRED: ' +
             ', '.join(x['anchor'] for x in inventory['unresolved_array_formulas']))
    chain = _chain(inventory['owner_markers'], campaign_id, section_id, schema)
    active = chain[-1] if chain else None
    previous_ranges = [x['marker']['range'] for x in chain]
    reuse = bool(active and height <= active['range']['endRowIndex'] - active['range']['startRowIndex']
                 and width == active['range']['endColumnIndex'] - active['range']['startColumnIndex'])
    if reuse:
        region = active['range']
        _require(region['startColumnIndex'] == 0, 'OWNER_CONFLICT: B01 block must start in column A')
        mode = 'REUSE'; sequence = active['sequence']
        previous_ranges = previous_ranges[:-1]
    else:
        _require(not inventory['unbounded_reservations'], 'UNBOUNDED_RESERVED_RANGE: cannot append beyond whole-column/sheet reservation')
        start = inventory['last_used_or_reserved_row'] + 2  # 0-based; leave exactly two blank rows.
        region = dict(sheetId=SHEET_ID, startRowIndex=start, endRowIndex=start + height,
                      startColumnIndex=0, endColumnIndex=width)
        mode = 'CONTINUATION' if active else 'APPEND'
        sequence = len(chain) + 1
    for other in inventory['owner_markers']:
        if active is other and reuse:
            continue
        _require(not _intersects(region, other['range']), 'OWNER_CONFLICT: intersects another owned block')
    for item in inventory['reservations']:
        if not _intersects(region, item['range']):
            continue
        # A merge wholly inside our block is owned formatting. User named,
        # protected and explicitly reserved regions still require review.
        if item['kind'] == 'merge' and reuse and _contains(region, item['range']):
            continue
        _require(False, 'PLACEMENT_RESERVED_RANGE_CONFLICT: ' + item['kind'])
    if reuse:
        # A formula inside may be ours, but its inferred spill cannot exceed
        # the owned range. Outside anchors may not spill into our region.
        for item in inventory['array_formula_extents']:
            anchor = _parse_a1(item['anchor'])
            extent = dict(anchor, endRowIndex=anchor['startRowIndex'] + item['height_bound'], endColumnIndex=inventory['sheet_columns'])
            if _intersects(region, extent):
                _require(_contains(region, extent), 'ARRAY_SPILL_CROSSES_OWNERSHIP: ' + item['anchor'])
    coordinate = _a1(region['startRowIndex'], region['endRowIndex'], region['startColumnIndex'], region['endColumnIndex'])
    marker = dict(owner=OWNER, campaign_id=campaign_id, section_id=section_id, schema=schema,
                  range=coordinate, source_hash=source_hash, sequence=sequence)
    if sequence > 1:
        marker['continuation_of'] = chain[0]['marker']['range']
        marker['previous_range'] = chain[-2]['marker']['range'] if reuse else active['marker']['range']
    outside = outside_owned_snapshot(grid, metadata, coordinate)
    return dict(schema='B01_PLACEMENT_PLAN_v2', sheet_id=SHEET_ID, title=TITLE, mode=mode,
                start_row=region['startRowIndex'] + 1, end_row=region['endRowIndex'],
                start_col=region['startColumnIndex'] + 1, end_col=region['endColumnIndex'],
                range=coordinate, grid_range=region, requested_rows=height,
                capacity_rows=region['endRowIndex'] - region['startRowIndex'],
                required_grid_rows=max(inventory['sheet_rows'], region['endRowIndex']),
                append_grid_rows=max(0, region['endRowIndex'] - inventory['sheet_rows']),
                previous_ranges=previous_ranges, prior_ranges=previous_ranges,
                owner_marker=marker, marker_note=json.dumps(marker, sort_keys=True, ensure_ascii=False, separators=(',', ':')),
                new_sheet_count=0, historical_last_used_row=inventory['last_used_or_reserved_row'],
                protected_outside_sha256=outside['snapshot_sha256'])
