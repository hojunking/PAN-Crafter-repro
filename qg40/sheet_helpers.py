"""QG40-local structural guards for existing Sheet cells.

These helpers have no connection or mutation side effects. Their behavior is
independent of the separately deployed historical reporting worker.
"""
from fh12.upload import column


def fetch_controls(ws, rows):
    """Read formulas, validation, chips, merges and protected/typed regions."""
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
    sheets = [sheet for sheet in meta.get('sheets', [])
              if sheet.get('properties', {}).get('sheetId') == int(ws.id)]
    if len(sheets) != 1:
        raise ValueError('Cannot verify target Sheet controls')
    return dict(sheet=sheets[0], named_ranges={v['namedRangeId']: v['range']
                                             for v in meta.get('namedRanges', [])})


def _covers(region, sheet_id, row, col):
    return (region.get('sheetId', sheet_id) == sheet_id and
            region.get('startRowIndex', 0) <= row - 1 < region.get('endRowIndex', float('inf')) and
            region.get('startColumnIndex', 0) <= col - 1 < region.get('endColumnIndex', float('inf')))


def controlled_reason(controls, row, col):
    sheet = controls['sheet']
    sid = sheet['properties']['sheetId']
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
