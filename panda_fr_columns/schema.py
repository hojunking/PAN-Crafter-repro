"""Physical FR67 layout and reversible mapping to frozen logical fixed64 rows."""
from copy import deepcopy
from directshift48.reporting import HEADERS
from sheet_publish_v2.transport import TransportError, _cell_value, _typed

BOOK = '1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE'
TABS = {261001010: 'WV3-ablations', 261001011: 'QB-ablations', 261001012: 'GF2-ablations'}
EXTRA = ['HQNR (corrected PAN)↑', 'D_s (corrected PAN)↓', 'D_lambda (corrected PAN)↓']
HEADERS67 = list(HEADERS[:7]) + EXTRA + list(HEADERS[7:])
SCHEMA = 'PAN_FR67_V1'


def layout(header):
    if list(header[:67]) == HEADERS67:
        return 67
    if list(header[:64]) == list(HEADERS):
        return 64
    raise TransportError('Unknown Sheet header: refusing positional writes')


def collapse(row):
    return list(row[:7]) + list(row[10:67])


def expand(row, extra=None):
    if len(row) != 64:
        raise TransportError('Logical row must have exactly 64 cells')
    return list(row[:7]) + ([''] * 3 if extra is None else list(extra)) + list(row[7:])


def segments(start, end):
    """Logical [start,end) -> physical pieces, excluding the new H:J cells."""
    if not 0 <= start <= end <= 64:
        raise TransportError('Legacy write outside logical fixed64')
    return [(a, b, a if a < 7 else a + 3, b if b <= 7 else b + 3)
            for a, b in ((start, min(end, 7)), (max(start, 7), end)) if a < b]


def translate(request):
    """Map a known logical request, preserving corrected metrics on old writes."""
    if len(request) != 1:
        raise TransportError('One Sheets action per request required')
    kind, body = next(iter(request.items()))
    body = deepcopy(body)
    if kind == 'appendCells':
        for row in body['rows']:
            row['values'] = expand(row['values'], [{} for _ in range(3)])
        return [{kind: body}]
    if kind in ('updateCells', 'repeatCell'):
        r = body.get('range')
        if r is None or 'startColumnIndex' not in r or 'endColumnIndex' not in r:
            raise TransportError('Explicit column bounds required for legacy writes')
        output = []
        for a, b, c, d in segments(r['startColumnIndex'], r['endColumnIndex']):
            part = deepcopy(body)
            part['range'].update(startColumnIndex=c, endColumnIndex=d)
            if kind == 'updateCells':
                for row in part.get('rows', []):
                    if len(row.get('values', [])) != r['endColumnIndex'] - r['startColumnIndex']:
                        raise TransportError('Legacy cell payload width mismatch')
                    row['values'] = row['values'][a-r['startColumnIndex']:b-r['startColumnIndex']]
            output.append({kind: part})
        return output
    if ((kind == 'appendDimension' and body.get('dimension') == 'ROWS') or
            (kind == 'updateDimensionProperties' and body.get('range', {}).get('dimension') == 'ROWS')):
        return [{kind: body}]
    # Most historical publishers only use the actions above. Never guess at a
    # new mutation (especially clear/replace/sort/delete operations).
    raise TransportError('Unsupported legacy action on FR67: ' + kind)


def migration_requests(tab):
    sid = tab['properties']['sheetId']
    if tab['properties']['title'] != TABS[sid]:
        raise TransportError('Wrong pinned sheet identity')
    def rect(r0, r1, c0, c1):
        return dict(sheetId=sid, startRowIndex=r0, endRowIndex=r1,
                    startColumnIndex=c0, endColumnIndex=c1)
    expected = [rect(3,4,3,7), rect(3,4,7,15)]
    observed = [m for m in tab.get('merges', []) if m.get('startRowIndex') == 3
                and m.get('startColumnIndex', 0) < 15 and m.get('endColumnIndex', 0) > 3]
    if sorted(observed, key=str) != sorted(expected, key=str):
        raise TransportError('FR/RR header merges changed; inspect before migration')
    req = [{'unmergeCells': {'range': rect(3,4,3,15)}},
           {'insertDimension': {'range': dict(sheetId=sid, dimension='COLUMNS', startIndex=7, endIndex=10),
                                'inheritFromBefore': True}}]
    for a,b in ((3,7),(7,10),(10,18)):
        req.append({'mergeCells': {'range': rect(3,4,a,b), 'mergeType':'MERGE_ALL'}})
    req += [{'copyPaste': {'source':rect(4,5,3,6), 'destination':rect(4,5,7,10), 'pasteType':'PASTE_FORMAT'}},
            {'updateCells': {'range':rect(4,5,7,10), 'rows':[{'values':[
                {'userEnteredValue':_typed(v), 'note':'Same prediction and checkpoint as D:F. PAN reference replaced by frozen Aligner corrected PAN. Diagnostic comparison; official selection uses original PAN.'}
                for v in EXTRA]}], 'fields':'userEnteredValue,note'}}]
    for col, label in ((3,'FR · original PAN'),(7,'FR · corrected PAN'),(10,'RR · reconstruction quality')):
        req.append({'updateCells': {'range':rect(3,4,col,col+1),
            'rows':[{'values':[{'userEnteredValue':_typed(label)}]}], 'fields':'userEnteredValue'}})
    req.append({'updateCells': {'range':rect(1,2,0,1), 'rows':[{'values':[{
        'userEnteredValue':_typed('PAN_FR67_V1 · FR corrected PAN: H:J / 결과는 6행부터'),
        'note':'Reporting schema only. Frozen logical fixed64 rows map A:G + K:BO. H:J are corrected-PAN FR metrics. Use panda_fr_columns launcher; old direct uploaders must not write this layout.'}]}],
        'fields':'userEnteredValue,note'}})
    req += [{'repeatCell': {'range':rect(5,tab['properties']['gridProperties']['rowCount'],7,10),
                           'cell':{'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}},
                           'fields':'userEnteredFormat.numberFormat'}},
            {'updateDimensionProperties':{'range':dict(sheetId=sid,dimension='COLUMNS',startIndex=7,endIndex=10),
                                          'properties':{'pixelSize':155},'fields':'pixelSize'}},
            {'updateDimensionProperties':{'range':dict(sheetId=sid,dimension='ROWS',startIndex=4,endIndex=5),
                                          'properties':{'pixelSize':48},'fields':'pixelSize'}}]
    for p in tab.get('protectedRanges', []):
        if p.get('range', {}).get('endRowIndex') == 5:
            req.append({'updateProtectedRange':{'protectedRange':{'protectedRangeId':p['protectedRangeId'],
                'description':'PAN_FR67_V1 — protected header; use compatible reporting launcher'},'fields':'description'}})
    return req
