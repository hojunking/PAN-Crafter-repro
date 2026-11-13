"""Live schema detection and a compatibility view for immutable experiment code."""
from copy import deepcopy
from pathlib import Path
import time
from sheet_publish_v2.transport import (GoogleSheets as BaseGoogleSheets, TransportError,
    _cell_value, _padded, atomic_json, digest)
from .schema import BOOK, TABS, HEADERS67, layout, collapse, translate, migration_requests


class GoogleSheets(BaseGoogleSheets):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.layouts = {}
        self.extra_cells = {}
        self.reuse_once = None
        self.latest_snapshot = None

    def physical_request(self, method, url, **kw):
        return super()._request(method, url, **kw)

    def _request(self, method, url, **kw):
        if method != 'GET':
            if url != self.BASE + BOOK + ':batchUpdate' or set(kw) != {'json'}:
                raise TransportError('Only explicit batchUpdate is supported by the FR adapter')
            payload = deepcopy(kw['json']); requests = []
            for request in payload['requests']:
                body = next(iter(request.values()))
                sid = body.get('sheetId', body.get('range', {}).get('sheetId'))
                if sid in TABS:
                    if sid not in self.layouts:
                        raise TransportError('Snapshot the destination before any legacy write')
                    if self.layouts[sid] == 67:
                        from .schema import expand
                        kind = next(iter(request))
                        rows = body.get('rows', [])
                        ids = [r.get('values',[])[59].get('userEnteredValue',{}).get('stringValue')
                               if len(r.get('values',[]))==64 else None for r in rows]
                        full = kind=='appendCells' or (kind=='updateCells' and
                            body.get('range',{}).get('startColumnIndex')==0 and
                            body.get('range',{}).get('endColumnIndex')==64)
                        if full and rows and all(rid in self.extra_cells for rid in ids):
                            rewritten=deepcopy(body)
                            for row,rid in zip(rewritten['rows'],ids):
                                row['values']=expand(row['values'],self.extra_cells[rid])
                            if kind=='updateCells':rewritten['range']['endColumnIndex']=67
                            requests.append({kind:rewritten})
                        else:requests.extend(translate(request))
                    else:requests.append(request)
                else:
                    raise TransportError('Mutation outside the registered ablation sheets')
            payload['requests'] = requests; kw['json'] = payload
        return self.physical_request(method, url, **kw)

    def metadata(self, destination):
        if destination['spreadsheet_id'] != BOOK or TABS.get(destination['sheet_id']) != destination['sheet_title']:
            raise TransportError('Unregistered destination')
        metadata = self.physical_request('GET', self.BASE+BOOK, params={'fields':
            'spreadsheetId,properties(title),sheets(properties,merges,protectedRanges,basicFilter,tables)'})
        tabs = [s for s in metadata['sheets'] if s['properties']['sheetId'] == destination['sheet_id']]
        if len(tabs) != 1 or tabs[0]['properties']['title'] != destination['sheet_title']:
            raise TransportError('Live destination identity changed')
        return metadata, tabs[0]

    def native(self, destination):
        metadata, tab = self.metadata(destination)
        title = destination['sheet_title']; sid = destination['sheet_id']
        end = 'BO' if tab['properties']['gridProperties']['columnCount'] >= 67 else 'BL'
        count = tab['properties']['gridProperties']['rowCount']
        cells = [[] for _ in range(count-4)]
        hidden = {}
        ranges = [(n,min(count,n+self.page_rows-1)) for n in range(5,count+1,self.page_rows)]
        for offset in range(0,len(ranges),40):
            obj = self.physical_request('GET',self.BASE+BOOK,params={
                'ranges':[f"'{title}'!A{a}:{end}{b}" for a,b in ranges[offset:offset+40]],
                'includeGridData':'true','fields':
                'sheets(data(startRow,startColumn,rowMetadata(hiddenByUser),rowData(values(userEnteredValue,effectiveValue,note,dataValidation,chipRuns,textFormatRuns,userEnteredFormat(numberFormat)))))'})
            for sheet in obj.get('sheets',[]):
                for block in sheet.get('data',[]):
                    start = block.get('startRow',4)-4
                    for i,row in enumerate(block.get('rowData',[])):
                        if 0 <= start+i < len(cells): cells[start+i] = row.get('values',[])
                    for i,row in enumerate(block.get('rowMetadata',[])):
                        if row.get('hiddenByUser'):hidden[str(start+i+5)]=True
        header = [_cell_value(c) for c in cells.pop(0)]
        width = layout(header); self.layouts[sid] = width
        header = header[:width]
        cells = [r[:width] for r in cells]
        last = max((i+1 for i,row in enumerate(cells) if any(_cell_value(c) != '' or
                    any(c.get(k) for k in ('note','dataValidation','chipRuns','textFormatRuns')) for c in row)),default=0)
        cells = cells[:last]
        return {'metadata':metadata,'header':header,'width':width,'grid_row_count':count,'cells':cells,'hidden_rows':hidden,
                'rows':[[ _cell_value(c) for c in r]+['']*(width-len(r)) for r in cells],
                'notes':[r[0].get('note','') if r else '' for r in cells]}

    def snapshot(self, destination):
        if self.reuse_once is not None:
            result=self.reuse_once;self.reuse_once=None
            self.latest_snapshot=result
            return result
        native = self.native(destination)
        if native['width'] == 64:
            self.latest_snapshot=dict(native,header=_padded(native['header']),rows=[_padded(r) for r in native['rows']])
            return self.latest_snapshot
        # Keep an exact physical snapshot for backup/extra-column checks. The
        # immutable legacy validators see a separate reversible 64-column view.
        meta = deepcopy(native['metadata'])
        for tab in meta['sheets']:
            if tab['properties']['sheetId'] != destination['sheet_id']: continue
            tab['properties']['gridProperties']['columnCount'] = 64
            def unmap(r):
                for k in ('startColumnIndex','endColumnIndex'):
                    if k in r: r[k] = r[k] if r[k] <= 7 else max(7,r[k]-3)
                return r
            tab['merges'] = [m for m in map(unmap,tab.get('merges',[]))
                             if m.get('endColumnIndex',64)>m.get('startColumnIndex',0)]
            for item in tab.get('protectedRanges',[])+tab.get('tables',[]):
                if 'range' in item: unmap(item['range'])
            if 'basicFilter' in tab: unmap(tab['basicFilter']['range'])
        for row in native['cells']:
            for c in row[7:10]:
                _cell_value(c)
                if any(c.get(k) for k in ('dataValidation','chipRuns','textFormatRuns')):
                    raise TransportError('Native controls in corrected FR columns; refusing overwrite')
        self.latest_snapshot=dict(native,metadata=meta,header=collapse(native['header']),
                    rows=[collapse(r) for r in native['rows']],
                    cells=[collapse(r+[{}]*(67-len(r))) for r in native['cells']],physical=native)
        return self.latest_snapshot

    def ensure_layout(self, destination, state, *, apply=False):
        before = self.native(destination)
        if before['width'] == 67: return {'action':'SCHEMA_ALREADY_FR67','readback_verified':True}
        tab = next(t for t in before['metadata']['sheets'] if t['properties']['sheetId']==destination['sheet_id'])
        requests = migration_requests(tab)
        state=Path(state); stamp=str(time.time_ns())
        atomic_json(state/('schema_before_'+stamp+'.json'),before)
        atomic_json(state/'schema_requests.json',{'requests':requests,'destination':destination})
        if not apply: return {'action':'SCHEMA_MIGRATION_PLANNED','requests':requests}
        again=self.native(destination)
        if digest(before)!=digest(again): raise TransportError('Sheet changed while planning schema; retry fresh')
        self.physical_request('POST',self.BASE+BOOK+':batchUpdate',json={'requests':requests})
        after=self.native(destination)
        if after['width']!=67 or [collapse(r) for r in after['rows']]!=before['rows'] or after['notes']!=before['notes']:
            raise TransportError('Schema readback changed existing records; inspect backup')
        receipt={'action':'SCHEMA_FR67_VERIFIED','readback_verified':True,'destination':destination,
                 'before_sha256':digest(before),'after_sha256':digest(after)}
        atomic_json(state/'schema_verified.json',receipt)
        return receipt
