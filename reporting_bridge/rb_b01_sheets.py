"""Explicit, single-writer B01 Sheets bridge; importing never authenticates.

The adapter intentionally exposes only bounded cell/grid operations. Numeric
research modules and their source identities are not imported by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

SCHEMA = 'RB_B01_SHEETS_BRIDGE_v1'
CAMPAIGN = 'PANDA_REBUTTAL_B01_WV3_S135_20260928_v1'
SHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'
PROTECTED = ('paper', 'ablations', '유의미한결과')
VIEW_NAMES = ('WV3-main', 'QB-main', 'GF2-main', 'WV3-archive', 'QB-archive',
              'GF2-archive', 'WV2-archive', 'PC-Repro')
FIXED_IDS = {'paper': 1923997092, 'ablations': 1697831403,
             '유의미한결과': 410652761, '_records': 260928001,
             'WV3-main': 260928010}
NATIVE_TABS = ('_rb01_s1', '_rb01_s3', '_rb01_s5')
# Public view only. Raw source keeps its complete canonical 43-column schema.
_CURVE_READABLE = (4,5,6,10,12,13,18,19,20,21,16,17,23,22,11,14,15,
                   24,25,26,27,28,29,30,31,32,33)
CURVE_VIEW_COLUMNS = _CURVE_READABLE + tuple(i for i in range(43) if i not in _CURVE_READABLE)
OLD_RETURN = 'FILTER(allrows,CHOOSECOLS(allrows,1)<>"")'
NEW_RETURN = ('LET(rbload,LAMBDA(tab,IFNA(FILTER(INDIRECT("\'"&tab&"\'!A2:BL"),'
              'INDIRECT("\'"&tab&"\'!BH2:BH")<>""),MAKEARRAY(1,64,LAMBDA(rr,cc,"")))),'
              'joined,VSTACK(allrows,rbload("_rb01_s1"),rbload("_rb01_s3"),'
              'rbload("_rb01_s5")),FILTER(joined,CHOOSECOLS(joined,1)<>""))')


class BridgeError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def formula_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _contract():
    from reporting_bridge import rb_b01_contract
    return rb_b01_contract


def column(index):
    result = ''
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def quoted(title):
    return "'" + title.replace("'", "''") + "'"


def same(actual, expected):
    if actual is None: actual = ''
    if expected is None: expected = ''
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual,(int,float)) and not isinstance(actual,bool) and actual==expected
    if isinstance(expected, float):
        return (isinstance(actual, (float, int)) and math.isfinite(actual)
                and math.isfinite(expected) and
                abs(actual - expected) <= 1e-12 * max(1, abs(expected)))
    return actual == expected


def rows_equal(actual, expected):
    return all(same(actual[i] if i < len(actual) else '', value)
               for i, value in enumerate(expected)) and not any(actual[len(expected):])


def _without_layout(grid):
    """Compare cells independent of Google omitting empty trailing rows/cells."""
    result = {}
    for sheet in grid.get('sheets', []):
        sid = sheet.get('properties', {}).get('sheetId')
        data = {}
        for block in sheet.get('data', []):
            for i, row in enumerate(block.get('rowData', [])):
                for j, cell in enumerate(row.get('values', [])):
                    # effective/formatted values are not user edits and can be volatile.
                    value = {k: cell[k] for k in ('userEnteredValue', 'userEnteredFormat',
                                                 'note', 'dataValidation', 'textFormatRuns') if k in cell}
                    if value:
                        data[f"{block.get('startRow',0)+i}:{block.get('startColumn',0)+j}"] = value
        result[str(sid)] = {'cells': data, 'merges': sheet.get('merges', [])}
    return result


def cell_from_grid(grid):
    for sheet in grid.get('sheets', []):
        for block in sheet.get('data', []):
            for row in block.get('rowData', []):
                if row.get('values'): return row['values'][0]
    return {}


def patch_formula(formula):
    """Replace only the terminal LET expression, retaining legacy code verbatim."""
    if NEW_RETURN in formula:
        if formula.count(NEW_RETURN) != 1 or any(formula.count('rbload("'+t+'")') != 1 for t in NATIVE_TABS):
            raise BridgeError('FORMULA_REVIEW_REQUIRED: duplicate bridge union')
        suffix = formula.split(NEW_RETURN, 1)[1]
        if any(c not in ') \r\n\t' for c in suffix):
            raise BridgeError('FORMULA_REVIEW_REQUIRED: bridge is not terminal return')
        return formula
    if any(t in formula for t in NATIVE_TABS) or 'rbload' in formula:
        raise BridgeError('FORMULA_REVIEW_REQUIRED: unrecognized existing bridge')
    if formula.count(OLD_RETURN) != 1:
        raise BridgeError('FORMULA_REVIEW_REQUIRED: terminal legacy expression not unique')
    left, right = formula.split(OLD_RETURN)
    if any(c not in ') \r\n\t' for c in right):
        raise BridgeError('FORMULA_REVIEW_REQUIRED: legacy expression is not terminal')
    return left + NEW_RETURN + right


def observation_map(rows):
    mapped = {}
    for row in rows:
        rid = row[59] if len(row) > 59 else ''
        if not rid: continue  # display grouping/header rows are not observations
        if rid == 'Result_ID': continue
        if rid in mapped: raise BridgeError('Duplicate historical Result_ID: ' + str(rid))
        mapped[rid] = (list(row) + [''] * 64)[:64]
    return mapped


def _errors(rows):
    return sorted({str(value) for row in rows for value in row
                   if isinstance(value, str) and value.startswith(('#REF!', '#VALUE!', '#ERROR!', '#DIV/0!', '#N/A', '#NAME?', '#NUM!'))})


class GoogleSheetsAdapter:
    """Lazy existing service-account auth, opening ONLY the approved document ID."""
    def __init__(self, spreadsheet_id=SHEET_ID, root=None):
        if spreadsheet_id != SHEET_ID:
            raise BridgeError('Unexpected spreadsheet ID')
        self.spreadsheet_id = spreadsheet_id
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self._book = None
        self._metadata_cache = None

    @property
    def book(self):
        if self._book is None:
            import gspread
            # Reuse the existing credential without importing an old uploader.
            client = gspread.service_account(filename=str(self.root / 'gspread' / 'account.json'))
            client.http_client.set_timeout(30)
            self._book = client.open_by_key(self.spreadsheet_id)
        return self._book

    def metadata(self):
        self._metadata_cache=self._read(lambda:self.book.fetch_sheet_metadata(params={'fields': 'spreadsheetId,properties,sheets(properties,merges,protectedRanges)'}))
        return self._metadata_cache

    def grid(self, ranges):
        return self._read(lambda:self.book.fetch_sheet_metadata(params={'includeGridData': True, 'ranges': [self._bounded(r) for r in ranges],
            'fields': 'sheets(properties(sheetId,title),merges,data(startRow,startColumn,rowData(values(userEnteredValue,userEnteredFormat,note,dataValidation,textFormatRuns))))'}))

    def values(self, range_name):
        return self._read(lambda:self.book.values_get(self._bounded(range_name), params={'valueRenderOption': 'UNFORMATTED_VALUE',
                              'dateTimeRenderOption': 'SERIAL_NUMBER'})).get('values', [])

    @staticmethod
    def _read(call):
        for attempt in range(3):
            try:return call()
            except Exception as exc:
                status=getattr(getattr(exc,'response',None),'status_code',None)
                transient=status in (429,500,502,503,504) or isinstance(exc,TimeoutError) or type(exc).__name__ in ('ReadTimeout','ConnectTimeout','ConnectionError')
                if not transient or attempt==2:raise
                time.sleep(2**attempt)

    def batch_update(self, requests):
        result=self.book.batch_update({'requests': requests})
        if any(any(k in request for k in ('addSheet','appendDimension','updateSheetProperties')) for request in requests):
            self._metadata_cache=None
        return result

    def write_raw(self, entries):
        if entries:
            return self.book.values_batch_update({'valueInputOption': 'RAW',
                'data': [{'range': name, 'values': values} for name, values in entries]})

    def _bounded(self, value):
        if '!' in value and not re.search(r':[A-Z]+$',value): return value
        title=value.split('!',1)[0].strip("'").replace("''", "'")
        props=next(s['properties'] for s in (self._metadata_cache or self.metadata())['sheets'] if s['properties']['title']==title)
        shape=props['gridProperties']
        if '!' not in value:return value+'!A1:'+column(shape['columnCount'])+str(shape['rowCount'])
        return value+str(shape['rowCount'])

    def local_writer_server(self):
        path=self.root/'work_dir/_panda_rb/local_owner.json'
        if not path.is_file():raise BridgeError('Missing local B01 writer ownership record')
        return json.loads(path.read_text())['server']


class SheetsBridge:
    def __init__(self, adapter, progress_callback=None):
        self.adapter = adapter
        self.progress_callback = progress_callback
        self.spreadsheet_id = adapter.spreadsheet_id
        if self.spreadsheet_id != SHEET_ID: raise BridgeError('Unexpected spreadsheet ID')

    def _progress(self,state,details):
        if self.progress_callback is not None:self.progress_callback(state,details)

    def _tabs(self):
        metadata = self.adapter.metadata()
        if metadata.get('spreadsheetId', self.spreadsheet_id) != self.spreadsheet_id:
            raise BridgeError('Returned document ID differs')
        tabs = {s['properties']['title']: s['properties'] for s in metadata['sheets']}
        for title, sid in FIXED_IDS.items():
            if tabs.get(title, {}).get('sheetId') != sid:
                raise BridgeError('SCHEMA_CONFLICT: fixed tab ID/title ' + title)
        return metadata, tabs

    def _schemas(self):
        c = _contract()
        return {**{t: list(c.NATIVE_HEADERS) for t in NATIVE_TABS},
                '_rb02_points': list(c.STRESS_HEADERS), '_rb_b01_status': list(c.STATUS_HEADERS),
                'RB02-curves': [c.STRESS_HEADERS[i] for i in CURVE_VIEW_COLUMNS],
                'RB-B01': ['Panel', 'Cohort', 'Selection / Mode', 'Case', 'Metric / Radius',
                           'N', 'Mean / Measured', 'Sample SD / Expected', 'Missing / Failed', 'Provenance']}

    def _owner_note(self, title):
        return json.dumps({'bridge': SCHEMA, 'campaign_id': CAMPAIGN, 'title': title,
                           'header_sha256': digest(self._schemas()[title])}, sort_keys=True)

    def _owned(self, title, tabs):
        if title not in tabs: return False
        width = len(self._schemas()[title])
        expected = self._schemas()[title]
        row = self.adapter.values(f'{quoted(title)}!A1:{column(width)}1')
        cell = cell_from_grid(self.adapter.grid([f'{quoted(title)}!A1']))
        if not row or row[0] != expected or cell.get('note') != self._owner_note(title):
            raise BridgeError('SCHEMA_CONFLICT: new tab ownership/header ' + title)
        if tabs[title]['gridProperties']['columnCount'] != width:
            raise BridgeError('SCHEMA_CONFLICT: new tab column count ' + title)
        return True

    def inspect(self):
        metadata, tabs = self._tabs()
        headers = self.adapter.values("'_records'!A1:BL1")
        if headers != [list(_contract().NATIVE_HEADERS)]:
            raise BridgeError('SCHEMA_CONFLICT: canonical header drift')
        a2 = cell_from_grid(self.adapter.grid(["'_records'!A2"]))
        formula = a2.get('userEnteredValue', {}).get('formulaValue')
        if not formula: raise BridgeError('FORMULA_REVIEW_REQUIRED: A2 is not a formula')
        records = self.adapter.values("'_records'!A2:BL")
        main = self.adapter.values("'WV3-main'!A6:BL")
        snapshot = dict(schema=SCHEMA, spreadsheet_id=self.spreadsheet_id, metadata=metadata,
            headers=headers[0], records_a2=a2, formula_sha256=formula_sha(formula),
            views=self.adapter.grid([f'{quoted(t)}!A5:BL6' for t in VIEW_NAMES if t in tabs]),
            protected=self.adapter.grid([quoted(t) for t in PROTECTED]),
            observations=observation_map(records),
            errors={'records': _errors(records), 'main': _errors(main)},
            ownership={t: self._owned(t, tabs) for t in self._schemas()})
        snapshot['snapshot_sha256'] = digest(snapshot)
        return snapshot

    def plan(self, prepared, snapshot):
        self._check_snapshot(snapshot)
        original = snapshot['records_a2']['userEnteredValue']['formulaValue']
        patched = patch_formula(original)
        _, tabs = self._tabs()
        existing = {t: self._owned(t, tabs) for t in self._schemas()}
        c=_contract();diff=[]
        source_values={t:self.adapter.values(f'{quoted(t)}!A2:{column(len(self._schemas()[t]))}')
                       for t in (*NATIVE_TABS,'_rb02_points') if existing[t]}
        expected=[]
        for record in prepared.get('native_records',[]):
            title='_rb01_'+record['server']
            row=list(c.build_native_row(record,spreadsheet_id=self.spreadsheet_id,
                gid=tabs.get(title,{}).get('sheetId')))
            expected.append((title,row[57],row,59,51))
        for record in prepared.get('stress_records',[]):
            expected.append(('_rb02_points',c.stress_source_row(record),list(c.build_stress_row(record)),0,39))
        desired_ids={r[i] for _,_,r,i,_ in expected}
        for title,number,row,idcol,readcol in expected:
            data=source_values.get(title,[]);actual=data[number-2] if len(data)>number-2 else []
            adjusted=list(actual)+['']*max(0,len(row)-len(actual))
            if adjusted[readcol] in ('UPLOAD_PENDING','READBACK_VERIFIED'):adjusted[readcol]=row[readcol]
            state='NEW' if not any(actual) else 'UNCHANGED' if rows_equal(adjusted,row) else 'CONFLICT'
            diff.append({'source':title,'row':number,'id':row[idcol],'action':state})
        for title,data in source_values.items():
            idcol=59 if title in NATIVE_TABS else 0
            for number,row in enumerate(data,2):
                rid=row[idcol] if len(row)>idcol else ''
                if rid and rid not in desired_ids:
                    diff.append({'source':title,'row':number,'id':rid,'action':'CONFLICT','reason':'CUMULATIVE_PACKAGE_REQUIRED'})
        return dict(schema=SCHEMA, spreadsheet_id=self.spreadsheet_id,
            snapshot_sha256=snapshot['snapshot_sha256'], prepared_sha256=digest(prepared),
            original_formula=original, patched_formula=patched,
            original_formula_sha256=formula_sha(original), patched_formula_sha256=formula_sha(patched),
            formula_changed=original != patched, create_tabs=[t for t,v in existing.items() if not v],
            native_rows=len(prepared.get('native_records', [])), stress_rows=len(prepared.get('stress_records', [])),
            writer_server='s1', protected_tabs=list(PROTECTED), existing_views_untouched=True,
            observation_diff=diff, diff_counts={state:sum(d['action']==state for d in diff)
                for state in ('NEW','UNCHANGED','CONFLICT')}, errors=prepared.get('errors',[]),
            missing_or_failed=[{'run_id':r['run_id'],'status':r.get('verification_status'),
                               'error':r.get('error',r.get('errors'))}
                              for r in prepared.get('statuses',[]) if r.get('verification_status')!='EVIDENCE_VERIFIED'])

    def _check_snapshot(self, snapshot):
        unsigned = {k:v for k,v in snapshot.items() if k != 'snapshot_sha256'}
        if snapshot.get('schema') != SCHEMA or snapshot.get('spreadsheet_id') != self.spreadsheet_id or digest(unsigned) != snapshot.get('snapshot_sha256'):
            raise BridgeError('Invalid/tampered snapshot')

    def _writer(self, server):
        if server != 's1': raise BridgeError('Only coordinated s1 writer may setup/upload')
        if not hasattr(self.adapter,'local_writer_server') or self.adapter.local_writer_server()!='s1':
            raise BridgeError('Local server ownership is not s1')

    def _write_formula(self, sheet_id, row, col, formula):
        self.adapter.batch_update([{'updateCells': {'range': {'sheetId': sheet_id,
            'startRowIndex': row-1, 'endRowIndex': row, 'startColumnIndex': col-1, 'endColumnIndex': col},
            'rows': [{'values': [{'userEnteredValue': {'formulaValue': formula}}]}],
            'fields': 'userEnteredValue'}}])

    def _regression(self, snapshot):
        _, tabs = self._tabs()
        if self.adapter.values("'_records'!A1:BL1") != [snapshot['headers']]:
            raise BridgeError('HISTORICAL_REGRESSION: headers')
        if _without_layout(self.adapter.grid([quoted(t) for t in PROTECTED])) != _without_layout(snapshot['protected']):
            raise BridgeError('HISTORICAL_REGRESSION: protected table value/formula/note/format/merge')
        if _without_layout(self.adapter.grid([f'{quoted(t)}!A5:BL6' for t in VIEW_NAMES if t in tabs])) != _without_layout(snapshot['views']):
            raise BridgeError('HISTORICAL_REGRESSION: existing view formulas/notes/formats')
        rows = self.adapter.values("'_records'!A2:BL")
        now = observation_map(rows)
        for rid, old in snapshot['observations'].items():
            if rid not in now or not rows_equal(now[rid], old):
                raise BridgeError('HISTORICAL_REGRESSION: existing Result_ID ' + rid)
        for key, current in [('records', _errors(rows)), ('main', _errors(self.adapter.values("'WV3-main'!A6:BL")))]:
            if set(current) - set(snapshot['errors'][key]):
                raise BridgeError('VIEW_REVIEW_REQUIRED: new spreadsheet error ' + key)
        # Relative order, hidden flags, bounds and protected ranges may not be reset.
        old_tabs = snapshot['metadata']['sheets']
        old_order = [s['properties']['sheetId'] for s in sorted(old_tabs,key=lambda s:s['properties'].get('index',0))]
        current_order = [p['sheetId'] for p in sorted(tabs.values(),key=lambda p:p.get('index',0)) if p['sheetId'] in old_order]
        if old_order != current_order: raise BridgeError('HISTORICAL_REGRESSION: tab ordering')
        live_metadata=self.adapter.metadata()
        live_sheets={s['properties']['sheetId']:s for s in live_metadata['sheets']}
        for sheet in old_tabs:
            p = sheet['properties']; live = tabs.get(p['title'], {})
            if p.get('hidden',False) != live.get('hidden',False):
                raise BridgeError('HISTORICAL_REGRESSION: existing tab hidden flag')
            if sheet.get('protectedRanges',[]) != live_sheets[p['sheetId']].get('protectedRanges',[]):
                raise BridgeError('HISTORICAL_REGRESSION: protected ranges')
        return {'historical_observations': len(snapshot['observations']), 'protected_tables_unchanged': True,
                'existing_view_formulas_unchanged': True, 'historical_ids_preserved': True}

    def setup(self, plan, snapshot, *, apply=False, writer_server='s1'):
        self._writer(writer_server); self._check_snapshot(snapshot)
        if plan.get('snapshot_sha256') != snapshot['snapshot_sha256']:
            raise BridgeError('Plan does not match snapshot')
        original = snapshot['records_a2']['userEnteredValue']['formulaValue']
        if plan.get('original_formula') != original or plan.get('patched_formula') != patch_formula(original):
            raise BridgeError('Plan formula differs from verified minimal patch')
        if plan.get('diff_counts',{}).get('CONFLICT',0):raise BridgeError('EVIDENCE_CONFLICT: dry-run plan has conflicting rows')
        if not apply: return {'status': 'DRY_RUN', 'plan': plan}
        self._regression(snapshot)
        _, tabs = self._tabs()
        # Fail before creating ANY tab when an existing title is not ours.
        for title in self._schemas(): self._owned(title, tabs)
        created = []
        for title, header in self._schemas().items():
            if title in tabs: continue
            count = 2400 if title == '_rb02_points' else (2600 if title == 'RB02-curves' else 1000 if title == 'RB-B01' else 32)
            # Atomic addSheet + owned header/note: choose an unused deterministic ID.
            sid = 269300000 + list(self._schemas()).index(title)
            if any(p['sheetId'] == sid for p in tabs.values()):
                raise BridgeError('SCHEMA_CONFLICT: requested new sheet ID occupied')
            request = {'addSheet': {'properties': {'sheetId': sid, 'title': title,
                'hidden': title.startswith('_'), 'gridProperties': {'rowCount': count,
                    'columnCount': len(header), 'frozenRowCount': 1, 'frozenColumnCount': min(3,len(header))}}}}
            cells = [{'userEnteredValue': {'stringValue': h}} for h in header]
            cells[0]['note'] = self._owner_note(title)
            self.adapter.batch_update([request, {'updateCells': {'start': {'sheetId':sid,'rowIndex':0,'columnIndex':0},
                'rows':[{'values':cells}], 'fields':'userEnteredValue,note'}},
                {'repeatCell': {'range': {'sheetId':sid,'startRowIndex':0,'endRowIndex':1},
                    'cell': {'userEnteredFormat': {'textFormat': {'bold':True}, 'wrapStrategy':'WRAP'}},
                    'fields':'userEnteredFormat'}}])
            formatting=[{'updateDimensionProperties':{'range':{'sheetId':sid,'dimension':'COLUMNS',
                'startIndex':0,'endIndex':len(header)},'properties':{'pixelSize':125},'fields':'pixelSize'}},
                {'repeatCell':{'range':{'sheetId':sid,'startRowIndex':1},'cell':{'userEnteredFormat':{
                    'wrapStrategy':'CLIP','numberFormat':{'type':'NUMBER','pattern':'0.0000'}}},
                    'fields':'userEnteredFormat.wrapStrategy,userEnteredFormat.numberFormat'}}]
            if title in NATIVE_TABS:
                formatting += [{'repeatCell':{'range':{'sheetId':sid,'startRowIndex':1,'startColumnIndex':15,'endColumnIndex':22},
                    'cell':{'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.000'}}},
                    'fields':'userEnteredFormat.numberFormat'}},
                    {'repeatCell':{'range':{'sheetId':sid,'startColumnIndex':28},
                    'cell':{'userEnteredFormat':{'backgroundColor':{'red':.94,'green':.94,'blue':.94}}},
                    'fields':'userEnteredFormat.backgroundColor'}}]
            else:
                technical_start=9 if title=='RB-B01' else min(24,len(header)-1)
                formatting.append({'repeatCell':{'range':{'sheetId':sid,'startColumnIndex':technical_start},
                    'cell':{'userEnteredFormat':{'backgroundColor':{'red':.94,'green':.94,'blue':.94}}},
                    'fields':'userEnteredFormat.backgroundColor'}})
            integer_ranges=([(31,34),(38,41),(57,58),(62,63)] if title in NATIVE_TABS else
                [(6,8),(16,18),(32,34)] if title=='_rb02_points' else
                [(CURVE_VIEW_COLUMNS.index(i),CURVE_VIEW_COLUMNS.index(i)+1) for i in (6,7,16,17,32,33)] if title=='RB02-curves' else
                [(4,8),(10,12),(13,17),(18,22)] if title=='_rb_b01_status' else [(5,6)])
            formatting += [{'repeatCell':{'range':{'sheetId':sid,'startRowIndex':1,
                'startColumnIndex':a,'endColumnIndex':b},'cell':{'userEnteredFormat':{
                'numberFormat':{'type':'NUMBER','pattern':'0'}}},'fields':'userEnteredFormat.numberFormat'}}
                for a,b in integer_ranges]
            self.adapter.batch_update(formatting)
            created.append(title)
        _, tabs = self._tabs()
        for title in self._schemas(): self._owned(title,tabs)
        current = cell_from_grid(self.adapter.grid(["'_records'!A2"]))
        expected_cell = snapshot['records_a2']
        patched = plan['patched_formula']
        already = current.get('userEnteredValue',{}).get('formulaValue') == patched
        if (not already and current.get('userEnteredValue') != expected_cell.get('userEnteredValue')) or current.get('note') != expected_cell.get('note'):
            raise BridgeError('CONCURRENT_EDIT: _records!A2 changed since snapshot')
        changed = patched != original and not already
        if changed:
            for title in NATIVE_TABS:
                if any(any(row) for row in self.adapter.values(f'{quoted(title)}!A2:BL17')):
                    raise BridgeError('FORMULA_REVIEW_REQUIRED: initial union requires empty new native sources')
        growth=[]
        for title,need in [('_records',len(snapshot['observations'])+100),
                           ('WV3-main',len(self.adapter.values("'WV3-main'!A6:BL"))+106)]:
            have=tabs[title]['gridProperties']['rowCount']
            if have<need:growth.append({'appendDimension':{'sheetId':tabs[title]['sheetId'],
                                      'dimension':'ROWS','length':need-have}})
        if growth:self.adapter.batch_update(growth)
        try:
            if changed: self._write_formula(tabs['_records']['sheetId'],2,1,patched)
            regression = self._regression(snapshot)
        except Exception:
            if changed:
                after = cell_from_grid(self.adapter.grid(["'_records'!A2"]))
                if after.get('userEnteredValue',{}).get('formulaValue') == patched and after.get('note') == expected_cell.get('note'):
                    self._write_formula(tabs['_records']['sheetId'],2,1,original)
            raise
        curve_formula = ('=IFNA(CHOOSECOLS(SORT(FILTER(\'_rb02_points\'!A2:AQ,\'_rb02_points\'!A2:A<>""),'
                         '5,TRUE,6,TRUE,7,TRUE,11,TRUE,13,TRUE,14,TRUE,12,TRUE),'
                         + ','.join(str(i+1) for i in CURVE_VIEW_COLUMNS) + '),"")')
        curve_current = cell_from_grid(self.adapter.grid(["'RB02-curves'!A2"]))
        if curve_current.get('userEnteredValue') and curve_current.get('userEnteredValue',{}).get('formulaValue') != curve_formula:
            raise BridgeError('SCHEMA_CONFLICT: RB02-curves formula differs')
        if not curve_current.get('userEnteredValue'):
            self._write_formula(tabs['RB02-curves']['sheetId'],2,1,curve_formula)
        return {'status':'SETUP_COMPLETE','created_tabs':created,'formula_changed':changed,
                'formula_before_sha256':formula_sha(original),'formula_after_sha256':formula_sha(patched),
                'gids':{t:tabs[t]['sheetId'] for t in self._schemas()},'regression':regression}

    def _rows(self, prepared, tabs):
        c = _contract(); rows = []
        for record in prepared.get('native_records',[]):
            title = '_rb01_' + record['server']
            row = list(c.build_native_row(record,spreadsheet_id=self.spreadsheet_id,gid=tabs[title]['sheetId']))
            rows.append((title,int(row[57]),row,59,51))
        for record in prepared.get('stress_records',[]):
            row = list(c.build_stress_row(record))
            rows.append(('_rb02_points',c.stress_source_row(record),row,0,39))
        registry_slots={r['run_id']:i for i,r in enumerate(c.registry(),2)}
        for status in prepared.get('statuses',[]):
            row=list(c.build_status_row(status));row[23]='UPLOAD_PENDING'
            rows.append(('_rb_b01_status',registry_slots[status['run_id']],row,None,23))
        keys=[(t,n) for t,n,*_ in rows]
        if len(keys)!=len(set(keys)): raise BridgeError('Prepared duplicate deterministic slot')
        return rows

    def _summary_rows(self, prepared):
        """Formula-backed aggregates with the same explicit Student membership.

        Direction averages are formed inside each Student before STDEV. Failed
        points remain in the source and are visibly counted, never imputed zero.
        """
        summary=prepared.get('summary')
        if not summary:return [],[]
        c=_contract()
        native_refs={c.native_result_id(r):(quoted('_rb01_'+r['server']),
            c.source_row(r['repeat'],r['case_id'],r['selection_id'])) for r in prepared['native_records']}
        stress_refs={c.stress_record_id(r):(quoted('_rb02_points'),c.stress_source_row(r))
                     for r in prepared['stress_records']}
        ncols=dict(zip(c.NATIVE_METRICS,(4,5,6,7,8,9,10,11,12,13,14,15)))
        scols=dict(zip(c.STRESS_METRICS,(19,20,21,22,23,24)))
        def ref(rid,metric,stress=False):
            tab,num=(stress_refs if stress else native_refs)[rid]
            return f'{tab}!{column((scols if stress else ncols)[metric])}{num}'
        def dif(pair,metric,stress=False):
            a,b=ref(pair['left'],metric,stress),ref(pair['right'],metric,stress)
            return f'IF(COUNT({a},{b})=2,{a}-{b},"")'
        def raw(value):
            if value is None:value=''
            key='boolValue' if isinstance(value,bool) else 'numberValue' if isinstance(value,(int,float)) else 'stringValue'
            return {'userEnteredValue':{key:value}}
        cells=[];expected=[]
        def add(values,formulas=None):
            expected.append(values)
            row=[raw(v) for v in values]
            for idx,formula in (formulas or {}).items():row[idx]={'userEnteredValue':{'formulaValue':formula}}
            cells.append({'values':row})
        add(['Notes','','','','Registered Student is unit; selector aliases, scenes and directions are not extra seeds.',
             '', '', '', '', 'Native RR20:-21 / FR full512; stress fixed192. Partial cohorts; sample SD ddof=1.'])
        for server in summary.get('server_status',[]):
            s=server['server'];tab=quoted('_rb01_'+s)
            for metric,count,expected_n,formula in [
                ('Students',server['verified_students'],8,f'=COUNTIF({tab}!AQ2:AQ17,"EXACT_50000")'),
                ('Native observations',server['verified_native_observations'],16,f'=COUNTA({tab}!BH2:BH17)'),
                ('Curves',server['verified_curves'],16,f'=COUNTIFS(\'_rb02_points\'!F2:F2353,"{s}",\'_rb02_points\'!L2:L2353,"D000")')]:
                add(['Status',s,'','',metric,'',count,expected_n,expected_n-count,
                     f"numerical_failures={server['numerical_failures']}; invalid_geometry={server['invalid_geometry']}"],{6:formula})
            ledger=[r for r in prepared.get('statuses',[]) if r['server']==s]
            invalid=sum(r.get('verification_status')=='EVIDENCE_INVALID' for r in ledger)
            reasons=sum(bool(r.get('error') or r.get('errors')) for r in ledger)
            for metric,value,formula,description in [
                ('Evidence invalid',invalid,f'=COUNTIFS(\'_rb_b01_status\'!D2:D25,"{s}",\'_rb_b01_status\'!W2:W25,"EVIDENCE_INVALID")','Source validation failures, separate from delivery.'),
                # COUNTIFS("<>") can count RAW-written empty-string cells as
                # nonempty. LEN tests actual textual content, including after
                # Sheets canonicalizes an empty string into an empty CellData.
                ('Missing/error reasons',reasons,f'=SUMPRODUCT((\'_rb_b01_status\'!D2:D25="{s}")*(LEN(\'_rb_b01_status\'!Y2:Y25)>0))','Includes missing remote evidence; does not imply experiments never ran.'),
                ('Verified-run upload debt',0,f'=COUNTIFS(\'_rb_b01_status\'!D2:D25,"{s}",\'_rb_b01_status\'!W2:W25,"EVIDENCE_VERIFIED",\'_rb_b01_status\'!X2:X25,"<>READBACK_VERIFIED")','Delivery status only; source evidence states remain unchanged.')]:
                add(['Status',s,'','',metric,'',value,'','',description],{6:formula})
        for panel in ('native','paired','stress_radius'):
            for item in summary.get(panel,[]):
                metric=item['metric'];stress=panel=='stress_radius'
                if panel=='native':expr=[ref(rid,metric) for rid in item['member_ids']]
                elif panel=='paired':expr=[dif(pair,metric) for pair in item['member_pairs']]
                else:
                    expr=[]
                    for student in item['student_values']:
                        if student['value'] is None:continue
                        parts=([ref(rid,metric,True) for rid in student['member_ids']] if 'member_ids' in student
                               else [dif(pair,metric,True) for pair in student['member_pairs']])
                        args=','.join(parts)
                        expr.append(f'IF(COUNT({args})={len(parts)},AVERAGE({args}),"")')
                formulas={}
                if expr:
                    args=','.join(expr)
                    formulas[5]=f'=COUNT({args})'
                    formulas[6]=f'=IF(COUNT({args})={len(expr)},AVERAGE({args}),"")'
                    if len(expr)>1:formulas[7]=f'=IF(COUNT({args})={len(expr)},STDEV({args}),"")'
                provenance={'expected_n':item['expected_n'],'n_students':item['n_students'],
                    'full_six':item['full_six'],'flagged_n':item.get('flagged_n',0)}
                if panel=='paired':provenance.update(improved=item['improved'],ties=item['ties'],difference='case-QFULL')
                if stress:provenance.update(direction_count=item['direction_count'],mode_difference='A_ON-A_ZERO' if item['mode']=='A_ON_MINUS_A_ZERO' else None)
                add([panel,item['cohort_id'],item.get('selection_id',item.get('mode','')),
                    item['case_id'],metric+(f" / r={item['radius_hr']}" if stress else ''),
                    item['n'],item['mean'],item['sample_std'],
                    f"missing={item['missing_n']}; failed={item['failed_n']}",
                    json.dumps(provenance,sort_keys=True,separators=(',',':'))],formulas)
        for cohort,provenance in summary.get('cohorts',{}).items():
            add(['Provenance',cohort,'','','','','','','',json.dumps(provenance,sort_keys=True,separators=(',',':'))])
        return cells,expected

    def _update_summary(self,prepared,tabs):
        cells,expected=self._summary_rows(prepared)
        if not cells:return False
        sid=tabs['RB-B01']['sheetId']
        old=self.adapter.values("'RB-B01'!A2:J")
        count=max(len(cells),len(old))
        if tabs['RB-B01']['gridProperties']['rowCount']<count+1:
            self.adapter.batch_update([{'appendDimension':{'sheetId':sid,'dimension':'ROWS',
                'length':count+1-tabs['RB-B01']['gridProperties']['rowCount']}}])
        cells += [{'values':[{'userEnteredValue':{'stringValue':''}} for _ in range(10)]}
                  for _ in range(count-len(cells))]
        # Only explicit formulaValue cells execute; all labels remain stringValue.
        self.adapter.batch_update([{'updateCells':{'range':{'sheetId':sid,'startRowIndex':1,
            'endRowIndex':count+1,'startColumnIndex':0,'endColumnIndex':10},'rows':cells,
            'fields':'userEnteredValue'}}])
        return True

    def _verify_summary(self,prepared):
        _,expected=self._summary_rows(prepared)
        if not expected:return False
        actual=self.adapter.values(f"'RB-B01'!A2:J{len(expected)+1}")
        if len(actual)!=len(expected):raise BridgeError('SUMMARY_READBACK_MISMATCH: row count')
        for i,(a,e) in enumerate(zip(actual,expected),2):
            if not rows_equal(a,e):raise BridgeError('SUMMARY_READBACK_MISMATCH: RB-B01 row '+str(i))
        return True

    def _existing_rows(self, titles):
        return {t:self.adapter.values(f'{quoted(t)}!A2:{column(len(self._schemas()[t]))}') for t in titles}

    @staticmethod
    def _actual(existing,title,row):
        values=existing[title]
        return values[row-2] if len(values)>row-2 else []

    def upload(self, prepared, *, apply=False, writer_server='s1'):
        self._writer(writer_server)
        _,tabs=self._tabs()
        for t in self._schemas():
            if not self._owned(t,tabs): raise BridgeError('SETUP_REQUIRED: '+t)
        formula=cell_from_grid(self.adapter.grid(["'_records'!A2"])).get('userEnteredValue',{}).get('formulaValue','')
        if patch_formula(formula)!=formula:raise BridgeError('SETUP_REQUIRED: native union missing')
        rows=self._rows(prepared,tabs)
        existing=self._existing_rows({r[0] for r in rows})
        expected_ids={row[idcol] for _,_,row,idcol,_ in rows if idcol is not None}
        for title in (*NATIVE_TABS,'_rb02_points'):
            data=existing.get(title)
            if data is None:data=self.adapter.values(f'{quoted(title)}!A2:{column(len(self._schemas()[title]))}')
            idcol=59 if title in NATIVE_TABS else 0
            seen=set()
            for row in data:
                rid=row[idcol] if len(row)>idcol else ''
                if any(row) and not rid:raise BridgeError('SCHEMA_CONFLICT: nonempty source row without ID')
                if not rid:continue
                if rid in seen:raise BridgeError('Duplicate source observation ID')
                seen.add(rid)
                if rid not in expected_ids:raise BridgeError('CUMULATIVE_PACKAGE_REQUIRED: existing evidence omitted '+str(rid))
        writes=[]; readbacks=[]; new=0; unchanged=0; status_changes=0
        for title,number,row,idcol,readcol in rows:
            actual=self._actual(existing,title,number)
            if idcol is not None:
                if any(actual):
                    adjusted=list(actual)+['']*max(0,len(row)-len(actual))
                    if readcol is not None and adjusted[readcol] in ('UPLOAD_PENDING','READBACK_VERIFIED'):
                        adjusted[readcol]=row[readcol]
                    if not rows_equal(adjusted,row):
                        raise BridgeError(f'EVIDENCE_CONFLICT: occupied immutable source slot {title}!{number}')
                    unchanged+=1
                else:
                    new+=1; writes.append((f'{quoted(title)}!A{number}:{column(len(row))}{number}',[row]))
                readbacks.append((title,number,row,idcol,readcol))
            else:
                adjusted=list(actual)+['']*max(0,len(row)-len(actual))
                if len(adjusted)>25 and adjusted[25] and not row[25]:
                    raise BridgeError('STATUS_REGRESSION: existing verified source omitted')
                if readcol is not None and adjusted[readcol]=='READBACK_VERIFIED':adjusted[readcol]=row[readcol]
                if not rows_equal(adjusted,row):
                    status_changes+=1;writes.append((f'{quoted(title)}!A{number}:{column(len(row))}{number}',[row]))
                readbacks.append((title,number,row,idcol,readcol))
        receipt={'schema':SCHEMA,'status':'DRY_RUN','new_observations':new,'unchanged_observations':unchanged,
                 'status_rows_changed':status_changes,'planned_write_rows':len(writes),'prepared_sha256':digest(prepared)}
        if not apply: return receipt
        # A timeout after RAW write is resolved from actual cells before any retry.
        for start in range(0,len(writes),100):
            chunk=writes[start:start+100]
            try: self.adapter.write_raw(chunk)
            except Exception:
                if not all(len(self.adapter.values(name))==len(values) and
                           all(rows_equal(a,e) for a,e in zip(self.adapter.values(name),values))
                           for name,values in chunk):
                    raise
        self._progress('SOURCE_WRITTEN',{'written_rows':len(writes),'new_observations':new})
        now=self._existing_rows({r[0] for r in rows})
        marks=[]
        for title,number,row,idcol,readcol in readbacks:
            actual=self._actual(now,title,number)
            normalized=list(actual)+['']*max(0,len(row)-len(actual))
            if readcol is not None and normalized[readcol]=='READBACK_VERIFIED': normalized[readcol]=row[readcol]
            if not rows_equal(normalized,row): raise BridgeError('SOURCE_READBACK_MISMATCH: '+title+' '+str(number))
            if readcol is not None and (len(actual)<=readcol or actual[readcol]!='READBACK_VERIFIED'):
                marks.append((f'{quoted(title)}!{column(readcol+1)}{number}',[['READBACK_VERIFIED']]))
        for start in range(0,len(marks),100):self.adapter.write_raw(marks[start:start+100])
        final=self._existing_rows({r[0] for r in readbacks})
        for title,number,row,idcol,readcol in readbacks:
            checked=list(row)
            if readcol is not None:checked[readcol]='READBACK_VERIFIED'
            if not rows_equal(self._actual(final,title,number),checked):
                raise BridgeError('SOURCE_READBACK_MARK_MISMATCH: '+title+' '+str(number))
        self._progress('SOURCE_READBACK_VERIFIED',{'verified_rows':len(readbacks)})
        self._update_summary(prepared,tabs)
        receipt['summary_readback_verified']=self._verify_summary(prepared)
        if receipt['summary_readback_verified']:
            self._progress('SUMMARY_READBACK_VERIFIED',{'summary_sha256':digest(prepared['summary'])})
        receipt.update(status='SOURCE_READBACK_VERIFIED',source_readback_verified=True)
        return receipt

    def verify(self, prepared, snapshot):
        self._check_snapshot(snapshot)
        _,tabs=self._tabs()
        for t in self._schemas():
            if not self._owned(t,tabs): raise BridgeError('SETUP_REQUIRED: '+t)
        rows=self._rows(prepared,tabs)
        existing=self._existing_rows({r[0] for r in rows})
        native=[];stress=[]
        for title,number,row,idcol,readcol in rows:
            actual=self._actual(existing,title,number)
            if readcol is not None:row[readcol]='READBACK_VERIFIED'
            if not rows_equal(actual,row):raise BridgeError('SOURCE_READBACK_MISMATCH: '+title+' '+str(number))
            if title in NATIVE_TABS:native.append(row)
            if title=='_rb02_points':stress.append(row)
        records=observation_map(self.adapter.values("'_records'!A2:BL"))
        main=observation_map(self.adapter.values("'WV3-main'!A6:BL"))
        for row in native:
            for title,values in [('_records',records),('WV3-main',main)]:
                if row[59] not in values or not rows_equal(values[row[59]],row):
                    raise BridgeError('VIEW_READBACK_MISMATCH: '+title+' '+row[59])
        curve_rows=self.adapter.values("'RB02-curves'!A2:AQ")
        curve_ids={}
        public_id_col=CURVE_VIEW_COLUMNS.index(0)
        for row in curve_rows:
            rid=row[public_id_col] if len(row)>public_id_col else ''
            if rid:
                if rid in curve_ids:raise BridgeError('Duplicate stress view ID')
                curve_ids[rid]=row
        for row in stress:
            public_row=[row[i] for i in CURVE_VIEW_COLUMNS]
            if row[0] not in curve_ids or not rows_equal(curve_ids[row[0]],public_row):
                raise BridgeError('VIEW_READBACK_MISMATCH: RB02-curves '+row[0])
        regression=self._regression(snapshot)
        result={'schema':SCHEMA,'status':'VIEW_READBACK_VERIFIED','native_rows':len(native),
                'students':len({r[29] for r in native}),'stress_points':len(stress),
                'source_readback_verified':True,'formula_views_readback_verified':True,'regression':regression,
                'summary_readback_verified':self._verify_summary(prepared)}
        self._progress('VIEW_READBACK_VERIFIED',result)
        return result
