"""Synthetic fixtures only. This test never authenticates or reaches Google."""
import copy
import unittest
from unittest.mock import patch

from reporting_bridge import rb_b01_sheets as m
from reporting_bridge.rb_b01_contract import NATIVE_HEADERS


class FakeSheets:
    spreadsheet_id=m.SHEET_ID
    def __init__(self):
        self.tabs={};self.cells={};self.writes=[];self.break_union=False;self.timeout_after_write=False
        for i,title in enumerate((*m.PROTECTED,'_records',*m.VIEW_NAMES)):
            self.tabs[title]={'sheetId':m.FIXED_IDS.get(title,600+i),'title':title,'index':i,
                'hidden':title.startswith('_'),'gridProperties':{'rowCount':12000,'columnCount':64}}
            self.cells[title]={}
        for j,h in enumerate(NATIVE_HEADERS):self.cells['_records'][(0,j)]={'userEnteredValue':{'stringValue':h}}
        self.cells['_records'][(1,0)]={'userEnteredValue':{'formulaValue':'=LET(allrows,legacy,'+m.OLD_RETURN+')'},'note':'original note'}
        self.legacy=['legacy']+['']*63;self.legacy[59]='OLD_ID';self.legacy[3]=.9372458218
        for title in m.VIEW_NAMES:self.cells[title][(5,0)]={'userEnteredValue':{'formulaValue':'=legacy_view()'}}
        self.cells['paper'][(0,0)]={'userEnteredValue':{'numberValue':.9988},'note':'keep',
                                 'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}}
    def local_writer_server(self):return 's1'
    def metadata(self):return {'spreadsheetId':self.spreadsheet_id,'sheets':[{'properties':copy.deepcopy(p)} for p in self.tabs.values()]}
    @staticmethod
    def _range(text):
        title,rng=text.split('!',1) if '!' in text else (text,'A1:BL12000')
        title=title.strip("'")
        import re
        def loc(x):
            z=re.fullmatch(r'([A-Z]+)([0-9]*)',x);col=0
            for c in z[1]:col=col*26+ord(c)-64
            return (int(z[2] or 12000)-1,col-1)
        a,b=rng.split(':') if ':' in rng else (rng,rng)
        return title,loc(a),loc(b)
    def grid(self,ranges):
        out=[]
        for text in ranges:
            title,(r,c),(rr,cc)=self._range(text)
            data=[]
            for (i,j),cell in self.cells[title].items():
                if r<=i<=rr and c<=j<=cc:data.append({'startRow':i,'startColumn':j,'rowData':[{'values':[copy.deepcopy(cell)]}]})
            out.append({'properties':{'sheetId':self.tabs[title]['sheetId'],'title':title},'data':data})
        return {'sheets':out}
    def values(self,text):
        title,(r,c),(rr,cc)=self._range(text)
        if title in ('_records','WV3-main') and r>=1:
            formula=self.cells['_records'][(1,0)]['userEnteredValue']['formulaValue']
            result=[] if self.break_union and m.NEW_RETURN in formula else [copy.deepcopy(self.legacy)]
            if m.NEW_RETURN in formula:
                for t in m.NATIVE_TABS:
                    if t in self.tabs:result += [v for v in self.values("'"+t+"'!A2:BL") if len(v)>59 and v[59]]
            return result
        if title=='RB02-curves' and r>=1:
            return [[(v+['']*43)[i] for i in m.CURVE_VIEW_COLUMNS]
                    for v in self.values("'_rb02_points'!A2:AQ") if v and v[0]]
        result=[]
        maxrow=max((i for (i,j) in self.cells[title] if r<=i<=rr and c<=j<=cc),default=r-1)
        for i in range(r,maxrow+1):
            row=[]
            maxcol=max((j for (ii,j) in self.cells[title] if ii==i and c<=j<=cc),default=c-1)
            for j in range(c,maxcol+1):
                value=self.cells[title].get((i,j),{}).get('userEnteredValue',{})
                row.append(next(iter(value.values()),''))
            result.append(row)
        return result
    def batch_update(self,requests):
        self.writes.extend(copy.deepcopy(requests))
        for req in requests:
            if 'addSheet' in req:
                p=copy.deepcopy(req['addSheet']['properties']);p['index']=len(self.tabs);self.tabs[p['title']]=p;self.cells[p['title']]={}
            elif 'updateCells' in req:
                v=req['updateCells'];rg=v.get('range',v.get('start'));sid=rg['sheetId']
                title=next(t for t,p in self.tabs.items() if p['sheetId']==sid)
                r=rg.get('startRowIndex',rg.get('rowIndex'));c=rg.get('startColumnIndex',rg.get('columnIndex'))
                for i,row in enumerate(v['rows']):
                    for j,cell in enumerate(row['values']):self.cells[title].setdefault((r+i,c+j),{}).update(copy.deepcopy(cell))
            elif 'appendDimension' in req:
                v=req['appendDimension'];title=next(t for t,p in self.tabs.items() if p['sheetId']==v['sheetId'])
                self.tabs[title]['gridProperties']['rowCount']+=v['length']
    def write_raw(self,entries):
        self.writes.extend(copy.deepcopy(entries))
        for text,rows in entries:
            title,(r,c),_=self._range(text)
            for i,row in enumerate(rows):
                for j,value in enumerate(row):
                    kind='boolValue' if isinstance(value,bool) else 'numberValue' if isinstance(value,(int,float)) else 'stringValue'
                    self.cells[title].setdefault((r+i,c+j),{})['userEnteredValue']={kind:value}
        if self.timeout_after_write:
            self.timeout_after_write=False;raise TimeoutError('synthetic lost response after commit')


class SheetsTests(unittest.TestCase):
    def setUp(self):self.api=FakeSheets();self.bridge=m.SheetsBridge(self.api)
    def setup(self):
        self.snapshot=self.bridge.inspect();self.plan=self.bridge.plan({},self.snapshot)
        return self.bridge.setup(self.plan,self.snapshot,apply=True)
    def synthetic_row(self):
        row=['']*64;row[0]='=SYNTHETIC never upload';row[1]='s1';row[3]=.94561238900001
        row[28]=m.CAMPAIGN;row[29]='SYNTHETIC_RUN';row[41]='a'*64;row[42]='EXACT_50000'
        row[51]='UPLOAD_PENDING';row[53]=False;row[57]=2;row[59]='RB01_SYNTHETIC_ID'
        return row
    def test_formula_only_terminal_return_and_idempotence(self):
        original='=LET(parse,LAMBDA(tab,legacy),allrows,VSTACK(legacy),'+m.OLD_RETURN+'))'
        result=m.patch_formula(original)
        self.assertEqual(result.split(m.NEW_RETURN)[0],original.split(m.OLD_RETURN)[0])
        self.assertEqual(m.patch_formula(result),result)
        with self.assertRaises(m.BridgeError):m.patch_formula('='+m.OLD_RETURN+'+other')
    def test_empty_source_migration_preserves_legacy(self):
        receipt=self.setup();self.assertTrue(receipt['formula_changed'])
        self.assertEqual(self.api.values("'_records'!A2:BL"),[self.api.legacy])
        self.assertEqual(self.api.cells['_records'][(1,0)]['note'],'original note')
        n=len(self.api.writes);again=self.bridge.setup(self.plan,self.snapshot,apply=True)
        self.assertFalse(again['formula_changed']);self.assertEqual(n,len(self.api.writes))
    def test_rollback_only_own_formula(self):
        snapshot=self.bridge.inspect();plan=self.bridge.plan({},snapshot);self.api.break_union=True
        with self.assertRaisesRegex(m.BridgeError,'HISTORICAL_REGRESSION'):self.bridge.setup(plan,snapshot,apply=True)
        self.assertEqual(self.api.cells['_records'][(1,0)]['userEnteredValue']['formulaValue'],plan['original_formula'])
        self.assertIn('_rb01_s1',self.api.tabs)
    def test_header_drift(self):
        self.api.cells['_records'][(0,2)]['userEnteredValue']={'stringValue':'changed'}
        with self.assertRaisesRegex(m.BridgeError,'header drift'):self.bridge.inspect()
        self.assertEqual(self.api.writes,[])
    def test_existing_unowned_title_blocked_before_other_writes(self):
        self.api.tabs['_rb01_s1']={'sheetId':987,'title':'_rb01_s1','gridProperties':{'rowCount':17,'columnCount':64}}
        self.api.cells['_rb01_s1']={}
        with self.assertRaisesRegex(m.BridgeError,'ownership'):self.bridge.inspect()
        self.assertEqual(self.api.writes,[])
    def test_raw_formula_injection_retry_readback_and_no_duplicates(self):
        self.setup();row=self.synthetic_row()
        with patch.object(self.bridge,'_rows',return_value=[('_rb01_s1',2,row,59,51)]):
            self.api.timeout_after_write=True
            result=self.bridge.upload({},apply=True)
            self.assertEqual(result['new_observations'],1)
            self.assertEqual(self.api.cells['_rb01_s1'][(1,0)]['userEnteredValue'],{'stringValue':row[0]})
            again=self.bridge.upload({},apply=True)
            self.assertEqual(again['new_observations'],0)
            self.assertEqual(again['unchanged_observations'],1)
            self.assertTrue(self.bridge.verify({},self.snapshot)['formula_views_readback_verified'])
    def test_conflicting_payload_blocked(self):
        self.setup();row=self.synthetic_row()
        with patch.object(self.bridge,'_rows',return_value=[('_rb01_s1',2,row,59,51)]):
            self.bridge.upload({},apply=True)
            row[3]+=.001
            with self.assertRaisesRegex(m.BridgeError,'EVIDENCE_CONFLICT'):self.bridge.upload({},apply=True)
    def test_historical_table_note_regression(self):
        self.setup();self.api.cells['paper'][(0,0)]['note']='unauthorized change'
        with self.assertRaisesRegex(m.BridgeError,'protected table'):self.bridge._regression(self.snapshot)
    def test_s1_only_writer_not_merely_cli_flag(self):
        self.api.local_writer_server=lambda:'s3'
        with self.assertRaisesRegex(m.BridgeError,'ownership'):self.bridge.upload({},apply=True,writer_server='s1')
    def test_snapshot_tamper(self):
        snapshot=self.bridge.inspect();snapshot['observations']['OLD_ID'][3]=1
        with self.assertRaisesRegex(m.BridgeError,'tampered'):self.bridge.plan({},snapshot)
    def test_initial_union_requires_empty_native_sources(self):
        self.setup()
        self.api.cells['_records'][(1,0)]['userEnteredValue']['formulaValue']=self.plan['original_formula']
        self.api.write_raw([("'_rb01_s1'!A2:BL2",[self.synthetic_row()])])
        with self.assertRaisesRegex(m.BridgeError,'requires empty'):
            self.bridge.setup(self.plan,self.snapshot,apply=True)
        self.assertEqual(self.api.cells['_records'][(1,0)]['userEnteredValue']['formulaValue'],self.plan['original_formula'])
    def test_stage_callbacks_only_after_successful_source_checks(self):
        self.setup();row=self.synthetic_row();stages=[]
        self.bridge.progress_callback=lambda state,details:stages.append(state)
        with patch.object(self.bridge,'_rows',return_value=[('_rb01_s1',2,row,59,51)]):
            self.bridge.upload({},apply=True);self.bridge.verify({},self.snapshot)
        self.assertEqual(stages,['SOURCE_WRITTEN','SOURCE_READBACK_VERIFIED','VIEW_READBACK_VERIFIED'])
    def test_status_transport_separate_from_missing_evidence(self):
        from reporting_bridge.rb_b01_contract import registry
        self.setup()
        status=dict(registry()[0],verification_status='SOURCE_MISSING',upload_status='NOT_AVAILABLE')
        data={'statuses':[status]}
        self.bridge.upload(data,apply=True)
        actual=self.api.values("'_rb_b01_status'!A2:AA2")[0]
        self.assertEqual(actual[22],'SOURCE_MISSING');self.assertEqual(actual[23],'READBACK_VERIFIED')
        self.assertTrue(self.bridge.verify(data,self.snapshot)['source_readback_verified'])
    def test_number_serialization_exact_int_bool_and_precision(self):
        self.assertFalse(m.same(50000.00000001,50000));self.assertFalse(m.same(0,False))
        self.assertTrue(m.same(.9567891000000001,.9567891))
        self.assertFalse(m.same('0.95',.95))
    def test_public_stress_layout_preserves_all_raw_columns(self):
        from reporting_bridge.rb_b01_contract import STRESS_HEADERS
        self.setup()
        headers=self.bridge._schemas()['RB02-curves']
        self.assertEqual(headers[:10],['Case_ID','Server','Repeat','Mode','Radius_HR','Angle_deg',
                                      'ERGAS','PSNR','SAM','Edge_error_DN'])
        self.assertEqual(sorted(m.CURVE_VIEW_COLUMNS),list(range(43)))
        self.assertEqual(set(headers),set(STRESS_HEADERS))
        self.assertEqual(self.bridge._schemas()['_rb02_points'],list(STRESS_HEADERS))
        row=['']*43;row[0]='SYNTHETIC_STRESS_ID';row[4]='QFULL';row[5]='s1';row[6]=1
        row[10]='A_ON';row[12]=0.;row[18]=2.000012345678;row[39]='UPLOAD_PENDING'
        with patch.object(self.bridge,'_rows',return_value=[('_rb02_points',2,row,0,39)]):
            self.bridge.upload({},apply=True)
            self.assertTrue(self.bridge.verify({},self.snapshot)['formula_views_readback_verified'])
        shown=self.api.values("'RB02-curves'!A2:AQ")[0]
        self.assertEqual(shown[:5],['QFULL','s1',1,'A_ON',0.])
        self.assertEqual(shown[6],row[18])
        self.assertEqual(shown[m.CURVE_VIEW_COLUMNS.index(0)],row[0])
    def test_missing_reason_formula_ignores_raw_empty_string_cells(self):
        from reporting_bridge.rb_b01_contract import registry, canonical_native, build_status_row
        from reporting_bridge.rb_b01_summary import summarize
        from reporting_bridge.tests.test_rb_b01_contract_summary import synthetic_entry
        statuses=[dict(r,errors=[],error='' if r['server']=='s1' else 'Evidence not supplied')
                  for r in registry()]
        records=canonical_native(synthetic_entry())
        prepared=dict(native_records=records,stress_records=[],statuses=statuses,
                      summary=summarize(records,[],statuses))
        cells,expected=self.bridge._summary_rows(prepared)
        for server,expected_count in [('s1',0),('s3',8),('s5',8)]:
            index=next(i for i,row in enumerate(expected) if row[0]=='Status' and row[1]==server
                       and row[4]=='Missing/error reasons')
            formula=cells[index]['values'][6]['userEnteredValue']['formulaValue']
            self.assertIn('SUMPRODUCT(',formula);self.assertIn('LEN(',formula)
            self.assertNotIn('COUNTIFS(',formula)
            self.assertIn(f'D2:D25="{server}"',formula)
            self.assertEqual(expected[index][6],expected_count)
            # Empty strings have length zero both before RAW upload and after
            # Google represents their userEnteredValue as missing/empty cells.
            raw_rows=[build_status_row(row) for row in statuses]
            for empty in ('',None):
                actual=sum(row[3]==server and len(str(row[24] if row[24] else empty or ''))>0
                           for row in raw_rows)
                self.assertEqual(actual,expected_count)


if __name__=='__main__':unittest.main()
