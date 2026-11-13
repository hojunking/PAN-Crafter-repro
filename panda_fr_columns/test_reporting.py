import json
import unittest
from types import SimpleNamespace
from directshift48.reporting import HEADERS, INDEX
from sheet_publish_v2.transport import TransportError, _typed
from .schema import HEADERS67, collapse, expand, layout, translate, migration_requests
from .adapter import GoogleSheets
from .publication import extras, original_rows, without_companions, cell_equal


class MappingTests(unittest.TestCase):
    def test_full_row_roundtrip_and_unknown_header(self):
        row=list(range(64))
        self.assertEqual(collapse(expand(row,[9,8,7])),row)
        self.assertEqual(layout(HEADERS67),67)
        self.assertEqual(layout(list(HEADERS)+['']*3),64)
        with self.assertRaises(TransportError):layout(['wrong']*67)

    def test_legacy_updates_preserve_corrected_metrics(self):
        physical=expand(list(range(64)),['keepH','keepI','keepJ'])
        values=[{'userEnteredValue':_typed(f'v{i}')} for i in range(64)]
        req={'updateCells':{'range':{'sheetId':261001010,'startRowIndex':9,'endRowIndex':10,
             'startColumnIndex':0,'endColumnIndex':64},'rows':[{'values':values}],
             'fields':'userEnteredValue,note'}}
        for item in translate(req):
            b=item['updateCells'];r=b['range']
            physical[r['startColumnIndex']:r['endColumnIndex']]=[
                next(iter(c['userEnteredValue'].values())) for c in b['rows'][0]['values']]
        self.assertEqual(physical[7:10],['keepH','keepI','keepJ'])
        self.assertEqual(collapse(physical),[f'v{i}' for i in range(64)])
        self.assertEqual(physical[10],'v7') # RR ERGAS moves H -> K.
        self.assertEqual(physical[62],'v59') # Result_ID BH -> BK.

    def test_partial_ranges_formats_and_unsupported_mutations(self):
        req={'repeatCell':{'range':{'sheetId':261001010,'startColumnIndex':3,'endColumnIndex':25},
                           'cell':{},'fields':'userEnteredFormat.numberFormat'}}
        ranges=[r['repeatCell']['range'] for r in translate(req)]
        self.assertEqual([(r['startColumnIndex'],r['endColumnIndex']) for r in ranges],[(3,7),(10,28)])
        req={'appendDimension':{'sheetId':261001010,'dimension':'ROWS','length':3}}
        self.assertEqual(translate(req),[req])
        with self.assertRaises(TransportError):translate({'deleteDimension':{'range':{'dimension':'ROWS'}}})

    def test_atomic_dual_update_uses_same_row_id(self):
        adapter=object.__new__(GoogleSheets);adapter.layouts={261001010:67}
        adapter.extra_cells={'RID':[{'userEnteredValue':_typed(x)} for x in (.98,.01,.01)]}
        captured=[];adapter.physical_request=lambda *a,**kw:captured.append(kw['json'])
        row=['']*64;row[59]='RID'
        req={'updateCells':{'range':{'sheetId':261001010,'startColumnIndex':0,'endColumnIndex':64,
            'startRowIndex':8,'endRowIndex':9},'rows':[{'values':[{'userEnteredValue':_typed(v)} for v in row]}],
            'fields':'userEnteredValue,note'}}
        from .schema import BOOK
        adapter._request('POST',adapter.BASE+BOOK+':batchUpdate',json={'requests':[req]})
        b=captured[0]['requests'][0]['updateCells']
        self.assertEqual(b['range']['endColumnIndex'],67)
        self.assertEqual(len(b['rows'][0]['values']),67)
        self.assertEqual(b['rows'][0]['values'][7]['userEnteredValue']['numberValue'],.98)
        self.assertEqual(b['rows'][0]['values'][62]['userEnteredValue']['stringValue'],'RID')

    def test_migration_only_expected_header_layout(self):
        sid=261001010
        tab={'properties':{'sheetId':sid,'title':'WV3-ablations','gridProperties':{'rowCount':11000}},
             'merges':[dict(sheetId=sid,startRowIndex=3,endRowIndex=4,startColumnIndex=a,endColumnIndex=b)
                       for a,b in ((3,7),(7,15))]}
        requests=migration_requests(tab)
        self.assertEqual(sum('insertDimension' in r for r in requests),1)
        self.assertFalse(any('deleteDimension' in r for r in requests))
        tab['merges']=[]
        with self.assertRaises(TransportError):migration_requests(tab)


class EvidenceTests(unittest.TestCase):
    def test_sheets_empty_value_normalization(self):
        self.assertTrue(cell_equal({'note':'pending'},{'userEnteredValue':_typed(''),'note':'pending'}))
        self.assertFalse(cell_equal({'note':'wrong'},{'userEnteredValue':_typed(''),'note':'pending'}))
        self.assertFalse(cell_equal({'userEnteredValue':_typed(0)},{'userEnteredValue':_typed('')}))

    def setUp(self):
        self.row=['']*64
        for key,v in {'Run_ID':'RUN','Selected_step':1000,'Result_ID':'RID','Checkpoint_SHA':'sha',
                      'Campaign_ID':'CAMP','Server':'s1'}.items():self.row[INDEX[key]]=v
        self.record={'step':1000,'evidence_path':'evaluation_0001000.json','evidence_sha256':'evsha',
             'fr':{'metrics':{'D_lambda':.012345}},
             'fr_corrected_pan':{'metrics':{'HQNR':.9654321,'D_s':.0222222,'D_lambda':.012345}}}

    def test_verified_record_and_pending_values(self):
        v={'rows':[self.row],'proof':{('RUN',1000):self.record}}
        cells=extras(v)['RID']
        self.assertEqual([c['userEnteredValue']['numberValue'] for c in cells],[.9654,.0222,.0123])
        note=json.loads(cells[0]['note'])
        self.assertEqual(note['step'],1000)
        self.assertEqual(note['checkpoint_sha256'],'sha')
        self.assertEqual(note['raw_metrics']['HQNR'],.9654321)
        self.row[INDEX['Selected_step']]=0
        self.assertEqual(extras(v)['RID'][0]['userEnteredValue'],{'stringValue':''})

    def test_inconsistent_reference_rejected(self):
        self.record['fr_corrected_pan']['metrics']['D_lambda']=.9
        with self.assertRaises(ValueError):extras({'rows':[self.row],'proof':{('RUN',1000):self.record}})

    def test_companion_retirement_keeps_foreign_rows_and_indices(self):
        self.row[INDEX['Original description / notes']]=json.dumps({'publication':{'slot':'ROW0_CORRECTED'}})
        called=[]
        pub=SimpleNamespace(plan=SimpleNamespace(CAMPAIGN_ID='CAMP'),verify_owned=lambda *a:called.append(a[0]))
        foreign=['other']*64;snapshot={'rows':[foreign,self.row,foreign]}
        result=without_companions(snapshot,{},'s1',pub)
        self.assertEqual(len(result['rows']),3)
        self.assertEqual(result['rows'],[foreign,['']*64,foreign])
        self.assertEqual(called,[self.row])
        self.assertEqual(snapshot['rows'][1],self.row) # no source mutation


if __name__=='__main__':unittest.main()
