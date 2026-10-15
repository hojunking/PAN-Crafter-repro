"""Append-only integration safety. Synthetic workbook; never authenticates."""
import copy
import json
import unittest
from unittest.mock import patch

from reporting_bridge import rb_b01_contract as contract
from reporting_bridge import rb_b01_sheets as base
from reporting_bridge.rb_b01_summary import summarize
from reporting_bridge.tests.test_sheets import FakeSheets


class ExistingWorkbook(FakeSheets):
    """A pre-existing v1-migrated workbook, not produced by a live setup call."""
    def __init__(self):
        super().__init__()
        self.merges={};self.protected={};self.lose_next_block_reply=False
        old=base.SheetsBridge(self)
        for i,(title,header) in enumerate(old._schemas().items()):
            self.tabs[title]={'sheetId':269300000+i,'title':title,'index':len(self.tabs),
                'hidden':title.startswith('_'),'gridProperties':{'rowCount':2600,'columnCount':len(header)}}
            self.cells[title]={(0,j):{'userEnteredValue':{'stringValue':h}} for j,h in enumerate(header)}
            self.cells[title][(0,0)]['note']=old._owner_note(title)
        self.cells['_records'][(1,0)]['userEnteredValue']['formulaValue']='=LET(allrows,legacy,'+base.NEW_RETURN+')'
        self.cells['RB-B01'][(1,0)]={'userEnteredValue':{'stringValue':'PRESERVE prior public summary'},'note':'old public note'}
        self.cells['RB02-curves'][(1,0)]={'userEnteredValue':{'formulaValue':'=OLD_PUBLIC_CURVES()'}}
        self.cells['ablations'][(9,30)]={'userEnteredValue':{'stringValue':'Existing far-column table'},
            'note':'Existing user note','userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.000'}}}

    def metadata(self):
        value=super().metadata()
        for sheet in value['sheets']:
            sid=sheet['properties']['sheetId']
            sheet['merges']=copy.deepcopy(self.merges.get(sid,[]))
            sheet['protectedRanges']=copy.deepcopy(self.protected.get(sid,[]))
        return value

    def grid(self,ranges):
        value=super().grid(ranges)
        for sheet in value['sheets']:
            sheet['merges']=copy.deepcopy(self.merges.get(sheet['properties']['sheetId'],[]))
        return value

    def batch_update(self,requests):
        # No executable code path in this test is allowed to create/delete/move
        # a worksheet, even through a fake adapter.
        forbidden={'addSheet','deleteSheet','duplicateSheet','updateSheetProperties',
                   'insertDimension','deleteDimension','moveDimension'}
        if any(forbidden.intersection(request) for request in requests):
            raise AssertionError('Forbidden sheet/layout mutation in append-only mode')
        super().batch_update(requests)
        block_written=False
        for request in requests:
            if 'mergeCells' in request:
                rg=request['mergeCells']['range'];self.merges.setdefault(rg['sheetId'],[]).append(copy.deepcopy(rg))
            elif 'updateCells' in request:
                value=request['updateCells'];rg=value.get('range',value.get('start',{}))
                if rg.get('sheetId')==1697831403:block_written=True
                title=next(t for t,p in self.tabs.items() if p['sheetId']==rg['sheetId'])
                r=rg.get('startRowIndex',rg.get('rowIndex',0));c=rg.get('startColumnIndex',rg.get('columnIndex',0))
                fields=value.get('fields','').split(',')
                for i,row in enumerate(value.get('rows',[])):
                    for j,cell in enumerate(row.get('values',[])):
                        for key in fields:
                            if '.' not in key and key not in cell:self.cells[title].setdefault((r+i,c+j),{}).pop(key,None)
        if block_written and self.lose_next_block_reply:
            self.lose_next_block_reply=False
            raise TimeoutError('SYNTHETIC response lost after committed owned block')


def empty_prepared():
    statuses=[dict(row,verification_status='REMOTE_EVIDENCE_NOT_SUPPLIED',
        training_status='UNKNOWN',native_status='FILES_NOT_PROVIDED',errors=[],
        error='Synthetic evidence not supplied',upload_status='NOT_AVAILABLE')
        for row in contract.registry()]
    return dict(native_records=[],stress_records=[],statuses=statuses,
                summary=summarize([],[],statuses),package_hashes=[])


class AppendOnlyTests(unittest.TestCase):
    def setUp(self):
        from reporting_bridge.rb_b01_append import AppendOnlySheetsBridge
        self.api=ExistingWorkbook();self.bridge=AppendOnlySheetsBridge(self.api)
        self.prepared=empty_prepared()

    def test_inspect_and_plan_do_not_write(self):
        snapshot=self.bridge.inspect();plan=self.bridge.plan(self.prepared,snapshot)
        self.assertEqual(self.api.writes,[])
        self.assertEqual(plan['new_sheet_count'],0)

    def test_missing_target_is_not_created(self):
        self.api.tabs.pop('_rb01_s3');self.api.cells.pop('_rb01_s3')
        with self.assertRaisesRegex(ValueError,'TARGET_MISSING'):self.bridge.inspect()
        self.assertEqual(self.api.writes,[])

    def test_unmigrated_union_is_not_repaired(self):
        original='=LET(allrows,legacy,'+base.OLD_RETURN+')'
        self.api.cells['_records'][(1,0)]['userEnteredValue']['formulaValue']=original
        with self.assertRaisesRegex(ValueError,'union missing'):self.bridge.inspect()
        self.assertEqual(self.api.cells['_records'][(1,0)]['userEnteredValue']['formulaValue'],original)
        self.assertEqual(self.api.writes,[])

    def test_setup_validates_without_writing(self):
        snapshot=self.bridge.inspect();plan=self.bridge.plan(self.prepared,snapshot)
        result=self.bridge.setup(plan,snapshot,apply=True)
        self.assertEqual(result['writes'],0);self.assertEqual(result['new_sheet_count'],0)
        self.assertEqual(self.api.writes,[])

    def test_append_scans_far_columns_notes_and_effective_spills(self):
        self.api.cells['ablations'][(29,63)]={'note':'Reserved user note in BL30'}
        self.api.cells['ablations'][(39,8)]={'effectiveValue':{'numberValue':4.2}}
        snapshot=self.bridge.inspect();plan=self.bridge.plan(self.prepared,snapshot)
        self.assertEqual(plan['placement']['start_row'],43)
        self.assertEqual(plan['placement']['historical_last_used_row'],40)

    def test_unbounded_reservation_blocks_append(self):
        self.api.protected[1697831403]=[{'protectedRangeId':1,'range':{
            'sheetId':1697831403,'startRowIndex':200,'startColumnIndex':0,'endColumnIndex':1}}]
        snapshot=self.bridge.inspect()
        with self.assertRaisesRegex(ValueError,'UNBOUNDED_RESERVED'):
            self.bridge.plan(self.prepared,snapshot)
        self.assertEqual(self.api.writes,[])

    def test_upload_preserves_public_tabs_union_and_existing_ablations(self):
        snapshot=self.bridge.inspect()
        public={t:copy.deepcopy(self.api.cells[t]) for t in ('paper','유의미한결과','RB-B01','RB02-curves','_records')}
        old_cell=copy.deepcopy(self.api.cells['ablations'][(9,30)])
        count=len(self.api.tabs)
        result=self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(result['new_sheet_count'],0);self.assertEqual(len(self.api.tabs),count)
        self.assertEqual(result['placement']['start_row'],13)
        self.assertEqual(self.api.cells['ablations'][(9,30)],old_cell)
        for title,values in public.items():self.assertEqual(self.api.cells[title],values,title)
        receipt=self.bridge.verify(self.prepared,snapshot)
        self.assertTrue(receipt['regression']['outside_owned_block_unchanged'])
        self.assertEqual(receipt['students'],0)
        for request in self.api.writes:
            if isinstance(request,dict) and 'updateCells' in request:
                body=request['updateCells'];sid=body.get('range',body.get('start'))['sheetId']
                self.assertEqual(sid,1697831403)

    def test_same_payload_is_zero_duplicate_block_and_zero_new_rows(self):
        first=self.bridge.upload(self.prepared,apply=True)
        count=len(self.api.writes)
        again=self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(again['new_observations'],0)
        self.assertEqual(again['block_action'],'NO_OP')
        self.assertFalse(again['block_written'])
        self.assertEqual(again['placement']['range'],first['placement']['range'])
        self.assertEqual(len(self.api.writes),count)

    def test_manual_owned_value_or_note_blocks_before_source_write(self):
        first=self.bridge.upload(self.prepared,apply=True)
        key=(first['placement']['start_row']+4,0)
        for field,value in [('userEnteredValue',{'stringValue':'Manual user edit'}),('note','Manual user note')]:
            previous=copy.deepcopy(self.api.cells['ablations'][key])
            self.api.cells['ablations'][key][field]=value
            count=len(self.api.writes)
            with self.assertRaisesRegex(ValueError,'manually edited'):
                self.bridge.upload(self.prepared,apply=True)
            self.assertEqual(len(self.api.writes),count)
            self.api.cells['ablations'][key]=previous

    def test_lost_reply_recovers_existing_atomic_owner_block(self):
        self.api.lose_next_block_reply=True
        first=self.bridge.upload(self.prepared,apply=True)
        again=self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(first['placement']['range'],again['placement']['range'])
        self.assertEqual(again['block_action'],'NO_OP')
        self.assertEqual(again['new_observations'],0)

    def test_outside_content_edits_detected(self):
        snapshot=self.bridge.inspect();self.bridge.upload(self.prepared,apply=True)
        self.api.cells['ablations'][(9,30)]['note']='Unrelated modified content'
        with self.assertRaisesRegex(ValueError,'protected table/outside'):
            self.bridge.verify(self.prepared,snapshot)

    def test_any_sheet_creation_after_snapshot_is_regression(self):
        snapshot=self.bridge.inspect()
        self.api.tabs['unexpected']={'sheetId':987654321,'title':'unexpected','index':len(self.api.tabs),
            'gridProperties':{'rowCount':10,'columnCount':10}}
        self.api.cells['unexpected']={}
        with self.assertRaisesRegex(ValueError,'worksheet count/IDs'):
            self.bridge._regression(snapshot)

    def test_google_write_boundary_blocks_v1_and_public_paths_before_auth(self):
        adapter=base.GoogleSheetsAdapter()
        forbidden=[{'addSheet':{'properties':{'title':'bad'}}},
            {'duplicateSheet':{'sourceSheetId':1697831403}},
            {'deleteSheet':{'sheetId':269300006}},
            {'insertDimension':{'range':{'sheetId':1697831403,'dimension':'ROWS','startIndex':0,'endIndex':1}}},
            {'updateSheetProperties':{'properties':{'sheetId':1697831403,'hidden':True},'fields':'hidden'}},
            {'updateCells':{'range':{'sheetId':260928001},'rows':[],'fields':'userEnteredValue'}},
            {'updateCells':{'range':{'sheetId':269300006},'rows':[],'fields':'userEnteredValue'}},
            {'appendDimension':{'sheetId':1697831403,'dimension':'COLUMNS','length':2}}]
        for request in forbidden:
            with self.subTest(request=request),self.assertRaisesRegex(ValueError,'OUTPUT_POLICY'):
                adapter.batch_update([request])
        for title in ('paper','ablations','_records','WV3-main','RB-B01','RB02-curves'):
            with self.subTest(title=title),self.assertRaisesRegex(ValueError,'OUTPUT_POLICY'):
                adapter.write_raw([(f"'{title}'!A1",[['=literal']])])
        self.assertIsNone(adapter._book)

    def test_native_only_alias_does_not_invent_stress_or_extra_student(self):
        from reporting_bridge.tests.test_rb_b01_contract_summary import synthetic_entry
        self.prepared['native_records']=contract.canonical_native(synthetic_entry())
        self.prepared['statuses'][0].update(verification_status='EVIDENCE_VERIFIED',
            native_status='NATIVE_EVAL_COMPLETE',native_selectors=2,actual_updates=50000,error='')
        self.prepared['summary']=summarize(self.prepared['native_records'],[],self.prepared['statuses'])
        snapshot=self.bridge.inspect();first=self.bridge.upload(self.prepared,apply=True)
        verified=self.bridge.verify(self.prepared,snapshot)
        self.assertEqual(verified['native_rows'],2);self.assertEqual(verified['students'],1)
        self.assertEqual(verified['stress_points'],0)
        block=self.api.values(first['placement']['range'])
        primary=[row for row in block if len(row)>7 and row[0]=='QFULL' and row[2]=='EXACT_50000']
        self.assertTrue(primary)
        self.assertEqual(primary[0][3:8],[1,6,.95,'',1])
        again=self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(again['new_observations'],0);self.assertEqual(again['unchanged_observations'],2)

    def test_manual_owned_merge_is_conflict(self):
        first=self.bridge.upload(self.prepared,apply=True);p=first['placement']
        self.api.merges[1697831403]=[{'sheetId':1697831403,
            'startRowIndex':p['start_row']+4,'endRowIndex':p['start_row']+5,
            'startColumnIndex':0,'endColumnIndex':2}]
        count=len(self.api.writes)
        with self.assertRaisesRegex(ValueError,'merge'):
            self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(len(self.api.writes),count)

    def test_user_owned_style_is_preserved_on_noop(self):
        first=self.bridge.upload(self.prepared,apply=True);key=(first['placement']['start_row']+4,0)
        style={'backgroundColor':{'red':.8,'green':.1,'blue':.2}}
        self.api.cells['ablations'][key]['userEnteredFormat']=style
        again=self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(again['block_action'],'NO_OP')
        self.assertEqual(self.api.cells['ablations'][key]['userEnteredFormat'],style)

    def test_ownership_note_manual_annotation_is_not_silently_erased(self):
        first=self.bridge.upload(self.prepared,apply=True);key=(first['placement']['start_row']-1,0)
        marker=json.loads(self.api.cells['ablations'][key]['note'])
        marker['user_annotation']='Manual note to preserve'
        self.api.cells['ablations'][key]['note']=json.dumps(marker,ensure_ascii=False)
        count=len(self.api.writes)
        with self.assertRaisesRegex(ValueError,'SCHEMA_CONFLICT|OWNER'):
            self.bridge.upload(self.prepared,apply=True)
        self.assertEqual(len(self.api.writes),count)

    def test_existing_marker_field_changes_cannot_force_continuation(self):
        first=self.bridge.upload(self.prepared,apply=True);key=(first['placement']['start_row']-1,0)
        original=self.api.cells['ablations'][key]['note']
        for field,value in [('source_hash','f'*64),('range',
                f"'ablations'!A{first['placement']['start_row']}:AE{first['placement']['end_row']-1}")]:
            marker=json.loads(original);marker[field]=value
            self.api.cells['ablations'][key]['note']=json.dumps(marker,ensure_ascii=False)
            count=len(self.api.writes)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'SCHEMA_CONFLICT|OWNER'):
                self.bridge.upload(self.prepared,apply=True)
            self.assertEqual(len(self.api.writes),count)
            self.api.cells['ablations'][key]['note']=original


if __name__=='__main__':unittest.main()
