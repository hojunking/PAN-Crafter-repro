"""Two-cell banner migration never writes old WV3 or foreign text."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from panda_s1loss9 import reporting as R
from panda_s1loss9.best_publication import header as H


class Adapter:
    BASE='https://example.invalid/'
    def __init__(self,server):
        self.destination=R.destination(server);self.writes=[]
        self.cells={'D1':{'userEnteredValue':{'stringValue':H.OLD_D1[R.DATASETS[server]]},'note':'preserve note'},
                    'A3':{'userEnteredValue':{'stringValue':H.OLD_A3}}}
    def _request(self,method,url,**kwargs):
        if method=='GET':
            title=self.destination['sheet_title']
            assert kwargs['params']['ranges']==[f"'{title}'!D1:D1",f"'{title}'!A3:A3"]
            return {'spreadsheetId':self.destination['spreadsheet_id'],'sheets':[
                {'properties':{'sheetId':261001010,'title':'WV3-ablations'}},
                {'properties':{'sheetId':self.destination['sheet_id'],'title':title},
                 'data':[{'startRow':row,'startColumn':column,'rowData':[{'values':[deepcopy(self.cells[name])]}]}
                         for name,(row,column) in H.CELLS.items()]}]}
        assert method=='POST'
        self.writes.append(deepcopy(kwargs['json']))
        for req in kwargs['json']['requests']:
            u=req['updateCells'];r=u['range']
            assert r['sheetId']==self.destination['sheet_id']
            assert u['fields']=='userEnteredValue'
            assert r['endRowIndex']==r['startRowIndex']+1
            assert r['endColumnIndex']==r['startColumnIndex']+1
            name={rc:n for n,rc in H.CELLS.items()}[(r['startRowIndex'],r['startColumnIndex'])]
            self.cells[name]['userEnteredValue']=deepcopy(u['rows'][0]['values'][0]['userEnteredValue'])
        return {}


class BannerTests(unittest.TestCase):
    def test_both_sensor_targets_only_two_cells_verified_and_idempotent(self):
        for server in R.DATASETS:
            adapter=Adapter(server)
            with tempfile.TemporaryDirectory() as root:
                report=H.reconcile_header(adapter,adapter.destination,root,server,apply=True)
                self.assertTrue(report['readback_verified'])
                self.assertEqual(report['changed_cells'],['D1','A3'])
                self.assertEqual(adapter.cells['D1']['note'],'preserve note')
                self.assertEqual(len(adapter.writes),1)
                report=H.reconcile_header(adapter,adapter.destination,root,server,apply=True)
                self.assertEqual(report['changed_cells'],[])
                self.assertEqual(len(adapter.writes),1)
                self.assertEqual(adapter.cells['D1']['userEnteredValue']['stringValue'],H.desired(server)['D1'])
                self.assertTrue((Path(root)/'target_banner_verified.json').exists())

    def test_existing_sensor5_old_and_current_gf2_targets_are_accepted(self):
        for previous in (.548,.522):
            adapter=Adapter('s3')
            adapter.cells['D1']['userEnteredValue']['stringValue']=(
                f'SENSOR5 기준: HQNR ≥ 0.9645 AND ERGAS ≤ {previous:.3f} · 같은 모델 상태의 FR20 + RR20')
            adapter.cells['A3']['userEnteredValue']['stringValue']=H.desired('s3')['A3']
            with tempfile.TemporaryDirectory() as root:
                value=H.reconcile_header(adapter,adapter.destination,root,'s3',apply=True)
                self.assertTrue(value['readback_verified'])
                self.assertEqual(value['changed_cells'],['D1'] if previous==.548 else [])

    def test_foreign_header_destination_and_controls_never_overwritten(self):
        for mutation in ('foreign_text','WV3','formula','chip'):
            adapter=Adapter('s1');destination=deepcopy(adapter.destination)
            if mutation=='foreign_text':adapter.cells['A3']['userEnteredValue']['stringValue']='New user instruction'
            elif mutation=='WV3':destination.update(sheet_title='WV3-ablations',sheet_id=261001010)
            elif mutation=='formula':adapter.cells['A3']['userEnteredValue']={'formulaValue':'="test"'}
            else:adapter.cells['A3']['chipRuns']=[{'startIndex':0}]
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                H.reconcile_header(adapter,destination,root,'s1',apply=True)
            self.assertEqual(adapter.writes,[])

    def test_dry_run_makes_no_write(self):
        adapter=Adapter('s3')
        with tempfile.TemporaryDirectory() as root:
            result=H.reconcile_header(adapter,adapter.destination,root,'s3',apply=False)
            self.assertFalse(result['readback_verified']);self.assertEqual(adapter.writes,[])


if __name__=='__main__':unittest.main()
