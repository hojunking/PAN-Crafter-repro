#!/usr/bin/env python
"""FH12 upload tests: local fixtures and fake Sheets only, never network/GPU."""
import copy
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fh12 import upload as U
from fh12.common import atomic_json,object_sha,sha256,read_json
from fh12.plan import CAMPAIGN_ID,build_config,teacher_for


class FakeSheet:
    def __init__(self,headers,gid):
        self.id=gid; self.col_count=len(headers); self.row_count=4
        self.cells={2:['existing']*len(headers),3:headers.copy(),4:['historic-run','preserve-historical']}
        self.corrupt=False; self.writes=[]

    def row_values(self,row,**kwargs):
        values=self.cells.get(row,[]).copy()
        if self.corrupt and kwargs and row>=5: values[0]='corrupted'
        return values

    def get_all_values(self):
        return [self.cells.get(row,[]).copy() for row in range(1,max(self.cells)+1)]

    def add_cols(self,n): self.col_count+=n
    def add_rows(self,n): self.row_count+=n

    def batch_update(self,edits,**kwargs):
        self.writes.extend(copy.deepcopy(edits))
        for edit in edits:
            match=re.fullmatch(r'([A-Z]+)(\d+)',edit['range']); col=0
            for letter in match[1]: col=col*26+ord(letter)-64
            row=self.cells.setdefault(int(match[2]),[])
            row.extend(['']*max(0,col-len(row))); row[col-1]=edit['values'][0][0]


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fh12-upload-test-'); self.root=Path(self.temp.name)
        self.case=teacher_for('s3'); self.run=self.case.run_id; self.cfg=build_config(self.case)
        self.wd=self.root/'work_dir'/self.run; (self.wd/'meta').mkdir(parents=True)
        (self.wd/'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        server=self.root/'gspread/server.txt'; server.parent.mkdir(); server.write_text('s3(5090)\n')
        self.source={'git_release':'fixture-release','content_sha256':'fixture-source'}
        self.camp=self.root/'work_dir/_fh12/s3'
        self.data={'splits':{s:{'lpan_sha256':'LP-'+s} for s in ('train','val','rr','fr')}}
        atomic_json(self.camp/'dataset_manifest.json',self.data)
        atomic_json(self.camp/'lpan_manifest.json',{'recipe':{'id':'test-recipe','phase_id':'offset2'}})
        self.context=dict(run_id=self.run,campaign_id=CAMPAIGN_ID,config_sha256=object_sha(self.cfg),
                          source_identity=self.source,data_sha256=object_sha(self.data),official_complete=True)
        rr=dict(ergas=2.035,scc=.988,psnr=38.,sam=2.7,q8=.922,ssim=.976,official_complete=True)
        fr=dict(hqnr=.959,d_lambda=.016,d_s=.025)
        for name,label,step in [('raw_max','RAW_MAX',1010),('target_selection','TARGET',2020),
                                ('exact50k','EXACT50K',50000),('rr_val_selected','RR_VAL_SELECTED',3030),
                                ('e_min_diag','E_MIN_DIAG50',2020)]:
            folder=self.wd/'candidates'/str(step); folder.mkdir(parents=True,exist_ok=True)
            (folder/'model.safetensors').write_bytes(f'fixture {step}'.encode())
            identity=dict(update=step,model_sha256=sha256(folder/'model.safetensors'),config_sha256=object_sha(self.cfg))
            atomic_json(folder/'identity.json',identity)
            report=dict(self.context,selection_id=label,step=step,checkpoint_sha256=identity['model_sha256'],
                        checkpoint_identity=identity,rr=dict(rr),fr=dict(fr),val_ergas=2.1)
            if name=='target_selection': report.update(n_evaluated=50,n_eligible=3,target_status='official',joint_pass=True)
            if name=='e_min_diag': report.update(n_evaluated=50)
            atomic_json(self.wd/'official'/f'{name}.json',report)
        atomic_json(self.wd/'official/postrun_status.json',dict(self.context,actual_updates=50000,sheet_uploaded=False))
        atomic_json(self.wd/'official/profile.json',dict(config_sha256=object_sha(self.cfg),source_identity=self.source,
            params_m=3.,flops_g=100.,infer_ms=9.,mem_mb=200.,flops_is_estimate=True,
            flops_convention='test estimated MAC convention',scope='A+frontend+U'))
        self.gu=types.SimpleNamespace(CRED='account.json',SHEET='mock-only',ORIGIN_ROW=2)
        self.requested=[]

    def tearDown(self): self.temp.cleanup()

    def api(self,ws):
        def worksheet(name): self.requested.append(name); return ws
        book=types.SimpleNamespace(worksheet=worksheet)
        return types.SimpleNamespace(service_account=lambda **_:types.SimpleNamespace(open=lambda _:book))

    def invoke(self,ws):
        with patch.object(U,'legacy_constants',return_value=self.gu),patch.dict(sys.modules,{'gspread':self.api(ws)}):
            return U.upload_run(self.run,self.root)

    def test_official_values_all9_metrics_same_checkpoint(self):
        values=U.row_values(self.run,self.root)
        for prefix in ('RAW_MAX','Target','Exact50K','RR_VAL_SELECTED','E_MIN_DIAG50'):
            for label in list(U.RR_LABELS)+list(U.FR_LABELS): self.assertIn(prefix+' '+label,values)
        self.assertEqual(values['ERGAS↓'],values['RAW_MAX ERGAS↓'])
        self.assertEqual(values['HQNR↑'],values['RAW_MAX HQNR(raw)↑'])
        self.assertEqual(values['FH12 teacher seed'],71001)
        self.assertEqual(values['FH12 calibration status'],'PENDING_LOCAL_CALIBRATION')
        self.assertNotIn('NOA',values)
        self.assertNotIn('Q4↑',values)
        self.assertIn('FLOPs estimate',values['Notes'])

    def test_no_eligible_target_blank_only_target(self):
        path=self.wd/'official/target_selection.json'
        atomic_json(path,dict(self.context,selection_id='TARGET',selection=None,n_eligible=0,
                             n_evaluated=50,target_status='no_eligible',joint_pass=False))
        values=U.row_values(self.run,self.root)
        self.assertEqual(values['Target ERGAS↓'],'')
        self.assertEqual(values['Target checkpoint SHA256'],'')
        self.assertNotEqual(values['Exact50K ERGAS↓'],'')
        self.assertTrue(values['Target official'])

    def test_incomplete_status_rejected(self):
        path=self.wd/'official/postrun_status.json'; status=read_json(path);status['official_complete']=False
        atomic_json(path,status)
        with self.assertRaises(ValueError): U.row_values(self.run,self.root)

    def test_checkpoint_tamper_rejected(self):
        (self.wd/'candidates/1010/model.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA'): U.row_values(self.run,self.root)

    def test_context_mismatch_rejected(self):
        path=self.wd/'official/exact50k.json'; report=read_json(path);report['source_identity']={}
        atomic_json(path,report)
        with self.assertRaisesRegex(ValueError,'identity'): U.row_values(self.run,self.root)

    def test_dynamic_header_compound_upsert_and_preservation(self):
        ws=FakeSheet(['Run','Notes','NOA metric','Target ERGAS↓','Q4↑','HQNR↑'],U.GIDS['s3'])
        before=copy.deepcopy(ws.cells[4])
        first=self.invoke(ws); second=self.invoke(ws)
        self.assertEqual(first['row'],second['row']); self.assertEqual(first['row'],5)
        self.assertTrue(first['readback_verified']); self.assertEqual(ws.cells[4],before)
        self.assertEqual(self.requested,['WV3-s3(5090)']*2)
        labels=U.label_map(ws.cells[3]); self.assertEqual(labels['Target ERGAS↓'],4)
        self.assertEqual(ws.cells[5][labels['Target ERGAS↓']-1],2.035)
        self.assertEqual(ws.cells[5][labels['NOA metric']-1],'')
        self.assertEqual(ws.cells[5][labels['Q4↑']-1],'')
        self.assertIn('RR_VAL_SELECTED ERGAS↓',labels)

    def test_same_run_other_campaign_never_overwritten(self):
        ws=FakeSheet(['Run','FH12 campaign','FH12 run id'],U.GIDS['s3'])
        ws.cells[4]=[self.run,'OTHER_CAMPAIGN',self.run]
        receipt=self.invoke(ws)
        self.assertEqual(receipt['row'],5)
        self.assertEqual(ws.cells[4],[self.run,'OTHER_CAMPAIGN',self.run])

    def test_blank_run_historical_row_preserved(self):
        ws=FakeSheet(['Run','Notes'],U.GIDS['s3']);ws.cells[5]=['','historic annotation']
        self.assertEqual(self.invoke(ws)['row'],6)
        self.assertEqual(ws.cells[5],['','historic annotation'])

    def test_gid_mismatch_no_writes(self):
        ws=FakeSheet(['Run'],123)
        with self.assertRaisesRegex(ValueError,'gid'): self.invoke(ws)
        self.assertEqual(ws.writes,[])

    def test_duplicate_header_no_writes(self):
        ws=FakeSheet(['Run','Run'],U.GIDS['s3'])
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.invoke(ws)
        self.assertEqual(ws.writes,[])

    def test_duplicate_compound_key_no_writes(self):
        ws=FakeSheet(['Run','FH12 campaign','FH12 run id'],U.GIDS['s3'])
        ws.cells[4]=[self.run,CAMPAIGN_ID,self.run];ws.cells[5]=ws.cells[4].copy()
        with self.assertRaisesRegex(ValueError,'Duplicate FH12'): self.invoke(ws)
        self.assertEqual(ws.writes,[])

    def test_readback_failure_no_receipt(self):
        ws=FakeSheet(['Run'],U.GIDS['s3']);ws.corrupt=True
        with self.assertRaisesRegex(ValueError,'readback'): self.invoke(ws)
        self.assertFalse((self.wd/'official/upload_receipt.json').exists())

    def test_server_mismatch_before_network(self):
        (self.root/'gspread/server.txt').write_text('s1\n')
        with patch.object(U,'legacy_constants',side_effect=AssertionError('network must not be reached')):
            with self.assertRaisesRegex(ValueError,'another server'): U.upload_run(self.run,self.root)

    def test_numeric_readback_not_rounded(self):
        self.assertTrue(U.same_cell('0.958612345678',.958612345678))
        self.assertFalse(U.same_cell('0.9586',.958612345678))


if __name__=='__main__': unittest.main(verbosity=2)
