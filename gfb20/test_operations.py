import copy
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gfb20 import controller
from gfb20.common import read_json
from gfb20.deployment import frozen_checkout
from gfb20.plan import CAMPAIGN_ID,case_for,build_config,validate_config
from gfb20.upload import apply_upsert,flatten_summary


class Worksheet:
    title='GF2-B20-s3';id=123;col_count=1;row_count=1
    def __init__(self):self.rows=[];self.formats=[]
    def row_values(self,n,**kwargs):return self.rows[n-1].copy() if n<=len(self.rows) else []
    def add_cols(self,n):self.col_count+=n
    def add_rows(self,n):self.row_count+=n
    def get_all_values(self,**kwargs):return copy.deepcopy(self.rows)
    def update(self,range_name,values,**kwargs):
        import re
        letters,num=re.fullmatch(r'([A-Z]+)(\d+)',range_name).groups()
        col=0
        for c in letters:col=col*26+ord(c)-64
        row=int(num)
        while len(self.rows)<row:self.rows.append([])
        while len(self.rows[row-1])<col-1+len(values[0]):self.rows[row-1].append('')
        self.rows[row-1][col-1:col-1+len(values[0])]=values[0]
    def batch_format(self,formats):self.formats.extend(formats)


class OperationsTests(unittest.TestCase):
    def test_build_has36_configs_and_no_clock_or_training(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(controller,'verify_sources'):
            result=controller.build(Path(directory))
            self.assertEqual(result['configs'],36)
            self.assertFalse(result['clock_created'])
            self.assertFalse((Path(directory)/'work_dir').exists())
            paths=list((Path(directory)/'config/gfb20').glob('*.yaml'))
            self.assertEqual(len(paths),36)
            for path in paths:validate_config(read_json(path))

    def test_clock_write_is_immutable_across_servers_and_reruns(self):
        with tempfile.TemporaryDirectory() as directory:
            first=controller.write_window(directory,'2026-09-22T00:00:00Z')
            self.assertEqual(first,controller.write_window(directory,'2026-09-22T09:00:00+09:00'))
            with self.assertRaises(ValueError):controller.write_window(directory,'2026-09-22T01:00:00Z')
            self.assertEqual(controller.shared_window(directory).to_dict(),first)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):controller.shared_window(directory)

    def test_protected_servers_cannot_create_frozen_release(self):
        with tempfile.TemporaryDirectory() as directory:
            for server in ('s1','s2'):
                with self.assertRaises(ValueError):frozen_checkout(directory,server)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_future_clock_cannot_start_uncounted_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            future=dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)
            controller.write_window(directory,future)
            with patch.object(controller,'register_runtime') as register:
                with self.assertRaises(ValueError):controller.start(directory,'s3')
                register.assert_not_called()

    def test_new_tab_upsert_full_precision_and_readback(self):
        ws=Worksheet()
        values=dict(campaign_id=CAMPAIGN_ID,run_id='x',server='s3',readback_status='READBACK_PENDING',
                    **{'EXACT_FINAL hqnr':.9512345678912,'EXACT_FINAL rmse':7.123456789})
        receipt=apply_upsert(ws,values)
        self.assertTrue(receipt['readback_verified'])
        self.assertEqual(ws.rows[1][4],values['EXACT_FINAL hqnr'])
        self.assertEqual(ws.formats[0]['format']['numberFormat']['pattern'],'0.0000')
        apply_upsert(ws,values)
        self.assertEqual(len(ws.rows),2)
        self.assertEqual(ws.rows[1][3],'READBACK_VERIFIED')

    def test_old_tab_other_schema_duplicate_rows_never_overwritten(self):
        values=dict(campaign_id=CAMPAIGN_ID,run_id='x',server='s3',readback_status='READBACK_PENDING')
        ws=Worksheet();ws.title='GF2-s3(5090)'
        with self.assertRaises(ValueError):apply_upsert(ws,values)
        self.assertEqual(ws.rows,[])
        ws=Worksheet();ws.rows=[['old header']]
        with self.assertRaises(ValueError):apply_upsert(ws,values)
        self.assertEqual(ws.rows,[['old header']])
        ws=Worksheet();apply_upsert(ws,values);ws.rows.append(ws.rows[-1].copy())
        with self.assertRaises(ValueError):apply_upsert(ws,values)

    def test_flatten_preserves_rmse_cc_jqm_and_lifetime_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            record=dict(update=20000,val_ergas=.57,rr=dict(ergas=.55,rmse=4.123456789,cc=.98),
                fr=dict(hqnr=.95,d_s=.03,d_lambda=.02,jqm=.88,jqm_variant='SRF-substitute'))
            summary=dict(profile='BASE',local_updates=20000,lifetime_updates=120000,
                         selections={k:record for k in ('EXACT_FINAL','RR_VAL_SELECTED','RAW_AUX')})
            values=flatten_summary(summary,case_for('A01'),directory)
            self.assertEqual(values['EXACT_FINAL rmse'],4.123456789)
            self.assertEqual(values['EXACT_FINAL cc'],.98)
            self.assertEqual(values['EXACT_FINAL jqm'],.88)
            self.assertEqual(values['local_updates'],20000)
            self.assertEqual(values['lifetime_updates'],120000)
            self.assertNotIn('Exact100K',values)


if __name__=='__main__':unittest.main()
