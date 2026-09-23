import unittest
import copy
import tempfile
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import MagicMock,patch
from ablr2.plan import CAMPAIGN_ID,CASES,build_config,campaign_id
from ablr2.common import camp,read_json
from ablr2.upload import (plan_upsert, metric_formats, method_label, component_summary_values,
    plan_component_summary,apply_component_summary,spool_run,upload_run,retry_pending,SUMMARY_HEADERS,SUMMARY_BEGIN,SUMMARY_END)


def gf2_values(case_id='C17'):
    case=next(c for c in CASES if c.server_id=='s3' and c.case_id==case_id and c.sweep=='P01')
    value={'Run':method_label(case),'Method':method_label(case),'ABLR2 run id':case.run_id,
        'ABLR2 sensor':'GF2','ABLR2 server':'s3','ABLR2 campaign':campaign_id('GF2'),'ABLR2 role':'S',
        'ABLR2 case':case_id,'ABLR2 status':'OFFICIAL_EVAL_COMPLETE','ABLR2 actual updates':50000,
        'ABLR2 phase':'BOOT5','ABLR2 recipe':'R00','ABLR2 recipe revision':'r000','ABLR2 wave':'BOOT5',
        'ABLR2 sweep':'P01','ABLR2 selected SHA256':'checkpoint','ABLR2 upload status':'READBACK_PENDING',
        'HQNR↑':.963123456789,'D_lambda↓':.02,'D_s↓':.017,'ERGAS↓':.56,'SAM↓':1.,'SCC↑':.97,
        'Q4↑':.93,'PSNR↑':37.,'SSIM↑':.98,'RMSE↓':3.,'CC↑':.99,'JQM↑':.8,'Notes':'fresh50K'}
    return case,value


class UploadTests(unittest.TestCase):
    def values(self):
        return {'Run': 'ABLR2_WV3_s1_R00_P01_C00_1_FRESH50', 'ABLR2 campaign': CAMPAIGN_ID,
                'ABLR2 sensor': 'WV3', 'ABLR2 run id': 'ABLR2_WV3_s1_R00_P01_C00_1_FRESH50',
                'HQNR↑': .963123456789, 'Q8↑': .856712345, 'ABLR2 upload status': 'READBACK_PENDING'}

    def test_append_does_not_modify_legacy_row_and_keeps_precision(self):
        table = [[], [], ['Run', 'HQNR↑'], ['legacy_run', '.97']]
        plan = plan_upsert(table, table[2], self.values())
        self.assertEqual(plan['row'], 5)
        self.assertEqual(plan['values']['HQNR↑'], .963123456789)
        self.assertFalse(any(e['range'].endswith('4') for e in plan['edits']))

    def test_same_run_cannot_alias_historical_namespace(self):
        values = self.values()
        with self.assertRaises(ValueError):
            plan_upsert([[], [], ['Run'], [values['Run']]], ['Run'], values)

    def test_existing_observation_cannot_be_replaced(self):
        values = self.values()
        headers = list(values)
        row = list(values.values())
        row[headers.index('HQNR↑')] = .99
        with self.assertRaises(ValueError): plan_upsert([[], [], headers, row], headers, values)

    def test_four_decimals_are_display_only(self):
        formats = metric_formats({'HQNR↑': 1, 'Q8↑': 2, 'Q4↑': 3, 'ABLR2 RR_VAL_SELECTED ERGAS↓': 4}, 4)
        self.assertEqual(len(formats), 4)
        self.assertTrue(all(f['format']['numberFormat']['pattern'] == '0.0000' for f in formats))

    def test_native_gf2_label_has_no_seed_but_compound_run_identity_is_unique(self):
        case,values=gf2_values();self.assertNotIn(str(case.seed),values['Run'])
        self.assertNotIn('\n',values['Run']);self.assertIn('C17',values['Run'])
        plan=plan_upsert([[],[],['Run']],['Run'],values)
        self.assertEqual(plan['values']['HQNR↑'],.963123456789)
        table=[[],[],list(values),list(values.values())]
        self.assertEqual(plan_upsert(table,list(values),values)['row'],4)

    def test_summary_appends_owned_region_and_keeps_production_rows_intact(self):
        _,values=gf2_values();legacy=[['production layout'],['B20 result',.97],['P40 result',.98]]
        before=copy.deepcopy(legacy);first=plan_component_summary(legacy,values)
        self.assertEqual(first['first_row'],4);self.assertEqual(legacy,before)
        table=legacy+first['matrix'];same=plan_component_summary(table,values)
        self.assertEqual(first['matrix'],same['matrix'])
        _,c03=gf2_values('C03');updated=plan_component_summary(table,c03)
        self.assertEqual([r['Case'] for r in updated['records']],['C03','C17'])
        self.assertEqual(updated['records'][1]['HQNR↑'],.963123456789)

    def test_summary_cannot_expand_into_manual_following_region_or_changed_metrics(self):
        _,values=gf2_values();plan=plan_component_summary([],values);table=plan['matrix']
        _,c03=gf2_values('C03')
        with self.assertRaises(ValueError):plan_component_summary(table+[['manual following section']],c03)
        changed=dict(values);changed['HQNR↑']=.99
        with self.assertRaises(ValueError):plan_component_summary(table,changed)
        with self.assertRaises(ValueError):plan_component_summary(table+[[SUMMARY_BEGIN]],values)

    def test_summary_rejects_noncomponent_production_teacher_incomplete_and_seed_labels(self):
        _,values=gf2_values()
        for key,value in [('ABLR2 run id','P40_GF2_s3_C17'),('ABLR2 role','T'),('ABLR2 status','TRAINING'),
                          ('ABLR2 case','C18'),('ABLR2 actual updates',49999),('Method','P01 | C17 | seed 991001'),('Method','two\nlines')]:
            with self.subTest(key=key),self.assertRaises(ValueError):component_summary_values(dict(values,**{key:value}))

    def test_summary_raw_readback_and_formatting_only_owned_cells(self):
        _,values=gf2_values();table=[['untouched production']];plan=plan_component_summary(table,values)
        ws=MagicMock();ws.title='ablations';ws.id=123;ws.col_count=100;ws.row_count=1000
        ws.get_all_values.side_effect=[table,table,table+plan['matrix']]
        with patch('ablr2.upload.fetch_controls',return_value={}),patch('ablr2.upload.controlled_reason',return_value=None):
            receipt=apply_component_summary(ws,values)
        self.assertTrue(receipt['readback_verified'])
        edit=ws.batch_update.call_args.args[0][0]
        self.assertTrue(edit['range'].startswith('A2:'))
        self.assertTrue(all(x['format']['numberFormat']['pattern']=='0.0000' for x in ws.batch_format.call_args.args[0]))

    def test_network_failure_leaves_immutable_payload_before_any_api_call(self):
        case,values=gf2_values();cfg=build_config(case)
        with tempfile.TemporaryDirectory() as root,patch('ablr2.upload.find_config',return_value=cfg),patch('ablr2.upload.row_values',return_value=values):
            ws=MagicMock();ws.title='GF2-s3(5090)'
            with patch('ablr2.upload.write_locks',return_value=nullcontext()),patch('ablr2.upload.apply_upsert',side_effect=OSError('offline')):
                with self.assertRaises(OSError):upload_run(case.run_id,root,activated=True,worksheet=ws)
            folder=camp(root,'s3')/'upload_outbox'
            pending=read_json(folder/'pending'/(case.run_id+'.json'))
            payload=read_json(pending['payload_path'])
            self.assertEqual(payload['values']['HQNR↑'],.963123456789)
            self.assertEqual(pending['status'],'PENDING')
            self.assertEqual(spool_run(case.run_id,root),pending)
            self.assertEqual(len(list((folder/'payloads').glob('*.json'))),1)

    def test_outbox_retry_batch_is_bounded_and_does_not_train(self):
        from ablr2.common import atomic_json
        with tempfile.TemporaryDirectory() as root:
            folder=camp(root,'s3')/'upload_outbox/pending'
            for i in range(3):atomic_json(folder/f'run{i}.json',dict(run_id=f'run{i}',server='s3',status='PENDING',updated_at_utc=str(i)))
            with patch('ablr2.upload.upload_run',side_effect=OSError('offline')) as upload:
                result=retry_pending(root,'s3',limit=2,activated=True)
            self.assertEqual(upload.call_count,2);self.assertEqual(len(result),2)
            self.assertEqual(read_json(folder/'run0.json')['attempts'],1)
            self.assertNotIn('attempts',read_json(folder/'run2.json'))
            with self.assertRaises(PermissionError):retry_pending(root,'s3')

    def test_malformed_outbox_pointer_is_preserved_and_cannot_starve_healthy_api_batch(self):
        from ablr2.common import atomic_json
        with tempfile.TemporaryDirectory() as root:
            folder=camp(root,'s3')/'upload_outbox/pending';folder.mkdir(parents=True)
            bad=folder/'broken.json';bad.write_text('{interrupted')
            atomic_json(folder/'badtype.json',[])
            atomic_json(folder/'foreign.json',dict(run_id='foreign',server='s1',status='PENDING'))
            atomic_json(folder/'healthy.json',dict(run_id='healthy',server='s3',status='PENDING'))
            with patch('ablr2.upload.upload_run',return_value={'readback_verified':True}) as upload:
                result=retry_pending(root,'s3',limit=1,activated=True)
            upload.assert_called_once_with('healthy',root,activated=True)
            self.assertTrue(result['healthy']['readback_verified'])
            for run in ('broken','badtype','foreign'):self.assertEqual(result[run]['status'],'UPLOAD_BLOCKED_INTEGRITY')
            self.assertEqual(bad.read_text(),'{interrupted')
            self.assertEqual(read_json(folder.parent/'retry_status.json')['results'],result)


if __name__ == '__main__': unittest.main()
