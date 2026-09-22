"""Offline worksheet mocks only: no credential reads or external API writes."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from pcrepro.common import atomic_json, camp, object_sha, read, run_dir
from pcrepro.plan import CAMPAIGN_ID, RECIPE_ID, Case
from pcrepro.upload import (HEADER, SELECTIONS, apply_upsert, flatten_summary,
    validate_summary, upload_run, retry_pending, spool_run, status_summary)
from pcrepro.reporting import rebuild


def summary_for(case, *, complete=True, alias=False):
    def record(sha,update):
        rr=dict(ergas=2.0436123456789,sam=2.1,psnr=38.,ssim=.965,scc=.988,
                rmse=13.0123456789123,cc=.99,n_scenes=20)
        rr['q8' if case.num_bands==8 else 'q4']=.92345678912345
        fr=dict(hqnr=.958158987654321,d_lambda=.013,d_s=.027,jqm=.90987654321098,
            jqm_variant='JQM_SRF_substitute_v1',jqm_status='measured',n_scenes=20)
        return dict(update=update,checkpoint_sha256=sha,rr=rr,fr=fr,
            evaluation_manifest_sha256='e'*64,val_ergas=2.1,data_sha256='d'*64)
    first=record('a'*64,50000);second=copy.deepcopy(first) if alias else record('b'*64,34000)
    if alias:second['alias_of']='EXACT_50000'
    return dict(campaign_id=CAMPAIGN_ID,recipe_id=RECIPE_ID,run_id=case.run_id,case=case.to_dict(),
        complete=complete,status='OFFICIAL_EVAL_COMPLETE' if complete else 'FAILED_OOM',
        actual_updates=0 if case.dataset=='WV2' else 50000 if complete else 1234,
        started_at_utc='2026-09-22T14:30:00+00:00',completed_at_utc='2026-09-22T16:30:00+00:00',
        source_identity=dict(content_sha256='c'*64,git_release='0123456789abcdef'),
        data_sha256='d'*64,config_sha256='f'*64,evaluation_manifest_sha256='e'*64,
        paper_identity_status='PAPERSET_IDENTITY_VERIFIED',
        source_run_id=case.source_run_id,source_wv3_training_seconds=9000 if case.dataset=='WV2' else '',
        costs=dict(training_seconds=0 if case.dataset=='WV2' else 3600,validation_seconds=800,
                   test_seconds=200,profile_seconds=20,preprocessing_seconds=30,io_seconds=100),
        profile=dict(params_total=7212200,params_trainable=7212200,params_m=7.2122,
                     runtime=dict(gpu='RTX 5090'),input_shape=[1,1,256,256],flops_convention='MACx2'),
        selections=dict(EXACT_50000=first,RR_VAL_ERGAS_MIN=second))


class FakeWorksheet:
    def __init__(self,title='PC-Repro-s3',rows=None):
        self.title=title;self.id=917;self.col_count=1;self.row_count=1
        self.rows=copy.deepcopy(rows or []);self.formats=[];self.writes=[];self.fail_writes=False
    def row_values(self,row,**kwargs):return list(self.rows[row-1]) if row<=len(self.rows) else []
    def get_all_values(self,**kwargs):return copy.deepcopy(self.rows)
    def add_cols(self,n):self.col_count+=n
    def add_rows(self,n):self.row_count+=n
    def batch_format(self,formats):self.formats.extend(formats)
    def update(self,*,range_name,values,value_input_option):
        if self.fail_writes:raise OSError('simulated sheet outage')
        assert value_input_option=='RAW'
        self.writes.append((range_name,copy.deepcopy(values)))
        match=re.fullmatch(r'([A-Z]+)(\d+)',range_name);row=int(match.group(2));col=0
        for char in match.group(1):col=col*26+ord(char)-64
        while len(self.rows)<row+len(values)-1:self.rows.append([])
        for offset,source in enumerate(values):
            target=self.rows[row+offset-1]
            while len(target)<col+len(source)-1:target.append('')
            target[col-1:col-1+len(source)]=source


class PresentationTests(unittest.TestCase):
    def test_run_cell_only_model_setting_and_dynamic_q_band(self):
        for dataset in ('WV3','WV2','QB','GF2'):
            c=Case('s3',12,dataset);row=flatten_summary(summary_for(c),c)
            self.assertEqual(row['Run'],f'PAN-Crafter | C128 D224 LN | CM3A3 k3 | MARs lambda1 | PAN+MS in{c.num_bands+1} | 50K')
            self.assertNotIn(str(c.seed),row['Run']);self.assertNotIn('s3',row['Run'])
            self.assertEqual(tuple(row),HEADER)
            expected='q8' if c.num_bands==8 else 'q4';other='q4' if expected=='q8' else 'q8'
            self.assertEqual(row['EXACT_50000 rr_'+expected],.92345678912345)
            self.assertEqual(row['EXACT_50000 rr_'+other],'')

    def test_full_precision_actual_kst_dates_and_alias_columns(self):
        c=Case('s3',0,'WV3');row=flatten_summary(summary_for(c,alias=True),c)
        self.assertEqual(row['EXACT_50000 fr_hqnr'],.958158987654321)
        self.assertEqual(row['EXACT_50000 rr_rmse'],13.0123456789123)
        self.assertEqual(row['Date'],'2026-09-23')
        self.assertEqual(row['started_at_kst'],'2026-09-22T23:30:00+09:00')
        self.assertTrue(row['selections_alias_same_checkpoint'])
        self.assertEqual(row['RR_VAL_ERGAS_MIN alias_of'],'EXACT_50000')

    def test_wv2_zero_update_and_no_training_recharge(self):
        c=Case('s4',5,'WV2');row=flatten_summary(summary_for(c),c)
        self.assertEqual(row['actual_updates'],0);self.assertEqual(row['training_seconds'],0)
        self.assertEqual(row['source_wv3_training_seconds'],9000)
        self.assertEqual(row['source_run_id'],Case('s4',5,'WV3').run_id)
        for label in SELECTIONS:self.assertEqual(row[label+' source_checkpoint_sha256'],row[label+' checkpoint_sha256'])
        invalid=summary_for(c);invalid['costs']['training_seconds']=1
        with self.assertRaises(ValueError):flatten_summary(invalid,c)
        invalid=summary_for(c);invalid['source_run_id']=Case('s3',5,'WV3').run_id
        with self.assertRaises(ValueError):flatten_summary(invalid,c)

    def test_failure_status_rows_never_expose_partial_metrics(self):
        c=Case('s3',0,'GF2');row=flatten_summary(summary_for(c,complete=False),c)
        self.assertFalse(row['eval_complete']);self.assertEqual(row['actual_updates'],1234)
        self.assertEqual(row['status'],'FAILED_OOM')
        for label in SELECTIONS:
            self.assertTrue(all(row[k]=='' for k in HEADER if k.startswith(label+' ')))
        self.assertEqual(row['checkpoint_sha256'],'')

    def test_detect_wrong_q_metric_nans_and_unlabelled_jqm(self):
        c=Case('s3',0,'QB')
        for mutate in (lambda s:s['selections']['EXACT_50000']['rr'].update(q8=.9),
                       lambda s:s['selections']['EXACT_50000']['fr'].update(hqnr=float('nan')),
                       lambda s:s['selections']['EXACT_50000']['fr'].update(jqm_variant='official')):
            summary=summary_for(c);mutate(summary)
            with self.assertRaises(ValueError):validate_summary(summary,c)

    def test_actual_scene_counts_recorded_not_silently_forced20(self):
        c=Case('s3',0,'WV2');summary=summary_for(c)
        summary['paper_identity_status']='PAPERSET_IDENTITY_UNVERIFIED'
        for record in summary['selections'].values():record['rr']['n_scenes']=19
        row=flatten_summary(summary,c)
        self.assertEqual(row['EXACT_50000 rr_n_scenes'],19)
        self.assertEqual(row['paper_identity_status'],'PAPERSET_IDENTITY_UNVERIFIED')

    def test_identical_checkpoint_cannot_have_different_metrics(self):
        c=Case('s3',0,'WV3');summary=summary_for(c,alias=True)
        summary['selections']['RR_VAL_ERGAS_MIN']['fr']['hqnr']+=.001
        with self.assertRaisesRegex(ValueError,'different evaluation'):validate_summary(summary,c)


class WorksheetTests(unittest.TestCase):
    def test_dedicated_tab_upsert_raw_readback_and_four_decimal_format(self):
        c=Case('s3',0,'WV3');row=flatten_summary(summary_for(c),c);ws=FakeWorksheet()
        first=apply_upsert(ws,row);second=apply_upsert(ws,row)
        self.assertEqual(first['row'],2);self.assertEqual(second['row'],2);self.assertEqual(len(ws.rows),2)
        col=HEADER.index('EXACT_50000 fr_hqnr')
        self.assertEqual(ws.rows[1][col],.958158987654321)
        self.assertTrue(first['readback_verified']);self.assertTrue(ws.formats)
        self.assertTrue(all(f['format']['numberFormat']['pattern']=='0.0000' for f in ws.formats))

    def test_refuse_other_campaign_tab_header_overwrite_and_duplicates(self):
        c=Case('s3',0,'WV3');row=flatten_summary(summary_for(c),c)
        for ws in (FakeWorksheet('GF2-P40-s3'),FakeWorksheet(rows=[['old schema']])):
            with self.assertRaises(ValueError):apply_upsert(ws,row)
            self.assertEqual(ws.writes,[])
        ws=FakeWorksheet();apply_upsert(ws,row);ws.rows.append(list(ws.rows[-1]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):apply_upsert(ws,row)

    def test_readback_disagreement_never_reports_success(self):
        class WrongReadback(FakeWorksheet):
            def row_values(self,row,**kwargs):
                values=super().row_values(row,**kwargs)
                if row>1 and values:values[HEADER.index('EXACT_50000 fr_hqnr')]=.1
                return values
        c=Case('s3',0,'WV3')
        with self.assertRaisesRegex(ValueError,'readback mismatch'):
            apply_upsert(WrongReadback(),flatten_summary(summary_for(c),c))


class SpoolAndReportTests(unittest.TestCase):
    def test_explicit_activation_required_before_any_credential_or_payload_read(self):
        c=Case('s3',0,'WV3')
        with patch('pcrepro.upload.row_values',side_effect=AssertionError('no read')):
            with self.assertRaises(PermissionError):upload_run(c.run_id)
            with self.assertRaises(PermissionError):retry_pending('/tmp','s3')

    def test_durable_spool_precedes_api_and_failure_can_retry_beyond_two_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'WV3');row=flatten_summary(summary_for(c),c);ws=FakeWorksheet();ws.fail_writes=True
            with patch('pcrepro.upload.row_values',return_value=row):
                for attempt in range(3):
                    receipt=upload_run(c.run_id,root,activated=True,worksheet=ws)
                    self.assertFalse(receipt['readback_verified'])
                    envelope=read(camp(root,'s3')/'upload_spool'/(c.run_id+'.json'))
                    self.assertEqual(envelope['attempts'],attempt+1)
                ws.fail_writes=False
                receipts=retry_pending(root,'s3',limit=1,activated=True,worksheet=ws)
                self.assertTrue(receipts[c.run_id]['readback_verified'])
                envelope=read(camp(root,'s3')/'upload_spool'/(c.run_id+'.json'))
                self.assertEqual(envelope['attempts'],4)
                self.assertEqual(envelope['status'],'READBACK_VERIFIED')
                writes=len(ws.writes);upload_run(c.run_id,root,activated=True,worksheet=ws)
                self.assertEqual(len(ws.writes),writes)

    def test_api_factory_observes_already_saved_payload(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'GF2');row=flatten_summary(summary_for(c),c)
            def factory(*args):
                envelope=read(camp(root,'s3')/'upload_spool'/(c.run_id+'.json'))
                self.assertEqual(envelope['payload'],row);self.assertEqual(envelope['attempts'],1)
                raise OSError('offline')
            with patch('pcrepro.upload.row_values',return_value=row),patch('pcrepro.upload._worksheet',side_effect=factory):
                result=upload_run(c.run_id,root,activated=True)
            self.assertEqual(result['status'],'UPLOAD_PENDING')

    def test_updated_complete_payload_replaces_failure_same_run_id(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'WV3');ws=FakeWorksheet()
            for complete in (False,True):
                values=flatten_summary(summary_for(c,complete=complete),c)
                with patch('pcrepro.upload.row_values',return_value=values):
                    upload_run(c.run_id,root,activated=True,worksheet=ws)
            self.assertEqual(len(ws.rows),2)
            self.assertTrue(ws.rows[1][HEADER.index('eval_complete')])

    def test_retry_batch_limit_and_distinct_cycle_rows(self):
        with tempfile.TemporaryDirectory() as root:
            cases=[Case('s3',i,'WV3') for i in range(3)]
            payloads={c.run_id:flatten_summary(summary_for(c),c) for c in cases}
            with patch('pcrepro.upload.row_values',side_effect=lambda run,root:payloads[run]):
                for c in cases:spool_run(c.run_id,root)
                ws=FakeWorksheet();results=retry_pending(root,'s3',limit=2,activated=True,worksheet=ws)
                self.assertEqual(len(results),2);self.assertEqual(len(ws.rows),3)

    def test_corrupt_historical_spool_cannot_abort_other_uploads(self):
        with tempfile.TemporaryDirectory() as root:
            first,second=Case('s3',0,'WV3'),Case('s3',1,'WV3')
            payloads={c.run_id:flatten_summary(summary_for(c),c) for c in (first,second)}
            with patch('pcrepro.upload.row_values',side_effect=lambda run,root:payloads[run]):
                spool_run(first.run_id,root);spool_run(second.run_id,root)
                (camp(root,'s3')/'upload_spool'/(first.run_id+'.json')).write_text('{broken')
                ws=FakeWorksheet();result=retry_pending(root,'s3',activated=True,worksheet=ws)
            self.assertEqual(result[first.run_id]['status'],'UPLOAD_BLOCKED_INTEGRITY')
            self.assertTrue(result[second.run_id]['readback_verified'])
            self.assertEqual(len(ws.rows),2)

    def test_fabricated_verified_flag_without_matching_readback_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'WV3');payload=flatten_summary(summary_for(c),c)
            with patch('pcrepro.upload.row_values',return_value=payload):
                envelope=spool_run(c.run_id,root);envelope['status']='READBACK_VERIFIED'
                atomic_json(camp(root,'s3')/'upload_spool'/(c.run_id+'.json'),envelope)
                with self.assertRaisesRegex(ValueError,'matching raw-value readback'):
                    upload_run(c.run_id,root,activated=True,worksheet=FakeWorksheet())

    def test_failure_source_identity_required_and_never_invented(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'QB')
            with self.assertRaisesRegex(ValueError,'Persisted per-run'):status_summary(c.run_id,root)
            status=dict(campaign_id=CAMPAIGN_ID,run_id=c.run_id,recipe_id=RECIPE_ID,
                status='BLOCKED_DATA',actual_updates=0,source_identity=dict(files={},content_sha256=object_sha({})))
            atomic_json(run_dir(c,root)/'meta/status.json',status)
            summary=status_summary(c.run_id,root);self.assertEqual(summary['status'],'BLOCKED_DATA')
            status['run_id']=Case('s4',0,'QB').run_id
            atomic_json(run_dir(c,root)/'meta/status.json',status)
            with self.assertRaisesRegex(ValueError,'identity differs'):status_summary(c.run_id,root)

    def test_failed_postrun_outcome_preserves_training_costs_and_actual_updates(self):
        with tempfile.TemporaryDirectory() as root:
            c=Case('s3',0,'QB')
            identity=dict(campaign_id=CAMPAIGN_ID,run_id=c.run_id,recipe_id=RECIPE_ID,
                source_identity=dict(files={},content_sha256=object_sha({})))
            atomic_json(run_dir(c,root)/'meta/training_status.json',dict(identity,
                status='TRAINING_COMPLETE',actual_updates=50000,costs=dict(training_seconds=100)))
            atomic_json(run_dir(c,root)/'meta/status.json',dict(identity,
                status='FAILED_POSTRUN',actual_updates=50000,reason='missing FR asset'))
            summary=status_summary(c.run_id,root)
            self.assertEqual(summary['status'],'FAILED_POSTRUN')
            self.assertEqual(summary['reason'],'missing FR asset')
            self.assertEqual(summary['costs']['training_seconds'],100)
            self.assertEqual(summary['actual_updates'],50000)

    def test_cycle_reports_count_runs_not_selections_and_separate_seed_cohorts(self):
        with tempfile.TemporaryDirectory() as root:
            cases=[Case('s3',cycle,dataset) for cycle in (0,1) for dataset in ('WV3','WV2','QB','GF2')]
            payloads={}
            for c in cases:
                summary=summary_for(c,alias=True)
                atomic_json(run_dir(c,root)/'official/summary.json',summary)
                payloads[c.run_id]=flatten_summary(summary,c)
            with patch('pcrepro.reporting.row_values',side_effect=lambda run,root:payloads[run]):
                result=rebuild(root,'s3')
            self.assertEqual(result['unique_runs'],8);self.assertEqual(result['completed_training_runs'],6)
            self.assertEqual(result['completed_zero_shot_evaluations'],2)
            self.assertEqual(result['costs']['training_seconds'],6*3600)
            self.assertEqual(len(result['cohorts']),8)
            self.assertFalse(result['selections_count_as_extra_experiments'])
            self.assertFalse(result['performance_stop_enabled'])
            self.assertTrue(all(c['complete'] for c in result['cycles'].values()))


if __name__=='__main__':unittest.main()
