import copy
import math
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from ta2.common import CAMPAIGN, load_json
from ta2.common import atomic_json, digest, immutable_json
from ta2.reporting import (AnalysisSheet, COLUMNS, INDEX, SCHEMA, SHEET_ID, SPREADSHEET_ID,
    CapacityBlocked, Outbox, PublicationConflict, aggregate_records, gradient_records,
    asset_records, make_record, publication_capacity_estimate, response_records, status_record, summary_records)
from ta2.reporting import scientific_payload


def context(**kwargs):
    value = dict(case_id='TA2-M07', dataset='WV3', server='s1', seed=261006101,
                 replica=1, attempt=1, run_id='registered_run', completed_step=1010,
                 checkpoint_sha256='c'*64, source_revision='s'*64, selector='EXACT_FINAL',
                 label='One line\nexperiment', status='PLANNED_NOT_RUN')
    value.update(kwargs)
    return value


def record(scope='measurement', **fields):
    return make_record(context(), 'RESPONSE', scope, {'Gain': .9876543211234567, **fields})


def col_number(text):
    result = 0
    for char in text:
        result = result*26 + ord(char)-64
    return result-1


def area(text):
    plain = text.split('!')[-1]
    match = re.fullmatch(r'([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?', plain)
    if not match:
        raise AssertionError('Unexpected API range: '+text)
    return int(match[2]), int(match[4] or match[2]), col_number(match[1]), col_number(match[3] or match[1])


class FakeBook:
    id = SPREADSHEET_ID

    def __init__(self):
        self.rows = {7:list(COLUMNS), 8:['REPORT_ONLY']+['']*151}
        self.count = 2000
        self.other_cells = 0
        self.append_calls=[]; self.marker_calls=[]; self.format_calls=[]
        self.corrupt=False; self.timeout_after_append=False; self.concurrent=False
        self.validation=False

    def fetch_sheet_metadata(self, params):
        props = dict(sheetId=SHEET_ID,title='analysis',gridProperties=dict(rowCount=self.count,columnCount=152))
        if params.get('includeGridData'):
            blocks=[]
            for span in params['ranges']:
                first,last,left,right=area(span)
                rows=[]
                for number in range(first,last+1):
                    values=self.rows.get(number,['']*152)[left:right+1]
                    cells=[]
                    for value in values:
                        typ='boolValue' if isinstance(value,bool) else 'numberValue' if isinstance(value,(int,float)) else 'stringValue'
                        cells.append({'userEnteredValue':{typ:value}} if value!='' else {})
                    if self.validation:cells[0]['dataValidation']={'condition':{'type':'ONE_OF_LIST'}}
                    rows.append(dict(values=cells))
                blocks.append(dict(startRow=first-1,startColumn=left,rowData=rows))
            return dict(sheets=[dict(properties=props,data=blocks)])
        return dict(spreadsheetId=self.id,sheets=[dict(properties=props),dict(properties=dict(sheetId=1,title='paper',gridProperties=dict(rowCount=self.other_cells,columnCount=1)))])

    def values_get(self, text, params=None):
        first,last,left,right=area(text)
        result=[]
        for number in range(first,last+1):
            values=list(self.rows.get(number,[])[left:right+1])
            while values and values[-1]=='':values.pop()
            result.append(values)
        while result and not result[-1]:result.pop()
        if self.corrupt and first>8 and left==0 and right==151 and result:
            result[0][INDEX['Gain']]=.123
        return dict(values=result)

    def values_append(self, text, params, body):
        self.append_calls.append((text,copy.deepcopy(params),copy.deepcopy(body)))
        assert params['valueInputOption']=='RAW' and params['insertDataOption']=='INSERT_ROWS'
        if self.concurrent:
            at=max(self.rows)+1
            self.rows[at]=['STATUS']+['']*151
            self.concurrent=False
        number=max(max(self.rows)+1,area(text)[0])
        for offset,values in enumerate(body['values']):
            self.rows[number+offset]=list(values)
        last=number+len(body['values'])-1
        self.count=max(self.count,last)
        if self.timeout_after_append:
            self.timeout_after_append=False
            raise TimeoutError('Simulated uncertain transport after successful append')
        return dict(updates=dict(updatedRange=f'analysis!A{number}:EV{last}'))

    def values_batch_get(self, ranges, params=None):
        return dict(valueRanges=[dict(range=span,**self.values_get(span,params)) for span in ranges])

    def values_batch_update(self, body):
        self.marker_calls.append(copy.deepcopy(body))
        for request in body['data']:
            first,last,left,right=area(request['range'])
            assert first==last and left==right
            assert left in (INDEX['Readback_status'],INDEX['Upload_receipt_SHA'])
            self.rows[first][left]=request['values'][0][0]

    def batch_update(self, body):
        self.format_calls.append(copy.deepcopy(body))


class RecordsTest(unittest.TestCase):
    def test_status_real_updates_and_full_capacity_estimate(self):
        with self.assertRaises(ValueError):status_record(context(),'RUNNING',0)
        self.assertEqual(status_record(context(),'RUNNING',1)['Effective_updates'],1)
        estimated=publication_capacity_estimate(workbook_allocated_cells=2_000_000)
        self.assertEqual(estimated['status'],'REQUIRES_PUBLICATION_PARTITION_DECISION')
        self.assertGreater(estimated['added_cells_lower_estimate'],100_000_000)
        self.assertFalse(estimated['automatic_new_tabs'])

    def test_asset_numbers_and_smoke_are_not_formal_summary(self):
        manifest=dict(status='PASS',identity=dict(operator={'phase':2}),splits=dict(train=dict(
            sha256='a'*64,count=9714,lms_parity=dict(max_abs_dn=.001,rmse_dn=.00001,band_ncc=[.99,None]),
            phase_diagnostics=dict(summary=dict(lms_vs_gt=dict(mean_dy_dx=[.01,.02],valid_count=4,
                mean_absolute_component_pixels=.015))),statistics=dict(pan=dict(minimum=0,maximum=2047)))))
        smoke=dict(status='PASS',smoke_only=True,formal_optimizer_updates=0,cases=[dict(case_id='TA2-M07',
            logs=[{},{}],metric_smoke=dict(hqnr=.93,scc=.99,ergas=3.1),peak_cuda_bytes=1000000)])
        rows=asset_records(context(),manifest,smoke)
        self.assertTrue(any(r['Phase_dx']==.02 for r in rows))
        self.assertTrue(any(r['Metric_name']=='lms_parity/rmse_dn' and r['Estimate']==.00001 for r in rows))
        self.assertTrue(all(r['Record_type'] in ('ASSET','UNIT') for r in rows))
        self.assertTrue(all(r['HQNR']=='' for r in rows))

    def test_strict_columns_finite_blank_and_plain_text(self):
        row=record(Estimate=None,Reason='=IMPORTXML("https://not-executed","x")')
        self.assertEqual(len(row),152)
        self.assertEqual(row['Estimate'],'')
        self.assertEqual(row['Experiment'],'One line experiment')
        self.assertEqual(row['Status'],'MEASURED')
        for value in (float('nan'),float('inf'),'0.94',[.1,.2],True):
            with self.assertRaises(ValueError):record(Gain=value)
        with self.assertRaises(ValueError):record(Typo_metric=1)
        with self.assertRaises(ValueError):make_record(context(server='s2'),'STATUS','test')

    def test_same_key_changed_value_conflicts_but_retry_time_does_not(self):
        a=record();b=record(Gain=.5)
        self.assertEqual(a['Record_ID'],b['Record_ID'])
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory)
            path=out.enqueue(a)
            self.assertTrue(path.with_suffix('.csv').exists())
            self.assertEqual(out.enqueue(record()),path)
            with self.assertRaises(PublicationConflict):out.enqueue(b)

    def test_hqnr_selection_needs_real_fifty_candidate_receipt(self):
        report=dict(complete=True,rr_fr_same_checkpoint=True,completed_step=50000,checkpoint_sha256='x',
                    rr=dict(n_scenes=20,ergas=2.1,scc=.99),fr=dict(n_scenes=20,hqnr=.955,d_s=.02))
        with self.assertRaises(ValueError):summary_records(context(),report,'HQNR_MAX50')
        selected=dict(status='HQNR_SELECTION_COMPLETE',checkpoint_sha256='x',candidates_complete=50,candidates_expected=50)
        row=summary_records(context(),report,'HQNR_MAX50',selected)[0]
        self.assertTrue(row['Test_aware']);self.assertEqual(row['Completed_count'],50)
        self.assertEqual(summary_records(context(),report,'FIXED_PROGRESS')[0]['Record_type'],'CURVE')

    def test_gradient_protocol_and_response_numeric_mappings(self):
        gradients=dict(status='COMPLETE',seed=1,fixed_probe_batches=32,summary=[dict(group='fc2',
            raw_norm=dict(rec=1.,eps=2.,struct=3.),weighted_norm=dict(rec=1.,eps=.002,struct=.03),
            cosine=dict(rec_eps=None,rec_struct=.2,eps_struct=-.1),cosine_valid_counts=dict(rec_eps=0),
            total_weighted_norm=1.03,actual_parameter_update_norm=None)])
        rows=gradient_records(context(),gradients)
        self.assertEqual(rows[0]['Grad_struct_weighted_norm'],.03)
        self.assertEqual(rows[0]['Grad_cos'],'')
        self.assertTrue(any(r['Metric_name']=='total_weighted_norm' and r['Estimate']==1.03 for r in rows))
        response=dict(response_summary=[dict(split='fr',probe='AXIS16',mode='REESTIMATE_CURRENT_INPUT',
            measurement_domain='NATIVE_PATCH',support_scope='normal',radius=.5,gain=.9,gain_y=.8,gain_x=1.,
            numerator=9.,denominator=10.,component_mae=.1,epe=.2,n=80,n_clusters=20,
            numerator_y=4.,denominator_y=5.,numerator_x=5.,denominator_x=5.,mae_y=.2,mae_x=0.,gain_ci95=[.85,.95])])
        rows=response_records(context(),response)
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['Gain_num'],9.)
        self.assertEqual(rows[1]['Direction'],'y');self.assertEqual(rows[1]['Gain_den'],5.)

    def test_incomplete_three_seed_aggregate_keeps_missing_blank(self):
        data=dict(record_type='SEED_AGG',metric='hqnr',status='INCOMPLETE_3_SEEDS',n_success=2,
                  n_attempted=2,n_failed=0,mean=.94,std=.01,seed_values={'1':.93,'2':.95,'3':None},
                  seed_status={'1':'COMPLETE','2':'COMPLETE','3':'NOT_ATTEMPTED'})
        rows=aggregate_records(context(),data)
        self.assertEqual(rows[0]['Completed_repeats'],2)
        self.assertEqual(rows[-1]['Estimate'],'')
        self.assertEqual(rows[-1]['Status'],'NOT_ATTEMPTED')


class PublicationTest(unittest.TestCase):
    def test_atomic_append_own_markers_retry_and_foreign_rows_preserved(self):
        book=FakeBook();old=copy.deepcopy(book.rows)
        publisher=AnalysisSheet(book);row=record()
        receipt=publisher.publish(row)
        retry=publisher.publish(record())
        self.assertEqual(len(book.append_calls),1)
        self.assertTrue(receipt['appended']);self.assertFalse(retry['appended'])
        self.assertEqual(receipt['receipt_sha256'],retry['receipt_sha256'])
        self.assertEqual(book.rows[7],old[7]);self.assertEqual(book.rows[8],old[8])
        self.assertEqual(book.rows[9][INDEX['Readback_status']],'VERIFIED')

    def test_atomic_append_handles_new_other_server_row(self):
        book=FakeBook();book.concurrent=True
        receipt=AnalysisSheet(book).publish(record())
        self.assertEqual(receipt['row'],10)
        self.assertEqual(book.rows[9][0],'STATUS')

    def test_timeout_after_append_recovery_never_duplicates(self):
        book=FakeBook();book.timeout_after_append=True
        publisher=AnalysisSheet(book)
        with self.assertRaises(TimeoutError):publisher.publish(record())
        receipt=publisher.publish(record())
        self.assertFalse(receipt['appended']);self.assertEqual(len(book.append_calls),1)

    def test_existing_conflict_duplicate_or_header_mismatch_stop_writes(self):
        book=FakeBook();publisher=AnalysisSheet(book);publisher.publish(record())
        with self.assertRaises(PublicationConflict):publisher.publish(record(Gain=.5))
        self.assertEqual(len(book.append_calls),1)
        book.rows[10]=list(book.rows[9])
        with self.assertRaises(PublicationConflict):publisher.publish(record())
        book.rows[7][1]='wrong'
        with self.assertRaises(PublicationConflict):publisher.publish(record('different'))

    def test_readback_corruption_does_not_mark_verified(self):
        book=FakeBook();book.corrupt=True
        with self.assertRaises(PublicationConflict):AnalysisSheet(book).publish(record())
        self.assertFalse(book.marker_calls)

    def test_capacity_native_constraint_and_partial_outbox(self):
        book=FakeBook();book.other_cells=10_000_000-2000*152
        with self.assertRaises(CapacityBlocked):AnalysisSheet(book).publish(record())
        self.assertFalse(book.append_calls)
        book.other_cells=0;book.validation=True
        with self.assertRaises(PublicationConflict):AnalysisSheet(book).publish(record())
        book.validation=False
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue(record('a'));out.enqueue(record('b'))
            first=out.publish(AnalysisSheet(book),max_records=1)
            self.assertEqual(first['verified_records'],1);self.assertEqual(first['pending_records'],1)
            self.assertEqual(first['status'],'PARTIAL_OR_PENDING_PUBLICATION')
            second=out.publish(AnalysisSheet(book),max_records=1)
            self.assertEqual(second['verified_records'],2);self.assertEqual(second['pending_records'],0)
            self.assertFalse(second['experiment_complete'])

    def test_outbox_csv_integrity_failure_retains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record())
            path.with_suffix('.csv').write_text('changed')
            book=FakeBook();result=out.publish(AnalysisSheet(book))
            self.assertEqual(result['status'],'PARTIAL_OR_PENDING_PUBLICATION')
            self.assertEqual(result['errors'][0]['error_type'],'PublicationConflict')
            self.assertFalse(book.append_calls)


class FakeDelivery:
    def __init__(self):self.calls=[]
    def publish(self, record):
        self.calls.append(record['Record_ID'])
        return dict(record_id=record['Record_ID'], scientific_sha256=digest(scientific_payload(record)),
                    status='VALUES_READBACK_VERIFIED', spreadsheet_id='FAKE_ONLY_NO_NETWORK')


class SimulatedPowerLoss(BaseException):
    pass


class IndexedOutboxTest(unittest.TestCase):
    def test_due_queue_uses_bounded_index_without_global_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory)
            db=out._db()
            try:
                plan=db.execute("EXPLAIN QUERY PLAN SELECT * FROM records WHERE state='pending' AND priority=? AND next_attempt<=? ORDER BY next_attempt,sequence LIMIT ?",(0,1.,25)).fetchall()
            finally:db.close()
            detail=' '.join(str(row[3]) for row in plan)
            self.assertIn('pending_due',detail)
            self.assertNotIn('TEMP B-TREE',detail)

    def test_pending_hardlink_receipt_commit_keeps_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record())
            self.assertTrue((out.pending/path.name).samefile(path))
            result=out.publish(FakeDelivery(),max_records=1)
            self.assertEqual(result['verified_records'],1)
            self.assertFalse((out.pending/path.name).exists())
            self.assertTrue(path.exists());self.assertTrue(path.with_suffix('.csv').exists())
            receipt=load_json(out.root/'receipts'/path.name)
            self.assertEqual(receipt['payload_sha256'],digest({k:v for k,v in receipt.items() if k!='payload_sha256'}))

    def test_intent_before_files_repairs_after_producer_crash_without_history_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);original=record()
            with patch.object(out,'_materialize',side_effect=SimulatedPowerLoss):
                with self.assertRaises(SimulatedPowerLoss):out.enqueue(original)
            self.assertEqual(out.counts(),{'total':1,'verified':0,'pending':1})
            with patch.object(Path,'glob',side_effect=AssertionError('historical scan forbidden')):
                recovered=Outbox(directory)
                status=recovered.publish(FakeDelivery(),max_records=1)
            self.assertEqual(status['verified_records'],1)
            path=recovered.root/(digest(original['Record_ID'])+'.json')
            self.assertEqual(load_json(path)['record']['Created_UTC'],original['Created_UTC'])

    def test_json_before_csv_crash_retry_keeps_first_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);original=record()
            def interrupted(name,entry,**kwargs):
                immutable_json(out.root/name,entry)
                raise SimulatedPowerLoss()
            with patch.object(out,'_materialize',side_effect=interrupted):
                with self.assertRaises(SimulatedPowerLoss):out.enqueue(original)
            retry=dict(original,Created_UTC='2099-01-01T00:00:00Z')
            path=Outbox(directory).enqueue(retry)
            self.assertEqual(load_json(path)['record']['Created_UTC'],original['Created_UTC'])
            self.assertIn(original['Created_UTC'],path.with_suffix('.csv').read_text())

    def test_legacy_csv_before_json_recovers_timestamp_on_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);original=record()
            path=out.root/(digest(original['Record_ID'])+'.csv')
            path.write_bytes(out._csv(original))
            retried=dict(original,Created_UTC='2099-01-01T00:00:00Z')
            saved=out.enqueue(retried)
            self.assertEqual(load_json(saved)['record']['Created_UTC'],original['Created_UTC'])

    def test_receipt_before_index_commit_recovery_needs_no_remote_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record());delivery=FakeDelivery()
            def interrupted(row,entry,receipt):
                immutable_json(out.root/'receipts'/row['filename'],receipt)
                raise SimulatedPowerLoss()
            with patch.object(out,'_commit_verified',side_effect=interrupted):
                with self.assertRaises(SimulatedPowerLoss):out.publish(delivery,1)
            self.assertTrue((out.pending/path.name).exists())
            self.assertEqual(out.counts()['pending'],1)
            recovered=Outbox(directory);status=recovered.publish(delivery,1)
            self.assertEqual(len(delivery.calls),1)
            self.assertEqual(status['verified_records'],1)
            self.assertFalse((out.pending/path.name).exists())

    def test_index_commit_before_marker_cleanup_recovers_without_history_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record());delivery=FakeDelivery()
            cleanup=out._cleanup_verified
            call_count=[0]
            def interrupted(limit):
                call_count[0]+=1
                if call_count[0]==2:raise SimulatedPowerLoss()
                return cleanup(limit)
            with patch.object(out,'_cleanup_verified',side_effect=interrupted):
                with self.assertRaises(SimulatedPowerLoss):out.publish(delivery,1)
            self.assertEqual(out.counts()['verified'],1)
            self.assertTrue((out.pending/path.name).exists())
            with patch.object(Path,'glob',side_effect=AssertionError('historical scan forbidden')):
                result=Outbox(directory).publish(delivery,1)
            self.assertEqual(len(delivery.calls),1)
            self.assertFalse((out.pending/path.name).exists())
            self.assertEqual(result['attempted_this_batch'],0)

    def test_receipt_checksum_and_csv_binding_are_required_before_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record());entry=load_json(path)
            wrapped=out._sealed_receipt(FakeDelivery().publish(entry['record']),entry)
            wrapped['delivery']['status']='MODIFIED'
            atomic_json(out.root/'receipts'/path.name,wrapped)
            result=out.publish(FakeDelivery(),1)
            self.assertEqual(result['pending_records'],1)
            self.assertIn('checksum',result['errors'][0]['reason'])
            self.assertTrue((out.pending/path.name).exists())
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);path=out.enqueue(record());entry=load_json(path)
            wrapped=out._sealed_receipt(FakeDelivery().publish(entry['record']),entry)
            wrapped['csv_sha256']='changed';wrapped['payload_sha256']=digest(out._receipt_body(wrapped))
            atomic_json(out.root/'receipts'/path.name,wrapped)
            result=out.publish(FakeDelivery(),1)
            self.assertIn('different CSV',result['errors'][0]['reason'])

    def test_one_time_legacy_migration_keeps_verified_count_cached(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'analysis_outbox';root.mkdir()
            row=record();name=digest(row['Record_ID'])+'.json';raw=Outbox._csv(row)
            entry=dict(record=row,scientific_sha256=digest(scientific_payload(row)),csv_sha256=hashlib.sha256(raw).hexdigest())
            atomic_json(root/name,entry);(root/name).with_suffix('.csv').write_bytes(raw)
            legacy=AnalysisSheet(FakeBook()).publish(row)
            atomic_json(root/'receipts'/name,legacy)
            out=Outbox(directory)
            self.assertEqual(out.counts(),{'total':1,'verified':1,'pending':0})
            with patch.object(Path,'glob',side_effect=AssertionError('historical scan forbidden')):
                self.assertEqual(Outbox(directory).publish(FakeDelivery(),1)['verified_records'],1)

    def test_priority_failure_backoff_does_not_starve_new_central_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue(record())
            class Delivery(FakeDelivery):
                def publish(self,row):
                    if row['Record_type']=='RESPONSE':raise PermissionError('detail permission missing')
                    return super().publish(row)
            delivery=Delivery();failed=out.publish(delivery,1)
            self.assertEqual(failed['pending_records'],1)
            out.enqueue(make_record(context(),'SUMMARY','primary',HQNR=.95,Selection='EXACT_FINAL'))
            success=out.publish(delivery,1)
            self.assertEqual(success['verified_records'],1)
            self.assertEqual(success['pending_records'],1)
            self.assertFalse(success['errors'])
            self.assertEqual(out.publish(delivery,1)['attempted_this_batch'],0)

    def test_publisher_http_does_not_hold_enqueue_lock_or_sqlite_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue(record('first'))
            started,released=threading.Event(),threading.Event()
            class Delivery(FakeDelivery):
                def publish(self,row):
                    started.set()
                    if not released.wait(4):raise RuntimeError('test timeout')
                    return super().publish(row)
            with ThreadPoolExecutor(max_workers=2) as pool:
                background=pool.submit(out.publish,Delivery(),1)
                self.assertTrue(started.wait(2))
                try:
                    queued=pool.submit(out.enqueue,record('second')).result(timeout=2)
                    self.assertTrue(queued.exists())
                finally:released.set()
                status=background.result(timeout=2)
            self.assertEqual(status['total_records'],2);self.assertEqual(status['pending_records'],1)

    def test_pending_batch_work_is_bounded_not_proportional_to_verified_history(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);delivery=FakeDelivery()
            out.enqueue_many(record(str(i)) for i in range(60))
            self.assertEqual(out.publish(delivery,60)['verified_records'],60)
            out.enqueue(record('new'))
            from ta2 import reporting
            original_load=reporting.load_json;reads=[]
            def counted(path):reads.append(str(path));return original_load(path)
            with patch.object(Path,'glob',side_effect=AssertionError('historical scan forbidden')):
                with patch('ta2.reporting.load_json',side_effect=counted):
                    status=out.publish(delivery,1)
            self.assertEqual(status['verified_records'],61)
            self.assertLessEqual(len(reads),6)
            self.assertEqual(status['attempted_this_batch'],1)


class BatchDelivery(FakeDelivery):
    def __init__(self):super().__init__();self.batches=[]
    def publish_many(self, records):
        self.batches.append([r['Record_ID'] for r in records])
        return [super(BatchDelivery,self).publish(r) for r in records]


def ready_now(outbox):
    db=outbox._db()
    try:
        db.execute('UPDATE records SET next_attempt=0');db.commit()
    finally:db.close()


class BatchOutboxTest(unittest.TestCase):
    def test_batch_limit_and_backpressure_not_one_call_per_record(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue_many(record('batch/'+str(i)) for i in range(210))
            adapter=BatchDelivery();state=out.publish(adapter,max_records=205)
            self.assertEqual([len(b) for b in adapter.batches],[100,100,5])
            self.assertEqual(state['remote_batches_this_call'],3)
            self.assertEqual(state['verified_records'],205);self.assertEqual(state['pending_records'],5)

    def test_batches_do_not_mix_priorities_and_central_goes_first(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory)
            detail=record('detail')
            status=status_record(context(),'RUNNING',10)
            summary=make_record(context(),'SUMMARY','summary',Selection='EXACT_FINAL',HQNR=.95)
            out.enqueue_many([detail,status,summary]);adapter=BatchDelivery()
            out.publish(adapter,20)
            self.assertEqual(adapter.batches,[[summary['Record_ID']],[status['Record_ID']],[detail['Record_ID']]])

    def test_real_sheet_adapter_batches_append_and_readback_with_bounded_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue_many(record('sheet/'+str(i)) for i in range(50))
            book=FakeBook();adapter=AnalysisSheet(book)
            with patch.object(book,'fetch_sheet_metadata',wraps=book.fetch_sheet_metadata) as metadata:
                with patch.object(book,'values_batch_get',wraps=book.values_batch_get) as read:
                    state=out.publish(adapter,50)
            self.assertEqual(state['verified_records'],50)
            self.assertEqual(len(book.append_calls),1)
            self.assertEqual(len(book.append_calls[0][2]['values']),50)
            self.assertEqual(len(book.marker_calls),1)
            self.assertLessEqual(metadata.call_count,8)
            self.assertLessEqual(read.call_count,6)

    def test_uncertain_multirow_append_recovery_no_duplicate_or_premature_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue_many(record('uncertain/'+str(i)) for i in range(10))
            book=FakeBook();book.timeout_after_append=True;adapter=AnalysisSheet(book)
            failed=out.publish(adapter,10)
            self.assertEqual(failed['verified_records'],0);self.assertEqual(failed['pending_records'],10)
            self.assertEqual(len(book.append_calls),1)
            ready_now(out)
            recovered=out.publish(adapter,10)
            self.assertEqual(recovered['verified_records'],10)
            self.assertEqual(len(book.append_calls),1)

    def test_late_remote_group_failure_keeps_whole_batch_pending(self):
        class Partial(BatchDelivery):
            def __init__(self):super().__init__();self.remote={};self.fail=True
            def publish_many(self, records):
                self.batches.append([r['Record_ID'] for r in records])
                result=[]
                for i,row in enumerate(records):
                    if i==2 and self.fail:
                        self.fail=False
                        raise PermissionError('Later remote destination not writable')
                    previous=self.remote.setdefault(row['Record_ID'],scientific_payload(row))
                    if previous!=scientific_payload(row):raise AssertionError('Remote conflict')
                    result.append(FakeDelivery.publish(self,row))
                return result
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue_many(record('partial/'+str(i)) for i in range(5));adapter=Partial()
            first=out.publish(adapter,5)
            self.assertEqual(len(adapter.remote),2)
            self.assertEqual(first['verified_records'],0);self.assertEqual(first['pending_records'],5)
            self.assertFalse((out.root/'receipts').exists())
            ready_now(out)
            second=out.publish(adapter,5)
            self.assertEqual(second['verified_records'],5);self.assertEqual(len(adapter.remote),5)

    def test_all_receipts_validated_before_any_verified_commit(self):
        class Wrong(BatchDelivery):
            def publish_many(self,records):
                result=super().publish_many(records)
                result[-1]['status']='PARTIAL'
                return result
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory);out.enqueue_many(record('badreceipt/'+str(i)) for i in range(4))
            status=out.publish(Wrong(),4)
            self.assertEqual(status['verified_records'],0);self.assertEqual(status['pending_records'],4)
            self.assertFalse((out.root/'receipts').exists())


if __name__=='__main__':unittest.main()
