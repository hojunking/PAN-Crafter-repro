import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ta2.common import CAMPAIGN,atomic_json,digest,load_json
from ta2.publication import (PublicationRouter,DriveDetailFactory,PublicationPending,central_record,_acl,POLICY,
                            PREALLOCATED_SCHEMA,USER_APPROVED_PUBLIC_PARENT)
from ta2.reporting import (CapacityBlocked,PublicationConflict,SPREADSHEET_ID,SHEET_ID,COLUMNS,INDEX,
                          make_record,scientific_payload,AnalysisSheet)
from ta2.test_reporting import FakeBook,area


class BatchBook(FakeBook):
    """Network-free book with atomic multi-row append and API-call counters."""
    def __init__(self):super().__init__();self.api_calls=0
    def values_get(self,*args,**kwargs):
        self.api_calls+=1;return super().values_get(*args,**kwargs)
    def values_batch_get(self,ranges,params=None):
        self.api_calls+=1
        return {'valueRanges':[dict(range=r,values=FakeBook.values_get(self,r,params).get('values',[])) for r in ranges]}
    def fetch_sheet_metadata(self,params):
        self.api_calls+=1
        if not params.get('includeGridData'):return super().fetch_sheet_metadata(params)
        blocks=[]
        for target in params['ranges']:
            first,last,left,right=area(target);rows=[]
            for number in range(first,last+1):
                values=self.rows.get(number,['']*152)[left:right+1];cells=[]
                for value in values:
                    kind='boolValue' if isinstance(value,bool) else 'numberValue' if isinstance(value,(int,float)) else 'stringValue'
                    cells.append({'userEnteredValue':{kind:value}} if value!='' else {})
                if self.validation:cells[0]['dataValidation']={'condition':{'type':'ONE_OF_LIST'}}
                rows.append(dict(values=cells))
            blocks.append(dict(startRow=first-1,startColumn=left,rowData=rows))
        return dict(sheets=[dict(properties=dict(sheetId=SHEET_ID,title='analysis'),data=blocks)])
    def values_append(self,text,params,body):
        self.api_calls+=1;self.append_calls.append((text,params,body))
        if self.concurrent:
            self.rows[max(self.rows)+1]=['STATUS']+['']*151;self.concurrent=False
        first=max(max(self.rows)+1,area(text)[0])
        for i,values in enumerate(body['values']):self.rows[first+i]=list(values)
        last=first+len(body['values'])-1;self.count=max(self.count,last)
        if self.timeout_after_append:self.timeout_after_append=False;raise TimeoutError('Unknown successful batch append')
        return dict(updates=dict(updatedRange=f'analysis!A{first}:EV{last}'))
    def values_batch_update(self,body):self.api_calls+=1;return super().values_batch_update(body)
    def batch_update(self,body):self.api_calls+=1;return super().batch_update(body)


class Plan:
    def queue(self,server):
        return [dict(run_id=f'run{i}',queue_seq=i,case_id=f'TA2-B{i:02d}',server=server,dataset='WV3') for i in range(1,25)]


def row(kind='RESPONSE',**fields):
    context=dict(server='s1',dataset='WV3',case_id='TA2-B01',run_id='run1',seed=1,replica=1)
    return make_record(context,kind,'test',**fields)


class Adapter:
    def __init__(self,id,limit=None):self.id=id;self.rows={};self.limit=limit
    def publish(self,record):
        if record['Record_ID'] not in self.rows and self.limit is not None and len(self.rows)>=self.limit:
            error=CapacityBlocked('capacity');error.safe_to_rollover=True;raise error
        self.rows[record['Record_ID']]=record
        return dict(record_id=record['Record_ID'],scientific_sha256=digest(scientific_payload(record)),
                    spreadsheet_id=self.id,status='VALUES_READBACK_VERIFIED',row=len(self.rows))


class Factory:
    def __init__(self,limit=None):self.books={};self.limit=limit;self.calls=[]
    def ensure(self,key,title):
        self.calls.append(key)
        if key not in self.books:self.books[key]=Adapter('book-'+key,self.limit)
        book=self.books[key]
        return dict(key=key,title=title,spreadsheet_id=book.id,url='https://observed.example/'+book.id),book


class PublicationTests(unittest.TestCase):
    def test_router_construction_offline_and_server_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=Factory();router=PublicationRouter(temp,'s1',plan=Plan(),factory=factory,central=Adapter(SPREADSHEET_ID))
            self.assertEqual(factory.calls,[])
            self.assertEqual(router.bucket(row()),1)
            changed=row();changed['Run_ID']='run13';changed['Case_ID']='TA2-B13'
            self.assertEqual(router.bucket(changed),2)
            changed['Server']='s3'
            with self.assertRaises(PublicationConflict):router.bucket(changed)

    def test_bounded_fixed_central_policy_no_metric_value_selection(self):
        self.assertTrue(central_record(row('SUMMARY',Selection='EXACT_FINAL')))
        self.assertFalse(central_record(row('RESPONSE',Gain=1.1)))
        self.assertFalse(central_record(row('GRAD')))
        for metric in ('hqnr/mean','gain_fr/mean','proxy_fr_LMS_gradient/mean'):
            record=row('SEED_AGG',Completed_repeats=3,Status='COMPLETE_3_SEEDS',Metric_name=metric,Estimate=-1000)
            self.assertTrue(central_record(record))
            record['Completed_repeats']=2;self.assertFalse(central_record(record))
        self.assertFalse(central_record(row('SEED_AGG',Completed_repeats=3,Status='COMPLETE_3_SEEDS',Metric_name='psnr/mean')))
        self.assertTrue(central_record(row('PAIRED',Completed_repeats=3,Status='COMPLETE',Metric_name='hqnr',N_train_seeds=3)))
        self.assertFalse(central_record(row('CURVE',Step=2020,Metric_name='loss/rec')))
        self.assertTrue(central_record(row('CURVE',Step=25250)))
        self.assertFalse(central_record(row('CURVE',Step=2020)))
        long=row('CURVE',Step=2020);long['Run_ID']='TA2_F100K_R1'
        self.assertTrue(central_record(long))
        long['Step']=50000;self.assertFalse(central_record(long))
        self.assertTrue(central_record(row('STATUS',Status='RUNNING',Step=1)))
        self.assertFalse(central_record(row('STATUS',Status='RUNNING',Step=1000)))
        self.assertTrue(central_record(row('UNIT',Metric_name='disposable_smoke_updates')))

    def test_detail_and_central_destinations_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=Factory();central=Adapter(SPREADSHEET_ID)
            router=PublicationRouter(temp,'s1',plan=Plan(),factory=factory,central=central)
            detail=router.publish(row());summary=router.publish(row('SUMMARY',Selection='EXACT_FINAL'))
            self.assertNotEqual(detail['spreadsheet_id'],summary['spreadsheet_id'])
            self.assertEqual(summary['spreadsheet_id'],SPREADSHEET_ID)
            self.assertEqual(len(central.rows),2) # actual detail-link ASSET plus summary
            self.assertEqual(sum(len(b.rows) for b in factory.books.values()),1)
            self.assertEqual(router.publish(row())['spreadsheet_id'],detail['spreadsheet_id'])
            self.assertEqual(sum(len(b.rows) for b in factory.books.values()),1)

    def test_safe_rollover_preserves_attempt_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=Factory(limit=1);router=PublicationRouter(temp,'s1',plan=Plan(),factory=factory,central=Adapter(SPREADSHEET_ID))
            first=router.publish(row())
            second=row(Metric_name='other');r2=router.publish(second)
            self.assertNotEqual(first['spreadsheet_id'],r2['spreadsheet_id'])
            self.assertTrue(r2['destination_key'].endswith('P02'))
            self.assertEqual(router.publish(row())['spreadsheet_id'],first['spreadsheet_id'])
            files=list((Path(temp)/'publication/record_routes').rglob('capacity_*.json'))
            self.assertEqual(len(files),1);self.assertTrue(load_json(files[0])['no_append_confirmed'])

    def test_central_capacity_does_not_move_summary_into_details(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=Factory();router=PublicationRouter(temp,'s1',plan=Plan(),factory=factory,central=Adapter(SPREADSHEET_ID,limit=0))
            with self.assertRaises(CapacityBlocked):router.publish(row('SUMMARY',Selection='EXACT_FINAL'))
            self.assertEqual(factory.calls,[])

    def test_destination_failure_stays_pending_no_partial_receipt(self):
        class Denied(Factory):
            def ensure(self,*args):raise PublicationPending('quota')
        with tempfile.TemporaryDirectory() as temp:
            central=Adapter(SPREADSHEET_ID);router=PublicationRouter(temp,'s1',plan=Plan(),factory=Denied(),central=central)
            with self.assertRaises(PublicationPending):router.publish(row())
            self.assertEqual(central.rows,{})

    def test_immutable_record_route_prevents_redirect_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=Factory();router=PublicationRouter(temp,'s1',plan=Plan(),factory=factory,central=Adapter(SPREADSHEET_ID))
            record=row();router.publish(record)
            record['Gain']=.1
            with self.assertRaises(PublicationConflict):router.publish(record)

    def test_acl_only_existing_named_principals(self):
        grants=_acl([dict(type='user',emailAddress='owner@example.com',role='owner'),
                     dict(type='group',emailAddress='team@example.com',role='reader'),
                     dict(type='anyone',role='reader'),dict(type='domain',role='writer')])
        self.assertEqual({g['type'] for g in grants},{'user','group'})
        self.assertEqual(next(g for g in grants if g['type']=='user')['role'],'writer')
        with self.assertRaises(PublicationPending):_acl([dict(type='anyone',role='reader')])

    def test_creation_timeout_discovery_no_blind_duplicate_post(self):
        class TimedOut(DriveDetailFactory):
            def __init__(self,path):super().__init__(path,registry=path/'not-configured.json');self.posts=0
            def _folder(self):
                atomic_json(self.root/'folder.json',dict(id='folder',driveId='drive'));return 'folder'
            def _list(self,query):return []
            def _permissions(self,fileid):return [dict(type='user',role='owner',emailAddress='owner@example.com')]
            def _request(self,method,path,**kwargs):
                if method=='post':self.posts+=1;raise TimeoutError('unknown creation result')
                raise AssertionError('unexpected request')
        with tempfile.TemporaryDirectory() as temp:
            factory=TimedOut(Path(temp))
            with self.assertRaises(TimeoutError):factory.ensure('TA2_s1_B01_P01','title')
            with self.assertRaises(PublicationPending):factory.ensure('TA2_s1_B01_P01','title')
            self.assertEqual(factory.posts,1)

    def test_serviceaccount_mydrive_blocked_before_intent(self):
        class NoQuota(DriveDetailFactory):
            def _request(self,method,path,**kwargs):
                self.assertion=method
                return dict(id=SPREADSHEET_ID,parents=['root'])
        client=SimpleNamespace(http_client=SimpleNamespace(auth=SimpleNamespace(service_account_email='s@example.com',_subject=None)))
        with tempfile.TemporaryDirectory() as temp,mock.patch.dict('os.environ',{},clear=True):
            factory=NoQuota(Path(temp),client=client,registry=Path(temp)/'not-configured.json')
            with self.assertRaises(PublicationPending):factory.ensure('TA2_s1_B01_P01','title')
            self.assertEqual(list(Path(temp).rglob('*intent*.json')),[])
            self.assertEqual(factory.assertion,'get')

    def test_analysis_adapter_explicit_new_id_default_still_central_only(self):
        book=SimpleNamespace(id='detail-id')
        with self.assertRaises(PublicationConflict):AnalysisSheet(book)
        adapter=AnalysisSheet(book,spreadsheet_id='detail-id',max_cells=5_000_000)
        self.assertEqual(adapter.spreadsheet_id,'detail-id')

    def preallocated_factory(self,path,*,public=True):
        from ta2.test_reporting import FakeBook
        book=FakeBook();book.id='precreated-id'
        client=SimpleNamespace(open_by_key=lambda file_id:book)
        key='TA2_s1_B01_P01';title='PAN TA2 DETAIL S1 B01 P01 2026-10-06'
        parent=USER_APPROVED_PUBLIC_PARENT if public else 'private-parent'
        entry=dict(spreadsheet_id=book.id,parent_id=parent,title=title,
                   public_parent_inheritance_user_approved=public)
        registry=Path(path)/'registry.json'
        atomic_json(registry,dict(schema=PREALLOCATED_SCHEMA,campaign=CAMPAIGN,
                                 preallocated_only=True,entries={key:entry}))
        class Precreated(DriveDetailFactory):
            def _request(self,method,target,**kwargs):
                if method!='get':raise AssertionError('Preallocated destination must perform no Drive mutation')
                if target=='/files/'+book.id:
                    return dict(id=book.id,name=title,mimeType='application/vnd.google-apps.spreadsheet',
                                parents=[parent],capabilities=dict(canEdit=True),webViewLink='https://observed/book')
                if target=='/files/'+parent:return dict(id=parent,mimeType='application/vnd.google-apps.folder')
                raise AssertionError('Unexpected Drive read: '+target)
            def _permissions(self,file_id):
                base=[dict(type='user',role='owner',emailAddress='owner@example.com'),
                      dict(type='user',role='writer',emailAddress='service@example.com')]
                if public and file_id!=SPREADSHEET_ID:base.append(dict(type='anyone',role='writer'))
                return base
        factory=Precreated(Path(path)/'books',client=client,registry=registry)
        return factory,key,title,entry,registry,book

    def test_preallocated_user_owned_mydrive_requires_no_app_properties_or_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp)
            info,adapter=factory.ensure(key,title)
            self.assertEqual(info['spreadsheet_id'],book.id)
            self.assertEqual(info['visibility'],'user_approved_existing_public_parent')
            self.assertEqual(book.append_calls,[])
            self.assertEqual(book.format_calls,[])
            self.assertFalse((factory.root/'creation_intents').exists())
            self.assertEqual(factory.ensure(key,title)[0],info)
            with self.assertRaises(PublicationPending):factory.ensure('TA2_s1_B01_P02',title.replace('P01','P02'))

    def test_preallocated_identity_and_registry_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp,public=False)
            factory.ensure(key,title)
            changed=load_json(registry);changed['entries'][key]['spreadsheet_id']='redirected'
            atomic_json(registry,changed)
            with self.assertRaises(PublicationConflict):factory.ensure(key,title)
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp)
            doc=factory._request('get','/files/'+book.id);doc['parents']=['different-parent']
            with mock.patch.object(factory,'_request',return_value=doc):
                with self.assertRaises(PublicationConflict):factory.ensure(key,title)

    def test_preallocated_public_visibility_is_exact_parent_user_approval_only(self):
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp)
            changed=load_json(registry);changed['entries'][key]['public_parent_inheritance_user_approved']=False
            atomic_json(registry,changed)
            with self.assertRaises(PublicationPending):factory.ensure(key,title)
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp,public=False)
            changed=load_json(registry);changed['entries'][key]['public_parent_inheritance_user_approved']=True
            atomic_json(registry,changed)
            with self.assertRaises(PublicationConflict):factory.ensure(key,title)

    def test_preallocated_missing_parents_requires_exact_live_child_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp)
            original=factory._request
            def omit_parent(method,target,**kwargs):
                doc=original(method,target,**kwargs)
                if target=='/files/'+book.id:doc.pop('parents')
                return doc
            member=dict(id=book.id,name=title,mimeType='application/vnd.google-apps.spreadsheet',parents=[entry['parent_id']])
            with mock.patch.object(factory,'_request',side_effect=omit_parent),mock.patch.object(factory,'_list',return_value=[member]) as listing:
                info,adapter=factory.ensure(key,title)
            self.assertEqual(info['parent_verification'],'exact_folder_title_query_registered_id_mime_and_parent')
            self.assertIn("'"+entry['parent_id']+"' in parents",listing.call_args.args[0])
            self.assertIn("name = '"+title+"'",listing.call_args.args[0])

    def test_missing_parent_membership_does_not_accept_wrong_or_ambiguous_child(self):
        for variant in ('wrong_id','wrong_mime','wrong_parent','wrong_name','duplicate','empty'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp:
                factory,key,title,entry,registry,book=self.preallocated_factory(temp)
                original=factory._request
                def omit_parent(method,target,**kwargs):
                    doc=original(method,target,**kwargs)
                    if target=='/files/'+book.id:doc.pop('parents')
                    return doc
                member=dict(id=book.id,name=title,mimeType='application/vnd.google-apps.spreadsheet',parents=[entry['parent_id']])
                if variant=='wrong_id':member['id']='different'
                if variant=='wrong_mime':member['mimeType']='application/vnd.google-apps.folder'
                if variant=='wrong_parent':member['parents']=['different']
                if variant=='wrong_name':member['name']='different'
                matches=[] if variant=='empty' else [member,dict(member)] if variant=='duplicate' else [member]
                with mock.patch.object(factory,'_request',side_effect=omit_parent),mock.patch.object(factory,'_list',return_value=matches):
                    with self.assertRaises(PublicationConflict):factory.ensure(key,title)

    def test_explicit_wrong_parent_never_uses_membership_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            factory,key,title,entry,registry,book=self.preallocated_factory(temp)
            original=factory._request
            def wrong_parent(method,target,**kwargs):
                doc=original(method,target,**kwargs)
                if target=='/files/'+book.id:doc['parents']=[]
                return doc
            with mock.patch.object(factory,'_request',side_effect=wrong_parent),mock.patch.object(factory,'_list') as listing:
                with self.assertRaises(PublicationConflict):factory.ensure(key,title)
                listing.assert_not_called()

    def test_incomplete_or_repeating_drive_pagination_is_not_membership_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            factory=DriveDetailFactory(temp)
            with mock.patch.object(factory,'_request',return_value={'incompleteSearch':True,'files':[]}):
                with self.assertRaises(PublicationPending):factory._list('exact query')
            with mock.patch.object(factory,'_request',return_value={'nextPageToken':'repeated','files':[]}):
                with self.assertRaises(PublicationConflict):factory._list('exact query')

    def test_batch_api_requests_are_constant_per_batch_not_per_row(self):
        counts=[]
        for size in (2,50,100):
            book=BatchBook();adapter=AnalysisSheet(book)
            records=[row(Metric_name=f'm{i}',Gain=.1234567890123456) for i in range(size)]
            receipts=adapter.publish_many(records)
            self.assertEqual(len(receipts),size);self.assertEqual(len(book.append_calls),1)
            self.assertTrue(all(r['status']=='VALUES_READBACK_VERIFIED' for r in receipts))
            self.assertEqual(book.rows[8][0],'REPORT_ONLY')
            self.assertEqual(book.rows[9][INDEX['Gain']],.1234567890123456)
            counts.append(book.api_calls)
            retries=adapter.publish_many(records)
            self.assertFalse(any(r['appended'] for r in retries));self.assertEqual(len(book.append_calls),1)
        self.assertEqual(len(set(counts)),1)
        self.assertLess(counts[0],20)

    def test_batch_uncertain_append_and_concurrent_foreign_row_recovery(self):
        book=BatchBook();book.timeout_after_append=True;book.concurrent=True
        adapter=AnalysisSheet(book);records=[row(Metric_name=f'm{i}') for i in range(20)]
        with self.assertRaises(TimeoutError):adapter.publish_many(records)
        receipts=adapter.publish_many(records)
        self.assertEqual(len(book.append_calls),1);self.assertEqual(receipts[0]['row'],10)
        self.assertTrue(all(not r['appended'] for r in receipts))
        self.assertEqual(book.rows[9][0],'STATUS')

    def test_batch_conflict_duplicate_and_native_validation_fail_closed(self):
        book=BatchBook();adapter=AnalysisSheet(book);records=[row(Metric_name=f'm{i}') for i in range(3)]
        adapter.publish_many(records)
        changed=[dict(records[0],Gain=.5),row(Metric_name='new')]
        with self.assertRaises(PublicationConflict):adapter.publish_many(changed)
        self.assertEqual(len(book.append_calls),1)
        book.rows[12]=list(book.rows[9])
        with self.assertRaises(PublicationConflict):adapter.publish_many(records)
        clean=BatchBook();clean.validation=True
        with self.assertRaises(PublicationConflict):AnalysisSheet(clean).publish_many(records)
        self.assertEqual(clean.append_calls,[])

    def test_router_batch_groups_destinations_and_safe_partial_capacity(self):
        class Sheets:
            def __init__(self):self.books={}
            def ensure(self,key,title):
                if key not in self.books:
                    book=BatchBook();book.id='batch-'+key;book.count=8
                    # First part can hold two records, second is ample.
                    self.books[key]=(book,AnalysisSheet(book,spreadsheet_id=book.id,
                        max_cells=(10 if key.endswith('P01') else 1000)*152))
                book,adapter=self.books[key]
                return dict(key=key,title=title,spreadsheet_id=book.id,url='https://observed/'+book.id),adapter
        with tempfile.TemporaryDirectory() as temp:
            books=Sheets();central=Adapter(SPREADSHEET_ID)
            router=PublicationRouter(temp,'s1',plan=Plan(),factory=books,central=central)
            old=[row(Metric_name=f'm{i}') for i in range(2)]
            delivered=router.publish_many(old)
            new=[row(Metric_name=f'm{i}') for i in range(2,4)]
            retried=router.publish_many(old+new)
            self.assertEqual([r['spreadsheet_id'] for r in retried[:2]],[r['spreadsheet_id'] for r in delivered])
            self.assertTrue(all(r['destination_key'].endswith('P02') for r in retried[2:]))
            self.assertEqual(sum(len(book.append_calls) for book,adapter in books.books.values()),2)
            self.assertEqual(len(central.rows),2) # One manifest for each actual part.


if __name__=='__main__':unittest.main()
