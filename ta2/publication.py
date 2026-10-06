"""Explicitly enabled central-core / per-server detailed publication.

Construction and routing decisions are offline. Only publish() opens a Google
session. Failed/ambiguous creation is never blindly repeated; discovery by
appProperties recovers a successful remote creation before local acknowledgement.
No anonymous/domain sharing, other-tab writes, or existing-sheet reformat occurs.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from ta2.common import CAMPAIGN, ROOT, atomic_json, digest, immutable_json, load_json, now
from ta2.plan import Plan
from ta2.reporting import (AnalysisSheet, COLUMNS, HEADER_ROW, SHEET, SHEET_ID, SPREADSHEET_ID,
    CapacityBlocked, PublicationConflict, make_record, scientific_payload, validate_record)

POLICY='TA2_CORE_CENTRAL_DETAIL_12RUN_5MCELL_v1'
# Deliberately conservative local budgets, not Google's platform cell maximum.
LOCAL_SAFE_CELL_BUDGET=10_000_000
DETAIL_CELL_LIMIT=5_000_000
BUCKET_RUNS=12
CORE_METRICS=frozenset(('hqnr','scc','ergas','d_s','d_lambda','gain_train','gain_rr','gain_fr',
                        'proxy_rr_GT_gradient','proxy_fr_LMS_gradient'))
PAIRED_SEED_METRICS=frozenset(('hqnr','scc','ergas','gain_fr','proxy_fr_LMS_gradient'))
FIXED_PROGRESS={50000:frozenset((1010,25250,50000)),100000:frozenset((2020,50500,100000))}
DRIVE='https://www.googleapis.com/drive/v3'
USER_APPROVED_PUBLIC_PARENT='1lH61W1Enb0BGJYCzurnNIolOHaJFsx9e'
PREALLOCATED_SCHEMA='TA2_PREALLOCATED_DETAIL_BOOKS_v1'


class PublicationPending(RuntimeError):
    """External permissions/quota/ambiguous creation need attention, not retraining."""


def central_record(record):
    """Preregistered fixed lens, never routes based on favorable metric values."""
    kind=record['Record_type'];metric=record.get('Metric_name','')
    if kind=='SUMMARY':return record.get('Selection') in ('HQNR_MAX50','EXACT_FINAL')
    if kind=='CURVE':
        budget=100000 if '_F100K_' in record.get('Run_ID','') else 50000
        return record.get('Step') in FIXED_PROGRESS[budget] and metric==''
    if kind=='STATUS':
        if record.get('Status')=='RUNNING':return record.get('Step')==1
        return record.get('Status') in ('STARTED','COMPLETE','NUMERICAL_FAILURE','FAILED','PAUSED','PENDING_RECOVERY','BLOCKED')
    if kind=='ASSET':return ('lms' in metric.lower() and ('max_abs' in metric or 'rmse' in metric)) or metric in ('detail_manifest','dataset_status')
    if kind=='UNIT':return metric in ('case_count','smoke_case_count','disposable_smoke_updates','inference_finite','resume_equal')
    if kind=='SEED_AGG':
        return (record.get('Completed_repeats')==3 and record.get('Status')=='COMPLETE_3_SEEDS'
                and metric.endswith('/mean') and metric[:-5] in CORE_METRICS)
    if kind=='PAIRED':
        if record.get('Completed_repeats')!=3 or record.get('Status')!='COMPLETE':return False
        if metric.endswith('/mean') and metric[:-5] in CORE_METRICS:return True
        return bool(record.get('Train_seed')!='' and record.get('N_train_seeds')==3 and metric in PAIRED_SEED_METRICS)
    return False


def _acl(permissions):
    """A subset of already-authorized named principals; never broadens link access."""
    output={}
    for permission in permissions:
        if permission.get('deleted'):continue
        if permission.get('type') not in ('user','group'):continue
        email=permission.get('emailAddress')
        if not email:raise PublicationPending('Named central permission has no resolvable email; no guessed sharing')
        role=permission.get('role')
        if role in ('owner','writer','organizer','fileOrganizer'):role='writer'
        elif role in ('reader','commenter'):role='reader'
        else:raise PublicationPending('Unsupported named permission role; no implicit expansion')
        key=(permission['type'],email.lower())
        if key not in output or role=='writer':output[key]=dict(type=permission['type'],emailAddress=email,role=role)
    if not output:raise PublicationPending('No named central owner/editor ACL available for private detail access')
    return [output[key] for key in sorted(output)]


class DriveDetailFactory:
    """Small injected gspread transport; no secrets are logged or copied."""
    def __init__(self, directory, *, credentials=None, client=None, registry=None):
        self.root=Path(directory);self.credentials=credentials;self.client=client;self._ready={}
        self.registry=Path(registry) if registry else ROOT/'ta2/detail_books.json'

    def _preallocated(self,key,title):
        if not self.registry.exists():return None
        registry=load_json(self.registry)
        if registry.get('schema')!=PREALLOCATED_SCHEMA or registry.get('campaign')!=CAMPAIGN:
            raise PublicationConflict('Preallocated detail registry campaign/schema differs')
        entries=registry.get('entries',{})
        if not isinstance(entries,dict):raise PublicationConflict('Invalid preallocated detail registry')
        ids=[row.get('spreadsheet_id') for row in entries.values()]
        if len(set(ids))!=len(ids):raise PublicationConflict('Two detail keys alias the same spreadsheet ID')
        entry=entries.get(key)
        if entry is None:
            if registry.get('preallocated_only',True):
                raise PublicationPending('DETAIL_PART_NOT_PREALLOCATED: retain local records; user-owned file provisioning is required')
            return None
        if entry.get('title')!=title or not entry.get('spreadsheet_id') or not entry.get('parent_id'):
            raise PublicationConflict('Preallocated key/title/ID/parent differs from requested shard')
        if entry['spreadsheet_id']==SPREADSHEET_ID:raise PublicationConflict('Detail registry aliases central workbook')
        return entry

    def _verify_preallocated(self,entry,key,title):
        """Read-only identity/ACL checks: existing user-owned My Drive is valid."""
        file_id=entry['spreadsheet_id'];parent_id=entry['parent_id']
        doc=self._request('get',f'/files/{file_id}',params={'fields':
            'id,name,mimeType,trashed,webViewLink,parents,capabilities(canEdit)'})
        if (doc.get('id')!=file_id or doc.get('name')!=title or doc.get('trashed')
                or doc.get('mimeType')!='application/vnd.google-apps.spreadsheet'
                or not doc.get('capabilities',{}).get('canEdit')):
            raise PublicationConflict('Preallocated remote ID/title/parent/edit capability differs')
        parent=self._request('get',f'/files/{parent_id}',params={'fields':'id,mimeType,trashed'})
        if (parent.get('id')!=parent_id or parent.get('mimeType')!='application/vnd.google-apps.folder'
                or parent.get('trashed')):raise PublicationConflict('Preallocated parent is not the registered live folder')
        if 'parents' in doc:
            if doc['parents']!=[parent_id]:raise PublicationConflict('Preallocated file has explicit different parents')
            doc['parent_verification']='files_get_exact_parent'
        else:
            # Drive can hide a shared child's parents from files.get while a
            # known public-folder children query exposes its ancestry. Verify
            # membership independently; never infer it from registry alone.
            escape=lambda value:value.replace('\\','\\\\').replace("'","\\'")
            matches=self._list(f"trashed = false and '{escape(parent_id)}' in parents and name = '{escape(title)}'")
            if (len(matches)!=1 or matches[0].get('id')!=file_id or matches[0].get('name')!=title
                    or matches[0].get('mimeType')!='application/vnd.google-apps.spreadsheet'
                    or matches[0].get('parents')!=[parent_id] or matches[0].get('trashed')):
                raise PublicationConflict('Read-only exact-folder query did not uniquely prove registered child membership')
            doc['parents']=matches[0]['parents']
            doc['parent_verification']='exact_folder_title_query_registered_id_mime_and_parent'
        public_approved=entry.get('public_parent_inheritance_user_approved') is True
        if public_approved and parent_id!=USER_APPROVED_PUBLIC_PARENT:
            raise PublicationConflict('Public inheritance approval is restricted to the user-supplied folder')
        central=_acl(self._permissions(SPREADSHEET_ID))
        allowed={(p['type'],p['emailAddress'].lower()):p for p in central}
        parent_permissions=self._permissions(parent_id)
        for scope,permissions in (('parent',parent_permissions),('detail',self._permissions(file_id))):
            for permission in permissions:
                if permission.get('deleted') or permission.get('role')=='owner':continue
                if permission.get('type') in ('anyone','domain'):
                    # Preserve only explicitly approved existing inheritance;
                    # no code path here creates, deletes or changes a grant.
                    signature=lambda p:(p.get('type'),p.get('domain'),p.get('role'),bool(p.get('allowFileDiscovery',False)))
                    if not public_approved or not any(signature(p)==signature(permission) for p in parent_permissions):
                        raise PublicationPending('Unapproved public/domain visibility in preallocated destination')
                    continue
                principal=(permission.get('type'),permission.get('emailAddress','').lower())
                if principal not in allowed:raise PublicationPending('Preallocated destination has a new named principal')
                if (permission.get('role') in ('writer','organizer','fileOrganizer')
                        and allowed[principal]['role']!='writer'):
                    raise PublicationPending('Preallocated named principal exceeds existing central access')
        if not doc.get('webViewLink'):raise PublicationPending('Preallocated Drive URL was not observed')
        return doc

    def _ensure_preallocated(self,entry,key,title,receipt):
        entry_sha=digest(entry)
        if receipt.exists():
            saved=load_json(receipt)
            if (saved.get('preallocated_entry_sha256')!=entry_sha or saved.get('key')!=key
                    or saved.get('spreadsheet_id')!=entry['spreadsheet_id']):
                raise PublicationConflict('Immutable preallocated destination registry changed')
        else:saved=None
        doc=self._verify_preallocated(entry,key,title)
        book=self._client().open_by_key(entry['spreadsheet_id'])
        # Header/native structure must already have been provisioned; never
        # initialize or delete any user-created tab in this existing-file path.
        adapter=AnalysisSheet(book,spreadsheet_id=entry['spreadsheet_id'],max_cells=DETAIL_CELL_LIMIT)
        adapter.preflight(extra_rows=0)
        if saved is None:
            saved=dict(key=key,title=title,spreadsheet_id=doc['id'],url=doc['webViewLink'],
                parent_id=entry['parent_id'],schema='AAX_ANALYSIS152_v2',policy=POLICY,
                parent_verification=doc['parent_verification'],
                preallocated_entry_sha256=entry_sha,headers_readback=True,creation_reconciled=True,
                visibility='user_approved_existing_public_parent' if entry.get('public_parent_inheritance_user_approved')
                else 'named_central_principals_only',publication_does_not_create_permissions=True)
            immutable_json(receipt,saved)
        result=saved,adapter;self._ready[key]=result
        return result

    def _client(self):
        if self.client is None:
            import gspread
            import google.auth
            from .quota import LimitedHTTPClient
            credentials,_=google.auth.load_credentials_from_file(
                str(self.credentials or os.environ.get('TA2_GOOGLE_CREDENTIALS') or ROOT/'gspread/account.json'),
                scopes=['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
            self.client=gspread.authorize(credentials,http_client=LimitedHTTPClient)
        return self.client

    def _request(self,method,path,**kwargs):
        if path.startswith('/files'):
            kwargs['params']=dict(kwargs.get('params',{}),supportsAllDrives=True)
        return self._client().http_client.request(method,DRIVE+path,**kwargs).json()

    def _list(self,query):
        values=[];token=None;seen_tokens=set()
        while True:
            params=dict(q=query,pageSize=100,includeItemsFromAllDrives=True,
                        fields='nextPageToken,incompleteSearch,files(id,name,mimeType,appProperties,webViewLink,parents,ownedByMe,trashed,driveId)')
            if token:params['pageToken']=token
            page=self._request('get','/files',params=params)
            if page.get('incompleteSearch'):raise PublicationPending('Drive search is incomplete; destination uniqueness is unverified')
            values.extend(page.get('files',[]))
            token=page.get('nextPageToken')
            if not token:return values
            if token in seen_tokens:raise PublicationConflict('Drive pagination token repeated; membership is unverified')
            seen_tokens.add(token)

    def _permissions(self,file_id):
        rows=[];token=None
        while True:
            params=dict(pageSize=100,fields='nextPageToken,permissions(id,type,role,emailAddress,domain,allowFileDiscovery,deleted,permissionDetails)')
            if token:params['pageToken']=token
            page=self._request('get',f'/files/{file_id}/permissions',params=params);rows.extend(page.get('permissions',[]))
            token=page.get('nextPageToken')
            if not token:return rows

    def central(self):
        return AnalysisSheet(self._client().open_by_key(SPREADSHEET_ID))

    def _folder(self):
        path=self.root/'folder.json'
        if path.exists():
            saved=load_json(path)
            self._check_folder(saved['id'])
            return saved['id']
        explicit=os.environ.get('TA2_DETAIL_FOLDER')
        source=self._request('get',f'/files/{SPREADSHEET_ID}',params={'fields':'id,driveId,parents'})
        inherited=(source.get('parents') or [None])[0] if source.get('driveId') else None
        if explicit or inherited:
            folder=self._check_folder(explicit or inherited)
            immutable_json(path,dict(id=folder['id'],name=folder['name'],driveId=folder.get('driveId'),
                location_policy='explicit_TA2_DETAIL_FOLDER' if explicit else 'same_shared_drive_parent_as_central'))
            return folder['id']
        credentials=getattr(self._client().http_client,'auth',None)
        if getattr(credentials,'service_account_email',None) and not getattr(credentials,'_subject',None):
            raise PublicationPending('SERVICE_ACCOUNT_NO_DRIVE_STORAGE: set TA2_DETAIL_FOLDER to an authorized Shared Drive folder or TA2_GOOGLE_CREDENTIALS to authorized_user credentials; no create was attempted')
        values=self._list("trashed = false and 'me' in owners and 'root' in parents and mimeType = 'application/vnd.google-apps.folder' and name = 'ChatGPT'")
        if len(values)>1:raise PublicationPending('Ambiguous owned ChatGPT folder; no arbitrary location choice')
        if values:
            folder=values[0]
            if any(p.get('type') in ('anyone','domain') or p.get('type')=='group' or
                   (p.get('type')=='user' and p.get('role')!='owner') for p in self._permissions(folder['id'])):
                raise PublicationPending('ChatGPT folder is shared; refusing inherited access expansion for private detail books')
        else:
            intent=self.root/'folder_creation_intent.json'
            if intent.exists():raise PublicationPending('FOLDER_CREATION_UNRESOLVED: discovery found no unique folder; no blind recreate')
            immutable_json(intent,dict(name='ChatGPT',requested=now()))
            folder=self._request('post','/files',params={'fields':'id,name,mimeType'},
                json={'name':'ChatGPT','mimeType':'application/vnd.google-apps.folder','parents':['root']})
        if folder.get('mimeType')!='application/vnd.google-apps.folder':raise PublicationConflict('Folder identity differs')
        immutable_json(path,dict(id=folder['id'],name='ChatGPT',driveId=None,private=True))
        return folder['id']

    def _check_folder(self,file_id):
        doc=self._request('get',f'/files/{file_id}',params={'fields':'id,name,mimeType,trashed,driveId,capabilities(canAddChildren)'})
        if (doc.get('id')!=file_id or doc.get('mimeType')!='application/vnd.google-apps.folder'
                or doc.get('trashed') or not doc.get('capabilities',{}).get('canAddChildren')):
            raise PublicationPending('Detail folder is not a verified writable native Drive folder')
        source_acl={(p['type'],p['emailAddress'].lower()) for p in _acl(self._permissions(SPREADSHEET_ID))}
        for permission in self._permissions(file_id):
            if permission.get('deleted') or permission.get('role')=='owner':continue
            key=(permission.get('type'),permission.get('emailAddress','').lower())
            if permission.get('type') not in ('user','group') or key not in source_acl:
                raise PublicationPending('Detail folder would inherit access beyond named central principals')
        credentials=getattr(self._client().http_client,'auth',None)
        if getattr(credentials,'service_account_email',None) and not getattr(credentials,'_subject',None) and not doc.get('driveId'):
            raise PublicationPending('Service accounts need Shared Drive storage; an ordinary shared My Drive folder does not supply quota')
        return doc

    def ensure(self,key,title):
        """key binds one immutable remote destination, including rollover part."""
        preallocated=self._preallocated(key,title)
        if key in self._ready:
            info,adapter=self._ready[key]
            if info['title']!=title:raise PublicationConflict('Cached destination title differs')
            if preallocated is not None and info.get('preallocated_entry_sha256')!=digest(preallocated):
                raise PublicationConflict('Cached preallocated destination registry changed')
            return info,adapter
        self.root.mkdir(parents=True,exist_ok=True)
        receipt=self.root/'destinations'/(key+'.json')
        if preallocated is not None:return self._ensure_preallocated(preallocated,key,title,receipt)
        if receipt.exists():
            saved=load_json(receipt)
            if saved['key']!=key or saved['title']!=title:raise PublicationConflict('Immutable destination registry conflict')
            self._verify_metadata(saved['spreadsheet_id'],key,title)
            result=saved,AnalysisSheet(self._client().open_by_key(saved['spreadsheet_id']),
                                       spreadsheet_id=saved['spreadsheet_id'],max_cells=DETAIL_CELL_LIMIT)
            self._ready[key]=result
            return result
        tags={'ta2_campaign':CAMPAIGN,'ta2_destination':key,'ta2_schema':'AAX_ANALYSIS152_v2'}
        folder_id=self._folder()
        query=f"trashed = false and '{folder_id}' in parents and appProperties has {{ key='ta2_destination' and value='{key}' }}"
        existing=self._list(query)
        if len(existing)>1:raise PublicationConflict('Duplicate tagged detail books; do not merge/delete/recreate automatically')
        creation=self.root/'created'/(key+'.json');intent=self.root/'creation_intents'/(key+'.json')
        if creation.exists():
            created=load_json(creation)
            if len(existing)!=1 or existing[0]['id']!=created['id']:
                raise PublicationPending('Created detail ID is not currently visible; no replacement file is created')
            metadata=existing[0]
        elif existing:
            metadata=existing[0]
            if metadata.get('appProperties')!=tags or metadata.get('name')!=title:
                raise PublicationConflict('Tagged detail destination does not match exact registered campaign/schema/title')
            immutable_json(creation,dict(id=metadata['id'],key=key,title=title,recovered_by='Drive_appProperties'))
        else:
            if intent.exists():
                raise PublicationPending('DETAIL_CREATION_UNRESOLVED: timeout/quota/permission result has no discovered ID; no blind repeated creation')
            acl=_acl(self._permissions(SPREADSHEET_ID))
            immutable_json(intent,dict(key=key,title=title,tags=tags,acl_sha256=digest(acl),requested=now()))
            # appProperties are attached in the same create request. A timeout
            # is reconciled by tagged discovery, never by another POST.
            metadata=self._request('post','/files',params={'fields':'id,name,mimeType,webViewLink,appProperties,ownedByMe'},
                json={'name':title,'mimeType':'application/vnd.google-apps.spreadsheet','parents':[folder_id],'appProperties':tags})
            immutable_json(creation,dict(id=metadata['id'],key=key,title=title,recovered_by='creation_response'))
        metadata=self._verify_metadata(metadata['id'],key,title)
        book=self._client().open_by_key(metadata['id'])
        self._initialize(book,key,title)
        acl=_acl(self._permissions(SPREADSHEET_ID))
        current=self._permissions(metadata['id'])
        if any(p.get('type') in ('anyone','domain') for p in current):
            raise PublicationPending('New detail destination unexpectedly has public/domain access; no permission removal is attempted')
        allowed={(p['type'],p['emailAddress'].lower()):p for p in acl}
        for permission in current:
            if permission.get('role')=='owner':continue
            k=(permission.get('type'),permission.get('emailAddress','').lower())
            if k not in allowed:raise PublicationConflict('New detail book has an unregistered named principal')
        for grant in acl:
            known=[p for p in current if p.get('type')==grant['type'] and p.get('emailAddress','').lower()==grant['emailAddress'].lower()]
            if known:
                effective='writer' if known[0]['role'] in ('organizer','fileOrganizer') else known[0]['role']
                if effective not in (grant['role'],'owner'):
                    raise PublicationPending('Existing detail role differs from frozen central ACL; no privilege overwrite')
                continue
            self._request('post',f"/files/{metadata['id']}/permissions",params={'sendNotificationEmail':'false','fields':'id'},json=grant)
        verified=self._permissions(metadata['id'])
        for grant in acl:
            if not any(p.get('type')==grant['type'] and p.get('emailAddress','').lower()==grant['emailAddress'].lower()
                       and (p.get('role') in (grant['role'],'owner') or
                            grant['role']=='writer' and p.get('role') in ('organizer','fileOrganizer')) for p in verified):
                raise PublicationPending('Detail named-principal permission readback incomplete')
        final=self._verify_metadata(metadata['id'],key,title)
        if not final.get('webViewLink'):raise PublicationPending('Drive metadata did not expose observed detail URL')
        saved=dict(key=key,title=title,spreadsheet_id=final['id'],url=final['webViewLink'],schema='AAX_ANALYSIS152_v2',
                   acl_source_sha256=digest(acl),visibility='named_central_principals_only; no_public_or_domain_grants',
                   policy=POLICY,creation_reconciled=True,headers_readback=True)
        immutable_json(receipt,saved)
        result=saved,AnalysisSheet(book,spreadsheet_id=final['id'],max_cells=DETAIL_CELL_LIMIT)
        self._ready[key]=result
        return result

    def _verify_metadata(self,file_id,key,title):
        doc=self._request('get',f'/files/{file_id}',params={'fields':'id,name,mimeType,trashed,appProperties,webViewLink,ownedByMe,driveId,parents'})
        folder=load_json(self.root/'folder.json')
        if (doc['id']!=file_id or doc.get('name')!=title or doc.get('trashed')
                or doc.get('mimeType')!='application/vnd.google-apps.spreadsheet'
                or doc.get('appProperties',{}).get('ta2_destination')!=key
                or doc.get('appProperties',{}).get('ta2_campaign')!=CAMPAIGN
                or folder['id'] not in doc.get('parents',[])
                or not (doc.get('ownedByMe') or folder.get('driveId') and doc.get('driveId')==folder.get('driveId'))
                or doc.get('appProperties',{}).get('ta2_schema')!='AAX_ANALYSIS152_v2'):
            raise PublicationConflict('Remote detail identity/source ownership changed')
        return doc

    def _initialize(self,book,key,title):
        metadata=book.fetch_sheet_metadata(params={'fields':'spreadsheetId,sheets(properties)'})
        if metadata.get('spreadsheetId')!=book.id:raise PublicationConflict('New book metadata identity mismatch')
        sheets=metadata.get('sheets',[])
        matching=[s for s in sheets if s['properties']['sheetId']==SHEET_ID]
        if not matching:
            if len(sheets)!=1:raise PublicationConflict('Unexpected sheets in not-yet-initialized detail destination')
            old=sheets[0]['properties'];old_id=old['sheetId'];name=old['title'].replace("'","''")
            grid=old['gridProperties']
            if grid['rowCount']*grid['columnCount']>50000:
                raise PublicationConflict('Initial detail sheet unexpectedly large; no broad cleanup')
            # This is our privately-created, empty default tab, not any source
            # workbook tab. Never delete it if a value has appeared meanwhile.
            from ta2.reporting import column_letter
            area=f"'{name}'!A1:{column_letter(grid['columnCount']-1)}{grid['rowCount']}"
            if book.values_get(area).get('values'):raise PublicationConflict('New detail workbook is not empty')
            book.batch_update({'requests':[
                {'addSheet':{'properties':{'sheetId':SHEET_ID,'title':SHEET,'gridProperties':{
                    'rowCount':1000,'columnCount':152,'frozenRowCount':7,'frozenColumnCount':3}}}},
                {'deleteSheet':{'sheetId':old_id}}]})
        else:
            if len(sheets)!=1 or matching[0]['properties']['title']!=SHEET:
                raise PublicationConflict('Detail destination unexpectedly contains other tabs')
        header=book.values_get("'analysis'!A7:EV7",params={'valueRenderOption':'UNFORMATTED_VALUE'}).get('values',[])
        if header and header!=[list(COLUMNS)]:raise PublicationConflict('Detail header is nonempty and differs; no overwrite')
        if not header:
            book.batch_update({'requests':[
                {'updateCells':{'range':{'sheetId':SHEET_ID,'startRowIndex':6,'endRowIndex':7,'startColumnIndex':0,'endColumnIndex':152},
                    'rows':[{'values':[{'userEnteredValue':{'stringValue':c}} for c in COLUMNS]}],'fields':'userEnteredValue'}},
                {'repeatCell':{'range':{'sheetId':SHEET_ID,'startRowIndex':6,'endRowIndex':7,'startColumnIndex':0,'endColumnIndex':152},
                    'cell':{'userEnteredFormat':{'backgroundColor':{'red':.94,'green':.94,'blue':.94},
                        'textFormat':{'bold':True},'wrapStrategy':'WRAP'}},'fields':'userEnteredFormat'}},
                {'updateDimensionProperties':{'range':{'sheetId':SHEET_ID,'dimension':'ROWS','startIndex':6,'endIndex':7},
                    'properties':{'pixelSize':56},'fields':'pixelSize'}},
                {'setBasicFilter':{'filter':{'range':{'sheetId':SHEET_ID,'startRowIndex':6,'endRowIndex':1000,'startColumnIndex':0,'endColumnIndex':152}}}}
            ]})
        AnalysisSheet(book,spreadsheet_id=book.id,max_cells=DETAIL_CELL_LIMIT).preflight(extra_rows=0)


class PublicationRouter:
    def __init__(self,lane,server,*,plan=None,factory=None,central=None):
        if server not in ('s1','s3','s5'):raise ValueError('Only registered s1/s3/s5 publication lanes')
        self.lane=Path(lane);self.server=server;self.root=self.lane/'publication';self.root.mkdir(parents=True,exist_ok=True)
        self.plan=plan or Plan();self.queue=self.plan.queue(server)
        self.runs={r['run_id']:r for r in self.queue}
        self.case_seq={}
        for row in self.queue:self.case_seq.setdefault(row['case_id'],row['queue_seq'])
        self.factory=factory or DriveDetailFactory(self.root/'books');self._central=central
        immutable_json(self.root/'policy.json',dict(policy=POLICY,server=server,central_spreadsheet_id=SPREADSHEET_ID,
            core_metrics=sorted(CORE_METRICS),paired_seed_metrics=sorted(PAIRED_SEED_METRICS),
            bucket_runs=BUCKET_RUNS,detail_cell_limit=DETAIL_CELL_LIMIT,
            queue_sha256=digest(self.queue),central_capacity_no_fallback=True))

    @staticmethod
    def priority(record):return 0 if central_record(record) else 1

    def _central_adapter(self):
        if self._central is None:self._central=self.factory.central()
        return self._central

    def bucket(self,record):
        if record['Server']!=self.server:raise PublicationConflict('Cross-server record in local publication lane')
        row=self.runs.get(record['Run_ID'])
        seq=row['queue_seq'] if row else self.case_seq.get(record['Case_ID'])
        if seq is None and record['Record_type'] in ('ASSET','UNIT'):return 0
        if seq is None:raise PublicationConflict('Record has no registered case/run for deterministic detail shard')
        return (seq-1)//BUCKET_RUNS+1

    def _detail(self,bucket,part,record):
        key=f'TA2_{self.server}_B{bucket:02d}_P{part:02d}'
        title=f'PAN TA2 DETAIL {self.server.upper()} B{bucket:02d} P{part:02d} 2026-10-06'
        info,adapter=self.factory.ensure(key,title)
        if info['spreadsheet_id']==SPREADSHEET_ID:raise PublicationConflict('Detail destination unexpectedly aliases central workbook')
        path=self.root/'detail_manifests'/(key+'.json')
        if not path.exists():
            ctx=dict(record,Run_ID=key,Checkpoint_SHA='',Replica=None,Train_seed=None,Step=None,Selection='')
            manifest=make_record(ctx,'ASSET','detail_destination/'+key,
                Metric_name='detail_manifest',Estimate=bucket,Status='DESTINATION_VERIFIED',
                Asset_manifest_URI=info['url'],Raw_results_URI=info['url'],
                Reason=f'{POLICY}; shard part{part}; detailed numeric records, not independent training seeds')
            receipt=self._central_adapter().publish(manifest)
            immutable_json(path,dict(destination=info,central_manifest_receipt=receipt))
        return key,info,adapter

    def publish(self,record):
        record=validate_record(record)
        if record['Server']!=self.server:raise PublicationConflict('Wrong server publication record')
        with (self.root/'.router.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if central_record(record):
                return self._central_adapter().publish(record)
            bucket=self.bucket(record);tag=digest(record['Record_ID'])
            history=self.root/'record_routes'/tag
            attempts=sorted(history.glob('attempt_*.json')) if history.exists() else []
            if attempts:
                active=load_json(attempts[-1]);part=active['part'];generation=len(attempts)-1
                if active['record_id']!=record['Record_ID'] or active['scientific_sha256']!=digest(scientific_payload(record)):
                    raise PublicationConflict('Immutable routing identity differs')
                if (history/f'capacity_{generation:03d}.json').exists():part+=1;generation+=1
            else:
                state=self.root/'buckets'/f'{bucket:02d}.json'
                part=load_json(state)['part'] if state.exists() else 1;generation=0
            while True:
                key,info,adapter=self._detail(bucket,part,record)
                attempt=dict(record_id=record['Record_ID'],scientific_sha256=digest(scientific_payload(record)),
                             bucket=bucket,part=part,destination_key=key,spreadsheet_id=info['spreadsheet_id'],policy=POLICY)
                immutable_json(history/f'attempt_{generation:03d}.json',attempt)
                try:receipt=adapter.publish(record)
                except CapacityBlocked as exc:
                    if not getattr(exc,'safe_to_rollover',False):raise
                    immutable_json(history/f'capacity_{generation:03d}.json',dict(no_append_confirmed=True,destination_key=key))
                    part+=1;generation+=1
                    atomic_json(self.root/'buckets'/f'{bucket:02d}.json',dict(part=part,policy=POLICY))
                    continue
                if receipt.get('spreadsheet_id')!=info['spreadsheet_id']:
                    raise PublicationConflict('Published receipt destination differs from immutable router intent')
                return dict(receipt,publication_policy=POLICY,destination_key=key,central=False)

    @staticmethod
    def _send_many(adapter,records):
        receipts=(adapter.publish_many(records) if hasattr(adapter,'publish_many')
                  else [adapter.publish(record) for record in records])
        if len(receipts)!=len(records):raise PublicationConflict('Batch publisher omitted a delivery receipt')
        for record,receipt in zip(records,receipts):
            if (receipt.get('record_id')!=record['Record_ID']
                    or receipt.get('scientific_sha256')!=digest(scientific_payload(record))
                    or receipt.get('status') not in ('VALUES_READBACK_VERIFIED','ROUTED_READBACK_VERIFIED')):
                raise PublicationConflict('Batch delivery receipt identity/status differs')
        return receipts

    def publish_many(self,records):
        """Group up to100 records by immutable destination, never by outcomes.

        A group failure retains the entire local batch; earlier remote groups
        are recovered by exact IDs on retry. Only IDs proven absent can roll.
        """
        records=[validate_record(record) for record in records]
        if len(records)>100:raise ValueError('Router publication batch is bounded to100')
        if not records:return []
        if len(records)==1:return [self.publish(records[0])]
        if len({r['Record_ID'] for r in records})!=len(records):
            raise PublicationConflict('Duplicate record IDs in routed batch')
        if any(r['Server']!=self.server for r in records):raise PublicationConflict('Cross-server batch in local lane')
        with (self.root/'.router.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            delivered={}
            central=[r for r in records if central_record(r)]
            for receipt in self._send_many(self._central_adapter(),central) if central else []:
                delivered[receipt['record_id']]=receipt
            groups={}
            for record in records:
                if central_record(record):continue
                bucket=self.bucket(record);history=self.root/'record_routes'/digest(record['Record_ID'])
                attempts=sorted(history.glob('attempt_*.json')) if history.exists() else []
                if attempts:
                    active=load_json(attempts[-1]);part=active['part'];generation=len(attempts)-1
                    if (active['record_id']!=record['Record_ID']
                            or active['scientific_sha256']!=digest(scientific_payload(record))):
                        raise PublicationConflict('Immutable batch route identity differs')
                    if (history/f'capacity_{generation:03d}.json').exists():part+=1;generation+=1
                else:
                    state=self.root/'buckets'/f'{bucket:02d}.json'
                    part=load_json(state)['part'] if state.exists() else 1;generation=0
                groups.setdefault((bucket,part),[]).append((record,history,generation))
            pending=list(groups.items())
            while pending:
                (bucket,part),items=pending.pop(0)
                key,info,adapter=self._detail(bucket,part,items[0][0])
                if not hasattr(adapter,'publish_many') and len(items)>1:
                    # Injected legacy adapters provide no all-or-none capacity
                    # proof, so never assume a failed multi-row loop is unwritten.
                    pending[:0]=[((bucket,part),[item]) for item in items]
                    continue
                for record,history,generation in items:
                    attempt=dict(record_id=record['Record_ID'],scientific_sha256=digest(scientific_payload(record)),
                        bucket=bucket,part=part,destination_key=key,spreadsheet_id=info['spreadsheet_id'],policy=POLICY)
                    immutable_json(history/f'attempt_{generation:03d}.json',attempt)
                try:
                    receipts=self._send_many(adapter,[item[0] for item in items])
                except CapacityBlocked as exc:
                    if not getattr(exc,'safe_to_rollover',False):raise
                    absent=getattr(exc,'missing_record_ids',None)
                    if absent is None:
                        if len(items)!=1:raise PublicationConflict('Batch capacity failure did not prove which IDs are unwritten') from exc
                        absent=[items[0][0]['Record_ID']]
                    absent=set(absent);ids={item[0]['Record_ID'] for item in items}
                    if not absent or not absent<=ids:raise PublicationConflict('Invalid absent-ID rollover proof') from exc
                    already=[item for item in items if item[0]['Record_ID'] not in absent]
                    moving=[item for item in items if item[0]['Record_ID'] in absent]
                    if already:pending.insert(0,((bucket,part),already))
                    moved=[]
                    for record,history,generation in moving:
                        immutable_json(history/f'capacity_{generation:03d}.json',
                            dict(no_append_confirmed=True,destination_key=key))
                        moved.append((record,history,generation+1))
                    pending.append(((bucket,part+1),moved))
                    state=self.root/'buckets'/f'{bucket:02d}.json'
                    previous=load_json(state)['part'] if state.exists() else 1
                    atomic_json(state,dict(part=max(previous,part+1),policy=POLICY))
                    continue
                for receipt in receipts:
                    if receipt.get('spreadsheet_id')!=info['spreadsheet_id']:
                        raise PublicationConflict('Batch delivery differs from immutable destination')
                    delivered[receipt['record_id']]=dict(receipt,publication_policy=POLICY,destination_key=key,central=False)
            return [delivered[record['Record_ID']] for record in records]


def connect_router(lane,server):
    """No network until its publish(record) method is explicitly invoked."""
    return PublicationRouter(lane,server)
