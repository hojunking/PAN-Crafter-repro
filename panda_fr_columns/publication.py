"""One EQ40 row per selected checkpoint, with both FR reference triplets."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import time
from directshift48.reporting import INDEX
from sheet_publish_v2.transport import atomic_json, _typed, _matrix_equal, _cell_value
from .adapter import GoogleSheets
from .schema import BOOK

PACKAGES = ('panda_wv3_s2_eq40','panda_sensor_eq40')


def cell_equal(actual, expected):
    # Sheets omits userEnteredValue for a literal empty string. Compare its
    # semantic value, while retaining exact note/evidence checks.
    return (_matrix_equal([[_cell_value(actual)]],[[_cell_value(expected)]])
            and actual.get('note','')==expected.get('note',''))


def original_rows(value):
    pairs=[(row,note) for row,note in zip(value['rows'],value['notes'])
           if json.loads(note)['publication']['slot'].endswith('_ORIGINAL')]
    rows=[r for r,_ in pairs]; notes=[n for _,n in pairs]
    return dict(value,rows=rows,notes=notes)


def without_companions(snapshot,value,server,pub):
    """Hide, rather than destroy, already-posted corrected companion evidence.

    Row indices remain exact: placeholders only affect the logical ownership
    check. Actual Sheet cells are never cleared or moved.
    """
    rows=list(snapshot['rows'])
    for i,row in enumerate(rows):
        if row[INDEX['Campaign_ID']]!=pub.plan.CAMPAIGN_ID or row[INDEX['Server']]!=server:continue
        pub.verify_owned(row,value,server)
        note=json.loads(row[INDEX['Original description / notes']])
        if note['publication']['slot'].endswith('_CORRECTED'):rows[i]=['']*64
    return dict(snapshot,rows=rows)


def extras(value):
    output={}
    for row in value['rows']:
        rid=row[INDEX['Result_ID']];step=row[INDEX['Selected_step']]
        record=value['proof'].get((row[INDEX['Run_ID']],step)) if step else None
        numbers=['']*3; note='No evaluated checkpoint yet; corrected FR values intentionally blank.'
        if record:
            metrics=record['fr_corrected_pan']['metrics']
            numbers=[round(metrics[k],4) for k in ('HQNR','D_s','D_lambda')]
            if metrics['D_lambda']!=record['fr']['metrics']['D_lambda']:
                raise ValueError('Same reconstruction/LMS must have identical D_lambda')
            note=json.dumps({'schema':'PAN_FR67_V1','step':step,'run_id':row[INDEX['Run_ID']],
                'checkpoint_sha256':row[INDEX['Checkpoint_SHA']],
                'evaluation_file':record['evidence_path'],'evaluation_sha256':record['evidence_sha256'],
                'raw_metrics':metrics,'reference':'CORRECTED_PAN','official_selection_reference':'ORIGINAL_PAN'},
                ensure_ascii=False,sort_keys=True)
        output[rid]=[{'userEnteredValue':_typed(v),'note':note if j==0 else ''} for j,v in enumerate(numbers)]
    return output


def publish(package,root,server,*,apply=False,credential=None):
    if package not in PACKAGES:raise ValueError('Unknown dual-FR publisher')
    pub=importlib.import_module(package+'.publication')
    transport=importlib.import_module(package+'.transport')
    value=original_rows(pub.project(root,server));value['view_hash']=pub.plan.digest(value['rows'])
    root=Path(root);state=root/'publication/fr_columns'
    adapter=GoogleSheets(credential or pub.plan.ROOT/'gspread/account.json')
    # Startup performs migration. Polling never inserts columns or changes a
    # foreign tab: this also makes check-sheet strictly read-only.
    snap=adapter.snapshot(value['destination'])
    if snap['width']!=67:
        if apply:raise ValueError('FR67 missing; start via panda_fr_columns launcher to migrate once')
        return {'action':'SCHEMA_MIGRATION_REQUIRED','request_count':0,'readback_verified':False,
                'destination':value['destination'],'row_count':len(value['rows'])}
    adapter.extra_cells=extras(value)
    adapter.reuse_once=snap
    original=transport.row_requests
    transport.row_requests=lambda s,v,sv: pub.row_requests(without_companions(s,v,sv,pub),v,sv)
    try:
        receipt=transport.reconcile_rows(adapter,value,state,server,apply=apply)
    finally:transport.row_requests=original
    if not apply:return receipt
    fresh=adapter.latest_snapshot;requests=[];legacy=[];positions={}
    sid=value['destination']['sheet_id']
    for i,row in enumerate(fresh['rows']):
        rid=row[INDEX['Result_ID']]
        if rid in adapter.extra_cells:
            if rid in positions:raise ValueError('Duplicate desired row ID')
            positions[rid]=i+6
            wanted=adapter.extra_cells[rid]
            got=(fresh['physical']['cells'][i]+[{}]*67)[7:10]
            if any(not cell_equal(g,w) for g,w in zip(got,wanted)):
                requests.append({'updateCells':{'range':{'sheetId':sid,'startRowIndex':i+5,'endRowIndex':i+6,
                    'startColumnIndex':7,'endColumnIndex':10},'rows':[{'values':wanted}], 'fields':'userEnteredValue,note'}})
        elif row[INDEX['Campaign_ID']]==pub.plan.CAMPAIGN_ID and row[INDEX['Server']]==server:
            pub.verify_owned(row,value,server)
            if not json.loads(row[INDEX['Original description / notes']])['publication']['slot'].endswith('_CORRECTED'):
                raise ValueError('Unexpected retired original row')
            legacy.append(i+6)
    if set(positions)!=set(adapter.extra_cells):raise ValueError('Original rows missing after reconcile')
    for number in legacy:
        if not fresh['hidden_rows'].get(str(number)):
            requests.append({'updateDimensionProperties':{'range':{'sheetId':sid,'dimension':'ROWS',
                'startIndex':number-1,'endIndex':number},'properties':{'hiddenByUser':True},'fields':'hiddenByUser'}})
    for number in positions.values():
        if fresh['hidden_rows'].get(str(number)):
            requests.append({'updateDimensionProperties':{'range':{'sheetId':sid,'dimension':'ROWS',
                'startIndex':number-1,'endIndex':number},'properties':{'hiddenByUser':False},'fields':'hiddenByUser'}})
        if any(c.get('userEnteredFormat',{}).get('numberFormat',{}).get('pattern')!='0.0000'
               for c in fresh['physical']['cells'][number-6][3:28]):
            requests.append({'repeatCell':{'range':{'sheetId':sid,'startRowIndex':number-1,'endRowIndex':number,
                 'startColumnIndex':3,'endColumnIndex':28},
                 'cell':{'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}},
                 'fields':'userEnteredFormat.numberFormat'}})
    if requests:
        stamp=str(time.time_ns());atomic_json(state/('dual_fr_before_'+stamp+'.json'),fresh)
        atomic_json(state/('dual_fr_intent_'+stamp+'.json'),{'requests':requests})
        adapter.physical_request('POST',adapter.BASE+BOOK+':batchUpdate',json={'requests':requests})
    after=adapter.snapshot(value['destination']) if requests else fresh
    if any(not after['hidden_rows'].get(str(n)) for n in legacy):raise ValueError('Legacy companion hide readback mismatch')
    for rid,number in positions.items():
        row=after['rows'][number-6]
        expected=next(r for r in value['rows'] if r[INDEX['Result_ID']]==rid)
        if not _matrix_equal([row],[expected]):raise ValueError('Checkpoint/row changed during corrected-FR publication')
        got=(after['physical']['cells'][number-6]+[{}]*67)[7:10]
        for a,b in zip(got,adapter.extra_cells[rid]):
            if not cell_equal(a,b):
                raise ValueError('Corrected FR readback mismatch')
    receipt.update(owned_columns='A:BO',corrected_fr_columns='H:J',original_fr_columns='D:F',
        dual_fr_same_checkpoint_verified=True,legacy_companion_rows_hidden=legacy,
        sheet_rows=sorted(positions.values()),schema='PAN_FR67_V1')
    atomic_json(root/'publication/last_poll.json',receipt)
    atomic_json(state/'last_verified.json',receipt)
    return receipt


def install():
    import sheet_publish_v2.transport as base
    base.GoogleSheets=GoogleSheets
    for package in PACKAGES:
        pub=importlib.import_module(package+'.publication')
        pub.publish=lambda root,server,_package=package,**kw:publish(_package,root,server,**kw)
