"""WV3 main: publish Stage2 passes plus one representative, with donor evidence."""
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from directshift48.reporting import (HEADERS,INDEX,METRICS,serialize_row,atomic_json,
                                    canonical_json,utc_now)
from sheet_publish_v2.transport import _typed,_matrix_equal
from ta2.evaluation import file_sha
from . import plan
from .evaluation import verify_record

POLICY='WV3_MAIN_RGB_PASSES_AND_REPRESENTATIVE_V1'
R=SimpleNamespace(atomic_json=atomic_json,utc_now=utc_now)


def destination():
    return {'spreadsheet_id':'1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE',
            'sheet_title':'WV3-main','sheet_id':260928010}


@lru_cache(maxsize=4096)
def _verified_file(path,inode,size,mtime):
    return file_sha(path)


def artifact(meta):
    p=plan.local_path(meta['path']);s=p.stat()
    if _verified_file(str(p),s.st_ino,s.st_size,s.st_mtime_ns)!=meta['file_sha256']:
        raise ValueError('Published checkpoint file is missing or changed: '+str(p))


def choose(records,stage):
    if stage=='stage1' or not records:return []
    if stage!='stage2':raise ValueError('Unregistered publication stage')
    by_step={}
    for e in records:
        if e['step'] in by_step and by_step[e['step']]!=e:
            raise ValueError('Conflicting evaluations for the same checkpoint step')
        by_step[e['step']]=e
    records=[by_step[k] for k in sorted(by_step)]
    qualified=[e for e in records if e['fr']['metrics']['HQNR']>=plan.HQNR_MIN]
    rank=lambda e:(e['rr']['metrics']['ERGAS'],-e['fr']['metrics']['HQNR'],e['step'])
    rep=(min(qualified,key=rank) if qualified else
         min(records,key=lambda e:(-e['fr']['metrics']['HQNR'],e['rr']['metrics']['ERGAS'],e['step'])))
    selected=[(rep,'HQNR_QUALIFIED_MIN_ERGAS' if qualified else 'BEST_HQNR_FALLBACK','REPRESENTATIVE')]
    return selected+[(e,'PASS_ALL_STEPS',str(e['step'])) for e in records
        if e is not rep and e['fr']['metrics']['HQNR']>=plan.HQNR_MIN and e['rr']['metrics']['ERGAS']<=plan.ERGAS_MAX]


def approach(c):
    return (f"RGB full Stage1; S2 PANc+delta+MS; scale=1; "
            f"alpha={c['alpha']:g},gamma={c['gamma']:g},beta=0; "
            f"W112 D123 UBIL; L4/Tbic; lr={c['lr']:g} C{c['horizon']/1000:g}")


def donor_note(cfg,record):
    receipt=record.get('donor_selection')
    if not isinstance(receipt,dict) or not receipt:
        raise ValueError('Measured main result lacks its Stage1 donor selection evidence')
    if receipt.get('donor_run_id')!=cfg['donor_run_id']:
        raise ValueError('Measured main result uses another Stage1 donor run')
    if receipt.get('schema')=='WV3_MAIN_RGB_DONOR_SELECTION_V1':
        if receipt.get('receipt_sha256')!=plan.digest({k:v for k,v in receipt.items() if k!='receipt_sha256'}):
            raise ValueError('Stage1 donor selection receipt digest mismatch')
    # All 50 candidate records remain in the immutable evaluation JSON. Their
    # repetition in a single Sheet cell can exceed Google's 50k character limit.
    return {**{k:v for k,v in receipt.items() if k!='candidates'},
            'candidate_count':len(receipt.get('candidates',[])),
            'full_receipt_in_evaluation':True}


def make_row(cfg,status,record=None,selector='STATUS',slot='STATUS'):
    if cfg['stage']!='stage2':raise ValueError('Only Stage2 results belong in WV3-main')
    step=0 if record is None else record['step']
    # Reuse the queued row for its eventual representative rather than blanking it.
    id_slot='REPRESENTATIVE' if slot=='STATUS' else slot
    rid='WMAINRGB-'+plan.digest([plan.CAMPAIGN_ID,cfg['server'],cfg['run_id'],id_slot])
    passed=bool(record and record['fr']['metrics']['HQNR']>=plan.HQNR_MIN and record['rr']['metrics']['ERGAS']<=plan.ERGAS_MAX)
    label=('대표 · HQNR 최고 · HQNR 미달'
           if selector=='BEST_HQNR_FALLBACK' else '대표 · ERGAS 최소' if selector=='HQNR_QUALIFIED_MIN_ERGAS'
           else '조건 충족' if selector=='PASS_ALL_STEPS' else status)
    if selector=='HQNR_QUALIFIED_MIN_ERGAS':label+=' · 조건 충족' if passed else ' · ERGAS 미달'
    note={'schema_version':'PAN_FIXED64_V1','reset_epoch':'2026-10-13','configuration':cfg,
          'publication':{'policy':POLICY,'selector':selector,'slot':slot,'joint_pass':passed,
              'is_case_representative':slot=='REPRESENTATIVE','targets':plan.PLAN['targets']},
          'fixed_step_comparison':cfg.get('primary_step',cfg['horizon']),'reference_repeats':3,
          'evaluation_file':None if record is None else record['evidence_path'],
          'evaluation_sha256':None if record is None else record['evidence_sha256'],
          'model_state_sha256':None if record is None else record['model_sha256'],
          'artifact':None if record is None else record['checkpoint']}
    if record:
        note['donor_selection']=donor_note(cfg,record)
        if 'test_rre_rgb' in record:note['test_rre_rgb']=record['test_rre_rgb']
    else:
        note['donor_selection']={'run_id':cfg['donor_run_id'],'status':'PENDING_LOCAL_STAGE1_SELECTION'}
    note['reference_definition']={'L_spa':'GT RGB arithmetic mean [4,2,1]',
        'validation_selector':'RGB GT RRE50/RRE90 then ValERGAS then earlier step',
        'full_reference':'fresh local full Stage1 per repeat; frozen Aligner and Teacher',
        'student_input':'corrected PAN, corrected PAN minus original PAN, interp23 I_lrms'}
    fields={'Experiment':cfg['experiment'],'Server':cfg['server'],'Selection@Step':f'{label} · {step:,}',
        'Dataset':'WV3','Bucket':'main','Approach':approach(cfg),'Campaign_ID':plan.CAMPAIGN_ID,
        'Run_ID':cfg['run_id'],'Case_ID':cfg['case_id'],'Repeat':cfg['repeat'],'Seed':cfg['seed'],
        'Attempt':1,'Role':'Student+FrozenAligner',
        'Model / inputs':cfg['architecture']+' / '+cfg.get('student_input','PAN+MS'),
        'Teacher / reference':cfg['donor_run_id'], 'Updates':step,'Lifetime_updates':step,
        'Selected_step':step,'Selector':selector,'Status':status,'Readback':'VERIFIED',
        'Eval_scope':'OFFICIAL_FULL20_NATIVE' if record else 'STATUS_ONLY',
        'Test_aware':bool(record),'Result_ID':rid,
        'U_peak_lr':cfg['lr'],'A_peak_lr':cfg['lr_aligner'],
        'alpha':cfg['alpha'],'beta':0,
        'lambda_E':cfg.get('lambda_epsilon'),'Original description / notes':canonical_json(note)}
    if record:
        for split in ('rr','fr'):
            fields.update({METRICS[k]:v for k,v in record[split]['metrics'].items()})
        fields.update(Checkpoint_SHA=record['checkpoint']['file_sha256'],
            Date=record['recorded_at'],Eval_scope='OFFICIAL_FULL20_NATIVE',
            JQM_variant=record['jqm_variant'])
        fields['Params(M)']=record.get('inference_parameters',0)/1e6 if 'inference_parameters' in record else None
        fields['Eval(h)']=record['elapsed_seconds']/3600
    return serialize_row(fields),canonical_json(note)


def project(root,server):
    root=Path(root);states={}
    path=root/'campaign_state.json'
    if path.exists():states={r['run_id']:r for r in json.loads(path.read_text())['runs']}
    rows,notes,proof=[],[],{}
    configs={c['run_id']:c for c in plan.queue(server) if c['stage']=='stage2'}
    for cfg in configs.values():
        directory=root/'runs'/cfg['run_id'];records=[]
        for p in sorted(directory.glob('evaluation_*.json')):
            e=json.loads(p.read_text());verify_record(e,cfg);artifact(e['checkpoint'])
            if e['checkpoint']['model_sha256']!=e['model_sha256'] or e['checkpoint']['step']!=e['step']:
                raise ValueError('Evaluation does not match its retained checkpoint')
            e['evidence_path']=str(p);e['evidence_sha256']=file_sha(p);records.append(e)
            proof[(cfg['run_id'],e['step'])]=e
        state=states.get(cfg['run_id'],{'status':'QUEUED'})
        selected=choose(records,cfg['stage'])
        if not selected:
            row,note=make_row(cfg,state['status']);rows.append(row);notes.append(note)
        for e,selector,slot in selected:
            row,note=make_row(cfg,state['status'],e,selector,slot);rows.append(row);notes.append(note)
    return {'policy':POLICY,'destination':destination(),'rows':rows,'notes':notes,'conflicts':[],
            'configs':configs,'proof':proof,'view_hash':plan.digest(rows)}


def verify_owned(row,value,server):
    cfg=value['configs'].get(row[INDEX['Run_ID']])
    if cfg is None or cfg['server']!=server or row[INDEX['Dataset']]!='WV3':
        raise ValueError('Unrecognized row in this campaign; never erase it')
    note=json.loads(row[INDEX['Original description / notes']]);pub=note['publication']
    if note['configuration']!=cfg or pub['policy']!=POLICY:
        raise ValueError('Owned row has modified configuration/provenance')
    step=row[INDEX['Selected_step']]
    e=value['proof'].get((cfg['run_id'],step)) if step else None
    if step and e is None:raise ValueError('Owned row lacks immutable evaluation evidence')
    expected,expected_note=make_row(cfg,row[INDEX['Status']],e,pub['selector'],pub['slot'])
    if note.get('donor_selection')!=json.loads(expected_note)['donor_selection']:
        raise ValueError('Owned row Stage1 donor evidence mismatch')
    # Old selectors may no longer be optimal, but their original metrics and identity must be valid.
    for key in ('Result_ID','Experiment','Case_ID','Repeat','Seed','Selected_step','Checkpoint_SHA',*METRICS.values()):
        if row[INDEX[key]]!=expected[INDEX[key]]:raise ValueError('Owned row evidence mismatch: '+key)


def row_requests(snapshot,value,server):
    own={}
    for number,row in enumerate(snapshot['rows'],5):
        if row[INDEX['Campaign_ID']]!=plan.CAMPAIGN_ID or row[INDEX['Server']]!=server:continue
        verify_owned(row,value,server);rid=row[INDEX['Result_ID']]
        if rid in own:raise ValueError('Duplicate owned Result_ID')
        own[rid]=(number,row,snapshot['notes'][number-5])
    wanted={r[INDEX['Result_ID']]:(r,n) for r,n in zip(value['rows'],value['notes'])}
    if len(wanted)!=len(value['rows']):raise ValueError('Duplicate desired Result_ID')
    def cells(row,note):
        return {'values':[{'userEnteredValue':_typed(v),'note':note if i==0 else ''} for i,v in enumerate(row)]}
    def update(number,payload):
        return {'updateCells':{'range':{'sheetId':destination()['sheet_id'],'startRowIndex':number,
            'endRowIndex':number+1,'startColumnIndex':0,'endColumnIndex':64},'rows':[payload],
            'fields':'userEnteredValue,note'}}
    requests,additions=[],[]
    for rid,(row,note) in wanted.items():
        if rid not in own:additions.append(cells(row,note));continue
        number,old,old_note=own[rid]
        if not _matrix_equal([old],[row]) or old_note!=note:requests.append(update(number,cells(row,note)))
    for rid,(number,_,_) in own.items():
        if rid not in wanted:requests.append(update(number,{'values':[{} for _ in range(64)]}))
    if additions:requests.append({'appendCells':{'sheetId':destination()['sheet_id'],'rows':additions,'fields':'userEnteredValue,note'}})
    return requests


def publish(root,server,*,apply=False,credential=None):
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import reconcile_rows
    root=Path(root);adapter=GoogleSheets(credential or plan.ROOT/'gspread/account.json')
    value=project(root,server)
    receipt=reconcile_rows(adapter,value,root/'publication',server,apply=apply)
    atomic_json(root/'publication/last_poll.json',receipt)
    return receipt
