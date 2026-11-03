"""Fixed64 own-row publication: Stage1 Val-RRE; Stage2 all passes + representative."""
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

POLICY='WV3_RGB_ABLATION_VAL_RRE_REPRESENTATIVE_V1'
R=SimpleNamespace(atomic_json=atomic_json,utc_now=utc_now)


def destination():
    return {'spreadsheet_id':'1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE',
            'sheet_title':'WV3-ablations','sheet_id':261001010}


@lru_cache(maxsize=4096)
def _verified_file(path,inode,size,mtime):
    return file_sha(path)


def artifact(meta):
    p=plan.local_path(meta['path']);s=p.stat()
    if _verified_file(str(p),s.st_ino,s.st_size,s.st_mtime_ns)!=meta['file_sha256']:
        raise ValueError('Published checkpoint file is missing or changed: '+str(p))


def choose(records,stage):
    if not records:return []
    if stage=='stage1':
        valid=[e for e in records if 'validation' in e]
        if not valid:return []
        row=min(valid,key=lambda e:(e['validation']['RRE50'],e['validation']['RRE90'],e['validation']['Val_ERGAS'],e['step']))
        return [(row,'BEST_VAL_RRE','REPRESENTATIVE')]
    qualified=[e for e in records if e['fr']['metrics']['HQNR']>=plan.HQNR_MIN]
    rank=lambda e:(e['rr']['metrics']['ERGAS'],-e['fr']['metrics']['HQNR'],e['step'])
    rep=(min(qualified,key=rank) if qualified else
         min(records,key=lambda e:(-e['fr']['metrics']['HQNR'],e['rr']['metrics']['ERGAS'],e['step'])))
    selected=[(rep,'HQNR_QUALIFIED_MIN_ERGAS' if qualified else 'BEST_HQNR_FALLBACK','REPRESENTATIVE')]
    return selected+[(e,'PASS_ALL_STEPS',str(e['step'])) for e in records
        if e is not rep and e['fr']['metrics']['HQNR']>=plan.HQNR_MIN and e['rr']['metrics']['ERGAS']<=plan.ERGAS_MAX]


def approach(c):
    if c['stage']=='stage1':
        return (f"S1 {c['alignment']}; rec+{c['lambda_abs']:g}spa+{c['lambda_epsilon']:g}eps; "
                f"RGB mean/bicubic; ValRRE RGB; A/Tlr={c['lr_aligner']:g}/{c['lr']:g}; W112 BASE C50")
    return (f"S2 {c['student_input']}; a={c['alpha']:g},g={c['gamma']:g},b=0; "
            f"{c['weight']}; S100 W112 UBIL; L4/Tbic; lr=1e-4 C30")


def make_row(cfg,status,record=None,selector='STATUS',slot='STATUS'):
    step=0 if record is None else record['step']
    rid='WABLRGB-'+plan.digest([plan.CAMPAIGN_ID,cfg['server'],cfg['run_id'],slot])
    passed=bool(record and record['fr']['metrics']['HQNR']>=plan.HQNR_MIN and record['rr']['metrics']['ERGAS']<=plan.ERGAS_MAX)
    label=('대표 · Val-RRE' if selector=='BEST_VAL_RRE' else '대표 · HQNR 최고 · HQNR 미달'
           if selector=='BEST_HQNR_FALLBACK' else '대표 · ERGAS 최소' if selector=='HQNR_QUALIFIED_MIN_ERGAS'
           else '조건 충족' if selector=='PASS_ALL_STEPS' else status)
    if selector=='HQNR_QUALIFIED_MIN_ERGAS':label+=' · 조건 충족' if passed else ' · ERGAS 미달'
    note={'schema_version':'PAN_FIXED64_V1','reset_epoch':'2026-10-13','configuration':cfg,
          'publication':{'policy':POLICY,'selector':selector,'slot':slot,'joint_pass':passed,
              'is_case_representative':slot=='REPRESENTATIVE','targets':plan.PLAN['targets']},
          'fixed_step_comparison':cfg['primary_step'],'reference_repeats':3,
          'evaluation_file':None if record is None else record['evidence_path'],
          'evaluation_sha256':None if record is None else record['evidence_sha256'],
          'model_state_sha256':None if record is None else record['model_sha256'],
          'artifact':None if record is None else record['checkpoint']}
    if record and 'validation' in record:
        note['validation']={k:v for k,v in record['validation'].items() if not isinstance(v,(dict,list))}
    if record:
        note['test_rre_rgb']=record['test_rre_rgb']
    note['reference_definition']={'L_spa':'GT RGB arithmetic mean [4,2,1]',
        'validation_selector':'RGB GT RRE50/RRE90 then ValERGAS then earlier step',
        'Scharr_baseline':'stored LMS RGB mean [4,2,1]',
        'full_reference':'fresh S1_A5_ALL for each repeat; no historical mean8 Ours reuse'}
    fields={'Experiment':cfg['experiment'],'Server':cfg['server'],'Selection@Step':f'{label} · {step:,}',
        'Dataset':'WV3','Bucket':'ablations','Approach':approach(cfg),'Campaign_ID':plan.CAMPAIGN_ID,
        'Run_ID':cfg['run_id'],'Case_ID':cfg['case_id'],'Repeat':cfg['repeat'],'Seed':cfg['seed'],
        'Attempt':1,'Role':'Stage1 reconstruction' if cfg['stage']=='stage1' else 'Student+FrozenAligner',
        'Model / inputs':cfg['architecture']+' / '+cfg.get('student_input','PAN+MS'),
        'Teacher / reference':cfg.get('donor') or '', 'Updates':step,'Lifetime_updates':step,
        'Selected_step':step,'Selector':selector,'Status':status,'Readback':'VERIFIED',
        'Eval_scope':'OFFICIAL_FULL20_NATIVE' if record else 'STATUS_ONLY',
        'Test_aware':bool(record),'Result_ID':rid,
        'U_peak_lr':cfg['lr'],'A_peak_lr':cfg['lr_aligner'],
        'alpha':cfg.get('alpha'),'beta':0 if cfg['stage']=='stage2' else None,
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
    configs={c['run_id']:c for c in plan.queue(server)}
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
    expected,_=make_row(cfg,row[INDEX['Status']],e,pub['selector'],pub['slot'])
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
