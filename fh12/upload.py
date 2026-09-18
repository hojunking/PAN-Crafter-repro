"""FH12 official-JSON-only Sheets upload. No training, inference or new tabs."""
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import sys

import yaml

from fh12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow
from fh12.plan import CAMPAIGN_ID, GRID_STEPS, build_config, case_for

SCHEMA = 'FH12_OFFICIAL_SELECTIONS_v1'
GIDS = dict(s1=994031662,s2=991648123,s3=284220763,s4=2026091404,s5=823586191)
TABS = {s:'WV3-'+('s3(5090)' if s=='s3' else s) for s in GIDS}
RR_LABELS = {'ERGAS↓':'ergas','SAM↓':'sam','PSNR↑':'psnr','SSIM↑':'ssim','SCC↑':'scc','Q8↑':'q8'}
FR_LABELS = {'HQNR(raw)↑':'hqnr','D_lambda↓':'d_lambda','D_s↓':'d_s'}


def legacy_constants():
    """Only credentials/header constants; never call the legacy model collector."""
    name='_fh12_sheet_constants'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'gspread/gspread_upload.py')
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def optional_json(path):
    return read_json(path) if Path(path).is_file() else {}


def selection_values(prefix,report):
    empty=report.get('selection_id')=='TARGET' and report.get('target_status')=='no_eligible'
    values={prefix+' step':'' if empty else report['step'],
            prefix+' checkpoint SHA256':'' if empty else report['checkpoint_sha256'],
            prefix+' official':True}
    for label,key in RR_LABELS.items(): values[prefix+' '+label]='' if empty else report['rr'][key]
    for label,key in FR_LABELS.items(): values[prefix+' '+label]='' if empty else report['fr'][key]
    return values


def validate_report(report,run,cfg,context,wd):
    if not report.get('official_complete') or report.get('run_id')!=run or report.get('campaign_id')!=CAMPAIGN_ID:
        raise ValueError('FH12 report is incomplete or belongs to another run/campaign')
    for key in ('config_sha256','source_identity'):
        if report.get(key)!=context.get(key): raise ValueError(f'FH12 report identity mismatch: {key}')
    if report['config_sha256']!=object_sha(cfg): raise ValueError('FH12 config/report mismatch')
    if report.get('selection_id')=='TARGET' and report.get('target_status')=='no_eligible':
        if report.get('n_eligible')!=0 or report.get('selection','missing') is not None:
            raise ValueError('Malformed FH12 no_eligible target')
        return
    step=int(report['step'])
    if step not in GRID_STEPS: raise ValueError('FH12 selected step outside fixed grid')
    identity=read_json(wd/'candidates'/str(step)/'identity.json')
    if (report.get('checkpoint_identity')!=identity or identity.get('update')!=step or
            identity.get('config_sha256')!=object_sha(cfg) or
            report.get('checkpoint_sha256')!=identity.get('model_sha256') or
            sha256(wd/'candidates'/str(step)/'model.safetensors')!=identity.get('model_sha256')):
        raise ValueError('FH12 report/checkpoint SHA mismatch')
    if not report.get('rr',{}).get('official_complete'):
        raise ValueError('FH12 selected RR is not official')
    for domain,keys in [('rr',RR_LABELS.values()),('fr',FR_LABELS.values())]:
        for key in keys:
            if not math.isfinite(float(report[domain][key])):
                raise ValueError(f'Nonfinite FH12 metric: {domain}.{key}')


def row_values(run,root=ROOT):
    root=Path(root); case=case_for(run); wd=root/'work_dir'/run
    cfg=yaml.safe_load((wd/'meta/config.resolved.yaml').read_text())
    if cfg!=build_config(case): raise ValueError('FH12 upload config differs from frozen registry')
    status=read_json(wd/'official/postrun_status.json')
    if not status.get('official_complete') or status.get('actual_updates')!=50000:
        raise ValueError('FH12 upload requires completed official exact50K evaluation')
    reports={name:read_json(wd/'official'/f'{name}.json') for name in
             ('raw_max','target_selection','exact50k','rr_val_selected','e_min_diag')}
    for report in reports.values(): validate_report(report,run,cfg,status,wd)
    raw,target,exact,val,emin=(reports[k] for k in ('raw_max','target_selection','exact50k','rr_val_selected','e_min_diag'))
    if exact['step']!=50000 or target.get('n_evaluated')!=50:
        raise ValueError('FH12 exact50K/full candidate certification incomplete')
    if any(r.get('data_sha256')!=raw.get('data_sha256') for r in reports.values()):
        raise ValueError('FH12 selections use different data identities')
    profile=read_json(wd/'official/profile.json')
    if profile.get('config_sha256')!=object_sha(cfg) or profile.get('source_identity')!=status['source_identity']:
        raise ValueError('FH12 profile config/source mismatch')
    camp=root/'work_dir/_fh12'/case.server_id
    data=read_json(camp/'dataset_manifest.json'); lp=read_json(camp/'lpan_manifest.json')
    if object_sha(data)!=raw.get('data_sha256'): raise ValueError('FH12 dataset identity differs from reports')
    refpath=camp/'references'/case.teacher_run_id/'reference_manifest.json'; ref=optional_json(refpath)
    if case.role=='S' and not ref: raise ValueError('FH12 Student reference manifest missing')
    teacher_sha=exact['checkpoint_sha256'] if case.role=='T' else ref.get('teacher_checkpoint_sha256','')
    if ref:
        if ref.get('teacher_run_id')!=case.teacher_run_id: raise ValueError('FH12 wrong local Teacher reference')
        teacher_sha=ref.get('teacher_checkpoint_sha256',ref.get('teacher_model_sha256',teacher_sha))
        if not teacher_sha: raise ValueError('FH12 Teacher reference checksum missing')
        if ref.get('lpan_manifest_sha256')!=sha256(camp/'lpan_manifest.json'):
            raise ValueError('FH12 Teacher calibration LP identity mismatch')
        if case.role=='T' and teacher_sha!=exact['checkpoint_sha256']:
            raise ValueError('FH12 calibrated Teacher is not this exact50K')
        if case.role=='S' and any(r['checkpoint_identity'].get('reference_sha256')!=object_sha(ref)
                for r in reports.values() if r.get('checkpoint_identity')):
            raise ValueError('FH12 Student results use a different Teacher calibration')
    start=optional_json(wd/'meta/training_start_manifest.json'); training=optional_json(wd/'meta/training_status.json')
    init=optional_json(wd/'init_manifest.json')
    values={'Run':run,'캠페인':'FH12','FH12 campaign':CAMPAIGN_ID,'FH12 run id':run,
        'FH12 schema':SCHEMA,'FH12 server':case.server_id,'FH12 Role':case.role,
        'FH12 teacher input':case.teacher_input_layout,'FH12 student input':case.input_layout if case.role=='S' else '',
        'FH12 input layout':case.input_layout,'FH12 W':case.width,'FH12 D':''.join(map(str,case.depth)),
        'FH12 teacher seed':case.teacher_seed,'FH12 student seed':case.student_seed or '',
        'FH12 teacher run':case.teacher_run_id,'FH12 teacher_ref_sha':teacher_sha,
        'FH12 tau_R':ref.get('tau_R',''),'FH12 q_ref':ref.get('q_ref',''),
        'FH12 reference SHA256':sha256(refpath) if ref else '',
        'FH12 LP recipe':lp['recipe']['id'],'FH12 LP phase':lp['recipe']['phase_id'],
        'FH12 LP manifest SHA256':sha256(camp/'lpan_manifest.json'),
        'FH12 dataset SHA256':object_sha(data),'FH12 init policy':cfg['fh12']['init_policy'],
        'FH12 init manifest SHA256':object_sha(init),
        'FH12 training/evaluator release':status['source_identity'].get('git_release',''),
        'FH12 source SHA256':status['source_identity'].get('content_sha256',''),
        'FH12 actual updates':50000,'FH12 started UTC':start.get('started_at_utc',''),
        'FH12 deadline UTC':start.get('window',{}).get('deadline_utc',''),
        'FH12 completed UTC':status.get('completed_at_utc',''),
        'FH12 calibration status':'READY' if ref else 'PENDING_LOCAL_CALIBRATION',
        'FH12 inference scope':profile.get('scope',''),'FH12 FLOPs convention':profile.get('flops_convention',''),
        'FH12 FLOPs estimated':profile.get('flops_is_estimate',True),
        'Params(M)':profile['params_m'],'FLOPs(G)':profile['flops_g'],'Infer(ms)':profile['infer_ms'],
        'Mem(MB)':profile.get('mem_mb'),'Train(h)':float(training.get('training_seconds',0))/3600,
        'Date':start.get('started_at_utc','')[:10],
        'Notes':('FH12; main RR/FR = RAW_MAX same checkpoint; Target is test-aware H>=.9585 then E; '
                 'RR_VAL_SELECTED uses validation; E_MIN_DIAG50 is test oracle. '
                 'FLOPs estimate: '+profile.get('flops_convention','')+'. Offline LP cache generation excluded from inference.'),
        'Target selector':'FH12_H9585_E_SCC_PSNR_STEP_v1','Target n eligible':target['n_eligible'],
        'Target status':target['target_status'],'Target joint pass':target.get('joint_pass',False),
        'RR_VAL_SELECTED val ERGAS':val['val_ergas'],
        'E_MIN_DIAG50 n evaluated':emin.get('n_evaluated'),'E_MIN_DIAG50 independent test':False}
    for label,key in RR_LABELS.items(): values[label]=raw['rr'][key]
    for label,key in FR_LABELS.items(): values['HQNR↑' if key=='hqnr' else label]=raw['fr'][key]
    for prefix,report in [('RAW_MAX',raw),('Target',target),('Exact50K',exact),('RR_VAL_SELECTED',val),('E_MIN_DIAG50',emin)]:
        values.update(selection_values(prefix,report))
    for split,info in data['splits'].items(): values[f'FH12 LP {split} SHA256']=info['lpan_sha256']
    return {k:'' if v is None else v for k,v in values.items()}


def label_map(headers):
    labels={}
    for i,label in enumerate(headers,1):
        label=str(label).strip()
        if label:
            if label in labels: raise ValueError(f'Duplicate Sheet header: {label}')
            labels[label]=i
    return labels


def column(index):
    result=''
    while index:
        index,digit=divmod(index-1,26); result=chr(65+digit)+result
    return result


def a1(row,col): return f'{column(col)}{row}'


def same_cell(actual,expected):
    if expected=='': return actual in ('',None)
    if isinstance(expected,bool): return str(actual).lower()==str(expected).lower()
    if isinstance(expected,(int,float)):
        try: return abs(float(actual)-expected)<=1e-12*max(1.,abs(expected))
        except (ValueError,TypeError): return False
    return str(actual)==str(expected)


def header_group(label):
    for prefix in ('RAW_MAX','Target','Exact50K','RR_VAL_SELECTED','E_MIN_DIAG50'):
        if label.startswith(prefix+' '): return prefix
    return 'FH12'


def upload_run(run,root=ROOT):
    root=Path(root); values=row_values(run,root); case=case_for(run)
    from tools.fh12_runner import detect_server
    local=detect_server(root,(root/'work_dir/_fh12/local_server.txt').read_text().strip()
                        if (root/'work_dir/_fh12/local_server.txt').is_file() else None)
    if local!=case.server_id: raise ValueError('Refusing upload into another server tab')
    gu=legacy_constants()
    import gspread
    lockpath=root/'work_dir/.gspread_write.lock'; lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        book=gspread.service_account(filename=str(root/'gspread'/Path(gu.CRED).name)).open(gu.SHEET)
        ws=book.worksheet(TABS[case.server_id])  # no creation/fallback API
        if int(ws.id)!=GIDS[case.server_id]: raise ValueError('FH12 worksheet gid mismatch')
        headers=ws.row_values(gu.ORIGIN_ROW+1); labels=label_map(headers)
        if 'Run' not in labels: raise ValueError('Existing Sheet Run schema missing')
        # Read occupied extent so a historical row with a blank Run is never overwritten.
        table=ws.get_all_values()
        matches=[]
        for number,row in enumerate(table[gu.ORIGIN_ROW+1:],gu.ORIGIN_ROW+2):
            def cell(label):
                col=labels.get(label,0)
                return row[col-1] if 0<col<=len(row) else ''
            if cell('FH12 campaign')==CAMPAIGN_ID and cell('FH12 run id')==run:
                if cell('Run')!=run: raise ValueError('FH12 compound-key row has conflicting Run')
                matches.append(number)
        if len(matches)>1: raise ValueError('Duplicate FH12 campaign/run rows; refusing ambiguous upsert')
        rownum=matches[0] if matches else max(len(table)+1,gu.ORIGIN_ROW+2)
        missing=[label for label in values if label not in labels]
        first=max(len(headers),len(ws.row_values(gu.ORIGIN_ROW)))+1
        last=first+len(missing)-1
        if last>ws.col_count: ws.add_cols(last-ws.col_count)
        if rownum>ws.row_count: ws.add_rows(rownum-ws.row_count)
        edits=[]
        for col,label in enumerate(missing,first):
            labels[label]=col
            edits.extend([{'range':a1(gu.ORIGIN_ROW,col),'values':[[header_group(label)]]},
                          {'range':a1(gu.ORIGIN_ROW+1,col),'values':[[label]]}])
        for label,value in values.items(): edits.append({'range':a1(rownum,labels[label]),'values':[[value]]})
        ws.batch_update(edits,value_input_option='RAW')
        observed=ws.row_values(rownum,value_render_option='UNFORMATTED_VALUE')
        for label,want in values.items():
            col=labels[label]; got=observed[col-1] if col<=len(observed) else ''
            if not same_cell(got,want): raise ValueError(f'FH12 readback mismatch: row {rownum}, {label}')
        receipt=dict(schema=SCHEMA,campaign_id=CAMPAIGN_ID,run_id=run,server=case.server_id,
                     gid=ws.id,worksheet=TABS[case.server_id],row=rownum,readback_verified=True,
                     raw_max_checkpoint_sha256=values['RAW_MAX checkpoint SHA256'],
                     target_checkpoint_sha256=values['Target checkpoint SHA256'],
                     exact50k_checkpoint_sha256=values['Exact50K checkpoint SHA256'],uploaded_at_utc=utcnow())
        atomic_json(root/'work_dir'/run/'official/upload_receipt.json',receipt)
        return receipt
