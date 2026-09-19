"""FH20R1 JSON-only existing-tab upsert; never inference or new-sheet creation."""
import fcntl
import json
import math
from pathlib import Path
import yaml
from fh12.upload import (GIDS,TABS,RR_LABELS,FR_LABELS,legacy_constants,label_map,
                         selection_values,a1,same_cell)
from fh20r1.common import ROOT,atomic_json,object_sha,read_json,sha256,source_identity,utcnow
from fh20r1.plan import CAMPAIGN_ID,GRID_STEPS,build_config,case_for
from fh20r1.ledger import union_seconds


def optional(path): return read_json(path) if Path(path).is_file() else {}


def validate_reuse(run,root=ROOT):
    """Re-prove local semantic equivalence from bytes, not a stale TRUE flag."""
    from fh20r1.preflight import find_reusable_runs
    from fh12.postrun import select_records
    root=Path(root); c=case_for(run); cfg=build_config(c)
    doc=read_json(root/'work_dir'/run/'meta/reuse_reference.json')
    if (doc.get('campaign_id')!=CAMPAIGN_ID or doc.get('run_id')!=run
        or doc.get('credited_new_seconds')!=0 or doc.get('validated') is not True
        or doc.get('official_complete') is not True or doc.get('normal_same_step_A_U') is not True):
        raise ValueError('Invalid zero-credit FH20R1 semantic reuse claim')
    source=doc.get('source_run_id','')
    if not source or Path(source).name!=source or source==run:
        raise ValueError('Invalid local semantic reuse source')
    origin=root/'work_dir'/source
    if Path(doc['source_official_dir']).resolve()!=(origin/'official').resolve():
        raise ValueError('Reuse source official path differs from declared run')
    if (origin/'meta/reuse_reference.json').exists():
        raise ValueError('Recursive reuse is forbidden; link the original measured run')
    bridge=read_json(root/cfg['fh20r1']['reference_bridge']); release=source_identity(root)
    if (bridge.get('alias')!=c.teacher_alias or bridge.get('server')!=c.server_id
        or bridge.get('status')!='PASS' or bridge.get('complete') is not True
        or bridge.get('consumer_source_identity')!=release):
        raise ValueError('Reuse reference bridge context mismatch')
    verified=find_reusable_runs(root,c.server_id,bridge)['runs'].get(run)
    if verified is None or any(doc.get(k)!=v for k,v in verified.items()):
        raise ValueError('Semantic reuse proof no longer matches full source artifacts')
    source_cfg=yaml.safe_load((origin/'meta/config.resolved.yaml').read_text())
    grid=read_json(origin/'official/raw_grid.json')
    def numerical(name):
        return name.startswith(('fh12/','model/','pa/','tools/metrics/')) or name in ('tools/eval_dlpan.py','tools/repair_lpan.py')
    required={name:value for name,value in release['files'].items() if numerical(name)}
    if not required or any(grid['source_identity']['files'].get(k)!=v for k,v in required.items()):
        raise ValueError('Reuse numerical source coverage/revision differs')
    expected_ref=object_sha(bridge['origin_reference'])
    if source_cfg['trainer']=='fh20r1':
        source_bridge=read_json(root/source_cfg['fh20r1']['reference_bridge'])
        if source_bridge['origin_reference']!=bridge['origin_reference']:
            raise ValueError('Reuse Teacher calibration reference differs')
        expected_ref=object_sha(source_bridge)
    for record in grid['records']:
        identity=record['checkpoint_identity']
        if (identity.get('reference_sha256')!=expected_ref
            or any(identity.get(k)!=grid.get(k) for k in ('config_sha256','data_sha256','source_identity'))):
            raise ValueError('Reuse candidate context/calibration differs from the grid')
    selected=select_records(grid['records'])
    for name,key in (('raw_max','raw_max'),('target_selection','target'),('exact50k','exact50k'),
                     ('rr_val_selected','rr_val_selected'),('e_min_diag','e_min_diag')):
        report=read_json(origin/'official'/f'{name}.json'); expected=selected[key]
        if expected is None:
            if report.get('selection','missing') is not None or report.get('target_status')!='no_eligible':
                raise ValueError('Reuse no-eligible source report drift')
        elif (report.get('step')!=expected['update'] or
              any(report.get(k)!=expected.get(k) for k in ('rr','fr','val_ergas','checkpoint_identity'))):
            raise ValueError('Reuse source selected metrics differ from their official grid')
    return doc,bridge,source_cfg,grid,release


def reuse_row_values(run,root=ROOT):
    """Existing measured results, explicitly linked; no new measurement or time."""
    doc,bridge,source_cfg,grid,release=validate_reuse(run,root)
    source=doc['source_run_id']; c=case_for(run); cfg=build_config(c); root=Path(root)
    if source_cfg['trainer']=='fh12':
        from fh12.upload import row_values as old_values
        measured=old_values(source,root)
    else:
        measured=row_values(source,root)
    values={k:v for k,v in measured.items() if not k.startswith(('FH12 ','FH20R1 '))}
    receipt=optional(root/'work_dir'/source/'official/upload_receipt.json')
    values.update({'Run':run,'캠페인':'FH20R1','FH20R1 campaign':CAMPAIGN_ID,'FH20R1 run id':run,
        'FH20R1 block':c.block_id,'FH20R1 server':c.server_id,'FH20R1 role':c.role,
        'FH20R1 profile':c.profile,'FH20R1 input':c.input_layout,'FH20R1 W':c.width,
        'FH20R1 D':''.join(map(str,c.depth)),'FH20R1 Teacher alias':c.teacher_alias,
        'FH20R1 Teacher layout':c.teacher_input_layout,'FH20R1 Teacher seed':c.teacher_seed,
        'FH20R1 Student seed':c.student_seed,'FH20R1 Teacher SHA256':bridge['teacher_checkpoint_sha256'],
        'FH20R1 reused':True,'FH20R1 source run':source,'FH20R1 source sheet gid':receipt.get('gid',''),
        'FH20R1 source sheet row':receipt.get('row',''),'FH20R1 source official directory':doc['source_official_dir'],
        'FH20R1 source proof SHA256':doc['identity_sha256'],'FH20R1 source actual updates':50000,
        'FH20R1 source official complete':True,'FH20R1 actual updates':0,'FH20R1 official complete':False,
        'FH20R1 execution release':release.get('git_release',''),
        'FH20R1 source measured release':grid['source_identity'].get('git_release',''),
        'FH20R1 evaluator SHA256':grid['source_identity'].get('content_sha256',''),
        'FH20R1 data SHA256':grid['data_sha256'],'FH20R1 tau_R':bridge['tau_R'],'FH20R1 q_ref':bridge['q_ref'],
        'FH20R1 q cache SHA256':bridge['origin_reference'].get('q_cache_sha256',''),
        'FH20R1 LR schedule':cfg['fh20r1']['lr_schedule_id'],
        'FH20R1 run effective seconds':0,'FH20R1 committed train seconds':0,
        'FH20R1 committed eval seconds':0,'FH20R1 committed diagnostic seconds':0,'Train(h)':0,
        'Notes':f'REUSED measured FH12/FH20R1 result from {source}; no new training/evaluation or time credit. All metrics retain original same-step checkpoint provenance.'})
    return values


def row_values(run,root=ROOT):
    root=Path(root); c=case_for(run); wd=root/'work_dir'/run
    if (wd/'meta/reuse_reference.json').exists(): return reuse_row_values(run,root)
    cfg=yaml.safe_load((wd/'meta/config.resolved.yaml').read_text())
    status=read_json(wd/'official/postrun_status.json')
    if (cfg!=build_config(c) or not status.get('official_complete') or status.get('actual_updates')!=50000
        or status.get('config_sha256')!=object_sha(cfg)):
        raise ValueError('FH20R1 upload requires registered recipe and complete official 50K')
    names=('raw_max','target_selection','exact50k','rr_val_selected','e_min_diag')
    selection_ids=dict(zip(names,('RAW_MAX','TARGET','EXACT50K','RR_VAL_SELECTED','E_MIN_DIAG50')))
    reports={n:read_json(wd/'official'/f'{n}.json') for n in names}
    for name,r in reports.items():
        if (not r.get('official_complete') or r.get('run_id')!=run or r.get('campaign_id')!=CAMPAIGN_ID
            or r.get('config_sha256')!=object_sha(cfg) or r.get('source_identity')!=status['source_identity']
            or r.get('normal_same_step_A_U') is not True or r.get('selection_id')!=selection_ids[name]):
            raise ValueError(f'FH20R1 report context mismatch: {name}')
        if name=='target_selection' and r.get('target_status')=='no_eligible':
            if r.get('n_eligible')!=0 or r.get('selection','missing') is not None:
                raise ValueError('Malformed no_eligible Target')
            continue
        step=r['step']; folder=wd/'candidates'/str(step); ident=read_json(folder/'identity.json')
        if (step not in GRID_STEPS or ident!=r['checkpoint_identity'] or ident['update']!=step
            or any(ident.get(k)!=r.get(k) for k in ('config_sha256','data_sha256','source_identity'))
            or ident['model_sha256']!=r['checkpoint_sha256'] or sha256(folder/'model.safetensors')!=ident['model_sha256']):
            raise ValueError('FH20R1 selected checkpoint mismatch')
        if not r['rr'].get('official_complete'): raise ValueError('Unofficial selected RR')
        if not all(math.isfinite(float(r[d][k])) for d,keys in [('rr',RR_LABELS.values()),('fr',FR_LABELS.values())] for k in keys):
            raise ValueError('Nonfinite official metric')
    raw,target,exact,val,emin=(reports[n] for n in names)
    if exact['step']!=50000 or target.get('n_evaluated')!=50: raise ValueError('Incomplete 50K/grid certification')
    if any(r['data_sha256']!=raw['data_sha256'] for r in reports.values()): raise ValueError('Mixed selected datasets')
    data=read_json(root/cfg['fh20r1']['dataset_manifest'])
    if object_sha(data)!=raw['data_sha256']: raise ValueError('Data manifest changed')
    cost=read_json(wd/'official/profile.json')
    if cost.get('config_sha256')!=object_sha(cfg) or cost.get('source_identity')!=status['source_identity']:
        raise ValueError('Profile identity mismatch')
    bridgepath=root/cfg['fh20r1']['reference_bridge']; bridge=optional(bridgepath)
    if c.role=='S' and not bridge: raise ValueError('Student reference bridge missing')
    if bridge:
        if (bridge.get('alias',bridge.get('reference_alias'))!=c.teacher_alias
            or bridge.get('server')!=c.server_id or bridge.get('status')!='PASS'
            or bridge.get('complete') is not True
            or bridge.get('consumer_source_identity')!=status['source_identity']):
            raise ValueError('Wrong Teacher bridge context')
        bridge_sha=object_sha(bridge)
        if c.role=='S' and any(r['checkpoint_identity'].get('reference_sha256')!=bridge_sha
                for r in reports.values() if r.get('checkpoint_identity')):
            raise ValueError('Result/reference bridge identity mismatch')
    camp=root/'work_dir/_fh20r1'/c.server_id
    init=optional(wd/'init_manifest.json'); start=optional(wd/'meta/training_start_manifest.json')
    timing=optional(wd/'meta/timing_committed.json'); branch=optional(camp/'branch_record.json')
    segments=timing.get('segments',[])
    timing_breakdown={kind:union_seconds((s['start_utc'],s['end_utc']) for s in segments if s['kind']==kind)
                      for kind in ('train','eval','diagnostic')}
    effective_seconds=union_seconds((s['start_utc'],s['end_utc']) for s in segments
                                    if s['kind'] in timing_breakdown)
    values={'Run':run,'캠페인':'FH20R1','FH20R1 campaign':CAMPAIGN_ID,'FH20R1 run id':run,
        'FH20R1 block':c.block_id,'FH20R1 server':c.server_id,'FH20R1 role':c.role,
        'FH20R1 profile':c.profile,'FH20R1 input':c.input_layout,'FH20R1 W':c.width,'FH20R1 D':''.join(map(str,c.depth)),
        'FH20R1 Teacher alias':c.teacher_alias,'FH20R1 Teacher layout':c.teacher_input_layout,
        'FH20R1 Teacher seed':c.teacher_seed,'FH20R1 Student seed':c.student_seed,
        'FH20R1 Teacher SHA256':bridge.get('teacher_checkpoint_sha256',exact['checkpoint_sha256'] if c.role=='T' else ''),
        'FH20R1 bridge SHA256':sha256(bridgepath) if bridge else '',
        'FH20R1 tau_R':bridge.get('tau_R',''),'FH20R1 q_ref':bridge.get('q_ref',''),
        'FH20R1 calibration SHA256':bridge.get('origin_reference',{}).get('calibration_sha256',''),
        'FH20R1 q cache SHA256':bridge.get('origin_reference',{}).get('q_cache_sha256',''),
        'FH20R1 LR schedule':cfg['fh20r1']['lr_schedule_id'],'FH20R1 init policy':cfg['fh20r1']['init_policy'],
        'FH20R1 init hashes':json.dumps(init.get('hashes',{}),sort_keys=True),
        'FH20R1 data SHA256':raw['data_sha256'],'FH20R1 execution release':status['source_identity'].get('git_release',''),
        'FH20R1 evaluator SHA256':status['source_identity'].get('content_sha256',''),
        'FH20R1 actual updates':50000,'FH20R1 timing committed SHA256':object_sha(timing),
        'FH20R1 run effective seconds':effective_seconds,
        'FH20R1 committed train seconds':timing_breakdown['train'],
        'FH20R1 committed eval seconds':timing_breakdown['eval'],
        'FH20R1 committed diagnostic seconds':timing_breakdown['diagnostic'],
        'FH20R1 timing scope':'this run committed segments; interval union; not campaign total',
        'FH20R1 branch':branch.get('condition',branch.get('branch','STANDARD')),'FH20R1 branch reason':str(branch.get('reason','')),
        'FH20R1 LP phase':data['recipe']['phase_id'],'FH20R1 started UTC':start.get('started_at_utc',''),
        'FH20R1 completed UTC':status.get('completed_at_utc',''),'FH20R1 official complete':True,
        'FH20R1 cost scope':cost.get('scope',''),'FH20R1 FLOPs convention':cost.get('flops_convention',''),
        'Params(M)':cost['params_m'],'FLOPs(G)':cost['flops_g'],'Infer(ms)':cost['infer_ms'],'Mem(MB)':cost.get('mem_mb'),
        'Train(h)':optional(wd/'meta/training_status.json').get('training_seconds',0)/3600,
        'Notes':'FH20R1; main=RAW_MAX; all selected metrics same-step A/U. Target and E_MIN test-aware; A/U crosses diagnostics only.',
        'Target selector':'FH12_H9585_E_SCC_PSNR_STEP_v1','Target n eligible':target['n_eligible'],
        'Target status':target['target_status'],'Target joint pass':target.get('joint_pass',False),
        'RR_VAL_SELECTED val ERGAS':val['val_ergas'],'E_MIN_DIAG50 independent test':False,'E_MIN_DIAG50 n evaluated':50}
    for k,v in RR_LABELS.items(): values[k]=raw['rr'][v]
    for k,v in FR_LABELS.items(): values['HQNR↑' if v=='hqnr' else k]=raw['fr'][v]
    for prefix,name in zip(('RAW_MAX','Target','Exact50K','RR_VAL_SELECTED','E_MIN_DIAG50'),names):
        values.update(selection_values(prefix,reports[name]))
    for split,item in data['splits'].items(): values[f'FH20R1 LP {split} SHA256']=item['lpan_sha256']
    return {k:'' if v is None else v for k,v in values.items()}


def upload_run(run,root=ROOT):
    root=Path(root); values=row_values(run,root); c=case_for(run)
    values['FH20R1 upload status']='READBACK_PENDING'
    from tools.fh12_runner import detect_server
    pointer=root/'work_dir/_fh20r1/local_server.txt'
    local=detect_server(root,pointer.read_text().strip() if pointer.exists() else None)
    if local!=c.server_id: raise ValueError('FH20R1 upload server mismatch')
    gu=legacy_constants()
    import gspread
    with (root/'work_dir/.gspread_write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        book=gspread.service_account(filename=str(root/'gspread'/Path(gu.CRED).name)).open(gu.SHEET)
        ws=book.worksheet(TABS[local])
        if int(ws.id)!=GIDS[local]: raise ValueError('FH20R1 existing tab GID mismatch')
        headers=ws.row_values(gu.ORIGIN_ROW+1); labels=label_map(headers)
        if 'Run' not in labels: raise ValueError('Missing existing Run column')
        table=ws.get_all_values(); matches=[]
        for number,row in enumerate(table[gu.ORIGIN_ROW+1:],gu.ORIGIN_ROW+2):
            def cell(label):
                col=labels.get(label,0); return row[col-1] if 0<col<=len(row) else ''
            if cell('FH20R1 campaign')==CAMPAIGN_ID and cell('FH20R1 run id')==run:
                if cell('Run')!=run: raise ValueError('Conflicting compound key')
                matches.append(number)
        if len(matches)>1: raise ValueError('Duplicate FH20R1 compound-key rows')
        rownum=matches[0] if matches else max(len(table)+1,gu.ORIGIN_ROW+2)
        missing=[k for k in values if k not in labels]
        first=max(len(headers),len(ws.row_values(gu.ORIGIN_ROW)))+1
        last=first+len(missing)-1
        if last>ws.col_count: ws.add_cols(last-ws.col_count)
        if rownum>ws.row_count: ws.add_rows(rownum-ws.row_count)
        edits=[]
        for col,label in enumerate(missing,first):
            labels[label]=col
            edits.extend([dict(range=a1(gu.ORIGIN_ROW,col),values=[['FH20R1']]),
                          dict(range=a1(gu.ORIGIN_ROW+1,col),values=[[label]])])
        for label,value in values.items(): edits.append(dict(range=a1(rownum,labels[label]),values=[[value]]))
        ws.batch_update(edits,value_input_option='RAW')
        observed=ws.row_values(rownum,value_render_option='UNFORMATTED_VALUE')
        for label,want in values.items():
            col=labels[label]; got=observed[col-1] if col<=len(observed) else ''
            if not same_cell(got,want): raise ValueError(f'FH20R1 readback mismatch: {label}')
        # Publish success only after all submitted metrics/provenance were read
        # back. A failed first read leaves an explicit pending marker in Sheet.
        status_col=labels['FH20R1 upload status']
        ws.batch_update([dict(range=a1(rownum,status_col),values=[['READBACK_VERIFIED']])],value_input_option='RAW')
        confirmed=ws.row_values(rownum,value_render_option='UNFORMATTED_VALUE')
        if len(confirmed)<status_col or confirmed[status_col-1]!='READBACK_VERIFIED':
            raise ValueError('FH20R1 upload status readback mismatch')
        receipt=dict(campaign_id=CAMPAIGN_ID,run_id=run,server=local,gid=ws.id,row=rownum,
                     readback_verified=True,uploaded_at_utc=utcnow())
        atomic_json(root/'work_dir'/run/'official/upload_receipt.json',receipt)
        return receipt
