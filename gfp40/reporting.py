"""Complete finite panels; cross-Teacher conclusions use paired local effects."""
import csv
import json
import os
from pathlib import Path
import tempfile
from gfp40.common import ROOT,read,read_json,run_dir,camp,atomic_json,utcnow,object_sha
from gfp40.plan import (CAMPAIGN_ID,CASES,cases_for,registry_sha256,
    EXECUTION_POLICY,EXECUTION_POLICY_SHA256,CONFIRMATION_FAMILY)
from gfp40.policy import SCREEN_PAIRS,contrast,effect

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    fd,pending=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8-sig') as stream:
            out=csv.DictWriter(stream,fieldnames=fields);out.writeheader()
            for r in rows:out.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
            stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
    finally:
        if os.path.exists(pending):os.unlink(pending)

TRANSFER={'s3':(('A02','A03'),('A05','A04')),'s4':SCREEN_PAIRS['G025'],'s5':(('C01','C02'),('C04','C03'))}
SCHEDULE={98411:dict(NN='C15',NM='C16',MN='C17',MM='C18'),98412:dict(NN='C22',NM='C21',MN='C24',MM='C23')}

def confirmation_pairs(server):
    out=[]
    for seed in (98711,98712,98713):
        arms={c.arm:c.case_id for c in cases_for(server) if c.phase=='FIXED_CONFIRM' and c.stream_seed==seed}
        out.append((arms['CTRL'],arms['MIX']))
    return out

def confirmation_summary(rows):
    """Six mandatory pairs only; optional third seed is separately described."""
    expected=[pair for s in ('s3','s4','s5') for pair in confirmation_pairs(s)[:2]]
    effects=[]
    for c,m in expected:
        if all(rows.get(k,{}).get('complete') and rows[k].get('integrity_verified') for k in (c,m)):
            effects.append(dict(server=next(x.server for x in CASES if x.case_id==c),control=c,alternative=m,**effect(rows[c],rows[m])))
    mean=lambda key:sum(v[key] for v in effects)/len(effects) if effects else None
    by={s:[v for v in effects if v['server']==s] for s in ('s3','s4','s5')}
    complete=len(effects)==6
    success=bool(complete and sum(v['delta_h']>0 and v['delta_ds']<0 for v in effects)>=5
        and all(sum(v['delta_h'] for v in values)/2>=-.001 for values in by.values())
        and mean('delta_h')>=.002 and mean('relative_e')<=.005
        and all(v['relative_e']<=.01 and v['delta_dlambda']<=.001 for v in effects))
    return dict(required_pairs=6,completed_pairs=len(effects),complete=complete,operational_success=success,
        confirmation_family=CONFIRMATION_FAMILY,execution_policy_sha256=EXECUTION_POLICY_SHA256,
        effects=effects,teacher_blocks=by,mean_delta_h=mean('delta_h'),mean_relative_e=mean('relative_e'),
        independent_identically_distributed=False,p_value=None,test_aware=True)

def rebuild(root=ROOT,server='s3'):
    from gfp40.upload import flatten_summary,row_values
    folder=camp(root,server);state=read(folder/'status.json');rows={};completed=[];other=[]
    for c in cases_for(server):
        summary=read(run_dir(c,root)/'official/summary.json')
        if summary.get('complete'):
            row_values(c.run_id,root)
            rows[c.case_id]=dict(summary,integrity_verified=True)
            completed.append(flatten_summary(summary,c,root))
        else:
            tr=read(run_dir(c,root)/'meta/training_status.json');r=state.get('runs',{}).get(c.run_id,{})
            b=state.get('blocks',{}).get(c.block_id,{})
            other.append(dict(case_id=c.case_id,run_id=c.run_id,phase=c.phase,priority=c.priority,block_id=c.block_id,
                status=tr.get('status',r.get('status',b.get('status','NOT_STARTED'))),parent=c.parent,
                actual_updates=tr.get('actual_updates',0),expected_updates=c.updates,reason=r.get('reason',b.get('reason',''))))
    contrasts={'TRANSFER_G025':TRANSFER[server]}
    if server=='s3':contrasts.update(LONG60=(('A07','A08'),('A10','A09')),
        SCALE=(('A01','A02'),('A06','A05')))
    if server=='s4':contrasts.update(SCREEN_PAIRS)
    if server=='s5':contrasts.update(LOW=(('C05','C06'),('C08','C07')),HIGH=(('C09','C10'),('C12','C11')))
    for i,pair in enumerate(confirmation_pairs(server),1):contrasts[f'CONFIRM_{i}']=(pair,)
    paired=[dict(name=name,**contrast(rows,pairs)) for name,pairs in contrasts.items()]
    schedule=[]
    if server=='s5':
        for seed,paths in SCHEDULE.items():
            for label,identifier in paths.items():
                value=rows.get(identifier,{})
                schedule.append(dict(seed=seed,path=label,case=identifier,complete=value.get('complete',False),
                    lifetime_updates=120000,optimizer_reset_at=100000,
                    endpoint=value.get('selections',{}).get('EXACT_FINAL'),shared_trunks_not_independent_seeds=True))
    write_csv(folder/'completed_cases.csv',completed);write_csv(folder/'failed_or_deferred_cases.csv',other)
    write_csv(folder/'paired_effects.csv',paired);write_csv(folder/'schedule_2x2.csv',schedule)
    value=dict(campaign_id=CAMPAIGN_ID,server=server,completed=len(completed),registered=len(cases_for(server)),
        incomplete_failed_deferred=len(other),execution_policy=EXECUTION_POLICY,
        execution_policy_sha256=EXECUTION_POLICY_SHA256,registry_sha256=registry_sha256(),
        confirmation_family=CONFIRMATION_FAMILY,window=read(folder/'campaign_window.json'),
        setup_hours=state.get('setup_hours',0),family_hours=state.get('family_hours',0),
        diagnostic_hours=state.get('diagnostic_hours',0),replay_complete=state.get('replay_complete',False),
        replay_status=state.get('replay_status','DIAGNOSTICS_PENDING'),
        new_segments_wall_hours=sum(v.get('wall_hours',0) for v in state.get('runs',{}).values()),
        shared_trunk_counted_once=True,old_parent_cost_not_recharged=True,local_confirmation=confirmation_summary(rows),
        global_confirmation_status='REQUIRES_VERIFIED_THREE_SERVER_REPORTS',test_aware=True,at_utc=utcnow())
    if server=='s4':
        from gfp40.policy import select_family
        value['exploratory_screen']=dict(select_family(rows),exploratory_only=True,
            used_for_confirmation=False,confirmation_family=CONFIRMATION_FAMILY)
    atomic_json(folder/'campaign_summary.json',value)
    evidence=dict(campaign_id=CAMPAIGN_ID,server=server,window=value['window'],
        execution_policy=EXECUTION_POLICY,execution_policy_sha256=EXECUTION_POLICY_SHA256,
        registry_sha256=registry_sha256(),confirmation_family=CONFIRMATION_FAMILY,rows=rows)
    atomic_json(folder/'confirmation_evidence.json',dict(evidence,evidence_sha256=object_sha(evidence)))
    return value

def aggregate(paths,output):
    """Same preregistered recipe, independent local clocks, six primary pairs."""
    from gfp40.policy import CampaignWindow
    if len(paths)!=3:raise ValueError('Exactly one verified report from each server required')
    servers=set();rows={};windows={}
    for path in paths:
        report=read_json(path);body={k:v for k,v in report.items() if k!='evidence_sha256'}
        server=report.get('server')
        if (report.get('campaign_id')!=CAMPAIGN_ID or server not in ('s3','s4','s5') or server in servers
            or report.get('evidence_sha256')!=object_sha(body)):raise ValueError('Foreign/duplicate/tampered report')
        if (report.get('execution_policy')!=EXECUTION_POLICY
            or report.get('execution_policy_sha256')!=EXECUTION_POLICY_SHA256
            or report.get('registry_sha256')!=registry_sha256()
            or report.get('confirmation_family')!=CONFIRMATION_FAMILY
            or any(k.startswith('selection_lock') for k in report)):
            raise ValueError('Report execution policy/registry/fixed family differs')
        current_window=CampaignWindow.from_dict(report['window'])
        if current_window.to_dict().get('server')!=server:
            raise ValueError('Report window belongs to a different server')
        windows[server]=current_window.to_dict();servers.add(server)
        for identifier,row in report['rows'].items():
            case=next((c for c in CASES if c.case_id==identifier),None)
            if (case is None or case.server!=server or row.get('run_id')!=case.run_id
                or row.get('case_id')!=identifier or row.get('server')!=server
                or row.get('stage')!=case.phase or row.get('profile')!=case.arm
                or row.get('family')!=case.calibration_family
                or row.get('stream_seed')!=case.stream_seed
                or row.get('local_updates')!=case.updates
                or row.get('parent_step')!=case.parent_step
                or row.get('lifetime_updates')!=case.lifetime_student_updates
                or row.get('parent_run_id')!=case.parent_run_id
                or row.get('reference_key')!=case.reference_key):
                raise ValueError('Cross-server/case/registered recipe report row')
            if (row.get('execution_policy')!=EXECUTION_POLICY
                or row.get('execution_policy_sha256')!=EXECUTION_POLICY_SHA256
                or row.get('registry_sha256')!=registry_sha256()
                or row.get('confirmation_family')!=CONFIRMATION_FAMILY
                or any(k.startswith('selection_lock') for k in row)):
                raise ValueError('Run execution policy/registry/fixed family differs')
            rows[identifier]=row
    result=dict(campaign_id=CAMPAIGN_ID,execution_policy=EXECUTION_POLICY,
        execution_policy_sha256=EXECUTION_POLICY_SHA256,registry_sha256=registry_sha256(),
        family=CONFIRMATION_FAMILY,server_windows=windows,independent_server_clocks=True,
        primary_confirmation=confirmation_summary(rows),all_cases=list(rows),test_aware=True,at_utc=utcnow())
    atomic_json(output,result);return result
