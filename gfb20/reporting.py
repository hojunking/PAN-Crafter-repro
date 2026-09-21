"""Local finite-panel reports, including every negative, failed and deferred arm."""
import csv
import json
import os
from pathlib import Path
import tempfile
from gfb20.common import ROOT,read,read_json,run_dir,camp,atomic_json,utcnow
from gfb20.plan import CAMPAIGN_ID,cases_for
from gfb20.policy import CONTRASTS,summarize_contrast


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    fd,pending=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
            stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
    finally:
        if os.path.exists(pending):os.unlink(pending)


def rebuild(root=ROOT,server='s3'):
    from gfb20.upload import flatten_summary
    folder=camp(root,server);state=read(folder/'status.json')
    completed=[];other=[];rows={}
    for case in cases_for(server):
        summary=read(run_dir(case,root)/'official/summary.json')
        if summary.get('complete'):
            rows[case.case_id]=summary
            completed.append(flatten_summary(summary,case,root))
        else:
            training=read(run_dir(case,root)/'meta/training_status.json')
            run=state.get('runs',{}).get(case.run_id,{})
            block=state.get('blocks',{}).get(case.block_id,{})
            other.append(dict(campaign_id=CAMPAIGN_ID,case_id=case.case_id,run_id=case.run_id,server=server,
                stage=case.stage,block_id=case.block_id,status=training.get('status',run.get('status',block.get('status','NOT_STARTED'))),
                actual_updates=training.get('actual_updates',0),expected_updates=case.updates,
                parent_id=case.parent_id,reason=run.get('reason',block.get('reason',''))))
    effects=[]
    for name,pairs in CONTRASTS.items():
        if not name.startswith('S'+server[1]):continue
        result=summarize_contrast(rows,pairs)
        if not result['complete']:
            effects.append(dict(contrast=name,status='INCOMPLETE_OR_FAILED',pairs=[list(p) for p in pairs]));continue
        for effect in result['effects']:
            effects.append(dict(contrast=name,status=result['status'],**effect))
    write_csv(folder/'completed_cases.csv',completed)
    write_csv(folder/'paired_effects.csv',effects)
    write_csv(folder/'failed_or_deferred_cases.csv',other)
    value=dict(campaign_id=CAMPAIGN_ID,server=server,complete_cases=len(completed),
        incomplete_failed_or_deferred=len(other),registered_cases=len(cases_for(server)),
        endpoint_joint_goal_runs=[v['run_id'] for v in rows.values() if v.get('endpoint_joint_goal')],
        decision=read(folder/'promotion_decision.json'),window=read(Path(root)/'work_dir/_gfb20/campaign_window.json'),
        setup_hours=state.get('setup_hours',0.),mixed_calibration_hours=state.get('mixed_calibration_hours',0.),
        new_campaign_observed_wall_hours=sum(v.get('wall_hours',0.) for v in state.get('runs',{}).values()),
        old_parent_costs_not_recharged=True,shared_fresh_trunk_counted_once=True,
        independent_test=False,test_aware_development=True,forks_not_independent_fresh_seeds=True,at_utc=utcnow())
    atomic_json(folder/'campaign_summary.json',value)
    return value
