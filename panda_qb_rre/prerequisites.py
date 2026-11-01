"""Authenticate the actual completed, early-stopped and still-pending jobs."""
import hashlib
import json
from pathlib import Path
from . import registry as G

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def terminal(root,server):
    from panda_sensor_refine.orchestration import final_evidence,local
    from panda_sensor_focus import early_stop
    from .runner import TERMINAL
    from . import failures,plan
    from types import SimpleNamespace
    root=Path(root);folder=root/G.OUTPUT/G.CAMPAIGN_ID/server/'QB'
    state=read(folder/'campaign_state.json');rows=state.get('runs',[])
    # Revalidate every held branch, including unchanged checkpoint/trace SHA.
    from copy import deepcopy
    checked=deepcopy(state)
    failures.synchronize(SimpleNamespace(root=folder,server=server,state=checked,config=plan.default_config(server)))
    if any(checked.get(k,{})!=state.get(k,{}) for k in ('failed_runs','blocked_runs')):
        raise ValueError('Missing authenticated failed/blocked accounting')
    expected=set(G.RUNS[server]);observed={(r['case'],r['replica']) for r in rows}
    blocked={(r['case'],r['replica']) for r in state.get('blocked_runs',{}).values()}
    pending={tuple(job) for group in state.get('pending_groups',[]) for job in group['runs']}
    if (state.get('status') not in TERMINAL or len(observed)!=len(rows) or observed&pending
        or blocked&(observed|pending) or observed|pending|blocked!=expected
        or bool(pending)!=(state['status']=='BUDGET_COMPLETE_WITH_PENDING')):
        raise ValueError('QB_RRE completed/pending accounting is incomplete or duplicated')
    completed=[];stopped=[];failed=[]
    for row in rows:
        directory=folder/'runs'/row['run_id']/'attempt_01';cfg=read(directory/'config.json')
        recipe=G.job(server,row['case'],row['replica'])
        if (cfg.get('campaign_id')!=G.CAMPAIGN_ID or cfg.get('server')!=server or cfg.get('dataset')!='QB'
            or cfg.get('seed')!=recipe['seed'] or cfg.get('planned_horizon')!=recipe['updates']
            or local(row['directory'],root).resolve()!=directory.resolve()):
            raise ValueError('QB_RRE terminal run identity differs')
        if row['status']=='COMPLETED' and row['step']==recipe['updates']:
            completed.append({**final_evidence(folder,cfg),'status':'COMPLETED'})
        elif row['status']==early_stop.STATUS and cfg['phase']=='stage2':
            record=early_stop.verify(row,cfg,root)
            stopped.append(dict(run_id=row['run_id'],step=row['step'],decision=record['decision'],
                                early_stop_sha256=sha(directory/'early_stop.json')))
        elif row['status']=='FAILED' and row['run_id'] in state.get('failed_runs',{}):
            failed.append(state['failed_runs'][row['run_id']])
        else:raise ValueError('Missing full-horizon or authentic early-stop proof')
    if bool(failed)!=(state['status']==failures.STATUS) and not pending:
        raise ValueError('Failed branches cannot be reported as a completed queue')
    return dict(campaign_status=state['status'],completed_runs=completed,early_stopped_runs=stopped,
        completed_count=len(completed),early_stopped_count=len(stopped),pending_count=len(pending),
        failed_count=len(failed),failed_runs=failed,blocked_count=len(blocked),blocked_runs=sorted(blocked),
        pending_runs=sorted(pending),all_registered_accounted=True,
        full_queue_completed_without_early_stops=not stopped and not pending and not failed and not blocked,
        state_sha256=sha(folder/'campaign_state.json'))
