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
    root=Path(root);folder=root/G.OUTPUT/G.CAMPAIGN_ID/server/'QB'
    state=read(folder/'campaign_state.json');rows=state.get('runs',[])
    expected=set(G.RUNS[server]);observed={(r['case'],r['replica']) for r in rows}
    pending={tuple(job) for group in state.get('pending_groups',[]) for job in group['runs']}
    if (state.get('status') not in TERMINAL or len(observed)!=len(rows) or observed&pending
        or observed|pending!=expected or bool(pending)!=(state['status']=='BUDGET_COMPLETE_WITH_PENDING')):
        raise ValueError('QB40 completed/pending accounting is incomplete or duplicated')
    completed=[];stopped=[]
    for row in rows:
        directory=folder/'runs'/row['run_id']/'attempt_01';cfg=read(directory/'config.json')
        recipe=G.job(server,row['case'],row['replica'])
        if (cfg.get('campaign_id')!=G.CAMPAIGN_ID or cfg.get('server')!=server or cfg.get('dataset')!='QB'
            or cfg.get('seed')!=recipe['seed'] or cfg.get('planned_horizon')!=recipe['updates']
            or local(row['directory'],root).resolve()!=directory.resolve()):
            raise ValueError('QB40 terminal run identity differs')
        if row['status']=='COMPLETED' and row['step']==recipe['updates']:
            completed.append({**final_evidence(folder,cfg),'status':'COMPLETED'})
        elif row['status']==early_stop.STATUS and cfg['phase']=='stage2':
            record=early_stop.verify(row,cfg,root)
            stopped.append(dict(run_id=row['run_id'],step=row['step'],decision=record['decision'],
                                early_stop_sha256=sha(directory/'early_stop.json')))
        else:raise ValueError('Missing full-horizon or authentic early-stop proof')
    return dict(campaign_status=state['status'],completed_runs=completed,early_stopped_runs=stopped,
        completed_count=len(completed),early_stopped_count=len(stopped),pending_count=len(pending),
        pending_runs=sorted(pending),all_registered_accounted=True,
        full_queue_completed_without_early_stops=not stopped and not pending,state_sha256=sha(folder/'campaign_state.json'))
