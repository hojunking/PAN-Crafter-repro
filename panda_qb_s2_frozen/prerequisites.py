"""Authenticate all 22 fresh runs without counting archived baselines as jobs."""
import json
from pathlib import Path
from panda_sensor_refine.orchestration import final_evidence,local,sha
from panda_sensor_focus import early_stop
from . import registry as G,plan
from .runner import TERMINAL

def terminal(root,server):
    root=Path(root);folder=root/G.OUTPUT/G.CAMPAIGN_ID/server/'QB'
    state=json.loads((folder/'campaign_state.json').read_text());rows=state.get('runs',[])
    expected=set(G.RUNS[server]);observed={(r['case'],r['replica']) for r in rows}
    if state.get('status') not in TERMINAL or len(rows)!=len(observed) or observed!=expected:
        raise ValueError('Frozen Stage2 terminal accounting is incomplete or duplicated')
    completed=[];stopped=[]
    for row in rows:
        cfg=plan.resolve_case(server,row['case'],row['replica'])
        directory=folder/'runs'/cfg['run_id']/'attempt_01'
        saved=json.loads((directory/'config.json').read_text())
        if (row['run_id']!=cfg['run_id'] or row['phase']!='stage2'
            or local(row['directory'],root).resolve()!=directory.resolve()
            or any(saved.get(k)!=cfg[k] for k in ('campaign_id','seed','case','replica','planned_horizon'))):
            raise ValueError('Terminal run configuration differs')
        if row['status']=='COMPLETED' and row['step']==30000:
            completed.append({**final_evidence(folder,saved),'status':'COMPLETED'})
        elif row['status']==early_stop.STATUS:
            proof=early_stop.verify(row,cfg,root)
            stopped.append(dict(run_id=row['run_id'],step=row['step'],decision=proof['decision'],
                                early_stop_sha256=sha(directory/'early_stop.json')))
        else: raise ValueError('Missing full30K or authenticated early-stop evidence')
    if bool(stopped)!=(state['status']==TERMINAL[1]): raise ValueError('Early stop status differs')
    return dict(campaign_status=state['status'],completed_runs=completed,early_stopped_runs=stopped,
        completed_count=len(completed),early_stopped_count=len(stopped),pending_count=0,
        failed_count=0,blocked_count=0,stage1_training_runs=0,archived_baseline_comparisons=2,
        all_registered_accounted=True,full_queue_completed_without_early_stops=not stopped,
        state_sha256=sha(folder/'campaign_state.json'))
