"""s2 finishes only its original RGBN donor; s3 trains fresh local GF2 pairs."""
import json,shutil
from directshift48.reporting import atomic_json,utc_now
from . import plan


def original_qb_next(state):
    wanted=plan.donor_config('s2',1)['run_id']
    return next((r for r in state['runs'] if r['run_id']==wanted
                 and r['status'] not in ('COMPLETED','FAILED','BLOCKED_DONOR')),None)


def finish_qb_reference(device,publish):
    from panda_unified_campaign import runtime as original
    from panda_unified_campaign.release import verify
    verify(environment=True)
    root=plan.donor_root('s2')
    if not (root/'campaign_state.json').exists():
        raise FileNotFoundError('s2 original unified QB campaign is required; no copied donor or implicit new Stage1')
    state=original.read_state(root,'s2')
    for row in state['runs']:
        if row['case'].startswith('S2_') and row['status'] not in ('COMPLETED','FAILED','EARLY_STOPPED'):
            if not row['status'].startswith('SUPERSEDED'):
                row.update(previous_status=row['status'],status='SUPERSEDED_NOT_STARTED' if row['step']==0 else 'SUPERSEDED_PARTIAL',
                           replacement_campaign=plan.CAMPAIGN_ID,updated_at=utc_now())
    atomic_json(root/'campaign_state.json',state)
    if original_qb_next(state) is not None:
        saved=original.next_job;original.next_job=original_qb_next
        try:state=original.run('s2',device=device,publish=publish)
        finally:original.next_job=saved
    row=next(r for r in state['runs'] if r['run_id']==plan.donor_config('s2',1)['run_id'])
    if row['status']!='COMPLETED' or row['step']!=50000:
        return False
    state['status']='SUPERSEDED_AFTER_RGBN_STAGE1'
    atomic_json(root/'campaign_state.json',state)
    if publish:original.publish_safely(root,'s2')
    return True


def run(server,*,device='cuda:0',publish=True):
    from .release import verify
    from .runtime import initial_state,run as train
    from .preflight import run as preflight
    verify(environment=True)
    marker=plan.ROOT/'gspread/server.txt'
    if not marker.is_file() or marker.read_text().strip()!=server:
        raise ValueError('Execute on the declared server; local server marker differs')
    root=plan.campaign_root(server);root.mkdir(parents=True,exist_ok=True)
    if not (root/'campaign_state.json').exists():
        state=initial_state(server);state['status']='WAITING_LOCAL_STAGE1' if server=='s2' else 'READY'
        atomic_json(root/'campaign_state.json',state)
    if server=='s2' and not finish_qb_reference(device,publish):
        return {'status':'WAITING_ORIGINAL_RGBN_STAGE1'}
    proof=preflight(server,device)
    size=proof['checkpoint_bytes'];probe_stage=plan.queue(server)[0]['stage']
    required=sum((2 if probe_stage=='stage1' and c['stage']=='stage2' else 1)*
                 (len(c['evaluation_steps'])*size['evaluated']+2*size['complete_resume'])
                 for c in plan.queue(server))+3*2**30
    available=shutil.disk_usage(root).free
    if available<required:raise RuntimeError(f'Insufficient free storage: {available} < {required}')
    atomic_json(root/'storage_admission.json',dict(available_bytes=available,required_bytes=required))
    return train(server,device=device,publish=publish)
