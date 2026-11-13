"""Continue original Stage1 with its unchanged complete state, then EQ40 R1/R2."""
import json
from directshift48.reporting import atomic_json,utc_now
from panda_unified_campaign import runtime as original
from panda_unified_campaign import plan as original_plan
from panda_unified_campaign.release import verify as verify_original
from . import plan


def remaining_stage1(state):
    ids={plan.donor_config(r)['run_id'] for r in (1,2)}
    return next((j for j in state['runs'] if j['run_id'] in ids
                 and j['status'] not in ('COMPLETED','FAILED','BLOCKED_DONOR')),None)


def run(server,*,device='cuda:0',publish=True):
    from .release import verify
    from .runtime import run as run_stage2,initial_state
    plan.sensor_for(server);verify(environment=True);verify_original(environment=True)
    root=plan.campaign_root(server);root.mkdir(parents=True,exist_ok=True)
    oldroot=plan.donor_root();oldstate=original.read_state(oldroot,'s1')
    obsolete=original_plan.resolve('s1',original_plan.STAGE2_CASE,2)['run_id']
    row=next(r for r in oldstate['runs'] if r['run_id']==obsolete)
    if row['step']!=0 or (oldroot/'runs'/obsolete/'train.jsonl').exists():
        raise ValueError('Old Stage2 R2 already started; explicit cutover review required')
    row.update(status='SUPERSEDED_NOT_STARTED',reason='Corrected-only alpha0 gamma.25 40K follow-up',
               replacement_campaign=plan.CAMPAIGN_ID,updated_at=utc_now())
    atomic_json(oldroot/'campaign_state.json',oldstate)
    if not (root/'campaign_state.json').exists():
        state=initial_state(server);state['status']='WAITING_ORIGINAL_STAGE1'
        atomic_json(root/'campaign_state.json',state)
    atomic_json(root/'handoff.json',dict(at=utc_now(),old_campaign=original_plan.CAMPAIGN_ID,
        old_source=verify_original()['source_sha256'],new_source=verify()['source_sha256'],
        stage1_training_changed=False,continuation='FULL_OPTIMIZER_STREAM_RNG_RESUME',
        obsolete_stage2=obsolete,status='FINISHING_ORIGINAL_STAGE1'))
    if remaining_stage1(oldstate) is not None:
        old_next=original.next_job
        original.next_job=remaining_stage1
        try:oldstate=original.run('s1',device=device,publish=publish)
        finally:original.next_job=old_next
    donors=[j for j in oldstate['runs'] if j['run_id'] in {plan.donor_config(r)['run_id'] for r in (1,2)}]
    if len(donors)!=2 or any(j['status']!='COMPLETED' or j['step']!=50000 for j in donors):
        atomic_json(root/'handoff_progress.json',dict(at=utc_now(),status='WAITING_STAGE1',donors=donors))
        return oldstate
    oldstate['status']='SUPERSEDED_AFTER_STAGE1'
    atomic_json(oldroot/'campaign_state.json',oldstate)
    if publish:original.publish_safely(oldroot,'s1')
    atomic_json(root/'handoff_progress.json',dict(at=utc_now(),status='STARTING_EQ40',donors=donors))
    # Integration check is disposable and restores exact production seed streams.
    from .preflight import run as preflight
    preflight(device=device)
    return run_stage2(server,device=device,publish=publish)
