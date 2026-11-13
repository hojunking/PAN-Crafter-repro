"""Finish unchanged local Stage1 donors, then start fresh zero-safe Stage2 seeds."""
import importlib,json,shutil
from directshift48.reporting import atomic_json,utc_now
from . import plan


def finish_stage1(server,device,publish):
    package='panda_sensor_eq40' if server=='s3' else 'panda_unified_campaign'
    original=importlib.import_module(package+'.runtime')
    original_plan=importlib.import_module(package+'.plan')
    root=plan.donor_root(server)
    if server in ('s1','s2') and not (root/'campaign_state.json').is_file():
        raise FileNotFoundError('The original local unified Stage1 is required; no cross-server donor transfer')
    wanted={plan.donor_config(server,r)['run_id'] for r in (1,2)}
    state=original.read_state(root,server)
    def only_stage1(value):
        return next((r for r in value['runs'] if r['run_id'] in wanted and
                     r['status'] not in ('COMPLETED','FAILED','BLOCKED_DONOR')),None)
    if only_stage1(state) is not None:
        if server=='s3':
            from panda_sensor_eq40.preflight import run as check
            proof=check(server,device)
            size=proof['checkpoint_bytes']
            needed=sum((2 if c['stage']=='stage2' else 1)*
                (len(c['evaluation_steps'])*size['evaluated']+2*size['complete_resume'])
                for c in original_plan.queue(server))+3*2**30
            if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient space for GF2 Stage1/Stage2')
        previous=original.next_job;original.next_job=only_stage1
        try:state=original.run(server,device=device,publish=publish)
        finally:original.next_job=previous
    selected=[r for r in state['runs'] if r['run_id'] in wanted]
    if len(selected)!=len(wanted) or any(r['status']!='COMPLETED' or r['step']!=50000 for r in selected):
        return False
    # Keep configs/weights/evaluations intact. Prevent the old Student queue
    # from accidentally continuing after the new method's handoff.
    for p in {root,plan.prior_root(server)}:
        path=p/'campaign_state.json'
        if not path.exists():continue
        old=json.loads(path.read_text())
        for row in old['runs']:
            if row['case'].startswith('S2_') and row['status'] not in ('COMPLETED','FAILED','EARLY_STOPPED'):
                if not row['status'].startswith('SUPERSEDED'):
                    row.update(previous_status=row['status'],status='SUPERSEDED_ZEROQ',
                        replacement_campaign=plan.CAMPAIGN_ID,updated_at=utc_now())
        old.update(status='SUPERSEDED_BY_ZEROQ',replacement_campaign=plan.CAMPAIGN_ID)
        atomic_json(path,old)
    return True


def run(server,*,device='cuda:0',publish=True):
    from .release import verify
    from .runtime import initial_state,run as train
    from .preflight import run as check
    verify(environment=True)
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=server:raise ValueError('Wrong physical server')
    root=plan.campaign_root(server);root.mkdir(parents=True,exist_ok=True)
    if not (root/'campaign_state.json').exists():atomic_json(root/'campaign_state.json',initial_state(server))
    if not finish_stage1(server,device,publish):return {'status':'WAITING_LOCAL_STAGE1'}
    proof=check(server,device)
    needed=sum(len(c['evaluation_steps'])*proof['checkpoint_bytes']['evaluated']+
               2*proof['checkpoint_bytes']['complete_resume'] for c in plan.queue(server))+3*2**30
    if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient space for retained evaluations/resume')
    return train(server,device=device,publish=publish)
