"""Finish the active GF2 R2 cycle before starting QB; preserve all prior results."""
import fcntl
import json
import signal
import time
from directshift48.reporting import atomic_json, utc_now
from panda_input3_repeat import recipes as gf2
from . import plan

EXPECTED = gf2.queue('s3')
PREVIOUS_ROOT = plan.ROOT / 'work_dir/panda_input3_repeat' / gf2.CAMPAIGN_ID / 's3/GF2'


def finished():
    p=PREVIOUS_ROOT/'campaign_state.json'
    if not p.exists():raise ValueError('Expected existing s3 GF2 R2 queue is missing')
    state=json.loads(p.read_text())
    if (state.get('campaign_id')!=gf2.CAMPAIGN_ID or state.get('server')!='s3'
            or state.get('queue_sha256')!=plan.digest(EXPECTED)
            or [r['run_id'] for r in state['runs']]!=[c['run_id'] for c in EXPECTED]):
        raise ValueError('GF2 predecessor identity changed')
    if state.get('status') not in ('COMPLETED','FINISHED_WITH_FAILURES'):return False
    for row,cfg in zip(state['runs'],EXPECTED):
        if row['status'] not in ('COMPLETED','FAILED','BLOCKED_DONOR','EARLY_STOPPED'):return False
        if row['status']=='COMPLETED' and row['step']!=cfg['horizon']:
            raise ValueError('GF2 marked complete before its endpoint')
    with (PREVIOUS_ROOT/'runner.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def wait():
    stop=[False]
    def handler(signum,frame):stop[0]=True
    saved={s:signal.signal(s,handler) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while not finished():
            if stop[0]:return False
            atomic_json(plan.campaign_root('s3')/'waiting.json',
                {'status':'WAITING_FOR_GF2_R2','at':utc_now()})
            print('Waiting for s3 GF2 R2 to finish; existing jobs unchanged',flush=True)
            for _ in range(15):
                if stop[0]:return False
                time.sleep(2)
        atomic_json(plan.campaign_root('s3')/'waiting.json',{'status':'GF2_R2_FINISHED','at':utc_now()})
        return not stop[0]
    finally:
        for s,h in saved.items():signal.signal(s,h)
