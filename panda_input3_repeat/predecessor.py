"""Wait for the existing C30 cycle; never interrupt it or infer progress from Sheet."""
import fcntl
import json
import signal
import time
from directshift48.reporting import atomic_json, utc_now
from . import recipes

TERMINAL = ('COMPLETED', 'FAILED', 'BLOCKED_DONOR', 'EARLY_STOPPED')


def finished(server):
    root = recipes.previous.campaign_root(server)
    path = root / 'campaign_state.json'
    if not path.exists(): return False
    state = json.loads(path.read_text()); expected = recipes.previous.queue(server)
    if (state.get('campaign_id') != recipes.previous.CAMPAIGN_ID or state.get('server') != server
            or state.get('queue_sha256') != recipes.previous.original.digest(expected)
            or [r['run_id'] for r in state['runs']] != [c['run_id'] for c in expected]):
        raise ValueError('Predecessor queue identity differs')
    if state.get('status') not in ('COMPLETED', 'FINISHED_WITH_FAILURES'): return False
    for row, cfg in zip(state['runs'], expected):
        if row['status'] not in TERMINAL: return False
        if row['status'] == 'COMPLETED' and row['step'] != cfg['horizon']:
            raise ValueError('Predecessor marked complete before its endpoint')
    # Status is committed before final publication. Wait for the writer to leave.
    with (root / 'runner.lock').open('r') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return False
    return True


def wait(server):
    stopped = [False]
    def stop(signum, frame): stopped[0] = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not finished(server):
            if stopped[0]: return False
            atomic_json(recipes.campaign_root(server) / 'waiting.json',
                dict(status='WAITING_FOR_PREVIOUS_C30', server=server, at=utc_now()))
            print('Waiting for previous C30 cycle: ' + server, flush=True)
            for _ in range(15):
                if stopped[0]: return False
                time.sleep(2)
        atomic_json(recipes.campaign_root(server) / 'waiting.json',
            dict(status='PREVIOUS_C30_FINISHED', server=server, at=utc_now()))
        return not stopped[0]
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)
