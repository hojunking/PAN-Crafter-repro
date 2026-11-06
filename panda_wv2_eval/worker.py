"""Durable deferred worker: entire WV3 queue first, then three frozen evaluations."""
import os
import signal
import subprocess
import time
from .common import ROOT, OUT, PLAN, read, write, now, verify_release, dependency_ready


def gpu_pids():
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(x.strip()) for x in output.splitlines() if x.strip()}


def run():
    verify_release()
    if (ROOT/'gspread/server.txt').read_text().strip()!='s1':raise ValueError('WV2 queue belongs to s1')
    state = read(OUT/'state.json')
    stop={'value':False}
    def stopped(*_):stop['value']=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def save(status,**extra):
        state.update(status=status,updated_at=now(),**extra);write(OUT/'state.json',state)
    while not stop['value']:
        ready,reason=dependency_ready(read(ROOT/PLAN['dependency']['state_path']))
        if not ready:
            save('WAIT_WV3_QUEUE',reason=reason)
            print('WAIT_WV3_QUEUE',reason,flush=True);time.sleep(30);continue
        others=gpu_pids()-{os.getpid()}
        if others:
            save('WAIT_IDLE_GPU',reason='GPU PIDs: '+str(sorted(others)))
            time.sleep(30);continue
        break
    if stop['value']:save('STOPPED');return
    from .evaluate import evaluate
    from .publication import publish
    try:
        for case in PLAN['cases']:
            if stop['value']:save('STOPPED');return
            key=case['case_id'];state['runs'][key]={'status':'RUNNING'};save('RUNNING',active_case=key)
            publish(apply=True)
            report=evaluate(case)
            state['runs'][key]={'status':'COMPLETED','metrics':{s:report[s]['metrics'] for s in ('rr','fr')}}
            save('RUNNING',active_case=key)
            publish(apply=True)
        save('COMPLETED',active_case=None,reason='3/3 evaluated; Sheet readback verified')
    except Exception as exc:
        save('ERROR',error=repr(exc));raise


if __name__=='__main__':run()
