"""Start the registered alpha/gamma campaign after TUNE30 exits; never stop it."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from . import plan
from .runtime import code_pin
from .transition import PREVIOUS_CAMPAIGN, predecessor_receipt, container_gate


def readiness(root, server):
    path = Path(root)/'work_dir/panda_tune30'/PREVIOUS_CAMPAIGN/server/'WV3/campaign_state.json'
    if not path.is_file():
        return {'ready':False,'status':'WAITING_FOR_TUNE30_STATE'}
    state = json.loads(path.read_text())
    status = state.get('status')
    if status in ('FAILED','STOPPED_SAFE','FINISHED_WITH_EXPLICIT_SKIPS'):
        raise ValueError('Predecessor requires inspection: '+str(status))
    if status not in ('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_BUDGET_SKIPS'):
        return {'ready':False,'status':'WAITING_FOR_TUNE30','predecessor_status':status}
    receipt = predecessor_receipt(root,server)
    # State can become terminal just before its container exits. Wait for exit.
    result = subprocess.run(['docker','inspect','--format','{{json .State}}',
                             f'panda-tune30-hardonly-wv3-{server}-v1'],
                            capture_output=True,text=True,check=True)
    container = json.loads(result.stdout)
    if container.get('Running'):
        return {'ready':False,'status':'WAITING_FOR_CONTAINER_EXIT'}
    container_gate(server)
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip():
        return {'ready':False,'status':'WAITING_FOR_IDLE_GPU'}
    return {'ready':True,'status':'READY','predecessor':receipt}


def watch(server, directory, expected_pin):
    directory = Path(directory)
    with (directory/'watcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if code_pin() != expected_pin:
                    raise ValueError('Queued source changed; inspect and explicitly re-enqueue')
                value = readiness(plan.ROOT,server)
                atomic_json(directory/'status.json',dict(value,server=server,pid=os.getpid(),at=utc_now()))
                if value['ready']:
                    break
            except Exception as exc:
                atomic_json(directory/'status.json',{'status':'BLOCKED','error':repr(exc),
                    'server':server,'pid':os.getpid(),'at':utc_now(),'predecessor_untouched':True})
                raise
            time.sleep(60)
        atomic_json(directory/'status.json',{'status':'STARTING','server':server,
                                            'pid':os.getpid(),'at':utc_now()})
        result = subprocess.run(['bash','scripts/start_panda_hard36_costed.sh',server],
                                cwd=plan.ROOT,env=dict(os.environ,PAN_HOST_PYTHON=sys.executable))
        atomic_json(directory/'status.json',{'status':'LAUNCHED' if result.returncode==0 else 'START_FAILED',
            'exit_code':result.returncode,'server':server,'at':utc_now(),
            'predecessor_untouched':True})
        return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--enqueue',action='store_true')
    parser.add_argument('--pin',type=Path)
    args = parser.parse_args(argv)
    marker = plan.ROOT/'gspread/server.txt'
    if marker.read_text().strip()!=args.server:
        raise ValueError('Local server marker mismatch')
    directory = plan.ROOT/'work_dir/panda_hard36_launchers'/args.server
    directory.mkdir(parents=True,exist_ok=True)
    if not args.enqueue:
        if args.pin is None:
            parser.error('--pin is required for an internal watcher')
        return watch(args.server,directory,json.loads(args.pin.read_text()))
    with (directory/'enqueue.lock').open('a+') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory/'watcher.lock').open('a+') as watcher_lock:
            fcntl.flock(watcher_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = directory/'status.json'
        if existing.exists() and json.loads(existing.read_text()).get('status') in ('STARTING','LAUNCHED'):
            raise ValueError('Campaign already starting/launched; inspect existing state')
        token = str(time.time_ns())
        pin = directory/('source_'+token+'.json')
        atomic_json(pin,code_pin())
        log = directory/('watcher_'+token+'.log')
        with log.open('x') as output:
            child = subprocess.Popen([sys.executable,'-m','panda_hard36.queue','--server',args.server,
                                      '--pin',str(pin)],cwd=plan.ROOT,stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        receipt={'server':args.server,'pid':child.pid,'log':str(log),'pin':str(pin),'at':utc_now()}
        atomic_json(directory/'enqueue.json',receipt)
        print(json.dumps(receipt))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
