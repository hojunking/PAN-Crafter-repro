"""Finish current WV3 run -> sensor Stage1 + five cases -> all deferred WV3."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from .transition import (ROOT,campaign_root,state_of,predecessor_receipt,
                         container_gate,completed_run)
from . import wv3_launch

POLL_SECONDS=15
REQUEST='USER_20261026_DATASET5_THEN_RESUME_WV3'


def controller_pin():
    from panda_hard36.runtime import code_pin
    files=list((ROOT/'panda_sensor5').rglob('*.py'))
    files+=list((ROOT/'scripts').glob('*panda_sensor5*.sh'))
    return {**code_pin(),**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(files)}}


def inspect_optional(name):
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    if result.returncode:
        if 'No such object' in result.stderr or 'No such container' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def wait_exit(name):
    value=inspect_optional(name)
    if value is None:
        raise ValueError('Expected training container is missing: '+name)
    state=value['State']
    if state['Running']:
        return False
    if state.get('Status')!='exited' or state.get('ExitCode')!=0:
        raise ValueError('Training container did not exit normally: '+name)
    return True


def hold_hard_watcher(server):
    """Stop only an idle old queue watcher, never a trainer or publisher."""
    held=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid!=os.getuid():
                continue
            argv=(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if ('-m' not in argv or argv[argv.index('-m')+1]!='panda_hard36.queue'
                    or '--server' not in argv or argv[argv.index('--server')+1]!=server):
                continue
            generation=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
            current=(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if current!=argv or (path/'stat').read_text().rsplit(')',1)[1].split()[19]!=generation:
                raise ValueError('Old watcher process identity changed')
            os.kill(int(path.name),signal.SIGTERM)
            held.append({'pid':int(path.name),'argv':argv,'generation':generation})
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return held


def initial_cutover(server,directory):
    path=directory/'initial_cutover.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('request')!=REQUEST:
            raise ValueError('Unknown existing cutover request')
        return value
    held=hold_hard_watcher(server)
    candidates=[]
    for package in ('panda_tune30','panda_hard36'):
        short=package.removeprefix('panda_')
        name=f'panda-{short}-hardonly-wv3-{server}-v1'
        container=inspect_optional(name)
        if container and container['State']['Running']:
            candidates.append((package,name))
    if len(candidates)>1:
        raise ValueError('Multiple active WV3 campaigns; no trainer was signalled')
    package,name=candidates[0] if candidates else ('panda_tune30',f'panda-tune30-hardonly-wv3-{server}-v1')
    state=state_of(ROOT,package,server)
    if state is None or state['status']=='FAILED':
        raise ValueError('Missing/failed WV3 campaign requires inspection')
    active=[r for r in state['runs'] if r['status'] in ('RUNNING','STOPPED')]
    if any(r.get('poisoned') for r in active) or len(active)>1:
        raise ValueError('Current WV3 optimizer state requires inspection')
    if not active and state['status'] not in ('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_BUDGET_SKIPS'):
        raise ValueError('No unambiguous current WV3 run')
    value={'request':REQUEST,'at_unix':time.time(),'package':package,
        'target_run_id':active[0]['run_id'] if active else None,
        'observed_step':active[0]['step'] if active else None,
        'initial_status':state['status'],'source_container':name,
        'held_hard36_watchers':held,'mode':'SAFE_SAVE_THEN_SAME_RUN_COMPLETION'}
    atomic_json(path,value)
    if candidates:
        subprocess.run(['docker','kill','--signal=SIGTERM',name],check=True)
    return value


def ensure_cutover(server,directory):
    initial=initial_cutover(server,directory)
    package=initial.get('package','panda_tune30'); short=package.removeprefix('panda_')
    target=initial.get('target_run_id')
    if not target:
        source=initial.get('source_container',f'panda-{short}-hardonly-wv3-{server}-v1')
        return wait_exit(source)
    receipt_path=directory/f'wv3_{short}_finish-current.json'
    if not receipt_path.exists():
        source=initial.get('source_container',f'panda-{short}-hardonly-wv3-{server}-v1')
        container=inspect_optional(source)
        if container and container['State']['Running']:
            # Recover a watcher crash after durable intent but before SIGTERM.
            # Recheck the exact target before ever signalling a live trainer.
            state=state_of(ROOT,package,server)
            active=[r for r in state['runs'] if r['status'] in ('RUNNING','STOPPED')]
            if len(active)!=1 or active[0]['run_id']!=target:
                raise ValueError('Original current-run target changed before safe stop')
            subprocess.run(['docker','kill','--signal=SIGTERM',source],check=True)
            return False
        if not wait_exit(source):
            return False
        state=state_of(ROOT,package,server)
        others=[r for r in state['runs'] if r['status'] in ('RUNNING','STOPPED') and r['run_id']!=target]
        if others:
            raise ValueError('WV3 run changed across cutover; refusing to hide an extra started run')
        wv3_launch.launch(server,package,'finish-current',target)
        return False
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('target_run')!=target:
        raise ValueError('Current-run adapter target differs from cutover receipt')
    if not wait_exit(receipt['container']):
        return False
    predecessor_receipt(ROOT,server)
    container_gate(server)
    return True


def start_publisher(server,package):
    subprocess.run(['bash',f'scripts/start_{package}_best_publisher.sh',server],
                   cwd=ROOT,env=dict(os.environ,PAN_HOST_PYTHON=sys.executable),check=True)


def campaign_complete(package,server):
    state=state_of(ROOT,package,server)
    if state is None:
        return False
    if state.get('status') in ('FAILED','STOPPED_SAFE'):
        raise ValueError(package+' requires inspection: '+state['status'])
    if state.get('status')!='FINISHED_REGISTERED_QUEUE':
        return False
    plan=importlib.import_module(package+'.plan')
    ids=[plan.resolve_case(server,c,r)['run_id'] for g in plan.groups(server) for c,r in g['runs']]
    if {r['run_id'] for r in state['runs']}!=set(ids):
        raise ValueError('Terminal campaign does not contain exactly its registered runs')
    for run_id in ids:
        completed_run(ROOT,package,server,run_id)
    return True


def advance(server,directory,chain):
    stage=chain.get('stage','CUTOVER')
    if stage=='CUTOVER':
        if ensure_cutover(server,directory):
            start_publisher(server,initial_cutover(server,directory).get('package','panda_tune30'))
            chain['stage']='START_SENSOR5'
    elif stage=='START_SENSOR5':
        from .plan import dataset_for_server
        sensor=dataset_for_server(server).lower()
        name=f'panda-sensor5-stage12-{sensor}-{server}-v1'
        if inspect_optional(name) is None:
            subprocess.run(['bash','scripts/start_panda_sensor5_costed.sh',server],
                cwd=ROOT,env=dict(os.environ,PAN_HOST_PYTHON=sys.executable),check=True)
        chain.update(stage='WAIT_SENSOR5',sensor_container=name)
    elif stage=='WAIT_SENSOR5':
        if wait_exit(chain['sensor_container']) and campaign_complete('panda_sensor5',server):
            chain['stage']='RESUME_TUNE30'
    elif stage in ('RESUME_TUNE30','RESUME_HARD36'):
        package='panda_tune30' if stage=='RESUME_TUNE30' else 'panda_hard36'
        receipt_path=directory/f"wv3_{package.removeprefix('panda_')}_resume-all.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if wait_exit(receipt['container']):
                if not campaign_complete(package,server):
                    raise ValueError('Deferred WV3 container exited without full queue completion')
                chain['stage']='RESUME_HARD36' if package=='panda_tune30' else 'COMPLETED'
        else:
            state=state_of(ROOT,package,server)
            if state and state['status']=='FINISHED_REGISTERED_QUEUE':
                source=f"panda-{package.removeprefix('panda_')}-hardonly-wv3-{server}-v1"
                if not wait_exit(source):
                    return
                if not campaign_complete(package,server):
                    raise ValueError('WV3 completion evidence missing')
                chain['stage']='RESUME_HARD36' if package=='panda_tune30' else 'COMPLETED'
                return
            if package=='panda_hard36':
                wv3_launch.prepare_hard(server)
            receipt=wv3_launch.launch(server,package,'resume-all')
            chain[package+'_resume']=receipt
            start_publisher(server,package)
    else:
        if stage!='COMPLETED':
            raise ValueError('Unknown chain stage: '+str(stage))


def watch(server,directory,expected_pin):
    with (directory/'watcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=directory/'chain.json'
        chain=json.loads(path.read_text()) if path.exists() else {'stage':'CUTOVER','request':REQUEST}
        while True:
            try:
                if controller_pin()!=expected_pin:
                    raise ValueError('Queued SENSOR5/WV3 source changed; inspect before re-enqueue')
                advance(server,directory,chain)
                chain.update(server=server,pid=os.getpid(),at=utc_now(),status=chain['stage'])
                atomic_json(path,chain)
                atomic_json(directory/'status.json',chain)
                if chain['stage']=='COMPLETED':
                    return 0
            except Exception as exc:
                atomic_json(directory/'status.json',{'status':'BLOCKED','stage':chain.get('stage'),
                    'error':repr(exc),'server':server,'pid':os.getpid(),'at':utc_now()})
                raise
            time.sleep(POLL_SECONDS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s3'),required=True)
    p.add_argument('--enqueue',action='store_true')
    p.add_argument('--pin',type=Path)
    args=p.parse_args(argv)
    if (ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Local server marker mismatch')
    directory=ROOT/'work_dir/panda_sensor5_launchers'/args.server
    directory.mkdir(parents=True,exist_ok=True)
    if not args.enqueue:
        if args.pin is None:
            p.error('--pin required for internal watcher')
        return watch(args.server,directory,json.loads(args.pin.read_text()))
    with (directory/'enqueue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (directory/'watcher.lock').open('a+') as watcher:
            fcntl.flock(watcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        token=str(time.time_ns())
        pin=directory/('source_'+token+'.json'); atomic_json(pin,controller_pin())
        log=directory/('watcher_'+token+'.log')
        with log.open('x') as output:
            child=subprocess.Popen([sys.executable,'-m','panda_sensor5.queue','--server',args.server,
                '--pin',str(pin)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        receipt={'server':args.server,'pid':child.pid,'log':str(log),'pin':str(pin),'at':utc_now()}
        atomic_json(directory/'enqueue.json',receipt)
        print(json.dumps(receipt))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
