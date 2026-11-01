"""Explicit control-only recovery of an exited, support-failed QB40 queue.

The original admission and failed checkpoints stay immutable. This narrowly
scoped transition permits fresh remaining jobs, never an old trainer resume.
"""
import argparse
from copy import deepcopy
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from directshift48.reporting import atomic_json, utc_now
from panda_sensor_refine.orchestration import read, sha
from . import registry as G

SCHEMA = 'QB40_CONTROL_RECOVERY_SUPPORT_FAILURE_V1'
CONTROL_FILES = frozenset('panda_qb40/'+name+'.py' for name in (
    'runtime','runner','enqueue','prerequisites','publication','failures','recovery'))


def validate_revision(old, new):
    if {k:v for k,v in old.items() if k!='code'} != {k:v for k,v in new.items() if k!='code'}:
        raise ValueError('Recovery cannot change data, evaluator, config or environment')
    before, after = old['code'], new['code']
    changed = {k for k in before.keys() | after.keys() if before.get(k)!=after.get(k)}
    if not changed or not changed <= CONTROL_FILES or before.keys()-after.keys():
        raise ValueError('Recovery permits only the registered queue-control changes')
    return sorted(changed)


def prepare(folder, server, new_code):
    """Save a reviewable revision receipt; do not replace any scientific record."""
    from . import plan, failures
    folder = Path(folder)
    admission = folder/'admission.json'
    state_path = folder/'campaign_state.json'
    state, old = read(state_path), read(admission)
    new = {**deepcopy(old), 'code': new_code}
    changed = validate_revision(old,new)
    receipt_path = folder/'control_recovery.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt['expected_admission']!=new or receipt['original_admission_sha256']!=sha(admission)
                or receipt['server']!=server or receipt['schema']!=SCHEMA):
            raise ValueError('Existing recovery differs; no second implicit revision')
        return receipt
    if (state.get('status')!='FAILED' or not state.get('runs') or state.get('donors')
            or any(r.get('status')!='FAILED' for r in state['runs'])):
        raise ValueError('Recovery requires only failed Stage1 runs, no active/completed trainer')
    checked = deepcopy(state)
    camp = SimpleNamespace(root=folder,server=server,state=checked,config=plan.default_config(server))
    failures.synchronize(camp)
    if len(checked['failed_runs'])!=len(state['runs']):
        raise ValueError('Every old trainer must be an authenticated held failure')
    # Byte copies preserve the original FAILED status and source provenance.
    archive = folder/'recovery_before_control_v2'
    archive.mkdir(exist_ok=True)
    for source in (admission,state_path):
        target=archive/source.name
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise ValueError('Recovery archive already differs')
        if not target.exists(): target.write_bytes(source.read_bytes())
    receipt = dict(schema=SCHEMA,campaign_id=G.CAMPAIGN_ID,server=server,at=utc_now(),
        original_admission_sha256=sha(admission),original_state_sha256=sha(state_path),
        original_started_unix=state['started_unix'],expected_admission=new,
        changed_control_files=changed,failed_runs=checked['failed_runs'],
        blocked_runs=checked['blocked_runs'],old_trainer_resume_allowed=False,
        budget_clock_reset=False,method_changed=False)
    atomic_json(receipt_path,receipt)
    return receipt


def accept_admission(camp, pin):
    """Native environment must exactly match the prepared, control-only pin."""
    from . import failures
    path=camp.root/'control_recovery.json'
    if not path.exists():
        raise ValueError('Admission changed without an explicit QB40 recovery receipt')
    receipt=read(path);old=read(camp.root/'admission.json')
    validate_revision(old,pin)
    if (receipt.get('schema')!=SCHEMA or receipt.get('campaign_id')!=G.CAMPAIGN_ID
            or receipt.get('server')!=camp.server or receipt.get('expected_admission')!=pin
            or receipt.get('original_admission_sha256')!=sha(camp.root/'admission.json')
            or receipt.get('original_started_unix')!=camp.state['started_unix']
            or receipt.get('old_trainer_resume_allowed') is not False):
        raise ValueError('QB40 recovery provenance differs')
    for key,proof in receipt['failed_runs'].items():
        rows=[r for r in camp.state['runs'] if r['run_id']==key]
        if len(rows)!=1 or failures.verify(camp,rows[0])!=proof:
            raise ValueError('Original failed branch was changed or resumed')
    path=camp.root/'admission_control_v2.json'
    if path.exists() and read(path)!=pin:
        raise ValueError('Native recovery admission changed')
    atomic_json(path,pin)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',required=True,choices=('s1','s3'))
    parser.add_argument('--wait',action='store_true',help='Wait for unrelated local GPU work to finish')
    args=parser.parse_args(argv)
    from . import enqueue as E
    from .runtime import code_pin
    from panda_sensor_refine.launch import inspect_optional
    if (E.ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Recovery server differs from local marker')
    control=E.control_dir(args.server)
    folder=E.ROOT/G.OUTPUT/G.CAMPAIGN_ID/args.server/'QB'
    with ExitStack() as stack:
        for path in (control/'controller.lock',folder/'.controller.lock'):
            stream=stack.enter_context(path.open('a+'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=inspect_optional(E.container_name(args.server))
        intent,launch=read(control/'launch_intent.json'),read(control/'launch.json')
        E.authenticate_container(args.server,current,intent,launch)
        docker=current['State']
        if docker.get('Running') or docker.get('Status')!='exited' or docker.get('OOMKilled'):
            raise ValueError('Recovery requires the authenticated exited non-OOM container')
        if docker.get('ExitCode')!=1:
            raise ValueError('Recovery is only for the diagnosed failed queue')
        E.wait_for_gpu(args.wait,lambda stage,**extra:print(json.dumps(dict(stage=stage,**extra)),flush=True))
        receipt=prepare(folder,args.server,code_pin())
        archive=folder/'recovery_before_control_v2'
        for name in ('status.json','launch.json','launch_intent.json'):
            source=control/name;target=archive/('launcher_'+name)
            if not target.exists(): target.write_bytes(source.read_bytes())
        log=archive/'docker.log'
        if not log.exists():
            with log.open('wb') as output:
                subprocess.run(['docker','logs',launch['container_id']],stdout=output,
                               stderr=subprocess.STDOUT,check=True)
        atomic_json(control/'status.json',dict(stage='RECOVERY_STARTING',server=args.server,
            campaign_id=G.CAMPAIGN_ID,at=utc_now(),recovery_receipt=str(folder/'control_recovery.json')))
        # Same authenticated container/environment, new control code via its
        # original read-only source mount. No checkpoint or runtime pins edited.
        subprocess.run(['docker','start',launch['container_id']],check=True)
    with (control/'enqueue.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','panda_qb40.enqueue',
            '--server',args.server,'--wait'],cwd=E.ROOT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    watcher=None
    if args.server=='s1' and (E.ROOT/G.OUTPUT/'priority_s1/request.json').exists():
        from .priority import apply
        watcher=apply()
    print(json.dumps(dict(status='RECOVERY_LAUNCHED',server=args.server,launcher_pid=child.pid,
        held_stage1=len(receipt['failed_runs']),blocked_students=len(receipt['blocked_runs']),
        priority=watcher),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
