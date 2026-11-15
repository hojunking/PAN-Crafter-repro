"""Gracefully resume the same INPUT3 optimizer state with the new publisher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from directshift48.reporting import atomic_json
from panda_input3 import plan, enqueue as old
from panda_unified_campaign import enqueue as base
from .release import verify


def name(server):return f'panda-input3-targets-{server}-v1'


def command(server,image,*,gpu=False,detached=False,dlpan=None):
    cmd=old.command(server,image,gpu=gpu,detached=detached,dlpan=dlpan)
    cmd[-1]='panda_input3_targets'
    if detached:
        cmd[cmd.index('--name')+1]=name(server)
        i=cmd.index('--label')
        cmd[i:i]=['--label','panda.reporting='+verify()['source_sha256']]
    return cmd


def authenticate(info,server,*,new):
    proof=verify();module='panda_input3_targets' if new else 'panda_input3'
    old.authenticate(info,server,plan.CAMPAIGN_ID,proof['training_release'],
        [['python','-m',module,'run','--server',server]])
    if new and info['Config'].get('Labels',{}).get('panda.reporting') not in (
            proof['source_sha256'],*proof['compatible_reporting_upgrades']):
        raise ValueError('Reporting container belongs to a different release')


def pause(container,server):
    info=base.inspect(container)
    if not info:return
    authenticate(info,server,new=container==name(server))
    if info['State']['Running']:
        subprocess.run(['docker','kill','--signal=SIGTERM',container],check=True)
        deadline=time.monotonic()+240
        while base.inspect(container)['State']['Running']:
            if time.monotonic()>deadline:raise RuntimeError('Still saving; never force-kill training')
            time.sleep(2)
    subprocess.run(['docker','update','--restart=no',container],check=True,capture_output=True)
    state=json.loads((plan.campaign_root(server)/'campaign_state.json').read_text())
    from panda_donor_read_guard.guard import sha
    for job in state['runs']:
        if job['status']=='PAUSED' and job['step']:
            meta=json.loads((plan.campaign_root(server)/'runs'/job['run_id']/'latest.json').read_text())
            if meta['step']!=job['step'] or sha(plan.local_path(meta['path']))!=meta['file_sha256']:
                raise ValueError('Graceful-stop optimizer checkpoint does not match state')
    atomic_json(plan.campaign_root(server)/'publication/target_policy/cutover_state.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();proof=verify()
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
        raise ValueError('Run on the matching physical server')
    current=base.inspect(name(a.server))
    if current:
        authenticate(current,a.server,new=True)
        if current['State']['Running'] and current['Config']['Labels']['panda.reporting']==proof['source_sha256']:
            print('Already running '+name(a.server));return
    previous=base.inspect(old.name(a.server))
    if previous:authenticate(previous,a.server,new=False)
    for action in ('check','check-sheet'):
        subprocess.run(command(a.server,a.image,dlpan=a.dlpan)+[action,'--server',a.server],check=True)
    if a.check_only:return
    if current and current['Config']['Labels']['panda.reporting']!=proof['source_sha256']:
        pause(name(a.server),a.server)
        archive=name(a.server)+'-prior-reporting-'+str(time.time_ns())
        subprocess.run(['docker','rename',name(a.server),archive],check=True)
        current=None
    pause(old.name(a.server),a.server)
    # The old publisher is now stopped. Migration can safely retire its obsolete slots.
    subprocess.run(command(a.server,a.image,dlpan=a.dlpan)+['publish','--server',a.server,'--apply'],check=True)
    if current:subprocess.run(['docker','start',name(a.server)],check=True)
    else:subprocess.run(command(a.server,a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server',a.server],check=True)
    print('Resumed unchanged INPUT3 training with all-target publication: '+name(a.server),flush=True)


if __name__=='__main__':main()
