"""Pause declared prior queues; start the three-input comparison and retain all evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from directshift48.reporting import atomic_json,utc_now
from panda_unified_campaign import enqueue as base
from . import plan
from .release import verify

def name(server):
    plan.sensor_for(server)
    return f'panda-input3-{server}-v1'

def command(server,image,*,gpu=False,detached=False,dlpan=None):
    plan.sensor_for(server)
    cmd=base.command(server,image,gpu=gpu,dlpan=dlpan);image_id=cmd.pop()
    for i,item in enumerate(cmd):
        if item=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):
            cmd[i+1]+=',readonly'
    output=plan.ROOT/'work_dir/panda_input3';output.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_input3']
    if detached:
        cmd.remove('--rm');cmd+=['--detach','--name',name(server),'--label','panda.server='+server,
            '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id,'python','-m','panda_input3']

def authenticate(info,server,campaign,source,commands):
    labels=info['Config'].get('Labels') or {}
    expected={'panda.server':server,'panda.campaign':campaign,'panda.source':source}
    mounts=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
    if (any(labels.get(k)!=v for k,v in expected.items()) or len(mounts)!=1
        or Path(mounts[0]).resolve()!=plan.ROOT.resolve() or info['Config']['Cmd'] not in commands):
        raise ValueError('Container does not match this checkout/server/declared queue')

def pause_campaign(module,server,image,dlpan=None):
    import importlib
    from panda_donor_read_guard.guard import sha as file_sha
    old=importlib.import_module(module+'.plan');oldverify=importlib.import_module(module+'.release').verify
    commands=[['python','-m',module,'run','--server',server]]
    if module in ('panda_sensor_main50','panda_wv3_s2abl5'):commands.append(['python','-m','panda_donor_read_guard','--package',module,'--server',server])
    names=subprocess.check_output(['docker','ps','-a','--filter','label=panda.server='+server,
        '--filter','label=panda.campaign='+old.CAMPAIGN_ID,'--format','{{.Names}}'],text=True).splitlines()
    for container in names:
        info=base.inspect(container);authenticate(info,server,old.CAMPAIGN_ID,oldverify()['source_sha256'],commands)
        if info['State']['Running']:
            subprocess.run(['docker','kill','--signal=SIGTERM',container],check=True)
            deadline=time.monotonic()+240
            while base.inspect(container)['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Previous queue still saving; no SIGKILL; retry after exit')
                time.sleep(2)
        subprocess.run(['docker','update','--restart=no',container],check=True,capture_output=True)
    root=old.campaign_root(server);path=root/'campaign_state.json'
    if not path.exists():return
    state=json.loads(path.read_text());folder=plan.campaign_root(server)/'cutover';folder.mkdir(parents=True,exist_ok=True)
    before=folder/(module+'_previous_queue.json')
    if not before.exists():atomic_json(before,state)
    changed=False
    for job in state['runs']:
        if job['status'] in ('COMPLETED','FAILED','BLOCKED_DONOR','EARLY_STOPPED') or job['status'].startswith(('CANCELLED','HELD')):continue
        if job['step']:
            meta=json.loads((root/'runs'/job['run_id']/'latest.json').read_text())
            if (meta['step']>job['step'] or (job['status']=='PAUSED' and meta['step']!=job['step'])
                or file_sha(plan.local_path(meta['path']))!=meta['file_sha256']):
                raise ValueError('Previous graceful-stop checkpoint/state mismatch')
            job['retained_complete_resume_step']=meta['step']
        job.update(previous_status=job['status'],status='HELD_BY_USER',
            replacement_campaign=plan.CAMPAIGN_ID,updated_at=utc_now());changed=True
    if changed:
        state.update(status='PAUSED_BY_USER_INPUT3_PRIORITY',updated_at=utc_now());atomic_json(path,state)
        launcher=importlib.import_module(module+'.enqueue')
        if module=='panda_qb_s3_parallel':cmd=launcher.command(image,gpu=False,dlpan=dlpan)
        elif module=='panda_wv3_s2abl5':cmd=launcher.command(image,gpu=False)
        else:cmd=launcher.command(server,image,gpu=False,dlpan=dlpan)
        subprocess.run(cmd+['publish','--server',server,'--apply'],check=True)

def pause_old(server,image,dlpan=None):
    modules=(['panda_wv3_s2abl5'] if server=='s1' else
        ['panda_qb_donorgrid','panda_qb_lateval','panda_sensor_main50'] if server=='s2' else
        ['panda_qb_donorgrid','panda_qb_lateval','panda_qb_s3_parallel','panda_sensor_main50'])
    for module in modules:pause_campaign(module,server,image,dlpan)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();proof=verify()
    if not a.check_only and (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
        raise ValueError('Launch on matching physical server')
    current=base.inspect(name(a.server))
    if current:
        authenticate(current,a.server,plan.CAMPAIGN_ID,proof['source_sha256'],
            [['python','-m','panda_input3','run','--server',a.server]])
        if current['State']['Running']:print('Already running '+name(a.server));return
        if not a.resume and not a.check_only:raise ValueError('Inspect stopped queue then use --resume')
    for action in ('check','check-sheet'):
        subprocess.run(command(a.server,a.image,dlpan=a.dlpan)+[action,'--server',a.server],check=True)
    if a.check_only:return
    pause_old(a.server,a.image,a.dlpan)
    if current:subprocess.run(['docker','start',name(a.server)],check=True)
    else:subprocess.run(command(a.server,a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server',a.server],check=True)
    print('Queued '+name(a.server)+': existing local main donor -> RAW / RAW+CORRECTED / CORRECTED+DIFF',flush=True)

if __name__=='__main__':main()
