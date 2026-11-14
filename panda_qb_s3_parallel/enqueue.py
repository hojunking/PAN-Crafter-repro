"""s3 GF2 -> QB cutover. Preserve GF2 evidence and checkpoint on SIGTERM."""
import argparse,json,os,subprocess,time
from pathlib import Path
from panda_unified_campaign import enqueue as base
from . import plan
from .release import verify
NAME='panda-qb-s3-parallel-v1'

def command(image,*,gpu=False,detached=False,dlpan=None):
    # s2 here selects QB DATA MOUNTS only; config, provenance, marker and outputs are s3.
    cmd=base.command('s2',image,gpu=gpu,dlpan=dlpan);image_id=cmd.pop()
    for i,x in enumerate(cmd):
        if x=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):cmd[i+1]+=',readonly'
    for package in ('panda_qb_s3_parallel','panda_stage1_reference'):
        path=plan.ROOT/'work_dir'/package;path.mkdir(parents=True,exist_ok=True)
        cmd+=['--mount',f'type=bind,src={path},dst=/workspace/work_dir/{package}']
    if detached:
        cmd.remove('--rm');cmd+=['--detach','--name',NAME,'--label','panda.server=s3',
            '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id,'python','-m','panda_qb_s3_parallel']

def authenticate(info,campaign,source,commands):
    labels=info['Config'].get('Labels') or {}
    if any(labels.get(k)!=v for k,v in {'panda.server':'s3','panda.campaign':campaign,'panda.source':source}.items()):raise ValueError('Unexpected container identity')
    mounts=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
    if len(mounts)!=1 or Path(mounts[0]).resolve()!=plan.ROOT.resolve() or info['Config']['Cmd'] not in commands:raise ValueError('Unexpected checkout/command')

def stop_gf2(image,dlpan=None):
    from panda_sensor_main50 import plan as old
    from panda_sensor_main50.release import verify as oldverify
    from directshift48.reporting import atomic_json,utc_now
    campaign=old.CAMPAIGN_ID;sha=oldverify()['source_sha256']
    commands=[['python','-m','panda_sensor_main50','run','--server','s3'],
              ['python','-m','panda_donor_read_guard','--package','panda_sensor_main50','--server','s3']]
    names=subprocess.check_output(['docker','ps','-a','--filter','label=panda.server=s3','--filter','label=panda.campaign='+campaign,'--format','{{.Names}}'],text=True).splitlines()
    for name in names:
        info=base.inspect(name);authenticate(info,campaign,sha,commands)
        if info['State']['Running']:
            subprocess.run(['docker','kill','--signal=SIGTERM',name],check=True)
            deadline=time.monotonic()+240
            while base.inspect(name)['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('GF2 still saving/evaluating; no SIGKILL. Retry after graceful exit.')
                time.sleep(2)
    root=old.campaign_root('s3');path=root/'campaign_state.json'
    if path.exists():
        state=json.loads(path.read_text());backup=plan.campaign_root('s3')/'cutover'
        backup.mkdir(parents=True,exist_ok=True)
        before=backup/'gf2_before.json'
        if not before.exists():atomic_json(before,state)
        for job in state['runs']:
            if job['status']!='COMPLETED' and not job['status'].startswith('CANCELLED'):
                if job['step']:
                    latest=root/'runs'/job['run_id']/'latest.json';meta=json.loads(latest.read_text())
                    if meta['step']!=job['step']:raise ValueError('GF2 final checkpoint/state step differs')
                    from panda_donor_read_guard.guard import sha as file_sha
                    if file_sha(plan.local_path(meta['path']))!=meta['file_sha256']:raise ValueError('GF2 stop checkpoint failed SHA')
                job.update(previous_status=job['status'],status='CANCELLED_BY_USER',replacement_campaign=plan.CAMPAIGN_ID,updated_at=utc_now())
        state.update(status='CANCELLED_BY_USER_QB_PRIORITY',updated_at=utc_now());atomic_json(path,state)
        # Publish only that historical GF2 campaign's status, preserving its values/weights.
        from panda_sensor_main50.enqueue import command as oldcommand
        subprocess.run(oldcommand('s3',image,gpu=False,dlpan=dlpan)+['publish','--server','s3','--apply'],check=True)
    from panda_stage1_reference.enqueue import command as refcommand
    subprocess.run(refcommand('s3','GF2',image,dlpan)+['--apply'],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'));p.add_argument('--check-only',action='store_true')
    a=p.parse_args();proof=verify()
    for action in ('check','check-sheet'):subprocess.run(command(a.image,dlpan=a.dlpan)+[action,'--server','s3'],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s3':raise ValueError('Cutover executes on s3 only')
    current=base.inspect(NAME)
    if current:
        authenticate(current,plan.CAMPAIGN_ID,proof['source_sha256'],[['python','-m','panda_qb_s3_parallel','run','--server','s3']])
        if current['State']['Running']:print('Already running '+NAME);return
    stop_gf2(a.image,a.dlpan)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:raise RuntimeError('Another GPU process remains; it was not terminated')
    if current:subprocess.run(['docker','start',NAME],check=True)
    else:subprocess.run(command(a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server','s3'],check=True)
    print('Started '+NAME+'; local Stage1 then five Stage2 cases; GF2 preserved',flush=True)
if __name__=='__main__':main()
