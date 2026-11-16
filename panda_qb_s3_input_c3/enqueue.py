"""Start the s3-only C3 queue after GF2 R2; leave old held QB sweeps held."""
import argparse
import os
import subprocess
import sys
from panda_unified_campaign import enqueue as base
from panda_input3 import enqueue as core
from panda_input3_repeat import enqueue as gf2
from . import plan
from .release import verify
from .predecessor import finished

NAME='panda-qb-s3-input-c3-v1'


def command(image,*,gpu=False,detached=False,dlpan=None):
    # 's2' here selects only QB data mounts. Provenance/output/physical server remain s3.
    cmd=base.command('s2',image,gpu=gpu,dlpan=dlpan);image_id=cmd.pop()
    for i,item in enumerate(cmd):
        if item=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):
            cmd[i+1]+=',readonly'
    output=plan.ROOT/'work_dir/panda_qb_s3_input_c3';output.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_qb_s3_input_c3']
    if detached:
        cmd.remove('--rm');cmd+=['--detach','--name',NAME,'--label','panda.server=s3',
            '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id,'python','-m','panda_qb_s3_input_c3']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();proof=verify()
    if not a.check_only and (plan.ROOT/'gspread/server.txt').read_text().strip()!='s3':
        raise ValueError('Launch on physical s3 only')
    current=base.inspect(NAME)
    if current:
        core.authenticate(current,'s3',plan.CAMPAIGN_ID,proof['source_sha256'],
            [['python','-m','panda_qb_s3_input_c3','run','--server','s3']])
        if current['State']['Running']:print('Already queued/running '+NAME);return
        if not a.resume and not a.check_only:raise ValueError('Inspect stopped C3 queue then use --resume')
    for action in ('check','check-sheet'):
        subprocess.run(command(a.image,dlpan=a.dlpan)+[action,'--server','s3'],check=True)
    if a.check_only:return
    if not finished():
        active=base.inspect(gf2.name('s3'))
        if active and active['State']['Running']:
            from panda_input3_repeat.release import verify as gf2_verify
            from panda_input3_repeat.recipes import CAMPAIGN_ID as gf2_campaign
            core.authenticate(active,'s3',gf2_campaign,gf2_verify()['source_sha256'],
                [['python','-m','panda_input3_repeat','run','--server','s3']])
        else:
            cmd=[sys.executable,'-m','panda_input3_repeat.enqueue','--server','s3','--resume','--image',a.image]
            if a.dlpan:cmd+=['--dlpan',a.dlpan]
            subprocess.run(cmd,check=True)
    subprocess.run(command(a.image,dlpan=a.dlpan)+['publish','--server','s3','--apply'],check=True)
    if current:subprocess.run(['docker','start',NAME],check=True)
    else:subprocess.run(command(a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server','s3'],check=True)
    print('Queued existing local QB Stage1@34K -> C R1 -> C R2 -> C R3, 30K each',flush=True)


if __name__=='__main__':main()
