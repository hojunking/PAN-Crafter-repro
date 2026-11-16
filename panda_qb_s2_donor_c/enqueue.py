"""Start an isolated s2 queue; old training/evaluation files are read-only."""
import argparse
import os
import subprocess
from panda_unified_campaign import enqueue as base
from panda_input3.enqueue import authenticate
from . import plan
from .predecessor import roots,require_finished
from .release import verify

NAME='panda-qb-s2-donor-c-v1'


def command(image,*,gpu=False,detached=False,dlpan=None):
    cmd=base.command('s2',image,gpu=gpu,dlpan=dlpan);image_id=cmd.pop()
    for i,item in enumerate(cmd):
        if item=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):
            cmd[i+1]+=',readonly'
    for _,root in roots():
        folder=root/'publication';folder.mkdir(parents=True,exist_ok=True)
        cmd+=['--mount',f'type=bind,src={folder},dst=/workspace/{folder.relative_to(plan.ROOT)}']
    output=plan.ROOT/'work_dir/panda_qb_s2_donor_c';output.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_qb_s2_donor_c']
    if detached:
        cmd.remove('--rm');cmd+=['--detach','--name',NAME,'--label','panda.server=s2',
            '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id,'python','-m','panda_qb_s2_donor_c']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    parser.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();proof=verify()
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s2':raise ValueError('Launch on physical s2')
    current=base.inspect(NAME)
    if current:
        authenticate(current,'s2',plan.CAMPAIGN_ID,proof['source_sha256'],
            [['python','-m','panda_qb_s2_donor_c','run','--server','s2']])
        if current['State']['Running']:print('Already running '+NAME);return
        if not a.check_only and not a.resume:raise ValueError('Inspect stopped queue, then use --resume')
    require_finished()
    for action in ('check','check-sheet'):
        subprocess.run(command(a.image,dlpan=a.dlpan)+[action,'--server','s2'],check=True)
    if a.check_only:return
    # Recovery is explicit and verified before the GPU queue starts.
    subprocess.run(command(a.image,dlpan=a.dlpan)+['recover','--server','s2','--apply'],check=True)
    if current:subprocess.run(['docker','start',NAME],check=True)
    else:subprocess.run(command(a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server','s2'],check=True)
    started=base.inspect(NAME)
    if not started or not started['State']['Running']:raise RuntimeError('Queue did not remain running; inspect Docker logs')
    print('Started QB representative recovery + donor48/40/50 R1 + donor48 R2/R3. Logs: docker logs -f '+NAME)


if __name__=='__main__':main()
