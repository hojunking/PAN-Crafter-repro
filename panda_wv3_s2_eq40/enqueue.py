"""s1 cutover only: SIGTERM checkpoint, same Stage1 resume, then two EQ40 runs."""
import argparse,json,subprocess,time
from . import plan
from .release import verify
from panda_unified_campaign import enqueue as old

NAME='panda-eq40-s1-v1'

def command(image,*,detached=False,gpu=True):
    cmd=old.command('s1',image,gpu=gpu)
    output=plan.ROOT/'work_dir/panda_wv3_s2_eq40';output.mkdir(parents=True,exist_ok=True)
    image_id=cmd.pop()
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_wv3_s2_eq40']
    if detached:
        cmd.remove('--rm')
        cmd+=['--detach','--name',NAME,'--label','panda.campaign='+plan.CAMPAIGN_ID,
              '--label','panda.server=s1','--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();release=verify()
    existing=old.inspect(NAME)
    if existing:
        labels=existing['Config'].get('Labels',{})
        if labels.get('panda.source')!=release['source_sha256'] or labels.get('panda.campaign')!=plan.CAMPAIGN_ID:
            raise ValueError('Existing EQ40 container has another identity')
        if existing['State']['Running']:print('Already running '+NAME);return
        if not a.resume:raise ValueError('Use --resume for stopped EQ40 container')
    base=command(a.image,gpu=False)
    for action in ('check','check-sheet'):
        subprocess.run(base+['python','-m','panda_wv3_s2_eq40',action,'--server','s1'],check=True)
    if a.check_only:return
    previous=old.inspect(old.container_name('s1'))
    if previous and previous['State']['Running']:
        old.verify_container(previous,'s1')
        subprocess.run(['docker','kill','--signal=TERM',old.container_name('s1')],check=True)
        deadline=time.monotonic()+300
        while old.inspect(old.container_name('s1'))['State']['Running']:
            if time.monotonic()>deadline:raise RuntimeError('Graceful checkpoint stop pending; no force kill')
            time.sleep(2)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:raise RuntimeError('Another GPU process remains; no force stop: '+active)
    if not existing:
        subprocess.run(command(a.image)+['python','-m','panda_wv3_s2_eq40.preflight','--server','s1'],check=True)
    if existing:subprocess.run(['docker','start',NAME],check=True)
    else:subprocess.run(command(a.image,detached=True)+['python','-m','panda_wv3_s2_eq40','run','--server','s1'],check=True)
    print('Started '+NAME+'; Stage1 continues before EQ40',flush=True)

if __name__=='__main__':main()
