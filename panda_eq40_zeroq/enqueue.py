"""Launch the shared new Stage2 queue; checkpoint-stop only registered predecessors."""
import argparse,importlib,json,os,subprocess,time
from pathlib import Path
from . import plan
from .release import verify
from panda_unified_campaign.enqueue import inspect

def name(server):return f'panda-eq40-zeroq-{server}-v1'

def command(server,image,*,gpu=True,detached=False,dlpan=None):
    package='panda_wv3_s2_eq40' if server=='s1' else 'panda_sensor_eq40'
    old=importlib.import_module(package+'.enqueue')
    cmd=(old.command(image,gpu=gpu) if server=='s1' else
         old.command(server,image,gpu=gpu,dlpan=dlpan))
    image_id=cmd.pop()
    output=plan.ROOT/'work_dir/panda_eq40_zeroq';output.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_eq40_zeroq']
    if detached:
        cmd.remove('--rm')
        cmd+=['--detach','--name',name(server),'--label','panda.campaign='+plan.CAMPAIGN_ID,
              '--label','panda.server='+server,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[image_id]

def authenticate(info,server,package):
    oldplan=importlib.import_module(package+'.plan')
    oldrelease=importlib.import_module(package+'.release').verify()
    expected={'panda.server':server,'panda.campaign':oldplan.CAMPAIGN_ID,'panda.source':oldrelease['source_sha256']}
    labels=info['Config'].get('Labels') or {}
    if any(labels.get(k)!=v for k,v in expected.items()):raise ValueError('Predecessor identity mismatch')
    paths=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
    if len(paths)!=1 or Path(paths[0]).resolve()!=plan.ROOT.resolve():raise ValueError('Wrong checkout')
    direct=['python','-m',package,'run','--server',server]
    wrapped=['python','-m','panda_fr_columns','--package',package,'run','--server',server]
    if info['Config']['Cmd'] not in (direct,wrapped,direct+['--device','cuda:0'],wrapped+['--device','cuda:0']):
        raise ValueError('Unexpected predecessor entrypoint')

def stop_predecessors(server):
    pairs=([('panda-eq40-s1-v1','panda_wv3_s2_eq40')] if server=='s1' else
           [(f'panda-sensor-eq40-{server}-v1','panda_sensor_eq40')])
    if server=='s2':pairs.append(('panda-unified-s2-v1','panda_unified_campaign'))
    for oldname,package in pairs:
        for container in (oldname,oldname+'-fr67'):
            info=inspect(container)
            if not info or not info['State']['Running']:continue
            authenticate(info,server,package)
            subprocess.run(['docker','kill','--signal=TERM',container],check=True)
            deadline=time.monotonic()+300
            while inspect(container)['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Graceful stop pending; no forced kill')
                time.sleep(2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--wait',action='store_true')
    a=p.parse_args();verify();existing=inspect(name(a.server))
    if existing:
        authenticate(existing,a.server,'panda_eq40_zeroq')
        if existing['State']['Running']:print('Already running '+name(a.server));return
    for action in ('check','check-sheet'):
        subprocess.run(command(a.server,a.image,gpu=False,dlpan=a.dlpan)+
            ['python','-m','panda_eq40_zeroq',action,'--server',a.server],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:raise ValueError('Wrong server marker')
    stop_predecessors(a.server)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        if not a.wait:raise RuntimeError('Unrelated GPU work remains; use --wait')
        print('Waiting for existing GPU work',flush=True);time.sleep(30)
    if existing:subprocess.run(['docker','start',name(a.server)],check=True)
    else:subprocess.run(command(a.server,a.image,detached=True,dlpan=a.dlpan)+
        ['python','-m','panda_eq40_zeroq','run','--server',a.server],check=True)
    if not inspect(name(a.server))['State']['Running']:raise RuntimeError('New queue exited; inspect logs')
    print('Started '+name(a.server)+'; fresh Stage2, unchanged local Stage1',flush=True)
if __name__=='__main__':main()
