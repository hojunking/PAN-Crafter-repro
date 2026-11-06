"""Launch a waiting container without modifying or interrupting the WV3 trainer."""
import argparse
import json
import os
import shutil
import subprocess
from .common import ROOT, OUT, PLAN, sha, read, write, verify_release, source_evidence, now

NAME='panda-wv2-zero-shot-after-wv3-s1-v1'
IMAGE='sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887'


def initialize():
    release=verify_release()
    OUT.mkdir(parents=True,exist_ok=True)
    for case in PLAN['cases']:
        source_evidence(case)
        directory=OUT/'sources'/case['case_id'];directory.mkdir(parents=True,exist_ok=True)
        for key,name in [('checkpoint','model.pt'),('evaluation','wv3_evaluation.json'),('config','source_config.json')]:
            target=directory/name;source=ROOT/case[key]
            if not target.exists():shutil.copy2(source,target)
            if sha(target)!=sha(source):raise ValueError('Preserved source collision')
    path=OUT/'state.json'
    if path.exists():
        if read(path)['source_release']!=release:raise ValueError('Already queued with different code')
    else:write(path,{'campaign_id':PLAN['campaign_id'],'status':'WAIT_WV3_QUEUE','source_release':release,
                     'created_at':now(),'runs':{c['case_id']:{'status':'WAIT_WV3_QUEUE'} for c in PLAN['cases']}})


def inspect():
    p=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True)
    if p.returncode:
        if 'No such' in p.stderr:return None
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)[0]


def command(*,detached):
    cmd=['docker','run','--init','--pid=host','--gpus','all','--shm-size=2g',
         '--user',f'{os.getuid()}:{os.getgid()}','--workdir','/workspace']
    if detached:cmd+=['--detach','--name',NAME,'--restart','on-failure:3','--label','panda.campaign='+PLAN['campaign_id'],
                      '--label','panda.source='+verify_release()]
    else:cmd+=['--rm']
    env={'PYTHONPATH':'/workspace','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MKL_THREADING_LAYER':'GNU',
         'PANCRAFTER_DLPAN':'/opt/DLPan-Toolbox','NVIDIA_DRIVER_CAPABILITIES':'compute,utility'}
    for k,v in env.items():cmd+=['--env',k+'='+v]
    mounts=[(ROOT,'/workspace',False),(OUT,'/workspace/'+PLAN['output'],True)]
    for item in PLAN['data'].values():
        p=(ROOT/item['path']).resolve(strict=True);mounts.append((p,str(p),False))
    for src,dst,rw in mounts:
        if any(c in str(src)+dst for c in (',','\n','\r')):raise ValueError('Invalid bind path')
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+('' if rw else ',readonly')]
    return cmd+[IMAGE]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.status:
        info=inspect();print(json.dumps({'container':None if info is None else info['State'],
                                       'queue':read(OUT/'state.json') if (OUT/'state.json').exists() else None},indent=2));return
    initialize();info=inspect()
    if info:
        if (info['Config'].get('Labels',{}).get('panda.source')!=verify_release()
                or info['Config']['Cmd']!=['python','-m','panda_wv2_eval.worker']):raise ValueError('Unowned WV2 container')
        if info['State']['Running']:print('Already waiting/running:',NAME);return
        if read(OUT/'state.json')['status']=='COMPLETED':print('Already completed');return
        if not args.resume:raise ValueError('Stopped WV2 queue: inspect and use --resume')
        subprocess.run(['docker','start',NAME],check=True);return
    if args.resume:raise ValueError('No WV2 container to resume')
    subprocess.run(command(detached=False)+['python','-m','panda_wv2_eval','check'],check=True)
    subprocess.run(command(detached=False)+['python','-m','panda_wv2_eval','check-sheet'],check=True)
    result=subprocess.run(command(detached=True)+['python','-m','panda_wv2_eval.worker'],check=True,capture_output=True,text=True)
    write(OUT/'launch.json',{'container':NAME,'container_id':result.stdout.strip(),'image':IMAGE,
                           'source_release':verify_release(),'created_at':now()})
    print('Queued after all 18 WV3 runs:',result.stdout.strip())


if __name__=='__main__':main()
