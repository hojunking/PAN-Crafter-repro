"""Checkpoint-preserving restart using frozen training + independent FR reporting."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
from .__main__ import ALLOWED


def reporting_sha():
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--package',choices=tuple(ALLOWED))
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--wait',action='store_true')
    a=p.parse_args();package=a.package or ('panda_wv3_s2_eq40' if a.server=='s1' else 'panda_sensor_eq40')
    if a.server not in ALLOWED[package]:raise ValueError('Invalid server/package')
    plan=importlib.import_module(package+'.plan');launch=importlib.import_module(package+'.enqueue')
    release=importlib.import_module(package+'.release').verify()
    from panda_unified_campaign.enqueue import inspect
    oldname=(launch.NAME if package=='panda_wv3_s2_eq40' else
             launch.name(a.server) if package=='panda_sensor_eq40' else launch.container_name(a.server))
    name=oldname+'-fr67'
    def command(gpu=False,detached=False):
        if package=='panda_wv3_s2_eq40':cmd=launch.command(a.image,gpu=gpu,detached=detached)
        else:cmd=launch.command(a.server,a.image,gpu=gpu,detached=detached,dlpan=a.dlpan)
        if detached:
            cmd[cmd.index('--name')+1]=name
            cmd[-1:-1]=['--label','panda.reporting='+reporting_sha()]
        return cmd+['python','-m','panda_fr_columns','--package',package]
    def identity(info,reporting=False,upgrade=False):
        labels=info['Config'].get('Labels') or {}
        expected={'panda.campaign':plan.CAMPAIGN_ID,'panda.server':a.server,'panda.source':release['source_sha256']}
        if reporting and not upgrade:expected['panda.reporting']=reporting_sha()
        if reporting and upgrade and len(labels.get('panda.reporting',''))!=64:
            raise ValueError('Missing reporting release identity')
        if any(labels.get(k)!=v for k,v in expected.items()):raise ValueError('Container identity mismatch')
        mounts=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
        if len(mounts)!=1 or Path(mounts[0]).resolve()!=plan.ROOT.resolve():raise ValueError('Wrong checkout')
        expected_cmd=(['python','-m','panda_fr_columns','--package',package,'run','--server',a.server]
                      if reporting else ['python','-m',package,'run','--server',a.server])
        if info['Config']['Cmd'] not in (expected_cmd,expected_cmd+['--device','cuda:0']):
            raise ValueError('Unexpected entrypoint; no process stopped')
    existing=inspect(name)
    upgrade=bool(existing and (existing['Config'].get('Labels') or {}).get('panda.reporting')!=reporting_sha())
    if existing:
        identity(existing,True,upgrade)
        if existing['State']['Running'] and not upgrade:print('Already running '+name);return
    for action in ('check','check-sheet'):
        subprocess.run(command()+[action,'--server',a.server],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:raise ValueError('Wrong server marker')
    if upgrade:
        if existing['State']['Running']:
            subprocess.run(['docker','kill','--signal=TERM',name],check=True)
            deadline=time.monotonic()+300
            while inspect(name)['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Reporting upgrade checkpoint stop pending')
                time.sleep(2)
        subprocess.run(['docker','rename',name,name+'-previous-'+str(time.time_ns())],check=True)
        existing=None
    prior=inspect(oldname)
    if prior:
        identity(prior)
        if prior['State']['Running']:
            subprocess.run(['docker','kill','--signal=TERM',oldname],check=True)
            deadline=time.monotonic()+300
            while inspect(oldname)['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Checkpoint stop pending; no force kill')
                time.sleep(2)
    # Preserve the original QB coordinator's targeted cutover, if this is the
    # first EQ40 launch after completing its local Stage1 donor.
    if package=='panda_sensor_eq40' and a.server=='s2':
        from panda_unified_campaign import enqueue as old
        earlier=inspect(old.container_name('s2'))
        if earlier and earlier['State']['Running']:
            old.verify_container(earlier,'s2')
            subprocess.run(['docker','kill','--signal=TERM',old.container_name('s2')],check=True)
            deadline=time.monotonic()+300
            while inspect(old.container_name('s2'))['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Original QB checkpoint stop pending')
                time.sleep(2)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        if not a.wait:raise RuntimeError('Other GPU work remains; use --wait. No unrelated job was stopped.')
        print('Waiting for existing GPU work',flush=True);time.sleep(30)
    if existing:subprocess.run(['docker','start',name],check=True)
    else:subprocess.run(command(True,True)+['run','--server',a.server],check=True)
    info=inspect(name)
    if not info or not info['State']['Running']:raise RuntimeError('New runner exited; inspect its logs')
    print('Started '+name+'; unchanged model/optimizer/stream/RNG resume',flush=True)


if __name__=='__main__':main()
