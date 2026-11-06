"""Host-Python launcher: wait for idle GPU, mount originals read-only, run s3 review."""
import argparse
import json
import subprocess
from types import SimpleNamespace
from panda_sensor_visuals.common import ROOT,read,write
from . import plan as P
from .release import verify

NAME='panda-gf2-sharpness-s3-v1'


def launch(image,*,wait=False,resume=False,dlpan=None):
    if (ROOT/'gspread/server.txt').read_text().strip()!='s3':raise ValueError('Launch on s3')
    release=verify(environment=False)
    from panda_gf2_w128 import enqueue as base
    current=base.inspect('container',NAME,missing=True)
    control=ROOT/P.OUTPUT/'launch.json'
    if current:
        if not control.exists():raise ValueError('Existing unowned container; preserve and inspect')
        receipt=read(control);labels=current['Config'].get('Labels') or {}
        if (current['Id']!=receipt['container_id'] or labels.get('panda.source')!=release['source_sha256']
                or current['Config']['Cmd']!=['python','-u','-m','panda_gf2_sharpness','run']):
            raise ValueError('Recorded launch identity differs')
        if current['State']['Running']:return dict(status='ALREADY_RUNNING',container=NAME)
        status_path=ROOT/P.OUTPUT/'worker_status.json'
        if status_path.exists() and read(status_path).get('status')=='COMPLETED':
            return dict(status='ALREADY_COMPLETED',container=NAME)
        if not resume:raise ValueError('Stopped review preserved; use --resume after inspecting logs')
        base.wait_for_gpu(wait);subprocess.run(['docker','start',NAME],check=True)
        return dict(status='RESUMED',container=NAME)
    if resume:raise ValueError('No prior review container to resume')
    if not image:raise ValueError('--image is required for a new review')
    # Use the validated read-only dataset/donor mount builder, with this new output
    # root in this host-only process. Restore the imported module immediately.
    old=base.G
    try:
        base.G=SimpleNamespace(OUTPUT=P.OUTPUT,CAMPAIGN_ID=P.CAMPAIGN_ID)
        cmd,identity=base.command(image,name=NAME,token=P.CAMPAIGN_ID,dlpan=dlpan,
                                  source_sha256=release['source_sha256'])
    finally:base.G=old
    # Original Student and calibration directories may themselves be symlinks.
    insert=[]
    for family in ('panda_sensor_focus','panda_gf2_psnr'):
        path=ROOT/'work_dir'/family
        if not path.is_dir():raise FileNotFoundError(path)
        resolved=path.resolve()
        if any(c in str(resolved) for c in ',\n\r'):raise ValueError('Unsupported mount path')
        insert+=['--mount',f'type=bind,src={resolved},dst=/workspace/work_dir/{family},readonly']
    cmd[-1:-1]=insert
    base.wait_for_gpu(wait)
    result=subprocess.run(cmd+['python','-u','-m','panda_gf2_sharpness','run'],check=True,capture_output=True,text=True)
    value=dict(container_id=result.stdout.strip(),source_sha256=release['source_sha256'],identity=identity,
               command=cmd+['python','-u','-m','panda_gf2_sharpness','run'])
    write(control,value)
    return dict(status='STARTED',container=NAME,container_id=value['container_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image');p.add_argument('--wait',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--dlpan');p.add_argument('--status',action='store_true')
    args=p.parse_args()
    if args.status:
        from panda_gf2_w128.enqueue import inspect
        current=inspect('container',NAME,missing=True);status=ROOT/P.OUTPUT/'worker_status.json'
        print(json.dumps(dict(container=None if current is None else current['State'],worker=read(status) if status.exists() else None),indent=2));return
    print(json.dumps(launch(args.image,wait=args.wait,resume=args.resume,dlpan=args.dlpan),indent=2))


if __name__=='__main__':main()
