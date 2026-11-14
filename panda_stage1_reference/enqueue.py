"""CPU report launcher; does not interrupt or change running training."""
import argparse,subprocess
from pathlib import Path
from panda_unified_campaign import enqueue as base
from panda_unified_campaign.plan import ROOT

def command(server,sensor,image,dlpan=None):
    # Mount data by sensor, independently of the new s3 QB assignment.
    mapped={'WV3':'s1','QB':'s2','GF2':'s3'}[sensor]
    cmd=base.command(mapped,image,gpu=False,dlpan=dlpan);image_id=cmd.pop()
    for i,item in enumerate(cmd):
        if item=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):cmd[i+1]+=',readonly'
    out=ROOT/'work_dir/panda_stage1_reference';out.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={out},dst=/workspace/work_dir/panda_stage1_reference']
    return cmd+[image_id,'python','-m','panda_stage1_reference','--server',server,'--sensor',sensor]

def main():
    p=argparse.ArgumentParser();p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--sensor',choices=('WV3','QB','GF2'),required=True);p.add_argument('--apply',action='store_true')
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118');p.add_argument('--dlpan')
    a=p.parse_args();subprocess.run(command(a.server,a.sensor,a.image,a.dlpan)+(['--apply'] if a.apply else []),check=True)
if __name__=='__main__':main()
