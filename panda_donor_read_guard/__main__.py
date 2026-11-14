import argparse
import importlib
import json
import sys
from .guard import install,POLICY
from .release import identity

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',choices=('panda_wv3_s2abl5','panda_sensor_main50'),required=True)
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    a=p.parse_args()
    frozen=importlib.import_module(a.package+'.release').verify()
    root=install(a.package,a.server)
    from directshift48.reporting import atomic_json,utc_now
    atomic_json(root/'donor_read_guard/active.json',{'policy':POLICY,'at':utc_now(),
        'overlay':identity(),'frozen_training_release':frozen,
        'config_optimizer_and_existing_checkpoints_preserved':True})
    print('Installed donor read guard; frozen training source unchanged',flush=True)
    sys.argv=[a.package,'run','--server',a.server]
    importlib.import_module(a.package+'.__main__').main()

if __name__=='__main__':main()
