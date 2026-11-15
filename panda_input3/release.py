"""Additive release covering both WV3 ablation and sensor/QB lineage."""
import argparse
import hashlib
import json
from . import plan
from panda_qb_donorgrid.release import verify as verify_grid
from panda_wv3_s2abl5.release import verify as verify_wv3

PATH=plan.CONFIG/'release.json'

def generate():
    files={}
    for name in ('panda_qb_donorgrid','panda_wv3_s2abl5'):
        path=plan.ROOT/'config'/name/'release.json'
        for key,value in json.loads(path.read_text())['files'].items():
            if key in files and files[key]!=value:raise ValueError('Source release conflict: '+key)
            files[key]=value
        files[str(path.relative_to(plan.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p in sorted((plan.ROOT/'panda_input3').glob('*.py'))+sorted(plan.CONFIG.glob('*')):
        if p.is_file() and p!=PATH:files[str(p.relative_to(plan.ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    value={'schema':'PANDA_INPUT3_RELEASE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))

def verify(environment=False):
    verify_grid(environment=environment);verify_wv3(environment=False)
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete INPUT3 release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256'],'file_count':len(expected['files'])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
