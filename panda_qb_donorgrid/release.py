"""Additive release; existing active campaigns and their source pins stay unchanged."""
import argparse
import hashlib
import json
from . import plan
from panda_qb_lateval.release import verify as verify_base

PATH=plan.CONFIG/'release.json'

def generate():
    base=plan.ROOT/'config/panda_qb_lateval/release.json'
    files=dict(json.loads(base.read_text())['files'])
    files[str(base.relative_to(plan.ROOT))]=hashlib.sha256(base.read_bytes()).hexdigest()
    for p in sorted((plan.ROOT/'panda_qb_donorgrid').glob('*.py'))+sorted(plan.CONFIG.glob('*')):
        if p.is_file() and p!=PATH:
            files[str(p.relative_to(plan.ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    value={'schema':'PANDA_QB_DONORGRID_RELEASE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))

def verify(environment=False):
    verify_base(environment=environment)
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete QB DONORGRID release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256'],'file_count':len(expected['files'])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
