"""New campaign manifest; all existing main/Stage1 releases remain immutable."""
import argparse
import hashlib
import json
from . import plan
from panda_eq40_zeroq.release import verify as verify_base

PATH=plan.CONFIG/'release.json'
def generate():
    base=json.loads((plan.main.CONFIG/'release.json').read_text())
    files=dict(base['files'])
    files[str((plan.main.CONFIG/'release.json').relative_to(plan.ROOT))]=hashlib.sha256((plan.main.CONFIG/'release.json').read_bytes()).hexdigest()
    for p in sorted((plan.ROOT/'panda_wv3_s2abl5').glob('*.py'))+sorted(plan.CONFIG.glob('*.json')):
        if p!=PATH:files[str(p.relative_to(plan.ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    value={'schema':'PANDA_S2ABL5_RELEASE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))
def verify(environment=False):
    verify_base(environment=environment)
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete S2ABL5 release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256'],'file_count':len(expected['files'])}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
