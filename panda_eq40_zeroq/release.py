"""Pin both the frozen donor implementations and this new shared Stage2 method."""
import argparse,hashlib,json,importlib
from . import plan
PATH=plan.CONFIG/'release.json'
PRIOR=('panda_unified_campaign','panda_wv3_s2_eq40','panda_sensor_eq40')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def generate():
    files={}
    for package in PRIOR:
        p=plan.ROOT/'config'/package/'release.json';r=json.loads(p.read_text())
        files.update(r['files']);files[str(p.relative_to(plan.ROOT))]=sha(p)
    for folder in ('panda_eq40_zeroq','panda_fr_columns'):
        for p in (plan.ROOT/folder).rglob('*.py'):files[str(p.relative_to(plan.ROOT))]=sha(p)
    for p in plan.CONFIG.glob('*.json'):
        if p!=PATH:files[str(p.relative_to(plan.ROOT))]=sha(p)
    value={'schema':'PANDA_ZEROQ_SOURCE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))
def verify(*,environment=False):
    expected=json.loads(PATH.read_text())
    if expected['source_sha256']!=plan.digest({k:v for k,v in expected.items() if k!='source_sha256'}):raise ValueError('Release digest mismatch')
    changed=[p for p,h in expected['files'].items() if not (plan.ROOT/p).is_file() or sha(plan.ROOT/p)!=h]
    if changed:raise ValueError('Pull complete ZEROQ release: '+repr(changed))
    for package in PRIOR:importlib.import_module(package+'.release').verify(environment=environment)
    return dict(status='VERIFIED',source_sha256=expected['source_sha256'],file_count=len(expected['files']),environment_checked=environment)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
