import argparse,hashlib,json
from . import plan
from panda_sensor_main50.release import verify as verify_base
PATH=plan.CONFIG/'release.json'
def generate():
    base=plan.ROOT/'config/panda_sensor_main50/release.json'
    files=dict(json.loads(base.read_text())['files']);files[str(base.relative_to(plan.ROOT))]=hashlib.sha256(base.read_bytes()).hexdigest()
    for package in ('panda_qb_s3_parallel','panda_stage1_reference','panda_donor_read_guard'):
        for p in sorted((plan.ROOT/package).glob('*.py')):files[str(p.relative_to(plan.ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    value={'schema':'PANDA_QB_S3_PARALLEL_RELEASE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))
def verify(environment=False):
    verify_base(environment=environment)
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete s3 QB/reporting release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256'],'file_count':len(expected['files'])}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.parent.mkdir(parents=True,exist_ok=True);PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
