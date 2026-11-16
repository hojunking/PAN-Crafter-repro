"""Additive source release; retain all source-training and recovery digests."""
import argparse
import hashlib
import json
from panda_input3.release import verify as verify_training
from panda_input3_repeat.release import verify as verify_previous
from panda_qb_repfix.release import verify as verify_repfix
from . import plan

PATH=plan.CONFIG/'release.json'


def generate():
    files={str(p.relative_to(plan.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((plan.ROOT/'panda_qb_s2_donor_c').glob('*.py'))+sorted(plan.CONFIG.glob('*'))
           if p.is_file() and p!=PATH}
    value=dict(schema='QB_S2_DONOR_C_RELEASE_V1',training_release=verify_training()['source_sha256'],
        previous_release=verify_previous()['source_sha256'],reporting_release=verify_repfix()['source_sha256'],files=files)
    return dict(value,source_sha256=plan.digest(value))


def verify(environment=False):
    verify_training(environment=environment)
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete s2 donor-C release')
    return dict(status='VERIFIED',source_sha256=expected['source_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:
        PATH.parent.mkdir(parents=True,exist_ok=True);PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
