"""Additive wrapper release; all prior training/checkpoint pins remain unchanged."""
import argparse
import hashlib
import json
from panda_input3_original.release import verify as verify_reporting
from panda_input3.release import verify as verify_training
from panda_qb_s3_parallel.release import verify as verify_donor
from panda_input3_repeat.release import verify as verify_predecessor
from . import plan

PATH = plan.CONFIG / 'release.json'


def generate():
    value = dict(schema='QB_S3_INPUT_C3_RELEASE_V1',
        input_release=verify_reporting()['source_sha256'], donor_release=verify_donor()['source_sha256'],
        predecessor_release=verify_predecessor()['source_sha256'],
        files={str(p.relative_to(plan.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((plan.ROOT/'panda_qb_s3_input_c3').glob('*.py')) + sorted(plan.CONFIG.glob('*'))
               if p.is_file() and p!=PATH})
    return dict(value, source_sha256=plan.digest(value))


def verify(environment=False):
    verify_training(environment=environment)
    expected = json.loads(PATH.read_text())
    if generate() != expected: raise ValueError('Pull the complete s3 QB C3 release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256']}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
