"""Separate reporting release; original training SHA and resume pins stay exact."""
import argparse
import hashlib
import json
from pathlib import Path
from panda_input3 import plan
from panda_input3.release import verify as verify_training

PATH=Path(__file__).with_name('release.json')
PREVIOUS_REPORTING_RELEASES=('09aade3c536f1ecca2c9f747af8e6a344e64c621a817b11f82356997c285f5a5',)


def generate():
    files={str(p.relative_to(plan.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(PATH.parent.glob('*.py'))}
    value={'schema':'INPUT3_TARGET_REPORTING_RELEASE_V1',
           'training_release':verify_training()['source_sha256'],'files':files}
    return dict(value,source_sha256=plan.digest(value))


def verify():
    expected=json.loads(PATH.read_text())
    if generate()!=expected:
        raise ValueError('Pull the complete INPUT3 target-reporting release')
    return {'status':'VERIFIED','source_sha256':expected['source_sha256'],
            'training_release':expected['training_release'],
            'compatible_reporting_upgrades':list(PREVIOUS_REPORTING_RELEASES)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
