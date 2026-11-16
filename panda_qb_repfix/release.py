import argparse
import hashlib
import json
from pathlib import Path
from panda_input3_original.release import verify as previous
from panda_input3 import plan

PATH=Path(__file__).with_name('release.json')


def generate():
    value=dict(schema='QB_REPRESENTATIVE_FALLBACK_RELEASE_V1',previous_release=previous()['source_sha256'],
        files={str(p.relative_to(plan.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PATH.parent.glob('*.py'))})
    return dict(value,source_sha256=plan.digest(value))


def verify():
    expected=json.loads(PATH.read_text())
    if generate()!=expected:raise ValueError('Pull the complete QB representative recovery release')
    return dict(status='VERIFIED',source_sha256=expected['source_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify()))
