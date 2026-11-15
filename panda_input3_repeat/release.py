"""Additive release: retain every existing training and publication source pin."""
import argparse
import hashlib
import json
from pathlib import Path
from panda_input3_original.release import verify as verify_reporting
from panda_input3.release import verify as verify_training
from . import recipes

PATH = Path(__file__).with_name('release.json')


def generate():
    value = dict(schema='INPUT3_C30_R2_RELEASE_V1',
        previous_release=verify_reporting()['source_sha256'],
        files={str(p.relative_to(recipes.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(PATH.parent.glob('*.py'))})
    return dict(value, source_sha256=recipes.previous.original.digest(value))


def verify(environment=False):
    verify_training(environment=environment)
    expected = json.loads(PATH.read_text())
    if generate() != expected:
        raise ValueError('Pull the complete INPUT3-C30 R2 release')
    return dict(status='VERIFIED', source_sha256=expected['source_sha256'],
                previous_release=expected['previous_release'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--write', action='store_true'); a = p.parse_args()
    if a.write: PATH.write_text(json.dumps(generate(), indent=2) + '\n')
    print(json.dumps(verify()))
