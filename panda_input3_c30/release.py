"""Additive release; never change the source pins of the running 40/50K run."""
import argparse
import hashlib
import json
from pathlib import Path
from panda_input3.release import verify as verify_training
from panda_input3_targets.release import verify as verify_reporting
from . import recipes

PATH = Path(__file__).with_name('release.json')


def generate():
    files = {str(p.relative_to(recipes.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PATH.parent.glob('*.py'))}
    value = {'schema': 'INPUT3_C30_RELEASE_V1',
             'training_release': verify_training()['source_sha256'],
             'reporting_release': verify_reporting()['source_sha256'], 'files': files}
    return dict(value, source_sha256=recipes.original.digest(value))


def verify(environment=False):
    verify_training(environment=environment)
    expected = json.loads(PATH.read_text())
    if generate() != expected:
        raise ValueError('Pull the complete INPUT3-C30 release')
    return {k: expected[k] for k in ('source_sha256', 'training_release', 'reporting_release')} | {'status': 'VERIFIED'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--write', action='store_true'); a = p.parse_args()
    if a.write: PATH.write_text(json.dumps(generate(), indent=2) + '\n')
    print(json.dumps(verify()))
