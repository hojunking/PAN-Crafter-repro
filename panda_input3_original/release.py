"""Reporting-only release; retains original and C30 training/resume source pins."""
import argparse
import hashlib
import json
from pathlib import Path
from panda_input3_c30.release import verify as verify_training
from panda_input3_c30 import recipes

PATH = Path(__file__).with_name('release.json')


def generate():
    value = dict(schema='INPUT3_ORIGINAL_PAN_REPORTING_RELEASE_V1',
        training_release=verify_training()['source_sha256'],
        files={str(p.relative_to(recipes.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(PATH.parent.glob('*.py'))})
    return dict(value, source_sha256=recipes.original.digest(value))


def verify():
    expected = json.loads(PATH.read_text())
    if generate() != expected: raise ValueError('Pull the complete original-PAN reporting release')
    return dict(status='VERIFIED', source_sha256=expected['source_sha256'],
                training_release=expected['training_release'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--write', action='store_true'); a = p.parse_args()
    if a.write: PATH.write_text(json.dumps(generate(), indent=2) + '\n')
    print(json.dumps(verify()))
