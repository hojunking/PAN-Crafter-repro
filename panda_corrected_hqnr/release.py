"""Independent control-policy manifest; the numerical ZEROQ release stays pinned."""
import hashlib
import json
from pathlib import Path
from panda_eq40_zeroq import plan
from panda_eq40_zeroq.release import verify as verify_training

PATH = plan.ROOT / 'config/panda_corrected_hqnr/release.json'


def generate():
    files = {str(p.relative_to(plan.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob('*.py'))}
    data = {'schema': 'CORRECTED_HQNR_CONTROL_RELEASE_V1', 'files': files,
            'training_release': verify_training()['source_sha256']}
    return dict(data, source_sha256=plan.digest(data))


def verify(environment=False):
    base = verify_training(environment=environment)
    expected = json.loads(PATH.read_text())
    if generate() != expected:
        raise ValueError('Pull the complete corrected-HQNR control release')
    return {'policy_release': expected['source_sha256'], 'training_release': base['source_sha256']}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--write', action='store_true'); a = p.parse_args()
    if a.write:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(generate(), indent=2) + '\n')
    print(json.dumps(verify()))
