"""Portable immutable source closure and the unchanged WV3 numerical environment."""
import argparse
import hashlib
import json
from . import plan

PATH = plan.CONFIG / 'release.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    from panda_wv3_main_rgb.release import generate as original_release
    files = dict(original_release()['files'])
    extra = list((plan.ROOT / 'panda_wv3_hyper').rglob('*.py'))
    extra += list(plan.CONFIG.glob('*.json')) + list(plan.CONFIG.glob('*.csv'))
    extra += [plan.ROOT/p for p in (
        'edgeres48/__init__.py', 'edgeres48/docker_mounts.py', 'Dockerfile.env', 'requirements.txt',
        'config/panda_wv3_main_rgb/release.json', 'config/panda_wv3_ablation_rgb/release.json',
        'tools/eval_dlpan.py',
        'panda_band30/tests/__init__.py', 'panda_band30/tests/test_method.py',
        'panda_stage2_condq/tests/__init__.py', 'panda_stage2_condq/tests/test_method.py',
        'panda_stage2_condq/tests/test_donor_sources.py')]
    extra += list((plan.ROOT/'tools/metrics').glob('*.py'))
    extra += list((plan.ROOT/'model').rglob('*.py'))
    files.update({str(p.relative_to(plan.ROOT)): sha(p) for p in extra if p != PATH})
    value = {'schema': 'WV3_HYPER_SOURCE_RELEASE_V1', 'files': dict(sorted(files.items()))}
    return dict(value, source_sha256=plan.digest(value))


def verify(*, environment=False):
    expected = json.loads(PATH.read_text())
    if expected['source_sha256'] != plan.digest({k:v for k,v in expected.items() if k!='source_sha256'}):
        raise ValueError('Source manifest digest mismatch')
    changed = [p for p, value in expected['files'].items()
               if not (plan.ROOT/p).is_file() or sha(plan.ROOT/p) != value]
    if changed:
        raise ValueError('Pull the complete committed hyperparameter release: ' + ', '.join(changed))
    if environment:
        import numpy as np
        import torch
        from panda_stage2_condq.training import configure_precision
        from ta2.evaluation import evaluator_identity
        configure_precision()
        observed = dict(torch=torch.__version__.split('+')[0], numpy=np.__version__,
            cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            training_policy=dict(amp=False, tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_benchmark=torch.backends.cudnn.benchmark))
        if observed != json.loads((plan.CONFIG/'reference_environment.json').read_text()):
            raise ValueError('Use the reference Dockerfile.env environment: ' + repr(observed))
        if evaluator_identity() != json.loads((plan.CONFIG/'reference_evaluator.json').read_text()):
            raise ValueError('Official evaluator differs from the anchored reference')
    return dict(status='VERIFIED', source_sha256=expected['source_sha256'],
                file_count=len(expected['files']), environment_checked=environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--environment', action='store_true')
    args = parser.parse_args()
    if args.write:
        PATH.write_text(json.dumps(generate(), indent=2) + '\n')
    print(json.dumps(verify(environment=args.environment)))
