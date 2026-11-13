"""Portable immutable source closure and the unchanged WV3 numerical environment."""
import argparse
import hashlib
import json
from . import plan

PATH = plan.CONFIG / 'release.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    """Pin the local import closure plus declared non-code data dependencies.

    Pin original unified donor/runtime dependencies and the immutable WV3
    corrected-only utility, including their manifests and imported configs.
    """
    import ast
    from pathlib import Path
    pending=list((plan.ROOT/'panda_unified').rglob('*.py'))+list((plan.ROOT/'panda_sensor_eq40').rglob('*.py'))
    pending+=list((plan.ROOT/'model').rglob('*.py'))+list((plan.ROOT/'tools/metrics').glob('*.py'))
    pending+=[plan.ROOT/'tools/eval_dlpan.py',plan.ROOT/'edgeres48/docker_mounts.py']
    visited=set()
    def add_module(name):
        parts=name.split('.')
        for n in range(1,len(parts)+1):
            folder=plan.ROOT.joinpath(*parts[:n]);init=folder/'__init__.py'
            if init.is_file():pending.append(init)
        file=plan.ROOT.joinpath(*parts).with_suffix('.py')
        if file.is_file():pending.append(file)
    while pending:
        path=pending.pop()
        if path in visited:continue
        visited.add(path)
        module=path.relative_to(plan.ROOT).with_suffix('').parts
        package=module[:-1]
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:add_module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(package[:len(package)-node.level+1]) if node.level else []
                if node.module:prefix+=node.module.split('.')
                name='.'.join(prefix)
                if name:add_module(name)
                for alias in node.names:
                    if alias.name!='*':add_module('.'.join(prefix+[alias.name]))
    # Training registries and historical modules with import-time configs.
    for old in ('panda_wv3_ablation','panda_qb_s2_frozen'):
        manifest=json.loads((plan.ROOT/'config'/old/'release.json').read_text())
        visited.update(plan.ROOT/p for p in manifest['files']
            if not p.endswith('.py') and not p.endswith('/release.json'))
    visited.update((plan.ROOT/'config/panda_unified_campaign').glob('*.json'));visited.update((plan.ROOT/'config/panda_unified_campaign').glob('*.csv'))
    visited.update(plan.CONFIG.glob('*.json'));visited.update(plan.CONFIG.glob('*.csv'))
    visited.update(plan.ROOT/p for p in ('Dockerfile.env','requirements.txt','ablr2/sensor_sources.json',
        'config/bglobal_rr/qb_original_lms.json'))
    for package in ('panda_unified_campaign','panda_wv3_s2_eq40'):
        previous=plan.ROOT/'config'/package/'release.json'
        visited.add(previous)
        visited.update(plan.ROOT/f for f in json.loads(previous.read_text())['files'])
    visited.discard(PATH)
    missing=[str(p) for p in visited if not p.is_file()]
    if missing:raise ValueError('Missing source/data dependencies: '+repr(missing))
    files={str(p.relative_to(plan.ROOT)):sha(p) for p in sorted(visited)}
    value={'schema':'PANDA_SENSOR_EQ40_SOURCE_RELEASE_V1','files':files}
    return dict(value,source_sha256=plan.digest(value))


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
