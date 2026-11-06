"""Verify the complete portable source closure and the original numerical environment."""
import argparse
import hashlib
import json
from pathlib import Path
from panda_sensor_visuals.common import ROOT,read,write,sha

PATH=ROOT/'config/panda_gf2_sharpness/release.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def generate():
    # The validated GF2 package already includes the original focus/PSNR engines,
    # metrics, data adapters and their transitive dependencies.
    files=dict(read(ROOT/'config/panda_gf2_w128/release.json')['files'])
    for package in ('panda_gf2_sharpness','panda_sensor_visuals'):
        for path in (ROOT/package).rglob('*'):
            if path.is_file() and 'tests' not in path.parts and path.suffix in ('.py','.md','.txt'):
                files[str(path.relative_to(ROOT))]=sha(path)
    for path in (ROOT/'config/panda_gf2_sharpness').glob('*.json'):
        if path.name!='release.json':files[str(path.relative_to(ROOT))]=sha(path)
    for path in (ROOT/'config/panda_gf2_w128').glob('reference_*.json'):
        files[str(path.relative_to(ROOT))]=sha(path)
    value=dict(schema='PANDA_GF2_SHARPNESS_RELEASE_V1',files=dict(sorted(files.items())))
    return dict(value,source_sha256=digest(value))


def verify(*,environment=False):
    expected=read(PATH)
    if expected['source_sha256']!=digest({k:v for k,v in expected.items() if k!='source_sha256'}):
        raise ValueError('Release manifest digest differs')
    changed=[path for path,value in expected['files'].items() if not (ROOT/path).is_file() or sha(ROOT/path)!=value]
    if changed:raise ValueError('Incomplete/different source checkout: '+', '.join(changed))
    if environment:
        import numpy as np
        import torch
        from panda_stage2_condq.training import configure_precision
        from ta2.evaluation import evaluator_identity
        configure_precision()
        measured=dict(torch=torch.__version__.split('+')[0],numpy=np.__version__,cuda_build=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),training_policy=dict(amp=False,
                tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark))
        if measured!=read(ROOT/'config/panda_gf2_w128/reference_environment.json'):
            raise ValueError('Use Dockerfile.env reference environment: '+repr(measured))
        if evaluator_identity()!=read(ROOT/'config/panda_gf2_w128/reference_evaluator.json'):
            raise ValueError('Original metric implementation/environment differs')
    return dict(status='VERIFIED',source_sha256=expected['source_sha256'],files=len(expected['files']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--environment',action='store_true');a=p.parse_args()
    if a.write:write(PATH,generate())
    print(json.dumps(verify(environment=a.environment)))
