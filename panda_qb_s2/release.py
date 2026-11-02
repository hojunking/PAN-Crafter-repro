"""Portable source/environment verification; no training, CUDA allocation or cloud writes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'config/panda_qb_s2/release.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def generate():
    from .runtime import code_pin
    value = {'schema': 'PANDA_QB_S2_RELEASE_V1', 'files': code_pin()}
    return dict(value, source_sha256=digest(value))


def verify(*, check_environment=False):
    expected = json.loads(PATH.read_text())
    actual = generate()
    if expected != actual:
        changed = sorted(k for k in set(expected['files']) | set(actual['files'])
                         if expected['files'].get(k) != actual['files'].get(k))
        raise ValueError('s2 release source differs; use the complete committed checkout: ' + ', '.join(changed))
    if check_environment:
        import numpy as np
        import torch
        from ta2.evaluation import evaluator_identity
        from panda_stage2_condq.training import configure_precision
        configure_precision()
        reference = json.loads((ROOT / 'config/panda_qb_s2/reference_environment.json').read_text())
        measured = {'torch':torch.__version__.split('+')[0], 'numpy':np.__version__,
                    'cuda_build':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
                    'training_policy':{'amp':False, 'tf32':torch.backends.cuda.matmul.allow_tf32,
                                       'cudnn_benchmark':torch.backends.cudnn.benchmark}}
        if measured != reference:
            raise ValueError(f's2 runtime differs from the reference environment: {measured!r}')
        ref_eval = json.loads((ROOT / 'config/panda_qb_s2/reference_evaluator.json').read_text())
        if evaluator_identity() != ref_eval:
            raise ValueError('s2 evaluator or metric-library versions differ from reference')
    return {'status':'RELEASE_VERIFIED', 'source_sha256':actual['source_sha256'],
            'file_count':len(actual['files']), 'environment_checked':check_environment}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Developer: regenerate only after validation, then commit it')
    parser.add_argument('--environment',action='store_true')
    args=parser.parse_args()
    if args.write:
        PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify(check_environment=args.environment)))


if __name__=='__main__':
    main()
