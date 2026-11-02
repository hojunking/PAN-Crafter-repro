"""Portable source/environment checks for a complete s1/s3 checkout."""
import argparse
import json
from ta2.evaluation import file_sha
from . import plan

PATH=plan.CONFIG/'release.json'


def generate():
    from panda_tune30.runtime import code_pin
    files=code_pin()
    extra=list((plan.ROOT/'panda_wv3_ablation').rglob('*.py'))
    extra+=list(plan.CONFIG.glob('*.json'))+list(plan.CONFIG.glob('*.csv'))
    extra+=[plan.ROOT/'research_log/gspread_rules/PAN_FIXED64_V1.schema.json']
    files.update({str(p.relative_to(plan.ROOT)):file_sha(p) for p in extra if p!=PATH})
    value={'schema':'WV3_COMPONENT_SOURCE_RELEASE_V1','files':dict(sorted(files.items()))}
    return dict(value,source_sha256=plan.digest(value))


def verify(*,environment=False):
    expected=json.loads(PATH.read_text());actual=generate()
    if expected!=actual:
        changed=[p for p in set(expected['files'])|set(actual['files']) if expected['files'].get(p)!=actual['files'].get(p)]
        raise ValueError('Incomplete/changed ablation source checkout: '+', '.join(sorted(changed)))
    if environment:
        import numpy as np
        import torch
        from panda_stage2_condq.training import configure_precision
        from ta2.evaluation import evaluator_identity
        configure_precision()
        observed={'torch':torch.__version__.split('+')[0],'numpy':np.__version__,
            'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'training_policy':{'amp':False,'tf32':torch.backends.cuda.matmul.allow_tf32,
                               'cudnn_benchmark':torch.backends.cudnn.benchmark}}
        if observed!=json.loads((plan.CONFIG/'reference_environment.json').read_text()):
            raise ValueError('Runtime differs from the pinned reference: '+repr(observed))
        if evaluator_identity()!=json.loads((plan.CONFIG/'reference_evaluator.json').read_text()):
            raise ValueError('Official evaluator differs from the anchored reference')
    return {'status':'VERIFIED','source_sha256':actual['source_sha256'],
            'file_count':len(actual['files']),'environment_checked':environment}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true');parser.add_argument('--environment',action='store_true')
    args=parser.parse_args()
    if args.write:PATH.write_text(json.dumps(generate(),indent=2)+'\n')
    print(json.dumps(verify(environment=args.environment)))
