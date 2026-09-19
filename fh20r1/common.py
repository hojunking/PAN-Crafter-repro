"""FH20R1 provenance, separate from immutable historical FH12 execution identity."""
from pathlib import Path
from fh12.common import (ROOT,atomic_json,before_deadline,check_deadline,object_sha,
                         read_json,resolved_path,sha256,utcnow)
from fh12.common import source_identity as _legacy_source


def source_identity(root=ROOT):
    root=Path(root)
    source=_legacy_source(root)
    paths=list((root/'fh20r1').glob('*.py'))
    paths+=list((root/'tools').glob('fh20r1_*.py'))
    for path in sorted(paths):
        if path.name.endswith('_tests.py'): continue
        source['files'][str(path.relative_to(root))]=sha256(path)
    source['content_sha256']=object_sha(source['files'])
    source['numeric_method_revision']='FH12_SYNC_FREQ_NATIVE_TEACHER_v1'
    source['consumer_campaign']='WV3_FH20R1_20260919_v1'
    return source


def load_checkpoint_model(cfg,directory,device='cpu',expected_source=None):
    from safetensors.torch import load_file
    from fh12.model import build_model
    directory=Path(directory); identity=read_json(directory/'identity.json')
    path=directory/'model.safetensors'
    if identity.get('model_sha256')!=sha256(path) or identity.get('config_sha256')!=object_sha(cfg):
        raise ValueError('FH20R1 checkpoint bytes/config mismatch')
    if expected_source is not None and identity.get('source_identity')!=expected_source:
        raise ValueError('FH20R1 checkpoint execution release mismatch')
    f=cfg.get('fh20r1',cfg.get('fh12')); m=cfg['model_args']
    state=load_file(str(path)); a={k[8:]:v for k,v in state.items() if k.startswith('aligner.')}
    model,_=build_model(f['input_layout'],m['hidden_size'],m['depth'],cfg['seed'],role=f['role'],
                        teacher_aligner_state=a if f['role']=='S' else None)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),identity
