"""New campaign output namespace; old numerical paths and assets are read-only."""
from pathlib import Path
import json
import os
from fh12.common import (ROOT, atomic_json, before_deadline, check_deadline, object_sha,
                         read_json, resolved_path, sha256, utcnow)
from l100.common import locked
from gfb20.plan import CAMPAIGN_ID, METHOD_REVISION, verify_lane


def read(path, default=None):
    return read_json(path) if Path(path).is_file() else ({} if default is None else default)


def camp(root, server):
    verify_lane(server)
    return Path(root) / 'work_dir/_gfb20' / server


def run_dir(identifier, root=ROOT):
    from gfb20.plan import case_for
    return Path(root) / 'work_dir' / case_for(identifier).run_id


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value: raise ValueError('Immutable GFB20 artifact changed: '+str(path))
    else: atomic_json(path,value)
    return path


def read_config(path):
    text = Path(path).read_text()
    try: return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


def apply_runtime_policy(root=ROOT):
    import torch
    path = Path(root)/'gfb20/runtime_policy.json'
    p = read_json(path)
    if p['precision'] != 'float32' or p['mixed_precision'] != 'no':
        raise ValueError('GFB20 requires FP32 and AMP OFF')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=p['tf32_matmul']
    torch.backends.cudnn.allow_tf32=p['tf32_cudnn']
    torch.backends.cudnn.benchmark=p['cudnn_benchmark']
    torch.backends.cudnn.deterministic=p['cudnn_deterministic']
    torch.use_deterministic_algorithms(p['deterministic_algorithms'])
    return dict(policy=p,sha256=sha256(path))


def source_identity(root=ROOT):
    from l100.common import source_identity as base
    root=Path(root)
    result=base(root)
    from gfb20.plan import SOURCE_SHAS
    paths=list((root/'gfb20').glob('*.py'))+list((root/'tools').glob('gfb20_*'))
    paths += [root/'gfb20/runtime_policy.json']+[root/name for name in SOURCE_SHAS]
    for path in paths:
        if path.is_file() and not path.name.startswith('test_'):
            result['files'][str(path.relative_to(root))]=sha256(path)
    result.update(content_sha256=object_sha(result['files']),consumer_campaign=CAMPAIGN_ID,
                  numeric_method_revision=METHOD_REVISION,runtime_policy_sha256=sha256(root/'gfb20/runtime_policy.json'))
    return result


def append_event(path,event,**details):
    row=dict(campaign_id=CAMPAIGN_ID,event=event,at_utc=utcnow(),**details)
    path=Path(path)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    return row


class RuntimePaused(RuntimeError): pass
