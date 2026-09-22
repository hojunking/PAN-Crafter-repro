"""P40 identities without loading or changing archived campaign registries."""
import json
import os
from pathlib import Path
from fh12.common import (ROOT,atomic_json,before_deadline,check_deadline,object_sha,
    read_json,resolved_path,sha256,utcnow)
from qg40.common import locked
from gfp40.plan import CAMPAIGN_ID,METHOD_REVISION,verify_lane,SOURCE_SHAS

def read(path,default=None):return read_json(path) if Path(path).is_file() else ({} if default is None else default)
def camp(root,server):
    verify_lane(server)
    return Path(root)/'work_dir/_gfp40'/server
def run_dir(identifier,root=ROOT):
    from gfp40.plan import case_for
    return Path(root)/'work_dir'/case_for(identifier).run_id
def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Immutable P40 artifact changed: '+str(path))
    else:atomic_json(path,value)
    return path
def read_config(path):
    text=Path(path).read_text()
    try:return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)
def apply_runtime_policy(root=ROOT):
    import torch
    p=read_json(Path(root)/'gfp40/runtime_policy.json')
    if p['precision']!='float32' or p['mixed_precision']!='no':raise ValueError('FP32 only')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=p['tf32_matmul']
    torch.backends.cudnn.allow_tf32=p['tf32_cudnn']
    torch.backends.cudnn.benchmark=p['cudnn_benchmark']
    torch.backends.cudnn.deterministic=p['cudnn_deterministic']
    torch.use_deterministic_algorithms(p['deterministic_algorithms'])
    return dict(policy=p,sha256=sha256(Path(root)/'gfp40/runtime_policy.json'))
def source_identity(root=ROOT):
    import torch
    from qg40.common import source_identity as base
    root=Path(root);result=base(root)
    names=['g20/data.py','g20/model.py','g20/evaluation.py','g20/plan.py','g20/losses.py',
        'g20/postrun.py','g20/common.py','l100/data.py','l100/common.py','l100/plan.py','l100/references.py',
        'l100/runtime_policy.json','reporting_extra/sensor_sheet.py','reporting_extra/sensor_layout.py',
        'reporting_extra/sensor_backfill.py','gspread/gspread_upload.py','gspread/sheet_categories.py',
        'gfp40/runtime_policy.json','gfp40/parent_registry.json']+list(SOURCE_SHAS)
    paths=[root/n for n in names]+list((root/'gfp40').glob('*.py'))+list((root/'tools').glob('gfp40_*'))
    for path in paths:
        if not path.name.startswith('test_'):result['files'][str(path.relative_to(root))]=sha256(path)
    result.update(content_sha256=object_sha(result['files']),consumer_campaign=CAMPAIGN_ID,
        numeric_method_revision=METHOD_REVISION,runtime_policy_sha256=sha256(root/'gfp40/runtime_policy.json'),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return result
def append_event(path,event,**details):
    row=dict(campaign_id=CAMPAIGN_ID,event=event,at_utc=utcnow(),**details);path=Path(path)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return row
