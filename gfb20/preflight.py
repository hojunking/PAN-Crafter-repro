"""Local B20 parent/data/numerical admission; no prior-campaign writes."""
import io
from pathlib import Path
import unittest
from gfb20.common import (ROOT,camp,read,read_json,atomic_json,immutable_json,
    source_identity,apply_runtime_policy,check_deadline,object_sha,sha256,utcnow)
from gfb20.plan import verify_lane,verify_sources,PARENT_ASSETS


def method_checks():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'gfb20'),'test_*.py',top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    return dict(complete=result.wasSuccessful() and result.testsRun>0,tests_run=result.testsRun,output=stream.getvalue())


def bind_assets(root,server,bindings=None,deadline=None):
    from gfb20.assets import discover_bindings,bind_parent
    verify_lane(server)
    discovered=discover_bindings(server,root)
    explicit=read_json(bindings) if isinstance(bindings,(str,Path)) else (bindings or {})
    explicit=explicit.get('parents',explicit)
    if any(k not in discovered for k in explicit): raise ValueError('Nonlocal/unregistered parent in binding map')
    discovered.update(explicit)
    result={}
    data=None
    for parent_id,binding in discovered.items():
        check_deadline(deadline)
        try:
            parent=bind_parent(parent_id,server,binding,root,local_dataset_manifest=data)
        except FileNotFoundError as exc:
            result[parent_id]=dict(status='BLOCKED_MISSING_PARENT',reason=str(exc));continue
        except (ValueError,RuntimeError) as exc:
            result[parent_id]=dict(status='BLOCKED_INTEGRITY',reason=str(exc));continue
        data=parent['native_reference']['data'] if data is None else data
        path=immutable_json(camp(root,server)/'parents'/f'{parent_id}.json',parent)
        ref=parent['native_reference']
        refpath=immutable_json(camp(root,server)/'references'/f'{ref["native_reference_id"]}.json',ref)
        result[parent_id]=dict(status='BOUND',manifest=str(path),reference_manifest=str(refpath),
                               parent_model_sha256=parent['parent_model_sha256'])
    atomic_json(camp(root,server)/'parent_bindings.json',result)
    if data is None: raise ValueError('No verified local parent: see parent_bindings.json; no fallback Teacher allowed')
    from gfb20.native_data import bind_lane_data
    lane,provenance=bind_lane_data(data,server,root)
    immutable_json(camp(root,server)/'dataset_manifest.json',lane)
    immutable_json(camp(root,server)/'dataset_wrapper_provenance.json',provenance)
    return result,lane


def verify_native_data(root,server,data,deadline=None):
    from l100.data import validate_manifest,validate_source_bindings
    from qg40.bootstrap import default_sensor_spec
    from g20.data import _scan_source,verify_lp_cache
    validate_manifest(data,server)
    spec=default_sensor_spec(root,'GF2')
    validate_source_bindings(data,spec)
    for split,item in data['splits'].items():
        check_deadline(deadline)
        for p,h in (('dataroot','sha256'),('lpan_path','lpan_sha256')):
            if sha256(item[p])!=item[h]: raise ValueError('Original native data/cache bytes changed')
        scan=_scan_source(item['dataroot'],split,spec,deadline)
        if scan['count']!=item['count']: raise ValueError('Native source count differs')
        evidence=verify_lp_cache(item['dataroot'],item['lpan_path'],item['sha256'],deadline)
        if any(item.get(k)!=v for k,v in evidence.items()): raise ValueError('Native LP correspondence differs')
    receipt=dict(complete=True,dataset_manifest_sha256=object_sha(data),full_lp_verified=True,
                 previous_campaign_written=False,source_catalog_sha256=sha256(Path(root)/'qg40/sensor_sources.json'))
    atomic_json(camp(root,server)/'data_verification.json',receipt)
    return receipt


def execute(root=ROOT,server='s3',deadline=None,bindings=None,device='cuda'):
    import torch
    verify_lane(server);verify_sources(root);check_deadline(deadline)
    runtime=apply_runtime_policy(root)
    if device!='cuda' or not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
        raise ValueError('GFB20 production requires the assigned s3-s5 RTX5090; no CPU/4090 fallback')
    release=source_identity(root)
    folder=camp(root,server)
    old=read(folder/'preflight.json')
    if old and old['source_identity']!=release: raise ValueError('GFB20 source/runtime changed after preflight')
    parents,data=bind_assets(root,server,bindings,deadline)
    verify_native_data(root,server,data,deadline)
    checks=method_checks();atomic_json(folder/'method_regression.json',checks)
    if not checks['complete']: raise ValueError('GFB20 numerical regression failed')
    from g20.data import build_dataset
    from gfb20.evaluation import FRMetrics
    FRMetrics(build_dataset(data,'fr',root=root))
    from gfb20.assets import load_reference,load_parent
    first=next(v for v in parents.values() if v['status']=='BOUND')
    teacher,q,reference=load_reference(first['reference_manifest'],device,root,deadline=deadline)
    model,parent=load_parent(first['manifest'],device,root)
    train=build_dataset(data,'train',root=root)
    row=train.base(0)
    with torch.no_grad(): out=model(row[4][None].to(device),row[2][None].to(device),row[3][None].to(device))
    if out['y'].shape!=row[0][None].shape or not torch.isfinite(out['y']).all():
        raise ValueError('Real parent native forward failed')
    del teacher,q,reference,model,parent,train,out
    torch.cuda.empty_cache()
    value=dict(complete=True,server=server,source_identity=release,runtime=runtime,
        dataset_manifest_sha256=object_sha(data),parents=parents,method_tests=checks['tests_run'],
        gpu=torch.cuda.get_device_name(0),native_reference_online_parity=True,real_native_forward=True,
        previous_campaign_written=False,at_utc=utcnow())
    atomic_json(folder/'preflight.json',value)
    return value
