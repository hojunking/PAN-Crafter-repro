"""Local P40 parent/data/source admission; no prior-campaign writes."""
import difflib
import io
from pathlib import Path
import unittest
from gfp40.common import (ROOT,camp,read,read_json,atomic_json,immutable_json,
    source_identity,apply_runtime_policy,check_deadline,object_sha,sha256,utcnow)
from gfp40.plan import verify_lane,verify_sources


def audit_b20_source(root, server, archive=None):
    """Authenticate actual B20 checkout bytes, never equate commit names.

    Existing B20 preflight/source receipts are read-only. An explicit archived
    source identity permits an extracted archive instead of a Git worktree.
    """
    from gfp40.assets import NUMERICAL_PATHS, source_compatibility
    root = Path(root)
    old_lane = root / 'work_dir/_gfb20' / server
    deployment = read(old_lane / 'runtime_release.json')
    archive = archive or {}
    archive_root = Path(archive.get('root') or deployment.get('path') or '')
    if not archive_root.is_absolute() or not archive_root.is_dir():
        raise ValueError('BLOCKED_INTEGRITY: preserve B20 frozen checkout, or bind b20_source_archive.root')
    identity_path = Path(archive.get('source_identity_path') or old_lane / 'preflight.json')
    evidence = read_json(identity_path)
    origin = evidence.get('source_identity', evidence)
    if not origin.get('files') or object_sha(origin['files']) != origin.get('content_sha256'):
        raise ValueError('B20 source identity is absent or its content SHA differs')
    mismatched = []
    verified = {}
    for name, digest in origin['files'].items():
        if name.startswith('external/'):
            continue
        path = archive_root / name
        if not path.is_file() or sha256(path) != digest:
            mismatched.append(name)
        else:
            verified[name] = digest
    if mismatched:
        raise ValueError('B20 archive bytes differ from its actual runtime identity: ' + ', '.join(mismatched))
    if not all(name in verified for name in NUMERICAL_PATHS):
        raise ValueError('B20 source archive lacks core model/loss/warp/LP/data/metric evidence')
    current = source_identity(root)
    differences = [name for name in NUMERICAL_PATHS if verified[name] != current['files'].get(name)]
    if differences:
        raise ValueError('B20 shared numerical implementation differs: ' + ', '.join(differences))
    compatibility = source_compatibility(origin, root)
    deltas = {}
    for old in sorted((archive_root / 'gfb20').glob('*.py')):
        if old.name.startswith('test_'):
            continue
        name = str(old.relative_to(archive_root))
        if name not in verified:
            raise ValueError('Unbound B20 campaign source file: ' + name)
        new = root / 'gfp40' / old.name
        if new.is_file():
            diff = ''.join(difflib.unified_diff(old.read_text().splitlines(True),
                new.read_text().splitlines(True), fromfile=name, tofile='gfp40/' + old.name))
            deltas[old.name] = dict(original_sha256=verified[name], current_sha256=sha256(new), diff=diff)
        else:
            deltas[old.name] = dict(original_sha256=verified[name], status='NOT_REUSED_IN_P40')
    if not deltas:
        raise ValueError('B20 campaign implementation archive is missing')
    return dict(schema='GFP40_B20_SOURCE_AUDIT_v1', complete=True,
        archive_root=str(archive_root), identity_path=str(identity_path), identity_sha256=sha256(identity_path),
        original_recorded_git_release=origin.get('git_release'), original_source_identity=origin,
        source_archive_sha256=object_sha(verified), verified_archive_files=verified,
        current_source_identity=current, shared_numerical_files={name: verified[name] for name in NUMERICAL_PATHS},
        measured_runtime_compatibility=compatibility,
        campaign_deltas=deltas, commit_equivalence_assumed=False, previous_campaign_written=False)


def inspect_b20_handoff(root, server):
    """Report old C07--C10 status without importing or changing the B20 queue."""
    root = Path(root)
    state = read(root / 'work_dir/_gfb20' / server / 'status.json')
    rows = []
    if server == 's5':
        for run in sorted((root / 'work_dir').glob('GFB20*')):
            if not run.is_dir() or not any('_' + key + '_' in run.name for key in ('C07', 'C08', 'C09', 'C10')):
                continue
            status = read(run / 'meta/training_status.json')
            official = read(run / 'official/summary.json')
            latest = read(run / 'last/identity.json')
            fullstate = run / 'last/training_state.pt'
            available = fullstate.is_file()
            fullstate_matches = bool(available and latest.get('full_state') is True
                and sha256(fullstate) == latest.get('training_state_sha256'))
            rows.append(dict(run_id=run.name, training_status=status,
                official_complete=official.get('official_complete', False),
                fullstate_available=available, last_identity=latest,
                fullstate_byte_identity_verified=fullstate_matches,
                reused_as_p40=False, new_training_count=0))
    return dict(schema='GFP40_READONLY_B20_HANDOFF_v1', server=server,
        local_controller_state=state, cases=rows, previous_campaign_written=False,
        inference='Local artifact status only. GPU/process admission is checked separately.')


def method_checks():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'gfp40'),'test_*.py',top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    return dict(complete=result.wasSuccessful() and result.testsRun>0,tests_run=result.testsRun,output=stream.getvalue())


def bind_assets(root,server,bindings=None,deadline=None):
    from gfp40.assets import discover_bindings,bind_parent
    verify_lane(server)
    discovered=discover_bindings(server,root)
    explicit=read_json(bindings) if isinstance(bindings,(str,Path)) else (bindings or {})
    explicit=explicit.get('parents', {k:v for k,v in explicit.items()
                                     if k not in ('b20_source_archive', 'b20_reference')})
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
    from gfp40.native_data import bind_lane_data
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
        raise ValueError('GFP40 production requires the assigned s3-s5 RTX5090; no CPU/4090 fallback')
    release=source_identity(root)
    folder=camp(root,server)
    old=read(folder/'preflight.json')
    if old and old['source_identity']!=release: raise ValueError('GFP40 source/runtime changed after preflight')
    provided=read_json(bindings) if isinstance(bindings,(str,Path)) else (bindings or {})
    audit=audit_b20_source(root,server,provided.get('b20_source_archive'))
    immutable_json(folder/'b20_source_audit.json',audit)
    atomic_json(folder/'b20_handoff.json',inspect_b20_handoff(root,server))
    parents,data=bind_assets(root,server,bindings,deadline)
    verify_native_data(root,server,data,deadline)
    checks=method_checks();atomic_json(folder/'method_regression.json',checks)
    if not checks['complete']: raise ValueError('GFP40 numerical regression failed')
    from g20.data import build_dataset
    from gfp40.evaluation import FRMetrics
    FRMetrics(build_dataset(data,'fr',root=root))
    from gfp40.assets import load_reference,load_parent
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
        b20_source_archive_sha256=audit['source_archive_sha256'],
        dataset_manifest_sha256=object_sha(data),parents=parents,method_tests=checks['tests_run'],
        gpu=torch.cuda.get_device_name(0),native_reference_online_parity=True,real_native_forward=True,
        previous_campaign_written=False,at_utc=utcnow())
    atomic_json(folder/'preflight.json',value)
    return value
