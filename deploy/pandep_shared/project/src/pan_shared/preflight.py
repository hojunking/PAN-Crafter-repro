"""Read-only source inventory and explicit, campaign-bound execution gates."""
from __future__ import annotations
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
from .common import CAMPAIGN, SOURCE_COMMIT, SENSORS, atomic_json, canonical_sha, file_sha, read_json, seed_for, timestamp
from .registry import build_registry, validate_registry
from .safety import isolated_paths, inventory, SafetyStop, control

PROJECT = Path(__file__).resolve().parents[2]

def source_manifest():
    files = sorted([*PROJECT.glob('src/**/*.py'), *PROJECT.glob('tests/**/*.py'),
                    *PROJECT.glob('vendor_reference/**/*'), PROJECT/'launch.py',
                    PROJECT/'requirements.lock', PROJECT/'pyproject.toml'])
    hashes={str(p.relative_to(PROJECT)):file_sha(p) for p in files
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    value=dict(schema='PANDEP_SOURCE_v1',legacy_source_commit=SOURCE_COMMIT,files=hashes)
    value['source_sha256']=canonical_sha(value)
    return value

def environment_manifest():
    names=('torch','numpy','scipy','h5py','safetensors','scikit-image','opencv-python-headless','gspread')
    versions={name:importlib.metadata.version(name) for name in names}
    expected={line.split('==')[0]:line.split('==')[1] for line in (PROJECT/'requirements.lock').read_text().splitlines() if '==' in line and not line.startswith('#')}
    if versions!=expected or sys.version_info[:2]!=(3,11):
        raise SafetyStop('BLOCKED_ENVIRONMENT: use pinned isolated Python3.11 container; no legacy environment upgrades')
    value=dict(python=platform.python_version(),packages=versions,precision='fp32',amp=False,tf32=False)
    value['environment_sha256']=canonical_sha(value)
    return value

def subset_and_probes(dataset_manifest):
    sensors={}
    for sensor in SENSORS:
        split=dataset_manifest['sensors'][sensor]['splits']
        count=split['train']['count']; source=split['train']['sha256']
        ranked=sorted(range(count),key=lambda i:(hashlib.sha256(f'PANDEP_SUBSET_v1|20261001|{sensor}|{source}|{i}'.encode()).hexdigest(),i))
        half=ranked[:count//2]
        def probes(ids, kind, sha):
            return sorted(ids,key=lambda i:(hashlib.sha256(f'PANDEP_PROBE_v1|20261001|{sensor}|{kind}|{sha}|{i}'.encode()).hexdigest(),i))[:128]
        value=dict(full_count=count,half_ids=half,half_count=len(half),
            train_probe_ids=probes(half,'train',source),
            val_probe_ids=probes(range(split['val']['count']),'val',split['val']['sha256']))
        if len(value['train_probe_ids'])!=128 or len(value['val_probe_ids'])!=128:
            raise SafetyStop('BLOCKED_DATA_IDENTITY: insufficient fixed probe pool')
        value['ids_sha256']=canonical_sha(value); sensors[sensor]=value
    result=dict(schema='PANDEP_SUBSETS_v1',subset_seed=20261001,
        subset_order='SHA256 UTF8(PANDEP_SUBSET_v1|20261001|sensor|source_sha256|decimal_id), then ID',
        probe_order='SHA256 UTF8(PANDEP_PROBE_v1|20261001|sensor|split|source_sha256|decimal_id), then ID',sensors=sensors)
    result['subset_sha256']=canonical_sha(result)
    return result

def _read_only(path):
    """Linux read-only mount, not an assumption based on H5 open mode."""
    return bool(os.statvfs(path).f_flag & getattr(os,'ST_RDONLY',1))

def prepare(work_root, *, data_catalog, data_root, server_identity, legacy_roots, microbatch=48):
    from .data import prepare_catalog
    work=Path(work_root).resolve()
    paths=isolated_paths(PROJECT,work,legacy_roots,data_root)
    inv=inventory(work,server_identity)
    if inv['server']!='s2':
        raise SafetyStop('BLOCKED_SERVER: preflight actual server mapping must be s2')
    if not legacy_roots:
        raise SafetyStop('BLOCKED_PATH: explicitly identify the existing repository root')
    if not _read_only(Path(data_root)):
        raise SafetyStop('BLOCKED_PATH: benchmark source must be on a read-only mount')
    env=environment_manifest(); source=source_manifest(); registry=build_registry(microbatch)
    existing=work/'campaign_manifest.json'
    if existing.exists():
        old=read_json(existing)
        for key,value in [('source_sha256',source['source_sha256']),('environment_sha256',env['environment_sha256']),('registry_sha256',registry['registry_sha256'])]:
            if old.get(key)!=value:
                raise SafetyStop('BLOCKED_IDENTITY: existing campaign '+key+' differs; do not overwrite')
        if old.get('hostname')!=inv['hostname']:
            raise SafetyStop('BLOCKED_IDENTITY: registered original host differs')
    catalog=read_json(data_catalog)
    for profile in catalog['sensors'].values():
        paths_to_check=[Path(data_root)/v['path'] for v in profile['splits'].values()]
        proof=profile.get('source_provenance',{})
        paths_to_check += [Path(data_root)/proof[k] for k in ('raw_train_path','raw_val_path') if k in proof]
        if any(not _read_only(p.resolve(strict=True)) for p in paths_to_check):
            raise SafetyStop('BLOCKED_PATH: symlink-resolved source not mounted read-only')
    work.mkdir(parents=True,exist_ok=True)
    atomic_json(work/'preflight/inventory.json',inv)
    def check_cancel():
        if (work/'control.json').exists() and control(work)['action'] in ('STOP','PAUSE','STOP_AFTER_BLOCK'):
            raise SafetyStop('STOPPED_BY_OPERATOR: preflight interrupted before formal training')
    check_cancel()
    dataset=prepare_catalog(data_catalog,data_root,work/'cache/lpan',full_scan=True,check_cancel=check_cancel)
    subsets=subset_and_probes(dataset)
    for name,value in [('source_manifest',source),('environment_manifest',env),('dataset_manifest',dataset),('subset_manifest',subsets)]:
        path=work/(name+'.json')
        if path.exists() and read_json(path)!=value:
            raise SafetyStop('BLOCKED_IDENTITY: frozen '+name+' changed')
        atomic_json(path,value)
    streams={str(seed):{key:seed_for(seed,*key.split('/')) for key in
        ['sensor-order',*[f'sample-order/{s}' for s in SENSORS],*[f'augmentation/{s}' for s in SENSORS], 'probes']}
        for seed in (271001,271002,271003)}
    atomic_json(work/'seed_registry.json',dict(schema='PANDEP_SEEDS_v1',streams=streams,
        derivation='first64bits(canonical SHA256 JSON [PANDEP_RNG_v1,master,*parts]) modulo (2^63-1)',
        model_init='per canonical tensor name AND shape; no case/subset/order key'))
    manifest=dict(schema='PANDEP_CAMPAIGN_v4',campaign_id=CAMPAIGN,server='s2',hostname=inv['hostname'],
        **paths,server_identity_file=str(Path(server_identity).resolve()),data_root=str(Path(data_root).resolve()),
        data_catalog=str(Path(data_catalog).resolve()),source_sha256=source['source_sha256'],
        environment_sha256=env['environment_sha256'],dataset_sha256=dataset['dataset_sha256'],
        subset_sha256=subsets['subset_sha256'],registry_sha256=registry['registry_sha256'],
        microbatch=microbatch,private_data_enabled=False,training_started=False)
    atomic_json(work/'campaign_manifest.json',manifest)
    return manifest

def check_identity(work_root, *, require_registered=True):
    work=Path(work_root).resolve(); manifest=read_json(work/'campaign_manifest.json')
    if manifest.get('campaign_id')!=CAMPAIGN or manifest.get('server')!='s2':
        raise SafetyStop('BLOCKED_IDENTITY: campaign/server')
    isolated_paths(PROJECT,work,manifest['legacy_roots'],manifest['data_root'])
    if manifest['source_sha256']!=source_manifest()['source_sha256']:
        raise SafetyStop('BLOCKED_IDENTITY: source changed')
    if manifest['environment_sha256']!=environment_manifest()['environment_sha256']:
        raise SafetyStop('BLOCKED_IDENTITY: environment changed')
    if require_registered or (work/'cases.json').exists():
        registry=validate_registry(read_json(work/'cases.json'))
    else:
        registry=build_registry(manifest['microbatch'])
    if registry['registry_sha256']!=manifest['registry_sha256']:
        raise SafetyStop('BLOCKED_IDENTITY: registry changed')
    for name,field in [('dataset','dataset_sha256'),('subset','subset_sha256')]:
        value=read_json(work/(name+'_manifest.json'))
        if value.get(field)!=manifest[field] or canonical_sha({k:v for k,v in value.items() if k!=field})!=value[field]:
            raise SafetyStop('BLOCKED_IDENTITY: '+name+' manifest seal')
    if not _read_only(manifest['data_root']):
        raise SafetyStop('BLOCKED_PATH: benchmark mount is no longer read-only')
    return manifest


def reconfigure_preformal_microbatch(work_root, microbatch):
    """Explicit operator preflight-only choice. Never change a registered run."""
    work=Path(work_root).resolve()
    manifest=check_identity(work,require_registered=False)
    registry=build_registry(microbatch)
    if microbatch==manifest['microbatch']: return manifest
    gate=work/'preflight/gates.json'
    if ((work/'cases.json').exists() or any((work/'runs').glob('*/resume/last.json'))
            or (gate.exists() and read_json(gate).get('status')=='PASS')):
        raise SafetyStop('BLOCKED_IDENTITY: microbatch change requires a new recipe after registration/gate PASS')
    history=work/'preflight'/('microbatch_'+str(manifest['microbatch'])+'_before_'+str(microbatch)+'_'+str(time.time_ns())+'.json')
    atomic_json(history,{'previous_campaign':manifest,'explicit_preformal_microbatch':microbatch,
                         'prior_q00_and_failures_preserved':True,'at':timestamp()})
    manifest=dict(manifest,microbatch=microbatch,registry_sha256=registry['registry_sha256'])
    atomic_json(work/'campaign_manifest.json',manifest)
    return manifest
