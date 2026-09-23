"""Pre-result GF2 BOOT5 seed collision overlay; authored CSVs stay immutable."""
from dataclasses import replace
from pathlib import Path
import hashlib

from ablr2.common import ROOT,camp,read_json,read_config,object_sha,sha256,immutable_json
from ablr2.plan import EXTENSION_ID,EXTENSION_BOOT_CASES,Case

SCHEMA='ABLR2X_GF2_BOOT_SEED_RESOLUTION_v1'


def authored_cases():return tuple(c for c in EXTENSION_BOOT_CASES if c.server_id=='s3')


def scan_used_seeds(root=ROOT,extra_registries=()):
    """Only training-start evidence or explicitly bound used-seed registries.

    Configs, queued cases, design CSVs and ordinary seed-allocation ledgers are
    never considered executed experiments. Referenced Teacher seeds in Student
    configs are not mistaken for that Student's own initialization seed.
    """
    root=Path(root).resolve();work=(root/'work_dir').resolve();records=[]
    own=(camp(root,'s3')/'runs').resolve()
    for path in sorted(work.rglob('training_start_manifest.json')):
        if path.resolve().is_relative_to(own):continue
        start=read_json(path);run=path.parent.parent if path.parent.name=='meta' else path.parent
        seed=None;config=None
        for name in ('meta/config.resolved.yaml','meta/config.resolved.json','config.resolved.yaml','config.yaml'):
            candidate=run/name
            if candidate.is_file():
                value=read_config(candidate)
                if type(value.get('seed')) is int:seed=value['seed'];config=candidate;break
        if seed is None:
            seed=start.get('model_seed',start.get('seed'))
        if seed is None and (run/'init_manifest.json').is_file():
            seed=read_json(run/'init_manifest.json').get('seed')
        if type(seed) is not int or not 0<=seed<2**31-1:
            raise ValueError('Actual training seed cannot be authenticated: '+str(path))
        records.append(dict(seed=seed,run_id=start.get('run_id',run.name),
            evidence_path=str(path),evidence_sha256=sha256(path),
            config_path=str(config) if config else None,config_sha256=sha256(config) if config else None,
            proof='ACTUAL_TRAINING_START'))
    for path in map(Path,extra_registries):
        document=read_json(path)
        if document.get('schema')!='ACTUAL_USED_TRAINING_SEEDS_v1' or not isinstance(document.get('runs'),list):
            raise ValueError('Explicit actual-used training-seed registry required')
        for row in document['runs']:
            if (type(row.get('seed')) is not int or not 0<=row['seed']<2**31-1
                    or not row.get('run_id') or row.get('training_started') is not True):
                raise ValueError('Queued/planned seeds cannot masquerade as used training history')
            records.append(dict(seed=row['seed'],run_id=row['run_id'],evidence_path=str(path.resolve()),
                evidence_sha256=sha256(path),config_path=None,config_sha256=None,proof='EXPLICIT_ACTUAL_USED_REGISTRY'))
    return sorted(records,key=lambda row:(row['seed'],row['run_id'],row['evidence_path']))


def _mapping(history):
    used={row['seed'] for row in history};reserved={base+i for base in (981000,991000) for i in range(1,6)}
    mapping=[]
    for number in range(1,6):
        for role,base in (('T',981000),('S',991000)):
            authored=base+number;selected=authored;counter=0
            while selected in used or counter>0 and selected in reserved:
                counter+=1
                payload=f'{EXTENSION_ID}|GF2|BOOT5|P{number:02}|{role}|{authored}|collision={counter}'
                selected=1+int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')%(2**31-2)
                if selected in reserved:continue
                if selected not in used:break
            # A derived seed cannot consume another role/sweep's authored seed.
            if counter and selected in reserved:raise ValueError('Unexpected seed namespace collision')
            mapping.append(dict(sweep=f'P{number:02}',role=role,authored_seed=authored,
                resolved_seed=selected,collision_counter=counter))
            used.add(selected)
    return mapping


def remap_boot_cases(mapping):
    """Pure overlay transformation, including exact Teacher references."""
    lookup={(r['sweep'],r['role']):(r['authored_seed'],r['resolved_seed']) for r in mapping}
    if len(lookup)!=10:raise ValueError('Complete five Teacher/Student seed blocks required')
    result=[]
    for case in authored_cases():
        ts0,ts=lookup[(case.sweep,'T')];ss0,ss=lookup[(case.sweep,'S')]
        def rename(value):
            return value.replace(f'_TS{ts0}_',f'_TS{ts}_').replace(f'_SS{ss0}_',f'_SS{ss}_') if value else value
        result.append(replace(case,run_id=rename(case.run_id),teacher_run_id=rename(case.teacher_run_id),
            reference_id=rename(case.reference_id),paired_init_group=rename(case.paired_init_group),
            teacher_seed=ts if case.teacher_seed is not None else None,
            student_seed=ss if case.student_seed is not None else None))
    return tuple(result)


def validate_resolution(receipt):
    if (receipt.get('schema')!=SCHEMA or receipt.get('extension_id')!=EXTENSION_ID
            or receipt.get('server')!='s3' or receipt.get('sensor')!='GF2'
            or receipt.get('registered_before_any_GF2_ABL_training') is not True
            or receipt.get('original_CSV_modified') is not False
            or receipt.get('selection')!='PERFORMANCE_INDEPENDENT_SHA256_COLLISION_ONLY'
            or receipt.get('history_sha256')!=object_sha(receipt.get('history'))
            or receipt.get('authored_cases_sha256')!=object_sha([c.to_dict() for c in authored_cases()])
            or receipt.get('mapping')!=_mapping(receipt['history'])):
        raise ValueError('GF2 BOOT seed resolution provenance changed')
    cases=remap_boot_cases(receipt['mapping'])
    if receipt.get('resolved_cases_sha256')!=object_sha([c.to_dict() for c in cases]):
        raise ValueError('Resolved GF2 cases differ from deterministic collision mapping')
    return cases


def resolve_gf2_boot(root=ROOT,extra_registries=()):
    root=Path(root);folder=camp(root,'s3');path=folder/'boot_seed_resolution.json'
    if path.is_file():
        receipt=read_json(path);return validate_resolution(receipt),receipt
    if any((folder/'runs').rglob('training_start_manifest.json')):
        raise ValueError('Missing pre-result seed resolution after GF2 ablation training started')
    history=scan_used_seeds(root,extra_registries);mapping=_mapping(history);cases=remap_boot_cases(mapping)
    receipt=dict(schema=SCHEMA,extension_id=EXTENSION_ID,server='s3',sensor='GF2',
        history=history,history_sha256=object_sha(history),mapping=mapping,
        authored_cases_sha256=object_sha([c.to_dict() for c in authored_cases()]),
        resolved_cases_sha256=object_sha([c.to_dict() for c in cases]),
        registered_before_any_GF2_ABL_training=True,selection='PERFORMANCE_INDEPENDENT_SHA256_COLLISION_ONLY',
        original_CSV_modified=False)
    validate_resolution(receipt);immutable_json(path,receipt)
    return cases,receipt


def validate_resolved_boot_case(case,root=None):
    if case.server_id!='s3' or case.sensor!='GF2' or case.phase!='BOOT5':
        raise ValueError('Only GF2/s3 BOOT may use this collision overlay')
    if root is None:
        # Structure only: controller admission must additionally authenticate
        # the exact persisted mapping. Do not read a different host's ROOT.
        mapping=[]
        for number in range(1,6):
            for role,base in (('T',981000),('S',991000)):
                selected=base+number
                if f'P{number:02}'==case.sweep:
                    value=case.teacher_seed if role=='T' else case.student_seed
                    if value is not None:selected=value
                mapping.append(dict(sweep=f'P{number:02}',role=role,authored_seed=base+number,resolved_seed=selected))
        cases=remap_boot_cases(mapping)
    else:
        cases=validate_resolution(read_json(camp(root,'s3')/'boot_seed_resolution.json'))
    if case not in cases:raise ValueError('BOOT case is not its pre-registered collision resolution')
    return case


def validate_boot_admission(root,case):
    if case.server_id!='s3' or case.phase!='BOOT5':return case
    validate_resolved_boot_case(case,root)
    receipt=read_json(camp(root,'s3')/'boot_seed_resolution.json')
    original={(r['evidence_path'],r['evidence_sha256'],r['seed']) for r in receipt['history']}
    current=scan_used_seeds(root)
    for row in current:
        key=(row['evidence_path'],row['evidence_sha256'],row['seed'])
        if row['seed']==case.seed and key not in original:
            raise ValueError('New external seed collision after immutable BOOT pre-registration; no silent remap')
    return case


def resolved_boot_seeds(root,sweep):
    receipt=read_json(camp(root,'s3')/'boot_seed_resolution.json');validate_resolution(receipt)
    values={r['role']:r['resolved_seed'] for r in receipt['mapping'] if r['sweep']==sweep}
    return values['T'],values['S']
