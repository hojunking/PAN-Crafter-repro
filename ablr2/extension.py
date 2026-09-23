"""Append-only C17 debts and evidence-bound A17 anchors.

The original stage cursor, definitions and observations are not migration input
to be regenerated. Discover every registered full panel and overlay only C17.
"""
from dataclasses import asdict, replace
from pathlib import Path

from ablr2.common import (ROOT,camp,run_dir,read,read_json,atomic_json,immutable_json,
                         object_sha,sha256,source_identity,append_event)
from ablr2.plan import case_for,validate_case,Case,EXTENSION_ID,component_config,build_config

FULL_PHASES=('BOOT5','REFRESH5','VERIFY5')


def _register(root,case):
    validate_case(case)
    folder=camp(root,case.server_id);registry=read_json(folder/'cases.json')
    old=registry['cases'].get(case.run_id)
    if old is not None and old!=asdict(case):raise ValueError('Extension would change an existing task')
    if old is None:
        registry['cases'][case.run_id]=asdict(case)
        atomic_json(folder/'cases.json',registry)
        append_event(folder/'extension/task_ledger.jsonl','REGISTER_EXTENSION_CASE',case=asdict(case))
    return case


def c17_for(anchor,zero):
    if (anchor.case_id!='C03' or zero.case_id!='TZERO' or anchor.server_id!=zero.server_id
            or anchor.teacher_seed!=zero.seed or anchor.recipe_id!=zero.recipe_id):
        raise ValueError('C17 needs the matched local TZERO of its C03 panel')
    return validate_case(replace(anchor,case_id='C17',run_id=anchor.run_id.replace('_C03_','_C17_'),
        teacher_kind='TZERO',reference_id=zero.reference_id,teacher_run_id=zero.run_id))


def sync(root,server,state):
    """Discover actual stages, including partially/completely executed refreshes."""
    folder=camp(root,server)
    registry=read_json(folder/'cases.json')['cases']
    rows=[case_for(run,root) for run in registry]
    path=folder/'extension/debts.json'
    debts=read(path,dict(extension_id=EXTENSION_ID,debts={}))
    for order,stage in enumerate(state['stages']):
        if stage['kind'] not in FULL_PHASES:continue
        members=[case_for(run,root) for run in stage['run_ids']]
        for anchor in members:
            if anchor.case_id!='C03':continue
            candidates=[c for c in rows if c.case_id=='TZERO' and c.seed==anchor.teacher_seed
                and c.server_id==server and c.recipe_id==anchor.recipe_id]
            panel=[c for c in candidates if c.run_id in stage['run_ids'] and c.sweep==anchor.sweep]
            if len(panel)!=1:raise ValueError('Registered full panel lacks a unique matched TZERO')
            case=c17_for(anchor,panel[0])
            if case.run_id in registry:
                registered=case_for(case.run_id,root)
                if replace(registered,queue_rank=case.queue_rank)!=case:
                    raise ValueError('Existing C17 differs from its C03 panel')
                case=registered
            _register(root,case)
            definition=dict(run_id=case.run_id,anchor_run_id=anchor.run_id,stage_id=stage['stage_id'],
                stage_order=order,sweep=case.sweep,insert_after_run_id=anchor.run_id)
            old=debts['debts'].get(case.run_id)
            if old and any(old.get(k)!=v for k,v in definition.items()):
                raise ValueError('C17 debt definition changed')
            if old is None:debts['debts'][case.run_id]=dict(definition,status='PENDING')
    atomic_json(path,debts)
    return debts


def stage_cases(root,server,stage,*,completed_only=False,state=None):
    values=[case_for(run,root) for run in stage['run_ids']]
    if completed_only:
        values=[c for c in values if c.case_id!='C17' or state.get('runs',{}).get(c.run_id,{}).get('complete')]
    debts=read(camp(root,server)/'extension/debts.json',{'debts':{}})['debts']
    known={c.run_id for c in values}
    for row in debts.values():
        if row['stage_id']!=stage['stage_id'] or row['run_id'] in known:continue
        if completed_only and not state.get('runs',{}).get(row['run_id'],{}).get('complete'):continue
        case=case_for(row['run_id'],root)
        index=next(i for i,c in enumerate(values) if c.run_id==row['anchor_run_id'])
        values.insert(index+1,case);known.add(case.run_id)
    # Partial C17 coverage is debt, not an 18-component balanced matrix.
    if completed_only and sum(c.case_id=='C17' for c in values) not in (0,5):
        values=[c for c in values if c.case_id!='C17']
    return values


def expected_student_identity(case,data):
    from ablr2.model import build_model,state_hash
    from fh12.training import BatchStream
    model,initial=build_model(bands=case.num_bands,seed=case.seed,role='S',component=component_config('C02',case.recipe_id))
    del model
    stream=BatchStream(data['splits']['train']['count'],48,case.seed)
    cfg=build_config(case)
    return dict(init_U_sha256=initial['hashes']['U'],
        sampler_sha256=state_hash({'order':stream.order,'rotations':stream.rotations}),
        rng_roles=dict(data_order=case.seed+300000,augmentation=case.seed+400000,
            corruption=cfg['ablr2']['corruption_seed'],workers=case.seed+500000))


def repair_anchor(anchor):
    wave='REPAIR_'+object_sha(asdict(anchor))[:16]
    run=(f'ABLR2_{anchor.sensor}_{anchor.server_id}_{anchor.recipe_revision}_PAIR_REPAIR_'
         f'{wave}_{anchor.sweep_id}_C03_SS{anchor.seed}_FRESH50')
    return validate_case(replace(anchor,run_id=run,phase='PAIR_REPAIR',wave=wave))


def prepare_anchor(case,root=ROOT):
    """Called only after C03 and both exact Teacher endpoints are available.

    Absence of old-C03 comparability creates a separate C03 repair computation;
    absence of a valid TZERO never substitutes another Teacher or seed.
    """
    if case.case_id!='C17':raise ValueError('A17 anchor requested for another component')
    folder=camp(root,case.server_id);wd=run_dir(case.run_id,root)
    existing=read(wd/'meta/paired_anchor.json')
    if existing:
        if existing.get('c17_case')!=asdict(case):raise ValueError('C17 pairing identity changed')
        return case_for(existing['anchor_run_id'],root)
    from ablr2.references import validate_teacher_pair
    from ablr2.analysis import case_report
    anchor_id=case.run_id.replace('_C17_','_C03_')
    anchor=case_for(anchor_id,root)
    teacher_pair=validate_teacher_pair(anchor.teacher_run_id,case.teacher_run_id,root)
    data=read_json(folder/'dataset_manifest.json')
    expected=expected_student_identity(case,data)
    consumer=source_identity(root)
    mode='VERIFIED_ORIGINAL';reason=None
    try:
        report=case_report(anchor,root)
        if (report['data_sha256']!=object_sha(data)
                or any(report.get(k)!=v for k,v in expected.items())):
            raise ValueError('Initial U/native stream/data do not match current C17 computation')
    except (ValueError,FileNotFoundError) as exc:
        mode='PAIR_REPAIR';reason=f'{type(exc).__name__}: {exc}'
        anchor=_register(root,repair_anchor(anchor))
    receipt=dict(schema='ABLR2X_A17_PAIRING_v1',c17_case=asdict(case),anchor_run_id=anchor.run_id,
        original_c03_run_id=anchor_id,mode=mode,repair_reason=reason,source_identity=consumer,
        data_sha256=object_sha(data),expected_student_identity=expected,teacher_pair=teacher_pair,
        old_observation_rewritten=False,new_component=False)
    immutable_json(wd/'meta/paired_anchor.json',receipt)
    return anchor


def paired_anchor_for(case,root=ROOT):
    """Analysis authenticates actual completed tensors/streams, not seed labels."""
    from ablr2.analysis import case_report
    from ablr2.references import validate_teacher_pair
    path=run_dir(case.run_id,root)/'meta/paired_anchor.json'
    receipt=read_json(path)
    if receipt.get('schema')!='ABLR2X_A17_PAIRING_v1' or receipt.get('c17_case')!=asdict(case):
        raise ValueError('PAIRING_NOT_VERIFIED: missing exact C17 pairing receipt')
    anchor=case_for(receipt['anchor_run_id'],root)
    if (anchor.case_id!='C03' or anchor.seed!=case.seed or anchor.server_id!=case.server_id
            or anchor.recipe_id!=case.recipe_id or anchor.teacher_seed!=case.teacher_seed):
        raise ValueError('PAIRING_NOT_VERIFIED: C03 anchor contract differs')
    pair=validate_teacher_pair(anchor.teacher_run_id,case.teacher_run_id,root)
    if pair!=receipt['teacher_pair']:raise ValueError('Matched Teacher evidence changed')
    for item in (case,anchor):
        observed=case_report(item,root)
        if (observed['data_sha256']!=receipt['data_sha256']
                or observed['analysis_source_identity']!=receipt['source_identity']
                or any(observed[k]!=v for k,v in receipt['expected_student_identity'].items())):
            raise ValueError('PAIRING_NOT_VERIFIED: actual initial U/native sequence/source differs')
    return anchor


def ready_debts(root,server,state):
    """Oldest ready first; callers admit at most two before original work."""
    document=sync(root,server,state);ready=[]
    for row in sorted(document['debts'].values(),key=lambda r:(r['stage_order'],r['sweep'])):
        case=case_for(row['run_id'],root)
        if state['runs'].get(case.run_id,{}).get('complete'):
            row['status']='COMPLETE';continue
        if not state['runs'].get(row['anchor_run_id'],{}).get('complete'):
            row['status']='WAIT_C03';continue
        try:
            anchor=prepare_anchor(case,root)
        except (ValueError,FileNotFoundError,RuntimeError) as exc:
            row.update(status='REFERENCE_UNAVAILABLE',reason=f'{type(exc).__name__}: {exc}')
            continue
        ready.append((case,anchor));row['status']='READY'
    atomic_json(camp(root,server)/'extension/debts.json',document)
    return ready
