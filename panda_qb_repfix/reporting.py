"""Additive publication adapter: verify V4 history, update existing row IDs."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from directshift48.reporting import INDEX, canonical_json, atomic_json, utc_now
from panda_input3 import plan, publication as pub
from panda_input3_original import reporting as old
from panda_input3_targets import reporting as prior
from panda_fr_columns import publication as fr
from panda_fr_columns.adapter import GoogleSheets
from .policy import POLICY, REPRESENTATIVES, allowed, choose, passes, hqnr_passes

_installed = False


def make_row(cfg, status, record=None, selector='STATUS', slot='ROW0'):
    if cfg['sensor'] != 'QB' or record is not None and not allowed(record, cfg, selector):
        raise ValueError('Invalid QB representative/detail evidence')
    # The original constructor already supports BEST_HQNR_FALLBACK. Avoid V3's
    # obsolete HQNR admission gate while keeping all metric/artifact columns.
    row, raw = prior._make(cfg, status, record, selector, slot)
    note = json.loads(raw)
    row[INDEX['Experiment']] = cfg['experiment'] + ' [기존 PAN 선정]'
    if record is None:
        label = ('대기' if status == 'QUEUED' else '30K 새 실행으로 교체'
                 if status == 'REPLACED_BY_C30' else '등록 평가 없음')
        row[INDEX['Selection@Step']] = f'기존 PAN · {label} · {status}'
    else:
        rr = record['rr']['metrics']; missing = []
        if not hqnr_passes(record, cfg): missing.append('HQNR 미달')
        if rr['ERGAS'] > plan.targets('QB')['ERGAS_max']: missing.append('ERGAS 미달')
        outcome = ' · '.join(missing) if missing else '전체 목표 충족'
        label = ('대표 · HQNR 최고' if selector == 'BEST_HQNR_FALLBACK' else
                 '대표 · HQNR 충족 · ERGAS 최소' if selector in REPRESENTATIVES else '조건 충족')
        if 'FIXED' in selector: label += ' · ' + next(x for x in selector.split('|') if x.startswith('FIXED'))
        row[INDEX['Selection@Step']] = f"기존 PAN · {label} · {outcome} · {record['step']:,}"
    row[INDEX['Eval_scope']] = 'DUAL_FR_FIXED_REFERENCE_ORIGINAL_SELECTION' if record else 'STATUS_ONLY'
    note.update(checkpoint_selection_reference='original PAN', official_selection_reference='ORIGINAL_PAN',
        publication_reference_override=dict(from_reference=cfg['official_selection_reference'],
            to='ORIGINAL_PAN', scope='Reporting only; immutable training and evaluation preserved'))
    note['publication'].update(policy=POLICY, selection_override='ALWAYS_ONE_REPRESENTATIVE_PLUS_ALL_TARGET_DETAILS',
        selection_reference='ORIGINAL_PAN', joint_pass=passes(record,cfg), all_dataset_targets_pass=passes(record,cfg),
        is_case_representative=bool(record and selector in REPRESENTATIVES),
        fallback_enabled=True, fallback_used=selector=='BEST_HQNR_FALLBACK', diagnostic_extrema_enabled=False,
        immutable_training_config_preserved=True, no_qualified_checkpoint=not hqnr_passes(record,cfg),
        representative_gate='FINITE_EVIDENCE_HQNR_PRIORITY_WITH_FALLBACK',
        other_result_gate='ALL_DATASET_TARGETS_ORIGINAL_PAN_HQNR')
    note['reporting'].update(policy=POLICY, selection_reference='ORIGINAL_PAN')
    raw = canonical_json(note); row[INDEX['Original description / notes']] = raw
    return row, raw


@contextmanager
def old_context():
    fields = ('POLICY','make_row','joint_pass','all_targets_pass')
    saved = {k:getattr(pub,k) for k in fields}
    pub.POLICY, pub.make_row = old.POLICY, old.make_row
    pub.joint_pass = pub.all_targets_pass = old.passes
    try: yield
    finally:
        for k,v in saved.items(): setattr(pub,k,v)


def verify_owned(row, value, server):
    note = json.loads(row[INDEX['Original description / notes']])
    if note['publication']['policy'] != POLICY:
        with old_context(): return old.verify_owned(row, value, server)
    prior._verify(row, value, server)
    cfg = value['configs'][row[INDEX['Run_ID']]]
    step = row[INDEX['Selected_step']]
    record = value['proof'].get((cfg['run_id'],step)) if step else None
    expected, raw = make_row(cfg,row[INDEX['Status']],record,
        note['publication']['selector'],note['publication']['slot'])
    expected_note = json.loads(raw)
    for key in ('Selection@Step','Eval_scope'):
        if row[INDEX[key]] != expected[INDEX[key]]: raise ValueError('QB representative label mismatch: '+key)
    for key in ('publication','official_selection_reference','publication_reference_override','checkpoint_selection_reference'):
        if note.get(key) != expected_note[key]: raise ValueError('QB representative provenance mismatch: '+key)


def export_trajectories(root, value):
    import csv
    fields = ['run_id','step','original_HQNR','corrected_HQNR','ERGAS','PSNR',
              'original_Ds','corrected_Ds','Dlambda','checkpoint_sha256','evaluation_sha256']
    folder = Path(root)/'publication/qb_representative';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'all_evaluations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for (run,step),r in sorted(value['proof'].items()):
            a,b,rr=r['fr']['metrics'],r['fr_corrected_pan']['metrics'],r['rr']['metrics']
            if a['D_lambda']!=b['D_lambda']: raise ValueError('Same-output Dlambda mismatch')
            writer.writerow(dict(run_id=run,step=step,original_HQNR=a['HQNR'],corrected_HQNR=b['HQNR'],
                ERGAS=rr['ERGAS'],PSNR=rr['PSNR'],original_Ds=a['D_s'],corrected_Ds=b['D_s'],Dlambda=a['D_lambda'],
                checkpoint_sha256=r['checkpoint']['file_sha256'],evaluation_sha256=r['evidence_sha256']))


def publish(root, server, *, apply=False, credential=None):
    if plan.sensor_for(server) != 'QB': raise ValueError('QB publication only')
    root=Path(root);folder=root/'publication/qb_representative';folder.mkdir(parents=True,exist_ok=True)
    guard=root/'publication/target_policy';guard.mkdir(parents=True,exist_ok=True)
    with (guard/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=fr.original_rows(pub.project(root,server));export_trajectories(root,value)
        adapter=GoogleSheets(credential or plan.ROOT/'gspread/account.json')
        # Stable ROW0 IDs replace blank representatives. This upgrade adds no
        # deletion requirement; refuse unexpected retirement instead of deleting.
        snap=adapter.snapshot(value['destination'])
        from panda_input3.transport import _validate_snapshot
        _validate_snapshot(snap,value)
        retired=prior.retired_rows(snap,value,server)
        if retired: raise ValueError('Unexpected obsolete rows; preserve and inspect: '+repr(retired))
        if apply: atomic_json(folder/f'before_{__import__("time").time_ns()}.json',snap)
        result=fr.publish('panda_input3',root,server,apply=apply,credential=credential)
        result.update(target_policy=POLICY,selection_reference='ORIGINAL_PAN',fallback_enabled=True,at=utc_now())
        atomic_json(folder/'last_poll.json',result)
        return result


def install():
    global _installed
    if _installed:return
    old.install()
    previous=fr.extras
    def extras(value):
        result=previous(value)
        if value['policy']==POLICY:
            for cells in result.values():
                try:note=json.loads(cells[0]['note'])
                except json.JSONDecodeError:
                    cells[0]['note']='No registered evaluation yet; representative appears after evaluation.'
                    continue
                note.update(official_selection_reference='ORIGINAL_PAN',selection_policy=POLICY,
                            corrected_pan_role='AUXILIARY_SAME_CHECKPOINT')
                cells[0]['note']=canonical_json(note)
        return result
    fr.extras=extras
    prior.verify_owned=verify_owned
    pub.POLICY,pub.choose,pub.make_row,pub.verify_owned=POLICY,choose,make_row,verify_owned
    pub.joint_pass=pub.all_targets_pass=passes
    pub.publish=publish
    _installed=True
