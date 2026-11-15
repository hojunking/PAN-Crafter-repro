"""Add target gates without modifying training configs, releases or checkpoints."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

from directshift48.reporting import INDEX, canonical_json, atomic_json, utc_now
from panda_input3 import plan, publication as pub
from panda_input3.reporting import install as install_fr
from panda_fr_columns import publication as fr
from panda_fr_columns.adapter import GoogleSheets
from sheet_publish_v2.transport import _cell_value
from .policy import POLICY, passes, choose, allowed

LEGACY_POLICY = pub.POLICY
STRICT_V2_POLICY = 'INPUT3_ALL_DATASET_TARGETS_MIN_ERGAS_V2'
_make = pub.make_row
_verify = pub.verify_owned
_joint = pub.joint_pass
_installed = False


def make_row(cfg, status, record=None, selector='STATUS', slot='ROW0'):
    if record is not None and not allowed(record, cfg, selector):
        raise ValueError('Representative requires HQNR; other results require every dataset target')
    row, raw = _make(cfg, status, record, selector, slot)
    note = json.loads(raw)
    if record is None:
        label = '대기' if status == 'QUEUED' else 'HQNR 조건 충족 모델 없음'
        row[INDEX['Selection@Step']] = f'보정 PAN · {label} · {status}'
    elif selector == 'HQNR_QUALIFIED_MIN_ERGAS':
        targets=plan.targets(cfg['sensor']);rr=record['rr']['metrics'];missing=[]
        if rr['ERGAS']>targets['ERGAS_max']:missing.append('ERGAS 미달')
        if 'PSNR_min' in targets and rr['PSNR']<targets['PSNR_min']:missing.append('PSNR 미달')
        outcome=' · '.join(missing) if missing else '전체 목표 충족'
        row[INDEX['Selection@Step']] = f"보정 PAN · 대표 · HQNR 충족 · ERGAS 최소 · {outcome} · {record['step']:,}"
    note['publication'].update(policy=POLICY, selection_override='HQNR_REPRESENTATIVE_AND_ALL_TARGET_DETAILS',
        joint_pass=passes(record,cfg), all_dataset_targets_pass=passes(record,cfg),
        is_case_representative=bool(record and selector == 'HQNR_QUALIFIED_MIN_ERGAS'),
        fallback_enabled=False, diagnostic_extrema_enabled=False,
        immutable_training_config_preserved=True, no_qualified_checkpoint=record is None,
        representative_gate='HQNR_MIN',other_result_gate='ALL_DATASET_TARGETS')
    note['reporting']['policy'] = POLICY
    raw = canonical_json(note)
    row[INDEX['Original description / notes']] = raw
    return row, raw


@contextmanager
def legacy_view(policy=LEGACY_POLICY):
    """Only for validating previously published V1 evidence; never for writes."""
    saved = pub.POLICY, pub.make_row, pub.joint_pass
    try:
        pub.POLICY, pub.make_row, pub.joint_pass = policy, _make, _joint
        yield
    finally:
        pub.POLICY, pub.make_row, pub.joint_pass = saved


def verify_owned(row, value, server):
    policy = json.loads(row[INDEX['Original description / notes']])['publication']['policy']
    if policy in (LEGACY_POLICY, STRICT_V2_POLICY):
        with legacy_view(policy):
            return _verify(row, value, server)
    if policy != POLICY:
        raise ValueError('Unknown INPUT3 publication policy')
    _verify(row, value, server)
    step = row[INDEX['Selected_step']]
    cfg = value['configs'][row[INDEX['Run_ID']]]
    selector=json.loads(row[INDEX['Original description / notes']])['publication']['selector']
    if step and not allowed(value['proof'][(cfg['run_id'],step)],cfg,selector):
        raise ValueError('V3 row violates its representative/detail targets')


def retired_rows(snapshot, value, server):
    wanted = {r[INDEX['Result_ID']] for r in value['rows']}
    retired = []
    seen = set()
    for number, row in enumerate(snapshot['rows'], 6):
        if row[INDEX['Campaign_ID']] != plan.CAMPAIGN_ID or row[INDEX['Server']] != server:
            continue
        verify_owned(row,value,server)
        rid = row[INDEX['Result_ID']]
        if rid in seen:
            raise ValueError('Duplicate INPUT3 row ID; inspect before migration')
        seen.add(rid)
        if rid not in wanted:
            retired.append(number)
    return retired


def _column(n):
    out = ''
    while n:
        n,k = divmod(n-1,26);out=chr(65+k)+out
    return out


def remove_retired(adapter, value, server, root, *, apply):
    """Back up and remove verified obsolete rows; never leave empty data rows."""
    snap = adapter.snapshot(value['destination'])
    from panda_input3.transport import _validate_snapshot
    _validate_snapshot(snap,value)
    retired = retired_rows(snap,value,server)
    if not retired:
        return []
    dest=value['destination'];sid=dest['sheet_id']
    tab=next(t for t in snap['physical']['metadata']['sheets'] if t['properties']['sheetId']==sid)
    width=tab['properties']['gridProperties']['columnCount']
    if width > 67:
        extra=adapter.physical_request('GET',adapter.BASE+dest['spreadsheet_id'],params={
            'ranges':[f"'{dest['sheet_title']}'!BP{r}:{_column(width)}{r}" for r in retired],
            'includeGridData':'true','fields':'sheets(data(rowData(values(userEnteredValue,note,dataValidation,chipRuns,textFormatRuns))))'})
        for sheet in extra.get('sheets',[]):
            for block in sheet.get('data',[]):
                for row in block.get('rowData',[]):
                    if any(c for c in row.get('values',[])):
                        raise ValueError('A retired row has foreign cells beyond BO; preserve it')
    if not apply:
        return retired
    folder=Path(root)/'publication/target_policy';stamp=str(time.time_ns())
    atomic_json(folder/f'before_{stamp}.json',snap)
    fresh=adapter.snapshot(dest)
    if plan.digest(fresh['physical'])!=plan.digest(snap['physical']):
        raise ValueError('Sheet changed before retirement; retry from a fresh snapshot')
    requests=[{'deleteDimension':{'range':{'sheetId':sid,'dimension':'ROWS',
        'startIndex':n-1,'endIndex':n}}} for n in sorted(retired,reverse=True)]
    atomic_json(folder/f'retirement_{stamp}.json',{'rows':retired,'requests':requests})
    adapter.physical_request('POST',adapter.BASE+dest['spreadsheet_id']+':batchUpdate',json={'requests':requests})
    after=adapter.snapshot(dest)
    retained=[(r,n) for i,(r,n) in enumerate(zip(snap['rows'],snap['notes']),6) if i not in retired]
    observed=list(zip(after['rows'],after['notes']))
    if retained!=observed:
        raise ValueError('Retirement readback differs; inspect saved backup')
    return retired


def publish(root,server,*,apply=False,credential=None):
    root=Path(root);folder=root/'publication/target_policy';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=fr.original_rows(pub.project(root,server))
        adapter=GoogleSheets(credential or plan.ROOT/'gspread/account.json')
        retired=remove_retired(adapter,value,server,root,apply=apply)
        if not apply and retired:
            result={'action':'TARGET_POLICY_MIGRATION_REQUIRED','retired_rows':retired,
                'desired_rows':len(value['rows']),'policy':POLICY,'readback_verified':False}
        else:
            result=fr.publish('panda_input3',root,server,apply=apply,credential=credential)
            result.update(target_policy=POLICY,retired_rows=retired,
                all_published_evaluations_follow_policy=True,representative_gate='HQNR_MIN',
                other_result_gate='ALL_DATASET_TARGETS',at=utc_now())
        atomic_json(folder/'last_poll.json',result)
        return result


def install():
    global _installed
    if _installed:
        return
    install_fr()
    original_extras=fr.extras
    def target_extras(value):
        cells=original_extras(value)
        if value['policy']==POLICY:
            for row in value['rows']:
                if not row[INDEX['Selected_step']]:
                    cells[row[INDEX['Result_ID']]][0]['note']=(
                        'No checkpoint satisfies the HQNR representative gate yet; metric cells are intentionally blank.')
        return cells
    fr.extras=target_extras
    pub.POLICY=POLICY
    pub.choose=choose
    pub.joint_pass=passes
    pub.all_targets_pass=passes
    pub.make_row=make_row
    pub.verify_owned=verify_owned
    pub.publish=publish
    _installed=True
