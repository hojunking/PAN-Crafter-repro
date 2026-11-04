"""Three-decimal values, raw-evidence selection and complete measured costs."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

from directshift48.reporting import atomic_json, canonical_json
from panda_wv3_main_rgb import plan, publication as original
from . import cost

INDEX = original.INDEX
POLICY = 'WV3_MAIN_RGB_REPORTING_COST_3DP_V1'
BASE_VERIFY = original.verify_owned


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def timing(directory, step, records):
    """Sum committed optimizer steps and all official evaluations to selected step."""
    directory = Path(directory)
    seconds = 0.; first = None
    with (directory/'train.jsonl').open() as stream:
        for expected in range(1, step+1):
            line = stream.readline()
            if not line.endswith('\n'):
                raise ValueError('Incomplete committed training-time evidence')
            row = json.loads(line)
            if row['step'] != expected or not math.isfinite(row['seconds']) or row['seconds'] < 0:
                raise ValueError('Non-contiguous or invalid training timing')
            seconds += row['seconds']
            if first is None:
                first = stamp(row['at']) - timedelta(seconds=row['seconds'])
    selected = {e['step']:e for e in records if e['step'] <= step}
    if sorted(selected) != list(range(1000, step+1, 1000)) or step not in selected:
        raise ValueError('Cumulative evaluation timing requires every scheduled evaluation')
    eval_seconds = sum(e['elapsed_seconds'] for e in selected.values())
    wall = (stamp(selected[step]['recorded_at'])-first).total_seconds()
    if wall < 0 or not math.isfinite(eval_seconds) or eval_seconds < 0:
        raise ValueError('Invalid evaluation/wall timing')
    return {'Train(h)':seconds/3600, 'Eval(h)':eval_seconds/3600, 'Wall(h)':wall/3600,
        'scope':'Stage2 committed optimizer steps 1..selected_step; Train excludes eval/checkpoint/calibration/Sheet; Eval sums RR/FR+test-RRE evaluations; Wall is elapsed first optimizer step start to selected evaluation record, includes IO and any downtime; Stage1 cost excluded',
        'selected_step':step, 'optimizer_steps':step, 'evaluation_count':len(selected)}


def project(root, server):
    root = Path(root)
    value = original.project(root, server)
    costs = {}; timings = {}
    for row in value['rows']:
        run = row[INDEX['Run_ID']]; step = row[INDEX['Selected_step']]
        note = json.loads(row[INDEX['Original description / notes']])
        note['reporting'] = {'policy':POLICY, 'decimal_places':3,
            'selection_uses_unrounded_evidence':True,
            'raw_metrics':{k:row[INDEX[k]] for k in original.METRICS.values()}}
        if step:
            cfg = value['configs'][run]
            if not costs:
                costs = cost.load(root,cfg)
            elif costs['graph'] != cost.graph_signature(cfg):
                raise ValueError('Cannot share a profile across different inference graphs')
            if run not in timings:
                timings[run] = [e for (r,_),e in value['proof'].items() if r==run]
            times = timing(root/'runs'/run, step, timings[run])
            fields = costs['measurement']['sheet_cost']
            if fields['Params(M)'] != row[INDEX['Params(M)']]:
                raise ValueError('Profile parameter count does not match evaluated deployment')
            for k,v in fields.items(): row[INDEX[k]] = v
            for k in ('Train(h)','Eval(h)','Wall(h)'): row[INDEX[k]] = times[k]
            note['reporting'].update(cost_profile_sha256=costs['profile_sha256'],
                cost_profile='reporting/cost.json', cost_basis='SHARED_IDENTICAL_INFERENCE_GRAPH',
                cost_profile_checkpoint=costs['checkpoint'], cost_environment=costs['measurement']['environment'],
                cost_rr=costs['measurement']['observations']['rr']['sheet_fields'],
                cost_fr=costs['measurement']['observations']['fr']['sheet_fields'], timing=times)
            row[INDEX['Cost_scope']] += '; '+times['scope']
        # D:Y are result/cost columns. Preserve LR/step/seed/config precision.
        for i in range(3,25):
            if type(row[i]) in (int,float):row[i] = round(row[i],3)
        row[INDEX['Original description / notes']] = canonical_json(note)
    value['notes'] = [r[INDEX['Original description / notes']] for r in value['rows']]
    value.update(policy=POLICY, view_hash=plan.digest(value['rows']))
    return value


def verify_owned(row, value, server):
    """Admit original full-precision rows or validated rounded rows, never mix them."""
    note = json.loads(row[INDEX['Original description / notes']])
    reporting = note.get('reporting')
    if reporting is None:
        return BASE_VERIFY(row,value,server)
    if (reporting.get('policy') != POLICY or reporting.get('decimal_places') != 3
            or reporting.get('selection_uses_unrounded_evidence') is not True):
        raise ValueError('Unknown reporting provenance')
    cfg = value['configs'][row[INDEX['Run_ID']]]
    step = row[INDEX['Selected_step']]
    record = value['proof'].get((cfg['run_id'],step)) if step else None
    expected,_ = original.make_row(cfg,row[INDEX['Status']],record,
        note['publication']['selector'],note['publication']['slot'])
    restored = list(row)
    for key in original.METRICS.values():
        raw = expected[INDEX[key]]
        displayed = round(raw,3) if type(raw) in (int,float) else raw
        if row[INDEX[key]] != displayed or reporting['raw_metrics'].get(key) != raw:
            raise ValueError('Rounded metric differs from immutable evidence: '+key)
        restored[INDEX[key]] = raw
    BASE_VERIFY(restored,value,server)


@contextmanager
def verified_rounding():
    previous = original.verify_owned
    original.verify_owned = verify_owned
    try:yield
    finally:original.verify_owned = previous


def format_requests(sheet_rows):
    return [{'repeatCell':{'range':{'sheetId':original.destination()['sheet_id'],
            'startRowIndex':n-1,'endRowIndex':n,'startColumnIndex':3,'endColumnIndex':25},
        'cell':{'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.000'}}},
        'fields':'userEnteredFormat.numberFormat'}} for n in sheet_rows]


def publish(root,server,*,apply=False,credential=None):
    from sheet_publish_v2.transport import GoogleSheets
    from panda_wv3_main_rgb.transport import reconcile_rows
    root=Path(root);adapter=GoogleSheets(credential or plan.ROOT/'gspread/account.json')
    value=project(root,server)
    with verified_rounding():
        receipt=reconcile_rows(adapter,value,root/'reporting/publication',server,apply=apply)
    if apply:
        requests=format_requests(receipt['sheet_rows'])
        if requests:
            adapter._request('POST',adapter.BASE+original.destination()['spreadsheet_id']+':batchUpdate',
                json={'requests':requests})
    atomic_json(root/'reporting/last_poll.json',receipt)
    atomic_json(root/'publication/last_poll.json',receipt)
    return receipt
