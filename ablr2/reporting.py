"""Rebuild presentation tables from retained evidence, never filter failures."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile

from ablr2.common import camp, read_json, read, atomic_json, utcnow


def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({key for row in rows for key in row})
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
            if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()})
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',newline='') as output:
            output.write(stream.getvalue());output.flush();os.fsync(output.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def rebuild(root,server):
    folder=camp(root,server)
    state=read_json(folder/'state.json')
    ledger=folder/'all_attempts.jsonl'
    attempts=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.is_file() else []
    balanced,rechecks,verification,exploratory=[],[],[],[]
    for stage in state['stages']:
        if not stage.get('report_path'): continue
        report=read_json(stage['report_path'])
        if stage['kind'] in ('BOOT5','REFRESH5','VERIFY5'):
            rows=report.get('panelrows',[])
            if stage['kind']=='VERIFY5':
                verification.extend(dict(row,verification_status=stage.get('verification_status'),
                                         stage_id=stage['stage_id']) for row in rows)
            else:
                balanced.extend(rows)
            exploratory.extend(dict(row,reporting_role='EXPLORATORY_TEST_AWARE_ONLY_NOT_MAIN_TABLE') for row in rows)
        elif stage['kind']=='RECHECK5':
            rechecks.extend(dict(row,relation_id=stage['relation_id'],outcome=report['outcome'],
                stage_id=stage['stage_id'],independent_panel_claim=False) for row in report['records'].values())
    best={}
    for row in exploratory:
        key=(row['recipe_revision'],row['case_id'])
        if key not in best or row['RAW_MAX']['HQNR']>best[key]['RAW_MAX']['HQNR']: best[key]=row
    tables={'all_attempts':attempts,'complete_panel_metrics':balanced,'targeted_rechecks':rechecks,
            'exploratory_best':list(best.values()),'verification_results':verification}
    for name,rows in tables.items(): _csv(folder/'reports'/f'{name}.csv',rows)
    receipt=dict(rebuilt_at_utc=utcnow(),counts={key:len(rows) for key,rows in tables.items()},
                 full_precision=True,negative_observations_retained=True,source='IMMUTABLE_REPORTS_AND_APPEND_ONLY_ATTEMPTS')
    atomic_json(folder/'reports/table_manifest.json',receipt)
    return receipt
