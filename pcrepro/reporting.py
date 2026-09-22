"""Local, resumable cycle reports; repeated selections are not extra runs."""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile

from pcrepro.common import ROOT, atomic_json, camp, read, run_dir, utcnow
from pcrepro.plan import CAMPAIGN_ID, RECIPE_ID, SERVERS, case_for, cases_for
from pcrepro.upload import HEADER, COST_KEYS, row_values, _verify_envelope


def _atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def write_csv(path, rows, columns=HEADER):
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
            if isinstance(value,(dict,list,tuple)) else value for key,value in row.items() if key in columns})
    _atomic_text(path,stream.getvalue())


def discover_runs(root, server, cycle=None):
    """Only existing local case directories; never materialize future cycles."""
    if server not in SERVERS: raise ValueError('PC-Repro only s3/s4/s5')
    if cycle is not None and (isinstance(cycle,bool) or not isinstance(cycle,int) or cycle<0):
        raise ValueError('Nonnegative integer cycle required')
    found=[]
    for path in (Path(root)/'work_dir').glob('PCREPRO_*'):
        if not path.is_dir(): continue
        try: case=case_for(path.name)
        except ValueError: continue
        if case.server==server and (cycle is None or case.cycle==cycle):
            if any((path/relative).is_file() for relative in ('official/summary.json',
                    'meta/training_status.json','meta/status.json')):
                found.append(case)
    return sorted(found,key=lambda c:(c.cycle,[v.dataset for v in cases_for(server,c.cycle)].index(c.dataset)))


def _stats(values):
    values=[float(v) for v in values]
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Finite nonempty report population required')
    mean=sum(values)/len(values)
    return dict(n=len(values),mean=mean,minimum=min(values),maximum=max(values),
        sample_std=math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else None)


def _cohort_summary(rows):
    groups=[]
    for cohort in ('PAPER_SEED2025_SERVER_REPEAT','PREDECLARED_NEW_SEED_REPEAT'):
        for dataset in ('WV3','QB','GF2','WV2'):
            selected=[r for r in rows if r['eval_complete'] and r['seed_cohort']==cohort and r['dataset']==dataset]
            if not selected: continue
            for identity in sorted({r['paper_identity_status'] for r in selected}):
                subset=[r for r in selected if r['paper_identity_status']==identity]
                values={key:_stats([r['EXACT_50000 '+key] for r in subset])
                        for key in ('fr_hqnr','fr_d_s','fr_d_lambda','rr_ergas','rr_scc')}
                groups.append(dict(dataset=dataset,seed_cohort=cohort,selection='EXACT_50000',
                    paper_identity_status=identity,unique_runs=len(subset),run_ids=[r['run_id'] for r in subset],
                    seeds=[r['seed'] for r in subset],metrics=values,
                    same_seed_server_repeats_are_independent_seeds=False,
                    WV2_is_zero_shot=dataset=='WV2',statistical_significance_claim=False))
    return groups


def rebuild(root=ROOT,server='s3',cycle=None):
    cases=discover_runs(root,server,cycle);rows=[];problems=[]
    for case in cases:
        try: rows.append(row_values(case.run_id,root))
        except (ValueError,KeyError,OSError) as exc:
            problems.append(dict(run_id=case.run_id,dataset=case.dataset,cycle=case.cycle,
                status='REPORT_BLOCKED_INTEGRITY',reason=f'{type(exc).__name__}: {exc}'))
    ids=[r['run_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate run IDs in local report')
    costs={key:sum(r[key] for r in rows if isinstance(r[key],(int,float)) and not isinstance(r[key],bool))
           for key in COST_KEYS}
    missing_costs={key:[r['run_id'] for r in rows if r[key]==''] for key in COST_KEYS}
    cycles={}
    for number in sorted({c.cycle for c in cases} | ({cycle} if cycle is not None else set())):
        registered=cases_for(server,number)
        existing={r['run_id']:r for r in rows if r['cycle']==number}
        complete=[c.run_id for c in registered if existing.get(c.run_id,{}).get('eval_complete')]
        cycles[str(number)]=dict(cycle=number,expected_run_ids=[c.run_id for c in registered],
            completed_run_ids=complete,missing_run_ids=[c.run_id for c in registered if c.run_id not in existing],
            complete=len(complete)==4,
            statuses={c.run_id:existing.get(c.run_id,{}).get('status','NOT_RECORDED') for c in registered},
            actual_optimizer_updates=sum(existing.get(c.run_id,{}).get('actual_updates',0) for c in registered),
            expected_training_runs=3,expected_zero_shot_evaluations=1,
            next_cycle_condition='controller cursor and user controls only; never metric target')
    spool=[]
    for path in (camp(root,server)/'upload_spool').glob('*.json'):
        try:
            envelope=read(path);case=case_for(envelope['run_id'])
            if case.server!=server or path.stem!=case.run_id:raise ValueError('Foreign upload spool')
            _verify_envelope(envelope,case)
        except (ValueError,KeyError,OSError) as exc:
            spool.append(dict(run_id=path.stem,status='UPLOAD_BLOCKED_INTEGRITY',
                last_error=f'{type(exc).__name__}: {exc}'));continue
        if envelope.get('status')!='READBACK_VERIFIED':
            spool.append(dict(run_id=envelope.get('run_id'),attempts=envelope.get('attempts',0),
                last_error=envelope.get('last_error',''),status=envelope.get('status','UNKNOWN')))
    report=dict(schema='PCREPRO_LOCAL_REPORT_v1',campaign_id=CAMPAIGN_ID,recipe_id=RECIPE_ID,
        server=server,cycle_filter=cycle,reported_at_utc=utcnow(),unique_runs=len(rows),
        completed_experiments=sum(bool(r['eval_complete']) for r in rows),
        completed_training_runs=sum(r['eval_complete'] and r['dataset']!='WV2' for r in rows),
        completed_zero_shot_evaluations=sum(r['eval_complete'] and r['dataset']=='WV2' for r in rows),
        incomplete_or_failed=[dict(run_id=r['run_id'],status=r['status'],actual_updates=r['actual_updates'])
                              for r in rows if not r['eval_complete']],
        integrity_problems=problems,costs=costs,costs_missing_for_runs=missing_costs,
        WV2_source_training_cost_counted_again=False,selections_count_as_extra_experiments=False,
        cohorts=_cohort_summary(rows),cycles=cycles,upload_pending=spool,
        performance_stop_enabled=False,maximum_cycle_count=None,cross_server_barrier=False)
    folder=camp(root,server)/'reports'
    suffix='all_cycles' if cycle is None else f'cycle_{cycle:06d}'
    write_csv(folder/(suffix+'_runs.csv'),rows)
    _atomic_text(folder/(suffix+'_runs.jsonl'),''.join(json.dumps(row,sort_keys=True,
        ensure_ascii=False,allow_nan=False)+'\n' for row in rows))
    atomic_json(folder/(suffix+'_summary.json'),report)
    return report


def cycle_report(root, server, cycle): return rebuild(root,server,cycle)
