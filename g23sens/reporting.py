"""Fixed-T0, server-local paired sensitivity; never an adaptive sweep policy."""
from pathlib import Path
import math
import statistics

from g23sens.common import ROOT, camp, read_json, atomic_json, object_sha

METRICS=('hqnr','d_lambda','d_s','ergas','sam','psnr','ssim','scc','q8','rmse','cc','jqm')
SELECTIONS=('EXACT_50000','RR_VAL_ERGAS_MIN')
PAIR_KEYS=('runtime_content_sha256','teacher_checkpoint_sha256','initial_U_sha256',
    'initial_A_sha256','stream_sha256','q_raw_sha256','binding_sha256')


def paired_deltas(summary,baseline,selection):
    if not summary.get('complete') or not baseline.get('complete'):
        raise ValueError('Only complete official evaluations enter paired comparisons')
    case,base=summary['case'],baseline['case']
    if (base['case_id']!='BASE' or base['run_id']!=case['local_baseline_run_id']
            or any(case[k]!=base[k] for k in ('server','cycle','seed','pair_group_id'))):
        raise ValueError('Baseline must be same-server/same-cycle/same-seed BASE')
    a,b=summary['provenance'],baseline['provenance']
    if any(not a.get(k) or a.get(k)!=b.get(k) for k in PAIR_KEYS):
        raise ValueError('Paired source/data/Teacher/initialization/actual stream differs')
    result={}
    for key in METRICS:
        domain='fr' if key in ('hqnr','d_lambda','d_s','jqm') else 'rr'
        result[key]=summary['selections'][selection][domain][key]-baseline['selections'][selection][domain][key]
    return dict(values=result,baseline_run_id=base['run_id'],baseline_attempt=baseline['attempt'],
        baseline_summary_sha256=object_sha(baseline),summary_sha256=object_sha(summary),
        selection=selection,scope='same-server/same-cycle paired Student seed; fixed T0')


def _stats(values):
    if not values:return dict(n=0,mean=None,sd=None,raw=[])
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite report value')
    return dict(n=len(values),mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,raw=values)


def summarize(summaries,server):
    if server not in ('s4','s5'):raise ValueError('Only sensitivity lanes s4/s5')
    if any(s['case']['server']!=server for s in summaries):raise ValueError('Cross-server baseline pooling forbidden')
    accepted={};failures=[]
    for summary in summaries:
        case=summary['case'];key=(case['run_id'],summary['attempt'])
        if key in accepted and accepted[key]!=summary:raise ValueError('Conflicting duplicate attempt evidence')
        accepted[key]=summary
    byrun={}
    for s in accepted.values():byrun.setdefault(s['case']['run_id'],[]).append(s)
    complete={}
    for rid,attempts in byrun.items():
        valid=[s for s in attempts if s.get('complete')]
        if len(valid)>1:raise ValueError('Multiple successful attempts must not become extra seed evidence')
        if valid:complete[rid]=valid[0]
        failures.extend(dict(run_id=rid,attempt=s['attempt'],case_id=s['case']['case_id'],cycle=s['case']['cycle'],
            seed=s['case']['seed'],status=s['status'],reason=s.get('reason','')) for s in attempts if not s.get('complete'))
    raw=[];groups={};completed_cycles=set()
    for rid,s in sorted(complete.items()):
        c=s['case'];base=complete.get(c['local_baseline_run_id'])
        if base is None:raise ValueError('Completed variant has no completed local BASE')
        for selection in SELECTIONS:
            pair=paired_deltas(s,base,selection)
            metrics={k:s['selections'][selection]['fr' if k in ('hqnr','d_lambda','d_s','jqm') else 'rr'][k] for k in METRICS}
            row=dict(run_id=rid,attempt=s['attempt'],cycle=c['cycle'],seed=c['seed'],case_id=c['case_id'],
                selection=selection,selection_alias_of=s['selections'][selection].get('alias_of'),raw=metrics,paired_delta=pair)
            raw.append(row);groups.setdefault((c['case_id'],selection),[]).append(row)
    cycles={s['case']['cycle'] for s in accepted.values()}
    for cycle in cycles:
        terminal=[s for s in accepted.values() if s['case']['cycle']==cycle and (s.get('complete') or s['status']=='DIVERGED')]
        if len({s['case']['case_id'] for s in terminal})==7:completed_cycles.add(cycle)
    aggregate=[]
    for failure in failures:
        for selection in SELECTIONS:groups.setdefault((failure['case_id'],selection),[])
    for (case_id,selection),rows in sorted(groups.items()):
        if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Repeated attempts/cycles cannot inflate independent seeds')
        bad={f['run_id'] for f in failures if f['case_id']==case_id and f['status']=='DIVERGED'}
        aggregate.append(dict(case_id=case_id,selection=selection,independent_student_seeds=len(rows),
            raw={k:_stats([r['raw'][k] for r in rows]) for k in METRICS},
            paired_delta={k:_stats([r['paired_delta']['values'][k] for r in rows]) for k in METRICS},
            diverged_runs=len(bad),technical_failure_attempts=sum(f['case_id']==case_id and f['status']!='DIVERGED' for f in failures)))
    return dict(schema='G23SENS_LOCAL_PAIRED_REPORT_v1',server=server,raw=raw,aggregate=aggregate,failures=failures,
        completed_cycles=sorted(completed_cycles),initial_three_seed_summary_ready=len(completed_cycles)>=3,
        early_stop=False,scope='Fixed T0 conditional Student-seed sensitivity; no Teacher-seed or factor-interaction claim',
        duplicate_server_BASE_is_independent_seed=False,selection_alias_is_independent_run=False)


def build_report(root=ROOT,server='s4'):
    from g23sens.postrun import verify_summary_for_upload
    from g23sens.upload import status_summary
    records=[]
    folders={p.parent for p in (camp(root,server)/'runs').glob('*/attempt*/config.json')}
    folders.update(p.parent.parent for p in (camp(root,server)/'runs').glob('*/attempt*/meta/config.resolved.yaml'))
    for folder in sorted(folders):
        cfg=folder/'meta/config.resolved.yaml' if (folder/'meta/config.resolved.yaml').exists() else folder/'config.json'
        if not (folder/'meta/training_status.json').exists() and not (folder/'meta/attempt_failure.json').exists():continue
        from g23sens.common import read_config
        config=read_config(cfg);field=config['g23sens'];case=field['case'];attempt=field['attempt']
        status=read_json(folder/'meta/training_status.json',{})
        if (not (folder/'meta/attempt_failure.json').exists() and status.get('status') in ('RUNNING','PAUSED_SAFE','TRAIN_COMPLETE_EVAL_PENDING')
                and not (folder/'official/COMPLETE.json').exists()):continue
        summary=status_summary(case['run_id'],root,attempt=attempt)
        if summary.get('complete'):summary=verify_summary_for_upload(case['run_id'],root,attempt=attempt)
        records.append(summary)
    result=summarize(records,server)
    atomic_json(camp(root,server)/'reporting/paired_results.json',result)
    return result
