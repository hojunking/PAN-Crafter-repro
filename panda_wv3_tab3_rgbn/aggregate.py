"""All paired repeats at fixed 50K; no metric-driven exclusion or best run choice."""
import csv
import io
import json
import statistics
from pathlib import Path
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from . import plan
from .evaluation import verify_record
from .publication import artifact

METRICS = ('FR_D_lambda','FR_RRE50','RR_ERGAS','RR_SAM','RR_PSNR','RR_RRE50','FR_HQNR','FR_D_s')


def observations(record):
    fr,rr = record['fr']['metrics'],record['rr']['metrics']
    return dict(FR_D_lambda=fr['D_lambda'],FR_RRE50=record['test_rre']['fr']['RRE50'],
        RR_ERGAS=rr['ERGAS'],RR_SAM=rr['SAM'],RR_PSNR=rr['PSNR'],
        RR_RRE50=record['test_rre']['rr']['RRE50'],FR_HQNR=fr['HQNR'],FR_D_s=fr['D_s'])


def aggregate(root, server, *, write=False):
    root = Path(root); cases = []
    for case in plan.stage1_cases(server):
        rows = []
        for repeat in plan.PLAN['repeats']:
            cfg = plan.resolve(server,case,repeat)
            path = root/'runs'/cfg['run_id']/'evaluation_0050000.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text()); verify_record(record,cfg)
            if record['step'] != 50000 or record['checkpoint']['step'] != 50000:
                raise ValueError('Fixed50 summary cannot use another checkpoint')
            if record['checkpoint']['model_sha256'] != record['model_sha256']:
                raise ValueError('Evaluation/checkpoint state mismatch')
            artifact(record['checkpoint'])
            rows.append(dict(repeat=repeat,seed=cfg['seed'],run_id=cfg['run_id'],
                evaluation_sha256=file_sha(path),checkpoint=record['checkpoint'],
                values=observations(record),rre_diagnostics=record['test_rre']))
        complete = len(rows)==3
        values = {}
        for key in METRICS:
            numbers = [r['values'][key] for r in rows]
            valid = [v for v in numbers if v is not None]
            # Missing runs/invalid scene RRE must not become a selected subset.
            values[key] = {'n':len(valid), 'mean':statistics.mean(valid) if complete and len(valid)==3 else None,
                'std':statistics.stdev(valid) if complete and len(valid)==3 else None}
        cases.append(dict(case_id=case,table_row=plan.resolve(server,case,1)['table_row'],
            status='COMPLETE_3_REPEATS' if complete else 'INCOMPLETE',n_runs=len(rows),metrics=values,runs=rows))
    result = dict(campaign_id=plan.CAMPAIGN_ID,step=50000,ddof=1,expected_repeats=[1,2,3],
        policy=plan.PLAN['aggregation'],rre='Each run: median across all 20 scenes; then mean/sample std across 3 runs',
        cases=cases)
    if write:
        folder = root/'summary'; folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'table3_fixed50.json',result)
        buffer = io.StringIO(); writer = csv.writer(buffer)
        writer.writerow(['Case','n_runs',*[x for k in METRICS for x in (k+'_mean',k+'_std',k+'_n')]])
        lines = ['# Table 3 — RGBN, fixed 50K', '',
            'All 3 predeclared repeats; mean ± sample standard deviation (ddof=1). Incomplete groups remain blank.', '',
            '| Case | n | '+' | '.join(METRICS)+' |', '|---|---:|'+ '|'.join(['---:']*len(METRICS))+'|']
        for case in cases:
            writer.writerow([case['case_id'],case['n_runs'],*[v for k in METRICS for v in
                (case['metrics'][k]['mean'],case['metrics'][k]['std'],case['metrics'][k]['n'])]])
            cells = ['—' if case['metrics'][k]['mean'] is None else
                     f"{case['metrics'][k]['mean']:.4f} ± {case['metrics'][k]['std']:.4f}" for k in METRICS]
            lines.append('| '+case['table_row']+' | '+str(case['n_runs'])+' | '+' | '.join(cells)+' |')
        (folder/'table3_fixed50.csv').write_text(buffer.getvalue())
        (folder/'table3_fixed50.md').write_text('\n'.join(lines)+'\n')
    return result
