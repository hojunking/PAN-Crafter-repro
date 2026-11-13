"""Fixed40K only, all three registered seeds; no best-step substitution."""
import csv
import io
import json
import statistics
from pathlib import Path
from directshift48.reporting import atomic_json
from . import plan
from .evaluation import verify_record

METRICS={'HQNR_corrected':('fr_corrected_pan','HQNR'),'Ds_corrected':('fr_corrected_pan','D_s'),
    'Dlambda':('fr_corrected_pan','D_lambda'),'ERGAS':('rr','ERGAS'),'SAM':('rr','SAM'),
    'PSNR':('rr','PSNR'),'HQNR_original':('fr','HQNR'),'Ds_original':('fr','D_s')}

def collect(root,server):
    rows=[]
    for case in plan.PLAN['case_order']:
        observations=[]
        for repeat in (1,2,3):
            cfg=plan.resolve(server,case,repeat)
            directory=plan.source_directory() if cfg['reuse_existing'] else Path(root)/'runs'/cfg['run_id']
            path=directory/'evaluation_0040000.json'
            if not path.is_file():continue
            e=json.loads(path.read_text());verify_record(e,cfg)
            if e['step']!=40000 or e['checkpoint']['step']!=40000:
                raise ValueError('Fixed40 summary cannot substitute another step')
            observations.append({'repeat':repeat,'seed':cfg['seed'],'evaluation_file':str(path),
                'checkpoint_sha256':e['checkpoint']['file_sha256'],'reused':cfg['reuse_existing'],
                'metrics':{k:e[split]['metrics'][m] for k,(split,m) in METRICS.items()}})
        n=len(observations)
        rows.append({'case':case,'n':n,'expected_n':3,'complete':n==3,'step':40000,
            'observations':observations,'mean':{k:statistics.mean(o['metrics'][k] for o in observations) if n else None for k in METRICS},
            'std':{k:statistics.stdev(o['metrics'][k] for o in observations) if n>=2 else None for k in METRICS}})
    return {'schema':'S2ABL5_FIXED40_ALL_SEEDS_SAMPLE_STD_V1','std_ddof':1,'fixed_donor':plan.PLAN['fixed_donor'],
        'ready_for_three_repeat_table':all(r['complete'] for r in rows),'rows':rows}

def write(root,server):
    result=collect(root,server);folder=Path(root)/'summary';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'fixed40_mean_std.json',result)
    out=io.StringIO();fields=['case','n','expected_n','complete','step']+[k+s for k in METRICS for s in ('_mean','_std')]
    writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for r in result['rows']:
        writer.writerow({**{k:r[k] for k in fields[:5]},**{k+s:r[t][k] for k in METRICS for s,t in (('_mean','mean'),('_std','std'))}})
    (folder/'fixed40_mean_std.csv').write_text(out.getvalue())
    return result
