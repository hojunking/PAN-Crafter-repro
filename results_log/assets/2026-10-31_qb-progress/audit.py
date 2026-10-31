"""Read-only audit of authenticated local QB evaluations; no inference/training."""
import csv
import json
import math
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
PACKAGES = ('panda_s1loss9', 'panda_sensor_refine', 'panda_sensor_focus', 'panda_qb_refit')
TARGET = {'HQNR': .9315, 'ERGAS': 3.5}
read = lambda p: json.loads(p.read_text())
mean = lambda xs: st.mean(xs) if xs else None


def row(e):
    return {'step': e['state_before']['completed_update'],
            **{k: e['fr']['metrics'][k] for k in ('HQNR', 'D_s', 'D_lambda')},
            **{k: e['rr']['metrics'][k] for k in ('ERGAS', 'PSNR')},
            'weights_saved': e.get('weights_saved'), 'checkpoint_path': e.get('checkpoint_path'),
            'weights_file_sha256': e.get('weights_file_sha256'),
            'aligner_state_sha256': e.get('aligner_state_sha256')}


def representative(rows):
    eligible = [r for r in rows if r['HQNR'] >= TARGET['HQNR']]
    return min(eligible, key=lambda r: (r['ERGAS'], -r['HQNR'], r['step'])) if eligible else max(
        rows, key=lambda r: (r['HQNR'], -r['ERGAS'], -r['step']))


def correlations(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx,my=st.mean(xs),st.mean(ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(
        sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))


report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'target': TARGET, 'runs': [],
          'limitations': ['Exploratory adaptive evaluation; not an independent final benchmark.',
                         'Same replica uses same initialization and data stream where width unchanged.',
                         'Two seeds do not establish confidence intervals; adjacent checkpoints are correlated.',
                         'No GPU inference and no source, config, checkpoint, or queue mutations.']}
evidence = {}
for package in PACKAGES:
    for statepath in sorted((ROOT / 'work_dir' / package).glob('*/s1/QB/campaign_state.json')):
        state = read(statepath)
        for run in state['runs']:
            folder = statepath.parent / 'runs' / run['run_id'] / 'attempt_01'
            if run['case'].startswith('S1_') or not (folder/'config.json').exists():
                continue
            cfg = read(folder/'config.json')
            evs = [read(p) for p in sorted(folder.glob('evaluation_*.json'))]
            assert all(all(e['qualification'].get(k) for k in
                ('same_state_verified', 'metrics_complete', 'metrics_finite')) for e in evs)
            if not evs:
                continue
            rows = [row(e) for e in evs]
            spec = cfg['stage2_spec']; train = cfg['training_spec']
            record = {'package': package, 'case': run['case'], 'replica': cfg['replica'],
                'status': run['status'], 'recorded_step': run.get('step'), 'last_evaluation_step': rows[-1]['step'],
                'source': str(folder.relative_to(ROOT)),
                'settings': {**{k: spec[k] for k in ('alpha', 'gamma', 'student_width')},
                    'lr': train['lr'], 'updates': cfg['planned_horizon'], 'weight_decay': train['weight_decay'],
                    'batch': train['effective_batch_size'], 'donor': cfg['donor_version'],
                    'donor_checkpoint_sha256': cfg.get('donor_checkpoint_sha256'),
                    'lambda_abs': cfg['stage1_loss_parameters']['lambda_abs'],
                    'lambda_epsilon': cfg['stage1_loss_parameters']['lambda_epsilon'],
                    'epsilon_radius': cfg['epsilon_radius_rr']},
                'rows': rows, 'representative': representative(rows),
                'HQNR_pass_count': sum(r['HQNR']>=.9315 for r in rows),
                'joint_pass_count': sum(r['HQNR']>=.9315 and r['ERGAS']<=3.5 for r in rows),
                'unconditional_min_ergas': min(rows, key=lambda r:r['ERGAS']),
                'mean_25_30K': {k: mean([r[k] for r in rows if 25000<=r['step']<=30000])
                    for k in ('HQNR','ERGAS','PSNR','D_s','D_lambda')}}
            traces = []
            for line in (folder/'train.jsonl').read_text().splitlines():
                line = json.loads(line)
                if 'trace' in line:
                    traces.append((line['step'], line['trace']))
            record['training_windows'] = {}
            for a,b in ((5000,10000),(15000,20000),(25000,30000)):
                selected = [t for s,t in traces if a<=s<=b]
                record['training_windows'][f'{a}_{b}'] = {
                    k: mean([t[k] for t in selected if k in t]) for k in (
                        'gt_raw','hard_weighted','loss','d_e_mean','r_q_roi_mean','w_h_mean',
                        'w_h_p95','w_h_max','warp_support_fraction')}
            last = evs[-1]
            record['shift_stats'] = {}
            for mode in ('rr','fr'):
                norms=[x['correction_norms_px'] for x in last[mode]['per_scene']]
                record['shift_stats'][mode]={'mean':mean(norms),'median':st.median(norms),'max':max(norms)}
            record['trajectory_correlations'] = {
                'HQNR_Ds':correlations([r['HQNR'] for r in rows],[r['D_s'] for r in rows]),
                'HQNR_Dlambda':correlations([r['HQNR'] for r in rows],[r['D_lambda'] for r in rows])}
            report['runs'].append(record)
            evidence[(run['case'],cfg['replica'])] = {e['state_before']['completed_update']:e for e in evs}

baseline = {'R1':('QB_A200_E030','R1'), 'R2':('QB_BASE','R2')}
report['paired_30K'] = []
for run in report['runs']:
    key = (run['case'],run['replica']); control=baseline.get(run['replica'])
    if not control or key==control or 30000 not in evidence[key]:
        continue
    current=evidence[key][30000]; original=evidence[control][30000]
    rows_current={p['scene_id']:p for p in current['rr']['per_scene']}
    rows_original={p['scene_id']:p for p in original['rr']['per_scene']}
    delta={i:rows_current[i]['metrics']['ERGAS']-rows_original[i]['metrics']['ERGAS'] for i in rows_current}
    runkeys=('HQNR','D_s','D_lambda','ERGAS','PSNR')
    paired={'case':run['case'],'replica':run['replica'],'control':control,
        'same_donor':current['aligner_state_sha256']==original['aligner_state_sha256'],
        'max_rr_shift_delta':max(abs(a-b) for i in rows_current
            for a,b in zip(rows_current[i]['correction_dy_dx'],rows_original[i]['correction_dy_dx'])),
        'aggregate_delta':{k:row(current)[k]-row(original)[k] for k in runkeys},
        'improved_scene_count':sum(v<0 for v in delta.values()),
        'worse_scene_count':sum(v>0 for v in delta.values()),
        'median_scene_delta':st.median(delta.values()), 'scene_deltas':delta,
        'mean_delta_excluding_scene0':mean([v for k,v in delta.items() if k!='QB:rr:0'])}
    report['paired_30K'].append(paired)

report['counts']={'statuses':dict(Counter(r['status'] for r in report['runs'])),
                  'runs':len(report['runs']), 'evaluation_count':sum(len(r['rows']) for r in report['runs']),
                  'joint_pass_count':sum(r['joint_pass_count'] for r in report['runs'])}
report['best_H_qualified'] = min((dict(case=r['case'],replica=r['replica'],**v)
    for r in report['runs'] for v in r['rows'] if v['HQNR']>=.9315), key=lambda r:r['ERGAS'])
teacher_dir = next((ROOT/'work_dir/panda_s1loss9').glob('*/s1/QB/runs/*S1_A200_E030*/attempt_01'))
teacher_evals = [read(p) for p in teacher_dir.glob('evaluation_*.json')]
teacher = next(e for e in teacher_evals if e['state_before']['completed_update']==44000)
student = evidence[('QB_BASE','R2')][26000]
teacher_comparison = {'selected_teacher': row(teacher),
    'teacher_minimum_RR_diagnostic_only':row(min(teacher_evals,key=lambda e:e['rr']['metrics']['ERGAS'])),
    'student_best':row(student),'teacher_source':str(teacher_dir.relative_to(ROOT))}
for mode, metric in (('rr','ERGAS'),('fr','HQNR')):
    ts={x['scene_id']:x for x in teacher[mode]['per_scene']}
    ss={x['scene_id']:x for x in student[mode]['per_scene']}
    ds={i:ss[i]['metrics'][metric]-ts[i]['metrics'][metric] for i in ss}
    teacher_comparison[mode]={'metric':metric,'student_minus_teacher_mean':mean(list(ds.values())),
        'per_scene_delta':ds,'student_better_count':sum(v<0 if mode=='rr' else v>0 for v in ds.values()),
        'max_shift_delta':max(abs(a-b) for i in ss for a,b in zip(
            ss[i]['correction_dy_dx'],ts[i]['correction_dy_dx']))}
report['teacher_comparison']=teacher_comparison
selection=read(next((ROOT/'work_dir/panda_s1loss9').glob('*/s1/QB/donors/S1_A200_E030/rre_selection.json')))
validation=selection['selected']['validation']
report['validation_availability']={
    'stage2_trajectory':'No stored independent validation reconstruction trajectory located; official evaluation contains rr/fr test only.',
    'selected_stage1':{'step':selection['selected']['completed_update'],
        'Val_ERGAS':validation['Val_ERGAS'],'RRE50':validation['RRE50'],
        'population':validation['sample_count'],'not_comparable_to_RR_test_mean':True}}

OUT.mkdir(parents=True,exist_ok=True)
(OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
with (OUT/'trajectories.csv').open('w') as f:
    fields=['package','case','replica','status','step','HQNR','ERGAS','PSNR','D_s','D_lambda']
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for run in report['runs']:
        for r in run['rows']:
            writer.writerow({k:run[k] if k in run else r[k] for k in fields})
print(json.dumps({'counts':report['counts'],'best':report['best_H_qualified'],
    'new':[{'case':r['case'],'replica':r['replica'],'status':r['status'],
            'settings':r['settings'],'rep':{k:r['representative'][k] for k in ('step','HQNR','ERGAS','PSNR','D_s','D_lambda')},
            'mean_25_30K':r['mean_25_30K']}
            for r in report['runs'] if r['package']=='panda_qb_refit'],
    'paired_new':[{k:v for k,v in p.items() if k!='scene_deltas'} for p in report['paired_30K'] if p['case'].startswith('QB2_')]},indent=2))
