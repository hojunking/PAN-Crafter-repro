"""Reproduce the Oct 30 audit; local immutable QB evaluations + bounded Sheet snapshot."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('')
TARGETS = {'GF2': {'HQNR': .9645, 'ERGAS': .522, 'PSNR': 45.08},
           'QB': {'HQNR': .9315, 'ERGAS': 3.5}, 'WV3': {'HQNR': .9585, 'ERGAS': 2.040}}
read = lambda p: json.loads(p.read_text())


def qualifies(row, sensor):
    t = TARGETS[sensor]
    return row['HQNR'] >= t['HQNR'] and row['ERGAS'] <= t['ERGAS']


def representative(rows, sensor):
    eligible = [r for r in rows if r['HQNR'] >= TARGETS[sensor]['HQNR']]
    return min(eligible, key=lambda r: (r['ERGAS'], -r['HQNR'], r['step'])) if eligible else max(
        rows, key=lambda r: (r['HQNR'], -r['ERGAS'], -r['step']))


report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'targets': TARGETS,
          'local_QB': [], 'sheet_rows': {}, 'limitations': [
              'GF2: published checkpoints, not all unpublished evaluations or per-scene errors.',
              'GF2 new focus provenance confirms all H/E passes are published.',
              'Adaptive test-set tuning is exploratory; two seeds do not establish confidence intervals.']}
for package in ('panda_s1loss9', 'panda_sensor_refine', 'panda_sensor_focus'):
    for path in sorted((ROOT / 'work_dir' / package).glob('*/s1/QB/campaign_state.json')):
        state = read(path)
        for run in state['runs']:
            folder = path.parent / 'runs' / run['run_id'] / 'attempt_01'
            if run['case'].startswith('S1_') or not (folder / 'config.json').exists():
                continue
            cfg = read(folder / 'config.json')
            rows = []
            for p in sorted(folder.glob('evaluation_*.json')):
                e = read(p)
                assert all(e['qualification'].get(k) for k in ('same_state_verified', 'metrics_complete', 'metrics_finite'))
                rows.append({'step': e['state_before']['completed_update'],
                             **{k: e['fr']['metrics'][k] for k in ('HQNR', 'D_s', 'D_lambda')},
                             **{k: e['rr']['metrics'][k] for k in ('ERGAS', 'PSNR')}})
            if rows:
                report['local_QB'].append({'package': package, 'case': run['case'], 'replica': cfg['replica'],
                    'status': run['status'], 'rows': rows, 'representative': representative(rows, 'QB'),
                    'pass_count': sum(qualifies(r, 'QB') for r in rows),
                    'source': str(folder.relative_to(ROOT)),
                    'settings': {**{k: cfg['stage2_spec'][k] for k in ('alpha', 'gamma', 'student_width')},
                                 'lr': cfg['training_spec']['lr'], 'updates': cfg['planned_horizon']}})

sheet = read(OUT.parent / '2026-10-30_sensor-results-sheet.json')
sheet['WV3'] = read(OUT.parent / '2026-10-30_wv3-sheet-refresh.json')
for sensor in TARGETS:
    rows = []
    prior = read(OUT.parent / '2026-10-30_gf2-prior-sheet.json')['values'] if sensor == 'GF2' else []
    for r in prior + sheet[sensor]['values']:
        if len(r) < 41 or not isinstance(r[3], (int, float)) or not isinstance(r[7], (int, float)):
            continue
        row = {'case': r[30], 'replica': ('R'+str(r[31]) if isinstance(r[31], int) else r[31]), 'campaign': r[28], 'step': r[40],
               'HQNR': r[3], 'ERGAS': r[7], 'PSNR': r[10], 'D_s': r[4], 'D_lambda': r[5],
               'selection': r[2], 'status': r[50] if len(r) > 50 else None}
        row['HE_pass'] = qualifies(row, sensor)
        row['new_target_pass'] = row['HE_pass'] and (sensor != 'GF2' or row['PSNR'] >= 45.08)
        rows.append(row)
    report['sheet_rows'][sensor] = rows
gf2 = [r for r in report['sheet_rows']['GF2'] if 'SENSOR_FOCUS' in r['campaign']]
groups = sorted({(r['case'], r['replica']) for r in gf2})
report['GF2_focus'] = []
for case, rep in groups:
    rows = [r for r in gf2 if (r['case'], r['replica']) == (case, rep)]
    proof = next(p for p in sheet['GF2_provenance'] if p['cfg']['case'] == case and p['cfg']['replica'] == rep)
    pub = proof['publication']
    assert pub['last_logged_status'] == 'COMPLETED'
    assert len({r['step'] for r in rows if r['HE_pass']}) == pub['passed_count']
    report['GF2_focus'].append({'case': case, 'replica': rep, 'candidate_count': pub['candidate_count'],
        'HE_pass_count': pub['passed_count'], 'HQNR_pass_count': pub['hqnr_qualified_count'],
        'three_metric_pass_count': sum(r['new_target_pass'] for r in rows),
        'representative': next(r for r in rows if '대표' in r['selection']),
        'max_PSNR_HE_qualified': max((r for r in rows if r['HE_pass']), key=lambda r: r['PSNR'])})
best = max((r for r in gf2 if r['HE_pass']), key=lambda r: r['PSNR'])
report['GF2_best'] = best
report['GF2_PSNR_gap_dB'] = 45.08 - best['PSNR']
report['GF2_required_geometric_mean_MSE_reduction_fraction'] = 1 - 10 ** (-report['GF2_PSNR_gap_dB'] / 10)
OUT.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
with OUT.with_suffix('.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=['dataset', 'case', 'replica', 'step', 'HQNR', 'ERGAS', 'PSNR', 'D_s', 'D_lambda'], lineterminator='\n')
    w.writeheader()
    for run in report['local_QB']:
        w.writerows(dict(dataset='QB', case=run['case'], replica=run['replica'], **r) for r in run['rows'])
    for r in gf2:
        w.writerow({k: ('GF2' if k == 'dataset' else r[k]) for k in w.fieldnames})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
for run in report['GF2_focus']:
    rows = sorted([r for r in gf2 if (r['case'], r['replica']) == (run['case'], run['replica'])], key=lambda r: r['step'])
    axes[0].plot([r['step']/1000 for r in rows], [r['PSNR'] for r in rows], '.-', label=f"{run['case'].replace('GF2_', '')} {run['replica']}")
    axes[1].scatter([r['HQNR'] for r in rows], [r['PSNR'] for r in rows], s=18)
axes[0].axhline(45.08, color='black', ls='--')
axes[1].axhline(45.08, color='black', ls='--')
axes[1].axvline(.9645, color='black', ls='--')
axes[0].set(xlabel='Student updates (K)', ylabel='PSNR (dB)', title='GF2: published H/E-qualified checkpoints')
axes[0].legend(fontsize=7)
axes[1].set(xlabel='HQNR', ylabel='PSNR (dB)', title='GF2: joint target remains unmet')
for run in report['local_QB']:
    if run['package'] == 'panda_sensor_focus' or run['case'] in ('QB_BASE', 'QB_A200_E030'):
        rows = [r for r in run['rows'] if r['step'] >= 15000]
        axes[2].plot([r['step']/1000 for r in rows], [r['ERGAS'] for r in rows], label=run['case']+' '+run['replica'], alpha=.8)
axes[2].axhline(3.5, color='black', ls='--')
axes[2].set(xlabel='Student updates (K)', ylabel='ERGAS', title='QB: new trials remain near the plateau', ylim=(3.495,3.59))
axes[2].legend(fontsize=6)
for ax in axes: ax.grid(alpha=.2)
fig.savefig(OUT.with_suffix('.png'), dpi=180)
fig.savefig(OUT.with_suffix('.pdf'))
print(json.dumps({'GF2_best': best, 'GF2_gap': report['GF2_PSNR_gap_dB'],
                 'QB_focus': [{k:r[k] for k in ('case','replica','representative','pass_count')} for r in report['local_QB'] if r['package']=='panda_sensor_focus']}, ensure_ascii=False))
