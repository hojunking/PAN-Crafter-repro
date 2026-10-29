"""Read-only audit of saved QB evaluations and the bounded live Sheet snapshot.

No model loading, training imports, checkpoint changes, or Sheet writes.
Run from the repository root with a Python environment containing numpy/matplotlib.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results_log/assets/2026-10-29_sensor-results-analysis'
TARGETS = {'QB': (.9315, 3.5), 'GF2': (.9645, .522), 'WV3': (.9585, 2.040)}


def read(path):
    return json.loads(path.read_text())


def metrics(e):
    return {'step': e['state_before']['completed_update'],
            'HQNR': e['fr']['metrics']['HQNR'],
            'ERGAS': e['rr']['metrics']['ERGAS'],
            'D_s': e['fr']['metrics']['D_s'],
            'D_lambda': e['fr']['metrics']['D_lambda']}


def choose(rows, sensor):
    qualified = [r for r in rows if r['HQNR'] >= TARGETS[sensor][0]]
    return min(qualified, key=lambda r: (r['ERGAS'], -r['HQNR'], r['step'])) if qualified else max(rows, key=lambda r: (r['HQNR'], -r['ERGAS'], -r['step']))


report = {'captured_utc': datetime.now(timezone.utc).isoformat(),
          'targets': TARGETS, 'local_qb': [], 'donors': [], 'campaigns': [],
          'sheet_representatives': {}, 'same_step_comparisons': []}
raw = {}
traces = {}
for package in ('panda_s1loss9', 'panda_sensor_refine'):
    for statepath in sorted((ROOT / 'work_dir' / package).glob('*/s1/QB/campaign_state.json')):
        state = read(statepath)
        report['campaigns'].append({'path': str(statepath.relative_to(ROOT)), 'status': state['status'],
                                   'runs': [{k: r.get(k) for k in ('case', 'replica', 'status', 'step')} for r in state['runs']]})
        for run in state['runs']:
            if run['case'].startswith('S1_'):
                continue
            folder = statepath.parent / 'runs' / run['run_id'] / 'attempt_01'
            cfg = read(folder / 'config.json')
            ev = [read(p) for p in sorted(folder.glob('evaluation_*.json'))]
            if not ev:
                continue
            checks = [all(e['qualification'].get(k, False) for k in ('same_state_verified', 'metrics_complete', 'metrics_finite')) for e in ev]
            assert all(checks), (run['run_id'], checks)
            rows = [metrics(e) for e in ev]
            spec = cfg['stage2_spec']
            key = f"{run['case']}_R{cfg['replica']}" if isinstance(cfg['replica'], int) else f"{run['case']}_{cfg['replica']}"
            raw[key] = {r['step']: e for r, e in zip(rows, ev)}
            trace = [json.loads(s) for s in (folder / 'train.jsonl').read_text().splitlines()]
            trace = [r for r in trace if 'trace' in r and 'step' in r]
            traces[key] = {r['step']: r['trace'] for r in trace}
            last = ev[-1]
            shifts = {}
            for split in ('rr', 'fr'):
                v = np.asarray([s['correction_dy_dx'] for s in last[split]['per_scene']])
                norms = np.linalg.norm(v, axis=-1)
                shifts[split] = {'mean_norm': float(norms.mean()), 'median_norm': float(np.median(norms)),
                                 'max_norm': float(norms.max()), 'mean_dy_dx': v.mean(axis=0).tolist()}
            trace_means = {}
            for lo, hi in ((1000, 10000), (10000, 20000), (20000, 30000)):
                subset = [r['trace'] for r in trace if lo < r['step'] <= hi]
                trace_means[f'{lo}_{hi}'] = {k: float(np.mean([t[k] for t in subset])) for k in
                                           ('gt_raw', 'e_extra_raw', 'q_extra_raw', 'd_e_mean', 'q_roi_mean', 'r_q_roi_mean', 'w_h_mean', 'w_h_p95') if subset and all(k in t for t in subset)}
            report['local_qb'].append({'key': key, 'case': run['case'], 'status': run['status'],
                                      'train_step': run['step'], 'seed': cfg['seed'],
                                      'path': str(folder.relative_to(ROOT)),
                                      'donor_sha': cfg.get('donor_checkpoint_sha256'),
                                      'donor_step': cfg.get('donor_binding', {}).get('donor_step'),
                                      'stream_seeds': cfg.get('stream_seeds'),
                                      'student_seed': spec['seed'],
                                      'settings': {**{k: spec.get(k) for k in ('alpha', 'beta', 'gamma', 'student_width', 'total_updates')},
                                                   'lr': cfg['training_spec']['lr'],
                                                   'stage1_lambda_abs': cfg['stage1_loss_parameters']['lambda_abs'],
                                                   'stage1_lambda_epsilon': cfg['stage1_loss_parameters']['lambda_epsilon'],
                                                   'epsilon_radius_rr': cfg['epsilon_radius_rr'],
                                                   'scale': cfg['shift_scale']},
                                      'representative': choose(rows, 'QB'), 'max_HQNR': max(rows, key=lambda r: r['HQNR']),
                                      'min_ERGAS_unconstrained': min(rows, key=lambda r: r['ERGAS']),
                                      'joint_pass_steps': [r['step'] for r in rows if r['HQNR'] >= .9315 and r['ERGAS'] <= 3.5],
                                      'rows': rows, 'correction': shifts, 'trace_windows': trace_means})
        for donorpath in sorted((statepath.parent / 'donors').glob('*/rre_selection.json')):
            selected = read(donorpath)['selected']
            val = selected['validation']
            report['donors'].append({'case': donorpath.parent.name, 'step': selected['completed_update'],
                                     **{k: val[k] for k in ('RRE50', 'RRE90', 'Val_ERGAS')},
                                     'epsilon': {k: v for k, v in val['epsilon'].items() if isinstance(v, (int, float, str))}})

base = 'QB_A200_E030_R1'
report['matched_controls'] = []
baseline = next(r for r in report['local_qb'] if r['key'] == base)
for r in report['local_qb']:
    if r['key'] in ('QB_LR120_R1', 'QB_C40_R1', 'QB_H050_R1', 'QB_BASE_R2'):
        report['matched_controls'].append({'key': r['key'], 'same_donor': r['donor_sha'] == baseline['donor_sha'],
                                           'same_streams': r['stream_seeds'] == baseline['stream_seeds'],
                                           'same_student_init_seed': r['student_seed'] == baseline['student_seed']})
report['donor_shift_comparisons_10K'] = []
for key in ('QB_A200_E030_R4_R1', 'QB_A200_E010_R3_R1'):
    d = {'key': key, 'baseline': base}
    for split in ('rr', 'fr'):
        a = {s['scene_id']: np.asarray(s['correction_dy_dx']) for s in raw[base][10000][split]['per_scene']}
        b = {s['scene_id']: np.asarray(s['correction_dy_dx']) for s in raw[key][10000][split]['per_scene']}
        norms = np.asarray([np.linalg.norm(b[s] - a[s]) for s in a])
        d[split] = {'mean_vector_difference_px': float(norms.mean()), 'max_vector_difference_px': float(norms.max())}
    report['donor_shift_comparisons_10K'].append(d)
for key in raw:
    if key == base or 30000 not in raw[key] or 30000 not in raw[base]:
        continue
    paired = {'case': key, 'baseline': base, 'step': 30000}
    for split, metric in (('rr', 'ERGAS'), ('fr', 'HQNR')):
        a = {s['scene_id']: s['metrics'][metric] for s in raw[base][30000][split]['per_scene']}
        b = {s['scene_id']: s['metrics'][metric] for s in raw[key][30000][split]['per_scene']}
        assert a.keys() == b.keys()
        delta = np.array([b[s] - a[s] for s in a])
        paired[metric] = {'mean_delta': float(delta.mean()), 'median_delta': float(np.median(delta)),
                          'improved_scenes': int((delta < 0 if metric == 'ERGAS' else delta > 0).sum()),
                          'scene_count': len(delta), 'scene_deltas': dict(zip(a, delta.tolist()))}
    report['same_step_comparisons'].append(paired)

sheet = read(OUT.parent / '2026-10-29_sensor-results-sheet.json')
context = read(OUT.parent / '2026-10-29_sensor-results-sheet-context.json')
hard36 = read(OUT.parent / '2026-10-29_wv3-hard36-sheet.json')
wv3_controls = read(OUT.parent / '2026-10-29_wv3-controls-sheet.json')
for sensor, values in [('QB', sheet['tabs']['QB-ablations']['values']),
                       ('GF2', sheet['tabs']['GF2-ablations']['values']),
                       ('WV3', wv3_controls['values'] + context['wv3_tune']['values'] + hard36['values'])]:
    reps = []
    for r in values:
        if len(r) <= 40 or not isinstance(r[3], (int, float)):
            continue
        if '대표' not in str(r[2]):
            continue
        reps.append({'name': r[0], 'server': r[1], 'selection': r[2],
                     'HQNR': r[3], 'D_s': r[4], 'D_lambda': r[5], 'ERGAS': r[7],
                     'campaign': r[28], 'case': r[30], 'replica': r[31], 'step': r[40],
                     'params_M': r[17], 'FLOPs_G': r[18],
                     'joint_pass_current_target': r[3] >= TARGETS[sensor][0] and r[7] <= TARGETS[sensor][1]})
    report['sheet_representatives'][sensor] = reps

OUT.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
with OUT.with_suffix('.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['key', 'step', 'HQNR', 'ERGAS', 'D_s', 'D_lambda'], lineterminator='\n')
    writer.writeheader()
    for run in report['local_qb']:
        writer.writerows({'key': run['key'], **r} for r in run['rows'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
labels = {'QB_A200_E030_R1': 'BASE R1', 'QB_BASE_R2': 'BASE R2', 'QB_H050_R1': 'Hard x0.5 R1',
          'QB_LR120_R1': 'LR 1.2e-4 R1', 'QB_C40_R1': 'Cosine 40K R1',
          'QB_A200_E030_R4_R1': 'Stage1 radius 4 R1', 'QB_A200_E010_R3_R1': 'Stage1 lambda_eps .01 R1'}
for run in report['local_qb']:
    if run['key'] not in labels:
        continue
    for ax, metric in zip(axes[0], ('HQNR', 'ERGAS')):
        ax.plot([r['step'] / 1000 for r in run['rows']], [r[metric] for r in run['rows']],
                label=labels[run['key']], lw=1.6)
for ax, metric, threshold in zip(axes[0], ('HQNR', 'ERGAS'), TARGETS['QB']):
    ax.axhline(threshold, ls='--', color='black', lw=1, label=f'Target {threshold}')
    ax.set(xlabel='Student updates (K)', ylabel=metric, title=f'QB: {metric} trajectories')
    ax.grid(alpha=.2)
axes[0, 0].set_ylim(.83, .958)
axes[0, 1].set_ylim(3.49, 3.85)
axes[0, 0].legend(fontsize=8, loc='lower right')

ax = axes[1, 0]
for r in report['sheet_representatives']['GF2']:
    if not ('SENSOR5' in r['campaign'] and 'v2' in r['campaign']):
        continue
    if 'S1_' in r['case'] or 'TEACHER' in r['case']:
        continue
    ax.scatter(r['HQNR'], r['ERGAS'], s=55)
    ax.annotate(r['case'].replace('GF2_', ''), (r['HQNR'], r['ERGAS']), xytext=(4, 4), textcoords='offset points', fontsize=9)
ax.axvline(.9645, ls='--', color='black', lw=1)
ax.axhline(.522, ls='--', color='black', lw=1)
ax.set(xlabel='HQNR (higher better)', ylabel='ERGAS (lower better)', title='GF2: same RGB/L4 donor, published representatives', xlim=(.9638, .972), ylim=(.518, .600))
ax.grid(alpha=.2)

ax = axes[1, 1]
for r in report['sheet_representatives']['WV3']:
    if r['server'] != 's3' or not any(x in r['case'] for x in ('W128', 'A0125', 'LR120')):
        continue
    label = r['case'].replace('S100DIFFUBILC30', '') + ' R' + str(r['replica'])
    ax.scatter(r['HQNR'], r['ERGAS'], s=60, marker='o' if r['joint_pass_current_target'] else 'x')
    ax.annotate(label, (r['HQNR'], r['ERGAS']), xytext=(4, 4), textcoords='offset points', fontsize=9)
ax.axvline(.9585, ls='--', color='black', lw=1)
ax.axhline(2.040, ls='--', color='black', lw=1)
ax.set(xlabel='HQNR (higher better)', ylabel='ERGAS (lower better)', title='WV3: promising changes and Student seed variation', xlim=(.9582, .9610), ylim=(2.024, 2.092))
ax.grid(alpha=.2)
fig.suptitle('Saved evaluations + live Sheet snapshot | 2026-10-29\nSame-checkpoint metrics; Sheet snapshots do not certify remote runtime state', fontsize=13)
fig.savefig(OUT.with_suffix('.png'), dpi=180)
fig.savefig(OUT.with_suffix('.pdf'))
print(json.dumps({'runs': len(report['local_qb']), 'local_evaluations': sum(len(r['rows']) for r in report['local_qb']),
                  'local_joint_passes': sum(len(r['joint_pass_steps']) for r in report['local_qb']),
                  'artifacts': [str(OUT.with_suffix(s).relative_to(ROOT)) for s in ('.json', '.csv', '.png', '.pdf')]}, indent=2))
