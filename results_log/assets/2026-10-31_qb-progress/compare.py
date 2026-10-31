"""Summarize local trajectories and the bounded live Sheet snapshot; CPU only."""
import hashlib
import gzip
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
read = lambda p: json.loads(p.read_text())
audit = read(OUT / 'audit.json')
sheet = json.loads(gzip.decompress((OUT / 'sheet-snapshot.json.gz').read_bytes()))
runs = {(r['case'], r['replica']): r for r in audit['runs']}
base = {'R1': runs['QB_A200_E030', 'R1'], 'R2': runs['QB_BASE', 'R2']}
qb_rows = sheet['sheets']['QB-ablations']['values'][1:]
new = [r for r in audit['runs'] if r['package'] == 'panda_qb_refit']

report = {'snapshot_utc': sheet['captured_utc'], 's1_checks': [], 's1_representative_pairs': [],
          's3_pairs': [], 's3_rows': [], 'sheet_anomalies': []}
for r in qb_rows:
    if not isinstance(r[7], (int, float)):
        report['sheet_anomalies'].append({'case': r[30], 'replica': r[31], 'field': 'ERGAS', 'value': r[7]})
for r in new + [base['R2']]:
    v = r['representative']
    folder = ROOT / r['source']
    path = folder / 'models' / f"step_{v['step']:07d}.pt"
    side = read(path.with_suffix('.json'))['artifact']
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    sr = next(x for x in qb_rows if x[29] == folder.parent.name and x[38] == v['step'])
    valid = actual == side['file_sha256'] == sr[41]
    metrics_match = all(abs(v[k] - sr[c]) < 1e-12 for k, c in [('HQNR', 3), ('ERGAS', 7), ('PSNR', 10)])
    assert valid and metrics_match
    report['s1_checks'].append({'case': r['case'], 'replica': r['replica'], 'step': v['step'],
        'path': str(path.relative_to(ROOT)), 'sha256': actual, 'bytes': path.stat().st_size,
        'file_sidecar_sheet_hash_equal': valid, 'sheet_metrics_equal': metrics_match})
for r in new:
    rep = r['replica']; b = base[rep]['representative']; v = r['representative']
    report['s1_representative_pairs'].append({'case': r['case'], 'replica': rep,
        'HQNR': v['HQNR'], 'ERGAS': v['ERGAS'], 'step': v['step'],
        'delta_HQNR': v['HQNR'] - b['HQNR'], 'delta_ERGAS': v['ERGAS'] - b['ERGAS']})
for i, r in enumerate(qb_rows, 6):
    if r[1] != 's3':
        continue
    n = json.loads(r[60]); p = n['publication']; cfg = n['extra_parameters']['configuration']
    record = {'sheet_row': i, 'case': r[30], 'replica': r[31], 'step': r[38],
        'HQNR': r[3], 'ERGAS': r[7], 'PSNR': r[10], 'D_s': r[4], 'D_lambda': r[5],
        'status': r[50], 'last_observed_at': p['last_observed_at'],
        'candidate_count': p['candidate_count'], 'planned_horizon': p['planned_horizon'],
        'weights_saved': p['artifact']['weights_saved'], 'artifact_verification': p['artifact']['verification'],
        'checkpoint_sha256': p['artifact']['file_sha256'], 'checkpoint_path': p['artifact']['checkpoint_path'],
        'donor_sha256': cfg.get('donor_checkpoint_sha256'), 'donor_origin_server': cfg['donor_origin_server']}
    if cfg['stage2']:
        record['settings'] = {k: cfg['stage2_spec'][k] for k in ('alpha', 'beta', 'gamma')}
        record['settings'].update(lr=cfg['training_spec']['lr'], weight_decay=cfg['training_spec']['weight_decay'])
    else:
        record['validation'] = n['extra_parameters']['selector_receipt']['selected']['validation']
    report['s3_rows'].append(record)
controls = {r['replica']: r for r in report['s3_rows'] if r['case'] == 'QB3_CONTROL'}
for r in report['s3_rows']:
    if r['case'].startswith('S1_') or r['case'] == 'QB3_CONTROL':
        continue
    b = controls[r['replica']]
    assert r['donor_sha256'] == b['donor_sha256']
    report['s3_pairs'].append({'case': r['case'], 'replica': r['replica'],
        'delta_ERGAS': r['ERGAS'] - b['ERGAS'], 'delta_HQNR': r['HQNR'] - b['HQNR'],
        'comparison': 'same-server same-donor same-seed, independently selected representative steps'})

# C25: separate matched-update comparison from best-of-run comparison.
report['c25_matched25K'] = []
for rep in ('R1', 'R2'):
    r = runs['QB2_C25', rep]; b = base[rep]
    cur = next(v for v in r['rows'] if v['step'] == 25000)
    orig = next(v for v in b['rows'] if v['step'] == 25000)
    report['c25_matched25K'].append({'replica': rep,
        **{f'delta_{k}': cur[k] - orig[k] for k in ('HQNR', 'ERGAS', 'D_s', 'D_lambda')}})

report['runtime_snapshot'] = {}
for package in ('panda_qb_refit', 'panda_tune30'):
    state_path = next((ROOT / 'work_dir' / package).glob('*/s1/*/campaign_state.json'))
    state = read(state_path)
    report['runtime_snapshot'][package] = {'status': state['status'],
        'runs': [{k: r.get(k) for k in ('case', 'replica', 'status', 'step', 'started_at', 'completed_at')}
                 for r in state['runs']]}
report['runtime_snapshot']['priority'] = read(ROOT / 'work_dir/panda_qb_refit/priority_s1/status.json')
for pkg in ('panda_qb_refit', 'panda_tune30'):
    p = next((ROOT / 'work_dir' / pkg).glob('**/last_poll.json'))
    report['runtime_snapshot'][pkg]['publisher'] = read(p)
(OUT / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
cases = [('BASE', None), ('LR080', 'QB2_C30LR080'), ('LR050', 'QB2_C30LR050'),
         ('C25', 'QB2_C25'), ('AG000', 'QB2_C30AG000'), ('G025', 'QB2_C30G025')]
for label, name in cases:
    pair = [base[rep] if name is None else runs[name, rep] for rep in ('R1', 'R2')]
    steps = sorted(set(v['step'] for v in pair[0]['rows']) & set(v['step'] for v in pair[1]['rows']))
    means = [st.mean(next(v['ERGAS'] for v in r['rows'] if v['step'] == s) for r in pair) for s in steps]
    axs[0, 0].plot([s/1000 for s in steps], means, label=label, lw=1.7)
axs[0, 0].set(xlim=(10, 30), ylim=(3.49, 3.68), xlabel='Student updates (K)', ylabel='RR ERGAS',
              title='s1: mean RR ERGAS across two seeds')
axs[0, 0].axhline(3.5, color='black', ls='--', lw=1)
axs[0, 0].legend(fontsize=8, ncol=3)
for idx, rep in enumerate(('R1', 'R2')):
    vals = [(base[rep] if name is None else runs[name, rep])['representative']['ERGAS'] for _, name in cases]
    axs[0, 1].scatter([i + (idx-.5)*.18 for i in range(len(cases))], vals, label=rep, s=45)
axs[0, 1].set(xticks=range(len(cases)), xticklabels=[c[0] for c in cases], ylim=(3.495, 3.565),
              ylabel='Representative ERGAS', title='s1: no consistent paired improvement')
axs[0, 1].axhline(3.5, color='black', ls='--', lw=1); axs[0, 1].legend(fontsize=8)
s3cases = ['QB3_CONTROL', 'QB3_WD000', 'QB3_WD100', 'QB3_A075']
for idx, rep in enumerate((1, 2)):
    vals = [next(r['ERGAS'] for r in report['s3_rows'] if r['case'] == c and r['replica'] == rep) for c in s3cases]
    axs[1, 0].scatter([i + (idx-.5)*.18 for i in range(len(vals))], vals, label=f'R{rep}', s=45)
axs[1, 0].set(xticks=range(4), xticklabels=['CONTROL', 'WD=0', 'WD=0.1', 'alpha=0.75'],
              ylim=(3.495, 3.55), ylabel='Representative ERGAS', title='s3: all three changes reverse effect across seeds')
axs[1, 0].axhline(3.5, color='black', ls='--', lw=1); axs[1, 0].legend(fontsize=8)
for label, name in cases:
    pair = [base[rep] if name is None else runs[name, rep] for rep in ('R1', 'R2')]
    vs = [r['representative'] for r in pair]
    axs[1, 1].plot([v['HQNR'] for v in vs], [v['ERGAS'] for v in vs], 'o-', label=label, alpha=.8)
axs[1, 1].axhline(3.5, color='black', ls='--', lw=1)
axs[1, 1].axvline(.9315, color='black', ls='--', lw=1)
axs[1, 1].set(xlabel='FR HQNR', ylabel='RR ERGAS', ylim=(3.495, 3.565),
              title='s1: higher HQNR can coexist with worse RR')
axs[1, 1].legend(fontsize=8, ncol=2)
for ax in axs.flat:
    ax.grid(alpha=.18, axis='y'); ax.set_axisbelow(True)
fig.savefig(OUT / 'comparison.png', dpi=180)
fig.savefig(OUT / 'comparison.pdf')
print(json.dumps({'checkpoint_checks': len(report['s1_checks']), 's3_pairs': report['s3_pairs'],
                  'c25_matched25K': report['c25_matched25K'], 'sheet_anomalies': report['sheet_anomalies']}, indent=2))
