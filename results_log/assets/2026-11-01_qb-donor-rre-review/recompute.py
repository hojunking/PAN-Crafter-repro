"""CPU-only audit of existing evaluations; does not infer, train, or edit queues."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def evaluations(directory):
    rows = []
    for path in sorted(directory.glob('evaluation_*.json')):
        data = read(path)
        rows.append(dict(step=int(path.stem.split('_')[-1]),
                         H=data['fr']['metrics']['HQNR'],
                         E=data['rr']['metrics']['ERGAS']))
    return rows


def representative(rows):
    eligible = [row for row in rows if row['H'] >= .9315]
    return min(eligible, key=lambda row: row['E']) if eligible else max(rows, key=lambda row: row['H'])


def association(rows, metric):
    x = np.array([row['RRE50'] for row in rows])
    y = np.array([row[metric] for row in rows])
    return dict(n=len(rows), pearson=float(np.corrcoef(x, y)[0, 1]),
                slope_per_001px=float(np.polyfit(x, y, 1)[0] * .01))


receipts = {}
for package in ['panda_s1loss9', 'panda_sensor_refine']:
    for path in (ROOT / 'work_dir' / package).glob('*/s1/QB/donors/*/rre_selection.json'):
        selected = read(path)['selected']
        receipts[selected['source_run_id']] = (path, selected)

cases = {'QB_A200_E003', 'QB_A020_E003', 'QB_A050_E003', 'QB_A200_E0003',
         'QB_A200_E030', 'QB_A020_E0003', 'QB_A200_E030_R4', 'QB_A100_E030_R3'}
donors = []
diagnostics = read(ROOT / 'results_log/assets/2026-11-01_s1_qb-ergas/disp_all.json')
for key, row in diagnostics.items():
    if row['case'] not in cases or row['rep'] != 'R1' or row['step'] != 30000:
        continue
    path, selected = receipts[row['donor']]
    cfg, val = selected['source_config'], selected['validation']
    run_id = key.split('@')[0]
    official_path = next((ROOT / 'work_dir').glob(f'panda_*/*/s1/QB/runs/{run_id}/attempt_01/evaluation_0030000.json'))
    official = read(official_path)
    donors.append(dict(case=row['case'], donor=row['donor'], lambda_abs=cfg['lambda_abs'],
        lambda_epsilon=cfg['lambda_epsilon'], radius=cfg.get('epsilon_radius_rr', 3.),
        donor_seed=cfg['seed'], donor_step=selected['completed_update'],
        RRE50=val['RRE50'], RRE90=val['RRE90'], student_step=30000,
        H=official['fr']['metrics']['HQNR'], E=official['rr']['metrics']['ERGAS'],
        E_glob_oracle=row['E_glob_oracle'], diagnostic_E=row['E'],
        source_receipt=str(path.relative_to(ROOT)), source_evaluation=str(official_path.relative_to(ROOT))))
assert len(donors) == 8
donors.sort(key=lambda row: row['RRE50'])
stats = {group: {metric: association(rows, metric) for metric in ['E', 'E_glob_oracle']}
         for group, rows in [('all_complete', donors), ('radius3_only', [r for r in donors if r['radius'] == 3])]}

s1 = []
patterns = ['panda_s1loss9/*/s1/QB/runs/*QB_A200_E030*/attempt_01',
            'panda_sensor_refine/*/s1/QB/runs/*QB_BASE*R2*/attempt_01',
            'panda_qb_structure/*/s1/QB/runs/*QB4_CONTROL*R3*/attempt_01']
for replica, pattern in enumerate(patterns, 1):
    path = next((ROOT / 'work_dir').glob(pattern))
    rows = evaluations(path)
    s1.append(dict(replica=replica, seed=read(path / 'config.json')['seed'],
                   representative=representative(rows), final=rows[-1], source=str(path.relative_to(ROOT))))
sheet = read(OUT / 's3_sheet_snapshot.json')
s3 = sorted([r for r in sheet['rows'] if r['case'] in ['QB3_CONTROL', 'QB4_CONTROL']], key=lambda r: r['replica'])
paired = []
for local, remote in zip(s1, s3):
    assert local['seed'] == remote['seed']
    paired.append(dict(replica=local['replica'], seed=local['seed'],
        s1_E=local['representative']['E'], s3_E=remote['E'],
        delta_E=remote['E'] - local['representative']['E']))
baseline_means = dict(s1=float(np.mean([r['s1_E'] for r in paired])),
                      s3=float(np.mean([r['s3_E'] for r in paired])))
b96 = []
for case in ['QB5_B96C15LR100', 'QB5_B96C15LR200']:
    rows = [r for r in sheet['rows'] if r['case'] == case]
    b96.append(dict(case=case, mean_E=float(np.mean([r['E'] for r in rows])),
                    delta_vs_s3_control_R1R2=float(np.mean([r['E'] for r in rows]) - np.mean([r['E'] for r in s3[:2]])), rows=rows))

campaign = ROOT / 'work_dir/panda_qb40/PANDA_S13_QB40_LOCALSTAGE1_20261101_v1/s1/QB'
active_path = next(campaign.glob('runs/*C60LR050*R1*/attempt_01'))
active_rows = evaluations(active_path)
last_train = json.loads((active_path / 'train.jsonl').read_text().splitlines()[-1])
active = dict(state=read(active_path / 'run_state.json'), latest_train=last_train,
              latest_evaluation=active_rows[-1], representative=representative(active_rows),
              delta_at_30K_vs_baseline_R1=next(r['E'] for r in active_rows if r['step'] == 30000) - s1[0]['final']['E'],
              joint_pass_count=sum(r['H'] >= .9315 and r['E'] <= 3.5 for r in active_rows))
result = dict(recorded_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    scientific_changes=False, queue_changes=False,
    notes=['Oracle-shift-corrected ERGAS is a diagnostic using GT, not an official result.',
           'Eight complete runs include one radius-4 donor; early-stopped donors are not in endpoint correlation.',
           'Cross-server comparisons use selected representatives, not common final checkpoints.',
           'Same Student seeds are paired; Stage1 seed and server differ.',
           'Only two radius-3 lambda_epsilon=.03 donors are in this correlation subset.'],
    donors=donors, association=stats, s1_baselines=s1, s3_baselines=s3,
    cross_server_paired=paired, baseline_representative_means=baseline_means,
    s3_batch96=b96, current_c60=active)
(OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['recorded_at', 'association', 'cross_server_paired', 'baseline_representative_means', 's3_batch96', 'current_c60']}, indent=2))
