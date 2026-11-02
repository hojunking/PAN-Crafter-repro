"""Read-only, CPU-only QB evaluation audit. No inference or queue changes."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARGET_H, TARGET_E = .9315, 3.5


def read(path):
    return json.loads(path.read_text())


def one(pattern):
    paths = list((ROOT / 'work_dir').glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return paths[0]


def metrics(data, step):
    fr, rr = data['fr']['metrics'], data['rr']['metrics']
    return dict(step=step, H=fr['HQNR'], E=rr['ERGAS'], Ds=fr['D_s'], Dl=fr['D_lambda'])


def representative(rows):
    eligible = [r for r in rows if r['H'] >= TARGET_H]
    return dict(**(min(eligible, key=lambda r: r['E']) if eligible else max(rows, key=lambda r: r['H'])),
                rule='HQNR_QUALIFIED_MIN_ERGAS' if eligible else 'BEST_HQNR_FALLBACK')


def run(pattern):
    path = one(pattern)
    rows = [metrics(read(p), int(p.stem.split('_')[-1])) for p in sorted(path.glob('evaluation_*.json'))]
    return dict(source=str(path.relative_to(ROOT)), seed=read(path / 'config.json')['seed'],
                rows=rows, final=rows[-1], representative=representative(rows),
                h_qualified_count=sum(r['H'] >= TARGET_H for r in rows),
                joint_pass_count=sum(r['H'] >= TARGET_H and r['E'] <= TARGET_E for r in rows))


def means(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ['H', 'E', 'Ds', 'Dl']}


patterns = ['panda_s1loss9/*/s1/QB/runs/*QB_A200_E030*/attempt_01',
            'panda_sensor_refine/*/s1/QB/runs/*QB_BASE*R2*/attempt_01',
            'panda_qb_structure/*/s1/QB/runs/*QB4_CONTROL*R3*/attempt_01']
a2 = [run(p) for p in patterns]
a4 = [run(f'panda_qb_rre/*/s1/QB/runs/*QB5_A400_E030*R{i}*/attempt_01') for i in (1, 2, 3)]
assert [r['seed'] for r in a2] == [r['seed'] for r in a4]
paired, scene_changes = [], []
for replica, (old, new) in enumerate(zip(a2, a4), 1):
    delta = {k: new['final'][k] - old['final'][k] for k in ['H', 'E', 'Ds', 'Dl']}
    assert old['final']['step'] == new['final']['step'] == 30000
    scenes_old = read(ROOT / old['source'] / 'evaluation_0030000.json')['fr']['per_scene']
    scenes_new = read(ROOT / new['source'] / 'evaluation_0030000.json')['fr']['per_scene']
    assert [s['scene_id'] for s in scenes_old] == [s['scene_id'] for s in scenes_new]
    rows = []
    for before, after in zip(scenes_old, scenes_new):
        x, y = before['metrics'], after['metrics']
        dl = -(y['D_lambda'] - x['D_lambda']) * (1 - (x['D_s'] + y['D_s']) / 2)
        ds = -(y['D_s'] - x['D_s']) * (1 - (x['D_lambda'] + y['D_lambda']) / 2)
        dh = y['HQNR'] - x['HQNR']
        assert np.isclose(dl + ds, dh)
        rows.append(dict(scene_id=before['scene_id'], delta_H=dh, Dl_contribution=dl, Ds_contribution=ds))
    scene_changes.append(dict(replica=replica, scenes=rows))
    paired.append(dict(replica=replica, delta=delta,
                       h_worse_scenes=sum(s['delta_H'] < 0 for s in rows),
                       Ds_contribution=float(np.mean([s['Ds_contribution'] for s in rows])),
                       Dl_contribution=float(np.mean([s['Dl_contribution'] for s in rows]))))
fixed_means = dict(A2=means([r['final'] for r in a2]), A4=means([r['final'] for r in a4]))
fixed_delta = {k: fixed_means['A4'][k] - fixed_means['A2'][k] for k in fixed_means['A2']}
ds_fraction = float(np.mean([r['Ds_contribution'] for r in paired]) / fixed_delta['H'])

donors = []
for group, pattern in [('A2', 'panda_s1loss9/*/s1/QB/donors/S1_A200_E030/rre_selection.json'),
                       ('A4', 'panda_qb_rre/*/s1/QB/donors/QB5_S1_A400_E030/rre_selection.json')]:
    path = one(pattern)
    selected = read(path)['selected']
    val = selected['validation']
    evaluation_path = ROOT / Path(selected['source_evaluation_path']).relative_to('/workspace')
    data = read(evaluation_path)
    shifts = {}
    for domain in ['rr', 'fr']:
        arr = np.array([s['correction_dy_dx'] for s in data[domain]['per_scene']])
        norms = np.linalg.norm(arr, axis=1)
        shifts[domain] = dict(mean_dy_dx=arr.mean(axis=0).tolist(), median_norm=float(np.median(norms)),
                              p90_norm=float(np.percentile(norms, 90)), max_norm=float(norms.max()),
                              per_scene=arr.tolist())
    donors.append(dict(group=group, source=str(path.relative_to(ROOT)), step=selected['completed_update'],
                       seed=selected['source_config']['seed'],
                       validation={k: v for k, v in val.items() if k != 'per_sample'}, shifts=shifts))
fr_delta = np.array(donors[1]['shifts']['fr']['per_scene']) - np.array(donors[0]['shifts']['fr']['per_scene'])
fr_delta_norm = np.linalg.norm(fr_delta, axis=1)
shift_change = dict(median_vector_difference=float(np.median(fr_delta_norm)),
                    p90_vector_difference=float(np.percentile(fr_delta_norm, 90)),
                    max_vector_difference=float(fr_delta_norm.max()))

sheet = read(OUT / 'sheet_snapshot.json')
students = [r for r in sheet['rows'] if r['role'] == 'Student+FrozenAligner']
s3a2 = sorted([r for r in students if r['server'] == 's3' and r['case'] in ['QB3_CONTROL', 'QB4_CONTROL']], key=lambda r: r['replica'])
s3a4 = sorted([r for r in students if r['server'] == 's3' and r['campaign'].endswith('QB_RRE_FOCUS_20261101_v1')], key=lambda r: r['replica'])
assert len(s3a2) == len(s3a4) == 3
s3paired = [dict(replica=x['replica'], delta_H=y['H'] - x['H'], delta_E=y['E'] - x['E']) for x, y in zip(s3a2, s3a4)]
valid = [r for r in students if isinstance(r['H'], (float, int)) and isinstance(r['E'], (float, int))]
passes = [r for r in valid if r['H'] >= TARGET_H and r['E'] <= TARGET_E]
missing = [dict(row=r['row'], case=r['case'], replica=r['replica']) for r in students if r not in valid]
c60 = run('panda_qb40/*/s1/QB/runs/*C60LR050*R1*/attempt_01')
missing_resolved = run('panda_sensor_refine/*/s1/QB/runs/*QB_H050*R1*/attempt_01')
late_training = {}
for label, item in [('A2', a2[0]), ('A4', a4[0])]:
    logs = [json.loads(line) for line in (ROOT / item['source'] / 'train.jsonl').read_text().splitlines()]
    traces = [r['trace'] for r in logs if r.get('step', 0) >= 25000 and 'trace' in r]
    fields = ['d_e_mean', 'q_roi_mean', 'r_q_roi_mean', 'w_h_mean', 'w_h_p95', 'warp_support_fraction']
    late_training[label] = dict(logged_batch_count=len(traces),
        means={k: float(np.mean([r[k] for r in traces])) for k in fields})
weight_checks = []
for item in a4:
    paths = [ROOT / item['source'] / 'models' / f'step_{row["step"]:07d}.pt' for row in item['rows']]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    weight_checks.append(dict(seed=item['seed'], checkpoints_present=len(paths),
                             final_path=str(paths[-1].relative_to(ROOT))))
result = dict(recorded_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              inference_performed=False, queue_changed=False,
              target=dict(HQNR_min=TARGET_H, ERGAS_max=TARGET_E),
              s1_A2=a2, s1_A4=a4, s1_paired_30K=paired, s1_fixed_30K_means=fixed_means,
              s1_fixed_30K_delta=fixed_delta, h_loss_Ds_fraction=ds_fraction,
              scene_changes=scene_changes, s1_donors=donors, s1_fr_shift_difference=shift_change,
              s3_A2=s3a2, s3_A4=s3a4, s3_paired_representatives=s3paired,
              s3_representative_means=dict(A2=means(s3a2), A4=means(s3a4)),
              s1_C60LR050=c60, sheet_student_joint_passes=passes,
              sheet_student_rows_missing_numeric_metrics=missing,
              sheet_missing_row_27_local_resolution=missing_resolved['representative'],
              s1_R1_late_training=late_training, s1_A4_checkpoint_presence=weight_checks,
              caveats=['s3 values are selected representatives; remote 30K and per-scene data unavailable.',
                       'Stage1 seed and server are confounded. Student seeds are paired.',
                       'One Stage1 donor per condition per server; three Student seeds are not three donor replicates.',
                       's1 A4 representatives use max-H fallback; all 30 checkpoint HQNR values miss the target.',
                       'Metric decomposition is exact algebra, not causal attribution.',
                       'Early-stopped or failed conditions are not evidence of a completed negative endpoint.'])
(OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['recorded_at', 's1_fixed_30K_means', 's1_fixed_30K_delta',
    'h_loss_Ds_fraction', 's1_fr_shift_difference', 's3_representative_means', 's3_paired_representatives',
    'sheet_student_joint_passes', 'sheet_student_rows_missing_numeric_metrics']}, indent=2))

# Display seed range descriptively, not as a confidence interval.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout='constrained')
for runs, label, color in [(a2, 'A2 / eps .03', '#2166ac'), (a4, 'A4 / eps .03', '#b2182b')]:
    for ax, metric, target in [(axes[0], 'H', TARGET_H), (axes[1], 'E', TARGET_E)]:
        steps = np.array([r['step'] for r in runs[0]['rows']])
        values = np.array([[r[metric] for r in run['rows']] for run in runs])
        ax.plot(steps / 1000, values.mean(axis=0), color=color, label=label)
        ax.fill_between(steps / 1000, values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
        ax.set_xlabel('Student step (K)')
        ax.grid(alpha=.2)
axes[0].axhline(TARGET_H, color='k', linestyle='--', linewidth=1)
axes[0].set(xlim=(5, 30), ylim=(.84, .96), ylabel='HQNR', title='s1: 3 Student seeds, mean + range')
axes[0].legend(fontsize=8, loc='lower right')
axes[1].axhline(TARGET_E, color='k', linestyle='--', linewidth=1)
axes[1].set(xlim=(15, 30), ylim=(3.49, 3.565), ylabel='ERGAS', title='s1: late-training ERGAS')
for runs, label, color in [(s3a2, 's3 A2 (selected)', '#2166ac'), (s3a4, 's3 A4 (selected)', '#b2182b')]:
    axes[2].scatter([r['E'] for r in runs], [r['H'] for r in runs], label=label, color=color)
    for r in runs:
        axes[2].annotate(f"R{r['replica']}", (r['E'], r['H']), xytext=(4, 4), textcoords='offset points', fontsize=8)
axes[2].axhline(TARGET_H, color='k', linestyle='--', linewidth=1)
axes[2].axvline(TARGET_E, color='k', linestyle='--', linewidth=1)
axes[2].set(xlim=(3.495, 3.542), ylim=(.929, .956), xlabel='ERGAS (lower is better)', ylabel='HQNR',
            title='s3: qualified representatives')
axes[2].legend(fontsize=8, loc='upper left')
axes[2].grid(alpha=.2)
fig.savefig(OUT / 'qb_a4_review.png', dpi=170)
fig.savefig(OUT / 'qb_a4_review.pdf')
