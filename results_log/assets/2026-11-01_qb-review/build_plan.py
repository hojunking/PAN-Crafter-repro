"""Build a declarative follow-up plan; does not register or launch experiments."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'config/panda_qb40'
SEEDS = {'R1': 261018101, 'R2': 261018102, 'R3': 261018103}
S1_SEEDS = {'s1': 261016101, 's3': 261016301}
STUDENT_H = {'s1': 2.41, 's3': 1.67}
STAGE1_H = {'s1': 3.10, 's3': 2.10}
DONORS = {
    'A200_E100': {'stage1_lambda_abs': 2., 'stage1_lambda_epsilon': .10},
    'A400_E030': {'stage1_lambda_abs': 4., 'stage1_lambda_epsilon': .03},
    'A400_E100': {'stage1_lambda_abs': 4., 'stage1_lambda_epsilon': .10},
}


def donor(server, name):
    return dict(server=server, case='QB5_S1_' + name, phase='stage1', replica='R1',
        seed=S1_SEEDS[server], donor='SELF_NEW_LOCAL_STAGE1', updates=50000,
        batch=48, microbatch=12, accumulation=4, warmup=100,
        lr_teacher=1e-4, lr_aligner=1e-5, lr_student=None,
        alpha=None, beta=None, gamma=None, width=112, depth=[1, 2, 3],
        input='CORRECTED', sampling='BASE', weight_decay=.01,
        epsilon_radius_rr=3., **DONORS[name], estimate_hours=STAGE1_H[server],
        dependencies=[], selection='UNCHANGED_FULL_50K_VALIDATION_RRE_SELECTOR',
        reason='Upper-side Stage1 coefficient comparison with the original local Stage1 seed')


def student(server, case, replica, **changes):
    row = dict(server=server, case='QB5_' + case, phase='stage2', replica=replica,
        seed=SEEDS[replica], donor='EXISTING_OWN_A200_E030', updates=30000,
        batch=48, microbatch=12, accumulation=4, warmup=100,
        lr_teacher=0., lr_aligner=0., lr_student=1e-4,
        alpha=.25, beta=0., gamma=.5, width=112, depth=[1, 2, 3],
        input='CORRECTED_DIFF', sampling='UBIL', weight_decay=.01,
        epsilon_radius_rr=3., stage1_lambda_abs=2., stage1_lambda_epsilon=.03,
        dependencies=[], selection='ALL_JOINT_PASSES_PLUS_H_QUALIFIED_MIN_E_ELSE_MAX_H',
        reason=case)
    if case in DONORS:
        row.update(donor='NEW_OWN_' + case, **DONORS[case],
                   dependencies=['QB5_S1_' + case + ':R1'],
                   reason='Whole Stage1 pipeline effect; paired standard Student')
    row.update(changes)
    row['estimate_hours'] = round(STUDENT_H[server] * row['updates'] * row['batch'] / (30000 * 48), 4)
    row['training_presentations'] = row['updates'] * row['batch']
    return row


def pair(server, case, **changes):
    return [student(server, case, rep, **changes) for rep in ('R1', 'R2')]


def branch(server, name):
    return [donor(server, name), *pair(server, name)]


def long_pair(server):
    return pair(server, 'C60LR050', updates=60000, lr_student=5e-5,
                reason='Twice the presentations, smaller LR, approximately matched integrated LR')


def batch_pair(server, lr):
    return pair(server, 'B96C15LR' + ('100' if lr == 1e-4 else '200'),
                updates=15000, batch=96, accumulation=8, warmup=50, lr_student=lr,
                reason='Matched 1.44M presentations; batch/update/LR recipe comparison')


def queues():
    s1 = [*branch('s1', 'A200_E100'), *long_pair('s1'),
          *branch('s1', 'A400_E030'), *batch_pair('s1', 1e-4), *batch_pair('s1', 2e-4)]
    s3 = [*batch_pair('s3', 1e-4), *batch_pair('s3', 2e-4),
          *branch('s3', 'A200_E100'), *branch('s3', 'A400_E030'),
          *long_pair('s3'), *branch('s3', 'A400_E100'),
          student('s3', 'A200_E100', 'R3'), student('s3', 'A400_E030', 'R3'),
          *pair('s3', 'W128H050', width=128, alpha=.125, gamma=.25,
                reason='Exploratory combined width/Hard recipe; not an isolated interaction estimate')]
    return {'s1': s1, 's3': s3}


def main():
    plan = dict(plan_id='PANDA_S13_QB40_PROPOSAL_20261101_v1', status='PLANNED_NOT_REGISTERED',
        dataset='QB', joint_target={'HQNR': .9315, 'ERGAS': 3.5},
        scope='NEW_FOLLOWUP_BUDGET_PER_SERVER_EXCLUDING_CURRENT_QUEUE',
        budget_hours_per_server=40., fresh_students=True, scale=1.,
        stage1_origin='INDEPENDENT_LOCAL_TRAINING_PER_SERVER_AND_COEFFICIENT_BRANCH',
        existing_stage1_sources={
            's1': dict(campaign='PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1',
                       case='S1_A200_E030', seed=261016101, observed_selected_step=44000),
            's3': dict(campaign='PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1',
                       case='S1_A200_E030', seed=261016301, observed_selected_step=39000)},
        warp='LANCZOS4_A_T_S', structure_reference='RGB_MEAN',
        ms_paths='UNCHANGED_STORED_LMS_ALIGNER_AND_INTERP23_RECONSTRUCTION_BASE',
        stage2_loss='EXISTING_GT_HARD_BETA_ZERO', optimizer='AdamW',
        optimizer_betas=[.9, .999], optimizer_eps=1e-8,
        stage2_scheduler='WARMUP_COSINE_TO_ZERO_FROM_FRESH_INITIALIZATION',
        evaluation_interval=1000,
        early_stop=dict(scope='QB_STAGE2_ONLY', steps=[8000, 9000, 10000],
                        all_hqnr_strictly_below=.90, absolute_updates_unchanged=True),
        reporting_sheet='QB-ablations',
        retention='EVERY_SCHEDULED_EVALUATION_CHECKPOINT_AND_COMPLETE_RESUME_STATE',
        excluded=['rotation_loss', 'TTA', 'SWA', 'EMA', 'soft_loss', 'band_weighted_loss',
                  'P0_student_followup', 'BASE_student_followup', 'WD0_expansion'],
        common_analysis_grid='15 relative-horizon checkpoints: C15 every1K, C30 every2K, C60 every4K; official publication unchanged',
        cost_scope='Full-run time estimate, not selected-checkpoint Wall(h); no early-stop discounts',
        cost_limitations=['B96 recipe and s3 W128 need actual preflight profiling.',
                          's3 Stage1 estimate partly extrapolated from the selected-step timing.',
                          'Reserve covers profiling, full donor selection/calibration and timing drift.'],
        runtime_policy='Plan only. No live queue edits, no process signals, no automatic launch.',
        future_admission='Before a run, require measured full-run prediction plus closeout reserve to fit remaining budget; retain unfunded runs as pending, not completed.',
        queues=queues(), budget={})
    OUT.mkdir(parents=True, exist_ok=True)
    for server, rows in plan['queues'].items():
        seen=set(); elapsed=0.
        for i, row in enumerate(rows, 1):
            key=row['case']+':'+row['replica']
            assert key not in seen and set(row['dependencies']) <= seen
            assert row['batch'] == row['microbatch']*row['accumulation']
            assert row['phase']=='stage1' or (row['beta']==0. and row['input']=='CORRECTED_DIFF')
            seen.add(key); elapsed+=row['estimate_hours']
            row.update(order=i, cumulative_estimate_hours=round(elapsed, 4), execution_status='PLANNED')
        assert elapsed < 40
        plan['budget'][server]=dict(stage1_runs=sum(r['phase']=='stage1' for r in rows),
            student_runs=sum(r['phase']=='stage2' for r in rows), total_jobs=len(rows),
            nominal_hours=round(elapsed, 4), reserve_hours=round(40-elapsed, 4),
            measured_30k_student_hours=STUDENT_H[server], stage1_allowance_hours=STAGE1_H[server])
        fields=list(rows[0])
        for row in rows:
            fields.extend(k for k in row if k not in fields)
        with (OUT/f'cases_{server}.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fields, lineterminator='\n');writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    (OUT/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(plan['budget'], indent=2))


if __name__=='__main__': main()
