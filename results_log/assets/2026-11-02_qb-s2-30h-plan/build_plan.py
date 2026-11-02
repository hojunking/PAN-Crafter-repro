"""Produce a reviewable s2 experiment proposal, not a runnable campaign."""
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from panda_qb_rre.runtime import code_pin

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


timings = []
campaign = ROOT / 'work_dir/panda_qb_rre/PANDA_S13_QB_RRE_FOCUS_20261101_v1/s1/QB'
for path in sorted(campaign.glob('runs/*/attempt_01/run_state.json')):
    state = read(path)
    assert state['status'] == 'COMPLETED'
    elapsed = (datetime.fromisoformat(state['completed_at']) - datetime.fromisoformat(state['started_at'])).total_seconds() / 3600
    timings.append(dict(run_id=state['run_id'], phase=state['phase'], updates=state['step'],
                        hours=elapsed, source=str(path.relative_to(ROOT))))
stage1_h = next(r['hours'] for r in timings if r['phase'] == 'donor')
student_times = [r['hours'] for r in timings if r['phase'] == 'stage2']
stage2_h = sum(student_times) / len(student_times)
base_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
pins = code_pin()
tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
dirty = set(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines())
assert not set(pins) - tracked
assert not set(pins) & dirty
save('reference_code_pin.json', dict(reference_commit=base_commit,
    role='EXISTING_QB_RRE_REFERENCE_NOT_NEW_S2_EXECUTION_MANIFEST', files=pins))

specs = [
    ('A200_E030_D101', 2., .03, 261016101, 'Baseline: retrain historical s1 Stage1 seed locally on s2.'),
    ('A400_E030_D101', 4., .03, 261016101, 'Paired A2/A4 comparison for the historical s1 seed.'),
    ('A200_E030_D301', 2., .03, 261016301, 'Baseline: retrain historical s3 Stage1 seed locally on s2.'),
    ('A400_E030_D301', 4., .03, 261016301, 'Paired A2/A4 comparison for the historical s3 seed.'),
    ('A300_E030_D101', 3., .03, 261016101, 'Intermediate absolute-loss weight, matched to the s1-seed pair.'),
    ('A400_E060_D101', 4., .06, 261016101, 'Change epsilon weight only relative to A4/E.03; test coefficient balance.'),
]
cases, jobs = [], []
for order, (suffix, absolute, epsilon, seed, purpose) in enumerate(specs, 1):
    donor_key = 'QB6_S1_' + suffix
    student_key = 'QB6_S2_' + suffix
    cases.append(dict(order=order, case='QB6_' + suffix, lambda_abs=absolute, lambda_epsilon=epsilon,
        stage1_seed=seed, student_seed=261018101, stage1_case=donor_key, student_case=student_key,
        purpose=purpose, nominal_hours=stage1_h + stage2_h,
        cumulative_nominal_hours=order * (stage1_h + stage2_h)))
    jobs.extend([
        dict(order=2*order-1, server='s2', dataset='QB', phase='stage1', case=donor_key,
             donor_case=donor_key, seed=seed, updates=50000, stage1_lambda_abs=absolute, stage1_lambda_epsilon=epsilon,
             alpha=None, gamma=None, beta=0., aligner_scale=1., estimate_hours=stage1_h),
        dict(order=2*order, server='s2', dataset='QB', phase='stage2', case=student_key,
             donor_case=donor_key, seed=261018101, updates=30000, stage1_lambda_abs=absolute, stage1_lambda_epsilon=epsilon,
             alpha=.25, gamma=.5, beta=0., aligner_scale=1., estimate_hours=stage2_h),
    ])
assert len(cases) == 6 and len(jobs) == 12
assert len({r['case'] for r in jobs}) == 12
assert len({r['seed'] for r in jobs if r['phase'] == 'stage2'}) == 1
for first, second in zip(jobs[::2], jobs[1::2]):
    assert first['case'] == second['donor_case']
cfg = read(next(campaign.glob('runs/*S1_A400_E030*/attempt_01/config.json')))
native_environment = read(campaign / 'admission.json')['environment']
proposal = dict(
    schema='PANDA_S2_QB_STAGE1_BALANCE_PROPOSAL_V1',
    status='PROPOSAL_NOT_REGISTERED_OR_EXECUTABLE',
    campaign_id_proposed='PANDA_S2_QB_STAGE1_BALANCE_20261102_v1',
    reference_commit=base_commit, server='s2', dataset='QB',
    created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    user_budget_interpretation='Use s1 measured timing; approximately 30h, no hard deadline.',
    wallclock_deadline_seconds=None, after_queue='STOP_NO_WV3_SUCCESSOR',
    target=dict(HQNR_min=.9315, ERGAS_max=3.5, same_checkpoint=True),
    cases=cases, jobs=jobs,
    common_stage1=dict(updates=50000, peak_lr_A=1e-5, peak_lr_T=1e-4,
        lambda_rec=1., warmup_updates=100, schedule='WARMUP_COSINE_TO_ZERO',
        epsilon_radius_rr=3., epsilon_distribution='UNIFORM_AREA_DISK', native_correction_clip=None,
        teacher_width=112, teacher_depth=[1, 2, 3], fresh_local_training=True,
        selection=cfg['checkpoint_selection']),
    common_stage2=dict(updates=30000, peak_lr=1e-4, warmup_updates=100,
        schedule='WARMUP_COSINE_TO_ZERO', alpha=.25, gamma=.5, beta=0.,
        student_seed=261018101, width=112, depth=[1, 2, 3], sampling='UBIL',
        input=['CORRECTED_PAN', 'CORRECTED_MINUS_ORIGINAL_PAN', 'Ilrms_4_BANDS'],
        input_channels=6, output_bands=4, aligner_scale=1., frozen_donor=True,
        teacher_guidance='EXISTING_HARD_WEIGHT_ONLY_NO_SOFT', fresh_student_each_case=True),
    shared=dict(effective_batch=48, microbatch=12, accumulation=4, optimizer='AdamW',
        weight_decay=.01, betas=[.9, .999], adam_epsilon=1e-8,
        train_precision='FP32', amp=False, tf32=False, cudnn_benchmark=False,
        original_training_augmentation='UNCHANGED_FROM_REFERENCE_NOT_ROTATION_CONSISTENCY',
        warp_A_T_S='LANCZOS4_WHOLE_SAMPLE_REFLECTION',
        structure_reference_S1_S2='RGB_MEAN_INDICES_2_1_0',
        aligner_MS='ALL_FOUR_BANDS_OF_REGISTERED_STORED_LMS',
        reconstruction_base='INTERP23_OF_REGISTERED_LRMS',
        qb_train_val_policy='ORIGINAL_STORED_LMS_OVERRIDE_AND_REGISTERED_MSFIX_LRMS',
        rotation_consistency=0., inference_TTA=False, weight_averaging=False, band_weighted_loss=False),
    sources=dict(catalog='ablr2/sensor_sources.json',
        catalog_sha256=cfg['baseline_resolved']['source_registry_sha256'],
        qb_lms_override='config/bglobal_rr/qb_original_lms.json',
        qb_lms_override_sha256=cfg['baseline_resolved']['lms_override_sha256'],
        train_count=17139, val_count=1905, RR_scenes=20, FR_scenes=20, peak_DN=2047),
    reporting=dict(tab='QB-ablations', server='s2', eval_interval=1000,
        selection='ALL_JOINT_PASSES_PLUS_ONE_HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_HQNR_PER_RUN',
        retain_all_evaluation_weights=True, publish_stage1_selected_teacher=True,
        approach_fields=['stage1_loss_pair', 'stage1_seed', 'student_seed', 'alpha_gamma_beta',
                         'lr_horizon', 'scale', 'DIFF_UBIL', 'batch', 'width']),
    failure_policy=dict(student_early_stop='ALL_OFFICIAL_HQNR_AT_8K_9K_10K_STRICTLY_BELOW_0.90',
        apply_teacher_HQNR_gate=False, partial_donor_allowed=False,
        stage1_failure='RECORD_FAILURE_SKIP_ONLY_DEPENDENT_STUDENT_CONTINUE_INDEPENDENT_CASES',
        independent_case_results_do_not_cancel_other_cases=True),
    timing=dict(reference_runs=timings, measured_stage1_hours=stage1_h,
        measured_stage2_mean_hours=stage2_h, nominal_12_job_hours=6*(stage1_h+stage2_h),
        planning_range_hours=[34, 36], setup_and_data_transfer_included=False,
        full_horizon_completed_times_not_selected_checkpoint_sheet_times=True),
    code_handoff_audit=dict(existing_reference_pinned_files=len(pins),
        untracked_reference_files=[], changed_reference_files=[],
        existing_cli_supports_s2=False, existing_launcher_requires_old_sensor5_container=True,
        new_runtime_implementation_required=True, reference_environment=native_environment),
)
save('cases.json', proposal)
with (OUT / 'cases.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(jobs[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(jobs)
print(json.dumps(dict(status=proposal['status'], cases=len(cases), jobs=len(jobs),
                     nominal_hours=proposal['timing']['nominal_12_job_hours'],
                     pinned_reference_files=len(pins)), indent=2))
