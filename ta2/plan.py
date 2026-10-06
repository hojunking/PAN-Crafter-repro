"""Strict binding to the supplied 52-case / 468-run bundle; no invented queue."""
import csv
import math
from collections import Counter
from pathlib import Path

from .common import BUNDLE, CAMPAIGN, digest, file_sha, load_json

LANES = {'s1': ('WV3', 8, 2047, (261006101, 261006102, 261006103)),
         's3': ('QB', 4, 2047, (261006301, 261006302, 261006303)),
         's5': ('GF2', 4, 1023, (261006501, 261006502, 261006503))}
CORE = {'TA2-' + c for c in 'B00 B02 B03 B04 B05 B07 M00 M01 M02 M03 M04 M05 M06 M07 U00 U03'.split()}
POLICIES = {'NONE': 'BYPASS', 'ZERO_WARP': 'ZERO', 'TRAIN_GLOBAL_MEDIAN': 'GLOBAL_FIXED',
            'PER_IMAGE_LMS_GRAD_FIXED': 'PER_IMAGE_GRAD', 'PER_IMAGE_LMS_INT_FIXED': 'PER_IMAGE_INTENSITY',
            'PER_IMAGE_BIC_GRAD_FIXED': 'PER_IMAGE_BICUBIC', 'PER_IMAGE_GT_TRAIN_LMS_EVAL': 'TRAIN_GT_PROXY',
            'LEARNED_GLOBAL_2VECTOR': 'GLOBAL_LEARNED', 'PER_IMAGE_LMS_SHUFFLED': 'SHUFFLED', 'CNN': 'LEARNED'}
DESCRIPTORS = {'MULTISCALE_GNCC_SQUARED': 'NCC', 'SINGLE_GNCC_SQUARED': 'NCC',
               'MULTISCALE_NGF_SQUARED': 'NGF', 'HUBER_PSEUDO_SHIFT_BETA025': 'PSEUDO_HUBER'}
EXTRAS = {'NONE', 'PRIVILEGED_TRAIN_PROXY_NOT_GT_AT_INFERENCE', '2_PARAMETERS_CANONICAL_FRAME',
          'PSEUDO_OFFSET_NOT_PHYSICAL_GT', 'FROZEN_TRAIN_BAND_WEIGHTS', 'STRUCT_RAMP_5000',
          'A_AUX_MULTICROP_48_56_64', 'A_AUX_MAGNIFY_1_2_4_8', 'A_AUX_MAGNIFY_WITH_SCALE_LAMBDA001'}
INTEGER = {'bands', 'max_dn', 'replica', 'seed', 'queue_seq', 'epsilon_every', 'updates', 'repeat_count'}
FLOAT = {'epsilon_lambda', 'epsilon_radius', 'struct_lambda', 'u_lr', 'a_lr'}
BOOLEAN = {'a_receives_rec', 'student_training', 'kd', 'q_weighting', 'panmix'}


def csv_records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed registry CSV')
        for key in row:
            if key in INTEGER:
                row[key] = int(row[key])
            elif key in FLOAT and row[key] != 'DATASET_LEGACY':
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError('Nonfinite registry coefficient')
            elif key in BOOLEAN:
                if row[key] not in ('True', 'False'):
                    raise ValueError('Boolean registry value is not explicit')
                row[key] = row[key] == 'True'
    return rows


class Plan:
    def __init__(self, bundle=BUNDLE):
        self.bundle = Path(bundle).resolve(strict=True)
        self.files = {}
        for line in (self.bundle / 'SHA256SUMS.txt').read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            relative = relative.lstrip('*')
            path = self.bundle / relative
            if not path.resolve().is_relative_to(self.bundle) or path.is_symlink():
                raise ValueError('Plan manifest escapes bundle')
            if file_sha(path) != expected:
                raise ValueError('Plan bundle SHA mismatch: ' + relative)
            self.files[relative] = expected
        required = {'experiment_spec.json', 'TeacherOnly_Cases_52.csv', 'TeacherOnly_Run_Registry_468.csv',
                    'analysis_columns_v2.json'} | {f'{server}_{lane[0]}_Queue_156.csv' for server, lane in LANES.items()}
        if not required <= self.files.keys():
            raise ValueError('Incomplete author-supplied plan bundle')
        spec = load_json(self.bundle / 'experiment_spec.json')
        self.cases = {r['case_id']: r for r in spec['cases']}
        self.runs = csv_records(self.bundle / 'TeacherOnly_Run_Registry_468.csv')
        self.columns = load_json(self.bundle / 'analysis_columns_v2.json')
        if (spec['campaign_id'] != CAMPAIGN or len(self.cases) != 52 or len(self.runs) != 468
                or len({r['run_id'] for r in self.runs}) != 468 or len(set(self.columns)) != 152
                or spec['repeat_until_success'] is not False or spec['wall_clock_limit'] is not None):
            raise ValueError('Campaign counts/schema/termination differ from the registered plan')
        if csv_records(self.bundle / 'TeacherOnly_Cases_52.csv') != spec['cases']:
            raise ValueError('Cases CSV differs from JSON definitions')
        self.queues = {}
        for server, (sensor, bands, max_dn, seeds) in LANES.items():
            queue = csv_records(self.bundle / f'{server}_{sensor}_Queue_156.csv')
            official = sorted((r for r in self.runs if r['server'] == server), key=lambda r: r['queue_seq'])
            if queue != official or [r['queue_seq'] for r in queue] != list(range(1, 157)):
                raise ValueError('Server queue differs from the author registry')
            if Counter(r['case_id'] for r in queue) != Counter({c: 3 for c in self.cases}):
                raise ValueError('Each case requires exactly three attempts/seeds')
            for row in queue:
                if (row['campaign_id'] != CAMPAIGN or row['dataset'] != sensor or row['bands'] != bands
                        or row['max_dn'] != max_dn or row['replica'] not in (1, 2, 3)
                        or row['seed'] != seeds[row['replica']-1]):
                    raise ValueError('Cross-server / seed registry mismatch')
                case = self.cases[row['case_id']]
                for key, value in case.items():
                    if key == 'epsilon_lambda' and value == 'DATASET_LEGACY':
                        value = .0003 if sensor == 'QB' else .0001
                    if row[key] != value:
                        raise ValueError(f'Resolved run differs from its case: {row["run_id"]}/{key}')
                if (row['training_role'] != 'TEACHER_ONLY' or row['init'] != 'FRESH'
                        or any(row[k] for k in ('student_training', 'kd', 'q_weighting', 'panmix'))
                        or row['checkpoint_primary'] != 'HQNR_MAX50' or row['checkpoint_causal'] != 'EXACT_FINAL'):
                    raise ValueError('Forbidden donor/student/loss/selector changes')
                resolved_config(row)
            if (Counter(r['case_id'] for r in queue[:48]) != Counter({c: 3 for c in CORE})
                    or any(r['queue_block'] != 'P0_CORE' for r in queue[:48])):
                raise ValueError('All three core repeats must precede the remaining blocks')
            self.queues[server] = queue
        if sum(r['updates'] for r in self.runs) != 25200000:
            raise ValueError('Total update budget differs')
        self.sha256 = digest(self.files)

    def queue(self, server):
        if server not in LANES:
            raise ValueError('TA2 only assigns s1/WV3, s3/QB and s5/GF2; s2/s4 are untouched')
        return [dict(row) for row in self.queues[server]]


def resolved_config(row, micro_batch=12):
    if micro_batch not in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48):
        raise ValueError('Microbatch must divide the fixed batch48')
    if row['extra'] not in EXTRAS:
        raise ValueError('Unknown case extension')
    if not isinstance(row['epsilon_lambda'], (int, float)):
        raise ValueError('DATASET_LEGACY must be resolved before runtime')
    cfg = dict(row)
    cfg.update(aligner_policy=POLICIES[row['aligner']],
        lambda_epsilon=float(row['epsilon_lambda']), lambda_struct=float(row['struct_lambda']),
        struct_target={'NONE': 'OFF'}.get(row['struct_target'], row['struct_target']),
        struct_descriptor=DESCRIPTORS[row['struct_descriptor']],
        struct_scales=[.8] if row['struct_descriptor'] == 'SINGLE_GNCC_SQUARED' else [.8, 1.6],
        struct_branch='NATIVE_CORRECTED' if row['struct_branch'] == 'HALF_NATIVE_HALF_CORRECTED_EPS' else 'NATIVE',
        band_weights='CAL_FROZEN' if row['extra'] == 'FROZEN_TRAIN_BAND_WEIGHTS' else 'UNIFORM',
        epsilon_detach_anchor=row['epsilon_target'] == 'STOPGRAD_NATIVE',
        struct_ramp_updates=5000 if row['extra'] == 'STRUCT_RAMP_5000' else 0,
        auxiliary={'A_AUX_MULTICROP_48_56_64': 'CROP', 'A_AUX_MAGNIFY_1_2_4_8': 'SCALE',
                   'A_AUX_MAGNIFY_WITH_SCALE_LAMBDA001': 'SCALE_CONSISTENCY'}.get(row['extra'], 'NONE'),
        lambda_scale=.001 if row['extra'] == 'A_AUX_MAGNIFY_WITH_SCALE_LAMBDA001' else 0.,
        lr_u=float(row['u_lr']), lr_a=float(row['a_lr']), total_updates=int(row['updates']),
        batch_size=48, micro_batch=micro_batch, auxiliary_micro_batch=1,
        aligner_margin=0, struct_margin=16, width=112, depth=[1, 2, 3],
        warmup_updates=100, optimizer='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=.01,
        precision='FP32', amp=False, tf32=False, deterministic_cuda_grid_backward=False,
        native_augmentation='FH12_FIXED_HV_THEN_UNIFORM_ROT4',
        lms_augmentation='READ_STORED_HR_FIRST_THEN_SAME_HR_VIEW',
        bicubic_order='INTERPOLATE_AUGMENTED_LRMS_ALIGN_CORNERS_FALSE',
        implementation_revision='TA2_IMPL_v1')
    return cfg
