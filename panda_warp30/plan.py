"""Paired operators, coordinate frames and strength; fixed loss throughout."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from panda30r.plan import baseline as old_baseline
from donor24.plan import evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SEED = 261016101
SEEDS = {'R1': 261018101, 'R2': 261018102}
HORIZON = 50000
DONORS = {'S1_U23': 'interp23', 'S1_BIC': 'bicubic'}
# name: upsampler, frame, shift strength, scientific question
CASES = {
    'U23_PAN': ('interp23', 'pan', 1., 'PAN-to-MS reference for all geometry comparisons'),
    'BIC_PAN': ('bicubic', 'pan', 1., 'End-to-end fresh Stage1+2 upsampler control; not same-donor geometry evidence'),
    'U23_MID': ('interp23', 'mid', 1., 'Symmetric input/target transport without output return'),
    'U23_MID_RETURN': ('interp23', 'mid_return', 1., 'Symmetric inputs with differentiable output return and original targets'),
    'U23_MS_RETURN': ('interp23', 'ms_return', 1., 'MS-to-PAN inputs with differentiable output return and original targets'),
    'U23_PAN_S050': ('interp23', 'pan', .5, 'Half total correction, distinct from half coordinate sharing'),
    'U23_PAN_S000': ('interp23', 'pan', 0., 'Zero correction contribution control; A still measured, T unchanged'),
    'U23_PAN_S025': ('interp23', 'pan', .25, 'Quarter strength without any hidden RR/FR unit conversion'),
}
CORE = ('U23_PAN', 'BIC_PAN', 'U23_MID', 'U23_MID_RETURN', 'U23_MS_RETURN', 'U23_PAN_S050')
REPEATS = ('U23_PAN', 'U23_MID', 'U23_MID_RETURN', 'U23_MS_RETURN')


def baseline(server):
    if server != 's3':
        raise ValueError('This fresh warp campaign is local s3 / WV3 only')
    return deepcopy(old_baseline(server))


def groups(server='s3'):
    baseline(server)
    return ([{'name': case, 'optional': False, 'runs': [(case, 'R1')]} for case in CORE]
            + [{'name': case, 'optional': True, 'runs': [(case, 'R1')]}
               for case in ('U23_PAN_S000', 'U23_PAN_S025')]
            + [{'name': 'FOUR_FRAME_R2', 'optional': True,
                'runs': [(case, 'R2') for case in REPEATS], 'admission': 'whole_group_or_skip'}])


def default_config(server):
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
            'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
            'budget': deepcopy(BUDGET), 'donors': list(DONORS), 'groups': groups(server),
            'donor_policy': 'FRESH_LOCAL_MATCHED_STAGE1_H50_BEST_HQNR_THEN_ERGAS',
            'donor_source_seed': SOURCE_SEED, 'allow_cross_server_donor': False,
            'allow_old_donor_reuse': False, 'checkpoint_selection': POLICY.manifest(),
            'sheet_selection': 'ALL_OFFICIAL_EVALUATED_CHECKPOINTS',
            'evaluation': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20',
            'fixed_loss': {'alpha': .25, 'beta': .05, 'gamma': .5},
            'optional_policy': 'PREREGISTERED_ORDER_WITH_MEASURED_WHOLE_GROUP_ADMISSION',
            'time_prior_hours': {'stage1': 4., 'stage2': 2.1},
            'pruning': 'NONE'}


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered 30h warp configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    from panda_stage2_condq.training import TrainingSpec
    from .spec import WarpSpec, geometry_manifest
    campaign = default_config(server)
    valid = {(c, r) for group in groups(server) for c, r in group['runs']}
    if ((case, replica) not in valid and (case, replica) not in {(c, 'R1') for c in DONORS}):
        raise ValueError('Unregistered warp case or repeat')
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Positive attempt required')
    donor = case in DONORS
    seed = SOURCE_SEED if donor else SEEDS[replica]
    seeds, b = stream_seeds(seed), campaign['baseline_resolved']
    upsampler, frame, scale = (DONORS[case], 'pan', 1.) if donor else CASES[case][:3]
    spec = WarpSpec('WV3', 'panda_condq', seeds['S_init'], .25, .05, .5, 1e-6,
                   mask_policy=LOSS_MASK_POLICY, upsampler=upsampler, frame=frame, shift_scale=scale)
    cfg = {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'case': case, 'loss_case': case,
        'replica': replica, 'attempt': attempt, 'seed': seed,
        'run_id': f'PW30_WV3_S3_{case}_{replica}_S{seed}',
        'run_display': f'WARP30 {case} {replica}',
        'stream_seeds': seeds, 'stream_seed_protocol': SEED_PROTOCOL,
        'baseline_resolved': deepcopy(b), 'schedule': 'H50', 'start_update': 0,
        'planned_horizon': HORIZON, 'evaluation_steps': list(evaluation_steps()),
        'retention_steps': [HORIZON], 'phase': 'donor' if donor else 'stage2', 'stage2': not donor,
        'engine_kind': 'stage1_warp' if donor else 'stage2_warp', 'initialization': 'FRESH',
        'aligner_kind': 'direct', 'upsampler': upsampler, 'frame': frame, 'shift_scale': scale,
        'geometry': geometry_manifest(spec),
        'lr_U': 1e-4, 'lr_A': 1e-5 if donor else 0., 'weight_decay': .01,
        'lambda_rec': 1., 'lambda_abs': 2. if donor else 0.,
        'lambda_epsilon': .003 if donor else 0., 'lambda_eps_RR': .003 if donor else 0.,
        'epsilon_radius_rr': 3., 'epsilon_distribution': 'UNIFORM_AREA_DISK',
        'hqnr_min': .959, 'ergas_max': 2.040, 'checkpoint_selection': POLICY.manifest(),
        'loss_mask_policy': LOSS_MASK_POLICY, 'mask_policy': 'MASK2',
        'baseline_alignment_protocol_digest': alignment_protocol_digest(b),
        'config_digest': canonical_sha256(campaign),
        'evidence_level': 'PAIRED_SCREEN_NOT_STATISTICAL_PROOF',
        'student_architecture': {'bands': 8, 'width': 112, 'depth': [1,2,3]},
        'stage2_spec': asdict(spec),
        'augmentation': {'hflip_probability': 1., 'vflip_probability': 1., 'rot90': True},
        'training_spec': asdict(TrainingSpec(lr=1e-4, total_updates=HORIZON,
                                            warmup_updates=100, full_audit_interval=100)),
        'donor_version': 'U23' if upsampler == 'interp23' else 'BIC',
        'question': 'Fresh paired Stage1 PAN-to-MS donor' if donor else CASES[case][3]}
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'], record['ERGAS'], record['completed_update'])


def catalog_csv(server='s3'):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['order','phase','case','replica','seed','upsampler','frame','shift_scale','optional','updates'])
    rows = [(c, 'R1', False) for c in DONORS]
    rows += [(c, r, group['optional']) for group in groups(server) for c, r in group['runs']]
    for i, (c, r, optional) in enumerate(rows, 1):
        cfg = resolve_case(server, c, r)
        writer.writerow([i,cfg['phase'],c,r,cfg['seed'],cfg['upsampler'],cfg['frame'],cfg['shift_scale'],optional,HORIZON])
    return output.getvalue()
