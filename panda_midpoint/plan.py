"""Three preregistered loss settings for the s3 symmetric-shift intervention.

The architecture, RR stream, optimizer and conditional-q objective are unchanged.
Only the Stage 2 coordinate policy changes. All cases share the same local mature
Teacher and frozen early Aligner; this queue never creates another Stage 1 donor.
"""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from donor24.plan import evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from panda30r.plan import baseline as prior_baseline
from panda_stage2_condq.selection import SelectionPolicy
from .contract import CAMPAIGN_ID, METHOD_REVISION, GEOMETRY

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ('s3',)
SEEDS = {'R1': 261018101, 'R2': 261018102}
DEFAULT_REPLICAS = ('R1',)
HORIZON = 50000
POLICY = SelectionPolicy(.959)
CASES = {
    'MID_B005': (.25, .05, .5, 'M_Q paired reference; mature donor with moderate hard weighting'),
    'MID_B010': (.25, .1, .5, 'Intermediate beta between .05 and .2; interpolation, not a measured best'),
    'MID_B020': (.25, .2, .5, 'M_B2 promising two-seed HQNR on s4; not established on s3'),
}


def baseline(server):
    if server not in SERVERS:
        raise ValueError('Only local s3/WV3 is registered for midpoint Stage 2')
    return deepcopy(prior_baseline(server))


def groups(server, replicas=DEFAULT_REPLICAS):
    baseline(server)
    if not replicas or len(set(replicas)) != len(replicas) or any(r not in SEEDS for r in replicas):
        raise ValueError('Explicit unique registered replicas required')
    return [{'name': case, 'optional': False, 'anchor': case == 'MID_B005',
             'budget_block': case, 'runs': [(case, replica) for replica in replicas],
             'evidence_level': 'PAIRED_ONE_SEED' if len(replicas) == 1 else 'PAIRED_TWO_SEEDS'} for case in CASES]


def default_config(server, replicas=DEFAULT_REPLICAS):
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'groups': groups(server, replicas), 'replicas': list(replicas), 'geometry': deepcopy(GEOMETRY),
        'donor_policy': 'EXISTING_LOCAL_PANDA30R_M_FINAL50K_EARLY_A_FROZEN',
        'allow_cross_server_donor': False, 'allow_fresh_donor_fallback': False,
        'checkpoint_selection': POLICY.manifest(),
        'sheet_selection': {'HQNR_min': .960, 'ERGAS_max': 2.045,
                            'otherwise': 'best_HQNR_per_run'},
        'evaluation': GEOMETRY['metric_references'],
        'queue_policy': 'ALL_REGISTERED_RUNS_NO_INHERITED_WALL_DEADLINE',
        'checkpoint_retention': 'ALL_OFFICIAL_EVALUATED_STATES_PLUS_SAFE_STOP',
        'pruning': 'NONE', 'estimated_hours': 'MEASURED_RUNTIME_ONLY_NOT_A_DEADLINE'}


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'), cfg.get('replicas', DEFAULT_REPLICAS))):
        raise ValueError('Unregistered midpoint campaign configuration')


def resolve_case(server, case, replica='R1', *, attempt=1, replicas=DEFAULT_REPLICAS):
    from panda_stage2_condq.training import TrainingSpec
    config = default_config(server, replicas)
    if case not in CASES or replica not in config['replicas']:
        raise ValueError('Unregistered midpoint case/replica')
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Positive integer attempt required')
    seed = SEEDS[replica]
    seeds = stream_seeds(seed)
    a, b, g, question = CASES[case]
    baseline_cfg = config['baseline_resolved']
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'case': case, 'loss_case': case,
        'replica': replica, 'attempt': attempt, 'seed': seed,
        'run_id': f'PMID_WV3_S3_{case}_{replica}_S{seed}',
        'run_display': f'MID50 {case} DM Af a{a:g} b{b:g} q{g:g} M2 {replica}',
        'stream_seeds': seeds, 'stream_seed_protocol': SEED_PROTOCOL,
        'baseline_resolved': deepcopy(baseline_cfg), 'geometry': deepcopy(GEOMETRY),
        'schedule': 'H50', 'start_update': 0, 'planned_horizon': HORIZON,
        'evaluation_steps': list(evaluation_steps()), 'retention_steps': [HORIZON],
        'phase': 'stage2', 'stage2': True, 'engine_kind': 'stage2_midpoint',
        'initialization': 'FRESH', 'donor_version': 'M', 'aligner_kind': 'direct',
        'lr_U': 1e-4, 'lr_A': 0., 'weight_decay': .01,
        'lambda_abs': 0., 'lambda_epsilon': 0.,
        'hqnr_min': .959, 'ergas_max': 2.040,
        'checkpoint_selection': deepcopy(config['checkpoint_selection']),
        'loss_mask_policy': LOSS_MASK_POLICY,
        'baseline_alignment_protocol_digest': alignment_protocol_digest(baseline_cfg),
        'config_digest': canonical_sha256(config),
        'evidence_level': 'PAIRED_ONE_SEED_SCREEN' if len(config['replicas']) == 1 else 'PAIRED_TWO_SEEDS',
        'student_architecture': {'bands': 8, 'width': 112, 'depth': [1, 2, 3]},
        'stage2_spec': {'sensor': 'WV3', 'variant': 'panda_condq', 'seed': seeds['S_init'],
            'alpha': a, 'beta': b, 'gamma': g, 'delta': 1e-6, 'mask_policy': LOSS_MASK_POLICY},
        'augmentation': {'hflip_probability': 1., 'vflip_probability': 1., 'rot90': True},
        'training_spec': asdict(TrainingSpec(lr=1e-4, total_updates=HORIZON,
                                            warmup_updates=100, full_audit_interval=100)),
        'question': question}


def rank(record):
    return POLICY.rank(record['HQNR'], record['ERGAS'], record['completed_update'])


def catalog_csv(server, replicas=DEFAULT_REPLICAS):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['order', 'case', 'replica', 'seed', 'alpha', 'beta', 'gamma',
                     'donor', 'local_updates', 'run_display', 'question'])
    order = 0
    for group in groups(server, replicas):
        for case, replica in group['runs']:
            order += 1
            cfg = resolve_case(server, case, replica, replicas=replicas)
            spec = cfg['stage2_spec']
            writer.writerow([order, case, replica, cfg['seed'], spec['alpha'], spec['beta'],
                spec['gamma'], 'M', HORIZON, cfg['run_display'], cfg['question']])
    return output.getvalue()
