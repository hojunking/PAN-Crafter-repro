"""Preregistered, paired within-server tests of existing PANDA components.

No ensemble, new architecture, new loss, or changed official metric is used.
Each server owns one A1/R1 lineage; E/L/M are interventions on that lineage,
not a cross-server search for the most favorable scalar score.
"""
from copy import deepcopy
import csv
import io
from pathlib import Path

from donor24.plan import baseline as old_baseline, evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from panda_stage2_condq.selection import SelectionPolicy

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1'
METHOD_REVISION = 'PANDA30R-DIRECT-MASK2-SCHEDULE-v1'
SERVERS = ('s1', 's3', 's4')
SEEDS = {'R1': 261018101, 'R2': 261018102}
SOURCE_SEED = 261016101
HORIZON = 50000
BUDGET = {'total_seconds': 30*3600, 'closeout_seconds': 1800, 'safety_factor': 1.2}
POLICY = SelectionPolicy(.959)

# name: donor version, conditional-q/GT, beta, LR, flip probability, question
STANDARD = {
    'E_Q': ('E', 'panda_condq', .05, 1e-4, 1., 'Early A/T reference; compare L_Q and M_Q'),
    'L_Q': ('L', 'panda_condq', .05, 1e-4, 1., 'Same lineage final50K A/T; maturity and alignment move together'),
    'M_G': ('M', 'gt_only', 0., 1e-4, 1., 'Mature donor A, GT-only paired control'),
    'M_Q': ('M', 'panda_condq', .05, 1e-4, 1., 'Early A frozen, T continued to50K; main common reference'),
    'M_B2': ('M', 'panda_condq', .2, 1e-4, 1., 'Only soft KD beta .05 to .2 vs M_Q'),
    'M_B5': ('M', 'panda_condq', .5, 1e-4, 1., 'Optional stronger soft KD beta .5; not an assumed improvement'),
    'M_LR5': ('M', 'panda_condq', .05, 5e-5, 1., 'Only Student peak LR halved vs M_Q'),
    'M_FLIP': ('M', 'panda_condq', .05, 1e-4, .5, 'Only H/V flip probability1 to.5; rotation unchanged'),
}
TAILS = {
    'T40_G': (40000, 'tail_gt', 'Frozen A, GT-only matched tail control'),
    'T40_EPS': (40000, 'tail_fc2_eps', 'A.fc2 first5K; rec + .003 epsilon; then refreeze'),
    'T40_ABS': (40000, 'tail_fc2_abs', 'A.fc2 first5K; rec + .003 epsilon + .2 Scharr; then refreeze'),
    'T20_G': (20000, 'tail_gt', 'Optional20K fork frozen-A matched GT tail'),
    'T20_EPS': (20000, 'tail_fc2_eps', 'Optional earlier20K fork, A.fc2 first5K then frozen'),
    'T20_ABS': (20000, 'tail_fc2_abs', 'Optional earlier20K fork, add weak absolute Scharr'),
}


def baseline(server):
    if server not in SERVERS:
        raise ValueError('Registered servers are s1/s3/s4, all WV3')
    result = deepcopy(old_baseline('s1'))  # old s3 registry is GF2: never reuse it.
    result.update(server=server, dataset='WV3')
    return result


def groups(server):
    baseline(server)
    if server == 's1':
        names = [('E_Q', False), ('L_Q', False), ('M_Q', False), ('M_G', False), ('M_B2', True)]
    elif server == 's3':
        names = [('M_G', False), ('M_Q', False), ('T40_G', False), ('T40_EPS', False),
                 ('T40_ABS', False), ('T20_G', True), ('T20_EPS', True), ('T20_ABS', True),
                 ('M_B2', True)]
    else:
        names = [('M_G', False), ('M_Q', False), ('M_B2', False), ('M_LR5', False),
                 ('M_FLIP', False), ('M_B5', True)]
    return [{'name': name, 'optional': optional, 'anchor': name in ('E_Q','L_Q','M_G','M_Q'),
             'budget_block': 'T20_TRIO' if name.startswith('T20_') else name,
             'runs': [(name, replica) for replica in SEEDS],
             'evidence_level': 'PAIRED_TWO_SEEDS'} for name, optional in names]


def default_config(server):
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'budget': deepcopy(BUDGET), 'groups': groups(server),
        'donor_policy': 'LOCAL_A1_R1_COMPLETED_H50_EARLY_AND_FINAL_SAME_LINEAGE_ELSE_ONE_FRESH_A1',
        'maturation': 'E_A_FROZEN_T_CONTINUE_ORIGINAL_ADAM_STREAM_RNG_LR_TO_LIFETIME50K',
        'mature_binding': 'FINAL50K_NOT_BEST_HQNR',
        'allow_exploratory_donor': True, 'allow_cross_server_donor': False,
        'calibration_reserve_seconds': 600.,
        'checkpoint_selection': POLICY.manifest(),
        'sheet_selection': {'HQNR_min': .960, 'ERGAS_max': 2.045, 'otherwise': 'best_HQNR_per_run'},
        'evaluation': 'ORIGINAL_REFERENCES_UNMASKED_RR20_FR20_SAME_STATE',
        'core_admission': 'ALL_CORE_SEED_PAIRS_RESERVED_BEFORE_STUDENT_QUEUE',
        'tail_rule': 'SAME_M_Q_PARENT_PER_SEED_NO_KD_NO_OLD_CALIBRATION_WHEN_A_CHANGES',
        'checkpoint_retention': 'ALL_OFFICIAL_EVALUATED_STATES_PLUS_SAFE_STOP; BEST_SELECTORS_SEPARATE',
        'pruning': 'NONE; OPTIONAL_GROUPS_SKIP_WHOLE; DURABLE_WALL30H_CLOSEOUT',
        'time_prior': 's3 G50K~1.6h Q50K~2.1h; local100update/full40/save pilot governs admission'}


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered campaign configuration')


def training_spec(*, lr=1e-4, total_updates=HORIZON, warmup_updates=100):
    from dataclasses import asdict
    from panda_stage2_condq.training import TrainingSpec
    return asdict(TrainingSpec(lr=lr, total_updates=total_updates,
                              warmup_updates=warmup_updates, full_audit_interval=100))


def resolve_case(server, case, replica='R1', *, attempt=1, source_update=None):
    config = default_config(server)
    allowed = {tuple(run) for group in groups(server) for run in group['runs']}
    if ((case, replica) not in allowed and (case, replica) not in {('MATURE','R1'),('FRESH_A1','R1')}):
        raise ValueError('Unregistered server/case/replica')
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Positive attempt required')
    seed = SOURCE_SEED if case in ('MATURE','FRESH_A1') else SEEDS[replica]
    b = config['baseline_resolved']
    cfg = {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'case': case, 'loss_case': case,
        'replica': replica, 'attempt': attempt, 'seed': seed,
        'run_id': f'P30R_WV3_{server.upper()}_{case}_{replica}_S{seed}',
        'stream_seeds': stream_seeds(seed), 'stream_seed_protocol': SEED_PROTOCOL,
        'baseline_resolved': deepcopy(b), 'schedule': 'H50', 'start_update': 0,
        'planned_horizon': HORIZON, 'evaluation_steps': list(evaluation_steps()),
        'lr_U': 1e-4, 'weight_decay': .01, 'hqnr_min': .959, 'ergas_max': 2.040,
        'checkpoint_selection': POLICY.manifest(), 'initialization': 'FRESH',
        'aligner_kind': 'direct', 'loss_mask_policy': LOSS_MASK_POLICY,
        'baseline_alignment_protocol_digest': alignment_protocol_digest(b),
        'config_digest': canonical_sha256(config), 'evidence_level': 'PAIRED_TWO_SEEDS',
        'student_architecture': {'bands':8,'width':112,'depth':[1,2,3]},
        'retention_steps': [], 'training_spec': training_spec()}
    if case in ('FRESH_A1','MATURE'):
        cfg.update(phase='donor',stage2=False,engine_kind='fresh_donor' if case=='FRESH_A1' else 'mature_teacher',
            lambda_rec=1., lambda_abs=2. if case=='FRESH_A1' else 0.,
            lambda_epsilon=.003 if case=='FRESH_A1' else 0.,
            lambda_eps_RR=.003 if case=='FRESH_A1' else 0.,
            lr_A=1e-5 if case=='FRESH_A1' else 0., donor_checkpoint=None,
            epsilon_radius_rr=3., epsilon_distribution='UNIFORM_AREA_DISK',
            epsilon_every_update=True, epsilon_reduction='RR_SAMPLE_MEAN',
            donor_version='E' if case=='FRESH_A1' else 'M',
            run_display=f'{case} A'+('train' if case=='FRESH_A1' else 'fixed')+' T50 M2',
            question='Fallback one local A1 lineage' if case=='FRESH_A1' else 'Mature T with early A fixed')
        if case == 'MATURE':
            if type(source_update) is not int or not 0 < source_update < HORIZON:
                raise ValueError('Maturation requires exact original completed_update')
            cfg.update(start_update=source_update, planned_horizon=HORIZON-source_update,
                initialization='CONTINUE_T_ADAM_STREAM_RNG_FROM_EARLY',
                evaluation_steps=sorted({s-source_update for s in evaluation_steps() if s>source_update}),
                retention_steps=[HORIZON-source_update])
    elif case in STANDARD:
        donor, variant, beta, lr, flip, question = STANDARD[case]
        a,g = (0.,0.) if variant=='gt_only' else (.25,.5)
        cfg.update(phase='stage2',stage2=True,engine_kind='stage2_condq',donor_version=donor,
            lr_A=0.,lambda_abs=0.,lambda_epsilon=0.,lr_U=lr,question=question,
            stage2_spec={'sensor':'WV3','variant':variant,'seed':cfg['stream_seeds']['S_init'],
                'alpha':a,'beta':beta,'gamma':g,'delta':1e-6,'mask_policy':LOSS_MASK_POLICY},
            augmentation={'hflip_probability':flip,'vflip_probability':flip,'rot90':True},
            training_spec=training_spec(lr=lr),
            run_display=f'{case} D{donor} Af a{a:g} b{beta:g} q{g:g} lr{lr:g} flip{flip:g} M2 {replica}')
        if server=='s3' and case=='M_Q':
            cfg['retention_steps']=[20000,40000,50000]
    else:
        start,kind,question=TAILS[case]
        horizon=HORIZON-start
        cfg.update(phase='stage2',stage2=True,engine_kind=kind,donor_version='M',
            start_update=start,planned_horizon=horizon,
            evaluation_steps=sorted({s-start for s in evaluation_steps() if s>start}),
            retention_steps=[horizon],schedule=f'TAIL{start//1000}TO50',
            initialization='FORK_S_ADAM_STREAM_RNG_FROM_SAME_M_Q_STEP',
            parent_case='M_Q',parent_replica=replica,parent_update=start,
            lr_U=1e-5,lr_A=0. if kind=='tail_gt' else 1e-6,
            adapt_updates=0 if kind=='tail_gt' else 5000,
            lambda_rec=1.,lambda_epsilon=0. if kind=='tail_gt' else .003,
            lambda_abs=.2 if kind=='tail_fc2_abs' else 0.,
            epsilon_radius_rr=3.,epsilon_distribution='UNIFORM_AREA_DISK',
            training_spec=training_spec(lr=1e-5,total_updates=horizon,warmup_updates=0),
            question=question,
            run_display=f'{case} M_Q@{start//1000}K '+('Af' if kind=='tail_gt' else 'Afc2x5K')+
                f' GT lrS1e-5 lrA{0. if kind=="tail_gt" else 1e-6:g} M2 {replica}')
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])


def catalog_csv(server):
    out=io.StringIO(); w=csv.writer(out)
    w.writerow(['order','group','case','replica','seed','optional','donor','start_update',
                'local_updates','run_display','question'])
    order=0
    for group in groups(server):
        for case,replica in group['runs']:
            order+=1; c=resolve_case(server,case,replica)
            w.writerow([order,group['name'],case,replica,c['seed'],group['optional'],c['donor_version'],
                        c['start_update'],c['planned_horizon'],c['run_display'],c['question']])
    return out.getvalue()
