"""Preregistered paired experiments; importing never launches a job."""
from copy import deepcopy
import csv
import io
from pathlib import Path

from donor24.plan import baseline, evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY
from panda_method.calibration import canonical_sha256
from panda_stage2_condq.selection import SelectionPolicy
from panda_stage2_condq.spec import MASK_POLICY
from directshift48.plan import alignment_protocol_digest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1'
METHOD_REVISION = 'PANDA40-S2D-MASK2-CONDQ-v1'
SERVERS = ('s1', 's3')
SEEDS = {'R1': 261016101, 'R2': 261016102}
HORIZON = 50000
BUDGET = {'total_seconds': 40*3600, 'closeout_seconds': 1800,
          'safety_factor': 1.2}
POLICY = SelectionPolicy(.959)

# Geometry is the only changed variable within a D/S pair. Existing A1 is
# HQNR-promising, not ERGAS-optimal; A3/A5 bracket epsilon/absolute pressure.
STAGE1 = {
    'D_A1': ('direct', 2., .003, 1e-5),
    'S_A1': ('s2d', 2., .003, 1e-5),
    'D_A3': ('direct', 3., .0015, 1e-5),
    'S_A3': ('s2d', 3., .0015, 1e-5),
    'D_A5': ('direct', 2., .0015, 1e-5),
    'S_A5': ('s2d', 2., .0015, 1e-5),
    'S_LR': ('s2d', 2., .003, 5e-6),
}
# One fixed A/T across all s3 cases. alpha, beta, gamma are deliberately
# moderate hypotheses, not claimed optima. G/E/ES/Q isolate each increment.
STAGE2 = {
    'G': ('gt_only', 0., 0., 0., LOSS_MASK_POLICY),
    'E': ('panda_e', .5, 0., 0., LOSS_MASK_POLICY),
    'ES': ('panda_e', .5, .05, 0., LOSS_MASK_POLICY),
    'Q': ('panda_condq', .5, .05, .5, LOSS_MASK_POLICY),
    'Q_FULL': ('panda_condq', .5, .05, .5, MASK_POLICY),
    'Q_WEAK': ('panda_condq', .25, .025, .25, LOSS_MASK_POLICY),
}


def groups(server):
    if server == 's1':
        return [{'name': label, 'optional': label == 'S2D_half_LR', 'runs': runs}
                for label, runs in [
            ('A1_geometry', [(c,r) for r in SEEDS for c in ('D_A1','S_A1')]),
            ('A3_geometry', [(c,r) for r in SEEDS for c in ('D_A3','S_A3')]),
            ('A5_geometry', [(c,r) for r in SEEDS for c in ('D_A5','S_A5')]),
            ('S2D_half_LR', [('S_LR',r) for r in SEEDS])]]
    if server == 's3':
        return [{'name': c, 'optional': c == 'Q_WEAK', 'runs': [(c,r) for r in SEEDS]}
                for c in STAGE2]
    raise ValueError('Only s1/s3 WV3 registered')


def default_config(server):
    if server not in SERVERS:
        raise ValueError('Only s1/s3 WV3 registered')
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'budget': dict(BUDGET), 'groups': deepcopy(groups(server)),
        'checkpoint_selection': POLICY.manifest(),
        'sheet_selection': {'HQNR_min': .960, 'ERGAS_max': 2.045,
                            'otherwise': 'best_HQNR_per_run'},
        'donor_policy': 'ONE_LOCAL_COMPLETED_MASK2_DIRECT_DONOR24_OR_FRESH_D_A1',
        'allow_exploratory_donor': True,
        'evaluation': 'ORIGINAL_REFERENCES_UNMASKED_RR20_FR20_SAME_STATE',
        'pruning': 'NONE_COMPLETE_ADMITTED_GROUPS_UNLESS_STOP_OR_DEADLINE'}


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered 40h campaign configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    config = default_config(server)
    if replica not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError('Unknown replica/attempt')
    # s3 D_A1 is only the registered missing-donor fallback, not another
    # changing donor per Stage 2 loss case.
    if case not in (STAGE1 if server == 's1' else {**STAGE2, 'D_A1': STAGE1['D_A1']}):
        raise ValueError('Unregistered case for this server')
    seed, b = SEEDS[replica], config['baseline_resolved']
    cfg = {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'case': case, 'loss_case': case,
        'replica': replica, 'attempt': attempt, 'seed': seed,
        'run_id': f'P40_WV3_{server.upper()}_{case}_{replica}_S{seed}',
        'stream_seeds': stream_seeds(seed), 'stream_seed_protocol': SEED_PROTOCOL,
        'baseline_resolved': deepcopy(b), 'schedule': 'H50',
        'planned_horizon': HORIZON, 'evaluation_steps': list(evaluation_steps()),
        'lr_U': 1e-4, 'weight_decay': .01, 'hqnr_min': .959, 'ergas_max': 2.040,
        'checkpoint_selection': POLICY.manifest(), 'initialization': 'FRESH',
        'baseline_alignment_protocol_digest': alignment_protocol_digest(b),
        'config_digest': canonical_sha256(config)}
    if case in STAGE1:
        kind,a,e,lr = STAGE1[case]
        cfg.update(phase='donor',stage2=False,aligner_kind=kind,lambda_rec=1.,
            lambda_abs=a,lambda_epsilon=e,lambda_eps_RR=e,lr_A=lr,
            loss_mask_policy=LOSS_MASK_POLICY,donor_checkpoint=None,
            epsilon_radius_rr=3.,epsilon_distribution='UNIFORM_AREA_DISK',
            epsilon_every_update=True,epsilon_reduction='RR_SAMPLE_MEAN',
            run_display=f'{case} {kind} a{a:g} eps{e:g} lrA{lr:g} M2 {replica}')
    else:
        variant,a,beta,gamma,mask = STAGE2[case]
        spec = {'sensor': 'WV3', 'variant': variant, 'seed': cfg['stream_seeds']['S_init'],
                'alpha': a, 'beta': beta, 'gamma': gamma, 'delta': 1e-6, 'mask_policy': mask}
        cfg.update(phase='stage2',stage2=True,aligner_kind='direct',lr_A=0.,
            lambda_abs=0.,lambda_epsilon=0.,loss_mask_policy=mask,stage2_spec=spec,
            student_architecture={'bands':8,'width':112,'depth':[1,2,3]},
            run_display=f'{case} a{a:g} b{beta:g} q{gamma:g} '+
                ('M2' if mask == LOSS_MASK_POLICY else 'FULL')+f' W112D123 P0 {replica}')
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'], record['ERGAS'], record['completed_update'])


def choose(records):
    return max(records, key=rank) if records else None


def catalog_csv():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['server','group','optional','case','replica','seed','phase','run_display','horizon'])
    for server in SERVERS:
        for group in groups(server):
            for c,r in group['runs']:
                cfg = resolve_case(server,c,r)
                writer.writerow([server,group['name'],group['optional'],c,r,cfg['seed'],
                                 cfg['phase'],cfg['run_display'],HORIZON])
    return out.getvalue()
