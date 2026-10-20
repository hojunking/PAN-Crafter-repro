"""Executable translation of the preregistered s1 all-band/bandwise queue."""
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path

from panda30r.plan import baseline as old_baseline
from donor24.plan import evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT/'research_log/PANDA_S1_AllBand_Warp_Bandwise_30H_2026-10-19_v1.json'
SOURCE_SEED = 261016101
SEEDS = {'R1': 261018101, 'R2': 261018102}
HORIZON = 50000
DONORS = ('G8_S1_U23', 'B8A_S1_U23', 'G8_S1_BIC')


def registered_rows():
    value = json.loads(PLAN_PATH.read_text())
    if value['campaign_id'] != CAMPAIGN_ID or value['server'] != 's1' or value['dataset'] != 'WV3':
        raise ValueError('Registered case-plan identity changed')
    rows = value['cases']
    if len(rows) != 14 or any(row['frame'] not in ('pan','mid','mid_return','band_pan') for row in rows):
        raise ValueError('Exactly 14 registered non-MS-only cases required')
    return rows


def baseline(server):
    if server != 's1':
        raise ValueError('This all-band/bandwise campaign is local s1/WV3 only')
    return deepcopy(old_baseline(server))


def groups(server='s1'):
    baseline(server)
    output = []
    for row in registered_rows():
        name = row['budget_group']
        if not output or output[-1]['name'] != name:
            output.append({'name':name, 'optional':row['priority'] != 'priority',
                           'runs':[], 'admission':'whole_group_or_skip'})
        output[-1]['runs'].append((row['case_id'],row['replica']))
    return output


def default_config(server='s1'):
    return {'campaign_id':CAMPAIGN_ID, 'method_revision':METHOD_REVISION,
        'server':server, 'dataset':'WV3', 'baseline_resolved':baseline(server),
        'case_plan_sha256':hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        'budget':deepcopy(BUDGET), 'donors':list(DONORS), 'groups':groups(server),
        'donor_policy':'FRESH_LOCAL_INTERLEAVED_STAGE1_H50_BEST_HQNR_THEN_ERGAS',
        'donor_source_seed':SOURCE_SEED, 'allow_cross_server_donor':False,
        'allow_old_donor_reuse':False, 'checkpoint_selection':POLICY.manifest(),
        'sheet_selection':'ALL_OFFICIAL_EVALUATED_CHECKPOINTS',
        'evaluation':'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20',
        'fixed_loss':{'alpha':.25, 'beta':.05, 'gamma':.5},
        'optional_policy':'PREREGISTERED_ORDER_WITH_MEASURED_WHOLE_GROUP_ADMISSION',
        'time_prior_hours':{'stage1':2.60, 'stage2':3.353},
        'priority_policy':'ORDERED_ATTEMPTS_NOT_GUARANTEED_COMPLETION', 'pruning':'NONE'}


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered 30h all-band configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    from panda_stage2_condq.training import TrainingSpec
    from .spec import BandSpec, geometry_manifest
    campaign = default_config(server)
    matches = [r for r in registered_rows() if (r['case_id'],r['replica']) == (case,replica)]
    if len(matches) != 1 or type(attempt) is not int or attempt < 1:
        raise ValueError('Registered case/replica and positive attempt required')
    row = matches[0]
    donor = row['phase'] == 'stage1'
    seed = row['seed']
    seeds = stream_seeds(seed)
    seeds['rr_band_augmentation'] = 1 + int.from_bytes(hashlib.sha256(
        f'donor30:{seed}:rr_band_augmentation'.encode()).digest()[:8],'big') % (2**63-1)
    bands = row['aligner_outputs'] == 16
    extra = {'upsampler':row['upsampler'], 'frame':row['frame'],
        'shift_scale':float(row['shift_scale']), 'aligner_outputs':row['aligner_outputs'],
        'alignment_target':row['stage1_alignment_target'], 'pan_channels':8 if bands else 1}
    spec = BandSpec('WV3','panda_condq',seeds['S_init'],.25,.05,.5,1e-6,
                    mask_policy=LOSS_MASK_POLICY, **extra)
    b = campaign['baseline_resolved']
    return {'campaign_id':CAMPAIGN_ID, 'method_revision':METHOD_REVISION,
        'server':server, 'dataset':'WV3', 'case':case, 'loss_case':case,
        'replica':replica, 'attempt':attempt, 'seed':seed,
        'run_id':f'PB30_WV3_S1_{case}_{replica}_S{seed}', 'run_display':f'BAND30 {case} {replica}',
        'stream_seeds':seeds, 'stream_seed_protocol':SEED_PROTOCOL+'+rr_band_augmentation',
        'baseline_resolved':deepcopy(b), 'schedule':'H50', 'start_update':0,
        'planned_horizon':HORIZON, 'evaluation_steps':list(evaluation_steps()),
        'retention_steps':[HORIZON], 'phase':'donor' if donor else 'stage2', 'stage2':not donor,
        'engine_kind':'stage1_band' if donor else 'stage2_band', 'initialization':'FRESH',
        'aligner_kind':'direct', **extra, 'geometry':geometry_manifest(spec),
        'band_augmentation':row['band_augmentation'],
        'band_augmentation_spec':{'common_radius':1.5,'residual_radius':.5,
            'distribution':'UNIFORM_AREA_DISK','branch':'HR_LMS_GT_ALIGNER_ONLY',
            'consistency':'u_aux-delta-stopgrad(u_native)'} if donor and bands else None,
        'lr_U':1e-4, 'lr_A':1e-5 if donor else 0., 'weight_decay':.01,
        'lambda_rec':1., 'lambda_abs':2. if donor else 0.,
        'lambda_epsilon':.003 if donor else 0., 'lambda_eps_RR':.003 if donor else 0.,
        'epsilon_radius_rr':3., 'epsilon_distribution':'UNIFORM_AREA_DISK',
        'hqnr_min':.959, 'ergas_max':2.040, 'checkpoint_selection':POLICY.manifest(),
        'loss_mask_policy':LOSS_MASK_POLICY, 'mask_policy':'MASK2',
        'baseline_alignment_protocol_digest':alignment_protocol_digest(b),
        'config_digest':canonical_sha256(campaign),
        'evidence_level':'PAIRED_SCREEN_COMPOUND_BAND_BRANCH_NOT_STATISTICAL_PROOF',
        'student_architecture':{'bands':8,'width':112,'depth':[1,2,3]},
        'stage2_spec':asdict(spec),
        'augmentation':{'hflip_probability':1.,'vflip_probability':1.,'rot90':True},
        'training_spec':asdict(TrainingSpec(lr=1e-4,total_updates=HORIZON,
                                           warmup_updates=100,full_audit_interval=100)),
        'donor_version':case if donor else row['donor_case'],
        'question':'Fresh mean8 global donor' if donor and not bands else
            'Compound band16/PAN8/paired MS-GT aux branch' if bands else
            'Fixed all-band donor coordinate/operator/strength comparison'}


def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])


def catalog_csv(server='s1'):
    output=io.StringIO(); writer=csv.writer(output)
    writer.writerow(['order','phase','case','replica','seed','upsampler','frame',
                     'shift_scale','aligner_outputs','pan_channels','optional','updates'])
    order=0
    for group in groups(server):
        for case,replica in group['runs']:
            order+=1; cfg=resolve_case(server,case,replica)
            writer.writerow([order]+[cfg[k] for k in ('phase','case','replica','seed','upsampler',
                'frame','shift_scale','aligner_outputs','pan_channels')]+[group['optional'],HORIZON])
    return output.getvalue()
