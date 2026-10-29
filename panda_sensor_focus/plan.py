"""Six QB and five GF2 fresh Students, each anchored to an exact original donor."""
from copy import deepcopy
from dataclasses import asdict
import csv, hashlib, io
from pathlib import Path
from donor30.plan import canonical_sha256, stream_seeds, LOSS_MASK_POLICY, SEED_PROTOCOL
from panda_sensor_refine import plan as previous
from .contract import *
ROOT=Path(__file__).resolve().parents[1]
PLAN_PATH=Path(__file__).resolve()
DATASETS={s:v['dataset'] for s,v in LANES.items()}
SEEDS={'R1':261018101,'R2':261018102}
SOURCE_SEED=261016101
HORIZON=60000
lms_overrides=previous.lms_overrides

def dataset_for_server(server):
    if server not in DATASETS: raise ValueError('Registered s1/QB or s3/GF2 required')
    return DATASETS[server]

def baseline(server):
    value=previous.baseline(server,3.)
    value['protocol_decisions']['sensor_focus']='Exact original donor; fresh Student width/Hard/LR tuning; no additional methods'
    return value

def groups(server):
    dataset_for_server(server)
    return [dict(name=f'{i:02d}_{case.lower()}_{rep.lower()}',case=case,optional=False,
        runs=[(case,rep)],admission='ALL_REGISTERED_RUNS_NO_TIME_CAP')
        for i,(case,rep) in enumerate(REGISTERED_RUNS[server])]

def default_config(server):
    sensor=dataset_for_server(server); lane=LANES[server]
    return dict(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,server=server,dataset=sensor,
        baseline_resolved=baseline(server),groups=groups(server),budget=deepcopy(BUDGET),
        cases={k:deepcopy(v) for k,v in CASES.items() if v['sensor']==sensor},
        fixed_donor=deepcopy(SOURCE_DONORS[server]),donors=[SOURCE_DONORS[server]['case']],
        donor_policy='EXACT_SHA_ORIGINAL_VAL_RRE_SELECTED_FIXED_DONOR_NO_STAGE1_TRAINING',
        stage1_runs=0,stage2_runs=len(REGISTERED_RUNS[server]),seeds=deepcopy(SEEDS),
        joint_target={'HQNR':lane['hqnr_min'],'ERGAS':lane['ergas_max']},
        sheet_selection='ALL_JOINT_PASSES_PLUS_HQNR_QUALIFIED_MIN_ERGAS_ELSE_BEST_HQNR',
        fixed_loss={'beta':0.,'rotation_consistency':0.},fixed_aligner_scale=1.,
        weight_averaging='EXCLUDED',band_weighted_loss='EXCLUDED',
        time_prior_seconds_per_update={'s1':.30,'s3':.20},
        after_campaign='RESUME_UNCHANGED_WV3_TUNE30_HARD36',
        evidence_level='EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY')

def validate_config(cfg):
    if canonical_sha256(cfg)!=canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered focus configuration')

def resolve_case(server,case,replica='R1',*,attempt=1):
    from .spec import MatureSpec,policy_manifest
    from .engine import ScheduleTrainingSpec
    if (case,replica) not in REGISTERED_RUNS.get(server,()) or type(attempt) is not int or attempt<1:
        raise ValueError('Unregistered focus case/replica/attempt')
    # Preserve original Student seeding, augmentation, optimizer and protocol fields.
    cfg=previous.resolve_case(server,DATASETS[server]+'_BASE','R2',attempt=attempt)
    source=SOURCE_DONORS[server]; p=deepcopy(CASES[case]); seed=SEEDS[replica]
    seeds=stream_seeds(seed)
    seeds['rr_band_augmentation']=1+int.from_bytes(hashlib.sha256(f'donor30:{seed}:rr_band_augmentation'.encode()).digest()[:8],'big')%(2**63-1)
    horizon=p['total_updates']; width=p['student_width']
    spec=MatureSpec(DATASETS[server],'panda_condq' if p['gamma']>0 else 'panda_e',seeds['S_init'],p['alpha'],0.,p['gamma'],1e-6,
        mask_policy=LOSS_MASK_POLICY,policy_id='S100',total_updates=horizon,
        sampling_variant='UBIL',input_variant='CORRECTED_DIFF',student_width=width)
    training=ScheduleTrainingSpec(lr=p['lr_S'],total_updates=horizon,warmup_updates=100,
        min_lr=0.,full_audit_interval=100,inherited_updates=0,schedule_kind='WARMUP_COSINE')
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,case=case,loss_case=case,
        replica=replica,seed=seed,run_id=f'{RUN_PREFIX}_{DATASETS[server]}_{server.upper()}_{case}_NOSOFT_{replica}_S{seed}',
        run_display=f'SENSOR_FOCUS {DATASETS[server]} {case} {replica}',stream_seeds=seeds,
        baseline_resolved=baseline(server),planned_horizon=horizon,continuation_updates=horizon,
        evaluation_steps=list(range(1000,horizon+1,1000)),retention_steps=[horizon],
        config_digest=canonical_sha256(default_config(server)),student_architecture={'bands':4,'width':width,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
        stage2_spec=asdict(spec),geometry=policy_manifest(spec),training_spec=asdict(training),
        student_sampling=sampling_manifest('UBIL','CORRECTED_DIFF',width),
        student_input=input_manifest('CORRECTED_DIFF',width),student_warp=deepcopy(WARP_MANIFEST),
        donor_version=source['case'],stage1_retrained=False,fixed_donor_source=deepcopy(source),
        loss_pair=source['case'],stage1_loss_parameters={k:source[k] for k in ('lambda_abs','lambda_epsilon')},
        lr_S=p['lr_S'],lr_U=p['lr_S'],tuning_parameters=p,
        comparison_control={'campaign_id':source['campaign_id'],'case':'QB_A200_E030' if server=='s1' else 'GF2_LONG',
            'server':server,'replica':'R1','seed':SEEDS['R1'],'fresh_student_no_weights_reused':True,
            'same_exact_donor_checkpoint':True,'student_seed_differs_from_control':replica!='R1'},
        question='Fixed exact donor: separate Student capacity, existing Hard coefficients and sensor-specific schedule/LR')
    validate_loss_pair_config(cfg)
    return cfg

def rank(record): return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])
def source_checkpoint_path(cfg): raise ValueError('Focus Students start fresh; no Student weight reuse')
def catalog_csv(server):
    out=io.StringIO(); w=csv.writer(out,lineterminator='\n')
    w.writerow(['order','server','dataset','case','replica','updates','width','lr','alpha','beta','gamma','scale','donor','selected_step','HQNR_min','ERGAS_max'])
    for i,(case,rep) in enumerate(REGISTERED_RUNS[server],1):
        c=resolve_case(server,case,rep);s=c['stage2_spec']
        w.writerow([i,server,c['dataset'],case,rep,c['planned_horizon'],s['student_width'],c['lr_S'],s['alpha'],0,s['gamma'],1,c['donor_version'],SOURCE_DONORS[server]['selected_step'],c['hqnr_min'],c['ergas_max']])
    return out.getvalue()
