"""Nine paired Stage1 loss settings, each followed by its own fresh Student."""
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path

from panda_mature30 import plan as previous
from donor30.plan import canonical_sha256, stream_seeds, LOSS_MASK_POLICY, SEED_PROTOCOL
from directshift48.plan import alignment_protocol_manifest as original_alignment_protocol
from .contract import (CAMPAIGN_ID,METHOD_REVISION,RUN_PREFIX,LANES,BUDGET,POLICY,
    STAGE1_SELECTION,CASES,WARP_MANIFEST,SOURCE_FIELDS,input_manifest,sampling_manifest,
    DONORS,LOSS_PAIRS,PAIR_PARAMETERS,stage1_pair,validate_loss_pair_config)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
DATASETS = {s:v['dataset'] for s,v in LANES.items()}
SOURCE_SEED = 261016101
SEEDS = {'R1':261018101}
HORIZON = 50000

def alignment_protocol_manifest(baseline):
    from .warp import boundary_manifest
    value = original_alignment_protocol(baseline)
    value.update(method=METHOD_REVISION,
                 native_warp='single_original_PAN_lanczos4_whole_sample_reflection_FP32',
                 warp=deepcopy(WARP_MANIFEST), boundary=boundary_manifest())
    value['epsilon']['warp'] = deepcopy(WARP_MANIFEST)
    return value

def alignment_protocol_digest(baseline):
    return canonical_sha256(alignment_protocol_manifest(baseline))

def dataset_for_server(server):
    if server not in DATASETS:
        raise ValueError('S1Loss9 is registered only for s1/QB and s3/GF2')
    return DATASETS[server]

def donor_receipt_path(server, donor_case):
    stage1_pair(donor_case)
    return f'work_dir/panda_s1loss9/{CAMPAIGN_ID}/{server}/{dataset_for_server(server)}/donors/{donor_case}/rre_selection.json' 

def lms_overrides(sensor):
    if sensor != 'QB': return {}
    value = json.loads((ROOT/'config/bglobal_rr/qb_original_lms.json').read_text())
    if value['sensor'] != sensor: raise ValueError('Wrong QB override sensor')
    return deepcopy(value['splits'])

def baseline(server):
    sensor = dataset_for_server(server)
    b = deepcopy(previous.baseline('s1'))
    b.update(server=server,dataset=sensor,
        teacher_architecture={'bands':4,'width':112,'depth':[1,2,3]},
        spreadsheet_tab=sensor+'-ablations',spreadsheet_gid=261001011 if sensor=='QB' else 261001012,
        lms_override_path='config/bglobal_rr/qb_original_lms.json' if sensor=='QB' else None,
        lms_override_sha256=hashlib.sha256((ROOT/'config/bglobal_rr/qb_original_lms.json').read_bytes()).hexdigest() if sensor=='QB' else None)
    b['qual_export'].update(display_range_DN=[0,LANES[server]['peak']],rgb_indices=[2,1,0])
    b['alignment_protocol'] = alignment_protocol_manifest(b)
    b['support_policy'] = deepcopy(b['alignment_protocol']['boundary'])
    b['protocol_decisions']['geometry'] = 'R3 epsilon and native correction use Lanczos4; support radius4 plus R3 fits view_margin8, fixed ROI8/stencil7'
    b['protocol_decisions']['s1loss9'] = 'RGB-mean reference Stage1 with all four input/output bands and unified Lanczos4, sensor-specific sources and validation; no old donor reuse'
    return b

def groups(server):
    sensor = dataset_for_server(server)
    # Finish both stages of one pair before starting the next donor. Every
    # Student has independent donor selection and full-train calibration.
    names = [name for key, _, _ in LOSS_PAIRS
             for name in ('S1_' + key, sensor + '_' + key)]
    return [{'name':f'{i:02d}_{name.lower()}','case':name,'optional':False,
             'runs':[(name,'R1')],'admission':'ALL_REGISTERED_RUNS_NO_TIME_CAP'}
            for i,name in enumerate(names)]

def evaluation_steps(horizon, inherited_updates=0):
    return list(range(inherited_updates+1000,horizon+1,1000))

def default_config(server):
    sensor = dataset_for_server(server); lane = LANES[server]
    return {'campaign_id':CAMPAIGN_ID,'method_revision':METHOD_REVISION,
        'plan_revision':'SENSOR_SPECIFIC_STAGE1_LOSS_3X3_PAIRED_STAGE2_20261026',
        'server':server,'dataset':sensor,'baseline_resolved':baseline(server),
        'budget':deepcopy(BUDGET),'donors':list(DONORS),'groups':groups(server),
        'cases':{k:deepcopy(v) for k,v in CASES.items() if v['sensor']==sensor},
        'seeds':deepcopy(SEEDS),'stage1_seed':SOURCE_SEED,
        'donor_policy':'FRESH_LOCAL_RGB_L4_INTERP23_COMPLETE_50K_VAL_RRE_SELECTION',
        'allow_cross_server_donor':False,'stage1_selection':deepcopy(STAGE1_SELECTION),
        'checkpoint_selection':POLICY.manifest(),
        'sheet_selection':'ALL_JOINT_PASSES_PLUS_HQNR_QUALIFIED_MIN_ERGAS_ELSE_BEST_HQNR',
        'evaluation':'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20_SINGLE_FORWARD_NO_TTA',
        'joint_target':{'HQNR':lane['hqnr_min'],'ERGAS':lane['ergas_max']},
        'fixed_loss':{'beta':0.,'delta':1e-6,'rotation_consistency':0.},
        'existing_hard_formula':'(1+alpha*d_e+gamma*d_e*r_q_bar)*mean_bands(abs(S-GT))',
        'fixed_aligner_scale':1.,'warp':deepcopy(WARP_MANIFEST),
        'weight_averaging':'EXCLUDED','band_weighted_loss':'EXCLUDED','rotation_consistency':'EXCLUDED',
        'time_prior_seconds_per_update':{'s1':.30,'s3':.20},
        'priority_policy':'FRESH_BASELINE_THEN_ONE_FACTOR_THEN_INTERACTIONS_BEFORE_SENSOR5_LR_AND_WV3',
        'evidence_level':'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
        'comparison_scope':'WITHIN_DATASET_MATCHED_PAIR_SPECIFIC_DONOR_SAME_INITIALIZATION_AND_STREAM_SEEDS',
        'stage1_loss_grid':deepcopy(PAIR_PARAMETERS),
        'stage1_objective':'L_rec + lambda_abs * L_abs + lambda_epsilon * L_epsilon',
        'baseline_control':'FRESH_IMPLEMENTATION_REPRODUCTION_NOT_INDEPENDENT_SEED_REPLICATION',
        'after_campaign':'RESUME_REMAINING_SENSOR5_LR_LONG_HARD_THEN_WV3_TUNE30_HARD36',
        'before_campaign':'FINISH_ACTIVE_SENSOR5_RUN_AND_SENSOR5_BASE_STUDENT',
        'lms_overrides':lms_overrides(sensor)}

def validate_config(cfg):
    if canonical_sha256(cfg)!=canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered S1Loss9 config')

def resolve_case(server,case,replica='R1',*,attempt=1):
    from .spec import MatureSpec, policy_manifest
    from .engine import ScheduleTrainingSpec
    from panda_stage2_condq.training import TrainingSpec
    sensor = dataset_for_server(server); lane = LANES[server]
    if ((case,replica) not in [r for g in groups(server) for r in g['runs']]
            or type(attempt) is not int or attempt<1):
        raise ValueError('Unregistered S1Loss9 case/replica/attempt')
    donor = case in DONORS
    pair = stage1_pair(case) if donor else CASES[case]['loss_pair']
    loss_parameters = deepcopy(PAIR_PARAMETERS[pair])
    seed = SOURCE_SEED if donor else SEEDS[replica]
    seeds = stream_seeds(seed)
    seeds['rr_band_augmentation'] = 1+int.from_bytes(hashlib.sha256(
        f'donor30:{seed}:rr_band_augmentation'.encode()).digest()[:8],'big')%(2**63-1)
    b = baseline(server); horizon = 50000 if donor else CASES[case]['total_updates']
    cfg = {'campaign_id':CAMPAIGN_ID,'method_revision':METHOD_REVISION,
        'server':server,'dataset':sensor,'case':case,'loss_case':case,'replica':replica,
        'attempt':attempt,'seed':seed,'run_id':f'{RUN_PREFIX}_{sensor}_{server.upper()}_{case}_NOSOFT_{replica}_S{seed}',
        'run_display':f'S1LOSS9 {sensor} {case} {replica}',
        'stream_seeds':seeds,'stream_seed_protocol':SEED_PROTOCOL+'+rr_band_augmentation',
        'baseline_resolved':b,'schedule':'H50' if donor else 'WARMUP_COSINE',
        'start_update':0,'inherited_updates':0,'planned_horizon':horizon,
        'continuation_updates':horizon,'evaluation_steps':evaluation_steps(horizon),
        'retention_steps':[horizon],'phase':'donor' if donor else 'stage2',
        'stage2':not donor,'engine_kind':'stage1_mature' if donor else 'stage2_mature',
        'initialization':'FRESH','aligner_kind':'direct','upsampler':'interp23',
        'frame':'pan','shift_scale':1.,'aligner_outputs':2,'pan_channels':1,
        'alignment_target':'SCHARR_OF_RGB_MEAN_GT','policy_id':'S100',
        'band_augmentation':'NONE','band_augmentation_spec':None,
        'lr_U':1e-4,'lr_A':1e-5 if donor else 0.,'weight_decay':.01,
        'lambda_rec':1.,'lambda_abs':loss_parameters['lambda_abs'] if donor else 0.,
        'lambda_epsilon':loss_parameters['lambda_epsilon'] if donor else 0.,
        'lambda_eps_RR':loss_parameters['lambda_epsilon'] if donor else 0.,
        'loss_pair':pair,'stage1_loss_parameters':loss_parameters,
        'epsilon_radius_rr':3.,'epsilon_distribution':'UNIFORM_AREA_DISK',
        'hqnr_min':lane['hqnr_min'],'ergas_max':lane['ergas_max'],
        'checkpoint_selection':deepcopy(STAGE1_SELECTION) if donor else POLICY.manifest(),
        'loss_mask_policy':LOSS_MASK_POLICY,'mask_policy':'MASK2',
        'loss_mask_policy_compatibility':'LEGACY_ENUM_FOR_SHARED_MASKED_REDUCTION_ONLY',
        'effective_warp_support':'LANCZOS4_FLOOR_MINUS3_THROUGH_PLUS4',
        'baseline_alignment_protocol_digest':alignment_protocol_digest(b),
        'config_digest':canonical_sha256(default_config(server)),
        'evidence_level':'EXPLORATORY_TEST_AWARE_PAIRED_SCREEN_NOT_CONFIRMATORY',
        'student_architecture':{'bands':4,'width':112,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
        'augmentation':{'hflip_probability':1.,'vflip_probability':1.,'rot90':True},
        'training_spec':asdict(TrainingSpec(lr=1e-4,total_updates=horizon,warmup_updates=100,full_audit_interval=100)),
        'donor_version':'S1_'+pair,'stage1_retrained':True,'inference_tta':False,
        'model_variant':'RAW','weight_averaging':'EXCLUDED','soft_distillation':'REMOVED',
        'student_views':1,'rotation_consistency_weight':0.,'pair_transform':'NONE',
        'counter_policy':'ABSOLUTE_STUDENT_UPDATE','question':'Stage1 L_abs/L_epsilon coefficient effects on full-scale paired Stage2 reconstruction',
        **{key:None for key in SOURCE_FIELDS}}
    if donor:
        cfg['geometry'] = {'schema':'PANDA_S1LOSS9_STAGE1_GEOMETRY_V1','shift_scale':1.,
            'frame':'pan','aligner_outputs':2,'alignment_target':'SCHARR_OF_RGB_MEAN_GT',
            'upsampler':'interp23','student_warp':'NOT_APPLICABLE_STAGE1',
            'teacher_warp':'lanczos4_reflection','output_coordinates':'MS','loss_coordinates':'MS'}
        validate_loss_pair_config(cfg)
        return cfg
    p = deepcopy(CASES[case])
    spec = MatureSpec(sensor,'panda_condq',seeds['S_init'],p['alpha'],0.,p['gamma'],1e-6,
        mask_policy=LOSS_MASK_POLICY,policy_id='S100',total_updates=horizon,
        sampling_variant='UBIL',input_variant='CORRECTED_DIFF',student_width=112)
    training = ScheduleTrainingSpec(lr=p['lr_S'],total_updates=horizon,warmup_updates=100,
        min_lr=0.,full_audit_interval=100,inherited_updates=0,schedule_kind='WARMUP_COSINE')
    cfg.update(stage2_spec=asdict(spec),geometry=policy_manifest(spec),
        sampling_variant='UBIL',input_variant='CORRECTED_DIFF',
        student_sampling=sampling_manifest('UBIL','CORRECTED_DIFF'),
        student_input=input_manifest('CORRECTED_DIFF'),student_warp=deepcopy(WARP_MANIFEST),
        lr_U=p['lr_S'],lr_S=p['lr_S'],training_spec=asdict(training),tuning_parameters=p,
        comparison_control={'campaign_id':CAMPAIGN_ID,'case':sensor+'_A200_E003','server':server,
            'replica':replica,'seed':seed,'fresh_student_no_weights_reused':True,
            'single_factor_comparison':sum(loss_parameters[k] != PAIR_PARAMETERS['A200_E003'][k] for k in loss_parameters)==1,
            'factor':'STAGE1_LOSS_COEFFICIENT_PAIR',
            'baseline_interpretation':'IMPLEMENTATION_REPRODUCTION_CONTROL_NOT_INDEPENDENT_SEED_REPLICATE'})
    validate_loss_pair_config(cfg)
    return cfg

def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])

def source_checkpoint_path(cfg):
    raise ValueError('S1Loss9 registers only fresh Students')

def catalog_csv(server):
    out=io.StringIO(); writer=csv.writer(out,lineterminator='\n')
    writer.writerow(['order','server','dataset','phase','case','replica','seed','updates','lr_A','lr_U','lambda_abs','lambda_epsilon','donor_case','alpha','beta','gamma','scale','input_channels','HQNR_min','ERGAS_max'])
    for i,g in enumerate(groups(server)):
        c=resolve_case(server,*g['runs'][0]); s=c.get('stage2_spec',{})
        writer.writerow([i,server,c['dataset'],c['phase'],c['case'],c['replica'],c['seed'],
            c['planned_horizon'],c['lr_A'],c['lr_U'],c['stage1_loss_parameters']['lambda_abs'],
            c['stage1_loss_parameters']['lambda_epsilon'],c['donor_version'],s.get('alpha'),0,s.get('gamma'),1,
            5 if c['phase']=='donor' else 6,c['hqnr_min'],c['ergas_max']])
    return out.getvalue()
