"""MD-derived Table A case registry. Never creates work or contacts another server."""
from copy import deepcopy
from pathlib import Path
import re

from maina_hqnr.common import (ROOT, camp, run_dir, object_sha, sha256,
    read_json, source_identity, verify_server)

CAMPAIGN_ID = CAMPAIGN = 'PANDA_MAINA_WV3_S45_HQNR_20260925_v1'
SOURCE_PLAN = 'research_log/PANDA_WV3_TableA_S45_HQNR_Continuous_ExperimentPlan_2026-09-25.md'
ORIGINAL_RUN = 'FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1'
ORIGINAL_RELEASE = 'cccedeeeffd7ed19686e5ea684489ceee23cf313'
TEACHER_RUN = 'FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1'
TEACHER_SHA = TEACHER_SHA256 = '04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519'
BRIDGE_SHA = 'b902e12b376c2cfdbd69d89c21b732224fec2a52183eca4d5983147e7598c92d'
CALIBRATION_SHA = '45cae955010ab044790442885206b1ffd4d8cecaf5bdf47121747c2fbcc51658'
Q_CACHE_SHA = '96774f7a7ddc73e151d03009066601d61c40b0b4f9d8bbc88cfc2e41a5ac1c66'
DATA_SHA = 'd2d49536a1e692343bb8c9f2dd64cfebe9e331d15df455b9941ece1c0128a4c2'
EVALUATOR_SHA = '79a7e4a154547b355e94e4198a5d85a5a4ecf220ebc039da1b3a3404a91bbc89'
INITIAL_A_SHA = '1e01b69360dbd7526b77e55718050fad405ade38eb59f56c215ae07edabc65cd'
Q_REF = Q0 = 0.4532603621482849
TAU_R = TAU0 = 0.012118559330701828
GRID_STEPS = tuple(range(1010, 50000, 1010))+(50000,)
GRID_SHA = object_sha(list(GRID_STEPS))
MAX_SEED = 2**32-1
CASES = {
    'BASE': dict(alpha=1., beta=.10, lambda_E=.002, axis='baseline', value=None),
    'AL05': dict(alpha=.5, beta=.10, lambda_E=.002, axis='alpha', value=.5),
    'AL15': dict(alpha=1.5, beta=.10, lambda_E=.002, axis='alpha', value=1.5),
    'BE005': dict(alpha=1., beta=.05, lambda_E=.002, axis='beta', value=.05),
    'BE020': dict(alpha=1., beta=.20, lambda_E=.002, axis='beta', value=.20),
    'ED0006': dict(alpha=1., beta=.10, lambda_E=.0006, axis='lambda_E', value=.0006),
    'ED006': dict(alpha=1., beta=.10, lambda_E=.006, axis='lambda_E', value=.006),
}
CASE_CODES = tuple(CASES)
SELECTION = dict(primary_selection='HQNR_MAX50', secondary_selection='EXACT_50000',
    selection_rule='MAX_RAW_ORIGINAL_FR20_MEAN_HQNR_THEN_LOWER_STEP_V1',
    selection_split='FR20', test_aware=True, independent_test=False,
    expected_candidate_count=50, hqnr_reference='RAW_ORIGINAL_PAN',
    checkpoint_tie_breaker='LOWER_COMPLETED_UPDATES', ergas_used_for_selection=False)


def seed_for(server, cycle):
    verify_server(server)
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
        raise ValueError('Cycle must be a nonnegative integer')
    seed = (73101 if cycle == 0 else 2100000+2*(cycle-1)) if server == 's4' else 2100001+2*cycle
    # Corruption RNG seed is seed+100000 in the inherited recipe; never wrap it.
    if seed+100000 > MAX_SEED:
        raise OverflowError('Student/corruption seed range exhausted; do not wrap')
    return seed


def order_for(server, cycle):
    seed_for(server, cycle)
    pairs = [['AL05','AL15'], ['BE005','BE020'], ['ED0006','ED006']]
    offset = (cycle+(0 if server == 's4' else 2)) % 3
    pairs = pairs[offset:]+pairs[:offset]
    if (cycle+(0 if server == 's4' else 1)) % 2:
        pairs = [pair[::-1] for pair in pairs]
    return ['BASE']+[code for pair in pairs for code in pair]


def _run_id(server, cycle, seed, code):
    return f'MAINAHQNR_{server.upper()}_C{cycle:06d}_SS{seed}_{code}_PLH_W104_D122_F1_FRESH50_v1'


def make_case(server, cycle, code):
    seed = seed_for(server, cycle)
    if code not in CASES:
        raise ValueError('Only the seven Table A codes are allowed')
    baseline = _run_id(server, cycle, seed, 'BASE')
    case = dict(campaign_id=CAMPAIGN_ID, server=server, cycle=cycle, seed=seed,
        code=code, case_id=code, run_id=_run_id(server,cycle,seed,code),
        local_baseline_run_id=baseline, baseline_run_id=baseline,
        position_in_cycle=order_for(server,cycle).index(code), anchor=server=='s4' and cycle==0,
        input_layout='PLH', student_layout='PLH', width=104, depth=[1,2,2],
        teacher_alias='F1', teacher_seed=71001, teacher_step=50000, teacher_sha=TEACHER_SHA,
        teacher_run_id=TEACHER_RUN, teacher_input_layout='P0', fixed_q_ref=Q_REF,
        fixed_tau_R=TAU_R, U_peak_lr=1e-4, A_peak_lr=3e-6, actual_target_updates=50000,
        **deepcopy(CASES[code]))
    case['case_spec_sha256'] = object_sha(case)
    return case


def validate_case(case):
    if not isinstance(case, dict) or case != make_case(case.get('server'),case.get('cycle'),case.get('code')):
        raise ValueError('Case differs from canonical Table A recipe')
    return case


def case_for(run_id):
    match = re.fullmatch(r'MAINAHQNR_(S[45])_C(\d{6,})_SS(\d+)_([A-Z0-9]+)_PLH_W104_D122_F1_FRESH50_v1',run_id)
    if not match:
        raise ValueError('Not a MAIN-A run ID')
    server, cycle, _, code = match.groups()
    case = make_case(server.lower(),int(cycle),code)
    if case['run_id'] != run_id:
        raise ValueError('Run ID does not match its canonical seed/cycle')
    return case


def cycle_cases(server, cycle):
    return [make_case(server,cycle,code) for code in order_for(server,cycle)]


def next_cursor(server, cycle, position):
    seed_for(server,cycle)
    if isinstance(position,bool) or not isinstance(position,int) or not 0 <= position < 7:
        raise ValueError('Position must be an integer in [0,6]')
    next_cycle, next_position = (cycle+1,0) if position == 6 else (cycle,position+1)
    seed_for(server,next_cycle)
    return dict(cycle=next_cycle,position=next_position)


def iter_cases(server, start_cycle=0, start_position=0):
    cursor = dict(cycle=start_cycle,position=start_position)
    while True:
        yield cycle_cases(server,cursor['cycle'])[cursor['position']]
        cursor = next_cursor(server,**cursor)


def verify_sources(root=ROOT):
    root = Path(root)
    original = read_json(root/'maina_hqnr/spec/original_source_identity.json')
    if not original or original.get('content_sha256') != EVALUATOR_SHA or object_sha(original['files']) != EVALUATOR_SHA:
        raise ValueError('Original main source manifest differs')
    # All 42 original paths are presently exact; fail closed if a later revision
    # changes one, rather than silently claiming parity from a stale git label.
    for name, expected in original['files'].items():
        if not (root/name).is_file() or sha256(root/name) != expected:
            raise ValueError('Original FH20R1 source changed: '+name)
    return original


def build_config(case, root=ROOT, bindings=None, attempt=0):
    from fh20r1.plan import build_config as original_builder, case_for as original_case
    validate_case(case); root = Path(root)
    cfg = deepcopy(original_builder(original_case(ORIGINAL_RUN)))
    cfg.update(seed=case['seed'], work_dir=str(run_dir(case,root,attempt).resolve()),
               trainer='maina_hqnr', resume=False)
    f = cfg['fh20r1']
    for key in ('budget_path','min_effective_hours','hqnr_threshold','ergas_goal','eligible_when'):
        f.pop(key,None)
    f.update(server_id=case['server'], block_id=f"MAINA_{case['server']}_C{case['cycle']:06d}",
        campaign_id=CAMPAIGN_ID, source_plan=SOURCE_PLAN, run_id=case['run_id'],
        config_path=str(run_dir(case,root,attempt)/'config.json'),
        student_seed=case['seed'], seed=case['seed'], corruption_seed=case['seed']+100000,
        alpha=case['alpha'], beta=case['beta'], lambda_E=case['lambda_E'],
        profile='BASE', calibration=dict(n_patches=3072,seed=1234,probe='AXIS16',
            reuse_original_F1=True, recalibration_permitted=False),
        q_ref=Q_REF, tau_R=TAU_R, retain_all_candidates=True,
        primary_selection='HQNR_MAX50', secondary_selection='EXACT_50000')
    if bindings is not None:
        if bindings.get('server') != case['server'] or bindings.get('campaign_id') != CAMPAIGN_ID:
            raise ValueError('Bindings belong to a different server/campaign')
        f['dataset_manifest'] = deepcopy(bindings['dataset_manifest'])
        f['reference_bridge'] = bindings['canonical_bridge_path']
        f['teacher_checkpoint'] = bindings['teacher']['checkpoint']
        for split,key in dict(train='train_feeder_args',val='val_feeder_args',
            rr='test_reduced_feeder_args',fr='test_full_feeder_args').items():
            cfg[key]['dataroot'] = bindings['dataset_manifest']['splits'][split]['dataroot']
    cfg['maina_hqnr'] = dict(case=deepcopy(case), attempt=attempt,
        bindings_path=str(camp(root,case['server'])/'runtime_bindings.json'),
        binding_sha256=object_sha(bindings) if bindings is not None else None,
        source_identity=source_identity(root) if bindings is not None else None,
        candidate_grid=list(GRID_STEPS), candidate_grid_sha=GRID_SHA,
        evaluator_sha=EVALUATOR_SHA, original_release=ORIGINAL_RELEASE,
        original_main_run=ORIGINAL_RUN, fixed_q_ref=Q_REF, fixed_tau_R=TAU_R,
        q_ref_multiplier=1., tau_R_multiplier=1., rA=.03,
        coefficient_override_scope=['alpha','beta','lambda_E'], **SELECTION)
    return cfg


def validate_config(cfg, root=ROOT, bindings=None, require_bound=False):
    meta = cfg['maina_hqnr']; case = meta['case']; validate_case(case)
    if require_bound and not meta.get('binding_sha256'):
        raise ValueError('Verified runtime bindings required before training')
    if bindings is None and meta.get('binding_sha256'):
        bindings = read_json(meta['bindings_path'])
        if bindings is None: raise ValueError('Runtime bindings are missing')
    if cfg != build_config(case,root,bindings,meta['attempt']):
        raise ValueError('Config differs from original main plus declared Table A override')
    return case


def campaign_spec():
    return dict(campaign_id=CAMPAIGN_ID, source_plan=SOURCE_PLAN,
        provenance='IMPLEMENTATION_DERIVED_FROM_MD; companion CSV/JSON not supplied',
        servers=['s4','s5'], cases=deepcopy(CASES), grid=list(GRID_STEPS),
        selection=SELECTION, reference=dict(teacher_sha=TEACHER_SHA,bridge_sha=BRIDGE_SHA,
            calibration_sha=CALIBRATION_SHA,q_cache_sha=Q_CACHE_SHA,data_sha=DATA_SHA,
            evaluator_sha=EVALUATOR_SHA,q_ref=Q_REF,tau_R=TAU_R),
        repeat='INDEPENDENT_UNBOUNDED_LOCAL_CYCLES', cross_server_lock=False,
        automatic_asset_regeneration=False, preview_is_not_a_run_limit=True)
