"""MD-pinned LOCAL-T v2 definitions; no admission, clock or runtime mutation.

The author-referenced CSV/DesignRegistry were not supplied. These finite IDs and
queues are explicitly derived from the supplied MD, not claimed as recovered
author registry bytes. Generated exports must retain that provenance.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from qg40.plan import SensorSpec, SplitBinding, sensor_spec as _sensor_spec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_GF2_L100_S345_LOCALT_20H_20260921_v2'
REGISTRY_REVISION = 'L100I1_MD_DERIVED_REGISTRY_v2'
METHOD_REVISION = 'QG40_SYNC_FREQ_C4_v1'
SOURCE_STATUS = 'MD_DERIVED_MISSING_DESIGN_ASSETS'
SOURCE_PLAN = 'research_log/PANDA_GF2_L100_S345_LOCALT_20H_ExperimentPlan_2026-09-21_v2.md'
SOURCE_SHAS = MappingProxyType({SOURCE_PLAN: '7583d453a321afda53eccc6560cd3fb198d296d029e5b9b4162ef832f49bddad'})
MISSING_DESIGN_ASSETS = (
    'PANDA_GF2_L100_S345_LOCALT_20H_Cases_2026-09-21_v2.csv',
    'PANDA_GF2_L100_S345_LOCALT_20H_DesignRegistry_2026-09-21_v2.json')
SERVERS = ('s3', 's4', 's5')
SHEET_TABS = MappingProxyType(dict(s3='GF2-s3(5090)', s4='GF2-s4', s5='GF2-s5'))
WINDOW_HOURS, ADMISSION_CUTOFF_HOURS, TRAIN_FINISH_HOURS = 20., 16., 18.
CLOSE_HOURS, SETUP_HOURS, CALIBRATION_HOURS = 2., .75, 1.
STUDENT_SEEDS = (95001, 95002)
GRID50 = tuple(range(1010, 50000, 1010)) + (50000,)
GRID100 = tuple(50000 if 2 * step == 50500 else 2 * step for step in GRID50)
BINDING_FIELDS = ('dataset_manifest', 'reference_manifest', 'sensor_spec', 'sensor_spec_path',
    'teacher_checkpoint', 'teacher_sha256', 'tau_R', 'q_ref', 'q_cache', 'q_cache_sha256',
    'runtime_policy_sha256')


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verify_sources(root=ROOT):
    for name, expected in SOURCE_SHAS.items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('LOCAL-T source changed; use a new registry revision: ' + name)
    return dict(SOURCE_SHAS)


def sensor_spec(sensor='GF2'):
    if sensor != 'GF2':
        raise ValueError('LOCAL-T is GF2 only')
    return _sensor_spec(sensor)


def grid_steps(updates):
    if isinstance(updates, bool) or updates not in (50000, 100000):
        raise ValueError('Only fresh50K and fresh100K horizons are registered')
    return GRID50 if updates == 50000 else GRID100


def diagnostic_steps(role, updates):
    grid_steps(updates)
    if role not in ('T', 'S'):
        raise ValueError('Role must be T or S')
    steps = (0, 10000, 25000, 50000, 75000, 100000)
    if role == 'S':
        steps = (0, 1000, 5000, 10000, 25000, 50000, 75000, 100000)
    return tuple(step for step in steps if step <= updates)


def fullstate_steps(role, updates):
    return tuple(sorted(set(diagnostic_steps(role, updates)) | {50000, updates}))


@dataclass(frozen=True)
class Profile:
    alpha: float = 1.
    beta: float = .1
    lambda_edge: float = .002
    student_a_peak_lr: float = 3e-6
    aligner_schedule: str = 'BASE'
    status: str = 'CORE'


PROFILES = MappingProxyType(dict(BASE=Profile(), H010=Profile(alpha=.1), E1=Profile(lambda_edge=.001),
    ARW=Profile(aligner_schedule='ARW', status='DEFERRED_USER_RELEASE'),
    E02=Profile(lambda_edge=.0002, status='DEFERRED_USER_RELEASE')))
STUDENT_PROFILES = tuple(PROFILES)


@dataclass(frozen=True)
class Reference:
    reference_id: str
    alias: str
    server_id: str
    teacher_seed: int
    updates: int
    teacher_case_id: str
    calibration_id: str

    @property
    def owner_server(self):
        return self.server_id

    @property
    def lambda_con(self):
        return 1e-4


def _reference(server, seed, updates, case):
    length = str(updates // 1000)
    return Reference(f'L100I1_R{length}_{server.upper()}_TS{seed}',
        f'L100I1_T{length}_{server.upper()}_TS{seed}', server, seed, updates,
        'L100I1-' + case, f'L100I1-CAL-{server.upper()}-{length}')


_REFS = (_reference('s3', 94001, 100000, 'T01'), _reference('s3', 94001, 50000, 'T02'),
         _reference('s4', 94002, 100000, 'T03'), _reference('s5', 94003, 100000, 'T04'))
REFERENCES = MappingProxyType({ref.reference_id: ref for ref in _REFS})


@dataclass(frozen=True)
class Case:
    case_id: str
    run_id: str
    server_id: str
    role: str
    reference_id: str
    teacher_seed: int
    student_seed: int | None
    profile: str
    updates: int
    block_id: str
    queue_rank: int
    within_block_order: int
    status: str = 'PLANNED'
    tier: str = 'CORE'
    sensor: str = 'GF2'
    reference_sha256: str | None = None

    @property
    def seed(self):
        return self.teacher_seed if self.role == 'T' else self.student_seed

    @property
    def width(self):
        return 112 if self.role == 'T' else 104

    @property
    def depth(self):
        return (1, 2, 3) if self.role == 'T' else (1, 2, 2)

    @property
    def input_layout(self):
        return 'P0' if self.role == 'T' else 'PLH'

    @property
    def teacher_input_layout(self):
        return 'P0'

    @property
    def teacher_alias(self):
        return REFERENCES[self.reference_id].alias

    @property
    def teacher_owner(self):
        return REFERENCES[self.reference_id].server_id

    @property
    def owner_server(self):
        return self.teacher_owner

    @property
    def teacher_updates(self):
        return REFERENCES[self.reference_id].updates

    @property
    def teacher_run_id(self):
        return case_for(REFERENCES[self.reference_id].teacher_case_id).run_id

    @property
    def sheet_tab(self):
        return SHEET_TABS[self.server_id]

    @property
    def reservation_hours(self):
        return ({50000: 1., 100000: 1.7} if self.role == 'T' else {50000: 1.3, 100000: 2.4})[self.updates]

    @property
    def lambda_con(self):
        return 1e-4 if self.role == 'T' else 0.

    @property
    def alpha(self):
        return PROFILES[self.profile if self.role == 'S' else 'BASE'].alpha

    @property
    def beta(self):
        return PROFILES[self.profile if self.role == 'S' else 'BASE'].beta

    @property
    def lambda_edge(self):
        return PROFILES[self.profile if self.role == 'S' else 'BASE'].lambda_edge

    @property
    def student_a_peak_lr(self):
        return PROFILES[self.profile].student_a_peak_lr if self.role == 'S' else None

    @property
    def pair_baseline_run_id(self):
        identifier = _BASELINE_IDS.get(self.case_id)
        return case_for(identifier).run_id if identifier else None

    @property
    def is_resolved(self):
        return self.tier == 'CORE' and self.status == 'PLANNED'

    def to_dict(self):
        return dict(asdict(self), seed=self.seed, width=self.width, depth=list(self.depth),
            input_layout=self.input_layout, teacher_input_layout='P0', teacher_alias=self.teacher_alias,
            teacher_owner=self.teacher_owner, teacher_updates=self.teacher_updates,
            teacher_run_id=self.teacher_run_id, sheet_tab=self.sheet_tab, kind='TRAIN',
            alpha=self.alpha, beta=self.beta, lambda_edge=self.lambda_edge,
            student_a_peak_lr=self.student_a_peak_lr, lambda_con=self.lambda_con,
            pair_baseline_run_id=self.pair_baseline_run_id, campaign_id=CAMPAIGN_ID,
            config_path=f'config/l100/{self.run_id}.yaml', is_resolved=self.is_resolved)


_BLOCKS = (
    ('s3', 'L100I1_S3_LOCAL_CHAIN_95001', ('T01', 'T02', 'S01', 'S02', 'S03')),
    ('s3', 'L100I1_S3_LENGTH_95002', ('S04', 'S05', 'S06')),
    ('s4', 'L100I1_S4_LOCAL_H010_95001', ('T03', 'S07', 'S08')),
    ('s4', 'L100I1_S4_H010_95002', ('S09', 'S10')),
    ('s5', 'L100I1_S5_LOCAL_E1_95001', ('T04', 'S11', 'S12')),
    ('s5', 'L100I1_S5_E1_95002', ('S13', 'S14')))
_PLACEMENTS = {'L100I1-' + case: (block, rank, order)
               for rank, (_server, block, cases) in enumerate(_BLOCKS)
               for order, case in enumerate(cases)}


def _make_case(short, server, role, teacher_updates, updates, student_seed=None, profile='BASE', order=0):
    ref = next(ref for ref in _REFS if ref.server_id == server and ref.updates == teacher_updates)
    case_id = 'L100I1-' + short
    block, rank, within = _PLACEMENTS.get(case_id, (f'L100I1_{server.upper()}_DEFERRED_{profile}', 100, order))
    deferred = short.startswith('X')
    profile = 'C100' if role == 'T' else profile
    suffix = f'FRESH{updates // 1000}K_v2'
    run = (f'L100I1_GF2_T_{server.upper()}_TS{ref.teacher_seed}_C100_P0_W112_D123_{suffix}' if role == 'T' else
           f'L100I1_GF2_S_{server.upper()}_R{teacher_updates // 1000}_TS{ref.teacher_seed}_'
           f'SS{student_seed}_{profile}_PLH_W104_D122_{suffix}')
    return Case(case_id, run, server, role, ref.reference_id, ref.teacher_seed, student_seed,
        profile, updates, block, rank, within, status='DEFERRED_USER_RELEASE' if deferred else 'PLANNED',
        tier='DEFERRED' if deferred else 'CORE')


CASES = (
    _make_case('T01', 's3', 'T', 100000, 100000), _make_case('T02', 's3', 'T', 50000, 50000),
    _make_case('T03', 's4', 'T', 100000, 100000), _make_case('T04', 's5', 'T', 100000, 100000),
    _make_case('S01', 's3', 'S', 50000, 50000, 95001), _make_case('S02', 's3', 'S', 50000, 100000, 95001),
    _make_case('S03', 's3', 'S', 100000, 100000, 95001), _make_case('S04', 's3', 'S', 100000, 100000, 95002),
    _make_case('S05', 's3', 'S', 50000, 100000, 95002), _make_case('S06', 's3', 'S', 50000, 50000, 95002),
    _make_case('S07', 's4', 'S', 100000, 100000, 95001), _make_case('S08', 's4', 'S', 100000, 100000, 95001, 'H010'),
    _make_case('S09', 's4', 'S', 100000, 100000, 95002, 'H010'), _make_case('S10', 's4', 'S', 100000, 100000, 95002),
    _make_case('S11', 's5', 'S', 100000, 100000, 95001), _make_case('S12', 's5', 'S', 100000, 100000, 95001, 'E1'),
    _make_case('S13', 's5', 'S', 100000, 100000, 95002, 'E1'), _make_case('S14', 's5', 'S', 100000, 100000, 95002),
    _make_case('X01', 's4', 'S', 100000, 100000, 95001, 'ARW'),
    _make_case('X02', 's4', 'S', 100000, 100000, 95002, 'ARW', order=1),
    _make_case('X03', 's5', 'S', 100000, 100000, 95001, 'E02'),
    _make_case('X04', 's5', 'S', 100000, 100000, 95002, 'E02', order=1))
CORE_CASES = tuple(c for c in CASES if c.tier == 'CORE')
DEFERRED_CASES = tuple(c for c in CASES if c.tier == 'DEFERRED')
_BY_ID = MappingProxyType({case.case_id: case for case in CASES})
_BY_RUN = MappingProxyType({case.run_id: case for case in CASES})
_BASELINE_IDS = {'L100I1-' + key: 'L100I1-' + value for key, value in dict(
    S01='S01', S02='S01', S03='S02', S04='S05', S05='S06', S06='S06',
    S07='S07', S08='S07', S09='S10', S10='S10', S11='S11', S12='S11', S13='S14', S14='S14',
    X01='S07', X02='S10', X03='S11', X04='S14').items()}


def case_for(identifier):
    try:
        return _BY_ID[identifier] if identifier in _BY_ID else _BY_RUN[identifier]
    except (KeyError, TypeError) as exc:
        raise ValueError('Unknown LOCAL-T case: ' + str(identifier)) from exc


def validate_case(case, *, allow_deferred=False):
    if not isinstance(case, Case) or case != case_for(case.case_id):
        raise ValueError('Modified or unregistered LOCAL-T case')
    if case.teacher_owner != case.server_id:
        raise ValueError('Cross-server Teacher references are forbidden')
    if not case.is_resolved and not allow_deferred:
        raise ValueError('DEFERRED_USER_RELEASE is not executable; a new user recipe revision is required')
    return case


def cases_for(server, include_deferred=False):
    if server not in SERVERS:
        raise ValueError('LOCAL-T only permits s3, s4, s5; s1/s2 remain untouched')
    return tuple(sorted((case for case in CASES if case.server_id == server and
        (include_deferred or case.is_resolved)), key=lambda c: (c.queue_rank, c.within_block_order)))


def teacher_for(reference_id):
    if reference_id not in REFERENCES:
        raise ValueError('Unknown local reference')
    return case_for(REFERENCES[reference_id].teacher_case_id)


@dataclass(frozen=True)
class Block:
    server_id: str
    block_id: str
    queue_rank: int
    run_ids: tuple[str, ...]
    cases: tuple[Case, ...]

    def to_dict(self):
        return dict(server_id=self.server_id, block_id=self.block_id, queue_rank=self.queue_rank,
                    run_ids=list(self.run_ids), case_ids=[c.case_id for c in self.cases],
                    calibration_ids=[REFERENCES[c.reference_id].calibration_id for c in self.cases if c.role == 'T'])


def blocks_for(server):
    cases_for(server)
    result = []
    for rank, (owner, name, ids) in enumerate(_BLOCKS):
        if owner == server:
            cases = tuple(case_for('L100I1-' + identifier) for identifier in ids)
            result.append(Block(server, name, rank, tuple(c.run_id for c in cases), cases))
    return tuple(result)


def block_for(identifier):
    for server in SERVERS:
        for block in blocks_for(server):
            if block.block_id == identifier:
                return block
    raise ValueError('Unknown canonical LOCAL-T block')


def build_config(case):
    case = validate_case(case_for(case) if isinstance(case, str) else case)
    spec = sensor_spec()
    meta = dict(case.to_dict(), case_contract=asdict(case), method_revision=METHOD_REVISION,
        registry_revision=REGISTRY_REVISION, source_status=SOURCE_STATUS, source_plan=SOURCE_PLAN,
        sensor_spec=spec.to_dict(), sensor_spec_path=None, num_bands=4, max_pixel=1023, batch_size=48,
        window_path='work_dir/_l100/campaign_window.json', dataset_manifest=None,
        reference_manifest=None, teacher_checkpoint=None, teacher_sha256=None,
        tau_R=None, q_ref=None, q_cache=None, q_cache_sha256=None,
        runtime_policy_path='l100/runtime_policy.json', runtime_policy_sha256=None,
        teacher_from_scratch=True, backbone_from_scratch=True,
        aligner_init='FRESH_ZERO_LAST_LINEAR' if case.role == 'T' else 'TEACHER_CLONE',
        init_policy='FH12_named_tensor_fresh9_zero_extra_v1', init_generalization='Cplus1_native_channels_zero_extra_LH',
        candidate_grid=list(grid_steps(case.updates)), diagnostic_steps=list(diagnostic_steps(case.role, case.updates)),
        fullstate_steps=list(fullstate_steps(case.role, case.updates)), teacher_ref_step=case.teacher_updates,
        no_global_lock=True, lambda_E=case.lambda_edge,
        aligner_lr=1e-5 if case.role == 'T' else case.student_a_peak_lr,
        consistency_weight=case.lambda_con, offset_weight=case.lambda_con, offset_every=2,
        offset_radius=2. if case.role == 'T' else 0., view_margin_hr=4, corruption_seed=case.seed + 100000,
        calibration=dict(n_patches=3072, seed=1234, probe='AXIS16', views='FIXED_HV_ROT4', batch_size=16,
                         subset_scope='SENSOR_SHARED_TRAIN_IDS'),
        hqnr_threshold=.964, ergas_goal=.552, ergas_strong_goal=.522,
        lr_schedule_id=f'COSINE_N{case.updates}_W100_BASE_v2', a_lr_switch_completed_updates=None,
        a_lr_after_multiplier=1.,
        lp=dict(sigma=1.98, kernel_size=41, padding='replicate', decimation='2::4,2::4',
                generation_dtype='float64', cache_dtype='float32'))
    return dict(seed=case.seed, work_dir=f'work_dir/{case.run_id}', trainer='l100', phase='train',
        num_iter=case.updates, num_warmup=100, batch_size=48, test_batch_size=1, num_worker=4,
        num_bands=4, max_pixel=1023., mixed_precision='no', learning_rate=1e-4,
        optimizer='AdamW', weight_decay=.01, betas=[.9, .999], eps=1e-8, lr_scheduler='cosine', log_iter=100,
        train_feeder_args=dict(dataroot=None, crop=False, hflip=True, vflip=True, rot=True, ms_size=16, return_meta=True),
        val_feeder_args=dict(dataroot=None), test_reduced_feeder_args=dict(dataroot=None), test_full_feeder_args=dict(dataroot=None),
        model_args=dict(hidden_size=case.width, depth=list(case.depth), out_channels=4, attn_locations=[],
                        mode_modulation=False, norm='ln', dropout=0.), l100=meta)


def case_from_config(cfg):
    try:
        case = Case(**cfg['l100']['case_contract'])
    except (KeyError, TypeError) as exc:
        raise ValueError('LOCAL-T requires a complete immutable case_contract') from exc
    validate_case(case)
    if cfg.get('trainer') != 'l100' or any(cfg['l100'].get(k) != v for k, v in case.to_dict().items()):
        raise ValueError('LOCAL-T case metadata differs from its immutable registered contract')
    return case


def validate_config(cfg, *, require_bound=False):
    case = case_from_config(cfg)
    expected = build_config(case)
    if Path(cfg.get('work_dir', '')).name != case.run_id:
        raise ValueError('Work-directory run identity differs')
    expected['work_dir'] = cfg['work_dir']
    try:
        for key in BINDING_FIELDS:
            expected['l100'][key] = copy.deepcopy(cfg['l100'][key])
        for key in ('train_feeder_args', 'val_feeder_args', 'test_reduced_feeder_args', 'test_full_feeder_args'):
            expected[key]['dataroot'] = cfg[key]['dataroot']
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing local config binding field') from exc
    if cfg != expected:
        raise ValueError('LOCAL-T config differs from the immutable numerical recipe')
    if require_bound and (not cfg['l100']['dataset_manifest'] or not valid_sha(cfg['l100']['runtime_policy_sha256'])):
        raise ValueError('Bind verified local data and runtime policy before training')
    return case


def registry_rows(server=None):
    return [c.to_dict() for c in (CASES if server is None else cases_for(server, True))]


def registry_document():
    return dict(campaign_id=CAMPAIGN_ID, registry_revision=REGISTRY_REVISION, method_revision=METHOD_REVISION,
        source_status=SOURCE_STATUS, source_sha256=dict(SOURCE_SHAS), missing_author_design_assets=list(MISSING_DESIGN_ASSETS),
        run_id_derivation='MD case/reference/seed/profile/horizon; not author CSV byte identity',
        cases=registry_rows(), profiles={k: dict(asdict(v), aligner_multiplier=(
            dict(initial=.1, hold_before_updates=5000, linear_ramp_end_updates=10000, final=1.)
            if k == 'ARW' else dict(constant=1.))) for k, v in PROFILES.items()},
        references={k: dict(asdict(v), owner_server=v.owner_server, tau_R=None, q_ref=None,
                            teacher_checkpoint_sha256=None, q_cache_sha256=None) for k, v in REFERENCES.items()},
        grids={'50000': list(GRID50), '100000': list(GRID100)},
        window=dict(hours=20, admission_cutoff_hours=16, train_finish_hours=18, close_hours=2,
                    t0_utc=None, deadline_utc=None), setup_hours=SETUP_HOURS, calibration_hours=CALIBRATION_HOURS,
        blocks=[b.to_dict() for s in SERVERS for b in blocks_for(s)],
        estimated_core_primary_hours=dict(s3=17.65, s4=13.05, s5=13.05), cross_server_reference_edges=0,
        deferred_auto_release=False, sensors={'GF2': sensor_spec().to_dict()})


def registry_sha256():
    return _object_sha(registry_document())


verify_sources()
if len(CASES) != 22 or len(CORE_CASES) != 18 or len(DEFERRED_CASES) != 4 or len(_BY_RUN) != 22:
    raise ValueError('LOCAL-T registry must have exactly 18 core and 4 deferred distinct cases')
