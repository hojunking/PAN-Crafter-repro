"""Pinned ABLR2 design and deterministic, sensor-local experiment definitions.

Importing this module reads design assets only. It never acquires a lease, starts
training, chooses a result-dependent seed, or modifies any previous campaign.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_ABL_S1WV3_S2QB_ADAPTIVE_LOOP_20260921_v2'
REGISTRY_REVISION = 'ABLR2_EXECUTABLE_v2'
METHOD_REVISION = 'ABLR2_MASKED_CPLUS3_COMPONENT_ROUTING_v1'
BUNDLE = 'research_log/PANDA_ABL_S1WV3_S2QB_AdaptiveRepeat_Bundle_2026-09-21_v2'
SOURCE_PLAN = BUNDLE + '/PANDA_ABL_S1_WV3_S2_QB_AdaptiveRepeat_ExperimentPlan_2026-09-21_v2.md'
SOURCE_CASES = BUNDLE + '/ABLR2_BOOT5_190_TrainingCases_2026-09-21.csv'
SOURCE_CATALOG = BUNDLE + '/ABLR2_ComponentCatalog_2026-09-21.csv'
SOURCE_GRAPH = BUNDLE + '/ABLR2_ComparisonGraph_2026-09-21.csv'
SOURCE_REGISTRY = BUNDLE + '/ABLR2_DesignRegistry_2026-09-21.json'
SOURCE_SHAS = MappingProxyType({
    SOURCE_PLAN: '310fe726fa774e3e98afeec74adc14c854d229bee62379acb5fd6e9eb375e33e',
    SOURCE_CASES: '96d3b407e294f021e13b686c2da882af8989d5344c7d7eccec8a00ddc7f4453a',
    SOURCE_CATALOG: 'cd4189de7bd650aecd3f6b859d2521935e2ddb8b493b66d2089b28a9240e551b',
    SOURCE_GRAPH: '47a2405c1291c1390e0f6d792bec2ed14c4e2c7778c3c5b6987927ce2ad6960d',
    SOURCE_REGISTRY: '38b9e084654f4e6ababbe594fbb476a965d6e1c3cb899b396f2462e51b6fdda9'})
SERVERS = ('s1', 's2')
_SAFE_TOKEN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
LANES = MappingProxyType({'s1': 'WV3', 's2': 'QB'})
SHEET_TABS = MappingProxyType({'s1': 'WV3-s1', 's2': 'QB-s2'})
MAIN_CASES = tuple(f'C{i:02}' for i in range(17))
GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)
BINDING_FIELDS = ('dataset_manifest', 'sensor_spec', 'sensor_spec_path', 'reference_manifest',
    'teacher_checkpoint', 'teacher_sha256', 'tau_R', 'q_ref', 'q_cache', 'q_cache_sha256',
    's_bar', 's_bar_sha256', 'runtime_policy_sha256')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verify_sources(root=ROOT):
    for path, expected in SOURCE_SHAS.items():
        if hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed ABLR2 source requires an explicit revision: ' + path)
    return dict(SOURCE_SHAS)


def _rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class SensorSpec:
    sensor: str
    band_order: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.sensor not in LANES.values():
            raise ValueError('ABLR2 supports WV3/s1 and QB/s2 only')
        if self.band_order is not None:
            object.__setattr__(self, 'band_order', tuple(self.band_order))
            if len(self.band_order) != self.num_bands or len(set(self.band_order)) != self.num_bands:
                raise ValueError('Band order must match the sensor and be unique')

    @property
    def num_bands(self): return 8 if self.sensor == 'WV3' else 4
    @property
    def bands(self): return self.num_bands
    @property
    def max_dn(self): return 2047
    @property
    def max_pixel(self): return self.max_dn
    @property
    def inverse_scale(self): return self.max_dn / 2
    @property
    def mtf_sensor(self): return self.sensor
    @property
    def rr_q(self): return 'Q8' if self.sensor == 'WV3' else 'Q4'
    @property
    def nominal_train_n(self): return 9714 if self.sensor == 'WV3' else 17139
    @property
    def nominal_val_n(self): return 1080 if self.sensor == 'WV3' else None
    @property
    def hqnr_threshold(self): return None
    @property
    def ergas_goal(self): return None
    @property
    def ergas_strong_goal(self): return None

    def to_dict(self):
        return dict(sensor=self.sensor, num_bands=self.num_bands, max_dn=self.max_dn,
            inverse_scale=self.inverse_scale, mtf_sensor=self.sensor,
            band_order=list(self.band_order) if self.band_order is not None else None,
            rr_q=self.rr_q, nominal_train_n=self.nominal_train_n, nominal_val_n=self.nominal_val_n,
            hqnr_threshold=None, ergas_goal=None, ergas_strong_goal=None)

    @classmethod
    def from_dict(cls, value):
        spec = cls(value['sensor'], value.get('band_order'))
        if value != spec.to_dict():
            raise ValueError('Changed ABLR2 sensor constants')
        return spec


def sensor_spec(value):
    if isinstance(value, SensorSpec): return value
    if isinstance(value, dict): return SensorSpec.from_dict(value)
    return SensorSpec(value)


def verify_lane(server, sensor=None):
    if server not in LANES or (sensor is not None and LANES[server] != sensor):
        raise ValueError('Only s1/WV3 and s2/QB may enter ABLR2; s3-s5 are protected')
    return LANES[server]


@dataclass(frozen=True)
class Recipe:
    alpha: float = 1.
    beta: float = .1
    lambda_edge: float = .002
    lambda_con: float = 1e-4
    student_a_schedule: str = 'BASE'


RECIPES = MappingProxyType(dict(R00=Recipe(), R01=Recipe(beta=.2), R02=Recipe(lambda_edge=.001),
    R03=Recipe(alpha=.1), R04=Recipe(lambda_con=3e-4),
    R05=Recipe(student_a_schedule='BASE_TIMES_1over3_AFTER_24240')))


def _catalog_row(row):
    row = dict(row)
    for key in ('mask_L', 'mask_H', 'updates'):
        row[key] = int(row[key])
    for key in ('alpha', 'beta', 'lambda_edge', 'u_peak_lr', 'a_peak_lr'):
        row[key] = float(row[key])
    for key in ('soft_trust', 'soft_advantage', 'teacher_predictions_used', 'teacher_q_used',
                'deployment_teacher_required', 'teacher_A_clone_used', 'default_pilot_enabled'):
        if row[key] not in ('True', 'False'): raise ValueError('Invalid catalog boolean')
        row[key] = row[key] == 'True'
    return row


verify_sources()
COMPONENTS = MappingProxyType({r['case_id']: MappingProxyType(_catalog_row(r)) for r in _rows(SOURCE_CATALOG)})
COMPARISON_GRAPH = tuple(_rows(SOURCE_GRAPH))
GRAPH = MappingProxyType({r['relation_id']: MappingProxyType(r) for r in COMPARISON_GRAPH})


def component_config(case_id, recipe_id='R00'):
    """Apply a registered common recipe before all case deletion overrides."""
    if recipe_id not in RECIPES or case_id not in MAIN_CASES:
        raise ValueError('Only the registered R00-R05 bank and C00-C16 are executable')
    result = dict(COMPONENTS[case_id])
    recipe = RECIPES[recipe_id]
    for field in ('alpha', 'beta', 'lambda_edge'):
        # A zero in the component catalogue is a removal, never a free coefficient.
        result[field] = getattr(recipe, field) if result[field] != 0 else 0.
    result['student_a_schedule'] = recipe.student_a_schedule
    result['a_peak_lr'] = 0. if result['aligner'] in ('IDENTITY', 'CLONE_FROZEN') else 3e-6
    result['teacher_kind'] = ('TC3' if recipe_id == 'R04' else 'TPLUS') if result['teacher'] == 'TPLUS' else result['teacher']
    result['requires_teacher'] = result['teacher'] != 'NONE'
    result['requires_calibration'] = bool(result['teacher_q_used'] or result['hard_mode'] != 'PLAIN'
        or (result['soft_mode'] == 'SELECTIVE' and result['soft_trust']))
    result['requires_tau'] = result['hard_mode'] != 'PLAIN' or (result['soft_mode'] == 'SELECTIVE' and result['soft_trust'])
    result['requires_q'] = result['teacher_q_used']
    return result


@dataclass(frozen=True)
class Case:
    case_id: str
    run_id: str
    server_id: str
    sensor: str
    role: str
    phase: str
    recipe_id: str
    recipe_revision: str
    wave: str
    sweep_id: str
    panel_id: str
    teacher_seed: int | None
    student_seed: int | None
    teacher_kind: str
    reference_id: str | None
    teacher_run_id: str | None
    queue_rank: int
    paired_init_group: str
    status: str = 'DEFINED_NOT_LAUNCHED'

    @property
    def sweep(self): return self.sweep_id
    @property
    def seed(self): return self.teacher_seed if self.role == 'T' else self.student_seed
    @property
    def width(self): return 112 if self.role == 'T' else 104
    @property
    def depth(self): return (1, 2, 3) if self.role == 'T' else (1, 2, 2)
    @property
    def updates(self): return 50000
    @property
    def num_bands(self): return sensor_spec(self.sensor).num_bands
    @property
    def bands(self): return self.num_bands
    @property
    def max_dn(self): return 2047
    @property
    def component(self): return component_config(self.case_id, self.recipe_id) if self.role == 'S' else None
    @property
    def effective_flags(self): return self.component
    @property
    def input_layout(self): return 'P0' if self.role == 'T' else self.component['layout']
    @property
    def stored_input_channels(self): return self.num_bands + (1 if self.role == 'T' else 3)
    @property
    def requires_teacher(self): return self.role == 'S' and self.component['requires_teacher']
    @property
    def requires_calibration(self): return self.role == 'S' and self.component['requires_calibration']
    @property
    def teacher_owner(self): return self.server_id if self.reference_id else None
    @property
    def teacher_updates(self): return 50000 if self.reference_id else None
    @property
    def sheet_tab(self): return SHEET_TABS[self.server_id]
    @property
    def alpha(self): return self.component['alpha'] if self.role == 'S' else 0.
    @property
    def beta(self): return self.component['beta'] if self.role == 'S' else 0.
    @property
    def lambda_edge(self): return self.component['lambda_edge'] if self.role == 'S' else 0.
    @property
    def lambda_con(self): return (RECIPES[self.recipe_id].lambda_con if self.case_id == 'TPLUS' else 0.) if self.role == 'T' else 0.
    @property
    def aligner_lr(self): return 1e-5 if self.role == 'T' else self.component['a_peak_lr']

    def to_dict(self):
        return dict(asdict(self), sweep=self.sweep, seed=self.seed, width=self.width, depth=list(self.depth),
            updates=self.updates, num_bands=self.num_bands, max_dn=self.max_dn,
            input_layout=self.input_layout, stored_input_channels=self.stored_input_channels,
            requires_teacher=self.requires_teacher, requires_calibration=self.requires_calibration,
            component=self.component, alpha=self.alpha, beta=self.beta, lambda_edge=self.lambda_edge,
            lambda_con=self.lambda_con, aligner_lr=self.aligner_lr, teacher_owner=self.teacher_owner,
            teacher_updates=self.teacher_updates, sheet_tab=self.sheet_tab, campaign_id=CAMPAIGN_ID)


def _boot_cases():
    rows = _rows(SOURCE_CASES)
    teacher_runs = {r['reference_id']: r['run_id'] for r in rows if r['role'] == 'T'}
    out = []
    for row in rows:
        ref = row['reference_id'] or None
        out.append(Case(row['case_id'], row['run_id'], row['server'], row['sensor'], row['role'],
            row['phase'], 'R00', row['recipe_id'], 'BOOT5', row['sweep_id'], row['panel_id'],
            int(row['teacher_seed']) if row['teacher_seed'] else None,
            int(row['student_seed']) if row['student_seed'] else None, row['teacher_kind'], ref,
            teacher_runs.get(ref), int(row['queue_rank']), row['seed_group']))
    return tuple(out)


CASES = BOOT_CASES = _boot_cases()
_BY_RUN = MappingProxyType({c.run_id: c for c in CASES})


def case_for(run_id, root=ROOT):
    if isinstance(run_id, Case): return validate_case(run_id)
    if run_id in _BY_RUN: return _BY_RUN[run_id]
    for server, sensor in LANES.items():
        path = Path(root) / f'work_dir/ablr2/{sensor}/{server}/cases.json'
        if not path.is_file(): continue
        document = json.loads(path.read_text())
        values = document.get('cases', document) if isinstance(document, dict) else document
        values = values.values() if isinstance(values, dict) else values
        for value in values:
            if value.get('run_id') != run_id: continue
            contract = value.get('case_contract', value)
            case = Case(**{field.name: contract[field.name] for field in fields(Case)})
            if case.server_id != server: raise ValueError('Run persisted in the wrong lane registry')
            return validate_case(case)
    raise ValueError('Run ID is not registered in BOOT5 or the local append-only task registry')


def validate_case(case):
    if not isinstance(case, Case): raise ValueError('A registered Case contract is required')
    verify_lane(case.server_id, case.sensor)
    if case.recipe_id not in RECIPES or case.role not in ('T', 'S'):
        raise ValueError('Unregistered recipe or role')
    if case.phase not in ('BOOT5', 'REFRESH5', 'RECHECK5', 'FIT_ROUND', 'VERIFY5'):
        raise ValueError('Unregistered campaign phase')
    for value in (case.run_id, case.recipe_revision, case.wave, case.sweep_id, case.panel_id,
                  case.paired_init_group, case.reference_id, case.teacher_run_id):
        if value is not None and (not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value)):
            raise ValueError('Run, wave and reference identifiers must be plain safe tokens, never paths')
    if case.recipe_revision != f'r{int(case.recipe_id[1:]):03}':
        raise ValueError('Recipe revision mapping is immutable: R00/r000 through R05/r005')
    if case.status != 'DEFINED_NOT_LAUNCHED':
        raise ValueError('Immutable case definitions cannot be rewritten with runtime outcomes')
    if not case.recipe_revision or not case.wave or not case.sweep_id or not case.panel_id or not case.paired_init_group:
        raise ValueError('Complete pre-registered wave, revision, panel and pairing identities are required')
    for seed in (case.teacher_seed, case.student_seed):
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31 - 1):
            raise ValueError('Invalid seed')
    if case.role == 'T':
        if case.case_id not in ('TPLUS', 'TZERO') or case.student_seed is not None or case.teacher_seed is None:
            raise ValueError('Invalid Teacher contract')
        expected_kind = 'TZERO' if case.case_id == 'TZERO' else ('TC3' if case.recipe_id == 'R04' else 'TPLUS')
        if case.teacher_kind != expected_kind or case.teacher_run_id != case.run_id or not case.reference_id:
            raise ValueError('Teacher endpoint reference identity differs')
    else:
        flags = case.component
        if case.student_seed is None or case.teacher_kind != flags['teacher_kind']:
            raise ValueError('Student component dependency differs')
        if case.requires_teacher:
            if case.teacher_seed is None or not case.reference_id or not case.teacher_run_id:
                raise ValueError('Teacher-dependent component lacks its exact local endpoint')
            if not case.teacher_run_id.startswith(f'ABLR2_{case.sensor}_{case.server_id}_'):
                raise ValueError('Cross-lane Teacher dependency is forbidden')
        elif any(x is not None for x in (case.teacher_seed, case.reference_id, case.teacher_run_id)):
            raise ValueError('Teacher-free cases cannot hold a hidden Teacher dependency')
    if not case.run_id.startswith(f'ABLR2_{case.sensor}_{case.server_id}_') or not case.run_id.endswith('_FRESH50'):
        raise ValueError('Run ID lies outside the isolated ABLR2 lane')
    if case.phase == 'BOOT5' and _BY_RUN.get(case.run_id) != case:
        raise ValueError('BOOT5 is exactly the immutable author-supplied CSV')
    if case.phase != 'BOOT5':
        seed_label = 'TS' if case.role == 'T' else 'SS'
        expected_run = (f'ABLR2_{case.sensor}_{case.server_id}_{case.recipe_revision}_{case.phase}_'
            f'{case.wave}_{case.sweep_id}_{case.case_id}_{seed_label}{case.seed}_FRESH50')
        if case.run_id != expected_run:
            raise ValueError('Dynamic run ID and immutable case contract disagree')
    if case.reference_id and (f'_{case.sensor}_{case.server_id}_' not in case.reference_id
            or f'_TS{case.teacher_seed}_{case.teacher_kind}_END50000' not in case.reference_id):
        raise ValueError('Reference is not the exact local seed/kind/endpoint')
    return case


def cases_for(server):
    verify_lane(server)
    return tuple(c for c in CASES if c.server_id == server)


def student_order(sweep_number):
    if not isinstance(sweep_number, int) or sweep_number < 1: raise ValueError('Positive sweep ordinal required')
    offset = 4 * (sweep_number - 1) % 17
    order = MAIN_CASES[offset:] + MAIN_CASES[:offset]
    return order[::-1] if sweep_number % 2 == 0 else order


def grid_steps(updates=50000):
    if updates != 50000 or isinstance(updates, bool): raise ValueError('ABLR2 core is fresh50K only')
    return GRID_STEPS


def diagnostic_steps(role, updates=50000):
    grid_steps(updates)
    if role not in ('T', 'S'): raise ValueError('Unknown role')
    return (0, 10000, 25000, 50000) if role == 'T' else (0, 1000, 5000, 10000, 25000, 50000)


def build_config(case):
    case = validate_case(case_for(case) if isinstance(case, str) else case)
    lane = f'work_dir/ablr2/{case.sensor}/{case.server_id}'
    meta = dict(case.to_dict(), case_contract=asdict(case), source_plan=SOURCE_PLAN,
        registry_revision=REGISTRY_REVISION, method_revision=METHOD_REVISION,
        sensor_spec=sensor_spec(case.sensor).to_dict(), sensor_spec_path=None,
        dataset_manifest=None, reference_manifest=None, teacher_checkpoint=None, teacher_sha256=None,
        tau_R=None, q_ref=None, q_cache=None, q_cache_sha256=None, s_bar=None, s_bar_sha256=None,
        runtime_policy_path='ablr2/runtime_policy.json', runtime_policy_sha256=None,
        lease_path=lane + '/lease.json', stop_path=lane + '/control.json',
        candidate_grid=list(GRID_STEPS), diagnostic_steps=list(diagnostic_steps(case.role)),
        fullstate_steps=list(sorted(set(diagnostic_steps(case.role)) | {24240})),
        teacher_ref_step=50000 if case.reference_id else None,
        consistency_weight=case.lambda_con, offset_weight=case.lambda_con,
        init_policy='FH12_named_tensor_fresh_Cplus1_zero_extra_LH_v1',
        checkpoint_rule='RR_VALIDATION_ARGMIN_ERGAS_THEN_LOWER_STEP',
        secondary_checkpoint='EXACT50K', best_hqnr_auxiliary=True, test_aware=True,
        trainable_a=(case.role == 'T' or case.component['aligner'] in ('FRESH_TRAINABLE', 'CLONE_TRAINABLE')),
        student_a_schedule=RECIPES[case.recipe_id].student_a_schedule if case.role == 'S' else 'BASE',
        a_lr_switch_completed_updates=24240 if case.role == 'S' and case.recipe_id == 'R05' else None,
        a_lr_after_multiplier=1 / 3 if case.role == 'S' and case.recipe_id == 'R05' else 1.,
        offset_every=2, offset_radius=2. if case.role == 'T' else 0., view_margin_hr=4,
        corruption_seed=case.seed + 100000, calibration=dict(n_patches=3072, seed=1234,
            probe='AXIS16', views='FIXED_HV_ROT4', batch_size=16, subset_scope='SENSOR_SHARED_TRAIN_IDS'),
        lp=dict(sigma=1.98, kernel_size=41, padding='replicate', decimation='2::4,2::4',
            generation_dtype='float64', cache_dtype='float32'))
    return dict(seed=case.seed, work_dir=lane + '/runs/' + case.run_id, trainer='ablr2', phase='train',
        num_iter=50000, num_warmup=100, batch_size=48, test_batch_size=1, num_worker=4,
        num_bands=case.num_bands, max_pixel=2047., mixed_precision='no', learning_rate=1e-4,
        optimizer='AdamW', weight_decay=.01, betas=[.9, .999], eps=1e-8, lr_scheduler='cosine', log_iter=100,
        train_feeder_args=dict(dataroot=None, crop=False, hflip=True, vflip=True, rot=True, ms_size=16, return_meta=True),
        val_feeder_args=dict(dataroot=None), test_reduced_feeder_args=dict(dataroot=None), test_full_feeder_args=dict(dataroot=None),
        model_args=dict(hidden_size=case.width, depth=list(case.depth), out_channels=case.num_bands,
            attn_locations=[], mode_modulation=False, norm='ln', dropout=0.), ablr2=meta)


def case_from_config(cfg):
    try: case = validate_case(Case(**cfg['ablr2']['case_contract']))
    except (KeyError, TypeError) as exc: raise ValueError('Complete ABLR2 case_contract required') from exc
    if cfg.get('trainer') != 'ablr2' or any(cfg['ablr2'].get(k) != v for k, v in case.to_dict().items()):
        raise ValueError('Immutable registered case metadata differs')
    return case


def validate_config(cfg, *, require_bound=False):
    case = case_from_config(cfg)
    expected = build_config(case)
    actual_dir = Path(cfg.get('work_dir', ''))
    if ('..' in actual_dir.parts or actual_dir.parts[-6:] != Path(expected['work_dir']).parts[-6:]
            or (not actual_dir.is_absolute() and actual_dir != Path(expected['work_dir']))):
        raise ValueError('Work directory must remain inside its own ABLR2 lane')
    expected['work_dir'] = cfg['work_dir']
    try:
        for key in BINDING_FIELDS: expected['ablr2'][key] = copy.deepcopy(cfg['ablr2'][key])
        for key in ('train_feeder_args', 'val_feeder_args', 'test_reduced_feeder_args', 'test_full_feeder_args'):
            expected[key]['dataroot'] = cfg[key]['dataroot']
    except KeyError as exc: raise ValueError('Missing runtime binding') from exc
    if cfg != expected: raise ValueError('Config differs from the registered numerical recipe')
    if sensor_spec(cfg['ablr2']['sensor_spec']).sensor != case.sensor: raise ValueError('Cross-sensor binding')
    if not case.requires_teacher and case.role == 'S':
        for key in ('reference_manifest', 'teacher_checkpoint', 'teacher_sha256', 'tau_R', 'q_ref', 'q_cache', 'q_cache_sha256', 's_bar', 's_bar_sha256'):
            if cfg['ablr2'][key] is not None: raise ValueError('Teacher-free run contains a hidden reference binding')
    if not case.requires_calibration and case.role == 'S':
        if any(cfg['ablr2'][k] is not None for k in ('tau_R', 'q_ref', 'q_cache', 'q_cache_sha256', 's_bar', 's_bar_sha256')):
            raise ValueError('Clone-only / uniform-KD component cannot consume calibration')
    if require_bound and (not cfg['ablr2']['dataset_manifest'] or not valid_sha(cfg['ablr2']['runtime_policy_sha256'])):
        raise ValueError('Verified dataset and runtime policy binding required')
    return case


def execution_signature(cfg, *, source_sha256, data_sha256, reference_identity=None):
    """Canonical numerical identity for explicitly receipted screen reuse only.

    A logical FIT job is not a new independent observation when the same seed,
    source/data/reference, effective operations, optimizer and selection rule
    already ran. The caller supplies verified reference provenance, not a name
    guessed from recipe strings. New REFRESH/VERIFY seeds still run in full.
    """
    case = validate_config(cfg)
    if not valid_sha(source_sha256) or not valid_sha(data_sha256):
        raise ValueError('Verified numerical source/data SHA256 required for equivalence')
    if case.requires_teacher != (reference_identity is not None):
        raise ValueError('Equivalence must preserve exact consumed reference provenance')
    numeric_flags = None
    if case.role == 'S':
        component = case.component
        keys = ('mask_L', 'mask_H', 'aligner', 'hard_mode', 'alpha', 'soft_mode', 'beta',
                'soft_trust', 'soft_advantage', 'lambda_edge', 'q_edge', 'q_aligner',
                'teacher_predictions_used', 'teacher_q_used', 'teacher_A_clone_used')
        numeric_flags = {key: component[key] for key in keys}
        if numeric_flags['soft_mode'] == 'OFF':
            numeric_flags['soft_trust'] = numeric_flags['soft_advantage'] = False
    payload = dict(schema='ABLR2_NUMERICAL_EXECUTION_ID_v1', sensor=case.sensor, server=case.server_id,
        source_sha256=source_sha256, data_sha256=data_sha256, reference_identity=reference_identity,
        role=case.role, seed=case.seed, component=numeric_flags, model=cfg['model_args'],
        input_channels=case.stored_input_channels, train_batch=cfg['batch_size'], updates=cfg['num_iter'],
        warmup=cfg['num_warmup'], optimizer=cfg['optimizer'], learning_rate=cfg['learning_rate'],
        betas=cfg['betas'], eps=cfg['eps'], weight_decay=cfg['weight_decay'],
        precision=cfg['mixed_precision'], runtime_policy_sha256=cfg['ablr2']['runtime_policy_sha256'],
        train_transforms={key: value for key, value in cfg['train_feeder_args'].items() if key != 'dataroot'},
        worker_count=cfg['num_worker'], aligner_lr=case.aligner_lr,
        student_a_schedule=cfg['ablr2']['student_a_schedule'] if case.role == 'S' and case.aligner_lr else None,
        consistency_weight=case.lambda_con, offset_every=2 if case.lambda_con else None,
        offset_radius=2. if case.lambda_con else None, corruption_seed=cfg['ablr2']['corruption_seed'] if case.lambda_con else None,
        candidate_grid=cfg['ablr2']['candidate_grid'], checkpoint_rule=cfg['ablr2']['checkpoint_rule'],
        init_policy=cfg['ablr2']['init_policy'], lp=cfg['ablr2']['lp'])
    return dict(sha256=object_sha(payload), numerical_identity=payload)


def derive_seed(sensor, phase, recipe_revision, wave, sweep, role, stream, master_seed, ledger):
    """Allocate once, independent of metrics; identical keys are idempotent.

    The caller persists the appended entries atomically before admitting work.
    No component name participates in common initialization/data stream keys.
    """
    sensor_spec(sensor)
    key = f'ABLR2|{sensor}|{phase}|{recipe_revision}|{wave}|{sweep}|{role}|{stream}|{master_seed}'
    existing = [row for row in ledger if row.get('key') == key]
    if len(existing) > 1: raise ValueError('Duplicate seed ledger key')
    if existing:
        record = existing[0]
        counter = record.get('collision_counter')
        if not isinstance(counter, int) or counter < 0: raise ValueError('Invalid seed collision receipt')
        payload = key if counter == 0 else key + f'|collision={counter}'
        expected = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**31 - 1)
        if record.get('seed') != expected: raise ValueError('Changed performance-independent seed ledger')
        return expected
    used = {row['seed'] for row in ledger}
    counter = 0
    while True:
        payload = key if counter == 0 else key + f'|collision={counter}'
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**31 - 1)
        if seed not in used: break
        counter += 1
    ledger.append(dict(key=key, seed=seed, collision_counter=counter, sensor=sensor, phase=phase,
        recipe_revision=recipe_revision, wave=wave, sweep=sweep, role=role, stream=stream,
        master_seed=master_seed, selection='PERFORMANCE_INDEPENDENT_SHA256'))
    return seed


def _dynamic_teacher(server, phase, recipe, revision, wave, sweep, seed, kind, rank, init_group):
    sensor = verify_lane(server)
    effective_kind = 'TC3' if kind == 'TPLUS' and recipe == 'R04' else kind
    run = f'ABLR2_{sensor}_{server}_{revision}_{phase}_{wave}_{sweep}_{kind}_TS{seed}_FRESH50'
    ref = f'{CAMPAIGN_ID}_{sensor}_{server}_{revision}_TS{seed}_{effective_kind}_END50000'
    return Case(kind, run, server, sensor, 'T', phase, recipe, revision, wave, sweep,
        f'{sensor}_{server}_{revision}_{phase}_{wave}_{sweep}', seed, None, effective_kind, ref, run, rank, init_group)


def _dynamic_student(server, phase, recipe, revision, wave, sweep, seed, component, rank, init_group, teachers):
    sensor = verify_lane(server)
    flags = component_config(component, recipe)
    teacher = teachers[flags['teacher']] if flags['requires_teacher'] else None
    if teacher is not None:
        validate_case(teacher)
        if teacher.server_id != server or teacher.teacher_kind != flags['teacher_kind']:
            raise ValueError('Reference pool is not local and compatible with the current recipe')
    run = f'ABLR2_{sensor}_{server}_{revision}_{phase}_{wave}_{sweep}_{component}_SS{seed}_FRESH50'
    return Case(component, run, server, sensor, 'S', phase, recipe, revision, wave, sweep,
        f'{sensor}_{server}_{revision}_{phase}_{wave}_{sweep}', teacher.seed if teacher else None, seed,
        flags['teacher_kind'], teacher.reference_id if teacher else None, teacher.run_id if teacher else None,
        rank, init_group)


def full_wave(server, phase, recipe_id, recipe_revision, wave, master_seed, seed_ledger):
    """Register all five full sweeps (95 tasks) before observing their results."""
    sensor = verify_lane(server)
    if phase == 'BOOT5':
        if (recipe_id, recipe_revision, wave) != ('R00', 'r000', 'BOOT5'):
            raise ValueError('BOOT5 identity is author supplied')
        return cases_for(server)
    if phase not in ('REFRESH5', 'VERIFY5') or recipe_id not in RECIPES:
        raise ValueError('Only REFRESH5/VERIFY5 create new balanced Teacher panels')
    result = []
    for number in range(1, 6):
        sweep = f'P{number:02}'
        ts = derive_seed(sensor, phase, recipe_revision, wave, sweep, 'T', 'INIT_DATA', master_seed, seed_ledger)
        ss = derive_seed(sensor, phase, recipe_revision, wave, sweep, 'S', 'INIT_DATA', master_seed, seed_ledger)
        teachers = {kind: _dynamic_teacher(server, phase, recipe_id, recipe_revision, wave, sweep, ts,
            kind, 10 if kind == ('TPLUS' if number % 2 else 'TZERO') else 20,
            f'{sensor}_{phase}_{wave}_{sweep}_TS{ts}_COMMON_T_INIT') for kind in ('TPLUS', 'TZERO')}
        result.extend(sorted(teachers.values(), key=lambda c: c.queue_rank))
        result.extend(_dynamic_student(server, phase, recipe_id, recipe_revision, wave, sweep, ss, comp,
            101 + rank, f'{sensor}_{phase}_{wave}_{sweep}_SS{ss}_COMMON_U_INIT', teachers)
            for rank, comp in enumerate(student_order(number)))
    return tuple(validate_case(c) for c in result)


def recheck_cases(server, relation_id, recipe_id, recipe_revision, wave, teacher_panels, master_seed, seed_ledger):
    """Five complete parent/child pairs plus FULL, with the ordered latest panel."""
    sensor = verify_lane(server)
    if relation_id not in GRAPH or len(teacher_panels) != 5: raise ValueError('Five local Teacher panels and a graph relation are required')
    graph = GRAPH[relation_id]
    components = tuple(dict.fromkeys((graph['parent'], graph['child'], 'C07')))
    result = []
    for number, teachers in enumerate(teacher_panels, 1):
        sweep = f'P{number:02}'
        # Relation is part of wave identity, not a performance-selected seed salt.
        ss = derive_seed(sensor, 'RECHECK5', recipe_revision, wave, sweep, 'S', 'INIT_DATA', master_seed, seed_ledger)
        for rank, component in enumerate(components, 101):
            result.append(_dynamic_student(server, 'RECHECK5', recipe_id, recipe_revision, wave, sweep,
                ss, component, rank, f'{sensor}_RECHECK5_{wave}_{sweep}_SS{ss}_COMMON_U_INIT', teachers))
    return tuple(validate_case(c) for c in result)


def screen_cases(server, recipe_ids, recipe_revisions, wave, blocks, relation_id=None):
    """Two paired fixed-P01/P02 screens; candidates differ only by registered recipe.

    ``blocks`` are two dicts containing teacher_seed, student_seed and ``teachers``
    (the fixed DEV TPLUS/TZERO pool). Each recipe gets FULL and the same requested
    relation controls. R04 receives a fresh same-seed TC3; its unchanged TZERO is
    reused only as a numerical-zero-consistency reference, never as TC3.
    Controller validates the fixed DEV pool/source identity and reuses equivalent
    existing jobs without counting them as additional independent observations.
    """
    sensor = verify_lane(server)
    if len(blocks) != 2 or not 1 <= len(recipe_ids) <= 3 or len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError('One incumbent and at most two unique candidates on exactly two DEV blocks')
    if any(recipe not in RECIPES for recipe in recipe_ids): raise ValueError('Unregistered recipe')
    graph = GRAPH[relation_id] if relation_id is not None else None
    components = tuple(dict.fromkeys(('C07',) + ((graph['parent'], graph['child']) if graph else ())))
    result = []
    for number, block in enumerate(blocks, 1):
        sweep = f'P{number:02}'
        expected_ts = (781000 if server == 's1' else 881000) + number
        expected_ss = (791000 if server == 's1' else 891000) + number
        if (block['teacher_seed'], block['student_seed']) != (expected_ts, expected_ss):
            raise ValueError('Fitting uses the fixed author BOOT5 P01/P02 seed blocks')
        for recipe in recipe_ids:
            revision = recipe_revisions[recipe]
            teachers = dict(block['teachers'])
            if recipe == 'R04':
                teacher = _dynamic_teacher(server, 'FIT_ROUND', recipe, revision, wave, sweep,
                    expected_ts, 'TPLUS', 10, f'{sensor}_P{number:02}_TS{expected_ts}_COMMON_T_INIT')
                result.append(teacher)
                teachers['TPLUS'] = teacher
            for rank, component in enumerate(components, 101):
                result.append(_dynamic_student(server, 'FIT_ROUND', recipe, revision, wave, sweep,
                    expected_ss, component, rank,
                    f'{sensor}_P{number:02}_SS{expected_ss}_COMMON_U_INIT', teachers))
    return tuple(validate_case(c) for c in result)


def registry_document():
    return dict(campaign_id=CAMPAIGN_ID, registry_revision=REGISTRY_REVISION, method_revision=METHOD_REVISION,
        source_sha256=dict(SOURCE_SHAS), source_status='AUTHOR_BUNDLE_VERIFIED', lanes=dict(LANES),
        cases=[c.to_dict() for c in CASES], components={k: dict(v) for k, v in COMPONENTS.items()},
        comparison_graph=list(COMPARISON_GRAPH), recipes={k: asdict(v) for k, v in RECIPES.items()},
        candidate_grid=list(GRID_STEPS), max_campaign_cycles=None, lease_hours=72,
        require_initial_operator_lease=True, optional_auto_release=False,
        thresholds=None, test_aware=True, primary_checkpoint='RR_VALIDATION_ARGMIN_ERGAS_THEN_LOWER_STEP',
        recipe_revision_mapping={'r000': 'R00'}, statistical_significance_claim=False)


def registry_sha256(): return object_sha(registry_document())


if len(CASES) != 190 or len(_BY_RUN) != 190 or len(GRAPH) != 17:
    raise ValueError('Author bundle must contain exactly 190 unique BOOT5 jobs and 17 relations')
for _case in CASES: validate_case(_case)
