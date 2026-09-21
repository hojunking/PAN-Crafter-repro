"""Author-bundle identities and finite 36-case registry; never creates a clock."""
from dataclasses import dataclass
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_GF2_B20_S345_20260922_v1'
BUNDLE = 'research_log/PANDA_GF2_B20_S345_20H_Bundle_2026-09-22'
SUMS_SHA = '32a752c09c44401df8f53bf09d33bb8a03dca35e7b2ad02d55766f82c416d307'
SERVERS = ('s3', 's4', 's5')
WINDOW_HOURS, ADMISSION_CUTOFF_HOURS, TRAIN_FINISH_HOURS = 20., 16., 18.
METHOD_REVISION = 'GFB20_FT20_RAMP_PANMIX_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root=ROOT):
    folder = Path(root) / BUNDLE
    sums = folder / 'SHA256SUMS.txt'
    if _sha(sums) != SUMS_SHA:
        raise ValueError('GFB20 source SHA256SUMS changed; new revision required')
    result = {BUNDLE + '/SHA256SUMS.txt': SUMS_SHA}
    for line in sums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        path = folder / name
        if path.resolve().is_relative_to(folder.resolve()) is False or _sha(path) != expected:
            raise ValueError('GFB20 author bundle mismatch: ' + name)
        result[BUNDLE + '/' + name] = expected
    return result


SOURCE_SHAS = MappingProxyType(verify_sources())
REGISTRY = json.loads((ROOT / BUNDLE / 'GFB20_DesignRegistry.json').read_text())
PARENT_ASSETS = json.loads((ROOT / BUNDLE / 'GFB20_ParentAssets.json').read_text())
PROFILES = MappingProxyType(REGISTRY['profiles'])
REFERENCES = MappingProxyType(REGISTRY['references'])
PARENTS = MappingProxyType(REGISTRY['parents'])
GRID20 = tuple(REGISTRY['ft']['official_steps'])
GRID100 = tuple(REGISTRY['fresh']['official_steps'])
SHEET_TABS = {s: 'GF2-B20-' + s for s in SERVERS}


def verify_lane(server):
    if server not in SERVERS:
        raise ValueError('GFB20 permits only s3/s4/s5; s1/WV3 and s2/QB are protected')
    return 'GF2'


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class Case:
    values: object

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def server_id(self): return self.server
    @property
    def updates(self): return self.train_updates
    @property
    def seed(self): return self.model_seed or self.stream_seed
    @property
    def width(self): return self.student_width
    @property
    def depth(self): return tuple(int(x) for x in self.student_depth.split(','))
    @property
    def input_layout(self): return self.student_input
    @property
    def reference_id(self): return self.reference_key
    @property
    def role(self): return 'S'
    @property
    def sensor(self): return 'GF2'
    @property
    def is_ft(self): return self.init_mode == 'PARENT_STUDENT_U_AND_A_WEIGHTS'
    @property
    def reservation_hours(self):
        budget = REGISTRY['budgets'][self.server]
        return budget['core_ft_hours_each' if self.is_ft else 'fresh100_hours_each']

    def to_dict(self): return copy.deepcopy(dict(self.values))


CASES = tuple(Case(MappingProxyType(copy.deepcopy(row))) for row in REGISTRY['cases'])
CORE_CASES = tuple(c for c in CASES if c.stage == 'CORE')


def case_for(identifier, root=ROOT):
    if isinstance(identifier, Case):
        canonical = case_for(identifier.case_id)
        if identifier != canonical: raise ValueError('Changed case contract')
        return canonical
    matches = [c for c in CASES if identifier in (c.case_id, c.run_id)]
    if len(matches) != 1: raise ValueError('Unknown GFB20 case: ' + str(identifier))
    return matches[0]


def cases_for(server):
    verify_lane(server)
    return tuple(sorted((c for c in CASES if c.server == server), key=lambda c: c.queue_rank))


@dataclass(frozen=True)
class Block:
    block_id: str
    server_id: str
    case_ids: tuple
    @property
    def cases(self): return tuple(case_for(c) for c in self.case_ids)
    @property
    def run_ids(self): return tuple(c.run_id for c in self.cases)
    @property
    def stage(self): return self.cases[0].stage
    def to_dict(self):
        return dict(block_id=self.block_id, server_id=self.server_id, case_ids=list(self.case_ids),
                    run_ids=list(self.run_ids), stage=self.stage)


def blocks_for(server):
    verify_lane(server)
    queue = json.loads((ROOT / BUNDLE / f'GFB20_Queue_{server}.json').read_text())
    return tuple(Block(b['block_id'], server, tuple(b['case_ids'])) for b in queue['blocks'])


def block_for(identifier):
    matches = [b for s in SERVERS for b in blocks_for(s) if b.block_id == identifier]
    if len(matches) != 1: raise ValueError('Unknown GFB20 block')
    return matches[0]


def grid_steps(updates):
    if isinstance(updates, bool) or updates not in (20000, 100000):
        raise ValueError('Only FT20K and fresh100K are registered')
    return GRID20 if updates == 20000 else GRID100


def diagnostic_steps(updates):
    grid_steps(updates)
    return tuple(REGISTRY['ft' if updates == 20000 else 'fresh']['diagnostic_steps'])


def fullstate_steps(updates): return diagnostic_steps(updates)


def registry_sha256(): return _sha(ROOT / BUNDLE / 'GFB20_DesignRegistry.json')


BINDING_FIELDS = ('dataset_manifest', 'reference_manifest', 'parent_manifest', 'augmentation_manifest',
    'mixed_calibration_manifest', 'window_path', 'runtime_policy_sha256', 'parent_anchor',
    'teacher_checkpoint', 'teacher_sha256', 'tau_R', 'q_ref', 'q_cache', 'q_cache_sha256',
    'profile_decision_sha256')


def build_config(case, selected_profile=None):
    case = case_for(case)
    profile = selected_profile or case.profile
    if case.profile == 'B_SELECTED':
        if profile not in ('B_SELECTED', 'K1', 'E100'): raise ValueError('Unregistered B_SELECTED binding')
    elif profile != case.profile:
        raise ValueError('Cannot change a registered case profile')
    meta = dict(case.to_dict(), case_contract=case.to_dict(), server_id=case.server, role='S',
        sensor='GF2', reference_id=case.reference_key, input_layout='PLH', max_pixel=1023,
        profile=profile, registry_sha256=registry_sha256(), method_revision=METHOD_REVISION,
        campaign_id=CAMPAIGN_ID, candidate_grid=list(grid_steps(case.updates)),
        diagnostic_steps=list(diagnostic_steps(case.updates)), fullstate_steps=list(fullstate_steps(case.updates)),
        runtime_policy_path='gfb20/runtime_policy.json',
        **{k: None for k in BINDING_FIELDS})
    meta['window_path'] = 'work_dir/_gfb20/campaign_window.json'
    return dict(seed=case.seed, work_dir=f'work_dir/{case.run_id}', trainer='gfb20', phase='train',
        num_iter=case.updates, num_warmup=100, batch_size=48, test_batch_size=1, num_worker=4,
        num_bands=4, max_pixel=1023., mixed_precision='no', learning_rate=case.u_peak_lr,
        optimizer='AdamW', weight_decay=.01, betas=[.9,.999], eps=1e-8, lr_scheduler='cosine', log_iter=100,
        train_feeder_args=dict(dataroot=None, crop=False, hflip=True, vflip=True, rot=True, ms_size=16, return_meta=True),
        val_feeder_args=dict(dataroot=None), test_reduced_feeder_args=dict(dataroot=None),
        test_full_feeder_args=dict(dataroot=None),
        model_args=dict(hidden_size=104, depth=[1,2,2], out_channels=4, attn_locations=[],
                        mode_modulation=False, norm='ln', dropout=0.), gfb20=meta)


def validate_config(cfg, require_bound=False):
    field = cfg['gfb20']
    case = case_for(field['case_id'])
    expected = build_config(case, field['profile'])
    expected['work_dir'] = cfg['work_dir']
    if Path(cfg['work_dir']).name != case.run_id: raise ValueError('Run directory differs from registry')
    for key in BINDING_FIELDS: expected['gfb20'][key] = copy.deepcopy(field[key])
    for key in ('train_feeder_args','val_feeder_args','test_reduced_feeder_args','test_full_feeder_args'):
        expected[key]['dataroot'] = cfg[key]['dataroot']
    if cfg != expected: raise ValueError('GFB20 immutable config/recipe differs')
    if require_bound:
        required = ('dataset_manifest','reference_manifest','window_path','runtime_policy_sha256')
        if case.is_ft: required += ('parent_manifest',)
        if field['profile'] in ('PANMIX','NAT_MIXCAL'):
            required += ('augmentation_manifest','mixed_calibration_manifest')
        if any(not field[k] for k in required) or field['profile'] == 'B_SELECTED':
            raise ValueError('Bind verified GFB20 local assets and selected profile before training')
        if not valid_sha(field['runtime_policy_sha256']): raise ValueError('Invalid runtime policy SHA')
        if case.profile == 'B_SELECTED' and not valid_sha(field['profile_decision_sha256']):
            raise ValueError('B_SELECTED requires immutable decision evidence')
    return case


def validate_registry():
    if len(CASES) != 36 or len(CORE_CASES) != 18 or len({c.run_id for c in CASES}) != 36:
        raise ValueError('Expected 36 distinct Students, 18 core, no Teacher training')
    if GRID20 != tuple(range(1000,20001,1000)) or GRID100 != tuple(range(2020,100000,2020))+(100000,):
        raise ValueError('Candidate grid changed (diagnostic50K must not replace50500)')
    for filename, expected in [('GFB20_Cases_All36.csv',CASES),('GFB20_Cases_Core18.csv',CORE_CASES)]:
        with (ROOT/BUNDLE/filename).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(expected): raise ValueError('CSV row count differs')
        for row, case in zip(rows,expected):
            if row != {k:('' if v is None else str(v)) for k,v in case.values.items()}:
                raise ValueError('CSV/JSON case mismatch: '+case.case_id)
    for server in SERVERS:
        flattened = [c for b in blocks_for(server) for c in b.cases]
        if flattened != list(cases_for(server)): raise ValueError('Queue/block case order differs')
        for c in flattened:
            if c.teacher_seed != REFERENCES[c.reference_key]['teacher_seed'] or c.server != REFERENCES[c.reference_key]['server']:
                raise ValueError('Cross-server reference or Teacher seed mismatch')
    return dict(cases=36,core=18,teacher_training=0,source_files=len(SOURCE_SHAS))


validate_registry()
