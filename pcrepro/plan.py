"""Finite local cycle generation, never a finite experiment or shared clock."""
import copy
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN_ID='PANCRAFTER_REPRO_S345_UNLIMITED_20260922_v1'
RECIPE_ID='PC_PAPER_EQ11_C128_D224_LN_MARS_v1'
SOURCE_PLAN='research_log/PANCRAFTER_Reproduction_CasePlan_KR_2026-09-22.md'
SOURCE_PLAN_SHA256='087773ffd59fcfb230198fd76d79a95dbaf96d34d0d1da66294d37f4d2e61eb2'
SERVERS=('s3','s4','s5')
DATASETS=('WV3','QB','GF2','WV2')
SHEET_TABS={s:'PC-Repro-'+s for s in SERVERS}
ORDERS=dict(s3=('WV3','WV2','QB','GF2'),s4=('QB','GF2','WV3','WV2'),s5=('GF2','WV3','WV2','QB'))
SENSORS={name:dict(bands=8 if name in ('WV3','WV2') else 4,max_dn=1023 if name=='GF2' else 2047) for name in DATASETS}
RECIPE=json.loads((ROOT/'pcrepro/recipe.json').read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
RECIPE_SHA256=digest(RECIPE)
GRID=tuple(range(1000,50001,1000))
def verify_sources(root=ROOT):
    if hashlib.sha256((Path(root)/SOURCE_PLAN).read_bytes()).hexdigest()!=SOURCE_PLAN_SHA256:
        raise ValueError('Reproduction source plan changed; declare a new revision')
    return {SOURCE_PLAN:SOURCE_PLAN_SHA256}
def verify_server(server):
    if server not in SERVERS:raise ValueError('PCREPRO is s3/s4/s5 only; s1/s2 are protected')
def cycle_seed(server,cycle):
    verify_server(server)
    if isinstance(cycle,bool) or not isinstance(cycle,int) or cycle<0:raise ValueError('Nonnegative integer cycle required')
    seed=2025 if cycle==0 else 1000000+3*(cycle-1)+(int(server[1])-3)
    if seed>2**32-1:raise ValueError('Seed range exhausted; never wrap')
    return seed
@dataclass(frozen=True)
class Case:
    server:str
    cycle:int
    dataset:str
    @property
    def seed(self):return cycle_seed(self.server,self.cycle)
    @property
    def stage(self):return 'ZERO_SHOT' if self.dataset=='WV2' else 'TRAIN'
    @property
    def train_dataset(self):return 'WV3' if self.dataset=='WV2' else self.dataset
    @property
    def num_bands(self):return SENSORS[self.dataset]['bands']
    @property
    def max_dn(self):return SENSORS[self.dataset]['max_dn']
    @property
    def updates(self):return 0 if self.dataset=='WV2' else 50000
    @property
    def recipe_id(self):return RECIPE_ID
    @property
    def run_id(self):
        name='WV3toWV2' if self.dataset=='WV2' else self.dataset
        return f'PCREPRO_{name}_{self.server}_C{self.cycle:06d}_S{self.seed}_{"EVAL" if self.dataset=="WV2" else "F50K"}_v1'
    @property
    def source_run_id(self):return Case(self.server,self.cycle,'WV3').run_id if self.dataset=='WV2' else None
    def to_dict(self):return dict(asdict(self),campaign_id=CAMPAIGN_ID,recipe_id=RECIPE_ID,
        seed=self.seed,stage=self.stage,train_dataset=self.train_dataset,num_bands=self.num_bands,
        max_dn=self.max_dn,updates=self.updates,run_id=self.run_id,source_run_id=self.source_run_id,
        repeat_kind='SAME_SEED_SERVER_REPEAT' if self.cycle==0 else 'PREDECLARED_DISTINCT_SEED')
    def __post_init__(self):
        cycle_seed(self.server,self.cycle)
        if self.dataset not in DATASETS:raise ValueError('Unknown reproduction dataset')
def cases_for(server,cycle=0):
    verify_server(server)
    return tuple(Case(server,cycle,d) for d in ORDERS[server])
def case_for(value):
    if isinstance(value,Case):return value
    match=re.fullmatch(r'PCREPRO_(WV3toWV2|WV3|QB|GF2)_(s[345])_C(\d{6,})_S(\d+)_(F50K|EVAL)_v1',str(value))
    if not match:raise ValueError('Unregistered reproduction run ID')
    dataset,server,cycle,seed,stage=match.groups()
    c=Case(server,int(cycle),'WV2' if dataset=='WV3toWV2' else dataset)
    if c.run_id!=value:raise ValueError('Run ID is not the predeclared cycle/seed')
    return c
def build_config(case,data_manifest=None,data_sha=None,source_identity=None):
    c=case_for(case);t=RECIPE['training']
    return dict(seed=c.seed,work_dir=f'work_dir/{c.run_id}',trainer='pcrepro',num_iter=c.updates,
        batch_size=t['batch_pairs'],num_workers=t['workers'],num_worker=t['workers'],
        learning_rate=t['learning_rate'],weight_decay=t['weight_decay'],betas=t['betas'],eps=t['eps'],
        num_warmup=t['warmup_updates'],mixed_precision='no',model_args=dict(num_bands=c.num_bands,max_pixel=c.max_dn),
        pcrepro=dict(c.to_dict(),recipe_sha256=RECIPE_SHA256,recipe=copy.deepcopy(RECIPE),
            data_manifest=data_manifest,data_sha=data_sha,source_identity=source_identity))
def validate_config(cfg,require_bound=False):
    f=cfg['pcrepro'];case=case_for(f['run_id'])
    expected=build_config(case,f['data_manifest'],f['data_sha'],f['source_identity'])
    if cfg!=expected:raise ValueError('Immutable PCREPRO recipe/config differs')
    if require_bound and not all(f.get(k) for k in ('data_manifest','data_sha','source_identity')):
        raise ValueError('Verified local data and execution source required')
    return case
