"""Finite FH20R1 registry: supplied Cases CSV plus MD Appendix A block order."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy

from fh12.common import object_sha
from fh12.plan import build_config as legacy_config, teacher_for as legacy_teacher, cases_for as legacy_cases

CAMPAIGN_ID='WV3_FH20R1_20260919_v1'
REGISTRY_REVISION='FH20R1_REGISTRY_20260919_v1'
METHOD_REVISION='FH12_SYNC_FREQ_NATIVE_TEACHER_v1'
SOURCE_PLAN='research_log/PAN_FH20R1_AllServers_20Hplus_ExperimentPlan_2026-09-19.md'
SOURCE_CASES='research_log/PAN_FH20R1_Cases_2026-09-19.csv'
SERVERS=('s1','s2','s3','s4','s5')
MIN_EFFECTIVE_HOURS=20.0
GRID_STEPS=tuple(range(1010,50000,1010))+(50000,)
FULLSTATE_STEPS=(10000,24240,50000)
S1_SUPPORT='S1_A_SUPPORT'
S1_FALLBACK='S1_NO_A_SUPPORT_OR_INCONCLUSIVE'
S2_SUPPORT='S2_N2PL_REFERENCE_READY'
S2_FALLBACK='S2_DONOR_OR_REFERENCE_UNAVAILABLE'
PREFIXES=('04ef8e756ae8b229','d2832f5a2d40536a','8a971c857a1f386f','9b6fd8ccc0e3e567','dff2b0c1b4e00422')
REFERENCE_SHAS=(
    '04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519',
    'd2832f5a2d40536a859369c011063fe65794c0b7159c93ab23e9976bcb98a1b2',
    '8a971c857a1f386f9a2cf130f8aaf674760d7fa0604d411c92ae7a35fffdb352',
    '9b6fd8ccc0e3e567f62bacd53d56bb6b2017f23b4c3aac75bdcca0d1ae36d260',
    'dff2b0c1b4e00422ce184d9e7590ba78e88d750bb42674997423542ad20916c2')
REFERENCE_ASSETS={f'F{i}':dict(alias=f'F{i}',server_id=f's{i}',server=f's{i}',
    input_layout=legacy_teacher(f's{i}').input_layout,teacher_layout=legacy_teacher(f's{i}').input_layout,
    teacher_seed=legacy_teacher(f's{i}').seed,teacher_run_id=legacy_teacher(f's{i}').run_id,
    expected_sha_prefix=PREFIXES[i-1],exact50k_sha_prefix=PREFIXES[i-1],
    expected_sha256=REFERENCE_SHAS[i-1],
    origin_manifest=f'work_dir/_fh12/s{i}/references/{legacy_teacher(f"s{i}").run_id}/reference_manifest.json',
    checkpoint=f'work_dir/{legacy_teacher(f"s{i}").run_id}/candidates/50000',
    expected_sha_source='supplied_Cases_CSV_whole_checkpoint_SHA256')
    for i in range(1,6)}
N2_TEACHER_RUN='FH20R1_S2_T_PL_W112_D123_WV3_TS71001_N2INIT50_v1'


@dataclass(frozen=True)
class Case:
    server_id:str
    role:str
    tier:str
    block_id:str
    input_layout:str
    width:int
    depth:tuple
    teacher_alias:str
    teacher_seed:int
    student_seed:int|None
    profile:str='BASE'

    @property
    def seed(self): return self.teacher_seed if self.role=='T' else self.student_seed
    @property
    def teacher_input_layout(self):
        return 'PL' if self.teacher_alias=='N2PL' else REFERENCE_ASSETS[self.teacher_alias]['input_layout']
    @property
    def teacher_run_id(self):
        return N2_TEACHER_RUN if self.teacher_alias=='N2PL' else REFERENCE_ASSETS[self.teacher_alias]['teacher_run_id']
    @property
    def run_id(self):
        if self.role=='T': return N2_TEACHER_RUN
        return (f'FH20R1_{self.server_id.upper()}_S_{self.input_layout}_W{self.width}_D{"".join(map(str,self.depth))}'
                f'_WV3_T{self.teacher_alias}_TS{self.teacher_seed}_SS{self.student_seed}_{self.profile}_FRESH50_v1')
    @property
    def reservation_hours(self):
        return 2.2 if self.role=='T' else {'s1':1.568,'s2':1.893,'s3':1.070,'s4':1.126,'s5':1.091}[self.server_id]
    @property
    def eligible_when(self):
        if self.role=='T': return 'S2_DONOR_VERIFIED'
        if self.server_id=='s1':
            return S1_SUPPORT if self.profile=='A10' else S1_FALLBACK if self.depth==(1,2,1) else 'ALWAYS'
        if self.server_id=='s2':
            return S2_SUPPORT if self.teacher_alias=='N2PL' else S2_FALLBACK if self.input_layout=='PLH' else 'ALWAYS'
        return 'ALWAYS'
    def to_dict(self):
        return dict(asdict(self),depth=list(self.depth),run_id=self.run_id,seed=self.seed,
                    teacher_run_id=self.teacher_run_id,teacher_input_layout=self.teacher_input_layout,
                    config_path=f'config/{self.run_id}.yaml',kind='TRAIN',eligible_when=self.eligible_when)


@dataclass(frozen=True)
class Block:
    server_id:str
    block_id:str
    tier:str
    primary_order:tuple
    alternative_orders:dict
    def order_for(self,branch='PRIMARY'):
        return self.alternative_orders.get(branch,self.primary_order)
    def to_dict(self):
        return dict(asdict(self),primary_order=list(self.primary_order),
                    alternative_orders={k:list(v) for k,v in self.alternative_orders.items()})


_cases={}; _blocks=[]
def _student(server,block,seed,layout='PLH',width=104,depth=(1,2,2),alias=None,profile='BASE',tier='CORE'):
    c=Case(server,'S',tier,block,layout,width,tuple(depth),alias or 'F'+server[1:],
           71002 if server=='s5' else 71001,seed,profile)
    if c.run_id in _cases and _cases[c.run_id]!=c: raise AssertionError('Ambiguous case ID')
    _cases[c.run_id]=c
    return c.run_id


for srv in SERVERS:
    if srv=='s2':
        teacher=Case('s2','T','CORE','FH20R1_S2_TEACHER_PREP','PL',112,(1,2,3),'N2PL',71001,None,'N2INIT')
        _cases[teacher.run_id]=teacher
    seeds={'s1':range(73101,73112),'s2':[72001,*range(73201,73209)],
           's3':[72002,*range(73301,73314)],'s4':[72001,73401,73402,73403,73404,72001,73405,73406,73407],
           's5':[72002,73501,73502,73503,73504,72002,73505,73506]}[srv]
    for index,seed in enumerate(seeds,1):
        core_n={'s1':7,'s2':5,'s3':10,'s4':7,'s5':6}[srv]
        tier='CORE' if index<=core_n else 'RESERVE'
        bid=f'FH20R1_{srv.upper()}_B{index:02d}'
        def add(**kwargs): return _student(srv,bid,seed,tier=tier,**kwargs)
        alt={}
        if srv=='s1':
            base=add(); a10=add(profile='A10'); shallow=add(depth=(1,2,1))
            primary=[base,a10] if index%2 else [a10,base]
            alt[S1_FALLBACK]=tuple([shallow,base] if index%2 else [base,shallow])
        elif srv=='s2':
            fresh=add(layout='PL'); n2=add(layout='PL',alias='N2PL'); plh=add(layout='PLH')
            primary=[fresh,n2] if index%2 else [n2,fresh]
            alt[S2_FALLBACK]=tuple([fresh,plh] if index%2 else [plh,fresh])
        elif srv=='s3':
            pair=([add(layout='PH',depth=(1,2,1)),add(layout='PH')] if index<=5 or index>=11
                  else [add(layout='PH'),add(layout='PH',width=112)])
            primary=pair if index%2 else pair[::-1]
        elif srv=='s4' and index==1:
            primary=[add(layout='PL')]
        elif srv=='s4' and index==6:
            primary=[add(width=112)]
        elif srv=='s4' and index==7 or srv=='s5' and index==6:
            primary=[add(width=112,depth=(1,2,1)),add(width=112)]
        elif srv=='s5' and index==1:
            primary=[add(layout='PL',depth=(1,2,1)),add(layout='PL')]
        else:
            quartet=[add(layout='PL',depth=(1,2,1)),add(layout='PL'),add(),add(depth=(1,2,1))]
            # Appendix A starts quartets B02 forward; reserves s5 B07 forward.
            forward=(index%2==0) if srv=='s4' or index<=5 else (index%2==1)
            primary=quartet if forward else quartet[::-1]
        _blocks.append(Block(srv,bid,tier,tuple(primary),alt))

CASES=tuple(_cases.values()); BLOCKS=tuple(_blocks)
assert len(CASES)==145 and len(BLOCKS)==51


def case_for(run_id):
    if run_id not in _cases: raise ValueError(f'Unregistered FH20R1 case: {run_id}')
    return _cases[run_id]
def cases_for(server):
    if server not in SERVERS: raise ValueError(f'Unknown FH20R1 server: {server}')
    return tuple(c for c in CASES if c.server_id==server)
def blocks_for(server):
    cases_for(server)
    return tuple(b for b in BLOCKS if b.server_id==server)
def active_cases(server,branch='PRIMARY',include_reserve=True):
    allowed={'PRIMARY','STANDARD',S1_SUPPORT,S1_FALLBACK} if server=='s1' else ({'PRIMARY','STANDARD',S2_SUPPORT,S2_FALLBACK} if server=='s2' else {'PRIMARY','STANDARD'})
    if branch not in allowed: raise ValueError(f'Branch {branch} does not belong to {server}')
    out=[case_for(N2_TEACHER_RUN)] if server=='s2' and branch!=S2_FALLBACK else []
    for block in blocks_for(server):
        if include_reserve or block.tier=='CORE': out.extend(case_for(i) for i in block.order_for(branch))
    return tuple(out)


def registry_rows(server=None):
    return [c.to_dict() for c in (CASES if server is None else cases_for(server))]
def registry_document():
    stages=[dict(stage_id=f'FH20R1_{s.upper()}_{kind}',server_id=s,kind=kind)
            for s in SERVERS for kind in ('IMPORT','DIAG')]
    stages.append(dict(stage_id='FH20R1_S2_CAL_N2PL',server_id='s2',kind='CALIBRATION'))
    return dict(campaign_id=CAMPAIGN_ID,registry_revision=REGISTRY_REVISION,
                source='supplied Cases CSV (definitions/full SHAs) and MD Appendix A (block order); companion Registry JSON not supplied',
                reference_assets=REFERENCE_ASSETS,stages=stages,blocks=[b.to_dict() for b in BLOCKS],cases=registry_rows())
def registry_sha256(): return object_sha(registry_document())


def build_config(case):
    c=case_for(case) if isinstance(case,str) else case
    cfg=copy.deepcopy(legacy_config(legacy_teacher(c.server_id) if c.role=='T' else legacy_cases(c.server_id)[1]))
    old=cfg.pop('fh12')
    meta=dict(c.to_dict(),campaign_id=CAMPAIGN_ID,registry_revision=REGISTRY_REVISION,
        method_revision=METHOD_REVISION,source_plan=SOURCE_PLAN,
        budget_path=f'work_dir/_fh20r1/{c.server_id}/campaign_budget.json',
        dataset_manifest=f'work_dir/_fh12/{c.server_id}/dataset_manifest.json',
        reference_alias=c.teacher_alias,
        reference_bridge=f'work_dir/_fh20r1/{c.server_id}/imported_refs/{c.teacher_alias}/bridge_manifest.json',
        teacher_checkpoint=f'work_dir/{c.teacher_run_id}/candidates/50000/model.safetensors',
        teacher_from_scratch=False,backbone_from_scratch=True,aligner_init='N2_A_ONLY' if c.role=='T' else 'TEACHER_CLONE',
        init_policy='FH12_named_tensor_fresh9_zero_extra_v1',candidate_grid=list(GRID_STEPS),
        fullstate_steps=list(FULLSTATE_STEPS),alpha=1.,beta=.1,lambda_E=0. if c.role=='T' else .002,
        aligner_lr=1e-5 if c.role=='T' else 3e-6,offset_weight=1e-4 if c.role=='T' else 0.,
        offset_every=2,offset_radius=2. if c.role=='T' else 0.,view_margin_hr=4,corruption_seed=c.seed+100000,
        calibration=dict(n_patches=3072,seed=1234,probe='AXIS16',reuse_actual_indices_from='F2'),
        hqnr_threshold=.9585,ergas_goal=2.040,teacher_ref_step=50000,
        lr_schedule_id='COSINE_W100_A10_T10000_DIV3_v1' if c.profile=='A10' else 'COSINE_W100_BASE_v1',
        a_lr_switch_completed_updates=10000 if c.profile=='A10' else None,
        a_lr_after_multiplier=1/3 if c.profile=='A10' else 1.,gradient_diagnostic_every=5000,
        min_effective_hours=MIN_EFFECTIVE_HOURS)
    cfg.update(seed=c.seed,work_dir=f'work_dir/{c.run_id}',trainer='fh20r1',fh20r1=meta)
    cfg['model_args'].update(hidden_size=c.width,depth=list(c.depth))
    return cfg
