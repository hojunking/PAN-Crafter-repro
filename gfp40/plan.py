"""Immutable MD/CSV-derived P40 registry, independent of archived B20 paths."""
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_GF2_P40_S345_20260922_v1'
METHOD_REVISION = 'GFP40_PANMIX_FAMILY_BANK_v1'
REGISTRY_REVISION = 'GFP40_MD_CSV_INDEPENDENT_G025_v2'
SOURCE_STATUS = 'MD_CSV_DERIVED_MISSING_AUTHOR_JSONS'
SOURCE_PLAN = 'research_log/PANDA_GF2_P40_S345_40H_PANMIX_ExperimentPlan_2026-09-22.md'
SOURCE_CASES = 'research_log/GFP40_Cases_All83.csv'
SOURCE_SHAS = MappingProxyType({
    SOURCE_PLAN:'246938acd8df82a379c0020358152aa9198be5a0a0df9b80e46def0ca3095eea',
    SOURCE_CASES:'5818dbb59c11316b985173b6022367e9233399414064b1e8affe97618c9b8c09'})
SERVERS = ('s3','s4','s5')
SHEET_TABS = {s:'GF2-P40-'+s for s in SERVERS}
WINDOW_HOURS, ADMISSION_CUTOFF_HOURS, TRAIN_FINISH_HOURS = 40.,34.,37.
GRID100 = tuple(range(2020,100000,2020))+(100000,)
GRID100_PROVENANCE = dict(commit='5d9f4d6', path='gfb20/plan.py',
    rule='Verified B20 GRID100, 50500 official; 50000 diagnostic only')

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

CONFIRMATION_FAMILY = 'G025'
EXECUTION_POLICY = dict(revision='GFP40_INDEPENDENT_G025_v2',
    authority='USER_REQUEST_REMOVE_INTERSERVER_LOCK',interserver_dependencies=False,
    clock='PER_SERVER_ACTUAL_START_40H',confirmation_family=CONFIRMATION_FAMILY,
    s4_screen='EXPLORATORY_ONLY_NO_CONFIRMATION_FEEDBACK',
    teacher='EXISTING_SERVER_LOCAL_REFERENCE',new_teacher_training=False)
EXECUTION_POLICY_SHA256 = _hash(EXECUTION_POLICY)

def verify_sources(root=ROOT):
    for name,expected in SOURCE_SHAS.items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('P40 source changed; a new revision is required: '+name)
    return dict(SOURCE_SHAS)

def verify_lane(server):
    if server not in SERVERS: raise ValueError('P40 is only s3/s4/s5; s1 and s2 are protected')
    return 'GF2'

def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _family(gammas,tokens):
    return dict(gammas=list(gammas),tokens=list(tokens),
        probabilities=[n/sum(tokens) for n in tokens],native_index=list(gammas).index(1.),
        sigma=1.,kernel_size=7,padding='reflect_no_repeated_edge',train_only=True)

FAMILIES = {
    'G025':_family((.75,1.,1.25),(1,2,1)),
    'G0125':_family((.875,1.,1.125),(1,2,1)),
    'G050':_family((.5,1.,1.5),(1,2,1)),
    'P075':_family((.75,1.,1.25),(1,6,1)),
    'P025':_family((.75,1.,1.25),(3,2,3)),
    'LOW':_family((.75,1.),(1,1)),
    'HIGH':_family((1.,1.25),(1,1))}
UPGRADE_ORDER = ('G0125','G050','P075','P025')
SELECTABLE = ('G025',)+UPGRADE_ORDER
INITIAL_FAMILIES = dict(s3=('G025',),s4=('G025','G050','G0125','P025','P075'),s5=('G025','LOW','HIGH'))
PARENT_ASSETS = json.loads((ROOT/'gfp40/parent_registry.json').read_text())
PARENTS = PARENT_ASSETS['parents']
REFERENCES = PARENT_ASSETS['references']
SETUP_HOURS = dict(s3=5.,s4=6.,s5=2.)
DIAGNOSTIC_HOURS = 1.5

def family_for(name):
    if name not in FAMILIES: raise ValueError('Family must be concrete, registered, and not MSTAR/NATIVE')
    return copy.deepcopy(FAMILIES[name])

@dataclass(frozen=True)
class Case:
    values: object
    def __getattr__(self,name):
        try:return self.values[name]
        except KeyError as exc:raise AttributeError(name) from exc
    def to_dict(self):return copy.deepcopy(dict(self.values))
    @property
    def server_id(self):return self.server
    @property
    def updates(self):return self.added_updates
    @property
    def is_ft(self):return self.parent!='FRESH'
    @property
    def parent_id(self):return self.parent
    @property
    def parent_step(self):return self.parent_updates
    @property
    def parent_run_id(self):
        if not self.is_ft:return ''
        return PARENTS[self.parent]['source_run_id'] if self.parent in PARENTS else case_for(self.parent).run_id
    @property
    def reference_key(self):return self.reference
    @property
    def reference_id(self):return self.reference
    @property
    def profile(self):return self.arm
    @property
    def stage(self):return self.phase
    @property
    def seed(self):return self.model_seed or self.stream_seed
    @property
    def width(self):return self.student_width
    @property
    def depth(self):return tuple(int(v) for v in self.student_depth.split(','))
    @property
    def input_layout(self):return self.student_layout
    @property
    def student_input(self):return self.student_layout
    @property
    def u_peak_lr(self):return self.lr_u
    @property
    def a_peak_lr(self):return self.lr_a
    @property
    def role(self):return 'S'
    @property
    def sensor(self):return 'GF2'
    @property
    def teacher_seed(self):return 94000+int(self.server[1])-2
    @property
    def init_mode(self):return 'PARENT_STUDENT_U_AND_A_WEIGHTS' if self.is_ft else 'FRESH_U_TEACHER_A_CLONE'
    @property
    def input_views(self):return self.input_distribution
    @property
    def reservation_hours(self):return {20000:.65,60000:1.65,100000:2.6}[self.updates]

def _read_cases():
    verify_sources()
    integers=('model_seed','stream_seed','added_updates','parent_updates','lifetime_student_updates',
        'warmup_updates','batch_size','teacher_width','student_width')
    floats=('lr_u','lr_a','alpha','beta','lambda_edge')
    with (ROOT/SOURCE_CASES).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key in integers:row[key]=int(row[key]) if row[key] else None
        for key in floats:row[key]=float(row[key])
        row['binding_required']={'True':True,'False':False}[row['binding_required']]
        row['depends_on']=tuple(json.loads(row['depends_on']))
        if row['phase']=='LOCKED_CONFIRM':
            row['source_phase']=row['phase']
            row['phase']='FIXED_CONFIRM'
        if row['calibration_family']=='MSTAR':
            row['source_run_id']=row['run_id']
            row['source_calibration_family']='MSTAR'
            row['run_id']=row['run_id'].replace('_MSTAR_','_G025_FIXED_')
            row['calibration_family']=CONFIRMATION_FAMILY
            if row['input_distribution']=='MSTAR':row['input_distribution']=CONFIRMATION_FAMILY
            row['depends_on']=tuple(d for d in row['depends_on'] if d!='MSTAR_LOCK')
            row['binding_required']=False
    return tuple(Case(MappingProxyType(row)) for row in rows)

CASES = _read_cases()
PRIMARY_CASES = tuple(c for c in CASES if c.priority=='PRIMARY')
CORE_CASES = tuple(c for c in CASES if c.phase!='FIXED_CONFIRM')

def case_for(identifier,root=ROOT):
    if isinstance(identifier,Case):identifier=identifier.case_id
    rows=[c for c in CASES if identifier in (c.case_id,c.run_id)]
    if len(rows)!=1:raise ValueError('Unregistered P40 case: '+str(identifier))
    return rows[0]

def cases_for(server):
    verify_lane(server)
    return tuple(c for c in CASES if c.server==server)

@dataclass(frozen=True)
class Block:
    block_id:str
    server_id:str
    case_ids:tuple
    @property
    def cases(self):return tuple(case_for(c) for c in self.case_ids)
    @property
    def run_ids(self):return tuple(c.run_id for c in self.cases)
    @property
    def stage(self):return self.cases[0].phase
    @property
    def optional(self):return self.cases[0].priority=='OPTIONAL'
    def to_dict(self):return dict(block_id=self.block_id,server_id=self.server_id,
        case_ids=list(self.case_ids),run_ids=list(self.run_ids),stage=self.stage,optional=self.optional)

def blocks_for(server):
    cases=cases_for(server)
    return tuple(Block(name,server,tuple(c.case_id for c in cases if c.block_id==name))
                 for name in dict.fromkeys(c.block_id for c in cases))

def block_for(identifier):
    rows=[b for s in SERVERS for b in blocks_for(s) if b.block_id==identifier]
    if len(rows)!=1:raise ValueError('Unregistered P40 block')
    return rows[0]

def grid_steps(updates):
    if isinstance(updates,bool) or updates not in (20000,60000,100000):raise ValueError('Unregistered horizon')
    return GRID100 if updates==100000 else tuple(range(updates//20,updates+1,updates//20))

def diagnostic_steps(updates):
    grid_steps(updates)
    return tuple(t for t in (0,1000,5000,10000,20000,40000,50000,60000,75000,100000) if t<=updates)

def fullstate_steps(updates):return diagnostic_steps(updates)

def registry_dict():
    return dict(campaign_id=CAMPAIGN_ID,revision=REGISTRY_REVISION,status=SOURCE_STATUS,
        execution_policy=copy.deepcopy(EXECUTION_POLICY),execution_policy_sha256=EXECUTION_POLICY_SHA256,
        source_files=dict(SOURCE_SHAS),families=FAMILIES,cases=[c.to_dict() for c in CASES],
        grids={str(n):list(grid_steps(n)) for n in (20000,60000,100000)},grid100_provenance=GRID100_PROVENANCE,
        queues={s:[b.to_dict() for b in blocks_for(s)] for s in SERVERS},
        missing_author_assets=['GFP40_ParentAssets.json','GFP40_DesignRegistry.json','GFP40_Queue_s3/s4/s5.json'])

def registry_sha256():return _hash(registry_dict())

BINDING_FIELDS=('dataset_manifest','reference_manifest','parent_manifest','augmentation_manifest',
    'mixed_calibration_manifest','window_path','runtime_policy_sha256','parent_anchor',
    'teacher_checkpoint','teacher_sha256','tau_R','q_ref','q_cache','q_cache_sha256',
    'source_archive_sha256')

def build_config(case,selected_family=None):
    case=case_for(case)
    family=selected_family or case.calibration_family
    if family!=case.calibration_family:
        raise ValueError('Cannot change predeclared family; confirmation is fixed G025')
    meta=dict(case.to_dict(),case_contract=case.to_dict(),server_id=case.server,role='S',sensor='GF2',
        input_layout='PLH',max_pixel=1023,reference_id=case.reference,reference_key=case.reference,
        parent_id=case.parent,profile=case.arm,family=family,a_peak_lr=case.lr_a,u_peak_lr=case.lr_u,
        init_mode=case.init_mode,registry_sha256=registry_sha256(),method_revision=METHOD_REVISION,
        execution_policy=copy.deepcopy(EXECUTION_POLICY),execution_policy_sha256=EXECUTION_POLICY_SHA256,
        candidate_grid=list(grid_steps(case.updates)),candidate_grid_sha256=_hash(grid_steps(case.updates)),
        diagnostic_steps=list(diagnostic_steps(case.updates)),fullstate_steps=list(fullstate_steps(case.updates)),
        runtime_policy_path='gfp40/runtime_policy.json',**{k:None for k in BINDING_FIELDS})
    meta['window_path']=f'work_dir/_gfp40/{case.server}/campaign_window.json'
    # JSON roundtrip is canonical; dependency lists must remain stable on resume.
    meta=json.loads(json.dumps(meta))
    return dict(seed=case.seed,work_dir=f'work_dir/{case.run_id}',trainer='gfp40',phase='train',
        num_iter=case.updates,num_warmup=100,batch_size=48,test_batch_size=1,num_worker=4,
        num_bands=4,max_pixel=1023.,mixed_precision='no',learning_rate=case.lr_u,
        optimizer='AdamW',weight_decay=.01,betas=[.9,.999],eps=1e-8,lr_scheduler='cosine',log_iter=100,
        train_feeder_args=dict(dataroot=None,crop=False,hflip=True,vflip=True,rot=True,ms_size=16,return_meta=True),
        val_feeder_args=dict(dataroot=None),test_reduced_feeder_args=dict(dataroot=None),test_full_feeder_args=dict(dataroot=None),
        model_args=dict(hidden_size=104,depth=[1,2,2],out_channels=4,attn_locations=[],mode_modulation=False,norm='ln',dropout=0.),gfp40=meta)

def validate_config(cfg,require_bound=False):
    f=cfg['gfp40'];case=case_for(f['case_id']);expected=build_config(case,f['family'])
    expected['work_dir']=cfg['work_dir']
    if Path(cfg['work_dir']).name!=case.run_id:raise ValueError('Run directory differs')
    for key in BINDING_FIELDS:expected['gfp40'][key]=copy.deepcopy(f[key])
    for key in ('train_feeder_args','val_feeder_args','test_reduced_feeder_args','test_full_feeder_args'):
        expected[key]['dataroot']=cfg[key]['dataroot']
    if cfg!=expected:raise ValueError('P40 immutable config/recipe differs')
    if require_bound:
        required=['dataset_manifest','reference_manifest','window_path','runtime_policy_sha256']
        if case.is_ft:required+=['parent_manifest']
        if case.arm!='NATIVE0':required+=['augmentation_manifest','mixed_calibration_manifest']
        if any(not f[k] for k in required):raise ValueError('Verified local assets required')
        if not valid_sha(f['runtime_policy_sha256']):raise ValueError('Invalid runtime SHA')
    return case

def validate_registry():
    if len(CASES)!=83 or len(PRIMARY_CASES)!=74 or len({c.run_id for c in CASES})!=83:raise ValueError('Wrong case counts')
    for s,n in zip(SERVERS,(19,31,33)):
        if len(cases_for(s))!=n:raise ValueError('Wrong lane count')
    if sum(c.updates for c in PRIMARY_CASES)!=2440000 or sum(c.updates for c in CASES)!=2860000:
        raise ValueError('Wrong total update budget')
    for c in CASES:
        if (c.campaign_id!=CAMPAIGN_ID or c.reference!='R'+c.server[1]+'_100' or c.batch_size!=48
            or (c.alpha,c.beta,c.lambda_edge)!=(1.,.1,.002) or c.depth!=(1,2,2) or c.width!=104
            or c.teacher_layout!='P0' or c.teacher_width!=112 or c.teacher_depth!='1,2,3'
            or c.student_layout!='PLH' or c.parent_updates+c.updates!=c.lifetime_student_updates
            or c.lr_u!=(1e-5 if c.is_ft else 1e-4) or c.lr_a!=(3e-7 if c.is_ft else 3e-6)):
            raise ValueError('CSV method contract changed: '+c.case_id)
        for d in c.depends_on:
            if case_for(d).server!=c.server or case_for(d).updates!=100000:
                raise ValueError('Bad dependency')
    return dict(cases=83,primary=74,optional=9,new_teachers=0)

validate_registry()
