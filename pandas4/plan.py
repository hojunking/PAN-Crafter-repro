"""Fresh local donor trio, then paired anchors and one-factor loss screening.

This plan never edits the executing s1/s3 registry or borrows their donors.
Only the shared scientific engines and original data protocol are reused.
"""
from copy import deepcopy
import csv
import io
from pathlib import Path

from donor24.plan import baseline as old_baseline, evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from panda_stage2_condq.selection import SelectionPolicy
from panda_stage2_condq.spec import MASK_POLICY

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'PANDA_S4_DONOR_STAGE2_40H_20261017_v1'
METHOD_REVISION = 'PANDA-S4-DIRECT-MASK2-CONDQ-40H-v1'
SERVERS = ('s4',)
SEEDS = {'R1':261016101, 'R2':261016102}
HORIZON = 50000
BUDGET = {'total_seconds':40*3600, 'closeout_seconds':1800, 'safety_factor':1.2}
POLICY = SelectionPolicy(.959)

STAGE1 = {
    'D_A1': (2., .003, 1e-5, 'HQNR-promising but nonreproduced gate; two fresh seeds'),
    'D_A3': (3., .0015, 1e-5, 'Lower-ERGAS donor hedge; one paired-seed screening run'),
}
STAGE2 = {
    'G': ('gt_only',0.,0.,0.,LOSS_MASK_POLICY,'GT-only paired reference'),
    'Q': ('panda_condq',.5,.05,.5,LOSS_MASK_POLICY,'Full new method paired reference'),
    'E': ('panda_e',.5,0.,0.,LOSS_MASK_POLICY,'Hard error weighting; compare G/R1'),
    'ES': ('panda_e',.5,.05,0.,LOSS_MASK_POLICY,'Add selective soft KD; compare E/R1 then Q/R1'),
    'Q_ALOW': ('panda_condq',.25,.05,.5,LOSS_MASK_POLICY,'Change only alpha vs Q/R1'),
    'Q_BHIGH': ('panda_condq',.5,.1,.5,LOSS_MASK_POLICY,'Change only beta vs Q/R1'),
    'Q_GHIGH': ('panda_condq',.5,.05,1.,LOSS_MASK_POLICY,'Change only gamma vs Q/R1'),
    'Q_FULL': ('panda_condq',.5,.05,.5,MASK_POLICY,'Full support policy with separately consistent tau_e'),
}


def baseline(server='s4'):
    if server!='s4':
        raise ValueError('This separate registry is s4/WV3 only')
    result=deepcopy(old_baseline('s1'))
    result.update(server='s4',dataset='WV3')
    return result


def donor_queue():
    # Three runs, NOT three seeds of a single setting. A3/R1 shares data/init
    # seeds with A1/R1, making the recipe comparison interpretable locally.
    return [('D_A1','R1'),('D_A1','R2'),('D_A3','R1')]


def stage2_groups():
    groups=[]
    for case in STAGE2:
        anchor=case in ('G','Q')
        replicas=tuple(SEEDS) if anchor else ('R1',)
        groups.append({'name':case,'anchor':anchor,
            'optional':case not in ('G','Q','E','ES'),
            'evidence_level':'PAIRED_TWO_SEEDS' if anchor else 'SINGLE_SEED_SCREENING',
            'runs':[(case,r) for r in replicas]})
    return groups


def default_config(server='s4'):
    if server!='s4':
        raise ValueError('This separate registry is s4/WV3 only')
    return {'campaign_id':CAMPAIGN_ID,'method_revision':METHOD_REVISION,
        'server':server,'dataset':'WV3','baseline_resolved':baseline(server),
        'budget':deepcopy(BUDGET),'donor_queue':donor_queue(),'stage2_groups':stage2_groups(),
        'minimum_stage2_anchor_runs':4,'stage2_reserve_multiplier':2.,
        'calibration_reserve_seconds':600.,
        'donor_policy':'EXACTLY_THREE_FRESH_LOCAL_COMPLETED_RUNS_THEN_ONE_FIXED_A_T',
        'allow_external_donor':False,'allow_exploratory_donor':True,
        'checkpoint_selection':POLICY.manifest(),
        'sheet_selection':{'HQNR_min':.960,'ERGAS_max':2.045,'otherwise':'best_HQNR_per_run'},
        'evaluation':'ORIGINAL_REFERENCES_UNMASKED_RR20_FR20_SAME_STATE',
        'pruning':'NONE_COMPLETE_ADMITTED_H50_UNLESS_STOP_OR_DEADLINE',
        'screening_interpretation':'SINGLE_SEED_SENSITIVITY_NOT_CONFIRMED_GAIN',
        'time_prior':'s1 Direct H50 2.56h; s4 GPU and Stage2 throughput UNKNOWN until preflight'}


def validate_config(cfg):
    if canonical_sha256(cfg)!=canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered s4 campaign configuration')


def resolve_case(server,case,replica='R1',*,attempt=1):
    config=default_config(server)
    allowed=set(donor_queue())|{tuple(run) for group in stage2_groups() for run in group['runs']}
    if ((case,replica) not in allowed or type(attempt) is not int or attempt<1):
        raise ValueError('Unregistered s4 recipe/replica/attempt')
    seed,b=SEEDS[replica],config['baseline_resolved']
    cfg={'campaign_id':CAMPAIGN_ID,'method_revision':METHOD_REVISION,'server':'s4',
        'dataset':'WV3','case':case,'loss_case':case,'replica':replica,'attempt':attempt,'seed':seed,
        'run_id':f'P4_WV3_S4_{case}_{replica}_S{seed}',
        'stream_seeds':stream_seeds(seed),'stream_seed_protocol':SEED_PROTOCOL,
        'baseline_resolved':deepcopy(b),'schedule':'H50','planned_horizon':HORIZON,
        'evaluation_steps':list(evaluation_steps()),'lr_U':1e-4,'weight_decay':.01,
        'hqnr_min':.959,'ergas_max':2.040,'checkpoint_selection':POLICY.manifest(),
        'initialization':'FRESH','aligner_kind':'direct',
        'baseline_alignment_protocol_digest':alignment_protocol_digest(b),
        'config_digest':canonical_sha256(config)}
    if case in STAGE1:
        a,e,lr,question=STAGE1[case]
        cfg.update(phase='donor',stage2=False,lambda_rec=1.,lambda_abs=a,lambda_epsilon=e,
            lambda_eps_RR=e,lr_A=lr,loss_mask_policy=LOSS_MASK_POLICY,donor_checkpoint=None,
            epsilon_radius_rr=3.,epsilon_distribution='UNIFORM_AREA_DISK',
            epsilon_every_update=True,epsilon_reduction='RR_SAMPLE_MEAN',question=question,
            run_display=f'{case} direct a{a:g} eps{e:g} lrA{lr:g} M2 {replica}')
    else:
        variant,a,beta,gamma,mask,question=STAGE2[case]
        spec={'sensor':'WV3','variant':variant,'seed':cfg['stream_seeds']['S_init'],
              'alpha':a,'beta':beta,'gamma':gamma,'delta':1e-6,'mask_policy':mask}
        cfg.update(phase='stage2',stage2=True,lr_A=0.,lambda_abs=0.,lambda_epsilon=0.,
            loss_mask_policy=mask,stage2_spec=spec,
            student_architecture={'bands':8,'width':112,'depth':[1,2,3]},question=question,
            evidence_level='PAIRED_TWO_SEEDS' if case in ('G','Q') else 'SINGLE_SEED_SCREENING',
            run_display=f'{case} a{a:g} b{beta:g} q{gamma:g} '+
                ('M2' if mask==LOSS_MASK_POLICY else 'FULL')+f' P0 {replica}')
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])


def choose(records):
    return max(records,key=rank) if records else None


def catalog_csv():
    out=io.StringIO(); writer=csv.writer(out)
    writer.writerow(['order','phase','group','case','replica','seed','anchor','optional',
                     'run_display','horizon','question'])
    order=0
    for group in [{'name':'DONOR_TRIO','runs':donor_queue(),'anchor':False,'optional':False},*stage2_groups()]:
        for case,replica in group['runs']:
            order+=1; cfg=resolve_case('s4',case,replica)
            writer.writerow([order,cfg['phase'],group['name'],case,replica,cfg['seed'],
                group['anchor'],group['optional'],cfg['run_display'],HORIZON,cfg['question']])
    return out.getvalue()
