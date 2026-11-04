"""Same QB science and data/seed protocol, separate Stage2-only campaign identity."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path
from donor30.plan import canonical_sha256, stream_seeds, SEED_PROTOCOL
from panda_qb_s2 import plan as source
from . import registry as G
from .contract import (CAMPAIGN_ID,METHOD_REVISION,RUN_PREFIX,LANES,BUDGET,DONORS,
                       CASES,POLICY,STAGE1_SELECTION,sampling_manifest,input_manifest,
                       validate_loss_pair_config)

ROOT=Path(__file__).resolve().parents[1]
DATASETS={'s2':'QB'}
SEEDS=G.SEEDS
baseline=source.baseline
lms_overrides=source.lms_overrides
evaluation_steps=source.evaluation_steps
rank=source.rank

def dataset_for_server(server):
    if server!='s2': raise ValueError('This campaign is s2/QB only')
    return 'QB'

def groups(server):
    dataset_for_server(server)
    return [dict(name=f"{r['order']:02d}_{r['case']}_{r['replica']}",case=r['case'],
        optional=False,runs=[(r['case'],r['replica'])],admission='COMPLETE_REGISTERED_RUN')
        for r in G.PLAN['queues'][server]]

def default_config(server):
    dataset_for_server(server)
    return dict(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,
        plan_revision='QB_S2_FROZEN_20261104',server=server,dataset='QB',
        baseline_resolved=baseline(server),groups=groups(server),budget=deepcopy(BUDGET),
        cases=deepcopy(CASES),seeds=deepcopy(SEEDS),donors=deepcopy(G.PLAN['donors']),
        archived_baselines=deepcopy(G.PLAN['archived_baselines']),stage1_runs=0,stage2_runs=22,
        joint_target=deepcopy(G.TARGET),allow_cross_server_donor=False,
        donor_policy='REUSE_EXACT_EXISTING_S2_A_T_AND_CALIBRATION_NO_RETRAIN',
        checkpoint_selection=POLICY.manifest(),after_campaign='STOP_NO_WV3_SUCCESSOR',
        lms_overrides=lms_overrides('QB'))

def validate_config(cfg):
    if canonical_sha256(cfg)!=canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered frozen Student campaign')

def resolve_case(server,case,replica='R1',*,attempt=1):
    from .spec import MatureSpec,policy_manifest
    from .engine import ScheduleTrainingSpec
    dataset_for_server(server)
    row=G.job(server,case,replica);p=deepcopy(CASES[case]);pin=G.donor(row['donor_case'])
    original_case=pin['source_case'].replace('_S1_','_S2_')
    cfg=source.resolve_case(server,original_case,attempt=attempt)
    seeds=stream_seeds(row['seed'])
    import hashlib
    seeds['rr_band_augmentation']=1+int.from_bytes(hashlib.sha256(
        f"donor30:{row['seed']}:rr_band_augmentation".encode()).digest()[:8],'big')%(2**63-1)
    spec=MatureSpec('QB','panda_condq',seeds['S_init'],p['alpha'],0.,p['gamma'],1e-6,
        total_updates=30000,student_width=p['student_width'])
    train=ScheduleTrainingSpec(lr=1e-4,total_updates=30000,warmup_updates=100,
        effective_batch_size=48,microbatch_size=12,min_lr=0.,full_audit_interval=100)
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,case=case,loss_case=case,
        replica=replica,seed=row['seed'],run_id=f'{RUN_PREFIX}_QB_S2_{case}_{replica}_S{row["seed"]}',
        run_display=f'QB S2 Frozen {case} {replica}',stream_seeds=seeds,
        config_digest=canonical_sha256(default_config(server)),stage1_retrained=False,
        stage2_spec=asdict(spec),geometry=policy_manifest(spec),training_spec=asdict(train),
        tuning_parameters=p,question=row['reason'],budget_estimate_hours=row['estimate_hours'],
        student_sampling=sampling_manifest('UBIL','CORRECTED_DIFF',p['student_width']),
        student_input=input_manifest('CORRECTED_DIFF',p['student_width']),
        donor_origin_server='s2',execution_server='s2',
        parent_run=pin['source_run_id'],frozen_source_campaign=pin['source_campaign_id'])
    cfg['student_architecture']['width']=p['student_width']
    validate_loss_pair_config(cfg)
    return cfg

def source_checkpoint_path(cfg):
    raise ValueError('Fresh Students only; no Student continuation source')

def catalog_csv(server):
    out=io.StringIO();w=csv.writer(out,lineterminator='\n')
    w.writerow(['order','case','replica','seed','donor','source_step','updates','width','alpha','gamma','beta','scale','lr','batch','microbatch'])
    for r in G.PLAN['queues'][server]:
        c=resolve_case(server,r['case'],r['replica']);p=CASES[r['case']]
        w.writerow([r['order'],r['case'],r['replica'],r['seed'],r['donor_case'],48000,30000,
                    p['student_width'],p['alpha'],p['gamma'],0,1,1e-4,48,12])
    return out.getvalue()
