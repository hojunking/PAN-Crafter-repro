"""Read-only reuse of the exact main Ours R1; no relabelled training artifacts."""
import json
from pathlib import Path
from directshift48.reporting import atomic_json, utc_now
from ta2.evaluation import file_sha
from panda_eq40_zeroq.evaluation import verify_record as verify_main
from . import plan

RECIPE_KEYS = ('seed','stream_seeds','width','depth','architecture','sampling_variant','student_input',
    'lr','warmup','min_lr','horizon','weight_decay','betas','adam_eps','batch','microbatch','precision',
    'alpha','gamma','beta','q_zero_policy','scale','upsampler','augmentation','warp','teacher_warp',
    'reference_mode','reference_bands','mask_policy','donor_run_id','inference_tta','rotation_consistency')

def validate_recipe(cfg):
    plan.validate(cfg)
    if not cfg['reuse_existing'] or cfg['case_id']!='E_OURS' or cfg['repeat']!=1:
        raise ValueError('Only exact Ours R1 can reuse the completed main run')
    source=plan.source_config()
    different=[k for k in RECIPE_KEYS if cfg[k]!=source[k]]
    if different:raise ValueError('Reuse recipe mismatch: '+repr(different))
    if cfg['weight_variant']!='PRODUCT' or not cfg['input_alignment']:
        raise ValueError('Wrong computational path for reuse')
    path=plan.source_directory()/'config.json'
    if json.loads(path.read_text())!=source:raise ValueError('Original main config changed')
    state=json.loads((plan.main.campaign_root('s1')/'campaign_state.json').read_text())
    jobs=[r for r in state['runs'] if r['run_id']==source['run_id']]
    if len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=40000:
        raise ValueError('Original main Ours R1 is not complete')
    return source

def verify_record(record,cfg):
    source=validate_recipe(cfg)
    verify_main(record,source)
    if (record['checkpoint']['config_sha256']!=source['config_sha256']
        or record['donor_selection']['selected']['file_sha256']!=cfg['fixed_donor']['checkpoint_file_sha256']
        or record['calibration_sha256']!=cfg['fixed_donor']['calibration_artifact_sha256']):
        raise ValueError('Original main weight/donor/calibration differs')

def evidence(cfg,path):
    source=validate_recipe(cfg)
    return {'kind':'EXACT_COMPLETED_MAIN_R1_REUSED_NO_NEW_TRAINING',
        'source_campaign_id':source['campaign_id'],'source_run_id':source['run_id'],
        'source_config_sha256':source['config_sha256'],'source_evaluation_path':str(path),
        'source_evaluation_sha256':file_sha(path),'recipe_keys_verified':list(RECIPE_KEYS),
        'fixed_donor_file_sha256':cfg['fixed_donor']['checkpoint_file_sha256'],
        'train_log_path':str(plan.source_directory()/'train.jsonl'),
        'comparison_step':40000,'source_bytes_modified':False}

def initialize(root,server):
    from .runtime import read_state
    state=read_state(root,server)
    for cfg,job in zip(plan.queue(server),state['runs']):
        if not cfg['reuse_existing']:continue
        source=validate_recipe(cfg)
        paths=sorted(plan.source_directory().glob('evaluation_*.json'))
        if [int(p.stem.split('_')[-1]) for p in paths]!=cfg['evaluation_steps']:
            raise ValueError('Reuse requires all 40 original evaluation records')
        for path in paths:
            record=json.loads(path.read_text());verify_record(record,cfg)
            ck=plan.local_path(record['checkpoint']['path'])
            if file_sha(ck)!=record['checkpoint']['file_sha256']:
                raise ValueError('Original retained weight SHA mismatch')
        audit={'run_id':cfg['run_id'],'source_run_id':source['run_id'],'status':'VERIFIED',
            'updates_reused':40000,'additional_training_updates':0,'evaluation_count':len(paths),
            'all_checkpoint_sha256_verified':True,'at':utc_now()}
        atomic_json(Path(root)/'reuse/ours_r1.json',audit)
        job.update(status='COMPLETED',step=40000,reused_from=source['run_id'],additional_training_updates=0)
    atomic_json(Path(root)/'campaign_state.json',state)
    return state
