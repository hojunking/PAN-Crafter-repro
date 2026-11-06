from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='work_dir/panda_gf2_sharpness'
CAMPAIGN_ID='PANDA_S3_GF2_SHARPNESS_REVIEW_20261106_v1'
TARGET={'HQNR':.9645,'ERGAS':.522,'PSNR':45.08}
DONOR_SHA='937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478'
EARLY=(1000,2000,5000)
INSPECT=(1000,2000,5000,10000,20000,30000,47000,60000)
REPLAY_SAVES=(500,1000,2000,3000,5000)
STOP_UPDATE=5000


def cases():
    result=[]
    for rep,seed in [('R1',261018101),('R2',261018102)]:
        for package,campaign,case,lr in [
            ('panda_sensor_focus','PANDA_S13_SENSOR_FOCUS_FIXEDDONOR_20261029_v1','GF2_C60LR150',1.5e-4),
            ('panda_gf2_psnr','PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1','GF2P_C60LR200',2e-4)]:
            run=f'PSFC_GF2_S3_{case}_NOSOFT_{rep}_S{seed}'
            result.append(dict(case=case,replica=rep,seed=seed,lr=lr,run_id=run,
                original_directory=f'work_dir/{package}/{campaign}/s3/GF2/runs/{run}/attempt_01'))
    return result


def validate_case(cfg,case):
    spec=cfg['stage2_spec'];training=cfg['training_spec']
    checks=[cfg['server']=='s3',cfg['dataset']=='GF2',cfg['phase']=='stage2',
        cfg['run_id']==case['run_id'],cfg['seed']==case['seed'],cfg['case']==case['case'],
        cfg['planned_horizon']==60000,training['total_updates']==60000,spec['total_updates']==60000,
        cfg['lr_S']==case['lr'],training['lr']==case['lr'],training['warmup_updates']==100,
        training['min_lr']==0,training['schedule_kind']=='WARMUP_COSINE',
        training['effective_batch_size']==48,training['microbatch_size']==12,training['weight_decay']==.01,
        spec['student_width']==112,spec['alpha']==.25,spec['gamma']==.5,spec['beta']==0,
        spec['input_variant']=='CORRECTED_DIFF',spec['sampling_variant']=='UBIL',
        cfg['shift_scale']==1,cfg['rotation_consistency_weight']==0,cfg['inference_tta'] is False,
        cfg['weight_averaging']=='EXCLUDED',cfg['donor_checkpoint_sha256']==DONOR_SHA,
        cfg['donor_binding']['donor_step']==42000,cfg['initialization']=='FRESH',cfg['inherited_updates']==0]
    if not all(checks):
        raise ValueError('Original GF2 source setting differs from the registered visual review: '+case['run_id'])


def manifest():
    return dict(campaign_id=CAMPAIGN_ID,server='s3',dataset='GF2',cases=cases(),
        inspect_steps=INSPECT,missing_early_replay_steps=EARLY,replay_saves=REPLAY_SAVES,
        execution_stop=STOP_UPDATE,optimizer_horizon=60000,donor_step=42000,donor_sha256=DONOR_SHA,
        stage1_training=False,cross_server_weight_transfer=False,target=TARGET,
        publication='LOCAL_DIAGNOSTIC_ONLY; existing scientific rows unchanged',
        missing_late_weights='Report unavailable; do not train to 47K/60K for this review',
        baseline='Current qualitative GF2_C60LR150 R1 47K; numerical goal GF2P_C60LR200 R2 47K')
