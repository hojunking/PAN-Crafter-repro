"""Two WV3 repetitions and two paired QB edge-reference groups, all fresh/local."""
import csv
import hashlib
import io
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/panda_unified_campaign'
PLAN=json.loads((CONFIG/'plan.json').read_text())
CAMPAIGN_ID,REVISION=PLAN['campaign_id'],PLAN['method_revision']
STAGE1_CASE='S1_RGBN_FULL'
STAGE2_CASE='S2_RGBN_OURS_LR110'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def seeds(seed):
    return {key:1+int.from_bytes(hashlib.sha256(f'donor30:{seed}:{key}'.encode()).digest()[:8],'big')%(2**63-1)
            for key in ('A_init','T_init','S_init','rr_data','rr_augmentation','rr_epsilon','rr_band_augmentation')}

def local_path(value):
    p=Path(value)
    return ROOT/p.relative_to('/workspace') if str(p).startswith('/workspace/') else p if p.is_absolute() else ROOT/p

def sensor_for(server):
    if server not in PLAN['servers']:raise ValueError('Only s1 WV3 and s2 QB are registered')
    return PLAN['servers'][server]

def targets(sensor):return dict(PLAN['targets'][sensor])

def reference_bands(sensor,mode):
    if sensor not in ('WV3','QB') or mode not in ('RGB','RGBN'):raise ValueError('Unknown edge reference')
    return list((4,2,1) if sensor=='WV3' else (2,1,0))+([] if mode=='RGB' else [6 if sensor=='WV3' else 3])

def campaign_root(server):return ROOT/'work_dir/panda_unified_campaign'/CAMPAIGN_ID/server/sensor_for(server)

def stage1_cases(server):return [STAGE1_CASE] if server=='s1' else ['S1_RGB_FULL','S1_RGBN_FULL']

def donor_case(cfg):return 'S1_'+cfg['reference_mode']+'_FULL'

def resolve(server,case,repeat):
    sensor=sensor_for(server)
    if type(repeat) is not int or repeat not in ((1,2) if server=='s1' else (1,)):
        raise ValueError('Unregistered repetition')
    is_s1=case in stage1_cases(server)
    mode='RGBN' if server=='s1' else case.split('_')[1]
    refs=reference_bands(sensor,mode)
    if server=='s1':
        if case not in (STAGE1_CASE,STAGE2_CASE):raise ValueError('Unknown WV3 case')
        recipe={'alpha':.25,'gamma':.25,'lr':1.1e-4}
    elif is_s1:recipe={}
    else:
        candidates={f'S2_{mode}_{r["recipe"]}':r for r in PLAN['qb_stage2_cases']}
        if case not in candidates:raise ValueError('Unknown QB case')
        recipe=candidates[case]
    seed=((261016100+repeat if server=='s1' else 261016301) if is_s1 else 261018100+repeat)
    width=112 if sensor=='WV3' else 128
    from .contract import MASK2_POLICY,WARP_CONTRACT,warp_manifest
    cfg=dict(case_id=case,server=server,sensor=sensor,dataset=sensor,bands=8 if sensor=='WV3' else 4,
        campaign_id=CAMPAIGN_ID,method_revision=REVISION,stage='stage1' if is_s1 else 'stage2',
        repeat=repeat,replica=f'R{repeat}',seed=seed,stream_seeds=seeds(seed),
        run_id=f'UNIFIED_{sensor}_{server.upper()}_{case}_R{repeat}_S{seed}',
        experiment=f'{sensor}-UNIFIED-{case.replace("_","-")} R{repeat} ({mode})',
        alignment='LEARNED' if is_s1 else 'LEARNED_FROZEN',
        alignment_target=f'SCHARR_OF_{mode}_MEAN_GT',reference_mode=mode,reference_bands=refs,
        stage1_spatial_bands=refs,stage1_spatial_weights=[1/len(refs)]*len(refs),stage2_q_bands=refs,
        validation_reference=f'{mode}_MEAN_GT',test_rre_reference=f'{mode}_GT_RR_{mode}_STORED_LMS_FR',
        lambda_abs=(2. if sensor=='WV3' else 4.) if is_s1 else 0.,
        lambda_epsilon=(.003 if sensor=='WV3' else .03) if is_s1 else 0.,
        alpha=0. if is_s1 else recipe['alpha'],gamma=0. if is_s1 else recipe['gamma'],beta=0.,
        horizon=50000 if is_s1 else 30000,primary_step=50000 if is_s1 else 29000,
        common_comparison_step=50000 if is_s1 else 29000,
        lr=1e-4 if is_s1 else recipe['lr'],lr_aligner=1e-5 if is_s1 else 0.,
        width=width,depth=[1,2,3],architecture=f'W{width}_D123_UBIL',stage1_backbone='UBIL',
        sampling_variant='UBIL',student_input='CORRECTED' if is_s1 else 'CORRECTED_DIFF',initialization='FRESH',
        weight_decay=.01,betas=[.9,.999],adam_eps=1e-8,batch=48,microbatch=12,warmup=100,min_lr=0.,
        upsampler='interp23',epsilon_radius=3.,precision='FP32',scale=1.,
        augmentation={'hflip_probability':1.,'vflip_probability':1.,'rot90':True},
        warp='LANCZOS4_REFLECTION_ALL_PATHS',teacher_warp='LANCZOS4_REFLECTION',
        warp_contract=WARP_CONTRACT,warp_manifest=warp_manifest(),mask_policy=MASK2_POLICY,
        stage1_and_teacher_warp='LANCZOS4',epsilon_injection_warp='LANCZOS4',
        validation_rr_margin=19,test_rr_rre_margin=19,test_fr_rre_margin=24,
        stage1_reuses_existing_weights=False,inference_tta=False,rotation_consistency=0.,
        weight_averaging=False,attempt=1,weight='1+alpha*d_e+gamma*d_e*r_q',
        baseline_variant='FRESH_SHARED_UBIL_ALL_L4_COMMON_SCHARR_VALID_MASK',
        source_registry='ablr2/sensor_sources.json',
        source_registry_sha256=hashlib.sha256((ROOT/'ablr2/sensor_sources.json').read_bytes()).hexdigest(),
        checkpoint_selection='VALIDATION_RRE50_RRE90_ERGAS_STEP' if is_s1 else 'HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_HQNR')
    cfg['early_stop']={'enabled':sensor=='QB' and not is_s1,'at_step':10000,'steps':[8000,9000,10000],'HQNR_below':.90,'rule':'ALL_THREE_STRICTLY_BELOW'}
    cfg['evaluation_steps']=list(range(1000,cfg['horizon']+1,1000))
    cfg['validation_steps']=list(range(1000,50001,1000)) if is_s1 else []
    cfg['donor_run_id']=None if is_s1 else resolve(server,donor_case(cfg),repeat)['run_id']
    cfg['donor_alignment_target']=cfg['alignment_target']
    cfg['config_sha256']=digest(cfg)
    return cfg

def queue(server):
    if server=='s1':return [resolve(server,c,r) for r in (1,2) for c in (STAGE1_CASE,STAGE2_CASE)]
    sensor_for(server)
    # Both fresh local donors first, then paired RGB/RGBN at each priority.
    return [resolve(server,c,1) for c in stage1_cases(server)]+[
        resolve(server,f'S2_{mode}_{row["recipe"]}',1)
        for row in PLAN['qb_stage2_cases'] for mode in PLAN['qb_modes']]

def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):raise ValueError('Unregistered configuration')

def catalog(server):
    stream=io.StringIO();keys=['order','server','sensor','stage','case_id','reference_mode','repeat','seed','width','horizon','lr','lr_aligner','lambda_abs','lambda_epsilon','alpha','gamma','donor_run_id','experiment']
    writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for i,cfg in enumerate(queue(server),1):writer.writerow(dict(order=i,**{k:cfg[k] for k in keys if k!='order'}))
    return stream.getvalue()
