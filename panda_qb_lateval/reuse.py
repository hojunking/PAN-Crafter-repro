"""Reuse only a complete same-donor, same-recipe local Q1/P1 trajectory."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from panda_sensor_eq40.calibration import CalibrationStats
from . import plan, assets

SCIENCE_KEYS=('bands','width','depth','architecture','sampling_variant','student_input','alpha','beta','gamma',
    'horizon','lr','lr_aligner','weight_decay','betas','adam_eps','batch','microbatch','warmup','min_lr',
    'upsampler','precision','scale','augmentation','warp','teacher_warp','warp_contract','mask_policy',
    'reference_mode','reference_bands','q_zero_policy','stream_seeds','initialization','source_registry_sha256',
    'inference_tta','rotation_consistency','weight_averaging','stage1_spatial_bands','stage2_q_bands')

def same_recipe(a,b):
    return (all(a.get(k)==b.get(k) for k in SCIENCE_KEYS)
            and not a['early_stop']['enabled'] and not b['early_stop']['enabled'])

def stats_equal(a,b):
    ac,aq=a._cpu_arrays();bc,bq=b._cpu_arrays()
    return (a.tau_e==b.tau_e and np.array_equal(ac,bc) and len(aq)==len(bq)
            and all(np.array_equal(x,y) for x,y in zip(aq,bq)))

def apply_if_identical(server,root,donor,stats,pins):
    cfg=plan.resolve(server);p,oldcfg=plan.baseline(server)
    oldroot=p.campaign_root(server);statepath=oldroot/'campaign_state.json'
    receipt_path=Path(root)/'reuse_decision.json'
    def skip(reason):
        atomic_json(receipt_path,{'reused':False,'reason':reason});return False
    if not statepath.exists():return skip('No original local Q1/P1 state')
    jobs=[r for r in assets.read(statepath)['runs'] if r['run_id']==oldcfg['run_id']]
    if len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=50000:
        return skip('Original local Q1/P1 is not a completed50K run')
    if not same_recipe(cfg,oldcfg):return skip('Different scientific recipe')
    oldpins=assets.read(oldroot/'admission.json')
    if any(oldpins[k]!=pins[k] for k in ('evaluator','train_inventory','split_manifests')):
        raise ValueError('Reuse data/evaluator mismatch')
    directory=oldroot/'runs'/oldcfg['run_id']
    if assets.read(directory/'config.json')!=oldcfg:raise ValueError('Reuse source config differs')
    verifier=importlib.import_module(p.__name__.rsplit('.',1)[0]+'.evaluation').verify_record
    records=[]
    for step in cfg['evaluation_steps']:
        path=directory/f'evaluation_{step:07d}.json';e=assets.read(path);verifier(e,oldcfg)
        if e.get('donor_selection',{}).get('selected',{}).get('file_sha256')!=donor.binding['donor_checkpoint_sha256']:
            return skip('Selected late-validation donor differs from original Q1/P1')
        ck=plan.local_path(e['checkpoint']['path'])
        if file_sha(ck)!=e['checkpoint']['file_sha256'] or assets.read(ck.with_suffix('.json'))!=e['checkpoint']:
            raise ValueError('Reuse checkpoint hash/sidecar mismatch')
        records.append((path,e))
    hashes={e['calibration_sha256'] for _,e in records}
    if len(hashes)!=1:raise ValueError('Original Q1/P1 changed calibration')
    oldstats=None
    for path in (oldroot/'donor/calibration').rglob('*.npz'):
        with np.load(path,allow_pickle=False) as archive:
            binding=json.loads(archive['metadata_json'].tobytes())['binding']
        candidate=CalibrationStats.load(path,expected_binding=binding)
        if candidate.artifact_sha256 in hashes:oldstats=candidate;break
    if oldstats is None:raise ValueError('Original Q1/P1 exact calibration missing')
    if not stats_equal(stats,oldstats):return skip('Calibration numerical tables differ; independent run required')
    from .runtime import initial_state
    from .evaluation import verify_record
    target=Path(root)/'runs'/cfg['run_id'];target.mkdir(parents=True,exist_ok=True)
    atomic_json(target/'config.json',cfg)
    for path,old in records:
        e=deepcopy(old)
        e.update(schema='PANDA_QB_LATEVAL_EVAL_V1',run_id=cfg['run_id'],config_sha256=cfg['config_sha256'],
            pan_warp='LANCZOS4_FIXED_REFERENCE_PAN',evaluation_reference=cfg['evaluation_reference'],
            student_input=cfg['student_input'],weight_variant=cfg['weight_variant'],selection_reference='CORRECTED_PAN',
            fixed_reference_donor_sha256=donor.binding['donor_checkpoint_sha256'],
            donor_selection=deepcopy(donor.selection_receipt),calibration_sha256=stats.artifact_sha256,
            reuse={'source_evaluation':str(path),'source_sha256':file_sha(path),
                'source_run_id':oldcfg['run_id'],'source_config':oldcfg,'independent_repeat':False,
                'same_recipe_same_donor_same_calibration_tables':True})
        e['joint_pass']=e['fr_corrected_pan']['metrics']['HQNR']>=.9315 and e['rr']['metrics']['ERGAS']<=3.5
        verify_record(e,cfg)
        dest=target/path.name
        if dest.exists() and assets.read(dest)!=e:raise ValueError('Existing reuse record differs')
        atomic_json(dest,e)
    state=initial_state(server);state['status']='COMPLETED'
    state['runs'][0].update(status='COMPLETED',step=50000,reused_from=oldcfg['run_id'])
    atomic_json(Path(root)/'campaign_state.json',state)
    atomic_json(receipt_path,{'reused':True,'source_run_id':oldcfg['run_id'],'new_training_updates':0,
        'checkpoint_count':50,'independent_repeat':False})
    return True
