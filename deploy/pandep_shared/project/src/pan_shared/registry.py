"""Frozen 18-run experiment registry; no legacy server/sensor lane mapping."""
from .common import CAMPAIGN, SENSORS, SOURCE_COMMIT, canonical_sha, atomic_json, read_json
from pathlib import Path

CASES = (
    ('C00','SHARED',SENSORS,104,1.0), ('C11','SHARED',SENSORS,128,1.0),
    ('C01','SHARED',SENSORS,104,.5), ('S01','SINGLE',('WV3',),104,1.0),
    ('S02','SINGLE',('GF2',),104,1.0), ('S03','SINGLE',('QB',),104,1.0))
ORDER = ('C00','S01','S02','S03','C11','C01')

def build_registry(microbatch=48):
    if microbatch not in (48,24,12):
        raise ValueError('Only preflight-fixed microbatch 48, 24 or 12 is allowed')
    runs=[]
    for repeat, seed in enumerate((271001,271002,271003),1):
        for case,mode,sensors,width,fraction in CASES:
            run=dict(campaign_id=CAMPAIGN,case_id=case,mode=mode,sensors=list(sensors),
                width=width,depth=[1,2,2],train_fraction=fraction,repeat=repeat,seed=seed,attempt=1,
                effective_batch_size=48,microbatch=microbatch,precision='fp32',
                exposure_stage_updates=150000 if mode=='SHARED' else 50000,yield_block_updates=50000)
            run['run_id']=f'PDSP_S2_R{repeat:02d}_{case}_W{width}_D122_F{int(fraction*100):03d}_PLH_SE{seed}_A01'
            run['config_sha256']=canonical_sha(run); runs.append(run)
    queue=[]
    for repeat in (1,2,3):
        lookup={r['case_id']:r for r in runs if r['repeat']==repeat}
        # First comparison: shared preview, Singles, then shared continuation.
        stages=[('C00',50000),('S01',50000),('S02',50000),('S03',50000),
                ('C00',100000),('C00',150000),('C11',50000),('C11',100000),
                ('C11',150000),('C01',50000),('C01',100000),('C01',150000)]
        queue.extend(dict(run_id=lookup[c]['run_id'],target_step=t,exposure_stage=1) for c,t in stages)
    result=dict(schema='PANDEP_SHARED_PLAN_v4',campaign_id=CAMPAIGN,server_allowlist=['s2'],
        source_commit=SOURCE_COMMIT,private_data_enabled=False,until_operator_stop=True,
        wallclock_limit_seconds=None,max_global_steps=None,early_stopping=False,
        runs=runs,initial_queue=queue,repeat_order=[1,2,3],extension_case_order=list(ORDER),
        subset_seed=20261001,primary_selector='EXACT_MATCHED_SENSOR_EXPOSURE',
        secondary_selector_shared='MIN_MACRO_SENSOR_VALIDATION_L1_THEN_EARLIER_STEP',
        secondary_selector_single='MIN_SENSOR_VALIDATION_L1_THEN_EARLIER_STEP',
        sheet_title='배포용 모델',sheet_id=1198707876)
    result['registry_sha256']=canonical_sha(result)
    return result

def validate_registry(value):
    runs=value.get('runs',[])
    if not runs: raise ValueError('Registry runs are missing')
    expected=build_registry(runs[0].get('microbatch'))
    if value!=expected: raise ValueError('BLOCKED_IDENTITY: registry differs from frozen v4 design')
    return value

def register(work_root, registry):
    registry=validate_registry(registry); work_root=Path(work_root)
    path=work_root/'cases.json'
    if path.exists() and read_json(path)!=registry:
        raise ValueError('BLOCKED_IDENTITY: cannot overwrite registered Run_ID definitions')
    if not path.exists(): atomic_json(path,registry)
    queue=work_root/'queue.json'
    definition=dict(campaign_id=CAMPAIGN,registry_sha256=registry['registry_sha256'],
                    initial_queue=registry['initial_queue'],extension_case_order=registry['extension_case_order'])
    if queue.exists() and read_json(queue)!=definition:
        raise ValueError('BLOCKED_IDENTITY: frozen queue changed')
    if not queue.exists(): atomic_json(queue,definition)
    return registry
