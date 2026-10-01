#!/usr/bin/env python3
"""Create a fixed, offline experiment registry. Does not train or access Sheets."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'PANDA_REBUTTAL_B02_M12_WV3_S135_20261001_v1'
SERVERS = ('s1','s3','s5')
SEEDS = {s:[261001000 + int(s[1:])*100 + r for r in range(1,5)] for s in SERVERS}
CASE_SPECS = [
 ('QFULL','STEP1_QROUTE','QFULL','QFULL','ADAPTIVE','ADAPTIVE',1.0,'Registered full method'),
 ('QMEAN','STEP1_QROUTE','TRAIN_MEAN','TRAIN_MEAN','ADAPTIVE','ADAPTIVE',1.0,'Replace both geometry weights by train-view mean'),
 ('QSHUF','STEP1_QROUTE','QSHUF','QSHUF','ADAPTIVE','ADAPTIVE',1.0,'Permute weights within train view and e stratum'),
 ('QESUR','STEP1_QROUTE','QESUR','QESUR','ADAPTIVE','ADAPTIVE',1.0,'e-ranked geometry weights with matched marginal distribution'),
 ('QEDGE','STEP1_QROUTE','QFULL','TRAIN_MEAN','ADAPTIVE','ADAPTIVE',1.0,'Adaptive geometry only on U edge supervision'),
 ('QALIGN','STEP1_QROUTE','TRAIN_MEAN','QFULL','ADAPTIVE','ADAPTIVE',1.0,'Adaptive geometry only on A hard supervision'),
 ('H0','STEP2_FITTING','QFULL','QFULL','PLAIN','ADAPTIVE',0.0,'Remove only hard difficulty emphasis'),
 ('HSPMEAN','STEP2_FITTING','QFULL','QFULL','SPATIAL_MEAN','ADAPTIVE',1.0,'Preserve hard-weight mass per sample; remove pixel allocation'),
 ('NOADV','STEP2_FITTING','QFULL','QFULL','ADAPTIVE','ONE',1.0,'Remove only advantage gate; keep trust and beta'),
 ('ADVMEAN','STEP2_FITTING','QFULL','QFULL','ADAPTIVE','TRUST_WEIGHTED_SPATIAL_MEAN',1.0,'Preserve trust-weighted advantage mass per sample'),
]
MODES = ('A_ON','A_ZERO_INFERENCE_ONLY','A_NATIVE_FIXED','KNOWN_SHIFT_INVERSE')
REFERENCE = 'FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1'
ANCHOR = 'FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1'


def write_csv(path:Path, rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def dump(path:Path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build():
    templates=[]
    for case,phase,we,wa,hm,am,alpha,description in CASE_SPECS:
        templates.append(dict(case_id=case,phase=phase,edge_weight=we,aligner_weight=wa,
            hard_mode=hm,advantage_mode=am,alpha=alpha,beta=.1,lambda_edge=.002,
            per_server_repeats=4,total_student_seeds=12,description=description))
    runs=[]
    for server in SERVERS:
        for repeat,seed in enumerate(SEEDS[server],1):
            for spec in templates:
                case=spec['case_id']
                rid=f'RBM12_WV3_{server.upper()}_R{repeat}_SS{seed}_{case}_F50K_v1'
                runs.append(dict(campaign_id=CAMPAIGN,run_id=rid,dataset='WV3',server=server,
                    repeat=repeat,seed=seed,paired_block_id=f'M12_{server}_R{repeat}_SS{seed}',
                    case_id=case,phase=spec['phase'],updates=50000,
                    primary_selection='EXACT_50000',secondary_selection='RR_VAL_ERGAS_MIN',
                    initial_U='FRESH_SHARED_WITHIN_BLOCK',initial_A='FROZEN_F1_CLONE_THEN_TRAINABLE',
                    teacher_run_id=REFERENCE,teacher_updates=50000,
                    work_dir=f'work_dir/_panda_rb/20261001/B02_M12/train/{server}/R{repeat}/{case}',
                    old_B01_reuse=False,status='SPECIFIED_NOT_STARTED'))
    native=[];curves=[]
    for r in runs:
        for sel in ('EXACT_50000','RR_VAL_ERGAS_MIN'):
            native.append(dict(task_id=r['run_id']+'__NATIVE__'+sel,source_run_id=r['run_id'],
                server=r['server'],repeat=r['repeat'],seed=r['seed'],case_id=r['case_id'],
                selection=sel,rr_scenes=20,fr_scenes=20,optimizer_updates=0,
                statistical_unit=r['run_id']))
        if r['phase']=='STEP1_QROUTE':
            for mode in MODES:
                curves.append(dict(task_id=r['run_id']+'__STRESS__'+mode,source_run_id=r['run_id'],
                    server=r['server'],repeat=r['repeat'],seed=r['seed'],case_id=r['case_id'],
                    selection='EXACT_50000',mode=mode,shifts=49,rr_scenes=20,
                    primary_roi='48:-48_FIXED160',auxiliary_roi='32:-32_FIXED192',
                    optimizer_updates=0,statistical_unit=r['run_id']))
    schedule=[]
    for si,server in enumerate(SERVERS):
        position=0
        def add(action,phase='',repeat='',case='',mode='',rid=''):
            nonlocal position
            position+=1
            schedule.append(dict(server=server,position=position,action=action,phase=phase,
                repeat=repeat,case_id=case,mode=mode,run_id=rid))
        add('VERIFY_ASSETS_AND_RECOVER_OLD_EVIDENCE','PREFLIGHT')
        add('DIAGNOSE_F1_INITIAL_AND_EXISTING_FINAL_ALIGNER','PREFLIGHT')
        for phase in ('STEP1_QROUTE','STEP2_FITTING'):
            base=[t['case_id'] for t in templates if t['phase']==phase]
            for rep in range(1,5):
                shift=(si*4+rep-1)%len(base)
                order=base[shift:]+base[:shift]
                for case in order:
                    r=next(x for x in runs if x['server']==server and x['repeat']==rep and x['case_id']==case)
                    add('TRAIN_FRESH_50K_AND_NATIVE',phase,rep,case,'',r['run_id'])
                    if phase=='STEP1_QROUTE':
                        for mode in MODES:
                            add('STRESS_INFERENCE',phase,rep,case,mode,r['run_id'])
                    add('QUEUE_VERIFIED_UPLOAD',phase,rep,case,'',r['run_id'])
            add('SAVE_PHASE_REPORT_NO_SCORE_GATE',phase)
        add('FINAL_REPORT_AND_STOP_FOR_REVIEW','COMPLETE')
    shifts=[dict(id='D000',radius_hr=0.,angle_degrees=None,dy=0.,dx=0.)]
    for radius in (.25,.5,1.,2.,3.,4.):
        for angle in range(0,360,45):
            dy=radius*math.sin(math.radians(angle));dx=radius*math.cos(math.radians(angle))
            # Preserve mathematical components; do not round or snap by outcome.
            shifts.append(dict(id=f'D{len(shifts):03d}',radius_hr=radius,angle_degrees=angle,dy=dy,dx=dx))
    reg=dict(schema='PANDA_RB_M12_PLAN_v1',campaign_id=CAMPAIGN,status='SPECIFIED_NOT_STARTED',
        dataset='WV3',servers=list(SERVERS),repeats_per_server=4,seed_table=SEEDS,
        statistical_unit='One freshly initialized Student per case and paired seed; fixed F1',
        primary_analysis='New M12 cohort only; old B01 and ABLR2 remain separate',
        method_anchor=dict(teacher_run_id=REFERENCE,teacher_update=50000,source_student_config_run=ANCHOR,
            student=dict(layout='PLH',width=104,depth=[1,2,2],norm='ln',attention=False,mode_modulation=False)),
        training=dict(updates=50000,batch_size=48,warmup=100,lr_U=1e-4,lr_A=3e-6,
            optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,precision='FP32',
            validation_grid=[1010*n for n in range(1,50)]+[50000],
            diagnostic_steps=[0,1000,10000,25000,50000]),
        templates=templates,training_runs=runs,native_tasks=native,stress_tasks=curves,
        expected=dict(training_runs=120,training_runs_per_server=40,native_observations=240,
            stress_curves=288,stress_predictions=282240,stress_primary_points=14112,
            per_case_students=12,independent_teacher_models=1),
        automatic_case_search=False,score_based_stopping=False,launch_performed=False)
    (ROOT/'planning').mkdir(exist_ok=True)
    dump(ROOT/'planning/experiment_registry.json',reg)
    dump(ROOT/'planning/shift_grid.json',dict(coordinate_order=['dy','dx'],unit='HR_pixel',shifts=shifts))
    write_csv(ROOT/'planning/case_templates.csv',templates)
    write_csv(ROOT/'planning/training_runs_120.csv',runs)
    write_csv(ROOT/'planning/native_evaluations_240.csv',native)
    write_csv(ROOT/'planning/stress_curves_288.csv',curves)
    write_csv(ROOT/'planning/server_schedule.csv',schedule)
    dump(ROOT/'planning/binding.template.json',dict(schema='M12_BINDING_TEMPLATE_NOT_READY',
        teacher_run_id=REFERENCE,teacher_update=50000,teacher_checkpoint_sha256=None,
        source_B01_binding_path=None,source_B01_binding_sha256=None,
        teacher_resolved_config_sha256=None,q_ref=None,tau_rec=None,
        q_cache_sha256=None,e_bar_cache_sha256=None,train_view_identity_sha256=None,
        data_content_identity=None,numerical_release_sha256=None,evaluator_sha256=None,
        initialization_and_stream_by_block={},ready_to_train=False,
        instruction='Resolve and verify actual immutable B01 F1/data/calibration assets; nulls block training.'))
    return reg,schedule

if __name__=='__main__':
    reg,schedule=build()
    print(json.dumps(reg['expected'],ensure_ascii=False,indent=2))
