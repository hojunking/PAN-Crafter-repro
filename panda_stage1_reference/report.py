"""Same-checkpoint Stage1 Teacher dual-FR report; no training or reselection."""
from pathlib import Path
import json,hashlib,math
from panda_unified_campaign.plan import ROOT,local_path,digest
from directshift48.reporting import atomic_json,utc_now,METRICS,serialize_row
from ta2.evaluation import file_sha

CAMPAIGN='PANDA_CURRENT_STAGE1_REFERENCE_20261114_v1'
def row_equal(actual,expected):
    from sheet_publish_v2.transport import _matrix_equal
    if len(actual)>67 or len(expected)!=67:return False
    actual=list(actual)+['']*(67-len(actual))
    return all(_matrix_equal([[a]],[[b]]) for a,b in zip(actual,expected))

def resolve_source(server,sensor,source='current'):
    if (server,sensor) not in [('s1','WV3'),('s2','QB'),('s3','GF2'),('s3','QB')]:
        raise ValueError('Explicit same-server dataset reference required')
    if server=='s1':
        from panda_wv3_s2abl5 import plan as p
        from panda_wv3_s2abl5.release import verify
        verify(); cfg=p.main.donor_config('s1',1); root=p.main.donor_root('s1'); fixed=p.PLAN['fixed_donor']
    elif server=='s3' and sensor=='QB':
        from panda_qb_s3_parallel import plan as p
        from panda_qb_s3_parallel.assets import select
        from panda_qb_s3_parallel.release import verify
        verify(); root=p.campaign_root('s3'); cfg,receipt=select(root)
        fixed={'step':receipt['selected']['step'],'checkpoint_file_sha256':receipt['selected']['file_sha256']}
    else:
        from panda_sensor_main50 import plan as p
        from panda_sensor_main50.release import verify
        verify();cfg=p.donor_config(server);root=p.donor_root(server);fixed=p.PLAN['donors'][sensor]
    directory=root/'runs'/cfg['run_id'];step=fixed['step']
    if json.loads((directory/'config.json').read_text())!=cfg:raise ValueError('Reference configuration differs')
    ev=directory/f'evaluation_{step:07d}.json'; record=json.loads(ev.read_text())
    if server=='s3' and sensor=='QB':
        from panda_qb_s3_parallel.evaluation import verify_record
    elif sensor=='GF2':
        from panda_sensor_eq40.evaluation import verify_record
    else:
        from panda_unified_campaign.evaluation import verify_record
    verify_record(record,cfg)
    ck=directory/'models'/f'step_{step:07d}.pt';meta=json.loads(ck.with_suffix('.json').read_text())
    if (record['checkpoint']!=meta or meta['step']!=step or meta['file_sha256']!=fixed['checkpoint_file_sha256']
        or meta['model_sha256']!=record['model_sha256'] or file_sha(ck)!=meta['file_sha256']):
        raise ValueError('Selected reference bytes/evaluation mismatch')
    return cfg,root,ev,record,ck,meta

def source_model(cfg,root,ck,meta):
    import torch
    if cfg['sensor']=='GF2':
        from panda_sensor_eq40 import assets,model,training
        source,registered=assets.sources('s3')
    elif cfg['server']=='s3':
        from panda_qb_s3_parallel import assets,model,training
        source,registered=assets.sources('s3')
    else:
        from panda_unified_campaign import assets,model,training
        source,registered=assets.sources(cfg['server'])
    from ta2.evaluation import evaluator_identity
    pins=json.loads((root/'admission.json').read_text())
    if (pins['evaluator']!=evaluator_identity() or registered.inventory.manifest()!=pins['train_inventory']
        or {s:source.manifest_sha(s) for s in ('train','val','rr','fr')}!=pins['split_manifests']):raise ValueError('Source/evaluator changed')
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    if saved['config']!=cfg or saved['pins']!=pins or saved['updates']!=meta['step']:raise ValueError('Saved donor differs')
    with torch.random.fork_rng(devices=[]):net=model.Stage1(cfg,registered.inventory.manifest())
    net.load_state_dict(saved['model'],strict=True);net.requires_grad_(False).eval()
    if training.model_hash(net)!=meta['model_sha256']:raise ValueError('A/T model hash differs')
    return net,source,pins,training

def validate_corrected(record,corrected,sensor):
    keys={'HQNR','D_s','D_lambda','JQM'}
    a=record['fr']['per_scene'];b=corrected['per_scene']
    if len(a)!=20 or len(b)!=20:raise ValueError('Full FR20 required')
    import numpy as np
    for i,(x,y) in enumerate(zip(a,b)):
        if x['scene_id']!=y['scene_id'] or y['scene_id']!=f'{sensor}:fr:{i}':raise ValueError('Scene ordering differs')
        if set(y['metrics'])!=keys or not all(math.isfinite(v) for v in y['metrics'].values()):raise ValueError('Invalid metrics')
        if abs(x['metrics']['D_lambda']-y['metrics']['D_lambda'])>1e-5:raise ValueError('D_lambda changed without changing output/LMS')
    for k in keys:
        if not math.isclose(float(np.mean([r['metrics'][k] for r in b])),corrected['metrics'][k],abs_tol=1e-12,rel_tol=0):raise ValueError('Aggregate does not match FR20')

def evaluate_source(server,sensor,source='current'):
    cfg,root,ev,record,ck,meta=resolve_source(server,sensor,source)
    folder=ROOT/'work_dir/panda_stage1_reference'/server/sensor/(cfg['run_id']+f"_{meta['step']}")
    path=folder/'evaluation.json';evsha=file_sha(ev)
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['checkpoint']['file_sha256']!=meta['file_sha256'] or cached['source_evaluation_sha256']!=evsha:raise ValueError('Report cache identity differs')
        validate_corrected(record,cached['fr_corrected_pan'],sensor)
        return cached,folder
    result={'schema':'PANDA_SELECTED_STAGE1_TEACHER_DUAL_FR_V1','recorded_at':utc_now(),'config':cfg,
        'checkpoint':meta,'model_sha256':record['model_sha256'],'source_evaluation':str(ev),
        'source_evaluation_sha256':evsha,'evaluated_role':'SAME_CHECKPOINT_ALIGNER_AND_TEACHER',
        'rr':record['rr'],'fr':record['fr'],'no_new_training':True,'no_checkpoint_reselection':True,
        'original_PAN_metrics_auxiliary':True,'inference_tta':False}
    if 'fr_corrected_pan' in record:
        corrected=record['fr_corrected_pan'];result['measurement']='REUSE_VERIFIED_NATIVE_DUAL_FR_EVALUATION'
    else:
        old=ROOT/'results_log/assets/2026-11-14-wv3-reference-corrected-pan/result.json'
        if server=='s1' and old.exists():
            probe=json.loads(old.read_text())
            if (probe['checkpoint_file_sha256']!=meta['file_sha256'] or probe['original_evaluation_sha256']!=evsha
                or not probe['original_metrics_reproduced_at_4_decimals']):raise ValueError('WV3 CPU evaluation provenance mismatch')
            corrected=probe['fr_corrected_pan'];result['measurement']='VERIFIED_EXISTING_CPU_FR20';result['probe_sha256']=file_sha(old)
        else:
            import numpy as np,torch
            from directshift48.evaluation import metric_functions_for,metric_values,expected_ids
            from panda_sensor_eq40.evaluation import dual_fr_metrics
            torch.set_num_threads(2)
            net,data,pins,training=source_model(cfg,root,ck,meta);fn=metric_functions_for(data,pins['evaluator'])['fr'];rows=[];deltas=[]
            with torch.inference_mode():
                for i,sid in enumerate(expected_ids(data,'fr')):
                    batch=data.batch('fr',[i],device='cpu');out=net.native_prediction(batch)
                    raw=((out['y'][0]+1)*(data.peak/2)).numpy();pan=((out['pan_aligned'][0]+1)*(data.peak/2)).numpy()
                    arrays={k:v[0] for k,v in data.raw('fr',[i]).items()};original=metric_values(fn(raw,arrays),'fr',data.bands)
                    deltas.extend(abs(original[k]-record['fr']['per_scene'][i]['metrics'][k]) for k in original)
                    rows.append({'scene_id':sid,'metrics':dual_fr_metrics(fn,raw,arrays,pan,original,data.bands),
                        'prediction_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'shift_dy_dx':out['u'][0].tolist()})
                    print(f'Stage1 {sensor} FR {i+1}/20',flush=True)
            if max(deltas)>1e-4 or training.model_hash(net)!=meta['model_sha256']:raise ValueError('CPU native evaluation failed reproduction/state check')
            corrected={'per_scene':rows,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
            result.update(measurement='CPU_NATIVE_FR20_SAME_EVALUATOR',max_per_scene_original_FR_difference=max(deltas))
    validate_corrected(record,corrected,sensor);result['fr_corrected_pan']=corrected
    atomic_json(path,result);return result,folder

def row_for(result,server):
    from panda_fr_columns.schema import expand
    cfg=result['config'];step=result['checkpoint']['step'];sensor=cfg['sensor']
    rid=f'S1REF|{server}|{sensor}|{cfg["run_id"]}|{step}'
    note=dict(result);note.pop('rr');note.pop('fr');note.pop('fr_corrected_pan')
    note['raw_metrics']={s:result[s]['metrics'] for s in ('rr','fr','fr_corrected_pan')}
    note['selection']='Actual Stage2 donor, not independently optimized Teacher or final50K'
    note=json.dumps(note,sort_keys=True,ensure_ascii=False)
    fields={'Experiment':f'{sensor}-Stage1-reference / Teacher {cfg["replica"]} (RGBN)', 'Server':server,
        'Selection@Step':f'현재 donor · Teacher · 보정 PAN · {step:,}', 'Dataset':sensor,'Bucket':'main',
        'Approach':f'Paired A+Teacher; {cfg["architecture"]}; RGBN; Lanczos4; scale1; common support; spa={cfg["lambda_abs"]}; eps={cfg["lambda_epsilon"]}',
        'Campaign_ID':CAMPAIGN,'Run_ID':cfg['run_id'],'Case_ID':'CURRENT_STAGE1_REFERENCE','Repeat':cfg['repeat'],
        'Seed':cfg['seed'],'Attempt':1,'Role':'Stage1 Aligner+Teacher','Model / inputs':cfg['architecture']+' / corrected PAN + I_lrms',
        'Teacher / reference':cfg['run_id'],'Updates':step,'Lifetime_updates':step,'Selected_step':step,
        'Checkpoint_SHA':result['checkpoint']['file_sha256'],'Selector':'ACTUAL_STAGE2_DONOR',
        'Status':'COMPLETED','Readback':'VERIFIED','Eval_scope':'FR20 ORIGINAL_AND_CORRECTED_PAN / SAME_TEACHER_OUTPUT',
        'Test_aware':True,'Cost_scope':'Inference costs not measured','Result_ID':rid,'Date':result['recorded_at'],
        'Original description / notes':note,'U_peak_lr':cfg['lr'],'A_peak_lr':cfg['lr_aligner'],'lambda_E':cfg['lambda_epsilon']}
    for s in ('rr','fr'):
        for k,v in result[s]['metrics'].items():fields[METRICS[k]]=round(v,4)
    return expand(serialize_row(fields),[round(result['fr_corrected_pan']['metrics'][k],4) for k in ('HQNR','D_s','D_lambda')]),note

def publish_source(server,sensor,*,source='current',apply=False):
    from panda_fr_columns.adapter import GoogleSheets
    from panda_fr_columns.schema import BOOK,HEADERS67
    from sheet_publish_v2.transport import _typed,_matrix_equal
    result,folder=evaluate_source(server,sensor,source);row,note=row_for(result,server)
    dest={'spreadsheet_id':BOOK,'sheet_title':sensor+'-ablations','sheet_id':{'WV3':261001010,'QB':261001011,'GF2':261001012}[sensor]}
    api=GoogleSheets(ROOT/'gspread/account.json');snap=api.native(dest)
    if snap['header']!=HEADERS67:raise ValueError('FR67 layout required')
    matches=[i for i,r in enumerate(snap['rows']) if len(r)>62 and r[62]==row[62]]
    if len(matches)>1:raise ValueError('Duplicate reference result IDs')
    cells=[{'userEnteredValue':_typed(v),'note':note if i==0 else '',**({'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}} if 3<=i<25 else {})} for i,v in enumerate(row)]
    if matches:
        i=matches[0];old=snap['rows'][i]
        if old[31]!=CAMPAIGN or old[44]!=row[44]:raise ValueError('Existing reference identity conflicts')
        if row_equal(old,row) and snap['notes'][i]==note:
            receipt={'action':'ALREADY_VERIFIED','sheet_row':i+6,'destination':dest,'readback_verified':True,
                'run_id':result['config']['run_id'],'step':result['checkpoint']['step'],'corrected_fr':result['fr_corrected_pan']['metrics']}
            atomic_json(folder/'publication.json',receipt);return receipt
        if any(c.get('dataValidation') or c.get('chipRuns') for c in snap['cells'][i]):raise ValueError('Reference cells contain controls')
        req={'updateCells':{'range':{'sheetId':dest['sheet_id'],'startRowIndex':i+5,'endRowIndex':i+6,'startColumnIndex':0,'endColumnIndex':67},'rows':[{'values':cells}],'fields':'userEnteredValue,note,userEnteredFormat.numberFormat'}}
    else:req={'appendCells':{'sheetId':dest['sheet_id'],'rows':[{'values':cells}],'fields':'userEnteredValue,note,userEnteredFormat.numberFormat'}}
    atomic_json(folder/'publication_intent.json',{'destination':dest,'request':req,'apply':apply})
    if not apply:return {'action':'DRY_RUN','new_rows':int(not matches),'destination':dest}
    if (ROOT/'gspread/server.txt').read_text().strip()!=server:raise ValueError('Publish on the matching local server')
    atomic_json(folder/'sheet_before.json',snap)
    api.physical_request('POST',api.BASE+BOOK+':batchUpdate',json={'requests':[req]})
    after=api.native(dest);found=[i for i,r in enumerate(after['rows']) if len(r)>62 and r[62]==row[62]]
    if len(found)!=1 or not row_equal(after['rows'][found[0]],row) or after['notes'][found[0]]!=note:raise ValueError('Reference Sheet readback mismatch')
    receipt={'action':'STAGE1_REFERENCE_VERIFIED','sheet_row':found[0]+6,'destination':dest,'readback_verified':True,
        'run_id':result['config']['run_id'],'step':result['checkpoint']['step'],'corrected_fr':result['fr_corrected_pan']['metrics']}
    atomic_json(folder/'publication.json',receipt);return receipt
