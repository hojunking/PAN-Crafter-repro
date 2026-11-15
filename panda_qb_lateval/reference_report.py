"""Publish the actual late-validation donor Teacher in the existing FR67 layout."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from directshift48.reporting import atomic_json,utc_now
from ta2.evaluation import file_sha
from panda_stage1_reference.report import CAMPAIGN,row_for,row_equal,validate_corrected,source_model
from . import plan,assets
ROOT=plan.ROOT


def prepare(server,device='cpu'):
    folder=plan.campaign_root(server)/'stage1_reference';path=folder/'evaluation.json'
    receipt=assets.read(plan.campaign_root(server)/'donor/selection.json')
    cfg=plan.donor_config(server);root=plan.donor_root(server);chosen=receipt['selected']
    directory=root/'runs'/cfg['run_id'];ev=directory/f"evaluation_{chosen['step']:07d}.json"
    record=assets.read(ev);ck=root/chosen['checkpoint_path'];meta=assets.read(ck.with_suffix('.json'))
    if file_sha(ck)!=chosen['file_sha256'] or file_sha(ev)!=chosen['evaluation_sha256']:
        raise ValueError('Selected Teacher report source changed')
    if path.exists():
        cached=assets.read(path)
        if cached['donor_selection_sha256']!=receipt['receipt_sha256']:
            raise ValueError('Teacher report donor changed')
        validate_corrected(record,cached['fr_corrected_pan'],'QB');return cached
    result={'schema':'PANDA_SELECTED_STAGE1_TEACHER_DUAL_FR_V1','recorded_at':utc_now(),
        'config':cfg,'checkpoint':meta,'model_sha256':record['model_sha256'],
        'source_evaluation':str(ev),'source_evaluation_sha256':file_sha(ev),
        'evaluated_role':'SAME_CHECKPOINT_ALIGNER_AND_TEACHER','rr':record['rr'],'fr':record['fr'],
        'no_new_training':True,'donor_selection_sha256':receipt['receipt_sha256'],
        'donor_selector':plan.SELECTOR,'donor_selection_uses_test':False,'inference_tta':False}
    if 'fr_corrected_pan' in record:
        corrected=record['fr_corrected_pan'];result['measurement']='VERIFIED_EXISTING_DUAL_FR'
    else:
        from directshift48.evaluation import metric_functions_for,metric_values,expected_ids
        from .evaluation import dual_fr_metrics
        net,data,pins,training=source_model(cfg,root,ck,meta);net.to(device)
        fn=metric_functions_for(data,pins['evaluator'])['fr'];rows=[];deltas=[]
        with torch.inference_mode():
            for i,sid in enumerate(expected_ids(data,'fr')):
                batch=data.batch('fr',[i],device=device);out=net.native_prediction(batch)
                raw=((out['y'][0]+1)*(data.peak/2)).cpu().numpy()
                pan=((out['pan_aligned'][0]+1)*(data.peak/2)).cpu().numpy()
                arrays={k:v[0] for k,v in data.raw('fr',[i]).items()}
                original=metric_values(fn(raw,arrays),'fr',4)
                deltas.extend(abs(original[k]-record['fr']['per_scene'][i]['metrics'][k]) for k in original)
                rows.append({'scene_id':sid,'metrics':dual_fr_metrics(fn,raw,arrays,pan,original,4)})
                print(f'Selected Teacher FR {i+1}/20',flush=True)
        if max(deltas)>1e-4 or training.model_hash(net)!=meta['model_sha256']:
            raise ValueError('Selected Teacher native FR reproduction differs')
        corrected={'per_scene':rows,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
        result.update(measurement='NATIVE_FR20_SAME_TEACHER',max_original_FR_difference=max(deltas))
    validate_corrected(record,corrected,'QB');result['fr_corrected_pan']=corrected
    atomic_json(path,result);return result


def publish_source(server,*,apply=False):
    sensor='QB'
    from panda_fr_columns.adapter import GoogleSheets
    from panda_fr_columns.schema import BOOK,HEADERS67
    from sheet_publish_v2.transport import _typed,_matrix_equal
    folder=plan.campaign_root(server)/'stage1_reference'
    result=json.loads((folder/'evaluation.json').read_text());row,note=row_for(result,server)
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
