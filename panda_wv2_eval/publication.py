"""Three owned Fixed64 rows on WV2-ablations; metrics only after full RR20/FR20."""
import json
from .common import ROOT, OUT, PLAN, digest, read, write, now
from directshift48.reporting import HEADERS, INDEX, METRICS, serialize_row
from sheet_publish_v2.transport import GoogleSheets, _typed, _padded, _intersects_data


def project():
    state = read(OUT/'state.json') if (OUT/'state.json').exists() else {}
    rows = []
    for order, case in enumerate(PLAN['cases'], 1):
        status = state.get('runs', {}).get(case['case_id'], {}).get('status', 'WAIT_WV3_QUEUE')
        path = OUT/'runs'/case['case_id']/'evaluation.json'
        report = None
        if path.exists():
            from .evaluate import validate_report
            report = validate_report(read(path), case)
            status = 'COMPLETED'
        fields = {'Experiment':case['experiment'], 'Server':'s1',
            'Selection@Step':f'WV3 고정 {case["source_step"]:,} · '+('WV2 평가 완료' if report else 'WV3 큐 완료 후 평가 대기'),
            'Dataset':'WV2', 'Bucket':'ablations', 'Campaign_ID':PLAN['campaign_id'],
            'Run_ID':'WV2ZS_'+case['case_id'], 'Case_ID':case['case_id'],
            'Repeat':case['source_repeat'], 'Seed':case['source_seed'], 'Attempt':1,
            'Role':'Frozen WV3 to WV2 zero-shot evaluation',
            'Approach':f'WV3→WV2 zero-shot; W{case["width"]}/D123/UBIL; L4; scale1; DIFF; source LR={case["source_lr"]}, a={case["source_alpha"]}, g={case["source_gamma"]}; beta0',
            'Model / inputs':f'Frozen PAN Aligner + W{case["width"]} Student; corrected PAN + difference + interp23 LRMS',
            'Teacher / reference':'No Teacher at inference; WV3-frozen A/S; no WV2 adaptation',
            'Parent_run':case['source_run_id'], 'Updates':0, 'Lifetime_updates':case['source_step'],
            'Selected_step':case['source_step'], 'Checkpoint_SHA':case['checkpoint_sha256'],
            'Selector':'FIXED_ON_WV3_BEFORE_WV2', 'Status':status, 'Readback':'OWN_ROW_VERIFIED',
            'Eval_scope':'WV2 native RR20+FR20; WV2 MTF; same frozen checkpoint', 'Test_aware':True,
            'Source_sheet':'WV3-main', 'Source_URL':'https://docs.google.com/spreadsheets/d/'+PLAN['destination']['spreadsheet_id']+'/edit',
            'Result_ID':'WV2ZS-'+digest([PLAN['campaign_id'],case['case_id'],case['checkpoint_sha256']]),
            'Review':'WV3 test-aware source selection; WV2 no tuning/selection; all 3 fixed models reported',
            'Sort_case':order, 'Date':PLAN['created_at']}
        if report:
            for split in ('rr','fr'):
                for key, value in report[split]['metrics'].items(): fields[METRICS[key]] = round(value,4)
            for key in ('Infer(ms)','Mem(MB)','Params(M)','FLOPs(G)'):
                fields[key] = round(report['cost'][key],4)
            fields.update({'Train(h)':0., 'Eval(h)':round(report['elapsed_seconds']/3600,4),
                           'Cost_scope':report['cost']['Cost_scope'], 'JQM_variant':report['JQM_variant']})
        note = {'campaign_id':PLAN['campaign_id'], 'source':case, 'target_dataset':'WV2',
                'training_updates':0, 'method':PLAN['method'], 'data':PLAN['data'],
                'dependency':PLAN['dependency'], 'raw_metrics':None if report is None else
                {s:report[s]['metrics'] for s in ('rr','fr')}, 'goal_filter':None,
                'all_three_models_reported':True, 'precision':4,
                'result_path':str(path.relative_to(ROOT)) if report else None}
        fields['Original description / notes'] = json.dumps(note,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        rows.append(serialize_row(fields))
    return rows


def requests(snapshot, rows):
    dest = PLAN['destination'];meta=snapshot['metadata']
    tabs=[s for s in meta['sheets'] if s['properties']['sheetId']==dest['sheet_id']]
    if (meta['spreadsheetId']!=dest['spreadsheet_id'] or len(tabs)!=1
            or tabs[0]['properties']['title']!=dest['sheet_title'] or tuple(snapshot['header'])!=HEADERS):
        raise ValueError('WV2 fixed64 destination/header mismatch')
    result=[];locations=[];occupied={i for i,r in enumerate(snapshot['rows'],6) if any(x!='' for x in r)}
    for desired in rows:
        rid=desired[INDEX['Result_ID']]
        matches=[i for i,r in enumerate(snapshot['rows'],6) if _padded(r)[INDEX['Result_ID']]==rid]
        if len(matches)>1:raise ValueError('Duplicate WV2 result identity')
        if matches: number=matches[0]
        else:
            number=6
            while number in occupied:number+=1
        occupied.add(number);locations.append(number)
        if number>snapshot['grid_row_count']:raise ValueError('WV2 Sheet grid exhausted')
        old=_padded(snapshot['rows'][number-6]) if number-6<len(snapshot['rows']) else ['']*64
        cells=snapshot['cells'][number-6] if number-6<len(snapshot['cells']) else []
        if matches and (old[INDEX['Campaign_ID']]!=PLAN['campaign_id'] or old[INDEX['Checkpoint_SHA']]!=desired[INDEX['Checkpoint_SHA']]):
            raise ValueError('WV2 owned row provenance changed')
        for cell in cells:
            if ('formulaValue' in cell.get('userEnteredValue',{}) or any(cell.get(k) for k in ('dataValidation','chipRuns','textFormatRuns'))):
                raise ValueError('Native controls/formulas in intended WV2 row')
            if not matches and cell.get('note'):raise ValueError('Unowned note-only row')
        for block in tabs[0].get('merges',[])+[p['range'] for p in tabs[0].get('protectedRanges',[]) if 'range' in p]:
            if block.get('startRowIndex',0)<=number-1<block.get('endRowIndex',number):
                raise ValueError('Protected/merged intended WV2 row')
        values=[]
        for column,v in enumerate(desired):
            pattern='0.0000' if 3<=column<25 else ('0.00000' if column in (48,49) else '0')
            cell={'userEnteredValue':_typed(v),'userEnteredFormat':{'numberFormat':{'type':'NUMBER','pattern':pattern}}}
            if column==0:cell['note']=desired[INDEX['Original description / notes']]
            values.append(cell)
        result.append({'updateCells':{'range':{'sheetId':dest['sheet_id'],'startRowIndex':number-1,
            'endRowIndex':number,'startColumnIndex':0,'endColumnIndex':64},'rows':[{'values':values}],
            'fields':'userEnteredValue,userEnteredFormat.numberFormat,note'}})
    return result,locations


def publish(*,apply=False):
    adapter=GoogleSheets(ROOT/'gspread/account.json')
    rows=project();snapshot=adapter.snapshot(PLAN['destination'])
    edits,locations=requests(snapshot,rows)
    if apply:
        write(OUT/'publication'/('before_'+now().replace(':','-')+'.json'),snapshot)
        adapter._request('POST',adapter.BASE+PLAN['destination']['spreadsheet_id']+':batchUpdate',json={'requests':edits})
        fresh=adapter.snapshot(PLAN['destination'])
        for number,row in zip(locations,rows):
            if fresh['rows'][number-6]!=row or fresh['notes'][number-6]!=row[INDEX['Original description / notes']]:
                raise ValueError('WV2 publication readback mismatch')
    receipt={'at':now(),'rows':locations,'readback_verified':apply,'destination':PLAN['destination']}
    write(OUT/'publication/last_poll.json',receipt)
    return receipt
