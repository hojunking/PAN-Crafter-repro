"""Bounded S1Loss9 target-banner migration; result and fixed64 header cells stay intact."""
from pathlib import Path
import time

from . import R

OLD_D1 = {
    'QB': '표시 기준: HQNR ≥ 0.923 AND ERGAS ≤ 3.500 · 같은 모델 상태의 FR20 + RR20',
    'GF2': '표시 기준: HQNR ≥ 0.965 AND ERGAS ≤ 0.522 · 같은 모델 상태의 FR20 + RR20',
}
OLD_A3 = '충족=통과한 모든 step / 대표(미달)=HQNR 최고 1개'
CELLS = {'D1':(0,3), 'A3':(2,0)}


def desired(server):
    sensor = R.DATASETS[server]
    h,e = R.thresholds(server)
    return {'D1':f'SENSOR5 기준: HQNR ≥ {h:.4f} AND ERGAS ≤ {e:.3f} · 같은 모델 상태의 FR20 + RR20',
            'A3':'SENSOR5: 통과 step 전부 + run별 대표(HQNR 충족 중 ERGAS 최소, 없으면 HQNR 최고)'}


def _read(adapter,destination):
    title=destination['sheet_title'].replace("'","''")
    result=adapter._request('GET',adapter.BASE+destination['spreadsheet_id'],params={
        'ranges':[f"'{title}'!{name}:{name}" for name in CELLS],
        'includeGridData':'true', 'fields':
        'spreadsheetId,sheets(properties(sheetId,title),data(startRow,startColumn,rowData(values(userEnteredValue,note,dataValidation,chipRuns,textFormatRuns))))'})
    if result.get('spreadsheetId')!=destination['spreadsheet_id']:
        raise ValueError('Target banner workbook differs')
    expected={rc:name for name,rc in CELLS.items()}; found={}
    for sheet in result.get('sheets',[]):
        props=sheet.get('properties',{})
        if props.get('sheetId')!=destination['sheet_id'] or props.get('title')!=destination['sheet_title']:
            if sheet.get('data'):
                raise ValueError('Target banner read returned data from a foreign tab')
            continue
        for block in sheet.get('data',[]):
            for y,row in enumerate(block.get('rowData',[]),block.get('startRow',0)):
                for x,cell in enumerate(row.get('values',[]),block.get('startColumn',0)):
                    name=expected.get((y,x))
                    if name is None or name in found:
                        raise ValueError('Target banner read exceeds the two registered cells')
                    entered=cell.get('userEnteredValue',{})
                    if set(entered)!={'stringValue'} or any(cell.get(k) for k in ('chipRuns','dataValidation','textFormatRuns')):
                        raise ValueError('Target banner must remain a plain literal; controls are preserved')
                    found[name]=cell
    if set(found)!=set(CELLS):
        raise ValueError('Missing registered target banner cells')
    return found


def reconcile_header(adapter,destination,state,server,*,apply=False):
    """Update only known old D1/A3 text, then verify both cells; never blind overwrite."""
    if destination!=R.destination(server):
        raise ValueError('S1Loss9 banner cannot target another server or WV3')
    before=_read(adapter,destination); target=desired(server)
    current={name:cell['userEnteredValue']['stringValue'] for name,cell in before.items()}
    legacy={'D1':OLD_D1[R.DATASETS[server]],'A3':OLD_A3}
    approved = {name:{legacy[name],target[name]} for name in CELLS}
    if R.DATASETS[server] == 'GF2':
        approved['D1'].add('SENSOR5 기준: HQNR ≥ 0.9645 AND ERGAS ≤ 0.548 · 같은 모델 상태의 FR20 + RR20')
    if any(current[name] not in approved[name] for name in CELLS):
        raise ValueError('Unrecognized target banner; foreign text was not overwritten')
    changes=[name for name in CELLS if current[name]!=target[name]]
    if not apply:
        return {'action':'TARGET_BANNER_DRY_RUN','changed_cells':changes,'readback_verified':False}
    state=Path(state); state.mkdir(parents=True,exist_ok=True)
    if changes:
        stamp=str(time.time_ns())
        R.atomic_json(state/('target_banner_backup_'+stamp+'.json'),{
            'destination':destination,'before':before,'desired':target})
        if _read(adapter,destination)!=before:
            raise ValueError('Target banner changed during preparation; no write made')
        requests=[]
        for name in changes:
            row,column=CELLS[name]
            requests.append({'updateCells':{'range':{'sheetId':destination['sheet_id'],
                'startRowIndex':row,'endRowIndex':row+1,'startColumnIndex':column,'endColumnIndex':column+1},
                'rows':[{'values':[{'userEnteredValue':{'stringValue':target[name]}}]}],
                'fields':'userEnteredValue'}})
        adapter._request('POST',adapter.BASE+destination['spreadsheet_id']+':batchUpdate',json={'requests':requests})
    after=_read(adapter,destination)
    if any(after[name]['userEnteredValue']['stringValue']!=target[name]
           or {k:v for k,v in after[name].items() if k!='userEnteredValue'}
              != {k:v for k,v in before[name].items() if k!='userEnteredValue'}
           for name in CELLS):
        raise ValueError('Target banner readback mismatch')
    receipt={'action':'S1LOSS9_TARGET_BANNER_VERIFIED','at':R.utc_now(),
        'destination':destination,'changed_cells':changes,'readback_verified':True,
        'values':target,'fixed64_header_and_result_rows_untouched':True}
    R.atomic_json(state/'target_banner_verified.json',receipt)
    return receipt
