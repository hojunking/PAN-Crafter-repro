from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ta2.evaluation import file_sha
from panda_wv3_main_rgb import evaluation, plan, publication as P
from panda_wv3_main_rgb.transport import prepare, _verify_append_response


def simple(step,h,e,**extra):
    return dict(step=step,fr={'metrics':{'HQNR':h}},rr={'metrics':{'ERGAS':e}},**extra)


def student_config():
    return next(c for c in plan.queue('s1') if c['stage']=='stage2')


def persist(root,step,h,e):
    c=student_config();d=Path(root)/'runs'/c['run_id'];d.mkdir(parents=True,exist_ok=True)
    ck=d/f'synthetic_{step}.pt';ck.write_bytes(f'SYNTHETIC_ONLY_{step}'.encode())
    record={'schema':'WV3_MAIN_RGB_EVAL_V1','run_id':c['run_id'],'config_sha256':c['config_sha256'],
        'step':step,'stage':'stage2','same_state_verified':True,'reference_warp':False,'inference_tta':False,
        'model_sha256':'a'*64,'eval_protocol_id':'b'*64,'recorded_at':'2026-11-03T00:00:00+00:00',
        'jqm_variant':'SYNTHETIC','elapsed_seconds':2,
        'donor_selection':{'donor_run_id':c['donor_run_id'],'step':49000,
            'checkpoint_sha256':'c'*64,'validation':{'RRE50':.12,'RRE90':.25,'Val_ERGAS':2.04},
            'reference':'RGB_GT_MEAN'},
        'checkpoint':{'path':str(ck),'file_sha256':file_sha(ck),'model_sha256':'a'*64,'step':step}}
    rre={'registration':{'valid':True,'shift':[.1,.2]},'rre_px':float(np.linalg.norm([.1,.2]))}
    for split,values in [('rr',{'CC':.99,'ERGAS':e,'PSNR':38.,'Q8':.92,'RMSE':26.,'SAM':2.7,'SCC':.98,'SSIM':.97}),
                         ('fr',{'HQNR':h,'D_s':.03,'D_lambda':.01,'JQM':.97})]:
        record[split]={'per_scene':[{'scene_id':f'WV3:{split}:{i}','metrics':values,
                                    'rre_rgb':deepcopy(rre)} for i in range(20)],
                       'metrics':{k:float(np.mean([v]*20)) for k,v in values.items()}}
    record['test_rre_rgb']={s:evaluation.rre_summary(record[s]['per_scene']) for s in ('rr','fr')}
    (d/f'evaluation_{step:07d}.json').write_text(json.dumps(record))
    return record


def snapshot(value,width=70):
    dest=P.destination()
    return {'metadata':{'spreadsheetId':dest['spreadsheet_id'],'sheets':[{'properties':{
        'sheetId':dest['sheet_id'],'title':dest['sheet_title'],'gridProperties':{'columnCount':width,'rowCount':3000}}}]},
        'header':list(P.HEADERS),'rows':deepcopy(value['rows']),'notes':deepcopy(value['notes']),
        'cells':[],'grid_row_count':3000}


class PublicationTests(unittest.TestCase):
    def test_inclusive_targets_representative_and_duplicate_checkpoint(self):
        rows=[simple(1000,.9585,2.040),simple(2000,.9586,2.039),simple(3000,.958499,1.99)]
        out=P.choose(rows+[deepcopy(rows[1])],'stage2')
        self.assertEqual([(r['step'],s) for r,s,_ in out],
                         [(2000,'HQNR_QUALIFIED_MIN_ERGAS'),(1000,'PASS_ALL_STEPS')])
        out=P.choose([simple(1000,.9585,2.1),simple(2000,.957,1.8)],'stage2')
        self.assertEqual((out[0][0]['step'],out[0][1]),(1000,'HQNR_QUALIFIED_MIN_ERGAS'))
        out=P.choose([simple(1000,.94,1.8),simple(2000,.95,2.4)],'stage2')
        self.assertEqual((out[0][0]['step'],out[0][1]),(2000,'BEST_HQNR_FALLBACK'))
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            P.choose([simple(1000,.96,2.03),simple(1000,.97,2.03)],'stage2')

    def test_stage1_filtered_and_all_eighteen_main_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=P.project(tmp,'s1')
            self.assertEqual(len(plan.queue('s1')),21)
            self.assertEqual(len(value['rows']),18)
            self.assertEqual(len({r[P.INDEX['Experiment']] for r in value['rows']}),18)
            self.assertTrue(all(c['stage']=='stage2' for c in value['configs'].values()))
            for row in value['rows']:
                self.assertEqual(len(row),64)
                self.assertTrue(row[P.INDEX['Experiment']].endswith('(RGB)'))
                self.assertEqual(row[P.INDEX['Bucket']],'main')
                self.assertIn('RGB full Stage1',row[P.INDEX['Approach']])
            self.assertEqual(P.choose([simple(1000,.99,1.)],'stage1'),[])
            stage1=next(c for c in plan.queue('s1') if c['stage']=='stage1')
            with self.assertRaisesRegex(ValueError,'Stage2'):P.make_row(stage1,'QUEUED')

    def test_status_row_becomes_representative_without_creating_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=P.project(tmp,'s1');snap=snapshot(first);c=student_config()
            old=next(r for r in first['rows'] if r[P.INDEX['Run_ID']]==c['run_id'])
            record=persist(tmp,1000,.96,2.03);new=P.project(tmp,'s1')
            rep=next(r for r in new['rows'] if r[P.INDEX['Run_ID']]==c['run_id'])
            self.assertEqual(rep[P.INDEX['Result_ID']],old[P.INDEX['Result_ID']])
            requests,adds=prepare(snap,new,'s1')
            self.assertEqual(adds,[]);self.assertEqual(len(requests),1)
            payload=requests[0]['updateCells']
            self.assertEqual(payload['range']['endColumnIndex'],64)
            self.assertTrue(payload['rows'][0]['values'][0]['userEnteredValue'])
            note=json.loads(rep[P.INDEX['Original description / notes']])
            self.assertEqual(note['donor_selection'],P.donor_note(c,record))

    def test_idempotence_foreign_rows_and_extra_columns_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03);value=P.project(tmp,'s1');snap=snapshot(value)
            foreign=deepcopy(snap['rows'][0]);foreign[P.INDEX['Campaign_ID']]='FOREIGN'
            foreign[P.INDEX['Run_ID']]='FOREIGN-RUN';foreign[P.INDEX['Result_ID']]='FOREIGN-ID'
            snap['rows'].insert(0,foreign);snap['notes'].insert(0,'foreign note')
            before=deepcopy(snap)
            self.assertEqual(prepare(snap,value,'s1'),([],[]))
            self.assertEqual(snap,before)
            persist(tmp,2000,.961,2.02);new=P.project(tmp,'s1')
            requests,adds=prepare(snap,new,'s1')
            self.assertEqual(len(adds),1)
            for req in requests:
                if 'updateCells' in req:
                    bounds=req['updateCells']['range']
                    self.assertGreater(bounds['startRowIndex'],5)
                    self.assertEqual(bounds['endColumnIndex'],64)

    def test_representative_change_preserves_passing_checkpoint_and_donor_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03);first=P.project(tmp,'s1');snap=snapshot(first)
            old=next(r for r in first['rows'] if r[P.INDEX['Selector']]=='HQNR_QUALIFIED_MIN_ERGAS')
            persist(tmp,2000,.961,2.02);new=P.project(tmp,'s1')
            rep=next(r for r in new['rows'] if r[P.INDEX['Selector']]=='HQNR_QUALIFIED_MIN_ERGAS')
            self.assertEqual(old[P.INDEX['Result_ID']],rep[P.INDEX['Result_ID']])
            self.assertEqual(rep[P.INDEX['Selected_step']],2000)
            self.assertEqual([r[P.INDEX['Selected_step']] for r in new['rows']
                              if r[P.INDEX['Selector']]=='PASS_ALL_STEPS'],[1000])
            tampered=deepcopy(snap)
            note=json.loads(tampered['rows'][0][P.INDEX['Original description / notes']])
            note['donor_selection']['checkpoint_sha256']='d'*64
            tampered['rows'][0][P.INDEX['Original description / notes']]=json.dumps(note)
            with self.assertRaisesRegex(ValueError,'donor evidence'):prepare(tampered,new,'s1')

    def test_incomplete_metrics_missing_weights_and_donor_evidence_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03)
            path=next(Path(tmp).rglob('evaluation_*.json'));record=json.loads(path.read_text())
            record['rr']['per_scene'].pop();path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Incomplete'):P.project(tmp,'s1')
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03);next(Path(tmp).rglob('*.pt')).unlink()
            with self.assertRaises(FileNotFoundError):P.project(tmp,'s1')
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03)
            path=next(Path(tmp).rglob('evaluation_*.json'));record=json.loads(path.read_text())
            del record['donor_selection'];path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'donor'):P.project(tmp,'s1')

    def test_append_range_is_main_and_never_shifted(self):
        dest=P.destination();self.assertEqual(dest['sheet_id'],260928010)
        response={'updates':{'updatedRange':"'WV3-main'!A7:BL7",'updatedColumns':64,'updatedRows':1,'updatedCells':64}}
        self.assertEqual(_verify_append_response(response,dest,1,1),"'WV3-main'!A7:BL7")
        response['updates']['updatedRange']="'WV3-main'!T7:CE7"
        with self.assertRaisesRegex(Exception,'append bounds'):_verify_append_response(response,dest,1,1)

    def test_donor_candidates_remain_in_evidence_and_sheet_note_stays_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=persist(tmp,1000,.960,2.03);c=student_config()
            record['donor_selection']['schema']='WV3_MAIN_RGB_DONOR_SELECTION_V1'
            record['donor_selection']['candidates']=[{'diagnostic':'x'*2000} for _ in range(50)]
            record['donor_selection']['receipt_sha256']=plan.digest(record['donor_selection'])
            path=next(Path(tmp).rglob('evaluation_*.json'));path.write_text(json.dumps(record))
            value=P.project(tmp,'s1');note=json.loads(value['notes'][0])
            self.assertEqual(note['donor_selection']['candidate_count'],50)
            self.assertNotIn('candidates',note['donor_selection'])
            self.assertLess(len(value['notes'][0]),50000)
            self.assertEqual(len(json.loads(path.read_text())['donor_selection']['candidates']),50)
            record['donor_selection']['donor_run_id']='ANOTHER-REPEAT'
            with self.assertRaisesRegex(ValueError,'another Stage1'):P.donor_note(c,record)

    def test_main_snapshot_preserves_strict_native_cell_and_header_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=P.project(tmp,'s1');snap=snapshot(value)
            bad=deepcopy(snap);bad['header'][0]='WRONG'
            with self.assertRaisesRegex(Exception,'header digest'):prepare(bad,value,'s1')
            bad=deepcopy(snap);bad['cells']=[[{'userEnteredValue':{'formulaValue':'=1+1'}}]]
            with self.assertRaisesRegex(Exception,'Formula'):prepare(bad,value,'s1')
            bad=deepcopy(snap);bad['metadata']['sheets'][0]['properties']['title']='WV3-ablations'
            with self.assertRaisesRegex(Exception,'ID/title'):prepare(bad,value,'s1')


if __name__=='__main__':unittest.main()
