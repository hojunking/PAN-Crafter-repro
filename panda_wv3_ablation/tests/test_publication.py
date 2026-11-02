from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ta2.evaluation import file_sha
from panda_wv3_ablation import plan,publication as P
from panda_wv3_ablation.transport import prepare,_verify_append_response


def simple(step,h,e,**extra):
    return dict(step=step,fr={'metrics':{'HQNR':h}},rr={'metrics':{'ERGAS':e}},**extra)


def persist(root,step,h,e):
    c=plan.queue('s1')[0];d=Path(root)/'runs'/c['run_id'];d.mkdir(parents=True,exist_ok=True)
    ck=d/f'synthetic_{step}.pt';ck.write_bytes(f'SYNTHETIC_ONLY_{step}'.encode())
    record={'schema':'WV3_ABLATION_EVAL_V1','run_id':c['run_id'],'config_sha256':c['config_sha256'],
        'step':step,'stage':'stage2','same_state_verified':True,'reference_warp':False,'inference_tta':False,
        'model_sha256':'a'*64,'eval_protocol_id':'b'*64,'recorded_at':'2026-11-02T00:00:00+00:00',
        'jqm_variant':'SYNTHETIC','elapsed_seconds':2,
        'checkpoint':{'path':str(ck),'file_sha256':file_sha(ck),'model_sha256':'a'*64,'step':step}}
    for split,values in [('rr',{'CC':.99,'ERGAS':e,'PSNR':38.,'Q8':.92,'RMSE':26.,'SAM':2.7,'SCC':.98,'SSIM':.97}),
                         ('fr',{'HQNR':h,'D_s':.03,'D_lambda':.01,'JQM':.97})]:
        record[split]={'per_scene':[{'scene_id':f'WV3:{split}:{i}','metrics':values} for i in range(20)],
                       'metrics':{k:float(np.mean([v]*20)) for k,v in values.items()}}
    (d/f'evaluation_{step:07d}.json').write_text(json.dumps(record))


def snapshot(value,width=70):
    dest=P.destination()
    return {'metadata':{'spreadsheetId':dest['spreadsheet_id'],'sheets':[{'properties':{
        'sheetId':dest['sheet_id'],'title':dest['sheet_title'],'gridProperties':{'columnCount':width,'rowCount':3000}}}]},
        'header':list(P.HEADERS),'rows':deepcopy(value['rows']),'notes':deepcopy(value['notes']),
        'cells':[],'grid_row_count':3000}


class PublicationTests(unittest.TestCase):
    def test_exact_targets_and_representative_without_ergas_gate(self):
        rows=[simple(1000,.9585,2.040),simple(2000,.9586,2.039),simple(3000,.958499,1.99)]
        out=P.choose(rows,'stage2')
        self.assertEqual([(r['step'],s) for r,s,_ in out],[(2000,'HQNR_QUALIFIED_MIN_ERGAS'),(1000,'PASS_ALL_STEPS')])
        out=P.choose([simple(1000,.9585,2.1),simple(2000,.957,1.8)],'stage2')
        self.assertEqual(out[0][0]['step'],1000)
        self.assertEqual(out[0][1],'HQNR_QUALIFIED_MIN_ERGAS')
        out=P.choose([simple(1000,.94,1.8),simple(2000,.95,2.4)],'stage2')
        self.assertEqual((out[0][0]['step'],out[0][1]),(2000,'BEST_HQNR_FALLBACK'))

    def test_stage1_uses_validation_not_test_hqnr(self):
        a=simple(1000,.99,1.,validation={'RRE50':.2,'RRE90':.3,'Val_ERGAS':2.})
        b=simple(2000,.90,3.,validation={'RRE50':.1,'RRE90':.5,'Val_ERGAS':3.})
        self.assertEqual(P.choose([a,b],'stage1')[0][0]['step'],2000)

    def test_experiment_names_fixed64_foreign_rows_and_wide_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=P.project(tmp,'s1');foreign=P.project(tmp,'s3')
            self.assertEqual(len(value['rows']),27)
            self.assertEqual(len({r[P.INDEX['Experiment']] for r in value['rows']}),27)
            self.assertTrue(all(len(r)==64 for r in value['rows']))
            snap=snapshot(value)
            snap['rows'].insert(0,foreign['rows'][0]);snap['notes'].insert(0,foreign['notes'][0])
            requests,adds=prepare(snap,value,'s1')
            self.assertEqual((requests,adds),([],[]))

    def test_representative_replacement_preserves_old_pass_and_own_row_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03)
            first=P.project(tmp,'s1');snap=snapshot(first)
            old_rep=next(r for r in first['rows'] if r[P.INDEX['Selector']]=='HQNR_QUALIFIED_MIN_ERGAS')
            persist(tmp,2000,.961,2.02)
            new=P.project(tmp,'s1')
            rep=next(r for r in new['rows'] if r[P.INDEX['Selector']]=='HQNR_QUALIFIED_MIN_ERGAS')
            self.assertEqual(old_rep[P.INDEX['Result_ID']],rep[P.INDEX['Result_ID']])
            self.assertEqual(rep[P.INDEX['Selected_step']],2000)
            passed=[r for r in new['rows'] if r[P.INDEX['Selector']]=='PASS_ALL_STEPS']
            self.assertEqual([r[P.INDEX['Selected_step']] for r in passed],[1000])
            req,adds=prepare(snap,new,'s1');self.assertEqual(len(adds),1)
            self.assertTrue(any('updateCells' in r for r in req))
            tampered=deepcopy(snap);tampered['rows'][0][P.INDEX['HQNR↑']]=.999
            with self.assertRaisesRegex(ValueError,'evidence'):prepare(tampered,new,'s1')

    def test_incomplete_metrics_or_missing_weights_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03)
            path=next(Path(tmp).rglob('evaluation_*.json'));record=json.loads(path.read_text())
            record['rr']['per_scene'].pop();path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Incomplete'):P.project(tmp,'s1')
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,1000,.960,2.03);next(Path(tmp).rglob('*.pt')).unlink()
            with self.assertRaises(FileNotFoundError):P.project(tmp,'s1')

    def test_append_cannot_shift_to_another_subtable(self):
        dest=P.destination()
        response={'updates':{'updatedRange':"'WV3-ablations'!A7:BL7",'updatedColumns':64,'updatedRows':1,'updatedCells':64}}
        self.assertEqual(_verify_append_response(response,dest,1,1),"'WV3-ablations'!A7:BL7")
        response['updates']['updatedRange']="'WV3-ablations'!T7:CE7"
        with self.assertRaisesRegex(Exception,'append bounds'):_verify_append_response(response,dest,1,1)
