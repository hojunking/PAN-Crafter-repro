"""Synthetic evidence tests; no test metrics are written outside temporary folders."""
import json
from pathlib import Path
import statistics
import tempfile
import unittest
import numpy as np
from ta2.evaluation import file_sha
from panda_wv3_tab3_rgbn import plan, publication, evaluation, aggregate


def record(folder, cfg, step, ergas):
    directory=Path(folder)/'runs'/cfg['run_id'];directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/f'synthetic_{step}.pt';checkpoint.write_bytes(b'synthetic-test-evidence')
    value={'schema':'PANDA_TAB3_RGBN_EVAL_V1','run_id':cfg['run_id'],
        'config_sha256':cfg['config_sha256'],'step':step,'same_state_verified':True,
        'reference_warp':False,'inference_tta':False,'edge_reference_bands':[4,2,1,6],
        'edge_reference':'RGBN_ARITHMETIC_MEAN','pan_warp':'LANCZOS4_REFLECTION_ALL_PATHS',
        'rre_margin':{'rr':19,'fr':24},'model_sha256':'test-state',
        'checkpoint':{'path':str(checkpoint),'file_sha256':file_sha(checkpoint),'step':step,'model_sha256':'test-state'},
        'recorded_at':'2026-11-13T00:00:00Z','elapsed_seconds':1.,'jqm_variant':'synthetic-fixture'}
    for split,metrics in (
        ('rr',dict(CC=.99,ERGAS=ergas,PSNR=38.,Q8=.92,RMSE=20.,SAM=2.76,SCC=.988,SSIM=.98)),
        ('fr',dict(HQNR=.960123456,D_s=.028,D_lambda=.012,JQM=.97))):
        rows=[{'scene_id':f'WV3:{split}:{i}','metrics':metrics,
               'rre_reference':{'registration':{'valid':True,'shift':[.3,.4]},'rre_px':.5}} for i in range(20)]
        value[split]={'per_scene':rows,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in metrics}}
    value['test_rre']={s:evaluation.rre_summary(value[s]['per_scene'],cfg) for s in ('rr','fr')}
    evaluation.verify_record(value,cfg)
    (directory/f'evaluation_{step:07d}.json').write_text(json.dumps(value))
    return value


class ResultsTests(unittest.TestCase):
    def test_only_endpoint_publishes_with_provenance_and_no_foreign_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=plan.resolve('s3','OURS',1)
            record(folder,cfg,30000,2.01);record(folder,cfg,40000,2.02)
            self.assertEqual(publication.project(folder,'s3')['rows'],[])
            record(folder,cfg,50000,2.03)
            view=publication.project(folder,'s3');self.assertEqual(len(view['rows']),1)
            row=view['rows'][0];index=publication.INDEX
            self.assertEqual(row[index['Selected_step']],50000)
            self.assertEqual(row[index['Selector']],'FIXED50K')
            self.assertEqual(row[index[publication.METRICS['HQNR']]],.9601)
            note=json.loads(view['notes'][0])
            self.assertEqual(note['reference_repeats'],3)
            self.assertEqual(note['test_rre']['rr']['RRE50'],.5)
            foreign=['']*64;foreign[index['Campaign_ID']]='other-campaign'
            requests=publication.row_requests({'rows':[foreign],'notes':['']},view,'s3')
            self.assertEqual(len(requests),1);self.assertIn('appendCells',requests[0])
            requests=publication.row_requests({'rows':[foreign,row],'notes':['',view['notes'][0]]},view,'s3')
            self.assertEqual(requests,[])

    def test_three_repeat_sample_std_uses_all_results(self):
        with tempfile.TemporaryDirectory() as folder:
            values=[2.01,2.05,2.09]
            for repeat,value in enumerate(values,1):record(folder,plan.resolve('s3','OURS',repeat),50000,value)
            result=aggregate.aggregate(folder,'s3',write=True)['cases'][-1]
            actual=[r['values']['RR_ERGAS'] for r in result['runs']]
            self.assertEqual(result['n_runs'],3)
            self.assertEqual(result['metrics']['RR_ERGAS']['mean'],statistics.mean(actual))
            self.assertEqual(result['metrics']['RR_ERGAS']['std'],statistics.stdev(actual))
            self.assertEqual([r['repeat'] for r in result['runs']],[1,2,3])


if __name__=='__main__':unittest.main()
