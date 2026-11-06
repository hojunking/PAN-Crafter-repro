from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from panda_wv2_eval.common import PLAN, dependency_ready, digest
from panda_wv2_eval.data import Sources, register_sensor


class WV2Tests(unittest.TestCase):
    def test_dependency_waits_for_entire_queue(self):
        d=PLAN['dependency']
        state={'campaign_id':d['campaign_id'],'queue_sha256':d['queue_sha256'],'status':'RUNNING',
               'runs':[{'run_id':r,'status':'COMPLETED','step':30000} for r in d['run_ids']]}
        self.assertFalse(dependency_ready(state)[0])
        state['status']='COMPLETED';state['runs'][-1]['step']=29000
        self.assertFalse(dependency_ready(state)[0])
        state['runs'][-1]['step']=30000
        self.assertTrue(dependency_ready(state)[0])
        state['runs'].pop()
        with self.assertRaises(ValueError):dependency_ready(state)

    def test_inference_batch_is_wv2_and_cannot_read_rr_gt(self):
        register_sensor();s=Sources.__new__(Sources)
        arrays={'pan':np.full((1,1,64,64),100.),'ms':np.full((1,8,16,16),100.),
                'lms':np.full((1,8,64,64),100.),'gt':np.full((1,8,64,64),999.)}
        s.raw=lambda *_:arrays
        b=s.batch('rr',[0]);self.assertIsNone(b.gt)
        self.assertEqual(b.lineage.sensor,'WV2');self.assertEqual(b.lineage.sample_ids,('WV2:rr:0',))
        self.assertTrue(torch.allclose(b.pan,torch.full_like(b.pan,2*100/2047-1)))

    def test_all_three_frozen_models_run_synthetic_wv2_graphs(self):
        from panda_wv2_eval.model import Deployment
        register_sensor();torch.set_num_threads(1)
        s=Sources.__new__(Sources)
        rng=np.random.default_rng(481)
        arrays={'pan':rng.uniform(0,1000,(1,1,64,64)),'ms':rng.uniform(0,1000,(1,8,16,16)),
                'lms':rng.uniform(0,1000,(1,8,64,64))}
        s.raw=lambda *_:arrays;b=s.batch('rr',[0])
        for case in PLAN['cases']:
            model=Deployment(case);before=model.weight_hash()
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertEqual(model.student.spec.width,case['width'])
            with torch.no_grad():prediction=model.native_prediction(b)
            self.assertEqual(tuple(prediction['y'].shape),(1,8,64,64))
            self.assertTrue(torch.isfinite(prediction['y']).all())
            self.assertTrue(torch.equal(prediction['x_in'][:,1:2],prediction['pan_aligned']-b.pan))
            self.assertEqual(model.weight_hash(),before)

    def test_fr_metrics_use_wv2_sensor_not_wv3(self):
        register_sensor()
        from ta2.evaluation import fr_scene
        p=np.zeros((8,512,512),dtype=np.float32);pan=p[:1];ms=np.zeros((8,128,128))
        with patch('tools.metrics.eval_fr.d_lambda_k',return_value=.01) as dl, \
             patch('tools.metrics.eval_fr.d_s',return_value=.02), \
             patch('tools.metrics.jqm.jqm',return_value={'JQM':.8,'w_source':'test','w':[.125]*8}) as jq:
            result=fr_scene(p,pan,p,ms,'WV2',None)
        self.assertEqual(dl.call_args.args[2],'wv2');self.assertEqual(jq.call_args.args[3],'WV2')
        self.assertAlmostEqual(result['hqnr'],.99*.98)

    def test_pending_publication_has_no_fabricated_metrics_and_preserves_foreign_row(self):
        from panda_wv2_eval import publication as P
        with tempfile.TemporaryDirectory() as temp,patch.object(P,'OUT',Path(temp)):
            rows=P.project()
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r[P.INDEX['Dataset']]=='WV2' and r[P.INDEX['Updates']]==0 for r in rows))
        self.assertTrue(all(r[P.INDEX['HQNR↑']]=='' and r[P.INDEX['ERGAS↓']]=='' for r in rows))
        foreign=['']*64;foreign[0]='KEEP';foreign[P.INDEX['Result_ID']]='OTHER'
        dest=PLAN['destination']
        snapshot={'metadata':{'spreadsheetId':dest['spreadsheet_id'],'sheets':[{'properties':{
            'sheetId':dest['sheet_id'],'title':dest['sheet_title']}}]},'header':list(P.HEADERS),
            'rows':[foreign],'cells':[[]],'grid_row_count':3000}
        edits,locations=P.requests(snapshot,rows)
        self.assertEqual(locations,[7,8,9]);self.assertEqual(len(edits),3)
        self.assertEqual(snapshot['rows'][0],foreign)


if __name__=='__main__':unittest.main()
