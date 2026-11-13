import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from panda_wv3_tab3_rgbn import plan, publication, aggregate
from panda_wv3_tab3_rgbn.enqueue import validate_old_container
from panda_wv3_tab3_rgbn.matching import ScharrAlignment
from panda_unified.core import warp_pan, support, ReferenceSpatial


class CampaignTests(unittest.TestCase):
    def test_round_major_paired_design_and_endpoint(self):
        configs = plan.queue('s3')
        self.assertEqual(len(configs),15)
        self.assertEqual([c['repeat'] for c in configs],[1]*5+[2]*5+[3]*5)
        self.assertEqual(len({c['run_id'] for c in configs}),15)
        for r in (1,2,3):
            block=[c for c in configs if c['repeat']==r]
            self.assertTrue(all(c['stream_seeds']==block[0]['stream_seeds'] for c in block))
        self.assertEqual([(c['lambda_abs'],c['lambda_epsilon']) for c in configs[:5]],
                         [(0,0),(0,0),(2,0),(0,.003),(2,.003)])
        for c in configs:
            self.assertEqual(c['evaluation_steps'],[30000,40000,50000])
            self.assertEqual(c['validation_steps'],[30000,40000,50000])
            self.assertEqual(c['publication_steps'],[50000])
            self.assertEqual(c['reference_bands'],[4,2,1,6])
            self.assertEqual(c['stage'],'stage1')
            self.assertFalse(c['early_stop']['enabled'])
        with self.assertRaises(ValueError): plan.queue('s1')

    def test_sheet_has_no_placeholders_or_intermediate_rows(self):
        cfg=plan.resolve('s3','OURS',1)
        self.assertEqual(publication.choose([], 'stage1',cfg),[])
        self.assertEqual(publication.choose([{'step':30000},{'step':40000}], 'stage1',cfg),[])
        self.assertEqual(publication.choose([{'step':30000},{'step':50000}], 'stage1',cfg),
                         [({'step':50000},'FIXED50K','ROW0')])
        with tempfile.TemporaryDirectory() as folder:
            view=publication.project(folder,'s3')
            self.assertEqual(view['rows'],[])
            self.assertEqual(view['destination']['sheet_title'],'WV3-ablations')
        with self.assertRaises(ValueError): publication.make_row(cfg,'RUNNING',None)

    def test_incomplete_repeats_are_not_selected_subsets(self):
        with tempfile.TemporaryDirectory() as folder:
            result=aggregate.aggregate(folder,'s3',write=True)
            for case in result['cases']:
                self.assertEqual(case['n_runs'],0)
                self.assertIsNone(case['metrics']['RR_ERGAS']['mean'])
            cfg=plan.resolve('s3','OURS',1)
            d=Path(folder)/'runs'/cfg['run_id'];d.mkdir(parents=True)
            record={'step':50000,'model_sha256':'x','checkpoint':{'step':50000,'model_sha256':'x'},
                'rr':{'metrics':{'ERGAS':2.,'SAM':3.,'PSNR':38.}},
                'fr':{'metrics':{'HQNR':.96,'D_s':.03,'D_lambda':.01}},
                'test_rre':{'rr':{'RRE50':.2},'fr':{'RRE50':.5}}}
            (d/'evaluation_0050000.json').write_text(json.dumps(record))
            with patch.object(aggregate,'verify_record'),patch.object(aggregate,'artifact'):
                one=aggregate.aggregate(folder,'s3')['cases'][-1]
                self.assertEqual(one['n_runs'],1)
                self.assertIsNone(one['metrics']['RR_ERGAS']['mean'])

    def test_cutover_cannot_stop_s1_or_another_checkout(self):
        info={'Config':{'Labels':{'panda.server':'s3'},'Cmd':['python','-m','old','--server','s3']},
              'Mounts':[{'Destination':'/workspace','Source':str(plan.ROOT)}]}
        validate_old_container(info,'s3')
        bad=copy.deepcopy(info);bad['Config']['Labels']['panda.server']='s1'
        with self.assertRaises(ValueError): validate_old_container(bad,'s3')
        bad=copy.deepcopy(info);bad['Mounts'][0]['Source']='/some/other/repo'
        with self.assertRaises(ValueError): validate_old_container(bad,'s3')
        bad=copy.deepcopy(info);bad['Config']={'Labels':{},'Cmd':['python','train.py']}
        with self.assertRaises(ValueError): validate_old_container(bad,'s3')
        bad=copy.deepcopy(info);bad['Config']['Labels']['panda.campaign']=plan.CAMPAIGN_ID
        with self.assertRaises(ValueError): validate_old_container(bad,'s3')


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_integer_translation_sign_and_exact_cache(self):
        generator=torch.Generator().manual_seed(73)
        pan=torch.rand(1,1,32,32,generator=generator)
        delta=torch.tensor([[1.,-2.]])
        lms=warp_pan(pan,delta).repeat(1,8,1,1)
        net=ScharrAlignment(candidate_chunk=8)
        estimate=net(pan,lms)
        torch.testing.assert_close(estimate,delta,rtol=0,atol=0)
        cached=net(pan,lms)
        torch.testing.assert_close(cached,estimate,rtol=0,atol=0)
        net._cache.clear()
        torch.testing.assert_close(net(pan,lms),estimate,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in net.parameters()),0)

    def test_rgbn_reference_and_mask_are_shared(self):
        image=torch.zeros(1,8,32,32)
        image[:,6]=4
        self.assertTrue(torch.equal(ReferenceSpatial('WV3','RGBN').reference(image),torch.ones(1,1,32,32)))
        pan=torch.zeros(1,1,64,64)
        self.assertEqual(int(support(pan,torch.tensor([[.5,.5]])).sum()),55*55)
        # Near-integer shifts must use exact integer tap indexing, not floor(x+u) in FP32.
        self.assertEqual(int(support(pan,torch.tensor([[-1e-6,-1e-6]])).sum()),55*55)


if __name__=='__main__':unittest.main()
