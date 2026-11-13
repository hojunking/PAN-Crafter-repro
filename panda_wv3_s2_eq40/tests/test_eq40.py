import unittest,tempfile,json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from panda_unified import core
from panda_unified_campaign import plan as oldplan,model as oldmodel
from panda_unified_campaign.tests.test_integration import synthetic_source,Stats
from panda_wv3_s2_eq40 import plan,model,training,evaluation,publication
from panda_wv3_s2_eq40.assets import rank_candidates
from panda_wv3_s2_eq40.coordinator import remaining_stage1
from donor30.masking import masked_mean

class EQ40Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_queue_donors_and_cutover(self):
        q=plan.queue('s1')
        for r,c in enumerate(q,1):
            self.assertEqual((c['horizon'],c['alpha'],c['gamma'],c['beta']),(40000,0,.25,0))
            self.assertEqual(c['donor_run_id'],oldplan.resolve('s1',oldplan.STAGE1_CASE,r)['run_id'])
        with self.assertRaises(ValueError):plan.queue('s2')
        from panda_unified_campaign.runtime import initial_state
        state=initial_state('s1');state['runs'][0]['status']='COMPLETED'
        self.assertEqual(remaining_stage1(state)['run_id'],q[1]['donor_run_id'])
        state['runs'][2]['status']='COMPLETED'
        self.assertIsNone(remaining_stage1(state))
    def test_selection_gates_then_original_hqnr(self):
        rows=[dict(step=i,rr=dict(ERGAS=e,PSNR=p),fr=dict(HQNR=h)) for i,e,p,h in
              [(1000,2.05,39,.99),(2000,2.03,37.99,.98),(3000,2.04,38,.92),(4000,2.02,38.1,.91)]]
        self.assertEqual(rank_candidates(rows)['step'],3000)
        with self.assertRaises(ValueError):rank_candidates(rows[:2])
    def test_stem_and_shared_initialization(self):
        cfg=plan.queue('s1')[0];seed=cfg['stream_seeds']['S_init']
        old=core.build_reconstructor('WV3',112,seed,'S');new=model.build_student('WV3',112,seed)
        self.assertEqual(new.backbone.input.in_channels,9)
        self.assertEqual(sum(p.numel() for p in old.parameters())-sum(p.numel() for p in new.parameters()),1008)
        for k,v in new.backbone.state_dict().items():
            original=old.backbone.state_dict()[k]
            if k=='input.weight':original=torch.cat([original[:,:1],original[:,2:]],1)
            torch.testing.assert_close(v,original,rtol=0,atol=0)
        with self.assertRaises(RuntimeError):new.load_state_dict(old.state_dict(),strict=True)
    def test_frozen_loss_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered=synthetic_source(tmp,'WV3');dc=plan.donor_config(1);cfg=deepcopy(plan.queue('s1')[0])
            cfg.update(batch=1,microbatch=1)
            ref=oldmodel.Stage1(dc,registered.inventory.manifest())
            with torch.no_grad():ref.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
            for m in (ref.aligner,ref.teacher):m.requires_grad_(False).eval()
            binding=registered.inventory.manifest()
            binding.update(sensor='WV3',reference_mode='RGBN',alignment_target=dc['alignment_target'],
                           warp=core.WARP_CONTRACT,scharr={'reference_indices':dc['reference_bands']})
            donor=SimpleNamespace(aligner=ref.aligner,teacher=ref.teacher,binding=binding,
                                  selection_receipt={},assert_frozen=lambda:None)
            with patch.object(plan,'validate'):
                net=model.Stage2(cfg,donor,Stats());trainer=training.Trainer(cfg,net)
            batch=registered.batch([0]);f=core.frozen_predictions(net.aligner,net.teacher,batch)
            e,g,q=core.reference_maps(net.spatial,f['teacher_y'],f['gt_loss'],f['pan_loss'])
            expected_weight=1+.25*(e/(e+.1))*.5
            y=net.native_prediction(batch)['y']
            expected=masked_mean(expected_weight*(y-batch.gt).abs().mean(1,keepdim=True),f['loss_mask'])
            with patch('pa.warp.warp_pan',side_effect=AssertionError('bicubic used')):
                terms=net(batch)
            torch.testing.assert_close(terms['loss'],expected)
            stream=training.stream_for(cfg,registered,'cpu');pins={'test':True}
            path=Path(tmp)/'resume.pt'
            training.save(path,trainer,stream,pins);training.load(path,trainer,stream,pins)
            trainer.step(stream.next_update());training.save(path,trainer,stream,pins)
            want=trainer.step(stream.next_update());want_hash=training.model_hash(net)
            training.load(path,trainer,stream,pins);got=trainer.step(stream.next_update())
            self.assertEqual(want,got);self.assertEqual(want_hash,training.model_hash(net));trainer.audit()
    def test_fr_replaces_only_pan(self):
        raw=np.ones((8,4,4));lms=np.zeros_like(raw);pan=np.ones((1,4,4));corrected=pan*2
        arrays={'pan':pan,'lms':lms};seen=[]
        def fn(pred,a):
            self.assertIs(pred,raw);self.assertIs(a['lms'],lms);seen.append(a['pan'])
            return dict(hqnr=.9,d_s=.08,d_lambda=.01,jqm=.8)
        got=evaluation.dual_fr_metrics(fn,raw,arrays,corrected,dict(D_lambda=.01),8)
        self.assertIs(seen[0],corrected);self.assertIs(arrays['pan'],pan)
        self.assertEqual(got['D_lambda'],.01)
    def test_sheet_status_pair_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            v=publication.project(tmp,'s1');self.assertEqual(len(v['rows']),4)
            self.assertEqual(len(set(r[publication.INDEX['Result_ID']] for r in v['rows'])),4)
            for row in v['rows']:self.assertEqual(len(row),64)
            self.assertIn('[corrected PAN]',v['rows'][1][0])

if __name__=='__main__':unittest.main()
