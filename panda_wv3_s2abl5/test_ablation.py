from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same,_cpu
from donor30.masking import masked_mean
from panda_unified import core
from panda_unified_campaign import model as donor_model, training as donor_training
from panda_unified_campaign.assets import _load_selected,calibration
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_eq40_zeroq.model import Stage2 as OriginalStage2
from panda_eq40_zeroq.weighting import conditional_rank
from . import plan,model,training,publication,summary

class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_round_major_three_seeds_and_single_fixed_donor(self):
        q=plan.queue('s1')
        self.assertEqual(len(q),15)
        self.assertEqual([c['case_id'] for c in q],plan.PLAN['case_order']*3)
        self.assertEqual([c['repeat'] for c in q],[1]*5+[2]*5+[3]*5)
        self.assertEqual(sum(c['reuse_existing'] for c in q),1)
        self.assertEqual(len({c['donor_run_id'] for c in q}),1)
        self.assertEqual({c['fixed_donor']['step'] for c in q},{27000})
        self.assertEqual({c['horizon'] for c in q},{40000})
        for r in (1,2,3):
            cases=[c for c in q if c['repeat']==r]
            self.assertEqual(len({plan.digest(c['stream_seeds']) for c in cases}),1)
        with self.assertRaises(ValueError):plan.resolve('s1','GLOBAL_RANK',1)
        with self.assertRaises(ValueError):plan.queue('s3')

    def test_all_case_paths_and_ours_main_parity_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_source(root,'WV3')
            dc=plan.main.donor_config('s1',1)
            teacher=donor_model.Stage1(dc,registered.inventory.manifest())
            with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
            tr=donor_training.Trainer(dc,teacher);stream=donor_training.stream_for(dc,registered,'cpu')
            pins={'source_release':'synthetic-test'};ck=root/'donor.pt'
            meta=donor_training.save(ck,tr,stream,pins,evaluated=True)
            receipt={'selected':{'step':0,'checkpoint_path':'donor.pt','file_sha256':meta['file_sha256'],
                'model_sha256':meta['model_sha256']},'receipt_sha256':'d'*64}
            donor=_load_selected(dc,receipt,registered,root,pins,device='cpu')
            stats=calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
            batch=registered.batch([0]);ref=None;initial=None
            for case in plan.PLAN['case_order']:
                cfg=deepcopy(plan.resolve('s1',case,1));cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):
                    net=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,net)
                parameters={k:v.detach().clone() for k,v in net.student.named_parameters()}
                if initial is None:initial=parameters
                self.assertTrue(_same(initial,parameters))
                out=net.native_prediction(batch)
                if ref is None:ref=out['pan_aligned'].clone()
                self.assertTrue(torch.equal(ref,out['pan_aligned']))
                self.assertTrue(torch.equal(out['x_in'][:,:1],batch.pan if case=='A_RAW_L1' else ref))
                f=core.frozen_predictions(net.aligner,net.teacher,batch)
                e,g,q=core.reference_maps(net.spatial,f['teacher_y'],batch.gt,ref);mask=f['loss_mask']
                cue=e/(e+stats.tau_e).clamp_min(torch.finfo(torch.float32).tiny)
                rank=torch.zeros_like(e);rank[mask]=conditional_rank(stats,g[mask],q[mask])
                expected={'A_RAW_L1':torch.ones_like(e),'B_ALIGN_L1':torch.ones_like(e),
                    'C_DE_ONLY':1+.25*cue,'D_RQ_ONLY':1+.25*rank,'E_OURS':1+.25*cue*rank}[case]
                got=model.weights(e,g,q,stats,mask,cfg)
                torch.testing.assert_close(got[mask],expected[mask]);self.assertFalse(got.requires_grad)
                result=net(batch)
                target=masked_mean(expected*(out['y']-batch.gt).abs().mean(1,keepdim=True),mask)
                torch.testing.assert_close(result['loss'],target)
                if case=='E_OURS':
                    original=OriginalStage2(plan.source_config(),donor,stats)
                    torch.testing.assert_close(original(batch)['loss'],result['loss'],rtol=0,atol=0)
                    self.assertTrue(torch.equal(original.native_prediction(batch)['y'],out['y']))
                train_stream=training.stream_for(cfg,registered,'cpu')
                trainer.step(train_stream.next_update());trainer.audit()
                if case=='A_RAW_L1':
                    checkpoint=root/'resume.pt';training.save(checkpoint,trainer,train_stream,pins)
                    boundary=_cpu(net.state_dict());next_terms=trainer.step(train_stream.next_update())
                    target_hash=training.model_hash(net)
                    training.load(checkpoint,trainer,train_stream,pins)
                    self.assertTrue(_same(boundary,net.state_dict()))
                    self.assertEqual(next_terms,trainer.step(train_stream.next_update()))
                    self.assertEqual(target_hash,training.model_hash(net))
                del net,trainer

    def test_zero_rank_exception_on_rank_only_and_product(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        e=torch.ones(1,1,4,4);g=torch.ones_like(e);q=torch.zeros_like(e);mask=torch.ones_like(e,dtype=torch.bool)
        for case in ('D_RQ_ONLY','E_OURS'):
            self.assertTrue(torch.equal(model.weights(e,g,q,Stats(),mask,plan.resolve('s1',case,1)),e))
        self.assertTrue((model.weights(e,g,q,Stats(),mask,plan.resolve('s1','C_DE_ONLY',1))>1).all())

    def test_paper_aggregation_keeps_missing_seed_explicit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(summary,'verify_record'):
            root=Path(tmp);case='A_RAW_L1'
            for r,v in [(1,2.),(2,2.2)]:
                c=plan.resolve('s1',case,r);d=root/'runs'/c['run_id'];d.mkdir(parents=True)
                e={'step':40000,'checkpoint':{'step':40000,'file_sha256':str(r)*64},
                   'fr_corrected_pan':{'metrics':{'HQNR':.96,'D_s':.03,'D_lambda':.01}},
                   'fr':{'metrics':{'HQNR':.92,'D_s':.07}},'rr':{'metrics':{'ERGAS':v,'SAM':2.7,'PSNR':38.}}}
                (d/'evaluation_0040000.json').write_text(json.dumps(e))
            with patch.object(plan,'source_directory',return_value=root/'no_reuse_fixture'):
                result=summary.collect(root,'s1')
            row=result['rows'][0]
            self.assertEqual(row['n'],2);self.assertFalse(row['complete'])
            self.assertAlmostEqual(row['mean']['ERGAS'],2.1)
            self.assertAlmostEqual(row['std']['ERGAS'],.2/(2**.5))
            self.assertFalse(result['ready_for_three_repeat_table'])

if __name__=='__main__':unittest.main()
