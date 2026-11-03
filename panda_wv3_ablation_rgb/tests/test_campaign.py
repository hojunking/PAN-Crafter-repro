from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from donor30.artifacts import _same
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_wv3_ablation import plan as old_plan
from panda_wv3_ablation_rgb import plan, model, training, evaluation, runtime, publication, enqueue
from pa.warp import warp_pan


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fresh_round_major_full_control_and_legacy_isolation(self):
        q=plan.queue('s3')
        self.assertEqual(len(q),18)
        self.assertEqual({c['run_id'] for c in q}&{c['run_id'] for c in old_plan.queue('s3')},set())
        for i in range(3):
            self.assertEqual([c['case_id'] for c in q[6*i:6*(i+1)]],[c['case_id'] for c in q[:6]])
            self.assertEqual({c['repeat'] for c in q[6*i:6*(i+1)]},{i+1})
        for c in q:
            plan.validate(c)
            self.assertTrue(c['experiment'].endswith('(RGB)'))
            self.assertEqual((c['horizon'],c['batch'],c['microbatch']),(50000,48,12))
            self.assertNotIn('donor_checkpoint',c)
            self.assertEqual(c['initialization'],'FRESH')
        self.assertEqual((q[5]['lambda_abs'],q[5]['lambda_epsilon']),(2,.003))
        self.assertNotEqual(plan.campaign_root('s3'),old_plan.campaign_root('s3'))
        self.assertEqual(plan.catalog('s3'),(plan.CONFIG/'cases_s3.csv').read_text())
        with self.assertRaises(ValueError):plan.queue('s1')
        state=runtime.initial_state('s3');state['runs'][0]['status']='FAILED'
        self.assertEqual(runtime.next_job(state)['case'],'S1_A1_SCHARR')
        # Same initialization streams for every matched case and the old reference.
        self.assertEqual(q[3]['stream_seeds'],old_plan.resolve('s3','S1_A3_REC_SPA',1)['stream_seeds'])

    def test_spatial_loss_uses_rgb_but_aligner_keeps_eight_inputs(self):
        m=model.Stage1(plan.resolve('s3','S1_A3_REC_SPA',1),{})
        self.assertEqual(m.aligner.ms_bands,8)
        pan=torch.zeros(1,1,64,64)
        pattern=torch.arange(64).float()[None,:].expand(64,64)/64
        active=[]
        for band in range(8):
            gt=torch.zeros(1,8,64,64);gt[0,band]=pattern
            if float(m.spatial.absolute_loss(pan,gt))>0:active.append(band)
        self.assertEqual(active,[1,2,4])

    def test_loss_switches_real_forward_backward_and_full_spatial_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered=synthetic_sources(Path(tmp));rr=registered.batch([0])
            g=torch.Generator().manual_seed(73)
            rr=replace(rr,**{k:torch.rand(getattr(rr,k).shape,generator=g)*1.6-.8
                             for k in ('pan','ms','native_lms','gt')})
            for key in ('S1_A0_RAW','S1_A2_REC','S1_A3_REC_SPA','S1_A4_REC_EPS','S1_A5_ALL'):
                cfg=plan.resolve('s3',key,1);m=model.Stage1(cfg,registered.inventory.manifest())
                with patch.object(m.spatial,'absolute_loss',wraps=m.spatial.absolute_loss) as spatial:
                    out=m(rr)
                self.assertEqual(spatial.call_count,int(bool(cfg['lambda_abs'])))
                torch.testing.assert_close(out['loss'],out['rec']+cfg['lambda_abs']*out['spatial']+cfg['lambda_epsilon']*out['epsilon'])
                if key=='S1_A5_ALL':
                    grads=torch.autograd.grad(out['spatial'],[m.aligner.fc2.weight,m.teacher.backbone.input.weight],
                                              retain_graph=True,allow_unused=True)
                    self.assertGreater(float(grads[0].abs().sum()),0)
                    self.assertIsNone(grads[1])
                out['loss'].backward()
                self.assertTrue(any(p.grad is not None for p in m.teacher.parameters()))

    def test_scharr_matches_rgb_and_ignores_non_rgb_reference_changes(self):
        from panda_wv3_ablation_rgb.matching import ScharrAlignment
        torch.manual_seed(91)
        p=torch.nn.functional.avg_pool2d(torch.rand(1,1,68,68),5,stride=1)
        shifts=torch.tensor([[.5,-1.]])
        lms=warp_pan(p,shifts).expand(-1,8,-1,-1).clone()
        match=ScharrAlignment()
        expected=match(p,lms)
        lms[:,[0,3,5,6,7]]=torch.rand_like(lms[:,[0,3,5,6,7]])
        torch.testing.assert_close(match(p,lms),expected,atol=0,rtol=0)
        torch.testing.assert_close(expected,shifts,atol=0,rtol=0)

    def test_complete_rgb_full_resume_and_cross_campaign_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_sources(root,count=3)
            cfg=plan.resolve('s3','S1_A5_ALL',1);cfg.update(batch=1,microbatch=1)
            def build():
                with patch('panda_wv3_ablation_rgb.plan.validate'):
                    t=training.Trainer(cfg,model.Stage1(cfg,registered.inventory.manifest()))
                return t,training.stream_for(cfg,registered,'cpu')
            t,s=build();t.step(s.next_update());training.save(root/'resume.pt',t,s,{'RGB':True})
            expected=t.step(s.next_update());weights=deepcopy(t.model.state_dict());opt=deepcopy(t.optimizer.state_dict())
            t2,s2=build();training.load(root/'resume.pt',t2,s2,{'RGB':True})
            self.assertEqual(t2.step(s2.next_update()),expected)
            self.assertTrue(_same(weights,t2.model.state_dict()));self.assertTrue(_same(opt,t2.optimizer.state_dict()))
            self.assertTrue(_same(s.state_dict(),s2.state_dict()))
            with self.assertRaisesRegex(ValueError,'Resume'):
                training.load(root/'resume.pt',t2,s2,{'mean8':True})

    def test_rgb_test_rre_target_and_failures_are_not_zero(self):
        gt=torch.randn(1,8,256,256);pan=torch.zeros(1,1,256,256)
        batch=SimpleNamespace(gt=gt,native_lms=gt)
        reg={'shift':[.3,.4],'valid':True}
        with patch('ta2.registration.register_band',return_value=reg) as register:
            r=evaluation.test_rre({'pan_aligned':pan},batch,'rr')
        np.testing.assert_array_equal(register.call_args.args[1],((gt[0,[4,2,1]].mean(0)+1)/2).numpy())
        self.assertEqual(r['rre_px'],.5)
        rows=[{'scene_id':f'WV3:rr:{i}','rre_rgb':deepcopy(r)} for i in range(20)]
        self.assertEqual(evaluation.rre_summary(rows)['RRE50'],.5)
        rows[0]['rre_rgb']['rre_px']=None
        self.assertIsNone(evaluation.rre_summary(rows)['RRE50'])
        self.assertEqual(evaluation.rre_summary(rows)['failure_count'],1)

    def test_sheet_names_and_old_rows_are_not_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=publication.project(tmp,'s3')
            self.assertEqual(len(value['rows']),18)
            for row in value['rows']:
                self.assertTrue(row[publication.INDEX['Experiment']].endswith('(RGB)'))
                self.assertIn('RGB mean',row[publication.INDEX['Approach']])
            foreign=['']*64
            foreign[publication.INDEX['Campaign_ID']]=old_plan.CAMPAIGN_ID
            foreign[publication.INDEX['Server']]='s3'
            reqs=publication.row_requests({'rows':[foreign],'notes':['']},value,'s3')
            self.assertTrue(all('updateCells' not in r for r in reqs))
            self.assertEqual(sum(len(r.get('appendCells',{}).get('rows',[])) for r in reqs),18)

    def test_native_evaluation_evidence_and_representative_publication(self):
        class Sources:
            sensor='WV3';peak=2047;bands=8
            def batch(self,split,ids,device=None):
                h=256 if split=='rr' else 512
                lms=torch.zeros(1,8,h,h)
                return SimpleNamespace(pan=torch.zeros(1,1,h,h),native_lms=lms,
                    gt=lms if split=='rr' else None,
                    lineage=SimpleNamespace(sample_ids=(f'WV3:{split}:{ids[0]}',)))
            def raw(self,split,ids):
                arrays={k:np.zeros((1,1)) for k in ('pan','lms','ms')}
                if split=='rr':arrays['gt']=np.zeros((1,1))
                return arrays
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.teacher=torch.nn.Linear(1,1)
            def native_prediction(self,b):
                return {'y':b.native_lms,'u':torch.zeros(1,2),'pan_aligned':b.pan}
        cfg=plan.resolve('s3','S1_A5_ALL',1);m=Model();m.train()
        trainer=SimpleNamespace(cfg=cfg,model=m,updates=1000,audit=lambda:None)
        def rr(raw,arrays):
            self.assertIn('gt',arrays)
            return {k:1. for k in ('ergas','scc','sam','psnr','ssim','q8','rmse','cc')}
        def fr(raw,arrays):
            self.assertNotIn('gt',arrays)
            return {k:.9 for k in ('hqnr','d_s','d_lambda','jqm')}
        with patch.object(evaluation,'evaluator_identity',return_value={'synthetic':True}), \
             patch.object(evaluation,'metric_functions_for',return_value={'rr':rr,'fr':fr}):
            record=evaluation.evaluate(trainer,Sources(),{'synthetic':True})
        evaluation.verify_record(record,cfg)
        self.assertTrue(m.training)
        self.assertEqual(record['test_rre_rgb']['fr']['failure_count'],20)
        self.assertIsNone(record['test_rre_rgb']['fr']['RRE50'])
        corrupt=deepcopy(record);corrupt['test_rre_rgb']['rr']['RRE50']=0
        with self.assertRaisesRegex(ValueError,'summary'):
            evaluation.verify_record(corrupt,cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'runs'/cfg['run_id'];directory.mkdir(parents=True)
            checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'fixture checkpoint')
            record.update(checkpoint={'path':str(checkpoint),'file_sha256':publication.file_sha(checkpoint),
                'model_sha256':record['model_sha256'],'step':1000},recorded_at='2026-11-03T00:00:00Z',
                validation={'RRE50':.2,'RRE90':.3,'Val_ERGAS':2.1})
            (directory/'evaluation_0001000.json').write_text(json.dumps(record))
            value=publication.project(root,'s3')
            row=next(r for r in value['rows'] if r[publication.INDEX['Run_ID']]==cfg['run_id'])
            self.assertEqual(row[publication.INDEX['Selector']],'BEST_VAL_RRE')
            self.assertTrue(json.loads(row[publication.INDEX['Original description / notes']])['test_rre_rgb'])
            publication.verify_owned(row,value,'s3')

    def test_launcher_dry_run_and_failed_preflight_do_not_stop_legacy(self):
        args=['--server','s3','--image','test','--replace-legacy']
        with patch.object(enqueue,'release_check',return_value='a'*64), \
             patch.object(enqueue,'inspect_container',return_value=None), \
             patch.object(enqueue,'command',return_value=['fake']), \
             patch.object(enqueue,'stop_legacy') as stop, \
             patch.object(enqueue.subprocess,'run'):
            enqueue.main(args+['--check-only']);stop.assert_not_called()
        with patch.object(enqueue,'release_check',return_value='a'*64), \
             patch.object(enqueue,'inspect_container',return_value=None), \
             patch.object(enqueue,'command',return_value=['fake']), \
             patch.object(enqueue,'stop_legacy') as stop, \
             patch.object(enqueue.subprocess,'run',side_effect=RuntimeError('preflight failed')):
            with self.assertRaisesRegex(RuntimeError,'preflight'):enqueue.main(args)
            stop.assert_not_called()

    def test_cutover_refuses_s1_or_another_checkout(self):
        info={'Config':{'Labels':{'panda.campaign':enqueue.LEGACY_CAMPAIGN,'panda.server':'s1'}},
              'Mounts':[{'Destination':'/workspace','Source':str(plan.ROOT)}],'State':{'Running':True}}
        with patch.object(enqueue,'inspect_container',return_value=info),patch.object(enqueue.subprocess,'run') as run:
            with self.assertRaises(ValueError):enqueue.stop_legacy()
            run.assert_not_called()
        info['Config']['Labels']['panda.server']='s3';info['Mounts'][0]['Source']='/another-checkout'
        with self.assertRaises(ValueError):enqueue.verify_container(info,campaign=enqueue.LEGACY_CAMPAIGN)


if __name__=='__main__':unittest.main()
