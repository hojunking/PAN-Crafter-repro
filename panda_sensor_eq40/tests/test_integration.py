from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from donor30.artifacts import _cpu,_same
from donor30.masking import masked_mean
from panda_unified import core
from panda_unified_campaign import model as original_model,training as original_training
from panda_unified_campaign.assets import _load_selected as original_load
from panda_unified_campaign.tests.test_integration import synthetic_source as old_synthetic_source
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha
import h5py
from panda_sensor_eq40 import plan,model,training,assets,publication,calibration,evaluation
from panda_sensor_eq40.coordinator import original_qb_next


def synthetic_source(root,sensor):
    old_synthetic_source(root,sensor,count=12)
    root=Path(root)
    if sensor=='GF2':
        with h5py.File(root/'train.h5','r+') as f:
            for k in f:f[k][:]=f[k][:]*.5
        c=json.loads((root/'catalog.json').read_text())
        c['sensors'][sensor]['splits']['train']['sha256']=file_sha(root/'train.h5')
        (root/'catalog.json').write_text(json.dumps(c))
    return RegisteredRRTrain.from_catalog(root/'catalog.json',root=root,sensor=sensor,
        lms_overrides=({s:{'path':'train.h5','sha256':file_sha(root/'train.h5'),
                       'reason':'Synthetic fixture with identical stored LMS'} for s in ('train','val')} if sensor=='QB' else None))


class SensorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_queues_and_local_donors(self):
        qb,gf=plan.queue('s2'),plan.queue('s3')
        self.assertEqual(len(qb),2);self.assertTrue(all(c['stage']=='stage2' for c in qb))
        self.assertEqual(qb[0]['donor_run_id'],qb[1]['donor_run_id'])
        self.assertNotEqual(qb[0]['seed'],qb[1]['seed'])
        self.assertEqual([c['stage'] for c in gf],['stage1','stage1','stage2','stage2'])
        self.assertNotEqual(gf[2]['donor_run_id'],gf[3]['donor_run_id'])
        for c in qb+gf:
            self.assertEqual(c['width'],128);self.assertEqual(c['reference_bands'],[2,1,0,3])
            self.assertEqual(c['student_input'],'CORRECTED');self.assertEqual(c['scale'],1)
            self.assertEqual(c['alpha'],0);self.assertEqual(c['beta'],0)
            if c['stage']=='stage2':self.assertEqual((c['gamma'],c['horizon']),(.25,40000))
        self.assertEqual([c['lr'] for c in qb],[1e-4]*2)
        self.assertEqual(gf[2]['lr'],2e-4)
        self.assertEqual(gf[0]['validation_steps'],[30000,40000,50000])
        from panda_unified_campaign.runtime import initial_state
        state=initial_state('s2');wanted=plan.donor_config('s2',1)['run_id']
        self.assertEqual(original_qb_next(state)['run_id'],wanted)
        next(r for r in state['runs'] if r['run_id']==wanted)['status']='COMPLETED'
        self.assertIsNone(original_qb_next(state))
    def test_real_checkpoint_calibration_student_freeze_and_resume(self):
        for server,sensor in [('s2','QB'),('s3','GF2')]:
            with self.subTest(sensor=sensor),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);registered=synthetic_source(root,sensor)
                dc=plan.donor_config(server,1)
                teacher=(original_model.Stage1(dc,registered.inventory.manifest()) if server=='s2' else
                         model.Stage1(dc,registered.inventory.manifest()))
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=(original_training.Trainer(dc,teacher) if server=='s2' else training.Trainer(dc,teacher))
                stream=training.stream_for(dc,registered,'cpu');pins=dict(source_release='synthetic-test')
                save=original_training.save if server=='s2' else training.save
                ck=root/'donor.pt';meta=save(ck,tr,stream,pins,evaluated=True)
                selected=dict(step=0,checkpoint_path='donor.pt',file_sha256=meta['file_sha256'],model_sha256=meta['model_sha256'])
                receipt=dict(selected=selected,receipt_sha256='d'*64)
                load=original_load if server=='s2' else assets._load_selected
                donor=load(dc,receipt,registered,root,pins,device='cpu')
                stats=assets.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
                self.assertGreater(stats.tau_e,0)
                again=assets.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
                self.assertEqual(stats.artifact_sha256,again.artifact_sha256)
                cfg=deepcopy(plan.resolve(server,plan.STAGE2_CASE,1));cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):
                    student=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,student)
                batch=registered.batch([0]);out=student.native_prediction(batch)
                self.assertEqual(out['x_in'].shape,(1,5,64,64));self.assertEqual(out['y'].shape,(1,4,64,64))
                f=core.frozen_predictions(student.aligner,student.teacher,batch)
                e,g,q=core.reference_maps(student.spatial,f['teacher_y'],f['gt_loss'],f['pan_loss'])
                w=core.adaptive_weight(e,g,q,stats,f['loss_mask'],cfg)
                self.assertGreater(float(w[f['loss_mask']].mean()),1.)
                self.assertLessEqual(float(w.max()),1.25)
                expected=masked_mean(w*(out['y']-batch.gt).abs().mean(1,keepdim=True),f['loss_mask'])
                torch.testing.assert_close(student(batch)['loss'],expected)
                train_stream=training.stream_for(cfg,registered,'cpu')
                trainer.step(train_stream.next_update());ck=root/'resume.pt'
                training.save(ck,trainer,train_stream,pins)
                second=trainer.step(train_stream.next_update());expected_hash=training.model_hash(student)
                training.load(ck,trainer,train_stream,pins)
                self.assertEqual(second,trainer.step(train_stream.next_update()))
                self.assertEqual(expected_hash,training.model_hash(student));trainer.audit()
    def test_gf2_stage1_gradients_and_validation_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=synthetic_source(tmp,'GF2');cfg=plan.resolve('s3',plan.STAGE1_CASE,1)
            net=model.Stage1(cfg,r.inventory.manifest());result=net(r.batch([0]));result['loss'].backward()
            self.assertTrue(torch.isfinite(net.aligner.fc2.bias.grad).all())
            self.assertGreater(float(net.aligner.fc2.bias.grad.abs().sum()),0)
            from panda_sensor_eq40.selection import protocol
            self.assertEqual(protocol('GF2','RGBN')['reference_band_indices'],[2,1,0,3])
    def test_sheet_routes_pairs_and_gf2_target(self):
        for server,title,count in [('s2','QB-ablations',4),('s3','GF2-ablations',8)]:
            with tempfile.TemporaryDirectory() as tmp:
                v=publication.project(tmp,server)
                self.assertEqual(v['destination']['sheet_title'],title);self.assertEqual(len(v['rows']),count)
                self.assertEqual(len({r[publication.INDEX['Result_ID']] for r in v['rows']}),count)
                for r in v['rows']:self.assertEqual(len(r),64)
        cfg=plan.resolve('s3',plan.STAGE2_CASE,1)
        rec=dict(fr=dict(metrics=dict(HQNR=.965)),rr=dict(metrics=dict(ERGAS=.52,PSNR=45.079)))
        self.assertTrue(publication.joint_pass(rec,cfg));self.assertFalse(publication.all_targets_pass(rec,cfg))
        rec['rr']['metrics']['PSNR']=45.08
        self.assertTrue(publication.all_targets_pass(rec,cfg))
    def test_publisher_s1_donor_selection_does_not_need_validation_at_every_step(self):
        cfg=plan.resolve('s3',plan.STAGE1_CASE,1)
        def r(step,e,p,h):return dict(step=step,rr=dict(metrics=dict(ERGAS=e,PSNR=p)),fr=dict(metrics=dict(HQNR=h)))
        a,b,c=r(1000,.53,45.2,.99),r(2000,.52,45.08,.96),r(3000,.519,45.09,.95)
        chosen=publication.choose([a,b,c],'stage1',cfg)
        self.assertEqual(chosen[0][0]['step'],2000)
        self.assertEqual(chosen[0][1],'DONOR_METRIC_GATES')
        self.assertEqual(publication.choose([a],'stage1',cfg)[0][1],'DONOR_GATES_NOT_YET_MET')

if __name__=='__main__':unittest.main()
