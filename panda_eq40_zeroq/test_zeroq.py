"""Degenerate CDFs, mask separation, and sensor-identical real training paths."""
from copy import deepcopy
import ast,inspect,json,tempfile,textwrap,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from donor30.masking import masked_mean
from panda_unified import core
from panda_unified_campaign import calibration as oldcal,model as oldmodel,training as oldtraining
from panda_sensor_eq40 import calibration as sensorcal,model as sensormodel,training as sensortraining
from panda_unified_campaign.tests.test_calibration_selection import binding
from panda_sensor_eq40.tests.test_integration import synthetic_source
from . import plan,model,training,publication
from .weighting import conditional_rank,adaptive_weight,POLICY


class ZeroQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def stats(self,values,sensor):
        ids,b= binding('WV3' if sensor=='WV3' else 'QB','RGBN',count=1)
        c=oldcal if sensor=='WV3' else sensorcal
        if sensor=='GF2':
            ids=['GF2:train:0'];b.update(sensor=sensor,sample_ids=ids,sample_ids_sha256=c.sample_ids_sha256(ids))
        builder=c.CalibrationBuilder(ids,b,k=1,memory_budget_bytes=8*1024**2,mask_policy=core.MASK_POLICY)
        q=torch.tensor(values,dtype=torch.float32).reshape(1,1,1,-1).expand(1,1,5,-1).contiguous()
        builder.add(ids,torch.ones_like(q),torch.zeros_like(q),q,sensor=sensor,loss_mask=torch.ones_like(q,dtype=torch.bool))
        return builder.finalize()

    def test_all_zero_mixed_zero_positive_bins_all_sensors(self):
        for sensor in ('WV3','QB','GF2'):
            for population,expected in (([0]*5,1.),([0,0,0,.1,.2],.6),([.01,.02,.03,.04,.05],0.)):
                with self.subTest(sensor=sensor,population=population):
                    stats=self.stats(population,sensor)
                    q=torch.tensor([-0.,0.,1e-10,.01,.1,.2,1.]);g=torch.zeros_like(q)
                    raw=stats.lookup_exact(g,q);safe=conditional_rank(stats,g,q)
                    self.assertAlmostEqual(raw[0].item(),expected,places=6)
                    self.assertTrue((safe[:2]==0).all())
                    self.assertTrue(torch.equal(safe[2:],raw[2:]))
                    self.assertFalse(safe.requires_grad)
                    with self.assertRaises(ValueError):conditional_rank(stats,g,q+float('nan'))
                    with self.assertRaises(ValueError):conditional_rank(stats,g,-torch.ones_like(q))

    def test_valid_zero_discrepancy_is_distinct_from_masked_pixels(self):
        stats=self.stats([0]*5,'WV3')
        mask=torch.zeros(1,1,5,5,dtype=torch.bool);mask[:,:,1:4,1:4]=True
        e=torch.ones_like(mask,dtype=torch.float32,requires_grad=True)
        g=torch.full_like(e,-999);q=torch.full_like(e,-999)
        g[mask]=0.;q[mask]=0.
        cfg=dict(alpha=0.,gamma=.25,beta=0.,q_zero_policy=POLICY)
        w=adaptive_weight(e,g,q,stats,mask,cfg)
        self.assertTrue((w==1).all());self.assertFalse(w.requires_grad)
        legacy=core.adaptive_weight(e,g,q,stats,mask,cfg)
        self.assertTrue((legacy[mask]>1).all())
        q[mask]=1e-10
        self.assertTrue(torch.equal(adaptive_weight(e,g,q,stats,mask,cfg),core.adaptive_weight(e,g,q,stats,mask,cfg)))
        q[mask]=0;cfg['alpha']=.25
        self.assertTrue((adaptive_weight(e,g,q,stats,mask,cfg)[mask]>1).all()) # independent alpha stays independent

    def test_same_method_paths_and_declared_dataset_differences(self):
        for server,width,lr,bands in [('s1',112,1.1e-4,8),('s2',128,1e-4,4),('s3',128,2e-4,4)]:
            for cfg in plan.queue(server):
                prior=plan.prior_config(server,cfg['repeat'])
                for k in ('lr','width','seed','stream_seeds','reference_bands','donor_run_id','horizon',
                          'alpha','gamma','beta','student_input','scale','teacher_warp','early_stop'):
                    self.assertEqual(cfg[k],prior[k])
                self.assertEqual((cfg['width'],cfg['lr'],cfg['bands']),(width,lr,bands))
                self.assertEqual(cfg['reference_mode'],'RGBN');self.assertEqual(cfg['q_zero_policy'],POLICY)
                self.assertNotEqual(cfg['config_sha256'],prior['config_sha256'])
                self.assertNotEqual(plan.campaign_root(server),plan.prior_root(server))
                self.assertEqual(cfg['horizon'],40000)
        self.assertIs(model.adaptive_weight,adaptive_weight)
        from . import geometry
        self.assertIs(geometry.warp_pan,core.warp_pan);self.assertIs(geometry.support,core.support)
        self.assertIs(geometry.frozen_predictions,core.frozen_predictions)
        def syntax(fn):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))))
        self.assertEqual(syntax(oldcal.CalibrationStats.lookup_exact),syntax(sensorcal.CalibrationStats.lookup_exact))
        self.assertEqual(syntax(oldmodel.Stage1.forward),syntax(sensormodel.Stage1.forward))
        self.assertEqual(syntax(oldmodel.build_aligner),syntax(sensormodel.build_aligner))

    def test_native_four_eight_band_updates_frozen_teacher_and_complete_resume(self):
        from panda_unified_campaign.assets import _load_selected as load_old
        from panda_sensor_eq40.assets import _load_selected as load_sensor,calibration
        for server in ('s1','s2','s3'):
            with self.subTest(server=server),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r=synthetic_source(root,plan.sensor_for(server))
                dc=plan.donor_config(server,1)
                source_model=oldmodel if server!='s3' else sensormodel
                source_training=oldtraining if server!='s3' else sensortraining
                teacher=source_model.Stage1(dc,r.inventory.manifest())
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=source_training.Trainer(dc,teacher);stream=source_training.stream_for(dc,r,'cpu')
                pins={'source_release':'synthetic-only'};ck=root/'donor.pt'
                meta=source_training.save(ck,tr,stream,pins,evaluated=True)
                receipt={'selected':{'step':0,'checkpoint_path':'donor.pt','file_sha256':meta['file_sha256'],
                    'model_sha256':meta['model_sha256']},'receipt_sha256':'d'*64}
                donor=(load_sensor if server=='s3' else load_old)(dc,receipt,r,root,pins,device='cpu')
                stats=calibration(donor,r,root/'calibration',device='cpu',batch_size=1)
                cfg=deepcopy(plan.queue(server)[0]);cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):
                    student=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,student)
                batch=r.batch([0]);out=student.native_prediction(batch)
                self.assertEqual(out['x_in'].shape[1],cfg['bands']+1)
                f=core.frozen_predictions(student.aligner,student.teacher,batch)
                e,g,q=core.reference_maps(student.spatial,f['teacher_y'],f['gt_loss'],f['pan_loss'])
                w=adaptive_weight(e,g,q,stats,f['loss_mask'],cfg)
                self.assertTrue(torch.equal(w,core.adaptive_weight(e,g,q,stats,f['loss_mask'],cfg)))
                expected=masked_mean(w*(out['y']-batch.gt).abs().mean(1,keepdim=True),f['loss_mask'])
                torch.testing.assert_close(student(batch)['loss'],expected)
                train_stream=training.stream_for(cfg,r,'cpu');trainer.step(train_stream.next_update())
                ck=root/'complete.pt';training.save(ck,trainer,train_stream,pins)
                second=trainer.step(train_stream.next_update());expected_hash=training.model_hash(student)
                training.load(ck,trainer,train_stream,pins)
                self.assertEqual(second,trainer.step(train_stream.next_update()))
                self.assertEqual(expected_hash,training.model_hash(student));trainer.audit()
                with self.assertRaisesRegex(ValueError,'mismatch'):training.load(ck,trainer,train_stream,{'source_release':'wrong-method'})

    def test_publication_names_and_two_rows_per_server(self):
        from panda_fr_columns.publication import original_rows
        for server in ('s1','s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                value=original_rows(publication.project(tmp,server))
                self.assertEqual(len(value['rows']),2)
                self.assertTrue(all('QZERO' in row[0] for row in value['rows']))
                self.assertEqual(value['destination']['sheet_title'],plan.sensor_for(server)+'-ablations')

if __name__=='__main__':unittest.main()
