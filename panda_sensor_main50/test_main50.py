from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_unified_campaign import model as oldmodel,training as oldtraining
from panda_sensor_eq40 import model as sensormodel,training as sensortraining
from panda_sensor_eq40.assets import calibration,_load_selected as sensor_load
from panda_unified_campaign.assets import _load_selected as old_load
from panda_eq40_zeroq import model as base_model
from panda_stage2_condq.training import TrainingSpec
from . import plan,model,training,publication,runtime,enqueue

class Main50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_five_full_horizon_runs_per_server_no_early_stop(self):
        for server in ('s2','s3'):
            configs=plan.queue(server)
            self.assertEqual(len(configs),5)
            self.assertEqual(len({plan.digest(c['stream_seeds']) for c in configs}),1)
            self.assertEqual(len({c['donor_run_id'] for c in configs}),1)
            for c in configs:
                self.assertEqual((c['horizon'],c['primary_step'],c['common_comparison_step']),(50000,50000,50000))
                self.assertEqual(c['evaluation_steps'],list(range(1000,50001,1000)))
                self.assertFalse(c['early_stop']['enabled'])
                self.assertEqual((c['width'],c['alpha'],c['beta'],c['scale']),(128,0,0,1))
                for step in (1000,10000,20000,40000,50000):
                    self.assertIsNone(runtime.early_stop_evidence(c,Path('/does-not-exist'),step))
                schedule=TrainingSpec(lr=c['lr'],total_updates=c['horizon'],warmup_updates=c['warmup'],
                    weight_decay=c['weight_decay'],effective_batch_size=c['batch'],microbatch_size=c['microbatch'])
                self.assertGreater(schedule.rate(40000),0.)
                self.assertEqual(schedule.rate(50000),0.)
        with self.assertRaises(ValueError):plan.queue('s1')

    def test_ten_real_optimizer_paths_same_init_frozen_donor_and_resume(self):
        for server in ('s2','s3'):
            with self.subTest(server=server),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r=synthetic_source(root,plan.sensor_for(server));dc=plan.donor_config(server)
                sm,st,loader=(oldmodel,oldtraining,old_load) if server=='s2' else (sensormodel,sensortraining,sensor_load)
                teacher=sm.Stage1(dc,r.inventory.manifest())
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=st.Trainer(dc,teacher);ds=st.stream_for(dc,r,'cpu');pins={'source_release':'synthetic'}
                meta=st.save(root/'donor.pt',tr,ds,pins,evaluated=True)
                receipt={'selected':{'step':0,'checkpoint_path':'donor.pt','file_sha256':meta['file_sha256'],
                    'model_sha256':meta['model_sha256']},'receipt_sha256':'d'*64}
                donor=loader(dc,receipt,r,root,pins,device='cpu')
                stats=calibration(donor,r,root/'calibration',device='cpu',batch_size=1)
                initial=None
                for ordinal,cfg in enumerate(plan.queue(server)):
                    cfg=deepcopy(cfg);cfg.update(batch=1,microbatch=1)
                    with patch.object(plan,'validate'):
                        net=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,net)
                    params={k:v.detach().clone() for k,v in net.student.named_parameters()}
                    if initial is None:initial=params
                    self.assertTrue(_same(initial,params))
                    batch=r.batch([0]);out=net.native_prediction(batch)
                    self.assertEqual(out['x_in'].shape,(1,5,64,64))
                    self.assertEqual(out['y'].shape,(1,4,64,64))
                    # Direct numerical parity with the frozen main method at
                    # identical hyperparameters; only registered recipe changes.
                    with patch.object(base_model.plan,'validate'):
                        reference=base_model.Stage2(cfg,donor,stats)
                    self.assertTrue(torch.equal(reference(batch)['loss'],net(batch)['loss']))
                    self.assertTrue(torch.equal(reference.native_prediction(batch)['y'],out['y']))
                    stream=training.stream_for(cfg,r,'cpu')
                    trainer.step(stream.next_update());trainer.audit()
                    if ordinal==0:
                        checkpoint=root/'resume.pt';training.save(checkpoint,trainer,stream,pins)
                        second=trainer.step(stream.next_update());sha=training.model_hash(net)
                        training.load(checkpoint,trainer,stream,pins)
                        self.assertEqual(second,trainer.step(stream.next_update()))
                        self.assertEqual(sha,training.model_hash(net))
                        trainer.audit()
                    del reference,net,trainer

    def test_zero_discrepancy_guard_remains(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        e=torch.ones(1,1,5,5);q=torch.zeros_like(e);mask=torch.ones_like(e,dtype=torch.bool)
        for server in ('s2','s3'):
            for cfg in plan.queue(server):
                weight=model.adaptive_weight(e,e,q,Stats(),mask,cfg)
                self.assertTrue(torch.equal(weight,e))

    def test_unqualified_metrics_still_have_rr_extrema_and_fixed50(self):
        cfg=plan.queue('s2')[0]
        def record(step,h,e,p):
            return {'step':step,'fr_corrected_pan':{'metrics':{'HQNR':h}},
                'fr':{'metrics':{'HQNR':.99-h}},'rr':{'metrics':{'ERGAS':e,'PSNR':p}}}
        data=[record(1000,.7,4.,36.),record(20000,.8,3.7,37.),record(50000,.6,3.55,38.)]
        selected=publication.choose(data,'stage2',cfg);slots={s:(r,t) for r,t,s in selected}
        self.assertEqual(slots['ROW0'][0]['step'],20000)
        self.assertEqual(slots['MIN_ERGAS'][0]['step'],50000)
        self.assertEqual(slots['MAX_PSNR'][0]['step'],50000)
        self.assertIn('FIXED50K',slots['STEP_0050000'][1])
        self.assertEqual(len(slots),len(selected))
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                value=publication.project(tmp,server)
                self.assertEqual(len(value['rows']),10) # two FR views before FR67 joining
                self.assertEqual(value['destination']['sheet_title'],plan.sensor_for(server)+'-ablations')
                for row in value['rows']:
                    self.assertEqual(row[publication.INDEX['Bucket']],'main')
                    self.assertIn('MAIN50',row[0])
                    self.assertEqual(json.loads(row[publication.INDEX['Original description / notes']])['reference_repeats'],1)
        gf=plan.queue('s3')[0]
        r=record(50000,.975,.516,45.06)
        self.assertTrue(publication.joint_pass(r,gf));self.assertFalse(publication.all_targets_pass(r,gf))
        r['rr']['metrics']['PSNR']=45.08
        self.assertTrue(publication.all_targets_pass(r,gf))

    def test_donor_roots_read_only_and_only_new_output_writable(self):
        fake=['docker','run','--rm','--mount','type=bind,src=/tmp/old,dst=/workspace/work_dir/panda_sensor_eq40','image']
        for server in ('s2','s3'):
            with patch.object(enqueue.base,'command',return_value=list(fake)):
                cmd=enqueue.command(server,'image')
            mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
            self.assertTrue(mounts[0].endswith(',readonly'))
            self.assertIn('dst=/workspace/work_dir/panda_sensor_main50',mounts[1])
            self.assertEqual(cmd[-3:],['python','-m','panda_sensor_main50'])

if __name__=='__main__':unittest.main()
