from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_unified_campaign import model as original_s1
from panda_eq40_zeroq import model as original_s2
from . import plan,model,training,assets,publication,runtime,enqueue

class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_plan_same_method_no_floor_full50k_local_only(self):
        cfgs=plan.queue('s3');self.assertEqual(len(cfgs),6)
        self.assertEqual((cfgs[0]['lambda_abs'],cfgs[0]['lambda_epsilon']),(4.,.03))
        self.assertEqual(len({plan.digest(c['stream_seeds']) for c in cfgs[1:]}),1)
        for c in cfgs:
            self.assertEqual((c['width'],c['scale'],c['alpha'],c['beta']),(128,1.,0.,0.))
            self.assertEqual(c['horizon'],50000);self.assertFalse(c['early_stop']['enabled'])
            self.assertEqual(c['reference_bands'],[2,1,0,3]);self.assertEqual(c['student_input'],'CORRECTED')
            for step in (10000,20000,50000):self.assertIsNone(runtime.early_stop_evidence(c,Path('/missing'),step))
        self.assertTrue(all(c['donor_run_id']==cfgs[0]['run_id'] for c in cfgs[1:]))
        with self.assertRaises(ValueError):plan.queue('s2')
    def test_real_training_parity_donor_freeze_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_source(root,'QB');dc=plan.donor_config('s3')
            donor_model=model.Stage1(dc,registered.inventory.manifest())
            with patch.object(original_s1.plan,'validate'):baseline=original_s1.Stage1(dc,registered.inventory.manifest())
            batch=registered.batch([0]);self.assertTrue(torch.equal(donor_model(batch)['loss'],baseline(batch)['loss']))
            with torch.no_grad():donor_model.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
            dt=training.Trainer(dc,donor_model);stream=training.stream_for(dc,registered,'cpu');pins={'source_release':'synthetic'}
            meta=training.save(root/'donor.pt',dt,stream,pins,evaluated=True)
            receipt={'selected':{'step':0,'checkpoint_path':'donor.pt','file_sha256':meta['file_sha256'],'model_sha256':meta['model_sha256']},'receipt_sha256':'d'*64}
            donor=assets._load_selected(dc,receipt,registered,root,pins,device='cpu')
            stats=assets.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
            initial=None
            for ordinal,cfg in enumerate(plan.queue('s3')[1:]):
                cfg=deepcopy(cfg);cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):net=model.Stage2(cfg,donor,stats);tr=training.Trainer(cfg,net)
                with patch.object(original_s2.plan,'validate'):ref=original_s2.Stage2(cfg,donor,stats)
                self.assertTrue(torch.equal(net(batch)['loss'],ref(batch)['loss']))
                pred=net.native_prediction(batch);self.assertEqual(pred['x_in'].shape,(1,5,64,64))
                self.assertTrue(torch.equal(pred['y'],ref.native_prediction(batch)['y']))
                weights={k:v.detach().clone() for k,v in net.student.named_parameters()}
                if initial is None:initial=weights
                self.assertTrue(_same(initial,weights))
                stream=training.stream_for(cfg,registered,'cpu');tr.step(stream.next_update());tr.audit()
                if ordinal==0:
                    ck=root/'resume.pt';training.save(ck,tr,stream,pins)
                    second=tr.step(stream.next_update());sha=training.model_hash(net)
                    training.load(ck,tr,stream,pins)
                    self.assertEqual(second,tr.step(stream.next_update()));self.assertEqual(sha,training.model_hash(net));tr.audit()
    def test_donor_selection_no_admission_gate_and_corrected_tie_break(self):
        def r(s,e,p,h):return dict(step=s,rr={'ERGAS':e,'PSNR':p},fr_corrected_pan={'HQNR':h},fr={'HQNR':1-h})
        self.assertEqual(plan.pick_donor([r(1,3.6,38,.9),r(2,3.55,37,.95)])['step'],2)
        self.assertEqual(plan.pick_donor([r(1,3.49,38,.9),r(2,3.48,38,.95),r(3,3.6,39,.99)])['step'],2)
    def test_zero_q_and_sheet_all_six_cases(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        x=torch.ones(1,1,5,5)
        for c in plan.queue('s3')[1:]:self.assertTrue(torch.equal(model.adaptive_weight(x,x,x*0,Stats(),x.bool(),c),x))
        with tempfile.TemporaryDirectory() as tmp:
            value=publication.project(tmp,'s3');self.assertEqual(len(value['rows']),12)
            self.assertEqual(value['destination']['sheet_title'],'QB-ablations')
    def test_cutover_does_not_accept_other_server(self):
        info={'Config':{'Labels':{'panda.server':'s2','panda.campaign':'x','panda.source':'y'},'Cmd':[]},'Mounts':[]}
        with self.assertRaises(ValueError):enqueue.authenticate(info,'x','y',[[]])
    def test_stage1_teacher_publication_is_distinct_from_student(self):
        cfg=plan.donor_config('s3')
        record={'step':50000,'rr':{'metrics':{'ERGAS':3.49,'PSNR':38.5}},
            'fr':{'metrics':{'HQNR':.75,'D_s':.22,'D_lambda':.03,'JQM':.8}},
            'fr_corrected_pan':{'metrics':{'HQNR':.94,'D_s':.03,'D_lambda':.03,'JQM':.95}},
            'checkpoint':{'file_sha256':'a'*64},'model_sha256':'b'*64,
            'evidence_path':'fixture','evidence_sha256':'c'*64,'recorded_at':'2026-11-14T00:00:00+00:00',
            'elapsed_seconds':1.,'jqm_variant':'fixture'}
        entries=publication.choose([record],'stage1',cfg)
        for rec,selector,slot in entries:
            row,_=publication.make_row(cfg,'COMPLETED',rec,selector,slot+'_ORIGINAL')
            self.assertEqual(row[publication.INDEX['Role']],'Stage1 Aligner+Teacher')
            self.assertEqual(row[publication.INDEX['HQNR↑']],.75)

if __name__=='__main__':unittest.main()
