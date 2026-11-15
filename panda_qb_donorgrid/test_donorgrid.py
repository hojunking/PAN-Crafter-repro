from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_eq40_zeroq import model as old_s2
from panda_stage2_condq.training import TrainingSpec
from . import plan, assets, model, training, runtime, publication, enqueue


class DonorGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_existing_sources_and_paired_grid(self):
        for server in ('s2','s3'):
            cfgs=plan.queue(server)
            self.assertEqual(len(cfgs),len(plan.CANDIDATE_STEPS)*len(plan.SCHEDULES))
            self.assertEqual(len({c['run_id'] for c in cfgs}),len(cfgs))
            self.assertEqual({c['stage'] for c in cfgs},{'stage2'})
            self.assertEqual({c['donor_step'] for c in cfgs},{40000,48000,50000})
            self.assertEqual(len({plan.digest(c['stream_seeds']) for c in cfgs}),1)
            self.assertEqual(plan.donor_config(server)['server'],server)
            for c in cfgs:
                plan.assert_method(c)
                self.assertFalse(c['reuse_existing'])
                self.assertEqual(c['horizon'],50000)
                self.assertEqual(c['donor_run_id'],plan.donor_config(server)['run_id'])
                self.assertFalse(c['early_stop']['enabled'])
                for step in (10000,20000,50000):
                    self.assertIsNone(runtime.early_stop_evidence(c,Path('/missing'),step))
            self.assertEqual(plan.targets('QB'),{'HQNR_min':.9315,'ERGAS_max':3.5,'PSNR_min':None})
        with self.assertRaises(ValueError):plan.queue('s1')

    def test_cosine_baseline_and_hold_through_50k(self):
        for cfg in plan.queue('s2'):
            schedule=training.CosineHold(cfg)
            end=cfg['schedule']['decay_end'];floor=cfg['schedule']['min_lr']
            ref=TrainingSpec(lr=cfg['lr'],total_updates=end,warmup_updates=cfg['warmup'],min_lr=floor)
            for step in (1,100,1000,end-1,end,50000):
                self.assertEqual(schedule.rate(step),ref.rate(min(step,end)))
            self.assertEqual(schedule.rate(50000),floor)
            with self.assertRaises(ValueError):schedule.rate(50001)
        baseline=training.CosineHold(plan.queue('s2')[0])
        self.assertTrue(all(baseline.rate(s)==TrainingSpec().rate(s) for s in range(1,50001)))

    def test_fixed_selection_separate_calibration_receipts(self):
        rows=[dict(step=s,checkpoint_path=str(s),file_sha256=str(s),model_sha256='m',
                   evaluation_sha256='e',rr={'ERGAS':999-s/1000},fr={'HQNR':s/100000})
              for s in plan.CANDIDATE_STEPS]
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dc=plan.donor_config(server)
                with patch.object(assets,'check_source',return_value=(dc,root/'source',{'source_release':'fixture'},rows)):
                    for cfg in plan.queue(server):
                        _,receipt,_,_=assets.select(cfg,None,root,{})
                        self.assertEqual(receipt['selected']['step'],cfg['donor_step'])
                        self.assertFalse(receipt['uses_test_for_selection'])
                        self.assertEqual(receipt['server'],server)
                        self.assertTrue((plan.reference_root(root,cfg['donor_step'])/'selection.json').exists())
                    self.assertEqual(len(list((root/'donors').glob('*/selection.json'))),3)

    def test_actual_existing_checkpoint_load_method_parity_and_resume(self):
        import importlib
        for server in ('s2','s3'):
            with self.subTest(server=server),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);registered=synthetic_source(root,'QB');dc=plan.donor_config(server)
                sm=importlib.import_module(assets.package(server)+'.model')
                st=importlib.import_module(assets.package(server)+'.training')
                teacher=sm.Stage1(dc,registered.inventory.manifest())
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=st.Trainer(dc,teacher);ds=st.stream_for(dc,registered,'cpu');pins={'source_release':'fixture'}
                # Serialize a source evaluated model; the fixture step has no optimizer history.
                tr.updates=40000
                meta=st.save(root/'donor.pt',tr,ds,pins,evaluated=True)
                selected=dict(step=40000,checkpoint_path='donor.pt',file_sha256=meta['file_sha256'],model_sha256=meta['model_sha256'])
                receipt=dict(selected=selected,receipt_sha256='d'*64,server=server,uses_test_for_selection=False)
                donor=assets.load_candidate(dc,selected,registered,root,pins,'cpu',receipt)
                stats=assets.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
                cfg=deepcopy(plan.queue(server)[0]);cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):
                    net=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,net)
                with patch.object(old_s2.plan,'validate'):baseline=old_s2.Stage2(cfg,donor,stats)
                batch=registered.batch([0]);out=net.native_prediction(batch)
                self.assertEqual(out['x_in'].shape,(1,5,64,64))
                self.assertTrue(torch.equal(out['y'],baseline.native_prediction(batch)['y']))
                self.assertTrue(torch.equal(net(batch)['loss'],baseline(batch)['loss']))
                stream=training.stream_for(cfg,registered,'cpu');trainer.step(stream.next_update())
                ck=root/'resume.pt';training.save(ck,trainer,stream,pins)
                second=trainer.step(stream.next_update());expected=training.model_hash(net)
                training.load(ck,trainer,stream,pins)
                self.assertEqual(second,trainer.step(stream.next_update()))
                self.assertEqual(expected,training.model_hash(net));trainer.audit()
                wrong=deepcopy(cfg);wrong['donor_step']=50000
                with patch.object(plan,'validate'),self.assertRaises(ValueError):model.Stage2(wrong,donor,stats)
                from panda_unified.core import support,ReferenceSpatial
                self.assertEqual(int(support(batch.pan,torch.tensor([[.5,.5]])).sum()),55*55)
                torch.testing.assert_close(ReferenceSpatial('QB','RGBN').reference(batch.gt),batch.gt.mean(1,keepdim=True))

    def test_resume_across_decay_to_hold(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.aligner=torch.nn.Linear(1,1).requires_grad_(False)
                self.teacher=torch.nn.Linear(1,1).requires_grad_(False)
                self.student=torch.nn.Linear(1,1)
            def forward(self,b):return {'loss':self.student(b.pan).square().mean()}
        class Stream:
            def state_dict(self):return {'fixture':True}
            def load_state_dict(self,value):assert value=={'fixture':True}
        cfg=deepcopy(plan.queue('s2')[0]);cfg.update(batch=1,microbatch=1,horizon=5,warmup=1,
            schedule={'name':'WARMUP_COSINE_HOLD','decay_end':3,'min_lr':1e-6})
        with tempfile.TemporaryDirectory() as tmp,patch.object(plan,'validate'):
            net=Tiny();trainer=training.Trainer(cfg,net);stream=Stream();b=[SimpleNamespace(pan=torch.ones(1,1))]
            for _ in range(3):trainer.step(b)
            path=Path(tmp)/'state.pt';training.save(path,trainer,stream,{})
            expected=[trainer.step(b),trainer.step(b)];sha=training.model_hash(net)
            training.load(path,trainer,stream,{})
            self.assertEqual(expected,[trainer.step(b),trainer.step(b)])
            self.assertEqual(sha,training.model_hash(net));self.assertEqual(expected[-1]['lr'],1e-6)

    def test_zero_q_and_sheet_routing(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        x=torch.ones(1,1,5,5)
        from panda_fr_columns.publication import original_rows
        for server in ('s2','s3'):
            for cfg in plan.queue(server):
                self.assertTrue(torch.equal(model.adaptive_weight(x,x,x*0,Stats(),x.bool(),cfg),x))
            with tempfile.TemporaryDirectory() as tmp:
                value=original_rows(publication.project(tmp,server))
                self.assertEqual(len(value['rows']),len(plan.queue(server)))
                self.assertEqual(value['destination']['sheet_title'],'QB-ablations')
                self.assertTrue(all(r[publication.INDEX['Bucket']]=='main' for r in value['rows']))
        c=plan.queue('s2')[0]
        records=[dict(step=4000,fr_corrected_pan=dict(metrics=dict(HQNR=.925)),rr=dict(metrics=dict(ERGAS=3.4,PSNR=38))),
                 dict(step=50000,fr_corrected_pan=dict(metrics=dict(HQNR=.932)),rr=dict(metrics=dict(ERGAS=3.6,PSNR=39)))]
        self.assertEqual(publication.choose(records,'stage2',c)[0][0]['step'],50000)
        self.assertFalse(any(publication.joint_pass(r,c) for r in records))

    def test_old_evidence_mounts_and_wrong_server_rejected(self):
        fake=['docker','run','--rm','--mount','type=bind,src=/tmp/old,dst=/workspace/work_dir/panda_unified_campaign','image']
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp,patch.object(plan,'ROOT',Path(tmp)),patch.object(enqueue.base,'command',return_value=list(fake)):
                cmd=enqueue.command(server,'image')
            mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
            self.assertTrue(mounts[0].endswith(',readonly'))
            self.assertIn('dst=/workspace/work_dir/panda_qb_donorgrid',mounts[1])
        info={'Config':{'Labels':{'panda.server':'s1'},'Cmd':[]},'Mounts':[]}
        with self.assertRaises(ValueError):enqueue.authenticate(info,'s2','x','y',[[]])

    def test_fr67_same_row_metrics_and_donor_provenance(self):
        from panda_fr_columns.publication import extras,original_rows
        from panda_fr_columns.schema import expand
        cfg=plan.queue('s2')[0]
        receipt={'schema':'PANDA_QB_DONORGRID_DONOR_V1','donor_run_id':cfg['donor_run_id'],
            'server':'s2','selected':{'step':40000,'file_sha256':'a'*64},
            'uses_test_for_selection':False,'candidates':[{}, {}, {}]}
        receipt['receipt_sha256']=plan.digest(receipt)
        rec={'step':50000,'rr':{'metrics':{'ERGAS':3.49994,'PSNR':38.4}},
            'fr':{'metrics':{'HQNR':.812345,'D_s':.15,'D_lambda':.02,'JQM':.8}},
            'fr_corrected_pan':{'metrics':{'HQNR':.934567,'D_s':.04,'D_lambda':.02,'JQM':.9}},
            'donor_selection':receipt,'checkpoint':{'file_sha256':'b'*64},'model_sha256':'c'*64,
            'evidence_path':'fixture','evidence_sha256':'d'*64,'recorded_at':'2026-11-15T00:00:00+00:00',
            'elapsed_seconds':1,'jqm_variant':'fixture'}
        rows=[];notes=[]
        for ref in ('ORIGINAL','CORRECTED'):
            row,note=publication.make_row(cfg,'COMPLETED',rec,'PASS_ALL_STEPS','STEP_0050000_'+ref)
            rows.append(row);notes.append(note)
        values=extras(original_rows({'rows':rows,'notes':notes,'proof':{(cfg['run_id'],50000):rec}}))
        self.assertEqual(len(values),1)
        corr=[c['userEnteredValue']['numberValue'] for c in next(iter(values.values()))]
        row67=expand(rows[0],corr)
        self.assertEqual(row67[3],.8123);self.assertEqual(row67[7:10],[.9346,.04,.02])
        self.assertEqual(len(row67),67)
        note=json.loads(notes[0]);self.assertEqual(note['donor_selection']['selected']['step'],40000)
        self.assertEqual(note['publication']['targets']['HQNR_min'],.9315)

    def test_cutover_preserves_completed_and_failed_jobs(self):
        from panda_sensor_main50 import plan as old,enqueue as old_launcher
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);oldroot=root/'old';oldroot.mkdir()
            state={'runs':[{'run_id':'done','status':'COMPLETED','step':50000},
                {'run_id':'failed','status':'FAILED','step':19},
                {'run_id':'remaining','status':'QUEUED','step':0}]}
            (oldroot/'campaign_state.json').write_text(json.dumps(state))
            with patch.object(plan,'ROOT',root),patch.object(old,'campaign_root',return_value=oldroot),\
                 patch.object(enqueue.subprocess,'check_output',return_value=''),\
                 patch.object(enqueue.subprocess,'run') as run,patch.object(old_launcher,'command',return_value=['fixture']):
                enqueue.stop_campaign('panda_sensor_main50','s2','image')
            after=json.loads((oldroot/'campaign_state.json').read_text())
            self.assertEqual([r['status'] for r in after['runs']],['COMPLETED','FAILED','CANCELLED_BY_USER'])
            self.assertEqual(run.call_args.args[0],['fixture','publish','--server','s2','--apply'])
        for server in ('s2','s3'):
            with patch.object(enqueue,'stop_campaign') as stop:
                enqueue.stop_old(server,'image')
                self.assertEqual([c.args[0] for c in stop.call_args_list],
                    ['panda_sensor_main50' if server=='s2' else 'panda_qb_s3_parallel','panda_qb_lateval'])


if __name__=='__main__':unittest.main()
