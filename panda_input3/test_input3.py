from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_unified.core import frozen_predictions,reference_maps,base_from_batch
from panda_eq40_zeroq import model as previous
from . import plan,model,inputs,assets,training,runtime,publication,enqueue


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_plan_paired_seed_same_recipe_and_donor(self):
        for server,sensor,width,step,horizon,lr in [('s1','WV3',112,27000,40000,1.1e-4),
                ('s2','QB',128,32000,50000,1e-4),('s3','GF2',128,42000,50000,2.5e-4)]:
            cfgs=plan.queue(server)
            self.assertEqual([c['student_input'] for c in cfgs],list(inputs.ORDERS))
            self.assertEqual(len({c['seed'] for c in cfgs}),1)
            for c in cfgs:
                self.assertEqual((c['sensor'],c['width'],c['donor_step'],c['horizon'],c['lr']),(sensor,width,step,horizon,lr))
                self.assertEqual((c['alpha'],c['beta'],c['gamma']),(0.,0.,.25))
                self.assertEqual(c['stage'],'stage2');self.assertEqual(c['reference_mode'],'RGBN')
                self.assertEqual(c['sampling_variant'],'UBIL');self.assertEqual(c['teacher_warp'],'LANCZOS4_REFLECTION')
                self.assertFalse(c['reuse_existing']);self.assertFalse(c['early_stop']['enabled'])
                self.assertIsNone(runtime.early_stop_evidence(c,Path('/missing'),10000))
                from panda_stage2_condq.training import TrainingSpec
                ref=TrainingSpec(lr=lr,total_updates=horizon)
                schedule=training.CosineHold(c)
                self.assertTrue(all(schedule.rate(k)==ref.rate(k) for k in (1,100,1000,horizon)))

    def test_serialized_local_donor_all_inputs_guidance_freeze_and_resume(self):
        import importlib
        for server in ('s1','s2','s3'):
            with self.subTest(server=server),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dc=plan.donor_config(server);registered=synthetic_source(root,dc['sensor'])
                package='panda_sensor_eq40' if server=='s3' else 'panda_unified_campaign'
                sm=importlib.import_module(package+'.model');st=importlib.import_module(package+'.training')
                sa=importlib.import_module(package+'.assets')
                teacher=sm.Stage1(dc,registered.inventory.manifest())
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=st.Trainer(dc,teacher);tr.updates=plan.FIXED[dc['sensor']]['step']
                stream=st.stream_for(dc,registered,'cpu');pins={'source_release':'fixture'}
                meta=st.save(root/'donor.pt',tr,stream,pins,evaluated=True)
                selected={'step':tr.updates,'checkpoint_path':'donor.pt','file_sha256':meta['file_sha256'],'model_sha256':meta['model_sha256']}
                receipt={'selected':selected,'receipt_sha256':'a'*64,'server':server,'donor_run_id':dc['run_id']}
                donor=assets.protect(sa._load_selected)(dc,receipt,registered,root,pins,device='cpu')
                stats=assets.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
                batch=registered.batch([0]);initial_dual=None;controls=[]
                for c in plan.queue(server):
                    cfg=deepcopy(c);cfg.update(batch=1,microbatch=1)
                    with patch.object(plan,'validate'):net=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,net)
                    with patch.object(previous.plan,'validate'):old=previous.Stage2(cfg,donor,stats)
                    out=net.native_prediction(batch)
                    expected=inputs.pan_views(batch.pan,out['pan_aligned'],cfg['student_input'])
                    torch.testing.assert_close(out['x_in'][:,:cfg['student_pan_channels']],expected,rtol=0,atol=0)
                    self.assertEqual(out['pan_aligned'].shape,(1,1,64,64))
                    old_pan=batch.pan if cfg['student_input']=='RAW' else out['pan_aligned']
                    torch.testing.assert_close(out['y'],old.reconstruct(batch,old_pan,base_from_batch(batch,'interp23'))['y'],rtol=1e-6,atol=1e-7)
                    if cfg['student_input']=='RAW_CORRECTED':initial_dual=out['y'].detach().clone()
                    if cfg['student_input']=='CORRECTED_DIFF':torch.testing.assert_close(initial_dual,out['y'],rtol=1e-6,atol=1e-7)
                    terms=net(batch);controls.append(tuple(float(terms[k]) for k in ('weight_mean','valid_fraction','q_zero_fraction')))
                    self.assertEqual(net.student.manifest()['input_order'][:cfg['student_pan_channels']],inputs.ORDERS[cfg['student_input']])
                    stream=training.stream_for(cfg,registered,'cpu');trainer.step(stream.next_update());trainer.audit()
                    if cfg['student_input']=='RAW_CORRECTED':
                        ck=root/'resume.pt';training.save(ck,trainer,stream,pins)
                        second=trainer.step(stream.next_update());sha=training.model_hash(net)
                        training.load(ck,trainer,stream,pins)
                        self.assertEqual(second,trainer.step(stream.next_update()));self.assertEqual(sha,training.model_hash(net))
                        other=inputs.build_student(cfg['sensor'],cfg['width'],cfg['stream_seeds']['S_init'],'CORRECTED_DIFF')
                        with self.assertRaises(RuntimeError):other.load_state_dict(net.student.state_dict())
                    del old,net,trainer
                self.assertEqual(len(set(controls)),1)

    def test_zero_q_no_extra_weight_for_all_inputs(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        x=torch.ones(1,1,5,5)
        for server in ('s1','s2','s3'):
            for c in plan.queue(server):
                self.assertTrue(torch.equal(model.adaptive_weight(x,x,x*0,Stats(),x.bool(),c),x))

    def test_sheet_sensor_and_same_row_corrected_reference(self):
        from panda_fr_columns.publication import original_rows,extras
        from panda_fr_columns.schema import expand
        for server in ('s1','s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                v=original_rows(publication.project(tmp,server));self.assertEqual(len(v['rows']),3)
                self.assertEqual(v['destination']['sheet_title'],plan.sensor_for(server)+'-ablations')
                self.assertTrue(all(r[publication.INDEX['Bucket']]=='ablation' for r in v['rows']))
            cfg=plan.queue(server)[0];receipt={'schema':'PANDA_INPUT3_FIXED_DONOR_V1','server':server,
                'donor_run_id':cfg['donor_run_id'],'selected':{'step':cfg['donor_step']}}
            receipt['receipt_sha256']=plan.digest(receipt)
            record={'step':cfg['horizon'],'donor_selection':receipt,
                'rr':{'metrics':{'ERGAS':2.123456,'PSNR':38.123456}},
                'fr':{'metrics':{'HQNR':.812345,'D_s':.15,'D_lambda':.02,'JQM':.8}},
                'fr_corrected_pan':{'metrics':{'HQNR':.934567,'D_s':.04,'D_lambda':.02,'JQM':.9}},
                'checkpoint':{'file_sha256':'b'*64},'model_sha256':'c'*64,'evidence_path':'fixture',
                'evidence_sha256':'d'*64,'recorded_at':'2026-11-15T00:00:00+00:00','elapsed_seconds':1.,'jqm_variant':'fixture'}
            row,note=publication.make_row(cfg,'COMPLETED',record,'FIXED'+str(cfg['horizon']//1000)+'K','ROW0_ORIGINAL')
            x=extras({'rows':[row],'proof':{(cfg['run_id'],cfg['horizon']):record}})
            corr=[c['userEnteredValue']['numberValue'] for c in next(iter(x.values()))]
            full=expand(row,corr);self.assertEqual(full[3],.8123);self.assertEqual(full[7:10],[.9346,.04,.02])
            self.assertEqual(json.loads(note)['configuration']['guidance_pan'],'CORRECTED_PAN_FOR_ALL_CASES')

    def test_pause_preserves_completed_and_retains_remaining_jobs(self):
        from panda_wv3_s2abl5 import plan as old,enqueue as old_launcher
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);oldroot=root/'old';oldroot.mkdir()
            state={'runs':[{'run_id':'done','status':'COMPLETED','step':40000},
                           {'run_id':'remaining','status':'QUEUED','step':0}]}
            (oldroot/'campaign_state.json').write_text(json.dumps(state))
            with patch.object(plan,'ROOT',root),patch.object(old,'campaign_root',return_value=oldroot),\
                 patch.object(enqueue.subprocess,'check_output',return_value=''),\
                 patch.object(enqueue.subprocess,'run'),patch.object(old_launcher,'command',return_value=['fixture']):
                enqueue.pause_campaign('panda_wv3_s2abl5','s1','image')
            updated=json.loads((oldroot/'campaign_state.json').read_text())
            self.assertEqual([r['status'] for r in updated['runs']],['COMPLETED','HELD_BY_USER'])
            self.assertEqual(updated['status'],'PAUSED_BY_USER_INPUT3_PRIORITY')
        info={'Config':{'Labels':{'panda.server':'s2'},'Cmd':[]},'Mounts':[]}
        with self.assertRaises(ValueError):enqueue.authenticate(info,'s1','x','y',[[]])


if __name__=='__main__':unittest.main()
