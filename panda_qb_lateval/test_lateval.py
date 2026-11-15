from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same
from panda_sensor_eq40.tests.test_integration import synthetic_source
from panda_unified_campaign import model as old_s1, training as old_training
from panda_eq40_zeroq import model as old_s2
from panda_qb_s3_parallel import model as s3_model, training as s3_training
from panda_stage2_condq.training import TrainingSpec
from . import plan,assets,model,training,publication,runtime,enqueue,reuse


class LatevalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_method_one_case_per_server_and_full50(self):
        for server in ('s2','s3'):
            self.assertEqual(len(plan.queue(server)),1)
            c=plan.resolve(server);d=plan.donor_config(server)
            plan.assert_method(c);plan.assert_method(d)
            self.assertEqual((c['horizon'],c['seed'],c['lr'],c['gamma']),(50000,261018101,1e-4,.25))
            self.assertEqual(c['donor_run_id'],d['run_id']);self.assertEqual(d['server'],server)
            self.assertFalse(c['early_stop']['enabled'])
            self.assertEqual(c['donor_selection']['candidate_steps'],[40000,48000,50000])
            self.assertEqual(plan.targets('QB')['HQNR_min'],.9315)
            _,baseline=plan.baseline(server);self.assertTrue(reuse.same_recipe(c,baseline))
            for step in (10000,20000,50000):self.assertIsNone(runtime.early_stop_evidence(c,Path('/missing'),step))
            self.assertEqual(TrainingSpec(total_updates=50000).rate(50000),0)
            for key,bad in [('reference_bands',[2,1,0]),('width',112),('teacher_warp','BICUBIC'),('sampling_variant','BASE')]:
                wrong=deepcopy(c);wrong[key]=bad
                with self.assertRaises(ValueError):plan.assert_method(wrong)
        with self.assertRaises(ValueError):plan.resolve('s1')

    def test_selection_ignores_test_metrics_and_missing_candidates_fail(self):
        def row(step,rre,r90,e,h=1.):return dict(step=step,validation=dict(RRE50=rre,RRE90=r90,Val_ERGAS=e),
            rr=dict(ERGAS=0.1 if step==40000 else 999),fr_corrected_pan=dict(HQNR=h))
        rows=[row(40000,.3,.4,1),row(48000,.2,.5,9),row(50000,.25,.3,0)]
        self.assertEqual(assets.pick(rows)['step'],48000)
        rows=[row(40000,.2,.4,2),row(48000,.2,.4,1),row(50000,.2,.4,1)]
        self.assertEqual(assets.pick(rows)['step'],48000)
        with self.assertRaises(ValueError):assets.pick(rows[:2])

    def test_real_local_load_calibration_method_parity_frozen_and_resume(self):
        for server in ('s2','s3'):
            with self.subTest(server=server),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r=synthetic_source(root,'QB');dc=plan.donor_config(server)
                sm,st=(old_s1,old_training) if server=='s2' else (s3_model,s3_training)
                teacher=sm.Stage1(dc,r.inventory.manifest())
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
                tr=st.Trainer(dc,teacher);ds=st.stream_for(dc,r,'cpu');pins={'source_release':'synthetic'}
                meta=st.save(root/'donor.pt',tr,ds,pins,evaluated=True)
                selected=dict(step=0,checkpoint_path='donor.pt',file_sha256=meta['file_sha256'],model_sha256=meta['model_sha256'])
                donor=assets.load_candidate(dc,selected,r,root,pins,'cpu')
                stats=assets.calibration(donor,r,root/'calibration',device='cpu',batch_size=1)
                self.assertTrue(reuse.stats_equal(stats,stats))
                cfg=deepcopy(plan.resolve(server));cfg.update(batch=1,microbatch=1)
                with patch.object(plan,'validate'):
                    net=model.Stage2(cfg,donor,stats);trainer=training.Trainer(cfg,net)
                with patch.object(old_s2.plan,'validate'):reference=old_s2.Stage2(cfg,donor,stats)
                batch=r.batch([0]);out=net.native_prediction(batch)
                self.assertEqual(out['x_in'].shape,(1,5,64,64))
                self.assertTrue(torch.equal(out['y'],reference.native_prediction(batch)['y']))
                self.assertTrue(torch.equal(net(batch)['loss'],reference(batch)['loss']))
                self.assertTrue(torch.equal(teacher.native_prediction(batch)['pan_aligned'],out['pan_aligned']))
                stream=training.stream_for(cfg,r,'cpu');trainer.step(stream.next_update());trainer.audit()
                ck=root/'resume.pt';training.save(ck,trainer,stream,pins)
                second=trainer.step(stream.next_update());expected=training.model_hash(net)
                training.load(ck,trainer,stream,pins)
                self.assertEqual(second,trainer.step(stream.next_update()));self.assertEqual(expected,training.model_hash(net))
                trainer.audit()
                # Both reconstruction/spatial reference use the same actual common mask.
                from panda_unified.core import support,ReferenceSpatial
                mask=support(batch.pan,torch.tensor([[.5,.5]]))
                self.assertEqual(int(mask.sum()),55*55)
                spatial=ReferenceSpatial('QB','RGBN')
                torch.testing.assert_close(spatial.reference(batch.gt),batch.gt.mean(1,keepdim=True))

    def test_zero_q_and_four_decimal_dual_pan_reporting(self):
        class Stats:
            tau_e=1.
            def lookup_exact(self,g,q):return torch.ones_like(q)
        x=torch.ones(1,1,5,5)
        for server in ('s2','s3'):
            c=plan.resolve(server)
            self.assertTrue(torch.equal(model.adaptive_weight(x,x,x*0,Stats(),x.bool(),c),x))
            with tempfile.TemporaryDirectory() as tmp:
                value=publication.project(tmp,server)
                self.assertEqual(value['destination']['sheet_title'],'QB-ablations')
                self.assertEqual(len(value['rows']),2)
                self.assertTrue(all(r[publication.INDEX['Bucket']]=='main' for r in value['rows']))
        c=plan.resolve('s2')
        rows=[dict(step=4000,fr_corrected_pan=dict(metrics=dict(HQNR=.925)),rr=dict(metrics=dict(ERGAS=3.4,PSNR=38))),
              dict(step=50000,fr_corrected_pan=dict(metrics=dict(HQNR=.932)),rr=dict(metrics=dict(ERGAS=3.6,PSNR=39)))]
        selected=publication.choose(rows,'stage2',c)
        self.assertEqual(selected[0][0]['step'],50000)
        self.assertFalse(publication.joint_pass(rows[0],c));self.assertFalse(publication.joint_pass(rows[1],c))

    def test_source_mounts_read_only_and_cutover_authentication(self):
        fake=['docker','run','--rm','--mount','type=bind,src=/tmp/old,dst=/workspace/work_dir/panda_unified_campaign','image']
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp,patch.object(plan,'ROOT',Path(tmp)),patch.object(enqueue.base,'command',return_value=list(fake)):
                cmd=enqueue.command(server,'image')
            mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
            self.assertTrue(mounts[0].endswith(',readonly'))
            self.assertIn('dst=/workspace/work_dir/panda_qb_lateval',mounts[1])
            self.assertEqual(cmd[-3:],['python','-m','panda_qb_lateval'])
        info={'Config':{'Labels':{'panda.server':'s1'},'Cmd':[]},'Mounts':[]}
        with self.assertRaises(ValueError):enqueue.authenticate(info,'s2','x','y',[[]])

    def test_cached_validation_identity_and_selection(self):
        # Exercises selection's read-only source reuse and an absent48K recomputation.
        from types import SimpleNamespace
        rows=[{'step':s,'file_sha256':str(s),'checkpoint_path':str(s)} for s in plan.CANDIDATE_STEPS]
        inventory={'inventory_sha256':'inv'}
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=plan.donor_config(server);directory=root/'old/runs'/cfg['run_id'];directory.mkdir(parents=True)
                for step in (40000,50000):
                    val={'RRE50':.3,'RRE90':.4,'Val_ERGAS':4.,'validation_inventory_sha256':'inv'}
                    (directory/f'validation_{step:07d}.json').write_text(json.dumps(val))
                    (directory/f'evaluation_{step:07d}.json').write_text(json.dumps({'validation':val}))
                r=SimpleNamespace(sources='fixture')
                def score(v,inv):return {k:v[k] for k in ('RRE50','RRE90','Val_ERGAS')}
                with patch.object(assets,'check_source',return_value=(cfg,root/'old',{'source_release':'source'},deepcopy(rows))),\
                     patch.object(assets,'validation_inventory',return_value=inventory),\
                     patch.object(assets,'validate_score',side_effect=score),\
                     patch.object(assets,'load_candidate',return_value=object()),\
                     patch.object(assets,'collect_validation',return_value={'RRE50':.2,'RRE90':.3,'Val_ERGAS':9.}) as collect:
                    _,receipt,_,_=assets.select(server,r,root/'new',{},'cpu')
                    self.assertEqual(receipt['selected']['step'],48000);self.assertFalse(receipt['uses_test_for_selection'])
                    self.assertEqual(collect.call_count,1)
                    _,again,_,_=assets.select(server,r,root/'new',{},'cpu')
                    self.assertEqual(receipt,again);self.assertEqual(collect.call_count,1)

    def test_native_validation_and_cached_score_integrity(self):
        from panda_stage2_condq.sources import RegisteredRRTrain
        from panda_unified_campaign.selection import validation_inventory,collect_validation
        from ta2.evaluation import file_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);synthetic_source(root,'QB')
            catalog=json.loads((root/'catalog.json').read_text())
            catalog['sensors']['QB']['splits']['val']=deepcopy(catalog['sensors']['QB']['splits']['train'])
            (root/'catalog.json').write_text(json.dumps(catalog))
            r=RegisteredRRTrain.from_catalog(root/'catalog.json',root=root,sensor='QB',
                lms_overrides={s:{'path':'train.h5','sha256':file_sha(root/'train.h5'),'reason':'fixture'} for s in ('train','val')})
            cfg=plan.donor_config('s2');net=old_s1.Stage1(cfg,r.inventory.manifest())
            inventory=validation_inventory(r.sources,'RGBN')
            actual=collect_validation(net,r.sources,inventory=inventory,include_ergas=True,epsilon=True,batch_size=12)
            score=assets.validate_score(actual,inventory)
            self.assertEqual(score['sample_count'],12)
            self.assertTrue(actual['protocol']['uses_test_for_selection'] is False)
            bad=deepcopy(actual);bad['RRE50']+=1
            with self.assertRaises(ValueError):assets.validate_score(bad,inventory)

    def test_fr67_pairs_keep_original_and_corrected_columns_and_provenance(self):
        from panda_fr_columns.publication import extras,original_rows
        from panda_fr_columns.schema import expand
        cfg=plan.resolve('s2')
        receipt={'schema':'PANDA_QB_LATEVAL_DONOR_V1','donor_run_id':cfg['donor_run_id'],
            'selected':{'step':48000,'file_sha256':'a'*64},'uses_test_for_selection':False,'candidates':[{}, {}, {}]}
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
        cells=next(iter(values.values()))
        corr=[c['userEnteredValue']['numberValue'] for c in cells]
        row67=expand(rows[0],corr)
        self.assertEqual(row67[3],.8123);self.assertEqual(row67[7:10],[.9346,.04,.02])
        self.assertEqual(len(row67),67)

    def test_source_admission_rejects_wrong_server_missing_or_tampered_checkpoint(self):
        from types import SimpleNamespace
        from ta2.evaluation import file_sha
        for server in ('s2','s3'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=plan.donor_config(server)
                directory=root/'runs'/cfg['run_id'];(directory/'models').mkdir(parents=True)
                pins={'source_release':'fixture','evaluator':{'id':'e'},'train_inventory':{'id':'t'},'split_manifests':{'id':'s'}}
                (root/'admission.json').write_text(json.dumps(pins))
                state={'campaign_id':cfg['campaign_id'],'server':server,'runs':[{'run_id':cfg['run_id'],'status':'COMPLETED','step':50000}]}
                (root/'campaign_state.json').write_text(json.dumps(state));(directory/'config.json').write_text(json.dumps(cfg))
                for step in plan.CANDIDATE_STEPS:
                    ck=directory/'models'/f'step_{step:07d}.pt';ck.write_bytes(b'byte-identity-fixture')
                    meta={'step':step,'path':str(ck),'file_sha256':file_sha(ck),'model_sha256':'m'*64,'config_sha256':cfg['config_sha256']}
                    ck.with_suffix('.json').write_text(json.dumps(meta))
                    ev={'checkpoint':meta,'step':step,'model_sha256':'m'*64,'eval_protocol_id':plan.digest(pins['evaluator']),
                        'rr':{'metrics':{}},'fr':{'metrics':{}}}
                    (directory/f'evaluation_{step:07d}.json').write_text(json.dumps(ev))
                registered=SimpleNamespace(verify=lambda:SimpleNamespace(manifest=lambda:pins['train_inventory']))
                source_eval=importlib.import_module(assets.package(server)+'.evaluation')
                with patch.object(plan,'donor_root',return_value=root),patch.object(source_eval,'verify_record'):
                    result=assets.check_source(server,registered,pins)
                    self.assertEqual([r['step'] for r in result[3]],list(plan.CANDIDATE_STEPS))
                    state['server']='s1';(root/'campaign_state.json').write_text(json.dumps(state))
                    with self.assertRaises(ValueError):assets.check_source(server,registered,pins)
                    state['server']=server;(root/'campaign_state.json').write_text(json.dumps(state))
                    ck=directory/'models/step_0048000.pt';ck.write_bytes(b'changed')
                    with self.assertRaises(ValueError):assets.check_source(server,registered,pins)
                    ck.unlink()
                    with self.assertRaises(FileNotFoundError):assets.check_source(server,registered,pins)

    def test_replacement_preserves_completed_and_failed_jobs(self):
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
                enqueue.stop_old('s2','image')
            after=json.loads((oldroot/'campaign_state.json').read_text())
            self.assertEqual([r['status'] for r in after['runs']],['COMPLETED','FAILED','CANCELLED_BY_USER'])
            self.assertTrue((plan.campaign_root('s2')/'cutover').name=='cutover')
            self.assertEqual(run.call_args.args[0],['fixture','publish','--server','s2','--apply'])


if __name__=='__main__':unittest.main()
