from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import plan, assets, predecessor, enqueue
from .install import install


class C3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        install()

    def test_three_c_students_only_same_method_fixed_s3_donor(self):
        allowed={'server','campaign_id','run_id','experiment','repeat','replica','seed','stream_seeds',
            'donor_run_id','donor_step','fixed_donor','donor_campaign_id','donor_selection',
            'official_selection_reference','checkpoint_selection','reference_repeats',
            'evaluation_reference','repetition_comparison','schedule_comparison','config_sha256'}
        rows=plan.queue('s3')
        self.assertEqual([c['seed'] for c in rows],[261018101,261018102,261018103])
        for c in rows:
            self.assertEqual({k:v for k,v in c.items() if k not in allowed},
                             {k:v for k,v in plan.TEMPLATE.items() if k not in allowed})
            self.assertEqual((c['stage'],c['student_input'],c['horizon']),('stage2','CORRECTED_DIFF',30000))
            self.assertEqual((c['server'],c['sensor'],c['donor_step']),('s3','QB',34000))
            self.assertEqual(c['donor_run_id'],plan.DONOR_CONFIG['run_id'])
            self.assertEqual(c['reference_bands'],[2,1,0,3])
            self.assertFalse(c['early_stop']['enabled']);plan.validate(c)
        for server in ('s1','s2'):
            with self.assertRaises(ValueError):plan.queue(server)
        with self.assertRaises(ValueError):plan.resolve('s3','A_RAW')
        with self.assertRaises(ValueError):plan.resolve('s3',plan.CASE,4)

    def test_receipt_rejects_wrong_server_step_or_modified_digest(self):
        fixed=deepcopy(plan.FIXED)
        receipt={'schema':'PANDA_QB_S3_PARALLEL_DONOR_V1','server':'s3','sensor':'QB',
            'reference_mode':'RGBN','donor_run_id':fixed['run_id'],
            'source_config_sha256':fixed['source_config_sha256'],
            'selected':{'step':fixed['step'],'file_sha256':fixed['checkpoint_file_sha256'],
                'model_sha256':fixed['model_sha256'],'evaluation_sha256':fixed['evaluation_sha256']}}
        receipt['receipt_sha256']=plan.digest(receipt);fixed['receipt_sha256']=receipt['receipt_sha256']
        with patch.object(plan,'FIXED',fixed):
            assets.verify_receipt(receipt)
            for key,value in [('server','s2'),('reference_mode','RGB'),('donor_run_id','foreign')]:
                bad=deepcopy(receipt);bad[key]=value
                with self.assertRaises(ValueError):assets.verify_receipt(bad)
            bad=deepcopy(receipt);bad['selected']['step']=32000
            with self.assertRaises(ValueError):assets.verify_receipt(bad)

    def test_real_s3_donor_serialization_three_students_freeze_and_resume(self):
        from panda_sensor_eq40.tests.test_integration import synthetic_source
        from panda_qb_s3_parallel import model as sm,training as st,assets as sa
        from panda_input3 import plan as active,model,training
        from panda_method.calibration import state_dict_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_source(root,'QB');dc=plan.donor_config('s3')
            teacher=sm.Stage1(dc,registered.inventory.manifest())
            with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625,-.375]))
            dt=st.Trainer(dc,teacher);dt.updates=34000;stream=st.stream_for(dc,registered,'cpu')
            pins={'source_release':'synthetic-test-only'}
            meta=st.save(root/'donor.pt',dt,stream,pins,evaluated=True)
            receipt={'selected':{'step':34000,'checkpoint_path':'donor.pt',
                'file_sha256':meta['file_sha256'],'model_sha256':meta['model_sha256']},
                'receipt_sha256':'d'*64,'server':'s3','donor_run_id':dc['run_id']}
            donor=sa._load_selected(dc,receipt,registered,root,pins,device='cpu')
            stats=sa.calibration(donor,registered,root/'calibration',device='cpu',batch_size=1)
            initial=[];controls=[];batch=registered.batch([0])
            for cfg in plan.queue('s3'):
                c=deepcopy(cfg);c.update(batch=1,microbatch=1)
                with patch.object(active,'validate'):
                    net=model.Stage2(c,donor,stats);tr=training.Trainer(c,net)
                initial.append(state_dict_sha256(net.student))
                pred=net.native_prediction(batch)
                self.assertEqual(tuple(pred['x_in'].shape),(1,6,64,64))
                torch.testing.assert_close(pred['x_in'][:,0:1],pred['pan_aligned'],rtol=0,atol=0)
                torch.testing.assert_close(pred['x_in'][:,1:2],pred['pan_aligned']-batch.pan,rtol=0,atol=0)
                terms=net(batch);controls.append(tuple(float(terms[k]) for k in ('weight_mean','valid_fraction','q_zero_fraction')))
                stream=training.stream_for(c,registered,'cpu');tr.step(stream.next_update());tr.audit()
                if c['repeat']==1:
                    checkpoint=root/'student_resume.pt';training.save(checkpoint,tr,stream,pins)
                    expected=tr.step(stream.next_update());sha=training.model_hash(net)
                    training.load(checkpoint,tr,stream,pins)
                    self.assertEqual(tr.step(stream.next_update()),expected)
                    self.assertEqual(training.model_hash(net),sha);tr.audit()
                del net,tr,stream
            self.assertEqual(len(set(initial)),3);self.assertEqual(len(set(controls)),1)
            with (root/'donor.pt').open('ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):sa._load_selected(dc,receipt,registered,root,pins,device='cpu')

    def test_qb_sheet_original_selection_and_dual_fr(self):
        from panda_input3 import publication as pub
        from panda_input3_original.test_reporting import original_record
        from panda_input3_original.policy import choose,passes
        from panda_fr_columns import publication as fr
        from directshift48.reporting import INDEX
        with tempfile.TemporaryDirectory() as tmp:
            value=fr.original_rows(pub.project(tmp,'s3'))
            self.assertEqual(value['destination']['sheet_title'],'QB-ablations')
            self.assertEqual(len(value['rows']),3)
            self.assertEqual(len({r[INDEX['Result_ID']] for r in value['rows']}),3)
        cfg=plan.queue('s3')[0]
        a=original_record(cfg,original_h=.9315,e=3.5,h=.8)
        self.assertTrue(passes(a,cfg))
        row,note=pub.make_row(cfg,'COMPLETED',a,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        self.assertEqual(row[INDEX['Dataset']],'QB');self.assertEqual(row[INDEX['Server']],'s3')
        self.assertEqual(row[INDEX['HQNR↑']],.9315)
        self.assertEqual(json.loads(note)['reference_repeats'],3)
        pub.verify_owned(row,dict(configs={cfg['run_id']:cfg},
            proof={(cfg['run_id'],1000):a},rows=[row],notes=[note]),'s3')
        extras=fr.extras(dict(policy=pub.POLICY,rows=[row],proof={(cfg['run_id'],1000):a}))
        self.assertEqual(next(iter(extras.values()))[0]['userEnteredValue']['numberValue'],.8)
        a['fr']['metrics']['HQNR']=.931499999;a['fr_corrected_pan']['metrics']['HQNR']=.999
        self.assertEqual(choose([a],'stage2',cfg),[])
        self.assertFalse(passes(original_record(cfg,e=3.5000001),cfg))

    def test_qb_data_mounts_keep_donor_and_prior_results_read_only(self):
        fake=['docker','run','--rm','--mount','type=bind,src=/repo,dst=/workspace,readonly','image']
        with patch.object(enqueue.base,'command',return_value=fake) as call:
            cmd=enqueue.command('image',detached=True)
        self.assertEqual(call.call_args.args[0],'s2')  # QB data registry, not GF2
        self.assertIn('panda.server=s3',cmd)
        self.assertEqual(cmd[-3:],['python','-m','panda_qb_s3_input_c3'])
        mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
        self.assertTrue(any(m.endswith('dst=/workspace,readonly') for m in mounts))
        self.assertFalse(any(m.endswith('dst=/workspace/work_dir/panda_qb_s3_parallel') for m in mounts))

    def test_wait_for_gf2_final_publication_without_mutating_state(self):
        expected=predecessor.EXPECTED
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'campaign_state.json';lockpath=root/'runner.lock';lockpath.touch()
            state={'campaign_id':predecessor.gf2.CAMPAIGN_ID,'server':'s3',
                'queue_sha256':plan.digest(expected),'status':'RUNNING',
                'runs':[{'run_id':c['run_id'],'status':'COMPLETED','step':30000} for c in expected]}
            with patch.object(predecessor,'PREVIOUS_ROOT',root):
                path.write_text(json.dumps(state));self.assertFalse(predecessor.finished())
                state['status']='COMPLETED';text=json.dumps(state);path.write_text(text)
                with lockpath.open('r') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);self.assertFalse(predecessor.finished())
                self.assertTrue(predecessor.finished());self.assertEqual(path.read_text(),text)


if __name__=='__main__':unittest.main()
