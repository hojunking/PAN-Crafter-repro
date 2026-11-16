from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import plan,predecessor,enqueue


class DonorCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from .install import install
        install()

    def test_five_predeclared_jobs_architecture_alpha_and_paired_streams(self):
        queue=plan.queue('s2')
        self.assertEqual([(c['donor_step'],c['repeat']) for c in queue],list(plan.JOBS))
        self.assertEqual(len({json.dumps(c['stream_seeds'],sort_keys=True) for c in queue[:3]}),1)
        self.assertEqual([c['seed'] for c in queue if c['donor_step']==48000],[261018101,261018102,261018103])
        for c in queue:
            self.assertEqual((c['student_input'],c['width'],c['stage1_backbone']),('CORRECTED_DIFF',128,'UBIL'))
            self.assertEqual((c['alpha'],c['beta'],c['gamma'],c['lr'],c['horizon']),(0,0,.25,1e-4,30000))
            self.assertEqual(c['reference_mode'],'RGBN');self.assertFalse(c['early_stop']['enabled']);plan.validate(c)
            for key,value in [('alpha',.25),('width',160),('donor_step',32000),('student_input','CORRECTED')]:
                wrong=deepcopy(c);wrong[key]=value
                with self.assertRaises(ValueError):plan.validate(wrong)
        with self.assertRaises(ValueError):plan.queue('s3')
        with self.assertRaises(ValueError):plan.resolve('s2','D40_C30',2)

    def test_predecessors_must_finish_and_no_other_queue_is_mutated(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(plan,'ROOT',Path(tmp)):
            for (name,root),(_,_,campaign) in zip(predecessor.roots(),predecessor.PREVIOUS):
                root.mkdir(parents=True);(root/'runner.lock').touch()
                prefix='INPUT3' if name=='original' else 'INPUT3C30';repeat=2 if name=='repeat' else 1
                rows=[dict(run_id=f'{prefix}_QB_S2_{c}_R{repeat}_S{261018100+repeat}',status='COMPLETED',step=30000)
                      for c in ('A_RAW','B_RAW_CORRECTED','C_CORRECTED_DIFF')]
                (root/'campaign_state.json').write_text(json.dumps(dict(campaign_id=campaign,server='s2',runs=rows)))
            before={str(r): (r/'campaign_state.json').read_bytes() for _,r in predecessor.roots()}
            predecessor.require_finished()
            self.assertEqual(before,{str(r):(r/'campaign_state.json').read_bytes() for _,r in predecessor.roots()})
            root=predecessor.roots()[-1][1];p=root/'campaign_state.json';s=json.loads(p.read_text())
            s['runs'][-1]['status']='RUNNING';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):predecessor.require_finished()

    def test_mounts_only_new_output_and_old_publication_are_writable(self):
        fake=['docker','run','--rm','--mount','type=bind,src=/repo,dst=/workspace,readonly',
              '--mount','type=bind,src=/old,dst=/workspace/work_dir/panda_unified_campaign','image']
        with tempfile.TemporaryDirectory() as tmp,patch.object(plan,'ROOT',Path(tmp)),\
             patch.object(enqueue.base,'command',return_value=fake):
            cmd=enqueue.command('image')
        mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
        self.assertIn('type=bind,src=/old,dst=/workspace/work_dir/panda_unified_campaign,readonly',mounts)
        writable=[m for m in mounts if not m.endswith(',readonly')]
        self.assertEqual(len(writable),4)
        self.assertTrue(all('/publication' in m or m.endswith('/workspace/work_dir/panda_qb_s2_donor_c') for m in writable))

    def test_real_transport_restores_fallback_and_dual_FR_without_foreign_changes(self):
        from panda_input3 import publication as pub
        from panda_input3_original.test_reporting import original_record
        from panda_input3_original import reporting as old
        from panda_qb_repfix import reporting as fix
        from panda_fr_columns import publication as fr
        from panda_fr_columns.adapter import GoogleSheets
        from panda_fr_columns.schema import BOOK,HEADERS67,expand
        from sheet_publish_v2.transport import _typed,_cell_value
        from directshift48.reporting import INDEX
        cfg=plan.queue('s2')[0];r=original_record(cfg,original_h=.90,h=.96,e=3.51)
        with fix.old_context():blank,note=old.make_row(cfg,'COMPLETED',None,'STATUS','ROW0_ORIGINAL')
        row,raw=pub.make_row(cfg,'COMPLETED',r,'BEST_HQNR_FALLBACK','ROW0_ORIGINAL')
        value=dict(policy=pub.POLICY,destination=pub.destination('s2'),rows=[row],notes=[raw],conflicts=[],
            configs={cfg['run_id']:cfg},proof={(cfg['run_id'],1000):r},view_hash=plan.digest([row]))
        class MemorySheets(GoogleSheets):
            def __init__(self):
                self.layouts={261001011:67};self.extra_cells={};self.reuse_once=None;self.latest_snapshot=None
                self.hidden={};self.writes=[]
                oldcells=[dict(userEnteredValue=_typed(x),note=note if j==0 else '') for j,x in enumerate(expand(blank,['']*3))]
                foreign=[dict(userEnteredValue=_typed('foreign' if j==0 else '')) for j in range(67)]
                self.grid=[oldcells,foreign]
            def native(self,destination):
                self.layouts[261001011]=67;cells=deepcopy(self.grid)
                return dict(metadata=dict(spreadsheetId=BOOK,sheets=[dict(properties=dict(sheetId=261001011,title='QB-ablations',
                    gridProperties=dict(rowCount=1000,columnCount=67)))]),header=list(HEADERS67),width=67,grid_row_count=1000,
                    cells=cells,hidden_rows=deepcopy(self.hidden),rows=[[_cell_value(c) for c in line] for line in cells],
                    notes=[line[0].get('note','') for line in cells])
            def physical_request(self,method,url,**kwargs):
                assert method=='POST' and url==self.BASE+BOOK+':batchUpdate'
                for req in kwargs['json']['requests']:
                    self.writes.append(deepcopy(req))
                    if 'updateCells' in req:
                        b=req['updateCells'];rg=b['range'];start=rg['startColumnIndex']
                        for offset,line in enumerate(b['rows']):
                            target=self.grid[rg['startRowIndex']-5+offset]
                            for j,c in enumerate(line['values']):
                                oldformat=target[start+j].get('userEnteredFormat')
                                target[start+j]=deepcopy(c)
                                if oldformat:target[start+j]['userEnteredFormat']=oldformat
                    elif 'repeatCell' in req:
                        b=req['repeatCell'];rg=b['range']
                        for i in range(rg['startRowIndex']-5,rg['endRowIndex']-5):
                            for j in range(rg['startColumnIndex'],rg['endColumnIndex']):
                                self.grid[i][j]['userEnteredFormat']=deepcopy(b['cell']['userEnteredFormat'])
                    else:raise AssertionError('Unexpected mutation '+repr(req))
                return {}
        adapter=MemorySheets();foreign=deepcopy(adapter.grid[1])
        with tempfile.TemporaryDirectory() as tmp,patch.object(pub,'project',return_value=value),\
             patch.object(fix,'GoogleSheets',return_value=adapter),patch.object(fr,'GoogleSheets',return_value=adapter):
            dry=fix.publish(tmp,'s2',apply=False);self.assertEqual(dry['action'],'DRY_RUN');self.assertFalse(adapter.writes)
            result=fix.publish(tmp,'s2',apply=True)
            self.assertTrue(result['readback_verified']);self.assertTrue(result['dual_fr_same_checkpoint_verified'])
            self.assertEqual(_cell_value(adapter.grid[0][3]),.90);self.assertEqual(_cell_value(adapter.grid[0][7]),.96)
            self.assertEqual(adapter.grid[1],foreign)
            count=len(adapter.writes);fix.publish(tmp,'s2',apply=True);self.assertEqual(len(adapter.writes),count)
            self.assertTrue((Path(tmp)/'publication/qb_representative/all_evaluations.csv').exists())

    def test_real_serialized_same_arch_donors_C_input_frozen_and_exact_resume(self):
        from panda_sensor_eq40.tests.test_integration import synthetic_source
        from panda_unified_campaign import model as sm,training as st,assets as sa
        from panda_sensor_eq40.assets import calibration
        from panda_input3 import plan as active,training,model
        from panda_method.calibration import state_dict_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_source(root,'QB');dc=plan.donor_config('s2')
            teacher=sm.Stage1(dc,registered.inventory.manifest());dt=st.Trainer(dc,teacher)
            stream=st.stream_for(dc,registered,'cpu');pins=dict(source_release='synthetic-test')
            hashes=[]
            for cfg in plan.queue('s2')[:3]:
                step=cfg['donor_step'];dt.updates=step
                with torch.no_grad():teacher.aligner.fc2.bias.copy_(torch.tensor([.625+step/1e6,-.375]))
                folder=root/str(step);folder.mkdir();meta=st.save(folder/'donor.pt',dt,stream,pins,evaluated=True)
                receipt=dict(selected=dict(step=step,checkpoint_path='donor.pt',file_sha256=meta['file_sha256'],model_sha256=meta['model_sha256']),
                    receipt_sha256='d'*64,server='s2',donor_run_id=dc['run_id'])
                donor=sa._load_selected(dc,receipt,registered,folder,pins,device='cpu')
                stats=calibration(donor,registered,folder/'calibration',device='cpu',batch_size=1)
                c=deepcopy(cfg);c.update(batch=1,microbatch=1)
                with patch.object(active,'validate'):
                    net=model.Stage2(c,donor,stats);tr=training.Trainer(c,net)
                hashes.append(state_dict_sha256(net.student));batch=registered.batch([0])
                out=net.native_prediction(batch)
                self.assertEqual(tuple(out['x_in'].shape),(1,6,64,64))
                torch.testing.assert_close(out['x_in'][:,1:2],out['pan_aligned']-batch.pan,rtol=0,atol=0)
                s=training.stream_for(c,registered,'cpu');tr.step(s.next_update());tr.audit()
                if step==48000:
                    ck=root/'resume.pt';training.save(ck,tr,s,pins)
                    expected=tr.step(s.next_update());sha=training.model_hash(net)
                    training.load(ck,tr,s,pins)
                    self.assertEqual(tr.step(s.next_update()),expected);self.assertEqual(training.model_hash(net),sha)
                    tr.audit()
                wrong=deepcopy(c);wrong['donor_step']=123
                with patch.object(active,'validate'),self.assertRaises(ValueError):model.Stage2(wrong,donor,stats)
                with (folder/'donor.pt').open('ab') as f:f.write(b'tamper')
                with self.assertRaises(ValueError):sa._load_selected(dc,receipt,registered,folder,pins,device='cpu')
            self.assertEqual(len(set(hashes)),1)


if __name__=='__main__':unittest.main()
