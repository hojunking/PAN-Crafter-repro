"""Executed 4-band model/loss/checkpoint checks; synthetic metrics are not results."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same, capture_rng
from donor30.masking import masked_mean
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.stream import StreamSpec, TrainStream
from ta2.evaluation import file_sha

from panda_sensor5 import plan
from panda_sensor5.aligner import BandPANAligner, AllBandScharr
from panda_sensor5.base_reconstruction import BandReconstructionSpec, build_reconstructor
from panda_sensor5.calibration import bind_donor, binding_for_geometry
from panda_sensor5.contract import CAMPAIGN_ID
from panda_sensor5.engine import build_fresh_stage2, advance, save_checkpoint, load_checkpoint
from panda_sensor5.method import frozen_predictions
from panda_sensor5.reconstruction import reconstruct_with_base
from panda_sensor5.selection import validation_inventory, validate_inventory
from panda_sensor5.spec import MatureSpec
from panda_sensor5.stage1 import build_stage1, Stage1Stream, advance_stage1, save_stage1, load_stage1
from panda_sensor5.warp import warp_pan
from pa.warp import warp_pan as teacher_warp

PINS = {'code_pin':{'synthetic_test':'a'*64},'evaluator_pin':{'synthetic_test':'b'*64}}


def sources(root,sensor,count=3):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(8723)
    peak=1023 if sensor=='GF2' else 2047
    path=root/'train.h5'
    with h5py.File(path,'w') as h:
        for key,c,side in [('pan',1,64),('ms',4,16),('lms',4,64),('gt',4,64)]:
            h[key]=rng.uniform(.2*peak,.8*peak,(count,c,side,side)).astype('f4')
    ref={'path':path.name,'sha256':file_sha(path)}
    catalog=root/'sources.json'
    catalog.write_text(json.dumps({'schema':'ABLR2_SENSOR_SOURCES_v1','sensors':{sensor:{
        'band_order':['B','G','R','NIR'],'splits':{'train':ref,'val':ref,'rr':ref,'fr':ref}}}}))
    overrides={split:{**ref,'reason':'synthetic same-file original LMS contract'}
               for split in ('train','val')} if sensor=='QB' else None
    return RegisteredRRTrain(EdgeDataSources(catalog,root=root,sensor=sensor,lms_overrides=overrides))


def make_donor(sensor, *, binding_source=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        a=BandPANAligner()
    spec=BandReconstructionSpec(4,112,(1,2,3),'P0','BICUBIC_MS',1)
    t=build_reconstructor(spec,seed=12,role='T')
    with torch.no_grad(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        a.fc2.bias.copy_(torch.tensor([1.,-.5]))
        t.backbone.output[-1].weight.normal_(0.,.003)
    a.requires_grad_(False).eval();t.requires_grad_(False).eval()
    binding={'sensor':sensor,'source_split':'RR_train_only',
        'sample_ids_sha256':canonical_sha256([sensor+':train:0']),
        'source_manifest_sha256':'a'*64,
        'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
        'donor_checkpoint_id':'c'*64,
        'reconstruction_spec':{'bands':4,'width':112,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
        'scharr':{'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[2,1,0],'roi_margin':8},
        'warp':'bicubic_border_FP32_align_corners_false_current_pan_px',
        'band30_teacher':{'campaign_id':CAMPAIGN_ID,'upsampler':'interp23','aligner_outputs':2,
                          'alignment_target':'SCHARR_OF_MEAN4_GT','pan_channels':1},
        'band30_reconstruction_spec':spec.manifest()}
    if binding_source is not None:
        binding.update(binding_source.inventory.manifest())
    return LoadedDonor(a,t,binding,{})


def fixture(sensor,case='BASE',registered=None):
    cfg=plan.resolve_case('s1' if sensor=='QB' else 's3',sensor+'_'+case)
    cfg['training_spec'].update(microbatch_size=1,effective_batch_size=1)
    spec=MatureSpec(**cfg['stage2_spec'])
    donor=bind_donor(make_donor(sensor,binding_source=registered),spec)
    rr=batch(sensor=sensor) if registered is None else registered.batch([0])
    key=spec.geometry_keys[0]
    frozen=frozen_predictions(donor.aligner,donor.teacher,rr,geometry_key=key)
    e,g,q=reference_maps(ScharrAbsolute(sensor),frozen['teacher_y'],rr.gt,frozen['pan_loss'])
    ids=list(rr.lineage.sample_ids)
    # The one-image fixture uses an exact one-image source and calibration.
    collector=CalibrationBuilder(ids,binding_for_geometry(donor.binding,key),k=5,
        memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    collector.add(ids,e,g,q,sensor=sensor,loss_mask=frozen['loss_mask'])
    cfg['donor_binding']=deepcopy(donor.binding)
    trainer=build_fresh_stage2(cfg,donor,{key:collector.finalize()},evaluator_pin=PINS['evaluator_pin'])
    return trainer,rr


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_stage1_mean4_includes_nir_and_cannot_use_rgb_target(self):
        for sensor in ('QB','GF2'):
            spatial=AllBandScharr(2,sensor=sensor)
            gt=torch.zeros(1,4,64,64);gt[:,3,:,32:]=1.
            pan=gt.mean(1,keepdim=True)
            self.assertEqual(float(spatial.absolute_loss(pan,gt)),0.)
            self.assertGreater(float(ScharrAbsolute(sensor).absolute_loss(pan,gt)),0.)
            self.assertEqual(spatial.manifest()['target_bands'],[0,1,2,3])

    def test_both_stage1_update_and_complete_resume(self):
        for sensor,server in [('QB','s1'),('GF2','s3')]:
            with self.subTest(sensor=sensor),tempfile.TemporaryDirectory() as tmp:
                rr=sources(tmp,sensor)
                cfg=plan.resolve_case(server,plan.DONORS[0])
                cfg['baseline_resolved'].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
                a=build_stage1(cfg,rr.sources);stream=Stage1Stream(rr.sources,cfg,device='cpu')
                self.assertEqual(a.stage.teacher.backbone.input.in_channels,5)
                self.assertEqual(a.stage.aligner.ms_bands,4)
                olda=state_dict_sha256(a.stage.aligner);oldt=state_dict_sha256(a.stage.teacher)
                advance_stage1(a,stream)
                self.assertNotEqual(olda,state_dict_sha256(a.stage.aligner))
                self.assertNotEqual(oldt,state_dict_sha256(a.stage.teacher))
                path=Path(tmp)/'stage1.pt';saved=save_stage1(path,a,stream,**PINS)
                expected=advance_stage1(a,stream)
                before=capture_rng()
                b,resumed,_=load_stage1(path,saved['file_sha256'],cfg,rr.sources,
                    restore_global_rng=False,**PINS)
                self.assertTrue(_same(before,capture_rng()))
                got=advance_stage1(b,resumed)
                self.assertTrue(_same(expected,got))
                self.assertTrue(_same(a.state_dict(),b.state_dict()))

    def test_student_full_shift_diff_and_hard4_loss_frozen_roles(self):
        for sensor in ('QB','GF2'):
            for case in ('BASE','HARD4'):
                with self.subTest(sensor=sensor,case=case):
                    trainer,rr=fixture(sensor,case);s=trainer.stage
                    olda=state_dict_sha256(s.aligner);oldt=state_dict_sha256(s.teacher)
                    oldstudent=state_dict_sha256(s.student)
                    self.assertEqual(s.student.backbone.input.in_channels,6)
                    self.assertEqual(s.student.spec.bands,4)
                    frozen=s.frozen_predictions(rr)
                    corrected=warp_pan(rr.pan,frozen['u'])
                    torch.testing.assert_close(corrected,frozen['pan_input'],rtol=0,atol=0)
                    teacher=reconstruct_with_base(s.teacher,teacher_warp(rr.pan,frozen['u']),
                        frozen['ms_base_frame'],rr.ms)['y']
                    torch.testing.assert_close(teacher,frozen['teacher_y'],rtol=0,atol=0)
                    out=s(rr,return_maps=True);m=out['maps'];e=(out['student']['y']-rr.gt).abs().mean(1,keepdim=True)
                    weight=1+s.spec.alpha*m['d_e']+s.spec.gamma*m['d_e']*m['r_q_bar']
                    torch.testing.assert_close(out['loss'],masked_mean(weight*e,out['loss_mask']),rtol=0,atol=0)
                    torch.testing.assert_close(out['student']['x_in'][:,1:2],corrected-rr.pan,rtol=0,atol=0)
                    step=trainer.step_accumulated([rr])
                    self.assertEqual(step['soft_weighted'],0.)
                    self.assertNotEqual(oldstudent,state_dict_sha256(s.student))
                    self.assertEqual(olda,state_dict_sha256(s.aligner));self.assertEqual(oldt,state_dict_sha256(s.teacher))
                    self.assertTrue(all(p.grad is None for net in (s.aligner,s.teacher) for p in net.parameters()))

    def test_student_complete_resume_preserves_next_adam_update(self):
        for sensor in ('QB','GF2'):
            with self.subTest(sensor=sensor),tempfile.TemporaryDirectory() as tmp:
                registered=sources(tmp,sensor,count=1)
                a,_=fixture(sensor,registered=registered)
                stream=TrainStream(registered,StreamSpec(23,34),microbatch_size=1,effective_batch_size=1)
                advance(a,stream)
                path=Path(tmp)/'student.pt';saved=save_checkpoint(path,a,stream,**PINS)
                expected=advance(a,stream)
                b,_=fixture(sensor,registered=registered)
                other=TrainStream(registered,StreamSpec(23,34),microbatch_size=1,effective_batch_size=1)
                b,other=load_checkpoint(path,saved['file_sha256'],b,other,**PINS)
                actual=advance(b,other)
                self.assertTrue(_same(expected,actual))
                self.assertTrue(_same(a.state_dict(stream_state=stream.state_dict()),
                                      b.state_dict(stream_state=other.state_dict())))

    def test_strict_saved_donor_import_and_configured_evaluation_identity(self):
        from panda_sensor5.donor import load_donor
        from panda_sensor5.evaluation import configured_pipeline, model_identity
        from panda_warp30.tests.test_stage1 import evaluation
        for sensor,server in [('QB','s1'),('GF2','s3')]:
            with self.subTest(sensor=sensor),tempfile.TemporaryDirectory() as tmp:
                registered=sources(tmp,sensor,count=1)
                cfg=plan.resolve_case(server,plan.DONORS[0])
                cfg['baseline_resolved'].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
                trainer=build_stage1(cfg,registered.sources)
                stream=Stage1Stream(registered.sources,cfg,device='cpu')
                advance_stage1(trainer,stream)
                configured_pipeline(trainer,registered.sources)
                identity=model_identity(trainer,registered.sources,evaluator_pin=PINS['evaluator_pin'])
                self.assertEqual(identity['dataset'],sensor)
                ev=evaluation(trainer)
                for split in ('rr','fr'):
                    for i,row in enumerate(ev[split]['per_scene']):row['scene_id']=f'{sensor}:{split}:{i}'
                path=Path(tmp)/'donor.pt'
                artifact=save_stage1(path,trainer,stream,evaluation=ev,**PINS)
                donor=load_donor(path,artifact['file_sha256'],registered,server=server)
                self.assertEqual(state_dict_sha256(donor.aligner),state_dict_sha256(trainer.stage.aligner))
                self.assertEqual(state_dict_sha256(donor.teacher),state_dict_sha256(trainer.stage.teacher))
                self.assertEqual(donor.binding['reconstruction_spec']['bands'],4)
                self.assertEqual(donor.binding['scharr']['rgb_indices'],[2,1,0])
                with self.assertRaises(ValueError):
                    load_donor(path,artifact['file_sha256'],registered,server='s3' if server=='s1' else 's1')
                student,_=fixture(sensor,registered=registered)
                configured_pipeline(student,registered.sources)
                sid=model_identity(student,registered.sources,evaluator_pin=PINS['evaluator_pin'])
                self.assertEqual(sid['dataset'],sensor)

    def test_validation_population_and_peak_are_sensor_specific(self):
        for sensor in ('QB','GF2'):
            with tempfile.TemporaryDirectory() as tmp:
                rr=sources(tmp,sensor,count=3)
                inventory=validation_inventory(rr.sources)
                self.assertEqual(inventory['sample_count'],3)
                self.assertEqual(inventory['sensor'],sensor)
                validate_inventory(inventory,rr.sources)
                wrong=deepcopy(inventory);wrong['peak']=2047 if sensor=='GF2' else 1023
                wrong['inventory_sha256']=canonical_sha256({k:v for k,v in wrong.items() if k!='inventory_sha256'})
                with self.assertRaises(ValueError):validate_inventory(wrong)

if __name__=='__main__':unittest.main()
