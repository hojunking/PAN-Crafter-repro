"""Synthetic CPU correctness checks; these are not scientific experiment results."""
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor
from panda_sensor_focus import plan,reporting
from panda_sensor_focus.contract import SOURCE_DONORS,REGISTERED_RUNS

PINS={'code_pin':{'synthetic_cpu_test':'a'*64},'evaluator_pin':{'synthetic_cpu_test':'b'*64}}

def donor(server,registered):
    d=SOURCE_DONORS[server];pkg=importlib.import_module(d['package']+'.plan')
    source=pkg.resolve_case(server,d['case']);stage=importlib.import_module(d['package']+'.stage1')
    trainer=stage.build_stage1(source,registered.sources)
    a,t=trainer.stage.aligner,trainer.stage.teacher
    for model in (a,t):model.requires_grad_(False).eval()
    binding={**registered.inventory.manifest(),'source_split':'RR_train_only',
        'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
        'donor_checkpoint_id':d['checkpoint_sha256'],'donor_checkpoint_sha256':d['checkpoint_sha256'],
        'donor_run_id':source['run_id'],'donor_step':d['selected_step'],
        'reconstruction_spec':{'bands':4,'width':112,'depth':[1,2,3],'input_layout':'P0','base_reference':'BICUBIC_MS'},
        'scharr':{'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[2,1,0],'roi_margin':8},
        'warp':'lanczos4_reflect_FP32_current_pan_px','band30_teacher':stage.teacher_binding(source),
        'mature30_teacher':stage.teacher_binding(source),'band30_reconstruction_spec':t.spec.manifest(),
        'source_campaign_id':d['campaign_id']}
    return LoadedDonor(a,t,binding,{'synthetic_test_only':True})

def bank(d,spec,registered):
    from panda_sensor_focus.calibration import binding_for_geometry
    from panda_sensor_focus.calibration_core import CalibrationBuilder
    ids=list(registered.sample_ids);key=spec.geometry_keys[0]
    v=torch.linspace(.1,.9,64*64).reshape(1,1,64,64)
    c=CalibrationBuilder(ids,binding_for_geometry(d.binding,key),k=5,memory_budget_bytes=8*1024**2,
        include_condq=True,mask_policy=spec.mask_policy)
    c.add(ids,v,v.flip(-1),v.flip(-2),sensor=spec.sensor,loss_mask=torch.ones_like(v,dtype=torch.bool))
    return {key:c.finalize()}

class FocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_finite_registration_original_streams_and_forbidden_variants(self):
        from panda_sensor_refine import plan as prior
        for server,count in [('s1',6),('s3',5)]:
            self.assertEqual(len(plan.groups(server)),count)
            self.assertEqual(plan.default_config(server)['stage1_runs'],0)
            for case,rep in REGISTERED_RUNS[server]:
                c=plan.resolve_case(server,case,rep);reporting._config(c)
                source=importlib.import_module(SOURCE_DONORS[server]['package']+'.plan')
                control=(source.resolve_case(server,'QB_A200_E030','R1') if server=='s1' else source.resolve_case(server,'GF2_LONG','R1'))
                if rep=='R1':self.assertEqual(c['stream_seeds'],control['stream_seeds'])
                self.assertEqual(c['baseline_resolved']['teacher_architecture']['width'],112)
                self.assertEqual(c['student_input']['channels'],6)
                self.assertEqual(c['shift_scale'],1)
                self.assertEqual(c['stage2_spec']['beta'],0)
                for key,val in [('shift_scale',0),('initialization','FORK_PRESERVE_COMPLETE_STATE')]:
                    bad=deepcopy(c);bad[key]=val
                    with self.assertRaises(ValueError):reporting._config(bad)
        with self.assertRaises(ValueError):plan.resolve_case('s3','GF2_C60','R1')
        with self.assertRaises(ValueError):plan.resolve_case('s1','S1_A200_E030','R1')

    def test_original_teacher_identity_student_width_training_and_exact_resume(self):
        from panda_sensor_focus.calibration import bind_donor
        from panda_sensor_focus.spec import MatureSpec
        from panda_sensor_focus.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
        from panda_sensor_focus.evaluation import configured_pipeline
        from panda_stage2_condq.stream import TrainStream,StreamSpec
        for server,case,rep in [('s1','QB_W128','R1'),('s3','GF2_C60W128','R1'),('s1','QB_G000','R1')]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as root:
                reg=sources(root,plan.DATASETS[server],count=1)
                cfg=plan.resolve_case(server,case,rep)
                cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
                d=donor(server,reg);original=deepcopy(d.binding)
                spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec);stats=bank(d,spec,reg)
                self.assertEqual(d.binding['band30_teacher'],original['band30_teacher'])
                self.assertEqual(d.binding['source_campaign_id'],original['source_campaign_id'])
                cfg['donor_binding']=deepcopy(d.binding)
                t=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
                configured_pipeline(t,reg.sources)
                self.assertEqual(t.stage.student.backbone.input.in_channels,6)
                self.assertEqual(t.stage.student.spec.width,cfg['student_architecture']['width'])
                self.assertEqual(t.stage.teacher.spec.width,112)
                self.assertEqual(sum(p.numel() for p in t.stage.student.parameters()),cfg['student_sampling']['student_parameters'])
                stream=TrainStream(reg,StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation']),microbatch_size=1,effective_batch_size=1)
                result=advance(t,stream)
                self.assertEqual(result['loss'],result['hard_weighted']);self.assertEqual(result['soft_weighted'],0.)
                path=Path(root)/'resume.pt';saved=save_checkpoint(path,t,stream,**PINS)
                expected=advance(t,stream);expected_state=deepcopy(t.state_dict())
                fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
                restream=TrainStream(reg,StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation']),microbatch_size=1,effective_batch_size=1)
                restored,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
                actual=advance(restored,restream)
                self.assertTrue(_same(actual,expected));self.assertTrue(_same(restored.state_dict(),expected_state))

    def test_donor_step_sha_and_cross_sensor_rejected(self):
        from panda_sensor_focus.contract import validate_donor_loss_pair
        with tempfile.TemporaryDirectory() as root:
            reg=sources(root,'GF2',count=1);d=donor('s3',reg);cfg=plan.resolve_case('s3','GF2_C60','R2')
            validate_donor_loss_pair(cfg,d.binding)
            for key,value in [('donor_step',41000),('donor_checkpoint_sha256','x'*64),('sensor','QB')]:
                bad=deepcopy(d.binding);bad[key]=value
                with self.assertRaises(ValueError):validate_donor_loss_pair(cfg,bad)
            self.assertNotIn('lambda_abs',d.binding['band30_teacher'])

    def test_cost_graphs_and_empty_publisher(self):
        from panda_sensor_focus.cost.profiling import registered_profiles
        from panda_sensor_focus.cost import signature
        from panda_sensor_focus.best_publication import project
        for server in ('s1','s3'):
            profiles=registered_profiles(server)
            self.assertEqual({c['student_architecture']['width'] for c,_ in profiles},{112,128})
            for c,step in profiles:signature(c,step)
            with tempfile.TemporaryDirectory() as root:
                value=project(root,server)
                self.assertFalse(value['conflicts']);self.assertFalse(value['rows'])
                self.assertEqual(value['publication_thresholds']['ergas_max'],3.5 if server=='s1' else .522)

if __name__=='__main__':unittest.main()
