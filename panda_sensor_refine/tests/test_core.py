"""CPU implementation checks; synthetic inputs are never experiment evidence."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256, canonical_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor
from panda_sensor_refine import plan
from panda_sensor_refine.contract import (CAMPAIGN_ID, LEGACY_CAMPAIGN_ID,
    validate_loss_pair_config, validate_donor_loss_pair)
from panda_sensor_refine.stage1 import (sample_epsilon, build_stage1, teacher_binding,
    Stage1Stream, advance_stage1, save_stage1, load_stage1)
from panda_sensor_refine.warp import warp_support_mask

PINS = {'code_pin':{'synthetic_cpu_test':'a'*64},'evaluator_pin':{'synthetic_cpu_test':'b'*64}}


def make_donor(cfg, registered, *, legacy=False):
    if legacy:
        from panda_s1loss9.stage1 import build_stage1 as builder, teacher_binding as binding_fn
    else:
        builder, binding_fn = build_stage1, teacher_binding
    trainer = builder(cfg, registered.sources)
    a,t = trainer.stage.aligner,trainer.stage.teacher
    for m in (a,t): m.requires_grad_(False).eval()
    binding = {**registered.inventory.manifest(), 'source_split':'RR_train_only',
        'aligner_state_sha256':state_dict_sha256(a),'teacher_state_sha256':state_dict_sha256(t),
        'donor_checkpoint_id':'c'*64,'donor_run_id':cfg['run_id'],'donor_step':1000,
        'reconstruction_spec':{'bands':4,'width':112,'depth':[1,2,3],
            'input_layout':'P0','base_reference':'BICUBIC_MS'},
        'scharr':{'kernel':'absolute_xy_/32','padding':'reflect1','rgb_indices':[2,1,0],'roi_margin':8},
        'warp':'lanczos4_reflect_FP32_current_pan_px',
        'band30_teacher':binding_fn(cfg),'mature30_teacher':binding_fn(cfg),
        'band30_reconstruction_spec':t.spec.manifest(),'source_campaign_id':cfg['campaign_id']}
    return LoadedDonor(a,t,binding,{'synthetic_test_only':True})


def bank(donor,spec,registered):
    from panda_sensor_refine.calibration import binding_for_geometry
    from panda_sensor_refine.calibration_core import CalibrationBuilder
    key=spec.geometry_keys[0]; ids=list(registered.sample_ids)
    values=torch.linspace(.1,.9,64*64).reshape(1,1,64,64)
    c=CalibrationBuilder(ids,binding_for_geometry(donor.binding,key),k=5,
        memory_budget_bytes=8*1024**2,include_condq=True,mask_policy=spec.mask_policy)
    c.add(ids,values,values.flip(-1),values.flip(-2),sensor=spec.sensor,
          loss_mask=torch.ones_like(values,dtype=torch.bool))
    return {key:c.finalize()}


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_radius_sampler_distribution_and_same_random_draws(self):
        from directshift48.method import sample_epsilon as original_sample
        r3=sample_epsilon(100000,torch.Generator().manual_seed(72),3.)
        self.assertTrue(torch.equal(r3,original_sample(100000,torch.Generator().manual_seed(72))))
        previous_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            actual=sample_epsilon(100000,torch.Generator().manual_seed(72),3.)
            self.assertEqual(actual.dtype,torch.float32)
            self.assertTrue(torch.equal(actual,r3))
        finally:
            torch.set_default_dtype(previous_dtype)
        r4=sample_epsilon(100000,torch.Generator().manual_seed(72),4.)
        torch.testing.assert_close(r4,r3*(4/3),rtol=2e-7,atol=5e-7)
        norm=torch.linalg.vector_norm(r4,dim=1)
        self.assertLessEqual(float(norm.max()),4.)
        self.assertGreater(float(norm.max()),3.99)
        self.assertAlmostEqual(float(norm.square().mean()),8.,delta=.06)
        for r in (0,2,5,float('nan')):
            with self.assertRaises(ValueError): sample_epsilon(1,torch.Generator(),r)

    def test_r4_clean_input_support_and_separate_native_guard(self):
        shifts=torch.tensor([[4.,0.],[-4.,0.],[0.,4.],[0.,-4.],[2.828427,2.828427]])
        support=warp_support_mask(64,64,shifts)
        self.assertTrue(bool(support[...,8:-8,8:-8].all()))
        native=warp_support_mask(64,64,torch.tensor([[0.,3.999],[0.,4.],[0.,4.0336914],[0.,-4.]]))
        self.assertEqual(native[...,7:-7,7:-7].flatten(1).all(1).tolist(),[True,False,False,True])

    def test_config_and_donor_radius_mismatch_rejected(self):
        cfg=plan.resolve_case('s1','QB_A200_E030_R4')
        bad=deepcopy(cfg);bad['epsilon_radius_rr']=3.
        with self.assertRaises(ValueError):validate_loss_pair_config(bad)
        t={k:cfg[k] for k in ('campaign_id','upsampler','aligner_outputs','pan_channels','alignment_target','epsilon_radius_rr')}
        t.update(lambda_abs=2.,lambda_epsilon=.03)
        binding={'band30_teacher':t,'mature30_teacher':deepcopy(t),'source_campaign_id':CAMPAIGN_ID}
        validate_donor_loss_pair(cfg,binding)
        bad=deepcopy(binding);bad['band30_teacher']['epsilon_radius_rr']=3.
        bad['mature30_teacher']=deepcopy(bad['band30_teacher'])
        with self.assertRaises(ValueError):validate_donor_loss_pair(cfg,bad)

    def test_r4_stage1_real_updates_exact_resume_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            reg=sources(root,'QB',count=1)
            cfg=plan.resolve_case('s1','S1_A200_E030_R4')
            cfg['baseline_resolved'].update(microbatch_RR=1,effective_batch_RR=1,accumulation_RR=1)
            t=build_stage1(cfg,reg.sources);stream=Stage1Stream(reg.sources,cfg,device='cpu')
            self.assertEqual(t.stage.pipeline_manifest()['eps']['radius'],4.)
            self.assertEqual(t.stage.alignment_manifest()['epsilon']['radius'],4.)
            self.assertEqual(t.stage.pipeline_manifest()['boundary']['epsilon_max_radius_pan_px'],4.)
            self.assertEqual(t.stage.teacher_binding(),teacher_binding(cfg))
            from panda_sensor_refine.evaluation import configured_pipeline
            configured_pipeline(t,reg.sources)
            advance_stage1(t,stream)
            path=Path(root)/'resume.pt';saved=save_stage1(path,t,stream,**PINS)
            expected=advance_stage1(t,stream)
            restored,restream,_=load_stage1(path,saved['file_sha256'],cfg,reg.sources,
                                          restore_global_rng=False,**PINS)
            actual=advance_stage1(restored,restream)
            self.assertTrue(_same(expected,actual))
            self.assertTrue(_same(t.state_dict(),restored.state_dict()))
            invalid=deepcopy(cfg);invalid['epsilon_radius_rr']=3.
            with self.assertRaises(ValueError):load_stage1(path,saved['file_sha256'],invalid,reg.sources,**PINS)
            state=deepcopy(t.stage.state_dict());state['_extra_state']['pipeline']['eps']['radius']=3.
            with self.assertRaises(ValueError):t.stage.load_state_dict(state)

    def test_original_teacher_types_and_fresh_r4_donors_build_new_student(self):
        from panda_s1loss9 import plan as original
        from panda_sensor_refine.calibration import bind_donor
        from panda_sensor_refine.engine import build_fresh_stage2
        from panda_sensor_refine.spec import MatureSpec
        from panda_sensor_refine.evaluation import configured_pipeline
        for legacy in (True,False):
            with self.subTest(legacy=legacy),tempfile.TemporaryDirectory() as root:
                reg=sources(root,'GF2',count=1)
                cfg=plan.resolve_case('s3','GF2_BASE','R2') if legacy else plan.resolve_case('s3','GF2_A200_E030_R4')
                source=original.resolve_case('s3','S1_A200_E030') if legacy else plan.resolve_case('s3','S1_A200_E030_R4')
                d=make_donor(source,reg,legacy=legacy);before=deepcopy(d.binding)
                spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
                self.assertEqual(d.binding['band30_teacher'],before['band30_teacher'])
                self.assertEqual(d.binding['source_campaign_id'],LEGACY_CAMPAIGN_ID if legacy else CAMPAIGN_ID)
                cfg['donor_binding']=deepcopy(d.binding)
                t=build_fresh_stage2(cfg,d,bank(d,spec,reg),evaluator_pin=PINS['evaluator_pin'])
                configured_pipeline(t,reg.sources)
                output=t.stage(reg.batch([0]))
                self.assertTrue(bool(torch.isfinite(output['loss'])))
                self.assertEqual(t.stage.student.backbone.input.in_channels,6)
                self.assertEqual(t.stage.spec.beta,0.)
                del t,d;gc.collect()

    def test_controller_isolates_failed_donor_and_finishes_other_cases(self):
        from panda_sensor_refine.runtime import Campaign
        class Fake(Campaign):
            def admission(self,catalog): self.migration={'external_donor':{'status':'COMPLETED'}}
            def register_external_donor(self): pass
            def register_donor(self,case): pass
            def validate_preflight(self,cfg): pass
            def run_case(self,cfg):
                if cfg['case']=='S1_A200_E030_R4':
                    raise ValueError('SUPPORT_VIOLATION: synthetic controller failure')
                self.state['runs'].append({'run_id':cfg['run_id'],'status':'COMPLETED'})
            def support_failure_receipt(self,cfg,exc):
                return {'run_id':cfg['run_id'],'case':cfg['case'],'step':7,'synthetic_test':True}
        with tempfile.TemporaryDirectory() as root:
            c=Fake('s1',output=root,device='cpu');state=c.run(catalog='unused')
            self.assertEqual(state['status'],'FINISHED_WITH_EXPLICIT_FAILURES')
            self.assertEqual(len(state['runs']),8)
            self.assertEqual(len(state['isolated_support_failures']),1)
            self.assertEqual(len(state['held_runs']),1)
            held=next(iter(state['held_runs'].values()))
            self.assertEqual(held['case'],'QB_A200_E030_R4')
            previous=len(state['runs']);c.run(catalog='unused')
            self.assertEqual(len(state['runs']),previous)

    def test_missing_external_support_donor_holds_only_four_students(self):
        from panda_sensor_refine.runtime import Campaign
        class Fake(Campaign):
            def admission(self,catalog):
                self.migration={'external_donor':{'status':'FAILED_SUPPORT','failure':{'synthetic':True}}}
            def register_external_donor(self): raise AssertionError('Must not import failed source')
            def register_donor(self,case): pass
            def validate_preflight(self,cfg): pass
            def run_case(self,cfg): self.state['runs'].append({'run_id':cfg['run_id'],'status':'COMPLETED'})
        with tempfile.TemporaryDirectory() as root:
            c=Fake('s3',output=root,device='cpu');state=c.run(catalog='unused')
            self.assertEqual(state['status'],'FINISHED_WITH_EXPLICIT_FAILURES')
            self.assertEqual(len(state['runs']),6)
            self.assertEqual(len(state['held_runs']),4)
            self.assertTrue(all(row['donor_case']=='S1_A200_E030' for row in state['held_runs'].values()))

    def test_disposable_support_failure_cannot_hide_started_production(self):
        from panda_sensor_refine.runtime import Campaign
        from ta2.evaluation import file_sha
        cfg=plan.resolve_case('s1','S1_A200_E030_R4')
        with tempfile.TemporaryDirectory() as root:
            c=Campaign('s1',output=root,device='cpu')
            receipt=c.preflight_support_failure_receipt(cfg,ValueError('SUPPORT_VIOLATION: disposable'))
            proof=json.loads(Path(receipt['receipt_path']).read_text())
            self.assertEqual(receipt['receipt_sha256'],file_sha(receipt['receipt_path']))
            self.assertEqual(receipt['evidence_kind'],'FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES')
            self.assertEqual(proof['production_updates'],0)
            self.assertFalse(proof['production_weights_started'])
            self.assertIsNone(c.preflight_support_failure_receipt(cfg,ValueError('unknown failure')))
            (c.root/'runs'/cfg['run_id']).mkdir(parents=True)
            self.assertIsNone(c.preflight_support_failure_receipt(cfg,ValueError('SUPPORT_VIOLATION: started')))

    def test_known_support_failures_require_verified_complete_checkpoint(self):
        from panda_sensor_refine.runtime import Campaign
        from ta2.evaluation import file_sha
        cfg=plan.resolve_case('s1','S1_A200_E030_R4')
        with tempfile.TemporaryDirectory() as root:
            c=Campaign('s1',output=root,device='cpu')
            directory=Path(root)/'run';(directory/'models').mkdir(parents=True)
            checkpoint=directory/'models'/'stop.pt';checkpoint.write_bytes(b'synthetic proof')
            artifact={'checkpoint_path':str(checkpoint),'file_sha256':file_sha(checkpoint),
                      'completed_update':7,'campaign_resume_complete':True}
            (directory/'models'/'latest_resume.json').write_text(json.dumps(artifact))
            row={'run_id':cfg['run_id'],'case':cfg['case'],'status':'FAILED','poisoned':False,
                 'step':7,'directory':str(directory)}
            c.state['runs']=[row]
            self.assertIsNotNone(c.support_failure_receipt(cfg,ValueError('SUPPORT_VIOLATION: test')))
            self.assertIsNone(c.support_failure_receipt(cfg,ValueError('unrelated error')))
            row['poisoned']=True
            self.assertIsNone(c.support_failure_receipt(cfg,ValueError('SUPPORT_VIOLATION: test')))
            row['poisoned']=False;checkpoint.write_bytes(b'tampered')
            with self.assertRaises(ValueError):c.support_failure_receipt(cfg,ValueError('SUPPORT_VIOLATION: test'))

if __name__=='__main__':unittest.main()
