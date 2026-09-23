"""ABLR2X additive C17/GF2 contracts; CPU synthetic evidence only."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ablr2.common import ROOT,read_json,atomic_json,object_sha,sha256,run_dir
from ablr2.plan import component_config,cases_for,build_config,sensor_spec
from ablr2.model import build_model,state_hash
from ablr2.losses import student_losses,routed_student_backward
from ablr2.training import _load_teacher
from ablr2 import evaluation as ev


class C17Tests(unittest.TestCase):
    def test_c17_changes_only_reference_initial_a_not_u_or_objective(self):
        torch.manual_seed(13)
        positive,_=build_model(bands=4,seed=71,role='T',width=8,depth=(1,1,1))
        negative,_=build_model(bands=4,seed=71,role='T',width=8,depth=(1,1,1))
        with torch.no_grad():next(negative.aligner.parameters()).add_(.05)
        originals=[state_hash(t.state_dict()) for t in (positive,negative)]
        students=[]
        for case_id,teacher in [('C03',positive),('C17',negative)]:
            c=component_config(case_id)
            student,init=build_model(bands=4,seed=991001,role='S',component=c,
                teacher_aligner_state=teacher.aligner.state_dict(),width=8,depth=(1,1,1))
            self.assertEqual(state_hash(student.aligner.state_dict()),state_hash(teacher.aligner.state_dict()))
            self.assertTrue(all(p.requires_grad for p in student.aligner.parameters()))
            self.assertFalse(set(map(id,student.parameters()))&set(map(id,teacher.parameters())))
            students.append(student)
        self.assertEqual(state_hash(students[0].backbone.state_dict()),state_hash(students[1].backbone.state_dict()))
        c=component_config('C17');student=students[1]
        pan,ms,lp=torch.randn(2,1,32,32),torch.randn(2,4,8,8),torch.randn(2,1,8,8)
        gt=torch.randn(2,4,32,32);out=student(pan,ms,lp)
        losses=student_losses(out,None,gt,c,bands=4)
        torch.testing.assert_close(losses['L_U'],(out['y']-gt).abs().mean(),rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.equal(losses['L_A'],.5*losses['L_U']))
        # The fresh residual head is zero; update U once before testing the
        # reconstruction gradient through PAN/warp into the trainable clone.
        optimizer=torch.optim.AdamW(student.parameters(),lr=.01,weight_decay=.01)
        routed_student_backward(student,losses);optimizer.step();optimizer.zero_grad(set_to_none=True)
        losses=student_losses(student(pan,ms,lp),None,gt,c,bands=4)
        before=state_hash(student.aligner.state_dict());routed_student_backward(student,losses)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in student.aligner.parameters()),0)
        optimizer.step()
        self.assertNotEqual(before,state_hash(student.aligner.state_dict()))
        self.assertEqual(originals,[state_hash(t.state_dict()) for t in (positive,negative)])

    def test_c17_loader_is_endpoint_only_and_matched_tzero(self):
        case=next(c for c in cases_for('s3') if c.case_id=='C17')
        cfg=build_config(case);data={'sensor':'GF2'};f=cfg['ablr2']
        ref=dict(reference_id=case.reference_id,teacher_run_id=case.teacher_run_id,
            teacher_seed=case.teacher_seed,teacher_kind='TZERO',sensor='GF2',owner_server='s3',
            data_sha256=object_sha(data),teacher_checkpoint_sha256='a'*64)
        f.update(teacher_checkpoint='/fixture/teacher.safetensors',teacher_sha256='a'*64)
        sentinel=object()
        with patch('ablr2.references.load_reference',side_effect=AssertionError('No q/calibration access')), \
                patch('ablr2.references.load_endpoint_only',return_value=(sentinel,ref)) as endpoint, \
                patch('ablr2.common.sha256',return_value='a'*64):
            self.assertEqual(_load_teacher(case,cfg,data,ROOT,'cpu',None),(sentinel,ref,None))
            endpoint.assert_called_once()
            ref['teacher_kind']='TPLUS'
            with self.assertRaisesRegex(ValueError,'exact sensor/seed'):_load_teacher(case,cfg,data,ROOT,'cpu',None)


class GF2Tests(unittest.TestCase):
    def test_source_binding_matches_verified_existing_corpus_not_qb(self):
        from ablr2.data import source_catalog
        spec,entry=source_catalog(ROOT,'s3')
        self.assertEqual(entry,read_json(ROOT/'qg40/sensor_sources.json')['sensors']['GF2'])
        self.assertEqual((spec.max_dn,spec.inverse_scale,spec.nominal_train_n,spec.nominal_val_n),(1023,511.5,19809,2201))
        self.assertEqual(ev.canonical_band_indices(spec),(0,1,2,3))
        self.assertNotIn('raw_train_path',entry['source_provenance'])

    def test_rr_real_metric_primitives_match_existing_gf2_evaluator(self):
        from g20.evaluation import rr_metrics as old_rr
        base=np.arange(256*256,dtype=np.float64).reshape(1,1,256,256)/100+10
        truth=np.broadcast_to(base,(20,4,256,256));pred=truth+np.sin(base/50)
        spec=replace(sensor_spec('GF2'),band_order=('B','G','R','NIR'))
        actual=ev.rr_metrics(pred,truth,spec,include_q=False)
        prior=old_rr(pred,truth,spec,include_q=False)
        for key in ev.RR_COMMON:self.assertEqual(actual[key],prior[key],key)
        self.assertEqual(actual['per_scene'],prior['per_scene'])
        self.assertEqual(actual['max_dn'],1023)

    def test_fr_uses_gf2_mtf_and_jqm_explicit_default_not_qb(self):
        spec=replace(sensor_spec('GF2'),band_order=('B','G','R','NIR'))
        engine=ev.FRMetrics.__new__(ev.FRMetrics)
        engine.spec,engine.order,engine.inverse_order=spec,tuple(range(4)),np.arange(4)
        engine.wald,engine.wald_sha256=object(),None
        engine.lms=np.broadcast_to(np.array(1.),(20,512,512,4))
        engine.pan=np.broadcast_to(np.array(2.),(20,512,512));engine.reference=[[.2]*4 for _ in range(20)]
        sr=np.broadcast_to(np.array(3.),(20,4,512,512))
        def mtf(data,sensor,ratio,wald):
            self.assertEqual((sensor,ratio),('gf2',4));return data
        with patch('tools.metrics.eval_fr.mtf_filter',side_effect=mtf), \
                patch('tools.metrics.eval_fr._blockproc_uqi',return_value=.21),patch.object(ev,'q2n',return_value=(.98,None)):
            result=engine(sr)
        self.assertEqual(result['mtf_gnyq_canonical'],[.3]*4);self.assertFalse(result['masking'])
        def jqm(pred,ms,pan,sensor,**options):
            self.assertEqual(sensor,'GF2');self.assertEqual(options['R'],1023)
            return dict(JQM=.7,QLR=.8,QHR=.9,w=np.ones(4)/4,w_source='fixture')
        with patch('tools.metrics.jqm.jqm',side_effect=jqm):
            value=ev.fr_jqm(sr,np.ones((20,4,128,128)),np.ones((20,1,512,512)),spec)
        self.assertEqual(value['pan_gnyq'],.15);self.assertIn('surrogate',value['gf2_pan_note'])


class MatchedTeacherTests(unittest.TestCase):
    def test_pair_requires_original_init_and_native_stream_not_final_weights(self):
        from ablr2.references import validate_teacher_pair
        cases=[c for c in cases_for('s3') if c.sweep=='P01' and c.role=='T']
        cases=sorted(cases,key=lambda c:c.case_id)
        with tempfile.TemporaryDirectory() as root:
            endpoints={}
            for i,case in enumerate(cases):
                cfg={'seed':case.seed};data={'fixture':'same-data'}
                identity={'update':50000,'source_identity':{'fixture':'same-release'},
                    'data_sha256':object_sha(data),'model_sha256':str(i)*64}
                context=dict(config_sha256=object_sha(cfg),data_sha256=object_sha(data),
                    source_identity=identity['source_identity'],reference_sha256=None)
                wd=run_dir(case.run_id,root)
                atomic_json(wd/'init_manifest.json',dict(context,role='T',seed=case.seed,hashes={'U':'a'*64,'A':'b'*64}))
                atomic_json(wd/'meta/training_start_manifest.json',dict(context,run_id=case.run_id,
                    sampler_hash='c'*64,rng_roles={'data_order':1,'augmentation':2,'workers':3,'corruption':i}))
                endpoints[case.run_id]=(case,cfg,data,identity,object(),{})
            with patch('ablr2.references.teacher_endpoint',side_effect=lambda wd,*_:endpoints[Path(wd).name]):
                value=validate_teacher_pair(cases[0].run_id,cases[1].run_id,root)
                self.assertTrue(value['passed']);self.assertFalse(value['final_weight_equality_required'])
                self.assertNotEqual(value['positive']['checkpoint_sha256'],value['zero']['checkpoint_sha256'])
                path=run_dir(cases[1].run_id,root)/'meta/training_start_manifest.json'
                row=read_json(path);row['sampler_hash']='d'*64;atomic_json(path,row)
                with self.assertRaisesRegex(ValueError,'native stream differ'):
                    validate_teacher_pair(cases[0].run_id,cases[1].run_id,root)


if __name__=='__main__':unittest.main()
