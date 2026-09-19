#!/usr/bin/env python3
"""CPU/mock-only checks of FH20R1 diagnostics, branch gates, and cache provenance."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import yaml

import fh20r1.diagnostics as D
from fh20r1.common import object_sha,sha256
from fh20r1.plan import S1_SUPPORT,S1_FALLBACK
from fh12.plan import cases_for,build_config


RELEASE={'unit_test_source':True}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def metrics(h=.950,ds=.03,e=2.06,improved=20):
    fr=dict(hqnr=h,d_s=ds,d_lambda=.02,n_scenes=20,
            per_scene=[dict(hqnr=h,d_s=ds,d_lambda=.02) for _ in range(20)])
    rr=dict(ergas=e,scc=.98,psnr=37.,sam=2.8,q8=.91,ssim=.97,official_complete=True,n_scenes=20)
    rr['per_scene']=[{k:rr[k] for k in ('ergas','scc','psnr','sam','q8','ssim')} for _ in range(20)]
    return dict(fr=fr,rr=rr,shift=dict(fr_mean=[0.,0.],rr_mean=[0.,0.],fr_max_abs=0.,rr_max_abs=0.))


def cross(layout,supports=(True,True)):
    rows=[dict(step_A=50000,step_U=50000,**metrics())]
    for a,support in zip((10100,24240),supports):
        rows.append(dict(step_A=a,step_U=50000,**metrics(h=.952 if support else .949,
                     ds=.028 if support else .031,e=2.061)))
    return dict(source_run='old_'+layout,layout=layout,records=rows,diagonal_verified=True,
                complete=True,dependencies={})


def native():
    return {split:[dict(scene_id=i,dy=.1*i,dx=-.2*i) for i in range(20)] for split in ('rr','fr')}


class HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.aligner=torch.nn.Linear(1,1); self.forward_calls=0
    def forward(self,pan,*_):
        self.forward_calls+=1
        return {'delta':torch.arange(len(pan)*2,dtype=torch.float32).reshape(len(pan),2)}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='fh20r1-diag-tests-')
        self.root=Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def test_s1_needs_four_comparisons_and_plh_support(self):
        self.assertEqual(D.decide_s1([cross('P0'),cross('PLH')])['branch'],S1_SUPPORT)
        self.assertEqual(D.decide_s1([cross('P0'),cross('PLH',(False,False))])['branch'],S1_FALLBACK)
        missing=D.decide_s1([cross('PLH')])
        self.assertEqual(missing['branch'],S1_FALLBACK)
        self.assertFalse(missing['decision_evidence_complete'])

    def test_s1_each_threshold_and_scene_count(self):
        for field,value in (('hqnr',.9509),('d_s',.0291)):
            b=cross('PLH')
            for row in b['records'][1:]: row['fr'][field]=value
            self.assertEqual(D.decide_s1([cross('P0'),b])['branch'],S1_FALLBACK)
        b=cross('PLH')
        for row in b['records'][1:]: row['rr']['ergas']=2.064
        self.assertEqual(D.decide_s1([cross('P0'),b])['branch'],S1_FALLBACK)
        b=cross('PLH')
        for row in b['records'][1:]:
            for scene in row['fr']['per_scene'][11:]: scene['d_s']=.03
        self.assertEqual(D.decide_s1([cross('P0'),b])['branch'],S1_FALLBACK)
        for row in b['records'][1:]: row['fr']['per_scene'][11]['d_s']=.029
        self.assertEqual(D.decide_s1([cross('P0'),b])['branch'],S1_SUPPORT)

    def test_bad_diagonal_duplicate_or_nan_is_integrity_failure(self):
        a=cross('P0'); a['diagonal_verified']=False
        with self.assertRaises(ValueError): D.decide_s1([a,cross('PLH')])
        a=cross('P0'); a['records'].append(a['records'][0])
        with self.assertRaises(ValueError): D.decide_s1([a,cross('PLH')])
        a=cross('P0'); a['records'][0]['rr']['scc']=float('nan')
        with self.assertRaises(ValueError): D.decide_s1([a,cross('PLH')])
        with self.assertRaises(ValueError): D.decide_s1([cross('PLH'),cross('PLH')])

    def test_cross_hook_collects_native_c_without_extra_forward(self):
        model=HookModel()
        def evaluate():
            # Expanded views have correct shapes without allocating real images.
            model(torch.zeros(1).expand(20,1,512,512))
            model(torch.zeros(1).expand(20,1,256,256))
            return {'stub':True}
        result,rows=D._capture_cross_c(model,evaluate)
        self.assertEqual(result,{'stub':True}); self.assertEqual(model.forward_calls,2)
        self.assertEqual(rows['fr'][19],dict(scene_id=19,dy=38.,dx=39.))
        self.assertFalse(model._forward_hooks)

    def test_cross_hook_cleanup_on_failure_and_missing_scene(self):
        model=HookModel()
        with self.assertRaises(RuntimeError):
            D._capture_cross_c(model,lambda:(_ for _ in ()).throw(RuntimeError('bad evaluation')))
        self.assertFalse(model._forward_hooks)
        with self.assertRaises(ValueError): D._capture_cross_c(model,lambda:{})

    def fixture_cross(self):
        case=next(c for c in cases_for('s1') if c.role=='S' and c.input_layout=='PLH')
        run=case.run_id; wd=self.root/'work_dir'/run
        cfg=build_config(case)
        config=wd/'meta/config.resolved.yaml'; config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump(cfg))
        data=dict(recipe={'test':1},augmentation={'test':2},splits={})
        for split in ('train','val','rr','fr'):
            raw=self.root/f'{split}.h5'; raw.write_bytes(b'raw'+split.encode())
            lp=self.root/f'{split}_lp.h5'; lp.write_bytes(b'lp'+split.encode())
            data['splits'][split]=dict(dataroot=str(raw),lpan_path=str(lp),sha256=sha256(raw),
                lpan_sha256=sha256(lp),sample_order_sha256='order')
        write(self.root/'work_dir/_fh12/s1/dataset_manifest.json',data)
        rows=[]
        for step in D.STEPS:
            folder=wd/'candidates'/str(step); folder.mkdir(parents=True)
            (folder/'model.safetensors').write_bytes(f'checkpoint{step}'.encode())
            identity=dict(update=step,model_sha256=sha256(folder/'model.safetensors'),config_sha256=object_sha(cfg),source_identity=RELEASE)
            write(folder/'identity.json',identity)
            rows.append(dict(update=step,checkpoint_identity=identity,**metrics()))
        write(wd/'official/raw_grid.json',dict(config_sha256=object_sha(cfg),data_sha256=object_sha(data),records=rows))
        return run,wd,data

    def test_missing_optional_student_is_unavailable_not_bad_identity(self):
        with self.assertRaises(D.DiagnosticUnavailable):
            D.cross_run('not_downloaded','s1',{},'cpu',self.root)
        run,wd,data=self.fixture_cross()
        identity=wd/'candidates/10100/identity.json'; bad=json.loads(identity.read_text())
        bad['model_sha256']='0'*64; write(identity,bad)
        with patch.object(D,'source_identity',return_value=RELEASE),self.assertRaises(ValueError):
            D.cross_run(run,'s1',data,'cpu',self.root)

    def test_full_cross_cache_identity_and_scene_offsets(self):
        run,wd,data=self.fixture_cross()
        def evaluate(model,*args,**kwargs):
            model(torch.zeros(1).expand(20,1,512,512)); model(torch.zeros(1).expand(20,1,256,256))
            return metrics()
        with patch.object(D,'source_identity',return_value=RELEASE), \
             patch.object(D,'_numerical_origin'),patch.object(D,'build_dataset',return_value=object()), \
             patch.object(D,'FRMetrics',return_value=object()), \
             patch.object(D,'load_checkpoint_model',side_effect=lambda *a,**k:(HookModel(),{})), \
             patch.object(D,'load_file',return_value={'aligner.weight':torch.zeros(1,1),'aligner.bias':torch.zeros(1)}), \
             patch.object(D,'_native_c',return_value=native()), \
             patch.object(D,'evaluate_model',side_effect=evaluate) as ev, \
             patch('fh20r1.ledger.record_interval') as ledger:
            report=D.cross_run(run,'s1',data,'cpu',self.root)
            self.assertTrue(report['complete']); self.assertEqual(len(report['records']),9)
            self.assertEqual(ev.call_count,6); self.assertEqual(ledger.call_count,9)
            self.assertTrue(all(len(r['native_c']['fr'])==20 for r in report['records']))
            again=D.cross_run(run,'s1',data,'cpu',self.root)
            self.assertTrue(again['complete']); self.assertEqual(ev.call_count,6)
            self.assertEqual(ledger.call_count,9)
            # Existing malformed cache must never quietly trigger fallback/re-eval.
            p=self.root/'work_dir/_fh20r1/s1/diagnostics/au_cross'/run/'A10100_U50000.json'
            saved=json.loads(p.read_text()); saved['identity']['source_A_sha256']='changed';write(p,saved)
            with self.assertRaises(ValueError): D.cross_run(run,'s1',data,'cpu',self.root)

    def test_missing_core_data_is_hard_failure(self):
        run,wd,data=self.fixture_cross()
        Path(data['splits']['rr']['dataroot']).unlink()
        with self.assertRaises(ValueError): D.cross_run(run,'s1',data,'cpu',self.root)

    def test_q_shape_finiteness_and_cache_sha(self):
        path=self.root/'q.npz'
        np.savez(path,q=np.full((9714,4),.46875),per_radius_q=np.zeros((9714,4,4)),native_delta=np.zeros((9714,4,2)))
        bridge=dict(alias='F1',q_ref=.46875,tau_R=.012,resolved_artifacts={'q_cache_path':str(path)},
                    resolved_artifact_sha256={'q_cache_path':sha256(path)})
        report=D.q_report(bridge,'s1',self.root)
        self.assertEqual(report['s_mean'],.5); self.assertFalse(report['performance_gate'])
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): D.q_report(bridge,'s1',self.root)

    def test_diagnose_only_optional_absence_becomes_fallback(self):
        bridge={'dataset_manifest':{},'alias':'F1'}
        with patch('fh20r1.references.load_reference',return_value=(None,{},bridge,None)), \
             patch.object(D,'source_identity',return_value=RELEASE), \
             patch.object(D,'q_report',return_value={'dependencies':{}}), \
             patch.object(D,'cross_run',side_effect=D.DiagnosticUnavailable('optional old Student absent')):
            report=D.diagnose('s1','cpu',self.root)
        self.assertEqual(report['branch'],S1_FALLBACK)
        self.assertEqual(len(report['unavailable']),2)
        self.assertTrue((self.root/'work_dir/_fh20r1/s1/branch_record.json').exists())

    def test_malformed_or_disappeared_existing_artifact_is_not_fallback(self):
        for error in (KeyError('model_sha256'),FileNotFoundError('core LP disappeared'),ValueError('bad checksum')):
            with patch('fh20r1.references.load_reference',return_value=(None,{}, {'dataset_manifest':{}},None)), \
                 patch.object(D,'source_identity',return_value=RELEASE), \
                 patch.object(D,'q_report',return_value={'dependencies':{}}), \
                 patch.object(D,'cross_run',side_effect=error):
                with self.assertRaises(type(error)): D.diagnose('s1','cpu',self.root)
        self.assertFalse((self.root/'work_dir/_fh20r1/s1/branch_record.json').exists())

    def test_completed_report_rechecks_dependencies(self):
        path=self.root/'source.bin';path.write_bytes(b'original')
        bridge={'dataset_manifest':{}}
        output=self.root/'work_dir/_fh20r1/s3/diagnostics/report.json'
        write(output,dict(identity=dict(bridge_sha256=object_sha(bridge),consumer_source_identity=RELEASE),
                          dependencies={str(path):sha256(path)},complete=True))
        path.write_bytes(b'changed')
        with patch('fh20r1.references.load_reference',return_value=(None,{},bridge,None)), \
             patch.object(D,'source_identity',return_value=RELEASE),self.assertRaises(ValueError):
            D.diagnose('s3','cpu',self.root)

    def test_branch_cannot_be_first_chosen_after_training(self):
        camp=self.root/'work_dir/_fh20r1/s1'; report=camp/'diagnostics/report.json'
        write(report,{'done':True})
        write(self.root/'work_dir/FH20R1_S1_TEST/meta/training_start_manifest.json',{'started':True})
        with self.assertRaises(ValueError):
            D._lock_s1_branch(camp,dict(branch=S1_FALLBACK,reason='test'),report)

    def test_n2_calibration_cli_reports_q_even_on_cached_reuse(self):
        import tools.fh20r1_reference as cli
        bridge={'alias':'N2PL','verified':True}
        for reuse in (False,True):
            path=self.root/('reused.json' if reuse else 'fresh.json')
            if reuse: write(path,{'existing':True})
            stdout=io.StringIO()
            with patch.object(cli,'ROOT',self.root), \
                 patch.object(cli,'bridge_path',return_value=path), \
                 patch.object(cli,'calibrate_n2pl',return_value=path) as calibrate, \
                 patch.object(cli,'load_reference',return_value=(None,{},bridge,None)) as load, \
                 patch.object(D,'q_report',return_value={}) as report, \
                 patch.object(sys,'argv',['fh20r1_reference.py','calibrate','--server','s2','--device','cpu']), \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(),0)
            calibrate.assert_called_once()
            load.assert_called_once_with('N2PL','s2',self.root,device='cpu')
            report.assert_called_once_with(bridge,'s2',self.root)
            self.assertEqual(json.loads(stdout.getvalue())['reused'],reuse)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
