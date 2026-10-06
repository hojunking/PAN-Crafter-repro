import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ta2.evaluation import (EvaluationLedger, hqnr_grid, select_hqnr_max50, seal, rr_scene,
                            evaluate_checkpoint, assert_checkpoint_model, preserve_runtime)
from ta2.diagnostics import (response_grid, response_metrics, correction_statistics, counterfactual_corrections,
                             gradient_probe, measure_responses, native_geometry, nested_crops, same_fov, cost_probe,
                             known_shift_control, run_diagnostics)
from ta2.aggregate import seed_aggregate, paired_difference, scene_paired_interval


def candidate(step, value=.95):
    rows = [dict(scene_index=i, hqnr=value, d_s=.02, d_lambda=.03) for i in range(20)]
    mean = float(np.mean([r['hqnr'] for r in rows]))
    fr = dict(hqnr=mean, d_s=float(np.mean([.02]*20)), d_lambda=float(np.mean([.03]*20)), per_scene=rows,
              n_scenes=20, reference='original_PAN_and_native_LMS', masking=False, support='full512')
    rr = dict(ergas=2., scc=.9, per_scene=[dict(scene_index=i, ergas=2., scc=.9) for i in range(20)], n_scenes=20)
    rr['scc'] = float(np.mean([.9]*20))
    return seal(dict(completed_step=step, complete=True, checkpoint_sha256=str(step), fr=fr, rr=rr, context={'run':'x'}))


class EvaluationTests(unittest.TestCase):
    def test_model_bytes_bound_to_checkpoint(self):
        from safetensors.torch import save_file
        net=torch.nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.safetensors';save_file(net.state_dict(),str(path))
            assert_checkpoint_model(net,path)
            with torch.no_grad():net.weight.add_(1)
            with self.assertRaises(ValueError):assert_checkpoint_model(net,path)

    def test_runtime_restores_rng_and_mixed_module_modes(self):
        import random
        net=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout())
        net.train();net[1].eval()
        rng=random.getstate();nrng=np.random.get_state();trng=torch.get_rng_state()
        with preserve_runtime(net):
            random.random();np.random.rand();torch.rand(8)
            self.assertFalse(net.training)
        self.assertTrue(net.training);self.assertFalse(net[1].training)
        self.assertEqual(rng,random.getstate());np.testing.assert_equal(nrng[1],np.random.get_state()[1])
        torch.testing.assert_close(trng,torch.get_rng_state())
    def test_grid_exact(self):
        self.assertEqual(hqnr_grid(50000), tuple(range(1010, 50000, 1010))+(50000,))
        self.assertEqual(len(hqnr_grid(100000)), 50)
        self.assertNotIn(50000, hqnr_grid(100000))

    def test_selector_full_precision_earliest_no_ergas(self):
        records = [candidate(s) for s in hqnr_grid(50000)]
        self.assertEqual(select_hqnr_max50(records, 50000)['completed_step'], 1010)
        records[-1] = candidate(50000, .950000000000001)
        self.assertEqual(select_hqnr_max50(records, 50000)['completed_step'], 50000)
        records[-1]['rr']['ergas'] = -100000
        self.assertEqual(select_hqnr_max50(records, 50000)['completed_step'], 50000)

    def test_selector_rejects_missing_duplicate_oracle_mask_partial(self):
        records = [candidate(s) for s in hqnr_grid(50000)]
        for bad in (records[:-1], records+[records[0]]):
            with self.assertRaises(ValueError): select_hqnr_max50(bad,50000)
        for field, value in (('record_type','ORACLE'), ('complete',False)):
            bad = copy.deepcopy(records); bad[0][field]=value
            with self.assertRaises(ValueError): select_hqnr_max50(bad,50000)
        for field,value in (('masking',True), ('n_scenes',19), ('support','masked')):
            bad=copy.deepcopy(records); bad[0]['fr'][field]=value
            with self.assertRaises(ValueError): select_hqnr_max50(bad,50000)

    def test_ledger_resume_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger=EvaluationLedger(temp,{'run':'x'},100000)
            ledger.register(candidate(50000))
            self.assertEqual(ledger.status()['candidates_complete'],0)
            for s in hqnr_grid(100000): ledger.register(candidate(s))
            recovered=EvaluationLedger(temp,{'run':'x'},100000)
            self.assertEqual(recovered.status()['status'],'HQNR_SELECTION_COMPLETE')
            with self.assertRaises(ValueError): recovered.register(candidate(2020,.96))

    def test_rr4_rr8_protocol_crop(self):
        rng=np.random.default_rng(5)
        for sensor,bands in [('WV3',8),('QB',4),('GF2',4)]:
            gt=rng.uniform(100,800,(bands,256,256)).astype(np.float32)
            pred=gt.copy(); pred[:,:20,:]=100000
            row=rr_scene(pred,gt,sensor)
            self.assertEqual(row['ergas'],0)
            self.assertAlmostEqual(row['q8' if bands==8 else 'q4'],1)
            self.assertEqual(row['rmse'],0)

    def test_gt_never_in_inference_and_raw_unclipped(self):
        class Net(torch.nn.Module):
            def forward(self,p,ms,l):
                return dict(prediction=l+3,correction=torch.zeros(1,2))
        class ForbiddenGT:
            def __getitem__(self,key): raise AssertionError('FR GT touched')
        fr=dict(pan=np.ones((20,1,8,8),np.float32),ms=np.ones((20,4,2,2),np.float32),
                lms=np.ones((20,4,8,8),np.float32),gt=ForbiddenGT())
        rr=dict(fr,gt=np.ones((20,4,8,8),np.float32))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'model'; checkpoint.write_bytes(b'model')
            with mock.patch('ta2.evaluation.evaluator_identity',return_value={}), \
                 mock.patch('ta2.evaluation.assert_checkpoint_model'), \
                 mock.patch('tools.metrics.eval_fr.load_dlpan',return_value=object()), \
                 mock.patch('ta2.evaluation.rr_scene',return_value=dict(ergas=1.,scc=.9,sam=1.,psnr=30.,ssim=.9,q4=.9,rmse=1.,cc=.9)), \
                 mock.patch('ta2.evaluation.fr_scene',return_value=dict(hqnr=.9,d_s=.05,d_lambda=.05,jqm=.9)):
                report=evaluate_checkpoint(Net(),{'rr':rr,'fr':fr},Path(temp)/'eval',sensor='QB',
                    checkpoint_path=checkpoint,completed_step=1010,device='cpu',export_raw=True)
                raw=np.load(Path(temp)/'eval/fr/unclipped_DN/scene_00.npy')
                self.assertGreater(raw.max(),2047)
                self.assertFalse(report['gt_in_inference'])
                self.assertTrue(report['rr_fr_same_checkpoint'])
                recovered=evaluate_checkpoint(Net(),{'rr':rr,'fr':fr},Path(temp)/'eval',sensor='QB',
                    checkpoint_path=checkpoint,completed_step=1010,device='cpu',export_raw=True)
                self.assertEqual(report,recovered)


class DiagnosticTests(unittest.TestCase):
    def test_positive_control_actual_estimator(self):
        from ta2.registration import estimate_shift
        p=torch.rand(1,1,64,64,generator=torch.Generator().manual_seed(12))
        row=known_shift_control(p,torch.tensor([[.5,-.25]]),estimator=estimate_shift)
        self.assertTrue(row['estimate']['valid'])
        self.assertLess(row['epe'],.15)
        self.assertEqual(row['independent_training_seeds'],0)

    def test_diagnostics_are_not_complete_with_empty_optional_records(self):
        # Use two actual input tensors but tiny probes; mandatory missing
        # callbacks remain PENDING rather than producing invented metrics.
        p=torch.rand(1,1,64,64);l=p.repeat(1,4,1,1)
        samples={'train':[(0,p,l,l),(1,p,l,l)]}
        def estimator(p,r,**kw):return dict(shift=[0.,0.],valid=True)
        grid=[dict(probe='AXIS16',radius=.25,epsilon=[.25,0.],ood=False,offset_index=0)]
        with mock.patch('ta2.diagnostics.response_grid',return_value=grid):
            result=run_diagnostics(samples=samples,correction=lambda s,p,l,i,m:p.new_zeros((1,2)),
                estimator=estimator,checkpoint_sha256='x',context={},include_sizes=False)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertIn('P06_FIXED32_BATCH_GRADIENTS_REQUIRED',result['pending'])
    def test_probe_geometry_counts_disk_determinism(self):
        grid=response_grid()
        for name,count in [('AXIS16',16),('DIAG16',16),('DISK64_HELD',64),('OOD_AXIS',8)]:
            rows=[r for r in grid if r['probe']==name]
            self.assertEqual(len(rows),count)
            for row in rows: self.assertAlmostEqual(np.linalg.norm(row['epsilon']),row['radius'])
        self.assertEqual(grid,response_grid())

    def test_constant_axis_mae_and_perfect_gain(self):
        eps=np.asarray([r['epsilon'] for r in response_grid() if r['probe']=='AXIS16'])
        constant=response_metrics(np.zeros_like(eps),eps)
        self.assertEqual(constant['component_mae'],.46875)
        self.assertEqual(constant['gain'],0)
        perfect=response_metrics(-eps,eps)
        self.assertEqual(perfect['gain'],1)
        self.assertEqual(perfect['epe'],0)

    def test_component_zero_denom_is_na(self):
        row=response_metrics([[0,-1]],[[0,1]])
        self.assertIsNone(row['gain_y']); self.assertEqual(row['gain_x'],1)

    def test_mean_norm_not_norm_mean(self):
        row=correction_statistics([[1,0],[-1,0]])
        self.assertEqual(row['norm_mean'],0); self.assertEqual(row['mean_norm'],1)

    def test_counterfactual_constants_train_only(self):
        c=counterfactual_corrections([[10,0],[20,0]],[[1,2],[3,4]])
        np.testing.assert_equal(c['train_mean'],[[2,3],[2,3]])
        np.testing.assert_equal(c['shuffled'],[[20,0],[10,0]])

    def test_gradient_raw_weighted_conflict_zero_na_no_grad_mutation(self):
        p=torch.nn.Parameter(torch.tensor([2.,3.])); p.grad=torch.tensor([11.,12.])
        rows=gradient_probe(dict(rec=p.sum(),eps=-p.sum(),struct=p.sum()*0),[('fc2.bias',p)],dict(rec=1,eps=.001,struct=.01))
        row=rows[0]
        self.assertAlmostEqual(row['cosine']['rec_eps'],-1)
        self.assertIsNone(row['cosine']['rec_struct'])
        self.assertAlmostEqual(row['weighted_norm']['eps']/row['raw_norm']['eps'],.001)
        torch.testing.assert_close(p.grad,torch.tensor([11.,12.]))
        self.assertEqual({r['group'] for r in rows},{'whole','fc2','bias'})

    def test_native_proxy_actual_rewarped_not_vector_subtraction(self):
        seen=[]
        def estimator(p,r,**kwargs):
            seen.append(np.array(p)); return dict(shift=[0.,0.],valid=True)
        p=torch.arange(64*64).float().reshape(1,1,64,64)/4096
        result=native_geometry(p,p,torch.tensor([[1.,0.]]),estimator=estimator)
        self.assertFalse(np.array_equal(seen[0],seen[1]))
        self.assertTrue(result['residual_reestimated_after_actual_warp'])

    def test_native_nested_vs_synthetic_fov(self):
        p=torch.rand(1,1,512,512); l=torch.rand(1,4,512,512)
        crops=list(nested_crops(p,l)); self.assertEqual([s for s,_,_ in crops],[64,128,256,512])
        pp,ll,meta=same_fov(p,l,64)
        self.assertEqual(pp.shape[-1],64); self.assertEqual(meta['scale'],1/8)
        self.assertFalse(meta['official_wald'])
        self.assertFalse(torch.equal(pp,crops[0][1]))

    def test_uncached_registration_cost(self):
        calls=[]
        result=cost_probe(lambda:calls.append('r'),lambda c:calls.append('u'),device='cpu',warmup=2,repeats=3)
        self.assertEqual(calls,['r','u']*5)
        self.assertAlmostEqual(result['registration_ms']+result['warp_u_ms'],result['pipeline_ms'])


class AggregateTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=s,status='COMPLETE',dataset='WV3',case_id='M07',selector='EXACT_FINAL',source_revision='r1',
                     replica=i,initial_u_sha256='u',initial_a_sha256='a',native_stream_sha256='s',hqnr=.9+i*.01)
                for i,s in enumerate([11,12,13])]

    def test_three_seed_std_failure_counts(self):
        rows=self.rows(); rows[-1]['status']='FAILED'
        result=seed_aggregate(rows,metric='hqnr',expected_seeds=[11,12,13])
        self.assertEqual(result['n_success'],2); self.assertEqual(result['n_failed'],1)
        self.assertEqual(result['n_attempted'],3); self.assertEqual(result['status'],'INCOMPLETE_3_SEEDS')
        self.assertAlmostEqual(result['std'],np.std([.9,.91],ddof=1))
        self.assertIsNone(result['seed_values']['13'])

    def test_duplicate_retry_or_oracle_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):seed_aggregate(rows+[rows[0]],metric='hqnr',expected_seeds=[11,12,13])
        rows[0]['record_type']='ORACLE'
        with self.assertRaises(ValueError):seed_aggregate(rows,metric='hqnr',expected_seeds=[11,12,13])

    def test_paired_identity_required_and_difference(self):
        t=self.rows(); c=copy.deepcopy(t)
        for r in c:r['case_id']='M05';r['hqnr']-=.01
        result=paired_difference(t,c,metric='hqnr',expected_seeds=[11,12,13])
        self.assertAlmostEqual(result['mean'],.01);self.assertEqual(result['n_pairs'],3)
        c[0]['native_stream_sha256']='different'
        with self.assertRaises(ValueError):paired_difference(t,c,metric='hqnr',expected_seeds=[11,12,13])

    def test_scene_ci_not_training_seed_std(self):
        t=[dict(scene_index=i,hqnr=.9+i*.001) for i in range(20)]
        c=[dict(scene_index=i,hqnr=.89+i*.001) for i in range(20)]
        row=scene_paired_interval(t,c,metric='hqnr')
        self.assertEqual(row['n_training_seeds'],1);self.assertEqual(row['n_scenes'],20)
        self.assertAlmostEqual(row['mean'],.01)


class ProbeIntegrationTests(unittest.TestCase):
    def test_fixed32_effective48_gradient_and_rng_isolation(self):
        from ta2.probes import measure_gradient_protocol
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.aligner=torch.nn.Linear(2,1,bias=False);self.global_shift=None;self.policy='LEARNED'
        class Bundle:
            datasets={'train':range(48)}
            def batch(self,split,ids,rots=None,device='cpu'):
                return {'x':torch.tensor(ids,device=device,dtype=torch.float32)-23.5}
        net=Net()
        cfg=dict(auxiliary='NONE',struct_descriptor='NCC',band_weights='UNIFORM',lambda_struct=.01,
                 struct_ramp_updates=0,lambda_epsilon=.001,lambda_scale=0,epsilon_every=1)
        def objective(model,batch,cfg,step,**kwargs):
            w=model.aligner.weight
            rec=w[0,0]*batch['x'].mean()+w[0,1]
            return dict(rec=rec,epsilon=-rec,struct=rec*2,scale=rec*0)
        rng=torch.get_rng_state().clone()
        with mock.patch('ta2.probes.compute_objective',side_effect=objective):
            result=measure_gradient_protocol(net,Bundle(),cfg,'cpu',1010)
        self.assertEqual(result['fixed_probe_batches'],32)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(sum(r['group']=='whole' for r in result['rows']),32)
        ids=np.random.Generator(np.random.PCG64(261006900)).integers(48,size=(32,48))
        for i in range(32):
            row=next(r for r in result['rows'] if r['group']=='whole' and r['probe_batch']==i)
            expected=np.linalg.norm([float((ids[i]-23.5).mean()),1])
            self.assertAlmostEqual(row['raw_norm']['rec'],expected,places=5)
            self.assertAlmostEqual(row['cosine']['rec_eps'],-1)
        torch.testing.assert_close(torch.get_rng_state(),rng)

    def test_protocol_preserves_B_vs_L_reference(self):
        from ta2.probes import run_protocol
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Parameter(torch.zeros(1));self.policy='BYPASS'
            def forward(self,p,ms,l,shift=None):return dict(prediction=l,correction=p.new_zeros((1,2)))
            def correction(self,p,ms,l):return p.new_zeros((len(p),2))
        class Dataset:
            def __len__(self):return 32
        class Bundle:
            root=Path('/tmp');manifest={};max_dn=2047; sensor='WV3'
            datasets={s:Dataset() for s in ('train','val','rr','fr')}
            def batch(self,split,ids,rots=None,device='cpu'):
                n=len(ids);x=dict(pan=torch.zeros(n,1,64,64),ms=torch.ones(n,8,16,16)*.25,
                                 lms=torch.ones(n,8,64,64)*.75)
                if split!='fr':x['gt']=torch.zeros(n,8,64,64)
                return x
        captured={}
        def diagnostics(**kwargs):
            captured.update(kwargs)
            # Native L is .875 in unit intensity, independently generated B .625.
            item=kwargs['samples']['rr'][0]
            self.assertTrue(torch.all(item[2]==.875));self.assertTrue(torch.allclose(item[4],torch.full_like(item[4],.625)))
            return seal(dict(status='COMPLETE',context=kwargs['context']))
        cfg={'a_reference':'BICUBIC_MS'}
        with tempfile.TemporaryDirectory() as temp:
            cp=Path(temp)/'model';cp.write_bytes(b'cp')
            with mock.patch('ta2.probes.assert_checkpoint_model'),mock.patch('tools.metrics.eval_fr.load_dlpan'),\
                 mock.patch('ta2.probes.cost_probe',return_value={'record_type':'COST'}),\
                 mock.patch('ta2.probes.run_diagnostics',side_effect=diagnostics):
                report=run_protocol(Net(),Bundle(),cfg,temp,cp,50000,device='cpu',gradient_records={'status':'NOT_APPLICABLE'})
        self.assertEqual(report['train_constant_samples'],32)


if __name__ == '__main__': unittest.main()
