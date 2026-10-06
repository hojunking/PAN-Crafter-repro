import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ta2.progress import progress_protocol, progress_records, progress_steps


class TinyModel(torch.nn.Module):
    def __init__(self, policy='BYPASS'):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0.));self.policy=policy
        self.aligner=None;self.global_shift=None;self.seen=[]
    def predict_correction(self, pan, reference):
        self.seen.append(float(reference.mean()))
        return pan.new_zeros((len(pan),2))+self.weight*0


class TinyBundle:
    def __init__(self):
        self.datasets={split:range(4) for split in ('train','val','rr','fr')}
        self.manifest={'test':'synthetic_only'};self.requests=[]
    def batch(self, split, ids, rots=None, device='cpu'):
        self.requests.append((split,tuple(ids),rots))
        size=256 if split=='rr' else 512 if split=='fr' else 64
        ramp=torch.linspace(-.5,.5,size,device=device)[None,None,None,:].expand(1,1,size,size).clone()
        result=dict(pan=ramp,ms=torch.zeros(1,4,size//4,size//4,device=device),
                    lms=torch.full((1,4,size,size),.2,device=device))
        if split!='fr':result['gt']=torch.full((1,4,size,size),.4,device=device)
        return result
    def case_shifts(self, case, split):
        return torch.tensor([[.25,-.5]]*4)


def cfg(reference='NATIVE_LMS'):
    return dict(total_updates=50000,a_reference=reference,case_id='TA2-B00')


class ProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fixed_grid_constant_gain_and_runtime_preservation(self):
        model,bundle=TinyModel(),TinyBundle();model.train();model.weight.grad=torch.tensor(9.)
        old_random=random.getstate();old_numpy=np.random.get_state();old_torch=torch.get_rng_state()
        report=progress_protocol(model,bundle,cfg(),1010,'cpu')
        self.assertEqual(len(report['response_rows']),4*4*16)
        self.assertTrue(all(row['gain']==0 for row in report['response_summary']))
        self.assertTrue(model.training);self.assertEqual(float(model.weight.grad),9.)
        self.assertEqual(random.getstate(),old_random)
        self.assertTrue(np.array_equal(np.random.get_state()[1],old_numpy[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),old_torch))
        self.assertEqual(bundle.requests,[(s,(i,),None) for s in ('train','val','rr','fr') for i in range(4)])
        self.assertFalse(report['official_metric_masking']);self.assertFalse(report['quality_selection_eligible'])

    def test_bicubic_reference_not_native_lms_and_no_fr_gt(self):
        model,bundle=TinyModel('LEARNED'),TinyBundle()
        progress_protocol(model,bundle,cfg('BICUBIC_MS'),25250,'cpu')
        self.assertTrue(all(abs(value)<1e-7 for value in model.seen))
        model.seen=[]
        progress_protocol(model,bundle,cfg(),25250,'cpu')
        self.assertTrue(all(abs(value-.2)<1e-6 for value in model.seen))

    def test_fixed_per_image_reestimated_and_held_modes_are_separate(self):
        model,bundle=TinyModel('PER_IMAGE_GRAD'),TinyBundle()
        calls=[]
        def estimator(p,reference,**kwargs):
            calls.append(kwargs)
            return {'shift':[.25,-.5]}
        with patch('ta2.probes.estimate_shift',side_effect=estimator):
            report=progress_protocol(model,bundle,cfg(),50000,'cpu')
        self.assertEqual({r['mode'] for r in report['response_summary']},{'REESTIMATE_CURRENT_INPUT','HOLD_NATIVE_CACHE'})
        self.assertEqual(len(report['response_rows']),2*4*4*16)
        self.assertTrue(all(r['max_dn']==1. for r in calls))

    def test_privileged_b06_gt_is_train_only(self):
        model,bundle=TinyModel('TRAIN_GT_PROXY'),TinyBundle()
        calls=[]
        def estimator(p,reference,**kwargs):
            calls.append((kwargs['split'],kwargs['target'],float(reference.mean())))
            return {'shift':[.25,-.5]}
        with patch('ta2.probes.estimate_shift',side_effect=estimator):
            report=progress_protocol(model,bundle,cfg(),1010,'cpu')
        self.assertTrue(all(target=='GT' and abs(mean-.7)<1e-6 for split,target,mean in calls if split=='train'))
        self.assertTrue(all(target=='LMS' and abs(mean-.6)<1e-6 for split,target,mean in calls if split!='train'))
        self.assertEqual(report['inputs']['train']['A_reference'],'GT_TRAIN_PRIVILEGED_PROXY')

    def test_unregistered_step_and_smaller_sample_set_rejected(self):
        self.assertEqual(progress_steps(100000),(2020,50500,100000))
        with self.assertRaises(ValueError):progress_protocol(TinyModel(),TinyBundle(),cfg(),1000,'cpu')
        bundle=TinyBundle();bundle.datasets['val']=range(3)
        with self.assertRaises(ValueError):progress_protocol(TinyModel(),bundle,cfg(),1010,'cpu')

    def test_curve_numeric_losses_updates_rho_and_no_reweight(self):
        context=dict(case_id='TA2-M07',dataset='WV3',server='s1',seed=261006101,replica=1,attempt=1,run_id='test')
        probe=dict(completed_step=1010,probe_seed=261006900,response_summary=[],correction_summary={})
        log=dict(completed_step=1010,rec=.1,epsilon=.2,weighted_epsilon=.0002,struct=.3,weighted_struct=.003,
                 epsilon_effective_count=48,actual_parameter_update={'U':.001,'A':.002,'global_shift':None},
                 actual_a_group_update={'fc2':.001},structural_statistics=[dict(sample_weight=.25,raw_rho=.8,
                     rho2=.64,valid_band_count=8.,low_texture_fraction=0.)])
        records=progress_records(context,log,probe)
        values={r['Metric_name']:r['Estimate'] for r in records}
        self.assertEqual(values['weighted_epsilon'],.0002)
        self.assertEqual(values['actual_parameter_update/A'],.002)
        self.assertEqual(values['actual_parameter_update/global_shift'],'')
        self.assertEqual(values['structural_microbatch0/raw_rho'],.8)
        self.assertTrue(all(r['Record_type']=='CURVE' for r in records))
        with self.assertRaises(ValueError):progress_records(context,dict(log,completed_step=1),probe)


if __name__=='__main__':unittest.main()
