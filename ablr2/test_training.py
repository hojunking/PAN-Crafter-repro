import unittest
from unittest.mock import patch
from contextlib import ExitStack
from pathlib import Path
import tempfile
import yaml
import numpy as np
import torch

from ablr2.plan import cases_for,build_config,component_config
from ablr2.model import build_model,state_hash
from ablr2.training import cosine_factor,aligner_factor,make_optimizer,make_scheduler,_load_teacher,rng_state
from fh12.training import BatchStream,restore_rng
from ablr2.calibration import compute_calibration


class TrainingTests(unittest.TestCase):
    def test_cosine_and_r05_completed_update_boundary(self):
        self.assertEqual(cosine_factor(0),0.)
        self.assertEqual(cosine_factor(99),.99)
        self.assertEqual(cosine_factor(100),1.)
        self.assertEqual(cosine_factor(50000),0.)
        schedule='BASE_TIMES_1over3_AFTER_24240'
        self.assertEqual(aligner_factor(24239,schedule),1.)
        self.assertEqual(aligner_factor(24240,schedule),1/3)
        with self.assertRaises(ValueError):cosine_factor(0,total=100000)

    def test_identity_and_frozen_no_a_optimizer(self):
        teacher,_=build_model(bands=4,seed=8,role='T',width=8,depth=(1,1,1))
        for case_id in ('C00','C01','C09','C16'):
            case=next(c for c in cases_for('s2') if c.case_id==case_id)
            cfg=build_config(case)
            model,_=build_model(bands=4,seed=case.seed,role='S',component=case.component,
                teacher_aligner_state=teacher.aligner.state_dict() if case.component['teacher_A_clone_used'] else None,
                width=8,depth=(1,1,1))
            optimizer=make_optimizer(model,cfg)
            self.assertEqual([g['name'] for g in optimizer.param_groups],['U'])
            self.assertEqual(len(make_scheduler(optimizer).get_last_lr()),1)

    def test_teacherfree_loader_never_imports_or_reads_reference(self):
        for case_id in ('C00','C01','C02','C10'):
            case=next(c for c in cases_for('s1') if c.case_id==case_id)
            with patch('ablr2.references.load_reference',side_effect=AssertionError('unexpected reference')):
                with patch('ablr2.references.load_endpoint_only',side_effect=AssertionError('unexpected endpoint')):
                    self.assertEqual(_load_teacher(case,build_config(case),{},None,'cpu',None),(None,None,None))

    def test_matched_teacher_and_student_stream_prefixes(self):
        cases=cases_for('s1')
        teachers=[c for c in cases if c.sweep=='P01' and c.role=='T']
        students=[c for c in cases if c.sweep=='P01' and c.role=='S']
        for collection in (teachers,students):
            hashes=[]
            for case in collection:
                stream=BatchStream(9714,48,case.seed)
                hashes.append(state_hash({'order':stream.order,'rotations':stream.rotations}))
            self.assertEqual(len(set(hashes)),1)

    def test_batchstream_resume_exact(self):
        original=BatchStream(120,48,17)
        original.cursor=2
        expected=list(original.remaining_batches())
        restored=BatchStream(120,48,17);restored.load_state_dict(original.state_dict())
        self.assertEqual(list(restored.remaining_batches()),expected)

    def test_calibration_full_train_view_mean_not_median_or_half(self):
        q=np.array([[.1,.2,.3,.4],[.9,1.,2.,4.]],dtype=np.float32)
        raw=dict(q_ref=.3,tau_R=.2)
        with patch('ablr2.calibration._compute',return_value=(raw,dict(q=q))):
            got,_=compute_calibration(None,None,[],device='cpu',synthetic_test=True)
        expected=float(np.mean(.3/(.3+q.astype(np.float64))))
        self.assertEqual(got['s_bar'],expected)
        self.assertNotEqual(got['s_bar'],.5)
        self.assertEqual(got['schema'],'ABLR2_CALIBRATION_v1')

    def test_rng_isolation_restores_python_numpy_torch(self):
        saved=rng_state()
        expected=torch.rand(3);restore_rng(saved)
        torch.rand(199);np.random.rand(99);restore_rng(saved)
        self.assertTrue(torch.equal(torch.rand(3),expected))

    def test_real_optimizer_safe_stop_and_exact_resume_cpu_synthetic(self):
        """One real update, deliberately synthetic data/runtime; no production PASS."""
        from ablr2.training import train_run
        from ablr2.common import RuntimePaused,atomic_json,immutable_json
        class Dataset(torch.utils.data.Dataset):
            base_count=64
            def __len__(self):return 64
            def __getitem__(self,index):
                i,r=index if isinstance(index,(tuple,list)) else (index,0)
                generator=torch.Generator().manual_seed(int(i)+int(r)*100)
                gt=torch.rand(4,32,32,generator=generator)
                ms=torch.rand(4,8,8,generator=generator)
                lp=torch.rand(1,8,8,generator=generator)
                pan=torch.rand(1,32,32,generator=generator)
                return gt,gt,ms,lp,pan,torch.tensor([i,r,1,1])
        def stop_after_one(cfg,root,completed_update=0):
            if completed_update>=1:raise RuntimePaused('synthetic update-boundary test')
        actual_builder=build_model
        def small_builder(**kwargs):
            kwargs.update(width=8,depth=(1,1,1))
            return actual_builder(**kwargs)
        case=next(c for c in cases_for('s2') if c.case_id=='C02')
        cfg=build_config(case);cfg['ablr2']['runtime_policy_sha256']='a'*64
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg['ablr2']['dataset_manifest']=str(root/'data.json')
            atomic_json(root/'data.json',dict(sensor='QB',num_bands=4,max_pixel=2047,
                                           splits={'train':{'count':64}}))
            path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            pinned=root/cfg['work_dir']/'meta/config.resolved.yaml'
            immutable_json(pinned,cfg)
            pinned_bytes=pinned.read_bytes()
            with ExitStack() as stack:
                stack.enter_context(patch('ablr2.common.apply_runtime_policy',return_value={'sha256':'a'*64}))
                stack.enter_context(patch('ablr2.common.source_identity',return_value={'content_sha256':'b'*64}))
                stack.enter_context(patch('ablr2.common.check_runtime',side_effect=stop_after_one))
                stack.enter_context(patch('ablr2.training.runtime_context',return_value=({'synthetic_test':True},None)))
                stack.enter_context(patch('ablr2.controller.authorize_train',return_value=case))
                stack.enter_context(patch('ablr2.data.build_dataset',return_value=Dataset()))
                stack.enter_context(patch('ablr2.training.build_model',side_effect=small_builder))
                # Sandbox denies multiprocessing's local socket; exercise the
                # same batch sampler synchronously, without changing the config.
                real_loader=torch.utils.data.DataLoader
                def synchronous_loader(*args,**kwargs):
                    kwargs['num_workers']=0
                    return real_loader(*args,**kwargs)
                stack.enter_context(patch('ablr2.training.DataLoader',side_effect=synchronous_loader))
                self.assertEqual(train_run(path,device='cpu',root=root),75)
                state=torch.load(root/cfg['work_dir']/'last/training_state.pt',map_location='cpu',weights_only=False)
                digest=state_hash(state['model_state'])
                self.assertEqual(state['update'],1)
                self.assertEqual(state['scheduler']['last_epoch'],1)
                self.assertEqual(int(state['exposure_counts'].sum()),48)
                self.assertEqual(train_run(path,device='cpu',root=root,resume=True),75)
                recovered=torch.load(root/cfg['work_dir']/'last/training_state.pt',map_location='cpu',weights_only=False)
                self.assertEqual(recovered['update'],1)
                self.assertEqual(state_hash(recovered['model_state']),digest)
                self.assertEqual(pinned.read_bytes(),pinned_bytes)

    def test_direct_train_requires_controller_admission_and_exact_lane(self):
        from ablr2.training import train_run
        case=next(c for c in cases_for('s2') if c.case_id=='C00')
        cfg=build_config(case)
        cfg['ablr2']['runtime_policy_sha256']='a'*64
        cfg['ablr2']['dataset_manifest']='unused.json'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            with patch('ablr2.controller.authorize_train',side_effect=ValueError('not admitted')) as auth, \
                 patch('ablr2.data.build_dataset',side_effect=AssertionError('must not read data')):
                with self.assertRaisesRegex(ValueError,'not admitted'):train_run(path,device='cpu',root=root)
                self.assertEqual(auth.call_count,1)
                cfg['work_dir']='/outside/'+cfg['work_dir']
                path.write_text(yaml.safe_dump(cfg))
                with self.assertRaisesRegex(ValueError,'exact registered local lane'):train_run(path,device='cpu',root=root)
                self.assertEqual(auth.call_count,1)


if __name__=='__main__':unittest.main()
