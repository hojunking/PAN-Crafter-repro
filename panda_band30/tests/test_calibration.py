"""Full synthetic train inventory and immutable per-frame calibration guards."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from panda_method.calibration import canonical_sha256
from panda_method.edge_data import augment_edge_batch
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.calibration import CalibrationStats
from panda_band30 import calibration as C
from panda_band30.method import frozen_predictions
from panda_band30.tests.test_method import make_donor, make_spec


class SyntheticInventory:
    """Only typed RR-train samples; the population has three known rows."""
    sensor = 'WV3'
    sample_ids = ('WV3:train:0','WV3:train:1','WV3:train:2')

    def __init__(self):
        self.verify_calls = 0
        self.seen = []
        self.rows = batch(count=3)

    def verify(self):
        self.verify_calls += 1

    def calibration_batches(self, size, *, device='cpu'):
        for first in range(0,3,size):
            sl = slice(first,min(first+size,3))
            line = replace(self.rows.lineage,sample_ids=self.sample_ids[sl],
                           augmentations=self.rows.lineage.augmentations[sl])
            rr = replace(self.rows,lineage=line,
                **{name:getattr(self.rows,name)[sl].to(device)
                   for name in ('pan','ms','native_lms','gt')})
            self.seen.extend(line.sample_ids)
            yield rr


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def prepared(self, outputs=2):
        registered = SyntheticInventory()
        donor = make_donor(outputs=outputs)
        donor.binding['sample_ids_sha256'] = canonical_sha256(list(registered.sample_ids))
        donor.binding['sample_count'] = 3
        return donor, registered

    def options(self,path,spec):
        return dict(spec=spec,output=path,k=5,batch_size=2,
                    memory_budget_bytes=8*1024**2,include_condq=True,device='cpu')

    def test_exact_all_rows_actual_loss_frame_and_geometry_binding(self):
        for frame in ('mid','mid_return'):
            donor, registered = self.prepared()
            spec = make_spec(frame,.5)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'stats.npz'
                progress=[]
                # Other collaborators may edit code; deterministic pin for this
                # synthetic test only, never used by production calibration.
                with patch.object(C,'code_pin',return_value={'synthetic':'fixed'}):
                    receipt=C.calibrate(donor,registered,**self.options(path,spec),on_progress=progress.append)
                bound=C.bind_donor(donor,spec)
                loaded=CalibrationStats.load(path,expected_binding=bound.binding,
                    expected_file_sha256=receipt['file_sha256'],expected_mask_policy=spec.mask_policy)
                self.assertEqual(receipt['sample_count'],3)
                self.assertEqual(receipt['n_q'],3*48*48)
                self.assertEqual(registered.seen,list(registered.sample_ids))
                self.assertEqual([p['seen'] for p in progress],[2,3])
                self.assertEqual(registered.verify_calls,2)
                errors=[]
                for rr in registered.calibration_batches(2):
                    f=frozen_predictions(donor.aligner,donor.teacher,rr,spec=spec)
                    errors.append((f['teacher_y']-f['gt_loss']).abs().mean(1,keepdim=True)[f['loss_mask']].numpy())
                self.assertEqual(loaded.tau_e,float(np.median(np.concatenate(errors).astype(np.float64))))
                for changed in (replace(spec,shift_scale=1.),replace(spec,frame='pan')):
                    with self.assertRaises(ValueError):
                        CalibrationStats.load(path,expected_binding=C.bind_donor(donor,changed).binding,
                                              expected_mask_policy=spec.mask_policy)
                with self.assertRaises(FileExistsError):
                    C.calibrate(donor,registered,**self.options(path,spec))

    def test_policy_bound_donor_cannot_be_relabelled(self):
        donor,_=self.prepared()
        bound=C.bind_donor(donor,make_spec('mid'))
        with self.assertRaisesRegex(ValueError,'pre-bound'):
            C.bind_donor(bound,make_spec('mid_return'))
        old=make_donor()
        old.binding.pop('band30_teacher')
        with self.assertRaisesRegex(ValueError,'fresh|Fresh'):
            C.bind_donor(old,make_spec())

    def test_incomplete_augmented_memory_rejected_publish_nothing(self):
        donor,registered=self.prepared()
        spec=make_spec('mid_return')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.npz'
            def interrupted(*args,**kwargs):
                yield batch()
                raise RuntimeError('INTERRUPTED')
            with patch.object(C,'code_pin',return_value={}), \
                 patch.object(registered,'calibration_batches',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'INTERRUPTED'):
                    C.calibrate(donor,registered,**self.options(path,spec))
            self.assertFalse(path.exists())
            augmented=augment_edge_batch(batch(),hflip=True,vflip=False,rot90=1)
            with patch.object(C,'code_pin',return_value={}), \
                 patch.object(registered,'calibration_batches',return_value=iter([augmented])):
                with self.assertRaisesRegex(ValueError,'unaugmented'):
                    C.calibrate(donor,registered,**self.options(path,spec))
            options=self.options(path,spec);options['memory_budget_bytes']=10
            with self.assertRaises(MemoryError):
                C.calibrate(donor,registered,**options)
            self.assertFalse(path.exists())

    def test_code_pin_includes_external_upsampler_and_shared_warp(self):
        pin=C.code_pin()
        for name in ('panda_band30/method.py','panda_band30/reconstruction.py',
                     'panda_midpoint/resampling.py','align/resample.py','pa/warp.py'):
            self.assertIn(name,pin)

    def test_band_full_inventory_q_operator_and_all_view_mask_bound(self):
        donor,registered=self.prepared(outputs=16)
        spec=make_spec('band_pan')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'band.npz'
            with patch.object(C,'code_pin',return_value={'synthetic':'fixed'}):
                receipt=C.calibrate(donor,registered,**self.options(path,spec))
            bound=C.bind_donor(donor,spec)
            loaded=CalibrationStats.load(path,expected_binding=bound.binding,
                expected_file_sha256=receipt['file_sha256'],expected_mask_policy=spec.mask_policy)
            self.assertEqual(receipt['sample_count'],3)
            self.assertEqual(receipt['n_q'],3*48*48)
            self.assertEqual(loaded.binding['band30_q_reference']['pan_reference'],
                             'MEAN_RGB_ALIGNED_PAN_VIEWS_BEFORE_SCHARR')
            self.assertEqual(loaded.binding['band30_reconstruction_spec']['input_channels'],16)
            wrong=loaded.binding
            wrong['band30_q_reference']['pan_reference']='SINGLE_ALIGNED_PAN'
            with self.assertRaises(ValueError):
                CalibrationStats.load(path,expected_binding=wrong,expected_mask_policy=spec.mask_policy)


if __name__ == '__main__':
    unittest.main()
