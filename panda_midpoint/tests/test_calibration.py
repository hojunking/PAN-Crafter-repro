"""Actual synthetic-H5 import and complete train-only midpoint calibration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from donor24 import plan as donor_plan
from donor24.evaluation import model_state_hash
from donor30.artifacts import SCHEMA as DONOR_SCHEMA
from donor30.runtime import build_trainer
from panda_stage2_condq.calibration import CalibrationStats
from panda_stage2_condq.donor import load_donor
from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.spec import MASK2_POLICY
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from ta2.evaluation import file_sha

from panda_midpoint import calibration as C
from panda_midpoint.method import frozen_midpoint_predictions
from panda_midpoint.runtime import Campaign
from panda_midpoint.spec import geometry_manifest


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        synthetic_sources(cls.root, count=3)
        # Nonconstant images and nonzero trained-head substitutes make target
        # transport observable; zero modules/constant images conceal warp bugs.
        generator = np.random.default_rng(71019)
        with h5py.File(cls.root/'train.h5', 'r+') as handle:
            for name in ('pan', 'ms', 'lms', 'gt'):
                handle[name][:] = generator.uniform(500, 1400, handle[name].shape).astype(np.float32)
        catalog_path = cls.root/'sources.json'
        catalog = json.loads(catalog_path.read_text())
        catalog['sensors']['WV3']['splits']['train']['sha256'] = file_sha(cls.root/'train.h5')
        catalog_path.write_text(json.dumps(catalog))
        cls.registered = RegisteredRRTrain.from_catalog(catalog_path, root=cls.root, sensor='WV3')
        cfg = donor_plan.resolve_case(donor_plan.default_config('s1'), 'A0')
        cfg['run_id'] = 'SYNTHETIC_MIDPOINT_CALIBRATION_NOT_AN_EXPERIMENT'
        trainer = build_trainer(cfg, cls.registered.sources)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(71019)
            trainer.stage.aligner.fc2.bias.copy_(torch.tensor([.8, -1.2]))
            trainer.stage.teacher.backbone.output[-1].weight.normal_(std=.0002)
        trainer.updates = 7
        identity = {'model_state_sha256': model_state_hash(trainer.stage), 'completed_update': 7,
                    'run_id': cfg['run_id'], 'dataset': 'WV3'}
        payload = {'schema': DONOR_SCHEMA, 'config': cfg, 'trainer': trainer.state_dict(),
                   'identity': identity, 'state_hash': identity['model_state_sha256']}
        cls.path = cls.root/'synthetic_donor.pt'
        torch.save(payload, cls.path)
        cls.sha = file_sha(cls.path)
        del trainer

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def donor(self):
        return load_donor(self.path, self.sha, self.registered)

    def kwargs(self, output):
        return dict(output=output, k=5, batch_size=2, memory_budget_bytes=4*1024**2,
                    include_condq=True, device='cpu', mask_policy=MASK2_POLICY)

    def test_complete_train_population_midpoint_tau_mask_and_binding(self):
        donor = self.donor()
        original_binding = dict(donor.binding)
        output = self.root/'complete_midpoint.npz'
        progress = []
        result = C.calibrate(donor, self.registered, on_progress=progress.append, **self.kwargs(output))
        bound = C.bind_donor(donor)
        self.assertEqual(donor.binding, original_binding)
        self.assertNotIn('midpoint_geometry', donor.binding)
        self.assertIs(bound.aligner, donor.aligner)
        self.assertIs(bound.teacher, donor.teacher)
        self.assertEqual(bound.binding['midpoint_geometry'], geometry_manifest())
        self.assertEqual(result['kind'], 'exact_condq')
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['n_q'], 3*48*48)
        self.assertEqual([row['seen'] for row in progress], [2, 3])
        loaded = CalibrationStats.load(output, expected_binding=bound.binding,
            expected_file_sha256=result['file_sha256'], expected_mask_policy=MASK2_POLICY)
        self.assertEqual(loaded.artifact_sha256, result['artifact_sha256'])
        errors = []
        for rr in self.registered.calibration_batches(2):
            out = frozen_midpoint_predictions(bound.aligner, bound.teacher, rr, sensor='WV3')
            error = (out['teacher_y'] - out['gt_midpoint']).abs().mean(1, keepdim=True)
            errors.append(error[out['loss_mask']].numpy())
        values = np.concatenate(errors).astype(np.float64)
        self.assertEqual(loaded.tau_e, float(np.median(values)))
        self.assertEqual(result['n_e'], len(values))
        self.assertLess(result['n_e'], 3*64*64)
        self.assertEqual(set(self.registered.sources._files), {('train', False)})
        with self.assertRaises(ValueError):
            CalibrationStats.load(output, expected_binding=original_binding,
                                  expected_mask_policy=MASK2_POLICY)
        with self.assertRaises(FileExistsError):
            C.calibrate(donor, self.registered, **self.kwargs(output))
        donor.assert_frozen()

    def test_original_mask2_archive_is_not_a_midpoint_calibration(self):
        from panda_stage2_condq.runtime import calibrate as original_calibrate
        donor = self.donor()
        output = self.root/'original_geometry.npz'
        original_calibrate(donor, self.registered, **self.kwargs(output))
        with self.assertRaises(ValueError):
            CalibrationStats.load(output, expected_binding=C.bind_donor(donor).binding,
                                  expected_mask_policy=MASK2_POLICY)

    def test_interrupted_and_memory_rejected_calibration_publish_nothing(self):
        donor = self.donor()
        output = self.root/'interrupted_midpoint.npz'
        def failing(*args, **kwargs):
            yield self.registered.batch([0])
            raise RuntimeError('SYNTHETIC_INTERRUPT')
        with patch.object(self.registered, 'calibration_batches', side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_INTERRUPT'):
                C.calibrate(donor, self.registered, **self.kwargs(output))
        self.assertFalse(output.exists())
        options = self.kwargs(output)
        options['memory_budget_bytes'] = 10
        with self.assertRaises(MemoryError):
            C.calibrate(donor, self.registered, **options)
        self.assertFalse(output.exists())
        donor.assert_frozen()

    def test_runtime_recovers_complete_archive_without_receipt_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign('s3', output=directory, device='cpu')
            campaign.registered = self.registered
            donor = C.bind_donor(self.donor())
            path = campaign.root/'calibration/M_MIDPOINT_MASK2.npz'
            C.calibrate(donor, self.registered, **self.kwargs(path))
            self.assertFalse(path.with_suffix('.json').exists())
            with patch.object(campaign, 'load_donor', return_value=donor), \
                    patch.object(C, 'calibrate', side_effect=AssertionError('must not recalibrate')):
                actual, receipt = campaign.calibration()
                self.assertEqual(actual, path)
                self.assertEqual(receipt['file_sha256'], file_sha(path))
                self.assertEqual(receipt['donor_binding'], donor.binding)
                self.assertTrue(path.with_suffix('.json').is_file())
                altered = json.loads(path.with_suffix('.json').read_text())
                altered['file_sha256'] = 'f'*64
                path.with_suffix('.json').write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, 'SHA changed'):
                    campaign.calibration()


if __name__ == '__main__':
    unittest.main()
