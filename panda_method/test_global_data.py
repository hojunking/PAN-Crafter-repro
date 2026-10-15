"""Disposable H5 integration tests; never turn synthetic b into a real asset."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from panda_method.global_data import (BAND_ORDER, GlobalRRDataSources,
    augment_global_batch, file_sha)
from panda_method.global_prior import GlobalPriorAsset
from panda_method.reconstruction import ReconstructionSpec
from panda_method.losses import SpatialLossSpec
from teacher48.global_training import build_global_rr_stage1, GlobalRRTrainer
from teacher48.prepare_global_prior import propose_train_prior, freeze_reviewed_proposal
from teacher48.test_newglobal_prior import config, source_fixture, review_for


torch.set_num_threads(1)


class Fixture:
    def __init__(self, root, sensor='GF2'):
        self.root, self.sensor = Path(root), sensor
        source = source_fixture(root, sensor=sensor)
        self.original = self.root / source['path']
        with h5py.File(self.original, 'a') as f:
            # Fixture deliberately uses broken MS/GT links to prove prior prep
            # cannot read them. Replace only these disposable test links.
            del f['ms']; del f['gt']
            n, c = f['lms'].shape[:2]
            # A finite negative interpolation lobe must survive the adapter.
            f['lms'][:, 0, 0, 0] = -7.
            f['ms'] = np.full((n, c, 16, 16), 31., dtype=np.float64)
            f['gt'] = np.clip(f['lms'][:] + 10., 0., 1023.)
        source['sha256'] = file_sha(self.original)
        source['band_order'] = list(BAND_ORDER[sensor])
        proposal = propose_train_prior(source, config(), root=root)
        self.asset = freeze_reviewed_proposal(proposal, review_for(proposal))
        self.overrides = None
        active = self.original
        if sensor == 'QB':
            active = self.root / 'train_msfix_synthetic.h5'
            with h5py.File(self.original) as f, h5py.File(active, 'w') as g:
                for key in ('pan', 'gt'):
                    g[key] = f[key][:]
                g['ms'] = np.full(f['ms'].shape, 91., dtype=np.float64)
                g['lms'] = f['lms'][:] + 100.
            self.overrides = {key: {'path': str(self.original), 'sha256': source['sha256'],
                                   'reason': 'synthetic original LMS user decision'}
                              for key in ('train', 'val')}
        rr_path = self.root / 'rr20_synthetic.h5'
        with h5py.File(rr_path, 'w') as f:
            for key, shape, value in (
                ('pan', (20, 1, 256, 256), 20.), ('ms', (20, c, 64, 64), 31.),
                ('lms', (20, c, 256, 256), 41.), ('gt', (20, c, 256, 256), 51.)):
                f.create_dataset(key, shape=shape, dtype='f8', fillvalue=value)
        self.active = active
        self.catalog = self.root / 'catalog.json'
        self.entry = {'band_order': list(BAND_ORDER[sensor]),
            'splits': {key: {'path': str(path), 'sha256': file_sha(path),
                             'source_identity': 'synthetic test only'}
                       for key, path in (('train', active), ('val', active),
                                         ('rr', rr_path), ('fr', rr_path))},
            'source_provenance': {'scope': 'synthetic-only, not geographic split proof'}}
        self.catalog.write_text(json.dumps({'schema': 'ABLR2_SENSOR_SOURCES_v1',
                                            'sensors': {sensor: self.entry}}))

    def sources(self):
        return GlobalRRDataSources(self.catalog, root=self.root, sensor=self.sensor,
                                   prior_asset=self.asset, lms_overrides=self.overrides)


class GlobalDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)

    def test_source_without_scene_mapping_uses_actual_stored_lms(self):
        sources = self.fixture.sources()
        rr = sources.batch('train', [0, 2])
        self.assertFalse(hasattr(rr.lineage, 'scene_ids'))
        with h5py.File(self.fixture.original) as f:
            expected = torch.tensor(f['lms'][[0, 2]], dtype=torch.float32) * (2./1023) - 1
        torch.testing.assert_close(rr.native_lms, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(rr.native_lms, torch.nn.functional.interpolate(rr.ms, scale_factor=4)))
        self.assertFalse(rr.prior_dy_dx.requires_grad)
        sources.training_sources().check(rr, 'RR')

    def test_qb_original_lms_and_repaired_ms_are_separately_bound(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, 'QB')
            sources = f.sources()
            rr = sources.batch('train', [0])
            with h5py.File(f.original) as h:
                expected = torch.tensor(h['lms'][0:1], dtype=torch.float32) * (2./2047) - 1
            torch.testing.assert_close(rr.native_lms, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(rr.ms, torch.full_like(rr.ms, 91*(2./2047)-1)))
            self.assertEqual(rr.lineage.train_source_sha256, file_sha(f.original))
            with self.assertRaisesRegex(ValueError, 'original-LMS'):
                GlobalRRDataSources(f.catalog, root=root, sensor='QB', prior_asset=f.asset)

    def test_qb_mismatched_row_identity_is_not_silently_mixed(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, 'QB')
            with h5py.File(f.active, 'a') as h:
                h['pan'][0, 0, 1, 1] += 1
            # Pin the changed fixture intentionally; comparison still catches
            # original-LMS vs repaired-PAN row identity mismatch.
            for split in ('train', 'val'):
                f.entry['splits'][split]['sha256'] = file_sha(f.active)
            f.catalog.write_text(json.dumps({'schema': 'ABLR2_SENSOR_SOURCES_v1',
                                             'sensors': {'QB': f.entry}}))
            with self.assertRaisesRegex(ValueError, 'different PAN/GT rows'):
                f.sources().batch('train', [0])

    def test_online_augmentation_changes_b_and_lms_together(self):
        sources = self.fixture.sources()
        original = sources.batch('train', [0])
        for k in range(4):
            actual = sources.batch('train', [0], augmentation=(True, True, k))
            expected = torch.rot90(original.native_lms.flip(-1).flip(-2), k, (-2, -1))
            torch.testing.assert_close(actual.native_lms, expected, rtol=0, atol=0)
            actual.validate(4, training=True, require_gt=True)
        with self.assertRaisesRegex(ValueError, 'Prior differs'):
            replace(original, prior_dy_dx=original.prior_dy_dx+.01).validate(4)

    def test_rr_whole_image_and_no_fr_fallback_or_eval_training(self):
        sources = self.fixture.sources()
        rr = sources.batch('rr', [19])
        self.assertEqual(tuple(rr.pan.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(rr.prior_dy_dx.shape), (1, 2))
        with self.assertRaisesRegex(ValueError, 'Only RR train'):
            rr.validate(4, training=True)
        with self.assertRaisesRegex(ValueError, 'FR bGlobal prior policy'):
            sources.batch('fr', [0])
        with self.assertRaisesRegex(ValueError, 'Only RR train'):
            sources.batch('val', [0]).validate(4, training=True)

    def test_source_mutation_and_wrong_asset_rejected(self):
        sources = self.fixture.sources()
        with h5py.File(self.fixture.original, 'a') as h:
            h['pan'][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'changed after verification'):
            sources.batch('train', [0])
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.fixture.sources()

    def test_stored_lms_negative_ringing_survives(self):
        # Direct round-trip checks do not clip even when the provided LMS
        # is outside nominal sensor DN (the source fixture has such values).
        sources = self.fixture.sources()
        raw = sources.raw('train', [0])['lms']
        self.assertLess(float(raw.min()), 0.)
        rr = sources.batch('train', [0])
        self.assertLess(float(rr.native_lms.min()), -1.)
        restored = (rr.native_lms.numpy()+1)*1023/2
        np.testing.assert_allclose(restored, raw, atol=1e-4, rtol=1e-6)

    def test_mislabeled_prior_and_unconnected_lpan_fail_early(self):
        payload = self.fixture.asset.manifest
        payload['normalization_and_band_handling']['band_order'].reverse()
        wrong = GlobalPriorAsset.from_dict(payload)
        with self.assertRaisesRegex(ValueError, 'Prior asset band order'):
            GlobalRRDataSources(self.fixture.catalog, root=self.tmp.name,
                                sensor='GF2', prior_asset=wrong)
        with self.assertRaisesRegex(ValueError, 'P0 only'):
            build_global_rr_stage1(self.fixture.sources(),
                teacher_spec=ReconstructionSpec(4, 8, (1, 1, 1), 'PLH', 'BICUBIC_MS'),
                spatial_spec=SpatialLossSpec('GF2', (2, 1, 0), 'forward_difference',
                                             'reflect', 3, 'synthetic_common_minus1_plus1'),
                aligner_view_margin=0, teacher_seed=51, aligner_seed=52, epsilon_seed=53,
                boundary_policy='report', calibration_protocol='synthetic train-only')

    def test_real_adapter_to_rr_only_optimizer_and_resume(self):
        sources = self.fixture.sources()
        def build():
            stage = build_global_rr_stage1(sources,
                teacher_spec=ReconstructionSpec(4, 8, (1, 1, 1), 'P0', 'BICUBIC_MS'),
                spatial_spec=SpatialLossSpec('GF2', (2, 1, 0), 'forward_difference', 'reflect',
                                             3, 'synthetic_common_minus1_plus1'),
                aligner_view_margin=0, teacher_seed=51, aligner_seed=52, epsilon_seed=53,
                boundary_policy='report', calibration_protocol='synthetic whole training no augment')
            return GlobalRRTrainer(stage, torch.optim.Adam(stage.parameters(), lr=1e-4))
        before_rng = torch.random.get_rng_state().clone()
        trainer = build()
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        rr = sources.batch('train', [0, 1], augmentation=(True, True, 1))
        metric = trainer.step(rr)
        self.assertEqual(metric['update'], 1)
        self.assertNotIn('eps_fr', metric)
        snapshot = trainer.state_dict()
        resumed = build(); resumed.load_state_dict(snapshot)
        a, b = trainer.step(rr), resumed.step(rr)
        self.assertEqual(a, b)
        reordered_stage = build().stage
        reordered = GlobalRRTrainer(reordered_stage,
            torch.optim.Adam(list(reversed(list(reordered_stage.parameters()))), lr=1e-4))
        before_param = next(reordered.stage.parameters()).detach().clone()
        with self.assertRaisesRegex(ValueError, 'name order mismatch'):
            reordered.load_state_dict(snapshot)
        torch.testing.assert_close(next(reordered.stage.parameters()), before_param, rtol=0, atol=0)
        reordered.optimizer.param_groups[0]['params'].reverse()
        with self.assertRaisesRegex(ValueError, 'name order changed'):
            reordered.step(rr)
        before = resumed.stage.forward_calls
        with self.assertRaises(ValueError):
            resumed.step(sources.batch('val', [0, 1]))
        self.assertEqual(before, resumed.stage.forward_calls)
        with self.assertRaisesRegex(ValueError, 'all and only'):
            GlobalRRTrainer(trainer.stage, torch.optim.Adam(trainer.stage.teacher.parameters(), lr=1e-4))


if __name__ == '__main__':
    unittest.main()
