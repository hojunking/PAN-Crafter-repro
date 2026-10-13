"""Disposable H5 -> EdgeMatch -> Stage1 update/resume; no real experiment."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from panda_method.edge_data import (BAND_ORDER, EdgeDataSources, augment_edge_batch,
                                    file_sha)
from panda_method.edge_match import EdgeMatchSpec
from panda_method.reconstruction import ReconstructionSpec
from teacher48.edge_training import build_edge_stage1, EdgeStage1Trainer


torch.set_num_threads(1)


class Fixture:
    def __init__(self, root, sensor='GF2'):
        self.root, self.sensor = Path(root), sensor
        c = 8 if sensor == 'WV3' else 4
        rng = np.random.RandomState(73)
        ref = rng.uniform(150., 700., (3, 1, 64, 64)).astype('f4')
        original = self.root / 'train_original.h5'
        with h5py.File(original, 'w') as f:
            f['pan'] = np.roll(ref, 1, axis=-2)
            lms = np.repeat(ref, c, axis=1)
            lms[:, 0, 0, 0] = -7.
            f['lms'] = lms
            f['gt'] = np.clip(lms + 10., 0., 1023.)
            f['ms'] = np.full((3, c, 16, 16), 31., dtype='f4')
        active = original
        self.overrides = None
        if sensor == 'QB':
            active = self.root / 'train_msfix.h5'
            with h5py.File(original) as f, h5py.File(active, 'w') as g:
                for key in ('pan', 'gt'):
                    g[key] = f[key][:]
                g['lms'] = f['lms'][:] + 100.
                g['ms'] = np.full((3, c, 16, 16), 91., dtype='f4')
            self.overrides = {split: {'path': str(original), 'sha256': file_sha(original),
                                      'reason': 'Synthetic original LMS decision'}
                              for split in ('train', 'val')}
        paths = {'train': active, 'val': active}
        for split, side in (('rr', 256), ('fr', 512)):
            path = self.root / (split + '_synthetic20.h5')
            with h5py.File(path, 'w') as f:
                for key, shape, value in (
                    ('pan', (20, 1, side, side), 20.),
                    ('ms', (20, c, side//4, side//4), 31.),
                    ('lms', (20, c, side, side), 41.)):
                    f.create_dataset(key, shape=shape, dtype='f4', fillvalue=value)
                if split == 'rr':
                    f.create_dataset('gt', shape=(20, c, side, side), dtype='f4', fillvalue=51.)
                else:
                    f['gt'] = h5py.ExternalLink('forbidden_fr_gt.h5', '/gt')
            paths[split] = path
        self.catalog = self.root / 'catalog.json'
        self.entry = {'band_order': list(BAND_ORDER[sensor]), 'splits': {
            key: {'path': str(path), 'sha256': file_sha(path), 'source_identity': 'synthetic-only'}
            for key, path in paths.items()}}
        self.original, self.active = original, active
        self.save_catalog()

    def save_catalog(self):
        self.catalog.write_text(json.dumps({'schema': 'ABLR2_SENSOR_SOURCES_v1',
                                            'sensors': {self.sensor: self.entry}}))

    def sources(self):
        return EdgeDataSources(self.catalog, root=self.root, sensor=self.sensor,
                               lms_overrides=self.overrides)


def trainer(sources, *, cadence=1):
    spec = EdgeMatchSpec(sources.sensor, 2, 3, 'min_norm_then_lexicographic', 0., 'zero', 'report')
    stage = build_edge_stage1(sources,
        teacher_spec=ReconstructionSpec(sources.bands, 8, (1, 1, 1), 'P0', 'BICUBIC_MS'),
        match_spec=spec, aligner_view_margin=0, roi_margin=4,
        teacher_seed=81, aligner_seed=82, epsilon_seed=83, boundary_policy='report')
    return EdgeStage1Trainer(stage, torch.optim.Adam(stage.parameters(), lr=1e-4),
                             gradient_log_interval=cadence)


class EdgeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_no_scene_or_global_assets_and_preserved_lms(self):
        sources = self.fixture.sources()
        self.assertFalse(hasattr(sources, 'asset'))
        rr = sources.batch('train', [0, 1])
        for key in ('prior_dy_dx', 'scene_ids', 'prior_mapping_sha256'):
            self.assertFalse(hasattr(rr, key))
            self.assertFalse(hasattr(rr.lineage, key))
        with h5py.File(self.fixture.original) as f:
            expected = torch.tensor(f['lms'][0:2]) * (2./1023) - 1
        torch.testing.assert_close(rr.native_lms, expected, rtol=0, atol=0)
        self.assertLess(float(rr.native_lms.min()), -1.)
        sources.training_sources().check(rr)

    def test_original_qb_lms_with_separate_fixed_lrms(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, 'QB')
            rr = fixture.sources().batch('train', [1])
            with h5py.File(fixture.original) as f:
                expected = torch.tensor(f['lms'][1:2]) * (2./2047) - 1
            torch.testing.assert_close(rr.native_lms, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(rr.ms, torch.full_like(rr.ms, 91*(2./2047)-1)))
            with self.assertRaisesRegex(ValueError, 'original LMS'):
                EdgeDataSources(fixture.catalog, root=directory, sensor='QB')

    def test_qb_row_mismatch_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, 'QB')
            sources = fixture.sources()
            with h5py.File(fixture.active, 'a') as f:
                f['pan'][0, 0, 10, 10] += 1.
            with self.assertRaisesRegex(ValueError, 'changed after'):
                sources.batch('train', [0])
            for split in ('train', 'val'):
                fixture.entry['splits'][split]['sha256'] = file_sha(fixture.active)
            fixture.save_catalog()
            with self.assertRaisesRegex(ValueError, 'different PAN/GT'):
                fixture.sources().batch('train', [0])

    def test_augmentation_changes_images_not_a_cached_prior(self):
        sources = self.fixture.sources()
        original = sources.batch('train', [0])
        actual = sources.batch('train', [0], augmentation=(True, False, 1))
        for key in ('pan', 'ms', 'native_lms', 'gt'):
            torch.testing.assert_close(getattr(actual, key),
                torch.rot90(getattr(original, key).flip(-1), 1, (-2, -1)), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'once'):
            augment_edge_batch(actual, hflip=False, vflip=False, rot90=0)

    def test_rr256_fr512_inference_and_no_fr_gt_or_training(self):
        sources = self.fixture.sources()
        stage = trainer(sources).stage.eval()
        with torch.inference_mode():
            for split, side in (('rr', 256), ('fr', 512)):
                batch = sources.batch(split, [19])
                out = stage.native_prediction(batch)
                self.assertEqual(tuple(out['y'].shape), (1, 4, side, side))
                self.assertEqual(tuple(out['u'].shape), (1, 2))
                self.assertTrue(torch.equal(out['u'], torch.zeros(1, 2)))
                if split == 'fr':
                    self.assertIsNone(batch.gt)
                with self.assertRaises(ValueError):
                    batch.validate(4, training=True, require_gt=True)
        fr = sources.batch('fr', [0])
        with self.assertRaisesRegex(ValueError, 'no HRMS GT'):
            replace(fr, gt=fr.native_lms).validate(4)

    def test_update_logging_gradient_routes_and_identical_resume(self):
        sources = self.fixture.sources()
        before_rng = torch.random.get_rng_state().clone()
        first = trainer(sources)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        rr = sources.batch('train', [0, 1], augmentation=(True, False, 1))
        metric = first.step(rr)
        json.dumps(metric, allow_nan=False)
        self.assertEqual(metric['update'], 1)
        self.assertEqual(metric['term_gradient_norm']['abs_rr']['teacher'], 0.)
        self.assertEqual(metric['term_gradient_norm']['weighted_eps_rr']['teacher'], 0.)
        self.assertGreater(metric['term_gradient_norm']['rec']['teacher'], 0.)
        self.assertEqual(metric['alignment']['native']['resolution'], 'RR')
        snapshot = first.state_dict()
        resumed = trainer(sources)
        resumed.load_state_dict(snapshot)
        self.assertEqual(first.step(rr), resumed.step(rr))
        for key, value in first.stage.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, resumed.stage.state_dict()[key], rtol=0, atol=0)
        rng, calls = resumed.stage._epsilon_rng.get_state().clone(), resumed.stage.forward_calls
        with self.assertRaises(ValueError):
            resumed.step(sources.batch('val', [0]))
        self.assertTrue(torch.equal(rng, resumed.stage._epsilon_rng.get_state()))
        self.assertEqual(calls, resumed.stage.forward_calls)

    def test_gradient_diagnostics_do_not_change_update(self):
        sources = self.fixture.sources()
        logged, sparse = trainer(sources, cadence=1), trainer(sources, cadence=100)
        rr = sources.batch('train', [0])
        for _ in range(2):
            a, b = logged.step(rr), sparse.step(rr)
            self.assertEqual(a['loss'], b['loss'])
        self.assertIsNone(b['term_gradient_norm'])
        for pa, pb in zip(logged.stage.parameters(), sparse.stage.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    def test_order_and_old_trainer_checkpoint_rejected(self):
        sources = self.fixture.sources()
        current = trainer(sources)
        saved = current.state_dict()
        bad = copy.deepcopy(saved); bad['schema'] = 'PANDA_BGLOBAL_RR_TRAINER_V1'
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            current.load_state_dict(bad)
        other_stage = trainer(sources).stage
        wrong = EdgeStage1Trainer(other_stage,
            torch.optim.Adam(list(reversed(list(other_stage.parameters()))), lr=1e-4),
            gradient_log_interval=1)
        before = next(other_stage.parameters()).detach().clone()
        with self.assertRaisesRegex(ValueError, 'name/group order mismatch'):
            wrong.load_state_dict(saved)
        torch.testing.assert_close(before, next(other_stage.parameters()), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
