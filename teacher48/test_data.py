"""Synthetic source-admission tests: never read production imagery or assets."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from teacher48.data import Sources, TrainingStream, GROUPS
from ta2.evaluation import digest, file_sha


BASELINE = {'RR_train_HW': [8, 8], 'FR_train_HW': [8, 8]}


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.h5 = self.root / 'synthetic.h5'
        with h5py.File(self.h5, 'w') as stream:
            stream['pan'] = np.arange(20*64, dtype=np.float32).reshape(20, 1, 8, 8) % 101
            stream['ms'] = np.full((20, 4, 2, 2), 21, dtype=np.float32)
            stream['lms'] = np.full((20, 4, 8, 8), 123, dtype=np.float32)
            stream['gt'] = np.full((20, 4, 8, 8), 234, dtype=np.float32)
        self.proof = self.root / 'proof.json'
        self.proof.write_text(json.dumps({'kind': 'synthetic unittest evidence, not training assets'}))
        proof_ref = {'path': self.proof.name, 'sha256': file_sha(self.proof)}
        priors = []
        for name in ('train-scene', 'held-scene'):
            priors.append(dict(source_scene_id=name, prior_shift_dy_dx=[.25, .5],
                prior_grid_name='canonical-pan', pixel_scale=[1., 1.], origin_phase=[0., 0.],
                n_windows=8, n_valid=7, inlier_fraction=.875, direction_concentration=.9,
                dispersion=.1, confidence=.9, estimator_version='synthetic-fixture',
                reference_type='native_LMS', scene_input_hash='a'*64,
                mapping_rule='explicit synthetic scene', accepted=True,
                acceptance_evidence='Unit test accepted synthetic prior; no scientific claim'))
        samples = {}
        for group in GROUPS:
            train = group.endswith('train')
            samples[group] = []
            for index in range(3 if train else 20):
                samples[group].append(dict(sample_id=f'{group}-sample-{index}',
                    scene_id=f'{group}-official-{index}', source_scene_id='train-scene' if train else 'held-scene',
                    acquisition_id='train-acquisition' if train else 'held-acquisition',
                    geometry_evidence='synthetic crop provenance', file_id='observations', row=index,
                    grid_name='canonical-pan', source_crop_native_pan_yxhw=[index*32, 0, 8, 8],
                    preparation_augmentation={'hflip': False, 'vflip': False, 'rot90': 0},
                    native_fr_training=(group == 'fr_train')))
        self.manifest = dict(schema='TEACHER48_SOURCES_V1', dataset='GF2', radiometry='native_DN',
            ratio=4, scene_priors=priors, samples=samples, grid_transforms=[],
            files={'observations': {'path': self.h5.name, 'sha256': file_sha(self.h5)}},
            uses_lpan=False, h5_keys={key: key for key in ('pan', 'ms', 'lms', 'gt')},
            split_separation_margin_native_pan=0,
            **{key: copy.deepcopy(proof_ref) for key in ('scene_mapping_evidence',
                'native_fr_training_evidence', 'lms_provenance_evidence', 'prior_acceptance_evidence',
                'evaluation_protocol_evidence')})
        self.path = self.root / 'sources.json'
        self.write()

    def write(self, *, refresh_audit=True):
        if refresh_audit:
            path = self.root / 'separation.json'
            path.write_text(json.dumps({'accepted': True, 'exact_duplicates': 0,
                'near_duplicates': 0, 'protocol': 'synthetic exhaustive fixture',
                'sample_index_sha256': digest(self.manifest['samples'])}))
            self.manifest['split_separation_evidence'] = {'path': path.name, 'sha256': file_sha(path)}
        self.path.write_text(json.dumps(self.manifest))

    def sources(self, *, scan=False):
        return Sources(self.path, dataset='GF2', baseline=BASELINE, scan=scan)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_training_only_native_lms_is_loaded_without_interpolation(self):
        sources = self.fixture.sources()
        rr = sources.batch('rr_train', [0])
        fr = sources.batch('fr_train', [0])
        self.assertEqual(rr.pan.shape, (1, 1, 8, 8))
        self.assertEqual(rr.ms.shape, (1, 4, 2, 2))
        self.assertTrue(torch.allclose(rr.ms, torch.full_like(rr.ms, 21*2/1023-1)))
        self.assertTrue(torch.allclose(rr.native_lms, torch.full_like(rr.native_lms, 123*2/1023-1)))
        self.assertFalse(torch.allclose(rr.native_lms.mean(), rr.ms.mean()))
        self.assertIsNone(fr.gt)
        rr.validate(4, training=True, require_gt=True)
        fr.validate(4, training=True)
        self.assertEqual(len(sources.expected_ids('rr_eval')), 20)
        with self.assertRaises(ValueError):
            sources.batch('rr_eval', [0]).validate(4, training=True)

    def test_requires_exact_full_twenty_official_scene_ids(self):
        self.fixture.manifest['samples']['rr_eval'].pop()
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'all20'):
            self.fixture.sources()

    def test_duplicate_official_ids_rejected(self):
        rows = self.fixture.manifest['samples']['fr_eval']
        rows[1]['scene_id'] = rows[0]['scene_id']
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'official scene IDs'):
            self.fixture.sources()

    def test_source_crop_overlap_refused_even_if_external_audit_claims_clean(self):
        a = self.fixture.manifest['samples']['rr_train'][0]
        b = self.fixture.manifest['samples']['fr_eval'][0]
        a['acquisition_id'] = b['acquisition_id']
        a['source_crop_native_pan_yxhw'] = list(b['source_crop_native_pan_yxhw'])
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.fixture.sources()

    def test_near_duplicate_audit_must_bind_current_sample_index(self):
        self.fixture.manifest['samples']['rr_train'][0]['geometry_evidence'] += ' changed'
        self.fixture.write(refresh_audit=False)
        with self.assertRaisesRegex(ValueError, 'complete sample index'):
            self.fixture.sources()

    def test_stale_data_hash_refused_without_even_loading_training(self):
        with h5py.File(self.fixture.h5, 'a') as stream:
            stream['ms'][0, 0, 0, 0] = 50
        with self.assertRaisesRegex(ValueError, 'Missing/stale evidence'):
            self.fixture.sources()

    def test_native_fr_evidence_missing_or_disclaimed_refused(self):
        self.fixture.manifest['native_fr_training_evidence']['path'] = 'absent.json'
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'Missing/stale evidence'):
            self.fixture.sources()
        self.fixture = Fixture(self.temp.name)
        self.fixture.manifest['samples']['fr_train'][0]['native_fr_training'] = False
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'native training'):
            self.fixture.sources()

    def test_zero_or_unaccepted_scene_prior_never_gets_fallback(self):
        prior = self.fixture.manifest['scene_priors'][0]
        prior['prior_shift_dy_dx'] = [0., 0.]
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'zero scene prior'):
            self.fixture.sources()
        prior['prior_shift_dy_dx'] = [.25, .5]
        prior['accepted'] = False
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'not explicitly accepted'):
            self.fixture.sources()

    def test_grid_conversion_not_automatically_factor_four(self):
        self.fixture.manifest['samples']['rr_train'][0]['grid_name'] = 'reduced-pan'
        self.fixture.write()
        sources = self.fixture.sources()
        with self.assertRaisesRegex(ValueError, 'no automatic x4'):
            sources.batch('rr_train', [0])
        self.fixture.manifest['grid_transforms'] = [dict(source_grid='canonical-pan', target_grid='reduced-pan',
            matrix_dy_dx=[[.25, 0.], [0., .25]], phase_dy_dx=[.03, -.01], evidence='synthetic verified phase')]
        self.fixture.write()
        batch = self.fixture.sources().batch('rr_train', [0])
        self.assertTrue(torch.allclose(batch.prior_dy_dx, torch.tensor([[.0925, .115]])))

    def test_scan_checks_geometry_and_native_dn(self):
        sources = self.fixture.sources()
        with self.assertRaisesRegex(ValueError, 'geometry'):
            sources._scan(BASELINE)  # our small H5 is intentionally not official-size eval
        with h5py.File(self.fixture.h5, 'a') as stream:
            stream['pan'][0, 0, 0, 0] = -1
        self.fixture.manifest['files']['observations']['sha256'] = file_sha(self.fixture.h5)
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'native sensor DN'):
            self.fixture.sources(scan=True)

    def test_online_augmentation_transforms_images_and_dy_dx_together(self):
        sources = self.fixture.sources()
        stream = TrainingStream(sources, 'rr_train', seed=1, augmentation_seed=2,
            microbatch=1, accumulation=1, augmentation={'hflip_probability': 1., 'vflip_probability': 0., 'rot90': False}, device='cpu')
        batch = stream()[0]
        original_index = int(batch.lineage.sample_ids[0].rsplit('-', 1)[-1])
        original = sources.batch('rr_train', [original_index])
        self.assertTrue(torch.equal(batch.pan, original.pan.flip(-1)))
        self.assertTrue(torch.allclose(batch.prior_dy_dx, torch.tensor([[.25, -.5]])))

    def test_prepared_h5_augmentation_transforms_only_prior_not_pixels_twice(self):
        row = self.fixture.manifest['samples']['rr_train'][0]
        row['preparation_augmentation'] = {'hflip': True, 'vflip': False, 'rot90': 1}
        self.fixture.write()
        sources = self.fixture.sources()
        batch = sources.batch('rr_train', [0])
        # hflip (.25,.5)->(.25,-.5), then CCW -> (.5,.25).
        self.assertTrue(torch.allclose(batch.prior_dy_dx, torch.tensor([[.5, .25]])))
        expected = torch.from_numpy(sources.arrays('rr_train', 0)['pan'])[None].mul(2/1023).sub(1)
        self.assertTrue(torch.equal(batch.pan, expected))

    def test_live_h5_modification_after_admission_is_rejected(self):
        sources = self.fixture.sources()
        with h5py.File(self.fixture.h5, 'a') as stream:
            stream['ms'][0, 0, 0, 0] = 66
        # Some mounted filesystems timestamp multiple writes in the same tick.
        # This exercises the explicit live-stat guard, not a claim that stat
        # metadata is a cryptographic replacement for the admission file hash.
        st = self.fixture.h5.stat()
        os.utime(self.fixture.h5, ns=(st.st_atime_ns, st.st_mtime_ns+1_000_000_000))
        with self.assertRaisesRegex(ValueError, 'changed|mutat|immutable|stale'):
            sources.arrays('rr_train', 0)

    def test_invalid_split_margin_cannot_bypass_crop_check(self):
        for value in (float('nan'), float('inf'), True, -1):
            with self.subTest(value=value):
                self.fixture.manifest['split_separation_margin_native_pan'] = value
                self.fixture.write()
                with self.assertRaises(ValueError):
                    self.fixture.sources()

    def test_source_crop_requires_native_integer_coordinates(self):
        row = self.fixture.manifest['samples']['rr_train'][0]
        for crop in ([0, 0, float('nan'), 8], [0, 0, True, 8], [0, 0, 8.5, 8], [-1, 0, 8, 8]):
            with self.subTest(crop=crop):
                row['source_crop_native_pan_yxhw'] = crop
                # NaN must not enter the canonical immutable sample-index digest.
                if any(isinstance(value, float) and not np.isfinite(value) for value in crop):
                    with self.assertRaises(ValueError):
                        self.fixture.write()
                    continue
                self.fixture.write()
                with self.assertRaises(ValueError):
                    self.fixture.sources()

    def test_stream_rejects_eval_group_and_empty_or_fractional_batch(self):
        sources = self.fixture.sources()
        kwargs = dict(seed=1, augmentation_seed=2, microbatch=1, accumulation=1,
            augmentation={'hflip_probability': 0., 'vflip_probability': 0., 'rot90': False}, device='cpu')
        for group, changes in (('rr_eval', {}), ('fr_eval', {}), ('rr_train', {'microbatch': 0}),
                               ('rr_train', {'accumulation': .5}), ('rr_train', {'microbatch': True})):
            with self.subTest(group=group, changes=changes), self.assertRaises(ValueError):
                TrainingStream(sources, group, **{**kwargs, **changes})

    def test_independent_shuffle_augmentation_and_branch_rng(self):
        sources = self.fixture.sources()
        kwargs = dict(seed=123, augmentation_seed=456, microbatch=2, accumulation=2,
            augmentation={'hflip_probability': .5, 'vflip_probability': .5, 'rot90': True}, device='cpu')
        first = TrainingStream(sources, 'rr_train', **kwargs)
        reference = TrainingStream(sources, 'rr_train', **kwargs)
        other = TrainingStream(sources, 'fr_train', **{**kwargs, 'seed': 987, 'augmentation_seed': 876})
        torch_state = torch.get_rng_state().clone()
        np_state = np.random.get_state()
        for _ in range(4):
            left = first()
            other(); other()
            right = reference()
            for a, b in zip(left, right):
                self.assertEqual(a.lineage.sample_ids, b.lineage.sample_ids)
                self.assertTrue(torch.equal(a.pan, b.pan))
                self.assertTrue(torch.equal(a.prior_dy_dx, b.prior_dy_dx))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(first.state(), reference.state())


if __name__ == '__main__':
    unittest.main()
