"""Offline CLI smoke on temporary synthetic assets; no corpus or training."""
import copy
import contextlib
import io
import json
import tempfile
import unittest

import h5py
import torch

from teacher48.prepare import main, match_rr, prepare_prior
from teacher48.prepare_assets import LMS_RECIPE
from teacher48.test_prepare_assets import NativeFixture
from teacher48.test_prepare_prior import config


class PrepareCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = NativeFixture(self.tmp.name)
        self.f.scene.update(pixel_scale=[1., 1.], origin_phase=[0., 0.])
        self.f.write()
        self.config = self.f.root/'policy.json'
        self.config.write_text(json.dumps(config()))
        self.rr = self.f.root/'rr.h5'
        with h5py.File(self.rr, 'w') as f:
            f['gt'] = self.f.ms[None, :, 5:12, 9:17]

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(list(map(str, args)))

    def test_inventory_new_output_and_existing_refusal(self):
        out = self.f.root/'inventory.json'
        result = self.run_cli('inventory', '--root', self.f.root, '--output', out)
        self.assertEqual(json.loads(out.read_text()), result)
        self.assertFalse(result['admission_granted'])
        with self.assertRaises(FileExistsError):
            self.run_cli('inventory', '--root', self.f.root, '--output', out)

    def test_existing_dangling_symlink_refused(self):
        out = self.f.root/'dangling.json'
        out.symlink_to(self.f.root/'absent.json')
        with self.assertRaises(FileExistsError):
            self.run_cli('inventory', '--root', self.f.root, '--output', out)
        self.assertFalse((self.f.root/'absent.json').exists())

    def test_crop_native_cli_stages_without_admission(self):
        out = self.f.root/'crops'
        result = self.run_cli('crop-native', '--catalog', self.f.path, '--output', out,
                              '--patch', 64, 64, '--stride', 64, 64, '--split-margin', 0)
        self.assertEqual(result['status'], 'STAGED_NOT_ADMITTED')
        self.assertTrue((out/'preparation.json').exists())
        with self.assertRaises(FileExistsError):
            self.run_cli('crop-native', '--catalog', self.f.path, '--output', out,
                         '--patch', 64, 64, '--stride', 64, 64, '--split-margin', 0)

    def test_prior_cli_no_gt_and_explicit_grid_provenance(self):
        out = self.f.root/'prior.json'
        result = self.run_cli('prior', '--catalog', self.f.path,
            '--scene-id', 'synthetic-a', '--config', self.config,
            '--max-pixels', 128*128, '--output', out)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['provenance']['gt_used'])
        self.assertEqual(result['pixel_scale'], [1., 1.])
        self.assertEqual(result['reference_type'], 'provided_native_lms_bandmean')
        self.assertEqual(json.loads(out.read_text()), result)

    def test_prior_limit_failure_leaves_no_json(self):
        out = self.f.root/'prior-failed.json'
        with self.assertRaisesRegex(ValueError, 'max-pixels'):
            self.run_cli('prior', '--catalog', self.f.path, '--scene-id', 'synthetic-a',
                         '--config', self.config, '--max-pixels', 100, '--output', out)
        self.assertFalse(out.exists())

    def test_prior_whole_acquisition_holdout_blocks_even_distant_region(self):
        self.f.spec['heldout_crops'] = [dict(acquisition_id='synthetic-acq-a',
                                           source_crop_native_pan_yxhw=[10000, 10000, 4, 4])]
        self.f.write()
        with self.assertRaisesRegex(ValueError, 'held-out'):
            prepare_prior(self.f.path, 'synthetic-a', self.config, 128*128)

    def test_prior_grid_mandatory_and_generated_lms_not_mislabelled(self):
        self.f.scene.pop('origin_phase')
        self.f.write()
        with self.assertRaisesRegex(ValueError, 'origin_phase'):
            prepare_prior(self.f.path, 'synthetic-a', self.config, 128*128)
        self.f.scene.update(origin_phase=[0., 0.], lms_mode=LMS_RECIPE)
        self.f.scene.pop('lms')
        self.f.write()
        result = prepare_prior(self.f.path, 'synthetic-a', self.config, 128*128)
        self.assertEqual(result['reference_type'], 'generated_native_ms_interp23tap_bandmean')
        self.assertIn('lms_recipe_source_sha256', result['provenance'])
        self.assertFalse(result['accepted'])

    def test_match_rr_cli_records_origin_but_does_not_admit(self):
        out = self.f.root/'matches.json'
        result = self.run_cli('match-rr', '--rr-h5', self.rr, '--row', 0,
            '--catalog', self.f.path, '--max-scene-pixels', 1024, '--output', out)
        self.assertEqual(result['status'], 'EXACT_GT_ONLY_CANDIDATE')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['candidates'][0]['candidate_native_pan_crop_yxhw'], [36, 68, 28, 32])
        self.assertEqual(result['candidates'][0]['native_ms_crop_yxhw'], [5, 9, 7, 8])
        self.assertTrue(result['all_catalog_scenes_checked'])
        self.assertEqual(len(result['rr_source']['sha256']), 64)
        self.assertIn('PHASE_UNVERIFIED', result['candidates'][0]['candidate_pan_mapping_status'])

    def test_match_rr_across_duplicate_scenes_remains_ambiguous(self):
        duplicate = copy.deepcopy(self.f.scene)
        duplicate['source_scene_id'] = 'synthetic-second'
        duplicate['acquisition_id'] = 'synthetic-acq-b'
        self.f.spec['scenes'].append(duplicate)
        self.f.write()
        result = match_rr(self.rr, 'gt', 0, self.f.path, 1024)
        self.assertEqual(result['status'], 'AMBIGUOUS')
        self.assertEqual(len(result['scenes']), 2)
        self.assertEqual(len(result['candidates']), 2)
        self.assertFalse(result['accepted'])

    def test_match_rr_size_limit_bad_row_and_catalog_fail(self):
        with self.assertRaisesRegex(ValueError, 'max-scene-pixels'):
            match_rr(self.rr, 'gt', 0, self.f.path, 100)
        with self.assertRaises(ValueError):
            match_rr(self.rr, 'gt', 9, self.f.path, 1024)
        self.f.spec.pop('heldout_crops')
        self.f.write()
        with self.assertRaisesRegex(ValueError, 'heldout_crops'):
            match_rr(self.rr, 'gt', 0, self.f.path, 1024)


if __name__ == '__main__':
    unittest.main()
