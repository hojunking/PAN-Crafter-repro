"""Synthetic header-only discovery tests; no production assets touched."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from teacher48.prepare_inventory import inventory, MAX_STRING


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, name, *, native=False, pan_hw=64, gt=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = 'native/' if native else ''
        with h5py.File(path, 'w') as stream:
            stream.create_dataset(prefix + 'pan', shape=(2, 1, pan_hw, pan_hw), dtype='uint16')
            stream.create_dataset(prefix + 'ms', shape=(2, 4, pan_hw // 4, pan_hw // 4), dtype='float32')
            if gt:
                stream.create_dataset('gt', shape=(2, 4, pan_hw, pan_hw), dtype='float32')
        return path

    def test_rr_is_not_native_and_no_shape_based_scene_ids(self):
        self.file('train.h5')
        result = inventory([self.root])
        record = result['files'][0]
        self.assertEqual(record['classification'], 'RR_PATCH_ONLY')
        self.assertFalse(result['admission_granted'])
        self.assertFalse(record['lineage_metadata']['verified_source_identity'])
        self.assertEqual(record['lineage_metadata']['scene_fields'], [])
        self.assertNotIn('sha256', record)

    def test_native_structure_is_candidate_only(self):
        path = self.file('observation.h5', native=True, pan_hw=256, gt=False)
        with h5py.File(path, 'a') as stream:
            stream.attrs['scene_id'] = 'actual-metadata-string'
            stream.attrs['acquisition_id'] = 'acquisition-12'
            stream.create_dataset('splits/patch_index', shape=(100, 4), dtype='int32')
        r = inventory([path])['files'][0]
        self.assertEqual(r['classification'], 'NATIVE_SCENE_CANDIDATE')
        self.assertEqual(r['role_from_path_only'], 'UNKNOWN_ROLE')
        self.assertEqual(r['lineage_metadata']['scene_fields'], ['@scene_id'])
        self.assertIn('splits/patch_index', r['lineage_metadata']['crop_fields'])
        self.assertFalse(r['admission_granted'])

    def test_generic_ratio_does_not_prove_native_or_training(self):
        self.file('train_new.h5', pan_hw=256, gt=False)
        r = inventory([self.root])['files'][0]
        self.assertEqual(r['classification'], 'RATIO4_OBSERVATIONS_UNKNOWN_ROLE')
        self.assertEqual(r['role_from_path_only'], 'UNKNOWN_ROLE')

    def test_benchmark_name_overrides_geometry(self):
        self.file('full_examples_mat20/test_sensor.h5', native=True, pan_hw=512, gt=False)
        r = inventory([self.root])['files'][0]
        self.assertEqual(r['classification'], 'EVAL_NOT_TRAIN')
        self.assertEqual(r['geometry_class'], 'NATIVE_SCENE_CANDIDATE')

    def test_no_dataset_values_read_and_attributes_bounded(self):
        path = self.file('observations.h5')
        with h5py.File(path, 'a') as stream:
            stream.attrs['description'] = 'a' * (MAX_STRING * 3)
            stream.attrs['massive'] = np.arange(2000)
            stream.attrs['nan_value'] = float('nan')
            stream.create_dataset('metadata/source_scene_id', data=b'DO NOT READ')
        with patch.object(h5py.Dataset, '__getitem__', side_effect=AssertionError('dataset read')):
            result = inventory([path])
        self.assertEqual(result['errors'], [])
        r = result['files'][0]
        self.assertLess(len(r['attributes']['description']), MAX_STRING + 32)
        self.assertTrue(r['attributes']['massive']['omitted_large_attribute'])
        self.assertIn('metadata/source_scene_id', r['lineage_metadata']['scene_fields'])
        json.dumps(result, allow_nan=False)

    def test_symlink_loop_file_alias_hardlink_and_overlapping_roots(self):
        path = self.file('real/a.HDF5')
        os.symlink(self.root / 'real', self.root / 'directory_alias')
        os.symlink(self.root, self.root / 'real' / 'loop')
        os.symlink(path, self.root / 'file_alias.h5')
        os.link(path, self.root / 'hardlink.hdf')
        result = inventory([self.root, self.root / 'real'])
        self.assertEqual(result['summary']['file_count'], 1)
        self.assertEqual(result['errors'], [])
        self.assertGreaterEqual(len(result['files'][0]['discovered_paths']), 3)

    def test_missing_root_and_bad_h5_report_errors(self):
        (self.root / 'broken.h5').write_bytes(b'not hdf5')
        result = inventory([self.root, self.root / 'missing'])
        self.assertEqual(result['summary']['file_count'], 0)
        self.assertEqual(result['summary']['error_count'], 2)
        self.assertFalse(result['admission_granted'])

    def test_output_sorted_and_deterministic(self):
        self.file('z.h5')
        self.file('a.h5')
        first = inventory([self.root])
        self.assertEqual(first, inventory([self.root]))
        self.assertEqual([Path(r['path']).name for r in first['files']], ['a.h5', 'z.h5'])

    def test_empty_root_list_rejected(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            inventory([])


if __name__ == '__main__':
    unittest.main()
