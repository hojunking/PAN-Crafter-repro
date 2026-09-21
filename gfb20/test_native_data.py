from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fh12.data import RECIPE, AUGMENTATION, canonical_sha, sha256_file
from g20.data import PROVENANCE_FIELDS, build_dataset
from qg40.plan import sensor_spec, SplitBinding
from gfb20.native_data import bind_lane_data, native_method_signature


def fixture(root, schema='QG40_DATA_v1'):
    root = Path(root)
    source = root / 'native.h5'
    lp = root / 'native_lp.h5'
    source.write_bytes(b'fixture source bytes; no samples regenerated')
    lp.write_bytes(b'fixture immutable native LP bytes')
    splits = {split: dict(dataroot=str(source), sha256=sha256_file(source),
        lpan_path=str(lp), lpan_sha256=sha256_file(lp), count=count,
        source_identity='GF2_original_'+split, sample_order_sha256='a'*64,
        lpan_canonical_sha256='b'*64, shapes={'native_fixture': [count]},
        prior_optional_statistics={'untouched': 7})
        for split, count in dict(train=19809, val=2201, rr=20, fr=20).items()}
    provenance = {key: 'fixture explicit original evidence' for key in PROVENANCE_FIELDS}
    provenance['units'] = 'DN'
    original = dict(schema=schema, server='s3', sensor='GF2', num_bands=4,
        max_pixel=1023, mtf_sensor='GF2', band_order=['B', 'G', 'R', 'NIR'],
        recipe=deepcopy(RECIPE), augmentation=deepcopy(AUGMENTATION),
        augmentation_sha256=canonical_sha(AUGMENTATION), source_provenance=provenance,
        splits=splits, opencv_version='original-recorded-version',
        unknown_historical_metadata={'keep_nested_values': [1, 2, 3]})
    catalog = replace(sensor_spec('GF2'), band_order=tuple(original['band_order']),
        splits=tuple((key, SplitBinding(value['dataroot'], value['sha256'], value['source_identity']))
                     for key, value in splits.items()))
    return original, catalog


class NativeLaneDataTests(unittest.TestCase):
    def test_qg40_wrapper_preserves_original_and_accepted_by_g20_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            original, catalog = fixture(folder)
            before = deepcopy(original)
            with patch('gfb20.native_data.default_sensor_spec', return_value=catalog) as bound:
                lane, provenance = bind_lane_data(original, 's3', folder)
            bound.assert_called_once_with(Path(folder), 'GF2')
            self.assertEqual(original, before)
            self.assertIsNot(lane['splits'], original['splits'])
            self.assertEqual(lane, dict(before, schema='G20_DATA_v1'))
            self.assertEqual(native_method_signature(lane), native_method_signature(original))
            self.assertEqual(provenance['original_manifest_sha256'], canonical_sha(original))
            self.assertEqual(provenance['lane_manifest_sha256'], canonical_sha(lane))
            self.assertNotEqual(provenance['original_manifest_sha256'], provenance['lane_manifest_sha256'])
            # Exercise the real builder's schema/recipe/file checks; the file
            # decoding constructor is mocked because these are byte fixtures.
            with patch('g20.data.G20Dataset') as constructor:
                build_dataset(lane, 'train', root=folder)
                constructor.assert_called_once()

    def test_g20_kept_identical_including_extra_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            original, catalog = fixture(folder, 'G20_DATA_v1')
            with patch('gfb20.native_data.default_sensor_spec', return_value=catalog):
                lane, receipt = bind_lane_data(original, 's3', folder)
            self.assertEqual(lane, original)
            self.assertEqual(receipt['original_manifest_sha256'], receipt['lane_manifest_sha256'])

    def test_relative_original_paths_resolve_without_rewriting_original(self):
        with tempfile.TemporaryDirectory() as folder:
            original, catalog = fixture(folder)
            for item in original['splits'].values():
                item['dataroot'], item['lpan_path'] = 'native.h5', 'native_lp.h5'
            before = deepcopy(original)
            with patch('gfb20.native_data.default_sensor_spec', return_value=catalog):
                lane, receipt = bind_lane_data(original, 's3', folder)
            self.assertEqual(original, before)
            self.assertEqual(len(receipt['path_resolutions']), 8)
            self.assertTrue(Path(lane['splits']['train']['dataroot']).is_absolute())

    def test_changed_lp_or_wrong_catalog_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            original, catalog = fixture(folder)
            original['splits']['train']['lpan_sha256'] = 'f'*64
            with patch('gfb20.native_data.default_sensor_spec', return_value=catalog):
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    bind_lane_data(original, 's3', folder)
            original, catalog = fixture(folder)
            wrong = replace(catalog, splits=tuple((key, replace(value, sha256='f'*64))
                                                  for key, value in catalog.splits))
            with patch('gfb20.native_data.default_sensor_spec', return_value=wrong):
                with self.assertRaisesRegex(ValueError, 'pinned GF2 catalog'):
                    bind_lane_data(original, 's3', folder)

    def test_foreign_lane_schema_or_sample_count_is_not_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            original, catalog = fixture(folder)
            for server in ('s1', 's2', 's4', 's5'):
                with self.assertRaises(ValueError):
                    bind_lane_data(original, server, folder)
            original['splits']['rr']['count'] = 19
            with self.assertRaisesRegex(ValueError, '20 GF2 source samples'):
                bind_lane_data(original, 's3', folder)
            original['schema'] = 'WV3_DATA'
            with self.assertRaises(ValueError):
                bind_lane_data(original, 's3', folder)


if __name__ == '__main__':
    unittest.main()
