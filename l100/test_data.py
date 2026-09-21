import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from l100 import data as d
from l100.common import atomic_json, camp


def manifest():
    return dict(schema='G20_DATA_v1', server='s3', sensor='GF2', num_bands=4,
        max_pixel=1023, mtf_sensor='GF2', band_order=['B', 'G', 'R', 'NIR'],
        recipe=copy.deepcopy(d.RECIPE), augmentation=copy.deepcopy(d.AUGMENTATION),
        augmentation_sha256=d.canonical_sha(d.AUGMENTATION),
        source_provenance={key: 'DN' if key == 'units' else 'bound-test-evidence' for key in d.PROVENANCE_FIELDS},
        splits={key: dict(count=count) for key, count in dict(train=19809, val=2201, rr=20, fr=20).items()})


class DataContractTests(unittest.TestCase):
    def test_exact_gf2_count_contract_without_opening_dataset(self):
        value = manifest()
        self.assertIs(d.validate_manifest(value, 's3'), value)

    def test_wrong_counts_including_wv3_rejected(self):
        for split, count in (('train', 9714), ('val', 1080), ('rr', 19), ('fr', 19)):
            value = manifest(); value['splits'][split]['count'] = count
            with self.assertRaisesRegex(ValueError, 'expected'):
                d.validate_manifest(value, 's3')

    def test_missing_and_extra_splits_rejected(self):
        value = manifest(); value['splits']['extra'] = dict(count=1)
        with self.assertRaisesRegex(ValueError, 'Exactly'):
            d.validate_manifest(value, 's3')
        value = manifest(); value['splits'].pop('val')
        with self.assertRaisesRegex(ValueError, 'Exactly'):
            d.validate_manifest(value, 's3')

    def test_owner_sensor_band_and_dn_contracts(self):
        for key, value in (('server', 's4'), ('sensor', 'QB'), ('max_pixel', 2047), ('num_bands', 8),
                           ('mtf_sensor', 'WV3')):
            data = manifest(); data[key] = value
            with self.assertRaises(ValueError):
                d.validate_manifest(data, 's3')

    def test_band_order_cannot_silently_permute(self):
        value = manifest(); value['band_order'] = ['R', 'G', 'B', 'NIR']
        with self.assertRaisesRegex(ValueError, 'ordering'):
            d.validate_manifest(value, 's3')

    def test_actual_augmentation_must_match_its_claimed_hash(self):
        value = manifest(); value['augmentation'] = {'hflip': False}
        with self.assertRaisesRegex(ValueError, 'augmentation'):
            d.validate_manifest(value, 's3')

    def test_lp_recipe_and_provenance_cannot_be_missing(self):
        value = manifest(); value['recipe'] = {'sigma': 2.}
        with self.assertRaises(ValueError):
            d.validate_manifest(value, 's3')
        value = manifest(); value['source_provenance'].pop('units')
        with self.assertRaises(ValueError):
            d.validate_manifest(value, 's3')

    def test_explicit_missing_manifest_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Explicit local data manifest'):
                d.prepare_data(Path(directory), 's3', manifest_path='does-not-exist.json')

    def test_explicit_manifest_must_not_silently_replace_bound_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(camp(root, 's3') / 'dataset_manifest.json', manifest())
            replacement = manifest(); replacement['splits']['train']['sha256'] = 'a' * 64
            atomic_json(root / 'replacement.json', replacement)
            with self.assertRaisesRegex(ValueError, 'already-bound'):
                d.prepare_data(root, 's3', manifest_path='replacement.json')

    def test_reused_data_must_match_pinned_catalog_not_just_own_checksums(self):
        value = manifest()
        bindings = {split: SimpleNamespace(path='/official/gf2/' + split + '.h5',
            sha256=str(index) * 64, source_identity='official-' + split)
            for index, split in enumerate(d.SPLITS)}
        catalog = SimpleNamespace(sensor='GF2', num_bands=4, max_dn=1023,
                                  split=lambda split: bindings[split])
        for split, binding in bindings.items():
            value['splits'][split].update(dataroot=binding.path, sha256=binding.sha256,
                                         source_identity=binding.source_identity)
        d.validate_source_bindings(value, catalog)
        for key, wrong in (('dataroot', '/arbitrary/self-consistent.h5'),
                           ('sha256', 'f' * 64), ('source_identity', 'unknown-provenance')):
            changed = copy.deepcopy(value); changed['splits']['train'][key] = wrong
            with self.assertRaisesRegex(ValueError, 'pinned GF2 catalog'):
                d.validate_source_bindings(changed, catalog)


if __name__ == '__main__':
    unittest.main()
