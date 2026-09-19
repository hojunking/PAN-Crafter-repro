"""Synthetic source-catalog binding tests; no real datasets or GPU needed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qg40.bootstrap import default_sensor_spec
from qg40.data import PROVENANCE_FIELDS


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'checkout'
        self.catalog = self.fixture(self.root)

    def fixture(self, root, sensor='QB', data_dir=None):
        (root / 'qg40').mkdir(parents=True)
        data = root / 'data'
        if data_dir is None:
            data.mkdir()
        else:
            data.symlink_to(data_dir, target_is_directory=True)
        sources = {}
        for split in ('train', 'val', 'rr', 'fr', 'raw_train', 'raw_val'):
            path = data / f'{split}.h5'
            path.write_bytes(f'known synthetic {split}'.encode())
            sources[split] = dict(path=f'data/{split}.h5',
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  source_identity=f'fixture-{sensor}-{split}')
        proof = {key: f'explicit fixture {key}' for key in PROVENANCE_FIELDS}
        proof['units'] = 'DN'
        if sensor == 'QB':
            proof['msfix_recipe'] = 'declared fixture recipe, not an audit'
            for split in ('train', 'val'):
                raw = sources[f'raw_{split}']
                proof[f'raw_{split}_path'] = raw['path']
                proof[f'raw_{split}_sha256'] = raw['sha256']
        catalog = dict(schema='QG40_SENSOR_SOURCES_v1', sensors={sensor: dict(
            band_order=['B', 'G', 'R', 'NIR'],
            splits={key: sources[key] for key in ('train', 'val', 'rr', 'fr')},
            source_provenance=proof)})
        self.write(catalog, root)
        return catalog

    def write(self, catalog, root=None):
        ((root or self.root) / 'qg40/sensor_sources.json').write_text(json.dumps(catalog))

    def test_relocatable_catalog_and_shared_symlink(self):
        other = self.base / 'other_checkout'
        self.fixture(other)
        first, second = (default_sensor_spec(root, 'QB') for root in (self.root, other))
        self.assertTrue(first.is_bound)
        self.assertEqual(first.band_order, ('B', 'G', 'R', 'NIR'))
        self.assertEqual(first.split('train').sha256, second.split('train').sha256)
        self.assertNotEqual(first.split('train').path, second.split('train').path)
        shared = self.base / 'shared'
        shared.mkdir()
        linked = self.base / 'linked_checkout'
        self.fixture(linked, data_dir=shared)
        spec = default_sensor_spec(linked, 'QB')
        self.assertEqual(spec.split('train').path, str(shared / 'train.h5'))
        self.assertEqual(dict(spec.source_provenance)['raw_val_path'], str(shared / 'raw_val.h5'))

    def test_gf2_needs_no_qb_raw_provenance(self):
        root = self.base / 'gf2'
        self.fixture(root, sensor='GF2')
        spec = default_sensor_spec(root, 'GF2')
        self.assertEqual((spec.num_bands, spec.max_dn), (4, 1023))
        self.assertNotIn('raw_train_path', dict(spec.source_provenance))

    def test_changed_source_and_raw_hash_rejected(self):
        for filename in ('train.h5', 'raw_train.h5', 'raw_val.h5'):
            with self.subTest(filename=filename):
                path = self.root / 'data' / filename
                original = path.read_bytes()
                path.write_bytes(original + b' changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    default_sensor_spec(self.root, 'QB')
                path.write_bytes(original)

    def test_missing_source_and_missing_catalog_rejected(self):
        path = self.root / 'data/train.h5'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            default_sensor_spec(self.root, 'QB')
        with self.assertRaisesRegex(ValueError, 'catalog'):
            default_sensor_spec(self.base / 'absent', 'QB')

    def test_absolute_and_traversal_paths_rejected(self):
        for field in ('split', 'raw'):
            for value in (str(self.root / 'data/train.h5'), '../checkout/data/train.h5'):
                with self.subTest(field=field, value=value):
                    catalog = copy.deepcopy(self.catalog)
                    entry = catalog['sensors']['QB']
                    if field == 'split':
                        entry['splits']['train']['path'] = value
                    else:
                        entry['source_provenance']['raw_train_path'] = value
                    self.write(catalog)
                    with self.assertRaisesRegex(ValueError, 'relative'):
                        default_sensor_spec(self.root, 'QB')

    def test_required_provenance_and_units(self):
        for key in (*PROVENANCE_FIELDS, 'msfix_recipe', 'raw_train_path', 'raw_val_sha256'):
            with self.subTest(missing=key):
                catalog = copy.deepcopy(self.catalog)
                del catalog['sensors']['QB']['source_provenance'][key]
                self.write(catalog)
                with self.assertRaises(ValueError):
                    default_sensor_spec(self.root, 'QB')
        catalog = copy.deepcopy(self.catalog)
        catalog['sensors']['QB']['source_provenance']['units'] = 'normalized'
        self.write(catalog)
        with self.assertRaisesRegex(ValueError, 'units must be DN'):
            default_sensor_spec(self.root, 'QB')

    def test_catalog_schema_sensor_band_order_and_split_contract(self):
        mutations = [lambda c: c.update(schema='wrong'),
                     lambda c: c['sensors'].clear(),
                     lambda c: c['sensors']['QB'].update(sensor='GF2'),
                     lambda c: c['sensors']['QB'].update(band_order=['a', 'b', 'c', 'd']),
                     lambda c: c['sensors']['QB']['splits'].pop('fr'),
                     lambda c: c['sensors']['QB']['splits']['train'].update(sha256='x' * 64),
                     lambda c: c['sensors']['QB']['splits']['train'].update(source_identity=' ')]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                catalog = copy.deepcopy(self.catalog)
                mutation(catalog)
                self.write(catalog)
                with self.assertRaises(ValueError):
                    default_sensor_spec(self.root, 'QB')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            default_sensor_spec(self.root, 'WV3')

    def test_malformed_json_and_no_side_effects(self):
        path = self.root / 'qg40/sensor_sources.json'
        path.write_text('{broken')
        with self.assertRaisesRegex(ValueError, 'catalog'):
            default_sensor_spec(self.root, 'QB')
        self.write(self.catalog)
        snapshot = lambda: {str(p.relative_to(self.root)): p.read_bytes()
                            for p in self.root.rglob('*') if p.is_file()}
        before = snapshot()
        with patch('qg40.data.prepare_data', side_effect=AssertionError('no preparation')), \
                patch('qg40.data._scan_source', side_effect=AssertionError('no scan')), \
                patch('qg40.data._create_lp_cache', side_effect=AssertionError('no cache')):
            spec = default_sensor_spec(self.root, 'QB')
        self.assertEqual(snapshot(), before)
        self.assertFalse((self.root / 'work_dir').exists())
        self.assertNotIn('status', spec.to_dict())


if __name__ == '__main__':
    unittest.main()
