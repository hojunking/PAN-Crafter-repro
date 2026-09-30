"""Synthetic evidence only: none of these fixtures are uploadable results."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reporting_bridge import rb_b01_collect as c


def synthetic_grid():
    points = [(0., None)] + [(r, a) for r in (.25, .5, 1., 2., 3., 4.) for a in range(0, 360, 45)]
    return dict(coordinate_order=['dy', 'dx'], shifts=[dict(id=f'D{i:03d}', radius_hr=r,
        angle_degrees=a, dy=0. if a is None else r * math.sin(math.radians(a)),
        dx=0. if a is None else r * math.cos(math.radians(a))) for i, (r, a) in enumerate(points)])


def synthetic_selection(alias=False):
    records = [dict(update=u, val_ergas=1. if u == (50000 if alias else 49490) else 2.) for u in c.VAL_GRID]
    return dict(complete=True, actual_updates=50000, test_FR_sweep=False, validation_records=records,
        primary=dict(selection_id='EXACT_50000', update=50000, directory='exact50000', checkpoint_sha256='a' * 64),
        secondary=dict(selection_id='RR_VAL_ERGAS_MIN', update=50000 if alias else 49490,
                       directory='val_selected', checkpoint_sha256=('a' if alias else 'b') * 64,
                       alias_of='EXACT_50000' if alias else None))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_stress(wd):
    grid = synthetic_grid()
    native = dict(context=dict(run_id='SYNTHETIC_ONLY'), selections={'EXACT_50000': {'checkpoint_sha256': 'a' * 64}})
    points, rows = [], []
    for shift in grid['shifts']:
        point = dict(shift, n_scenes=20, n_failures=0)
        for key in c.POINT_METRICS:
            point[key] = 1.
            point[key + '_scene_std'] = 0.
        for key in c.METRICS:
            point['delta_from_zero_' + key] = 0.
        points.append(point)
        for scene in range(20):
            rows.append(dict(scene_index=scene, shift_id=shift['id'], status='ok',
                             invalid_roi_sampling=False, **{key: 1. for key in c.POINT_METRICS}))
    folder = wd / 'stress/A_ON'
    folder.mkdir(parents=True)
    with (folder / 'per_scene.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(folder / 'curve_summary.json', dict(points=points))
    report = dict(schema='PANDA_RB02_CURVE_v1', complete=True,
        identity=dict(context=native['context'], mode='A_ON', source_selection='EXACT_50000',
                      update=50000, checkpoint_sha256='a' * 64, grid_sha256=c.object_sha(grid)),
        n_scenes=20, n_shifts=49, n_observations=980, n_independent_students=1,
        n_numerical_failures=0, n_invalid_geometry=0, curve=points,
        file_hashes={str(p.relative_to(wd)): c.file_sha(p) for p in folder.iterdir()})
    report['payload_sha256'] = c.object_sha(report)
    return report, native, grid


class CollectionTests(unittest.TestCase):
    def test_original_payload_hash_convention(self):
        value = {'z': 0., 'a': None}
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(c.object_sha(value), expected)

    def test_seal_checks_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'raw.npy').write_bytes(b'SYNTHETIC')
            report = dict(complete=True, file_hashes={'raw.npy': c.file_sha(root / 'raw.npy')})
            report['payload_sha256'] = c.object_sha(report)
            c.validate_seal(report, root)
            (root / 'raw.npy').write_bytes(b'TAMPER')
            with self.assertRaisesRegex(c.EvidenceError, 'bytes changed'):
                c.validate_seal(report, root)

    def test_payload_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            report = dict(complete=True, file_hashes={'a': 'a' * 64}, metric=1.)
            report['payload_sha256'] = c.object_sha(report)
            report['metric'] = 2.
            with self.assertRaisesRegex(c.EvidenceError, 'modified'):
                c.validate_seal(report, temp)

    def test_path_escape_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'; root.mkdir()
            (root / 'escape').symlink_to(Path(temp), target_is_directory=True)
            for path in ('../outside', '/etc/passwd', 'escape/file'):
                with self.assertRaises(c.EvidenceError):
                    c.safe_path(root, path)

    def test_selection_validation_only_and_earliest_tie(self):
        manifest = synthetic_selection()
        self.assertEqual(c.selected_checkpoints(manifest)[1]['update'], 49490)
        manifest['validation_records'][0]['val_ergas'] = 1.
        with self.assertRaisesRegex(c.EvidenceError, 'Selection differs'):
            c.selected_checkpoints(manifest)

    def test_selection_alias_explicit(self):
        manifest = synthetic_selection(alias=True)
        self.assertEqual(c.selected_checkpoints(manifest)[1]['alias_of'], 'EXACT_50000')
        manifest['secondary']['alias_of'] = None
        with self.assertRaisesRegex(c.EvidenceError, 'alias'):
            c.selected_checkpoints(manifest)

    def test_test_sweep_and_nonfinite_block(self):
        for mode in ('test', 'nan', 'grid'):
            manifest = synthetic_selection()
            if mode == 'test': manifest['test_FR_sweep'] = True
            if mode == 'nan': manifest['validation_records'][0]['val_ergas'] = float('nan')
            if mode == 'grid': manifest['validation_records'].pop()
            with self.assertRaises(c.EvidenceError):
                c.selected_checkpoints(manifest)

    def test_grid_alias_conflict(self):
        grid = synthetic_grid(); c.validate_grid(grid)
        grid['shifts'][1]['angle_deg'] = 90
        with self.assertRaisesRegex(c.EvidenceError, 'angle'):
            c.validate_grid(grid)

    def test_stress_csv_mean_and_full_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); report, native, grid = synthetic_stress(wd)
            c.validate_stress(report, wd, native, 'A_ON', grid)
            report['curve'][0]['ergas'] = 2.
            write_json(wd / 'stress/A_ON/curve_summary.json', dict(points=report['curve']))
            with self.assertRaisesRegex(c.EvidenceError, 'mean mismatch'):
                c.validate_stress(report, wd, native, 'A_ON', grid)

    def test_stress_selected_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); report, native, grid = synthetic_stress(wd)
            report['identity']['source_selection'] = 'RR_VAL_ERGAS_MIN'
            with self.assertRaisesRegex(c.EvidenceError, 'exact native50K'):
                c.validate_stress(report, wd, native, 'A_ON', grid)

    def test_stress_failures_not_cleaned(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); report, native, grid = synthetic_stress(wd)
            csv_path = wd / 'stress/A_ON/per_scene.csv'
            with csv_path.open(newline='') as stream: rows = list(csv.DictReader(stream))
            rows[0].update(status='nonfinite_metric', ergas='', invalid_roi_sampling='True')
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            point = report['curve'][0]
            point.update(n_failures=1, ergas=None, ergas_scene_std=None, delta_from_zero_ergas=None)
            for p in report['curve']: p['delta_from_zero_ergas'] = None
            report.update(n_numerical_failures=1, n_invalid_geometry=1)
            write_json(wd / 'stress/A_ON/curve_summary.json', dict(points=report['curve']))
            c.validate_stress(report, wd, native, 'A_ON', grid)
            self.assertIsNone(report['curve'][0]['ergas'])
            self.assertEqual(report['n_invalid_geometry'], 1)

    def test_frozen_validator_only_function_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = 'raise RuntimeError("NO TOP LEVEL EXECUTION")\n\ndef validate_artifacts(report, directory):\n    return report\n'
            files = {'panda_rb/evaluation.py': hashlib.sha256(source.encode()).hexdigest()}
            release_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            release = root / c.CAMPAIGN_DIR / 'releases' / release_id
            (release / 'panda_rb').mkdir(parents=True)
            (release / 'panda_rb/evaluation.py').write_text(source)
            write_json(release / 'runtime_release.json', dict(files=files, release_sha256=release_id))
            _, validator, receipt = c.frozen_validator(root, dict(release_sha256=release_id))
            self.assertEqual(validator({'synthetic': True}, root), {'synthetic': True})
            self.assertIn('isolated AST', receipt['validator'])

    def test_import_does_not_import_torch_or_experiment(self):
        import subprocess, sys
        script = 'import reporting_bridge.rb_b01_collect,sys; assert not any(x == "torch" or x.startswith("panda_rb") for x in sys.modules)'
        subprocess.run([sys.executable, '-B', '-c', script], check=True)

    def test_missing_source_status_is_not_unrun_claim(self):
        status = c._status(dict(run_id='SYNTHETIC'))
        self.assertEqual(status['native_status'], 'FILE_NOT_FOUND')
        self.assertIsNone(status['actual_updates'])
        self.assertEqual(status['native_selectors'], 0)
        self.assertEqual(status['curve_status']['A_ON']['shifts'], 0)

    def test_regenerated_aggregate_report_does_not_change_observation_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = dict(run_id='SYNTHETIC', case_id='QFULL', server='s1', repeat=1, seed=9281101)
            wd = root / c.CAMPAIGN_DIR / 'RB01/s1/R1/QFULL'
            selections = {s: dict(checkpoint_sha256='a' * 64, alias_of=None) for s in c.SELECTORS}
            report = dict(context={}, selections=selections, payload_sha256='b' * 64,
                          completed_at_utc='FIXED_SOURCE_DATE', device='cpu')
            write_json(wd / 'native/metrics.json', report)
            write_json(root / c.CAMPAIGN_DIR / 'control/s1/deployment.json', dict(server='s1'))
            write_json(root / c.PLAN_DIR / 'planning/shift_grid_v1.json', {})
            original = dict(schema='PANDA_RB_B01_LOCAL_REPORT_v1', servers=['s1'], created_at_utc='FIRST',
                rows=[dict(run_id='SYNTHETIC', selection_id=s, checkpoint_sha256='a' * 64,
                           alias_of=None, metrics={}) for s in c.SELECTORS])
            aggregate = root / 'aggregate.json'; write_json(aggregate, original)
            with mock.patch.object(c, 'registry', return_value=[case]), \
                    mock.patch.object(c, 'validate_grid'), \
                    mock.patch.object(c, 'frozen_validator', return_value=(root, lambda *a: None, {'validator': 'FIXED'})), \
                    mock.patch.object(c, 'validate_seal'), \
                    mock.patch.object(c, 'validate_native'), \
                    mock.patch.object(c, '_context_assets', side_effect=lambda *a: ({}, {}, {}, {}, {'selection_manifest_sha256': 'c' * 64})):
                first = c.collect_local(root, 's1', frozen_report_path=aggregate)
                original['created_at_utc'] = 'SECOND'; write_json(aggregate, original)
                second = c.collect_local(root, 's1', frozen_report_path=aggregate)
                third = c.collect_local(root, 's1')
            self.assertFalse(first['errors'])
            self.assertNotEqual(first['original_report_audit'], second['original_report_audit'])
            self.assertEqual(first['native'][0]['provenance'], second['native'][0]['provenance'])
            self.assertEqual(first['native'][0]['provenance'], third['native'][0]['provenance'])


if __name__ == '__main__':
    unittest.main()
