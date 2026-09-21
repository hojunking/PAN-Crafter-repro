import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ablr2.postrun import GRID_STEPS, select_records, validate_record, selection_report, find_config
from ablr2.plan import CASES, build_config, CAMPAIGN_ID
from ablr2.common import immutable_json
from ablr2.test_evaluation import record


class PostrunTests(unittest.TestCase):
    def test_json_published_config_preserves_scientific_numeric_types(self):
        case = CASES[0]
        cfg = build_config(case)
        with TemporaryDirectory() as folder:
            path = Path(folder) / cfg['work_dir'] / 'meta/config.resolved.yaml'
            immutable_json(path, cfg)
            loaded = find_config(case.run_id, folder)
        self.assertEqual(loaded, cfg)
        self.assertIsInstance(loaded['eps'], float)
        self.assertEqual(loaded['eps'], 1e-8)

    def test_sensor_correct_q_and_complete_validation(self):
        for sensor in ('WV3', 'QB'):
            validate_record(record(sensor), sensor)
        row = record('QB')
        row['rr']['q8'] = row['rr'].pop('q4')
        with self.assertRaises(ValueError): validate_record(row, 'QB')

    def test_val_tie_lower_step_not_hqnr(self):
        rows = [record('QB', n) for n in GRID_STEPS]
        rows[2]['val_ergas'] = .5
        rows[3]['val_ergas'] = .5
        result = select_records(rows, 'QB')
        self.assertEqual(result['rr_val_selected']['update'], GRID_STEPS[2])
        self.assertEqual(result['exact50k']['update'], 50000)

    def test_partial_duplicate_and_diagnostic_grid_rejected(self):
        rows = [record('QB', n) for n in GRID_STEPS]
        for invalid in (rows[:-1], rows[:-1] + [rows[0]], rows + [record('QB', 1000)]):
            with self.assertRaises(ValueError): select_records(invalid, 'QB')

    def test_false_scene_aggregation_rejected(self):
        row = record('QB')
        row['rr']['ergas'] += .1
        with self.assertRaises(ValueError): validate_record(row, 'QB')

    def test_primary_rules_and_identity_mode_are_explicit(self):
        case = next(c for c in CASES if c.case_id == 'C00')
        cfg = build_config(case)
        grid = dict(campaign_id=CAMPAIGN_ID, config_sha256='x', data_sha256='y', source_identity={}, reference_sha256=None)
        row = record(case.sensor)
        val = selection_report(cfg, grid, 'rr_val_selected', row)
        raw = selection_report(cfg, grid, 'raw_max', row)
        self.assertTrue(val['primary'])
        self.assertFalse(raw['primary'])
        self.assertTrue(raw['test_aware'])
        self.assertFalse(val['independent_test'])
        self.assertEqual(val['eval_mode'], 'IDENTITY')


if __name__ == '__main__': unittest.main()
