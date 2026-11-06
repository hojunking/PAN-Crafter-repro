import copy
import json
import unittest
from directshift48.reporting import INDEX, HEADERS
from .publication import promote, verified_legacy_row, LEGACY


class PrecisionTests(unittest.TestCase):
    def fixture(self):
        row = [''] * len(HEADERS)
        row[INDEX['HQNR↑']] = .959
        row[INDEX['Infer(ms)']] = 10.049
        row[INDEX['U_peak_lr']] = .00011
        row[INDEX['Selected_step']] = 29000
        note = {'reporting': {'policy': LEGACY['w128'], 'decimal_places': 3,
                 'raw_metrics': {'HQNR↑': .958549}, 'cost_rr': {'Infer(ms)': 10.049051}},
                'checkpoint_retention': {'status': 'REMOVED_BY_USER_POLICY'}}
        row[INDEX['Original description / notes']] = json.dumps(note)
        return {'rows': [row], 'notes': [json.dumps(note)], 'view_hash': 'old'}

    def test_restores_fourth_digit_from_evidence_and_keeps_config(self):
        before = self.fixture(); saved = copy.deepcopy(before)
        value = promote(before, 'w128'); row = value['rows'][0]
        self.assertEqual(before, saved)
        self.assertEqual(row[INDEX['HQNR↑']], .9585)
        self.assertEqual(row[INDEX['Infer(ms)']], 10.0491)
        self.assertEqual(row[INDEX['U_peak_lr']], .00011)
        self.assertEqual(row[INDEX['Selected_step']], 29000)
        self.assertEqual(json.loads(value['notes'][0])['checkpoint_retention'],
                         {'status': 'REMOVED_BY_USER_POLICY'})
        legacy = verified_legacy_row(row, 'w128')
        self.assertEqual(legacy[INDEX['HQNR↑']], .959)

    def test_rejects_tampered_display_and_missing_raw_cost(self):
        row = promote(self.fixture(), 'w128')['rows'][0]
        row[INDEX['HQNR↑']] = .9586
        with self.assertRaises(ValueError): verified_legacy_row(row, 'w128')
        value = self.fixture()
        note = json.loads(value['notes'][0]); note['reporting']['cost_rr'] = {}
        value['rows'][0][INDEX['Original description / notes']] = json.dumps(note)
        with self.assertRaises(ValueError): promote(value, 'w128')


if __name__ == '__main__': unittest.main()
