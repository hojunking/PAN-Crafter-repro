import unittest
from ablr2.plan import CAMPAIGN_ID
from ablr2.upload import plan_upsert, metric_formats


class UploadTests(unittest.TestCase):
    def values(self):
        return {'Run': 'ABLR2_WV3_s1_R00_P01_C00_1_FRESH50', 'ABLR2 campaign': CAMPAIGN_ID,
                'ABLR2 sensor': 'WV3', 'ABLR2 run id': 'ABLR2_WV3_s1_R00_P01_C00_1_FRESH50',
                'HQNR↑': .963123456789, 'Q8↑': .856712345, 'ABLR2 upload status': 'READBACK_PENDING'}

    def test_append_does_not_modify_legacy_row_and_keeps_precision(self):
        table = [[], [], ['Run', 'HQNR↑'], ['legacy_run', '.97']]
        plan = plan_upsert(table, table[2], self.values())
        self.assertEqual(plan['row'], 5)
        self.assertEqual(plan['values']['HQNR↑'], .963123456789)
        self.assertFalse(any(e['range'].endswith('4') for e in plan['edits']))

    def test_same_run_cannot_alias_historical_namespace(self):
        values = self.values()
        with self.assertRaises(ValueError):
            plan_upsert([[], [], ['Run'], [values['Run']]], ['Run'], values)

    def test_existing_observation_cannot_be_replaced(self):
        values = self.values()
        headers = list(values)
        row = list(values.values())
        row[headers.index('HQNR↑')] = .99
        with self.assertRaises(ValueError): plan_upsert([[], [], headers, row], headers, values)

    def test_four_decimals_are_display_only(self):
        formats = metric_formats({'HQNR↑': 1, 'Q8↑': 2, 'Q4↑': 3, 'ABLR2 RR_VAL_SELECTED ERGAS↓': 4}, 4)
        self.assertEqual(len(formats), 4)
        self.assertTrue(all(f['format']['numberFormat']['pattern'] == '0.0000' for f in formats))


if __name__ == '__main__': unittest.main()
