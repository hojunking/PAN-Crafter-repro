"""CPU/stdlib-only presentation regression tests; no Sheet, model or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from reporting_extra.sensor_sheet import (TRAIN_TIME_SCOPE, augment_notes,
                                          display_formats, merge_notes, metadata_values)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wd = Path(self.tmp.name) / 'G20_fixture_19990101'
        (self.wd / 'meta').mkdir(parents=True)
        self.cfg = {'trainer': 'g20', 'seed': 82001, 'work_dir': str(self.wd),
                    'model_args': {'hidden_size': 104, 'depth': [1, 2, 2]},
                    'g20': {'run_id': self.wd.name, 'input_layout': 'PLH'}}
        self.case = SimpleNamespace(run_id=self.wd.name, seed=82001, width=104,
                                    depth=(1, 2, 2), input_layout='PLH')
        self.training = {'training_complete': True, 'updated_at_utc': '2026-09-20T15:01:00Z',
                         'training_seconds': 3612.123456789}
        self.postrun = {'completed_at_utc': '2026-09-24T16:00:00Z'}

    def start_manifest(self, **changes):
        data = {'run_id': self.wd.name, 'config_sha256': hashlib.sha256(json.dumps(
            self.cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'started_at_utc': '2026-09-20T13:59:00Z'}
        data.update(changes)
        (self.wd / 'meta/training_start_manifest.json').write_text(json.dumps(data))

    def values(self):
        return metadata_values(self.wd, self.cfg, self.case, self.training, self.postrun, 'EXACT50K')

    def test_actual_end_date_in_kst_and_full_precision(self):
        self.start_manifest()
        originals = copy.deepcopy((self.cfg, vars(self.case), self.training, self.postrun))
        result = self.values()
        self.assertEqual(result['Date'], '2026-09-21')
        self.assertEqual(result['Model'], 'W104 D122')
        self.assertEqual(result['Seed'], 82001)
        self.assertEqual(result['Input'], 'PLH')
        self.assertEqual(result['Selection'], 'EXACT50K')
        self.assertEqual(result['Train(h)'], 3612.123456789 / 3600)
        self.assertEqual(result['Wall(h)'], 62 / 60)
        self.assertEqual(result['Train time scope'], TRAIN_TIME_SCOPE)
        self.assertEqual(result['G20 Date timezone'], 'Asia/Seoul')
        self.assertEqual(result['G20 completed UTC'], '2026-09-20T15:01:00+00:00')
        self.assertEqual(result['G20 official completed UTC'], '2026-09-24T16:00:00+00:00')
        self.assertEqual((self.cfg, vars(self.case), self.training, self.postrun), originals)

    def test_kst_midnight_boundary_not_utc_date(self):
        self.start_manifest()
        self.training['updated_at_utc'] = '2026-09-20T14:59:59Z'
        self.assertEqual(self.values()['Date'], '2026-09-20')
        self.training['updated_at_utc'] = '2026-09-20T15:00:00Z'
        self.assertEqual(self.values()['Date'], '2026-09-21')

    def test_missing_metadata_never_uses_postrun_or_run_name_as_training_date(self):
        self.training = {}
        result = self.values()
        for key in ('Date', 'Wall(h)', 'Train(h)', 'G20 started UTC', 'G20 completed UTC'):
            self.assertEqual(result[key], '', key)
        self.assertTrue(result['G20 official completed UTC'])

    def test_incomplete_ignores_status_update_falls_back_to_start(self):
        self.start_manifest(started_at_utc='2026-09-20T16:00:00Z')
        self.training.update(training_complete=False, updated_at_utc='2026-09-25T12:00:00Z')
        result = self.values()
        self.assertEqual(result['Date'], '2026-09-21')
        self.assertEqual(result['G20 completed UTC'], '')
        self.assertEqual(result['Wall(h)'], '')

    def test_recovered_status_is_not_a_finish_time(self):
        self.start_manifest()
        self.training['recovered_from_exact50k'] = True
        result = self.values()
        self.assertEqual(result['Date'], '2026-09-20')
        self.assertEqual(result['G20 completed UTC'], '')
        (self.wd / 'meta/finished_at.txt').write_text('2026-09-21T01:00:00+09:00\n')
        self.assertEqual(self.values()['Date'], '2026-09-21')
        self.assertEqual(self.values()['G20 completed UTC'], '2026-09-20T16:00:00+00:00')

    def test_legacy_offset_fallback_and_primary_precedence(self):
        (self.wd / 'meta/started_at.txt').write_text('2026-09-20T23:00:00+09:00\n')
        (self.wd / 'meta/finished_at.txt').write_text('2026-09-21T02:00:00+09:00\n')
        self.training = {}
        result = self.values()
        self.assertEqual(result['Wall(h)'], 3)
        self.assertEqual(result['Date'], '2026-09-21')
        self.start_manifest(started_at_utc='2026-09-20T13:00:00Z')
        self.training = {'training_complete': True, 'updated_at_utc': '2026-09-20T14:00:00Z'}
        self.assertEqual(self.values()['Wall(h)'], 1)
        self.assertEqual(self.values()['Date'], '2026-09-20')

    def test_rejects_reverse_time_and_ambiguous_legacy_timezone(self):
        self.start_manifest(started_at_utc='2026-09-20T16:00:00Z')
        with self.assertRaisesRegex(ValueError, 'precedes'):
            self.values()
        self.training = {}
        (self.wd / 'meta/finished_at.txt').write_text('2026-09-21T02:00:00')
        with self.assertRaisesRegex(ValueError, 'timezone'):
            self.values()

    def test_naive_utc_fields_are_explicitly_utc(self):
        self.start_manifest(started_at_utc='2026-09-20T13:59:00')
        self.training['updated_at_utc'] = '2026-09-20T15:01:00'
        self.assertEqual(self.values()['Date'], '2026-09-21')

    def test_start_manifest_identity_cannot_be_bypassed_with_legacy_fallback(self):
        (self.wd / 'meta/started_at.txt').write_text('2026-09-20T20:00:00+09:00')
        for changes in ({'run_id': 'wrong'}, {'config_sha256': 'wrong'}, {'config_sha256': None}):
            with self.subTest(changes=changes):
                self.start_manifest(**changes)
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    self.values()

    def test_case_config_fields_must_agree(self):
        for field, bad in (('seed', 82002), ('width', 112), ('depth', (1, 2, 3)),
                           ('input_layout', 'P0'), ('run_id', 'other-run')):
            original = getattr(self.case, field)
            with self.subTest(field=field):
                setattr(self.case, field, bad)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.values()
                setattr(self.case, field, original)

    def test_config_fallbacks_and_missing_fields_remain_blank(self):
        self.case = {}
        result = self.values()
        self.assertEqual((result['Seed'], result['Model'], result['Input']), (82001, 'W104 D122', 'PLH'))
        self.cfg = {'trainer': 'qg40'}
        result = self.values()
        self.assertEqual((result['Seed'], result['Model'], result['Input']), ('', '', ''))
        self.assertIn('QG40 Date timezone', result)

    def test_invalid_metadata_numbers_are_not_silently_truncated(self):
        for field, bad in (('seed', 82001.1), ('width', True), ('depth', [1, 2.5, 3]),
                           ('depth', [1, None, 3])):
            original = getattr(self.case, field)
            with self.subTest(field=field):
                setattr(self.case, field, bad)
                with self.assertRaises(ValueError):
                    self.values()
                setattr(self.case, field, original)
        for seconds in (-1, float('nan'), float('inf'), True):
            self.training['training_seconds'] = seconds
            with self.assertRaises(ValueError):
                self.values()


class NotesAndFormatsTests(unittest.TestCase):
    def test_merge_notes_preserves_distinct_prior_text_and_is_idempotent(self):
        incoming = '[Run summary: W104 D122] Official generated note.'
        old = '  Human note\nstrict FAILED; ACCEPTED_WITH_METRIC_EXCEPTION receipt=0123.  '
        merged = merge_notes(incoming, old)
        self.assertEqual(merged, incoming + '\n[Previous Sheet note] ' + old)
        self.assertEqual(merge_notes(incoming, merged), merged)
        self.assertEqual(merge_notes(merged, old), merged)
        self.assertEqual(merge_notes(merged, merged), merged)

    def test_merge_notes_blank_same_and_containment(self):
        for blank in ('', '  \n', None):
            self.assertEqual(merge_notes('new', blank), 'new')
            self.assertEqual(merge_notes(blank, 'old'), 'old')
            self.assertEqual(merge_notes(blank, blank), '')
        self.assertEqual(merge_notes('same', 'same'), 'same')
        self.assertEqual(merge_notes('new + old', 'old'), 'new + old')
        self.assertEqual(merge_notes('new', 'old + new'), 'old + new')
        with self.assertRaisesRegex(ValueError, 'text'):
            merge_notes('new', 123)

    def test_notes_preserve_all_exception_text_and_are_idempotent(self):
        values = {'Model': 'W104 D122', 'Seed': 0, 'Input': 'PLH', 'Selection': 'EXACT50K'}
        old = 'R0 strict FAILED; ACCEPTED_WITH_METRIC_EXCEPTION; receipt=abcdef.\nHuman note.'
        result = augment_notes(values, old)
        self.assertIn('W104 D122; seed=0; input=PLH; selection=EXACT50K', result)
        self.assertTrue(result.endswith(old))
        self.assertEqual(augment_notes(values, result), result)
        self.assertEqual(augment_notes({}, old), old)

    def test_display_formats_cover_costs_and_do_not_change_numbers(self):
        values = {'HQNR↑': .9561123456789, 'RAW_MAX ERGAS↓': .6361978992881856,
                  'Params(M)': 3.5283737, 'FLOPs(G)': 85.123456, 'Infer(ms)': 7.981234,
                  'Mem(MB)': 41.98123, 'Train(h)': 1.012345, 'Wall(h)': 3.12345,
                  'Seed': 82001, 'G20 Student seed': 92001, 'Exact50K step': 50000,
                  'Date': '2026-09-21', 'G20 started UTC': '2026-09-20T14:00:00+00:00',
                  'G20 Date timezone': 'Asia/Seoul', 'RAW_MAX signed Ds signed_mean': -.0632198,
                  'checkpoint SHA256': '00000001'}
        before = copy.deepcopy(values)
        labels = {label: index for index, label in enumerate(values, 1)}
        formats = display_formats(labels, 7, {'HQNR↑', 'ERGAS↓'}, ['RAW_MAX', 'Exact50K', 'RAW_MAX'])
        observed = {item['range']: item['format']['numberFormat'] for item in formats}
        patterns = ['0.0000', '0.0000', '0.0000', '0.0', '0.00', '0.0', '0.00', '0.00',
                    '0', '0', '0', '@', '@', '@', '0.0000']
        for letter, pattern in zip('ABCDEFGHIJKLMNO', patterns):
            self.assertEqual(observed[letter + '7']['pattern'], pattern)
            self.assertEqual(observed[letter + '7']['type'], 'TEXT' if pattern == '@' else 'NUMBER')
        self.assertEqual(len(formats), len(observed))
        self.assertNotIn('P7', observed)
        self.assertEqual(values, before)

    def test_duplicate_prefixes_and_same_column_have_one_format(self):
        formats = display_formats({'RAW_MAX HQNR↑': 27, 'HQNR↑': 27}, 23,
                                  {'HQNR↑'}, ['RAW_MAX', 'RAW_MAX'])
        self.assertEqual(formats, [{'range': 'AA23', 'format': {
            'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}}}])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            display_formats({'Date': 1, 'Seed': 1}, 4, set(), ())


if __name__ == '__main__':
    unittest.main()
