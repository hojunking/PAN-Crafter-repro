"""Request-only layout tests; no network or campaign imports."""
import copy
import unittest

from reporting_extra.sensor_layout import (COMMON_HEADERS, initialize_requests,
                                         migrate_front_requests, row_formats)


def apply_requests(headers, requests):
    """Small Sheets column model that carries data/formula sentinels with cells."""
    columns = [{'header': h, 'value': f'value-{i}', 'formula': f'=source_{i}'}
               for i, h in enumerate(headers)]
    originals = copy.deepcopy(columns)
    for request in requests:
        if 'insertDimension' in request:
            index = request['insertDimension']['range']['startIndex']
            columns.insert(index, {'header': '', 'value': None, 'formula': None})
        elif 'updateCells' in request:
            update = request['updateCells']
            assert update['fields'] == 'userEnteredValue'
            index = update['start']['columnIndex']
            columns[index]['header'] = update['rows'][0]['values'][0]['userEnteredValue']['stringValue']
        elif 'moveDimension' in request:
            move = request['moveDimension']
            source, destination = move['source']['startIndex'], move['destinationIndex']
            value = columns.pop(source)
            columns.insert(destination if destination < source else destination - 1, value)
        else:
            raise AssertionError('Unexpected migration mutation')
    return columns, originals


class SensorLayoutTests(unittest.TestCase):
    def test_exact_common_columns(self):
        self.assertEqual(len(COMMON_HEADERS), 29)
        self.assertEqual(COMMON_HEADERS[:3], ('', 'Run', '캠페인'))
        self.assertEqual(COMMON_HEADERS[21:23], ('Date', 'Notes'))
        self.assertEqual(COMMON_HEADERS[-1], 'Train time scope')

    def test_initialization_headers_groups_freeze_and_widths(self):
        requests = initialize_requests(42)
        updates = [r['updateCells'] for r in requests if 'updateCells' in r]
        self.assertEqual(updates[0]['start'], {'sheetId': 42, 'rowIndex': 2, 'columnIndex': 0})
        self.assertEqual([c['userEnteredValue']['stringValue'] for c in updates[0]['rows'][0]['values']], list(COMMON_HEADERS))
        self.assertEqual(updates[1]['start']['rowIndex'], 1)
        props = next(r['updateSheetProperties']['properties'] for r in requests if 'updateSheetProperties' in r)
        self.assertEqual(props['gridProperties'], {'frozenRowCount': 3, 'frozenColumnCount': 3})
        widths = {}
        for request in requests:
            if 'updateDimensionProperties' in request:
                update = request['updateDimensionProperties']
                self.assertEqual(update['range']['dimension'], 'COLUMNS')
                for index in range(update['range']['startIndex'], update['range']['endIndex']):
                    widths[index] = update['properties']['pixelSize']
        self.assertEqual(widths[1], 400)
        self.assertEqual(widths[21], 110)
        self.assertEqual(widths[22], 500)
        self.assertTrue(all(widths[i] == 90 for i in set(range(29)) - {1, 21, 22}))
        self.assertFalse(any('mergeCells' in r or 'deleteDimension' in r for r in requests))

    def test_row_formats_are_layout_only_with_real_label_positions(self):
        formats = row_formats(['Notes', 'custom', 'Date', 'Run'], 17)
        self.assertEqual(formats[0], {'range': 'A17:D17', 'format': {'verticalAlignment': 'MIDDLE'}})
        by_range = {r['range']: r['format'] for r in formats}
        self.assertEqual(by_range['A17']['wrapStrategy'], 'CLIP')
        self.assertEqual(by_range['D17']['wrapStrategy'], 'CLIP')
        self.assertEqual(by_range['C17']['horizontalAlignment'], 'CENTER')
        for entry in formats:
            self.assertLessEqual(set(entry['format']), {'verticalAlignment', 'horizontalAlignment', 'wrapStrategy'})
        self.assertEqual(row_formats([], 4), [])
        self.assertEqual(row_formats(COMMON_HEADERS, 4)[0]['range'], 'A4:AC4')

    def test_sparse_shuffled_label_mapping_uses_actual_one_based_columns(self):
        labels = {'Date': 31, 'Notes': 24, 'Run': 2, 'Unknown': 35, '캠페인': 4}
        formats = row_formats(labels, 19)
        self.assertEqual(formats[0]['range'], 'A19:AI19')
        by_range = {entry['range']: entry['format'] for entry in formats}
        self.assertEqual(by_range['B19']['wrapStrategy'], 'CLIP')
        self.assertEqual(by_range['X19']['wrapStrategy'], 'CLIP')
        self.assertEqual(by_range['AE19']['horizontalAlignment'], 'CENTER')
        reversed_labels = dict(reversed(list(labels.items())))
        self.assertEqual(row_formats(reversed_labels, 19), formats)
        self.assertEqual(row_formats({}, 19), [])

    def test_label_mapping_rejects_invalid_or_colliding_columns(self):
        for labels in ({'Run': 0}, {'Run': -1}, {'Run': True}, {'Run': '2'},
                       {'Run': 2, 'Notes': 2}, {None: 2}):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                row_formats(labels, 4)

    def test_permutation_moves_values_formulas_and_preserves_extra_order(self):
        headers = ['', 'custom-left', *reversed(COMMON_HEADERS[1:]), 'custom-right', '']
        final, requests = migrate_front_requests(3, headers)
        self.assertEqual(final[:29], list(COMMON_HEADERS))
        self.assertEqual(final[29:], ['custom-left', 'custom-right', ''])
        actual, original = apply_requests(headers, requests)
        self.assertEqual([c['header'] for c in actual], final)
        for column in original:
            self.assertIn(column, actual)
        self.assertTrue(all('moveDimension' in r for r in requests))
        self.assertEqual(migrate_front_requests(3, final), (final, []))

    def test_missing_date_inserts_only_new_header_and_preserves_all_cells(self):
        headers = [h for h in COMMON_HEADERS if h != 'Date'] + ['Owner']
        final, requests = migrate_front_requests(9, headers)
        self.assertEqual(final, [*COMMON_HEADERS, 'Owner'])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['insertDimension']['range']['startIndex'], 21)
        actual, original = apply_requests(headers, requests)
        self.assertTrue(all(column in actual for column in original))
        self.assertEqual(actual[21], {'header': 'Date', 'value': None, 'formula': None})

    def test_absent_leading_blank_does_not_repurpose_unnamed_user_column(self):
        headers = ['Run', '', 'Owner']
        final, requests = migrate_front_requests(3, headers)
        self.assertEqual(final[:29], list(COMMON_HEADERS))
        self.assertEqual(final[29:], ['', 'Owner'])
        actual, original = apply_requests(headers, requests)
        self.assertEqual([c['header'] for c in actual], final)
        self.assertTrue(all(column in actual for column in original))
        self.assertIsNone(actual[0]['value'])

    def test_already_common_prefix_and_extra_columns_are_idempotent(self):
        headers = [*COMMON_HEADERS, 'Owner', '', 'Custom formula']
        before = headers[:]
        self.assertEqual(migrate_front_requests(0, headers), (headers, []))
        self.assertEqual(headers, before)

    def test_empty_header_sequence_builds_complete_prefix(self):
        final, requests = migrate_front_requests(0, [])
        self.assertEqual(final, list(COMMON_HEADERS))
        actual, original = apply_requests([], requests)
        self.assertEqual([c['header'] for c in actual], final)
        self.assertFalse(original)

    def test_duplicate_nonempty_headers_are_rejected_but_blanks_preserved(self):
        for labels in (['Run', 'Run'], ['Owner', '', 'Owner']):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'unique'):
                migrate_front_requests(1, labels)
            with self.assertRaises(ValueError):
                row_formats(labels, 4)
        final, _ = migrate_front_requests(1, ['', '', 'Run', ''])
        self.assertEqual(final.count(''), 3)

    def test_validation_rejects_invalid_ids_rows_and_nonstring_labels(self):
        for sheet_id in (-1, True, '3'):
            with self.assertRaises(ValueError):
                initialize_requests(sheet_id)
            with self.assertRaises(ValueError):
                migrate_front_requests(sheet_id, [])
        for header_row in (0, 1, True):
            with self.assertRaises(ValueError):
                initialize_requests(3, header_row)
        for labels in ('Run', [None], ['Run', 3]):
            with self.assertRaises(ValueError):
                migrate_front_requests(3, labels)
        with self.assertRaises(ValueError):
            row_formats(['Run'], 0)


if __name__ == '__main__':
    unittest.main()
