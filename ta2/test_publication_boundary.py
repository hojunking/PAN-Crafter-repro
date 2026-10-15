"""Regression for the real admission formula that blocked all three lanes."""
import copy
import fcntl
import tempfile
import unittest

from ta2.reporting import AnalysisSheet, INDEX, Outbox, PublicationConflict
from ta2.test_publication import BatchBook
from ta2.test_reporting import record


class FormulaBook(BatchBook):
    def __init__(self):
        super().__init__()
        self.rows[8][0:3] = ['STATUS', 'ADMISSION_V2', 'TA2-LANE-S5']
        self.rows[8][INDEX['Record_ID']] = 'LANE:TA2-LANE-S5'
        self.native_cells = {(8, INDEX['Record_ID']):
            {'userEnteredValue': {'formulaValue': '="LANE:"&C8'}}}

    def fetch_sheet_metadata(self, params):
        result = super().fetch_sheet_metadata(params)
        if params.get('includeGridData'):
            for block in result['sheets'][0]['data']:
                for offset, row in enumerate(block['rowData']):
                    number = block['startRow'] + offset + 1
                    for col in range(len(row['values'])):
                        cell = self.native_cells.get((number, block['startColumn'] + col))
                        if cell is not None:
                            row['values'][col] = copy.deepcopy(cell)
        return result


class PreservedAdmissionBoundaryTest(unittest.TestCase):
    def publish(self, book, batch):
        publisher = AnalysisSheet(book)
        return (publisher.publish_many([record('a'), record('b')]) if batch
                else [publisher.publish(record('a'))])

    def test_single_and_batch_append_keep_admission_formula_and_are_idempotent(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                book = FormulaBook()
                previous = copy.deepcopy((book.rows, book.native_cells))
                receipts = self.publish(book, batch)
                self.assertTrue(all(r['status'] == 'VALUES_READBACK_VERIFIED' for r in receipts))
                self.assertTrue(all(r['row'] > 8 for r in receipts))
                self.assertEqual(book.rows[7], previous[0][7])
                self.assertEqual(book.rows[8], previous[0][8])
                self.assertEqual(book.native_cells, previous[1])
                self.assertTrue(all(not r['appended'] for r in self.publish(book, batch)))
                self.assertEqual(len(book.append_calls), 1)

    def test_other_boundary_formulas_and_non_admission_rows_still_block(self):
        for batch in (False, True):
            for variant in ('spill', 'wrong_column', 'wrong_reference', 'wrong_stage'):
                with self.subTest(batch=batch, variant=variant):
                    book = FormulaBook()
                    if variant == 'spill':
                        book.native_cells[(8, INDEX['Record_ID'])]['userEnteredValue']['formulaValue'] = '=ARRAYFORMULA(C8:C)'
                    elif variant == 'wrong_column':
                        book.native_cells[(8, 3)] = book.native_cells.pop((8, INDEX['Record_ID']))
                    elif variant == 'wrong_reference':
                        book.native_cells[(8, INDEX['Record_ID'])]['userEnteredValue']['formulaValue'] = '="LANE:"&C9'
                    else:
                        book.rows[8][1] = 'MEASURED'
                    with self.assertRaises(PublicationConflict): self.publish(book, batch)
                    self.assertFalse(book.append_calls)

    def test_header_target_formula_validation_and_chips_still_block(self):
        for batch in (False, True):
            for position in (7, 8, 9):
                for kind in ('formula', 'validation', 'chip'):
                    if position == 8 and kind == 'formula': continue
                    with self.subTest(batch=batch, position=position, kind=kind):
                        book = FormulaBook()
                        cell = ({'userEnteredValue': {'formulaValue': '=1+1'}} if kind == 'formula'
                            else {'dataValidation': {'condition': {'type': 'BOOLEAN'}}} if kind == 'validation'
                            else {'chipRuns': [{'startIndex': 0}]})
                        book.native_cells[(position, 2)] = cell
                        with self.assertRaises(PublicationConflict): self.publish(book, batch)
                        self.assertFalse(book.append_calls)

    def test_existing_record_formula_blocks_receipt_marker_rewrite(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                book = FormulaBook()
                self.publish(book, batch)
                book.native_cells[(9, INDEX['Record_ID'])] = {'userEnteredValue': {'formulaValue': '="LANE:"&C9'}}
                markers = len(book.marker_calls)
                with self.assertRaises(PublicationConflict): self.publish(book, batch)
                self.assertEqual(len(book.marker_calls), markers)
                self.assertEqual(len(book.append_calls), 1)

    def test_sparse_rows_never_call_table_inferred_values_append(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                book=FormulaBook()
                def forbidden(*args, **kwargs):
                    raise AssertionError('values.append may shift sparse rows right')
                book.values_append=forbidden
                self.publish(book,batch)
                self.assertEqual(book.rows[9][0],'RESPONSE')
                self.assertEqual(len(book.rows[9]),152)

    def test_exclusive_publication_owner_preserves_single_writer_and_enqueue(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Outbox(directory)
            out.enqueue(record('a'))
            with (out.root/'.publish.lock').open('a+') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    out.publish(AnalysisSheet(FormulaBook()))
                out.enqueue(record('b'))
                state=out.publish(AnalysisSheet(FormulaBook()),publisher_lock=owner)
                self.assertEqual(state['verified_records'],2)
                self.assertFalse(owner.closed)
                with self.assertRaises(BlockingIOError):
                    out.publish(AnalysisSheet(FormulaBook()))


if __name__ == '__main__':
    unittest.main()
