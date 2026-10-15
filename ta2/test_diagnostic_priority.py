"""Regression tests for diagnostic delivery under continuous aggregate load."""
import hashlib
import tempfile
import time
import unittest

from ta2.common import load_json
from ta2.reporting import Outbox, make_record, status_record
from ta2.test_reporting import context, record, BatchDelivery, FakeDelivery


def sample(kind, index=0):
    fields = {
        'correction': dict(Native_c_dy=0.0, Native_c_dx=.1, Mean_norm_c=.1),
        'gain': dict(Gain=.2, Gain_den=425.),
        'alignment_proxy': dict(Residual_proxy_px=1.2, Shift_estimator='gradient'),
        'gradient': dict(Grad_rec_norm=.003, Grad_scope='whole'),
        'aggregate': dict(Metric_name='hqnr/mean', Estimate=.95, Completed_repeats=1),
    }
    record_type = 'GRAD' if kind == 'gradient' else 'PAIRED' if kind == 'aggregate' else 'RESPONSE'
    return make_record(context(), record_type, 'fair/' + kind + '/' + str(index), **fields[kind])


class DiagnosticPriorityTest(unittest.TestCase):
    def test_all_diagnostics_deliver_despite_large_aggregate_backlog(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Outbox(directory)
            out.enqueue_many(sample('aggregate', i) for i in range(50))
            for kind in ('correction', 'gain', 'alignment_proxy', 'gradient'):
                out.enqueue_many(sample(kind, i) for i in range(10))
            summary = make_record(context(), 'SUMMARY', 'summary', HQNR=.95)
            out.enqueue(summary)
            delivered = BatchDelivery()
            result = out.publish(delivered, 20)
            self.assertEqual(result['verified_records'], 20)
            self.assertEqual(delivered.batches[0], [summary['Record_ID']])
            for kind in ('correction', 'gain', 'alignment_proxy', 'gradient', 'aggregate'):
                self.assertGreater(result['selected_by_class'][kind], 0)

    def test_legacy_and_late_old_producer_rows_reclassify_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Outbox(directory)
            path = out.enqueue(sample('correction'))
            before = (path.read_bytes(), path.with_suffix('.csv').read_bytes())
            with out._db() as db:
                db.execute("UPDATE records SET delivery_class=NULL,failures=9,last_error='old failure',next_attempt=?", (time.time()+3600,))
            result = out.publish(FakeDelivery(), 10)
            self.assertEqual(result['attempted_this_batch'], 0)
            with out._db() as db:
                row = db.execute('SELECT * FROM records').fetchone()
                self.assertEqual(row['delivery_class'], Outbox._delivery_class(load_json(path)['record']))
                self.assertEqual((row['failures'], row['last_error']), (9, 'old failure'))
                self.assertGreater(row['next_attempt'], time.time())
            self.assertEqual(before, (path.read_bytes(), path.with_suffix('.csv').read_bytes()))
            late = out.enqueue(sample('gradient'))
            with out._db() as db:
                db.execute('UPDATE records SET delivery_class=NULL WHERE filename=?', (late.name,))
            self.assertEqual(out.publish(FakeDelivery(), 10)['verified_records'], 1)

    def test_single_record_calls_persist_fairness_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Outbox(directory)
            for kind in ('correction', 'gain', 'alignment_proxy', 'gradient', 'aggregate'):
                out.enqueue_many(sample(kind, i) for i in range(30))
            seen = set()
            for _ in range(20):
                out = Outbox(directory)
                state = out.publish(FakeDelivery(), 1)
                seen.update(k for k,v in state['selected_by_class'].items() if v)
            self.assertTrue({'correction', 'gain', 'alignment_proxy', 'gradient', 'aggregate'} <= seen)

    def test_due_and_unclassified_queries_remain_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Outbox(directory)
            with out._db() as db:
                plan = db.execute("EXPLAIN QUERY PLAN SELECT * FROM records WHERE state='pending' AND delivery_class=? AND next_attempt<=? ORDER BY next_attempt,sequence LIMIT ?", (2, 1., 100)).fetchall()
            detail = ' '.join(str(row[3]) for row in plan)
            self.assertIn('delivery_due', detail)
            self.assertNotIn('TEMP B-TREE', detail)

    def test_invalid_legacy_row_does_not_block_other_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Outbox(directory)
            broken = out.enqueue(sample('correction'))
            broken.write_text('{invalid json')
            with out._db() as db: db.execute('UPDATE records SET delivery_class=NULL')
            out.enqueue(sample('gradient'))
            state = out.publish(FakeDelivery(), 10)
            self.assertEqual(state['verified_records'], 1)
            self.assertEqual(len(state['errors']), 1)


if __name__ == '__main__': unittest.main()
