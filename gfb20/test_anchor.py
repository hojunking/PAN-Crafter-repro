import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from g20.evaluation import signed_ds_details
from g20.test_evaluation import record
from gfb20.anchor import ATOL, RTOL, anchor_metrics, compare_metrics, expected_anchor, verify_and_save_parent_anchor
from gfb20.common import atomic_json, object_sha, read_json
from gfb20.plan import CAMPAIGN_ID, PARENT_ASSETS, case_for


def native_anchor(metrics, parent):
    """Synthetic scene covariance reproduces means, not product-of-means HQNR."""
    h, dl, ds = metrics['HQNR'], metrics['D_lambda'], metrics['D_s']
    a, b = .01, (h - (1 - dl) * (1 - ds)) / .01
    high = np.empty((20, 4)); low = np.full((20, 4), .5)
    rows = []
    for i in range(20):
        sign = 1 if i < 10 else -1
        li, si = dl + sign * a, ds + sign * b
        rows.append(dict(hqnr=(1 - li) * (1 - si), d_lambda=li, d_s=si))
        high[i] = .5 + si
    result = record(parent['parent_step'], h=h, e=metrics['ERGAS'], dl=dl)
    result['fr'].update(hqnr=h, d_lambda=dl, d_s=ds, per_scene=rows,
        signed_ds=signed_ds_details(high, low, [r['d_s'] for r in rows]))
    result.update(model_state_hash=parent['parent_tensor_hashes']['full'],
        parent_step=parent['parent_step'], candidate_eligible=False)
    return result


def old_parent(parent_id='P3HI'):
    old = PARENT_ASSETS['parents'][parent_id]
    return dict(parent_id=parent_id, server=old['owner_server'], parent_run_id=old['source_run_id'],
        parent_step=old['parent_step'], parent_model_sha256=old['published_model_sha256'] or 'b' * 64,
        parent_tensor_hashes={'full': 'a' * 64})


class AnchorTests(unittest.TestCase):
    def test_fixed_equality_not_performance_gate(self):
        self.assertEqual((ATOL, RTOL), (1e-6, 0.))
        targets = dict(HQNR=1., ERGAS=1., D_lambda=1., D_s=1.)
        observed = {k: v + .5e-6 for k, v in targets.items()}
        result = compare_metrics(targets, observed)
        self.assertTrue(result['passed'])
        self.assertFalse(result['automatic_relaxation'])
        self.assertFalse(result['substitute_parent_allowed'])
        observed['ERGAS'] += 1e-6
        self.assertFalse(compare_metrics(targets, observed)['passed'])
        observed['ERGAS'] = float('nan')
        failed = compare_metrics(targets, observed)
        self.assertFalse(failed['passed'])
        self.assertIsNone(failed['metrics']['ERGAS']['actual'])

    def test_fullprecision_bundle_and_saved_comparison(self):
        parent = old_parent()
        expected, source = expected_anchor(case_for('A01'), parent)
        self.assertEqual(expected['HQNR'], .9559600487382884)
        self.assertEqual(source['kind'], 'IMMUTABLE_AUTHOR_BUNDLE_FULL_PRECISION')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'comparison.json'
            anchor = native_anchor(expected, parent)
            result = verify_and_save_parent_anchor('A01', parent, anchor, path)
            self.assertTrue(result['passed'])
            self.assertEqual(result, read_json(path))
            verify_and_save_parent_anchor('A01', parent, anchor, path)

    def test_nonfinite_and_mismatch_failure_evidence_saved(self):
        parent = old_parent()
        values = PARENT_ASSETS['parents']['P3HI']['reported_metrics']
        for changed in (values['ERGAS'] + .001, float('nan')):
            with tempfile.TemporaryDirectory() as tmp:
                anchor = native_anchor(values, parent)
                anchor['rr']['ergas'] = changed
                path = Path(tmp) / 'comparison.json'
                with self.assertRaisesRegex(ValueError, 'BLOCKED_INTEGRITY'):
                    verify_and_save_parent_anchor('A01', parent, anchor, path)
                self.assertFalse(read_json(path)['passed'])

    def test_parent_identity_and_native_protocol_guard(self):
        parent = old_parent()
        values = PARENT_ASSETS['parents']['P3HI']['reported_metrics']
        anchor = native_anchor(values, parent)
        anchor['fr']['masking'] = True
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_and_save_parent_anchor('A01', parent, anchor, Path(tmp) / 'bad.json')
        for key, val in [('parent_step', 45450), ('parent_model_sha256', 'f' * 64), ('server', 's4')]:
            with self.assertRaises(ValueError): expected_anchor('A01', dict(parent, **{key: val}))
        old = old_parent('P3OLD')
        values, source = expected_anchor('A03', old)
        self.assertEqual(old['parent_step'], 50000)
        self.assertIsNone(source['published_model_sha256'])
        self.assertEqual(source['resolved_parent_model_sha256'], old['parent_model_sha256'])

    def test_fresh_trunk_exact_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trunk = case_for('A11')
            folder = root / 'work_dir' / trunk.run_id
            cfg = dict(work_dir=str(folder))
            cfgpath = folder / 'meta/config.resolved.yaml'
            atomic_json(cfgpath, cfg)
            identity = dict(update=100000, model_sha256='c' * 64, state_hash='a' * 64)
            identpath = folder / 'candidates/100000/identity.json'
            atomic_json(identpath, identity)
            row = record(100000); row.update(selection_id='EXACT_FINAL', checkpoint_identity=identity)
            summary = dict(campaign_id=CAMPAIGN_ID, run_id=trunk.run_id, case_id='A11',
                official_complete=True, config_sha256=object_sha(cfg), selections={'EXACT_FINAL': row})
            path = folder / 'official/summary.json'; atomic_json(path, summary)
            parent = dict(parent_id='P3NEW1', parent_kind='GFB20_FRESH_TRUNK', server='s3',
                parent_run_id=trunk.run_id, parent_step=100000, parent_config=str(cfgpath),
                parent_identity=str(identpath), parent_model_sha256=identity['model_sha256'],
                parent_tensor_hashes={'full': identity['state_hash']})
            values, source = expected_anchor('A12', parent, root)
            self.assertEqual(values, anchor_metrics(row))
            self.assertEqual(source['kind'], 'VERIFIED_FRESH_TRUNK_EXACT_FINAL')
            self.assertEqual(source['parent_model_sha256'], identity['model_sha256'])
            anchor = native_anchor(values, parent)
            comparison_path = root / 'ft_parent_comparison.json'
            first = verify_and_save_parent_anchor('A12', parent, anchor, comparison_path, root)
            summary.update(summary_at_utc='2026-09-22T18:00:00Z',
                           costs={'io_seconds': 9123.}, io_hours=2.534)
            # A measured evaluation duration may change when debt is repaid;
            # neither it nor summary bookkeeping changes the numeric anchor.
            summary['selections']['EXACT_FINAL']['seconds'] = 534.25
            atomic_json(path, summary)
            same_values, same_source = expected_anchor('A12', parent, root)
            self.assertEqual((same_values, same_source), (values, source))
            self.assertEqual(verify_and_save_parent_anchor('A12', parent, anchor, comparison_path, root), first)
            summary['selections']['EXACT_FINAL']['checkpoint_identity']['model_sha256'] = 'd' * 64
            atomic_json(path, summary)
            with self.assertRaises(ValueError): expected_anchor('A12', parent, root)


if __name__ == '__main__':
    unittest.main()
