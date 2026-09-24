"""No GPU/data/Sheet access: mandatory selector and balanced-report acceptance."""
import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch

from maina_hqnr.common import object_sha
from maina_hqnr.evaluation import PROTOCOL, JQM_VARIANT, preserve_rng, validate_report, evaluate_rr
from maina_hqnr.plan import make_case
from maina_hqnr.postrun import GRID, SELECTION_POLICY, select_candidates
from maina_hqnr.reporting import METRICS, PAIR_KEYS, paired_deltas, summarize
from maina_hqnr.upload import validate_summary


def _rehash(row):
    row.pop('row_sha256', None)
    row['row_sha256'] = object_sha(row)


def candidates():
    scenes = [dict(scene_index=i, scene_id=f'native:{i}', native_scene_sha256=object_sha(i)) for i in range(20)]
    pop = dict(count=20, scenes=scenes, scene_manifest_sha256=object_sha(scenes))
    records = []
    for index, step in enumerate(GRID):
        digest = object_sha(step)
        per_scene = []
        for scene in scenes:
            dl = .08+index*.0001; ds = .01
            row = dict(scene, d_lambda=dl, d_s=ds, hqnr=(1-dl)*(1-ds), checkpoint_sha256=digest)
            _rehash(row); per_scene.append(row)
        fr = {k: float(np.mean([r[k] for r in per_scene])) for k in ('hqnr', 'd_lambda', 'd_s')}
        fr.update(per_scene=per_scene, n_scenes=20, scene_ids=[s['scene_id'] for s in scenes],
                  scene_manifest_sha256=pop['scene_manifest_sha256'], discarded_samples=0, official_complete=True,
                  checkpoint_sha256=digest, support='full512', reference='native_PAN', masking=False,
                  alignment=False, hqnr_variant='raw-original', aggregation='mean_per_scene_HQNR')
        records.append(dict(update=step, checkpoint_sha256=digest,
            checkpoint_identity=dict(update=step, model_sha256=digest), fr=fr,
            metadata=dict(protocol=PROTOCOL, protocol_sha256=object_sha(PROTOCOL),
                optimizer_updates_during_evaluation=0, checkpoint_sha256=digest,
                data_sha256='d'*64, population=pop, evaluator=dict(sha256='e'*64))))
    return records


def set_hqnr(record, value):
    for row in record['fr']['per_scene']:
        row.update(d_lambda=1-value, d_s=0., hqnr=(1-(1-value)))
        _rehash(row)
    for key in ('hqnr', 'd_lambda', 'd_s'):
        record['fr'][key] = float(np.mean([row[key] for row in record['fr']['per_scene']]))


def complete_summary(server='s4', cycle=1, code='BASE', best_exact=False):
    case = make_case(server, cycle, code); records = candidates()
    if best_exact:
        set_hqnr(records[-1], .99)
    selected = select_candidates(records, 50000); result = {}
    for label, record in selected.items():
        fr = copy.deepcopy(record['fr']); digest = record['checkpoint_sha256']
        for row in fr['per_scene']:
            row['jqm'] = .91; _rehash(row)
        fr.update(jqm=float(np.mean([row['jqm'] for row in fr['per_scene']])), jqm_variant=JQM_VARIANT)
        rr = dict(per_scene=[], n_scenes=20, scene_ids=fr['scene_ids'], discarded_samples=0,
                  official_complete=True, checkpoint_sha256=digest, crop='20:-21', q_block=32)
        for index, source in enumerate(fr['per_scene']):
            row = {key: source[key] for key in ('scene_index', 'scene_id', 'native_scene_sha256', 'checkpoint_sha256')}
            row.update(ergas=2., sam=2.5, psnr=40., scc=.95, ssim=.98, q8=.93, rmse=15., cc=.99)
            _rehash(row); rr['per_scene'].append(row)
        for key in ('ergas', 'sam', 'psnr', 'scc', 'ssim', 'q8', 'rmse', 'cc'):
            rr[key] = float(np.mean([row[key] for row in rr['per_scene']]))
        result[label] = dict(update=record['update'], checkpoint_sha256=digest,
            checkpoint_identity=record['checkpoint_identity'], rr=rr, fr=fr,
            evaluation_manifest_sha256=object_sha(digest),
            alias_of='HQNR_MAX50' if label == 'EXACT_50000' and best_exact else None)
    return dict(campaign_id=case['campaign_id'], run_id=case['run_id'], case=case, attempt=0,
        complete=True, status='COMPLETE', actual_updates=50000, selection_policy=SELECTION_POLICY,
        candidate_grid_sha256=object_sha(list(GRID)), expected_candidates=50, evaluated_candidates=50,
        training_seconds=100., evaluation_seconds=50., wall_seconds=160.,
        started_at_utc='2026-09-25T01:00:00+00:00', completed_at_utc='2026-09-25T01:03:00+00:00',
        provenance={key: object_sha([server, cycle, key]) for key in PAIR_KEYS}, selections=result,
        paper_identity_status='PAPERSET_IDENTITY_UNVERIFIED')


class SelectorAcceptance(unittest.TestCase):
    def test_hqnr_wins_independently_of_ergas(self):
        rows = candidates(); rows[0]['val_ergas'] = 100.; rows[-1]['val_ergas'] = .01
        self.assertEqual(select_candidates(rows, 50000)['HQNR_MAX50']['update'], 1010)

    def test_full_precision_tie_uses_lower_step_not_ergas(self):
        rows = candidates(); set_hqnr(rows[0], .98); set_hqnr(rows[3], .98)
        rows[0]['ergas'] = 99.; rows[3]['ergas'] = .001
        self.assertEqual(select_candidates(rows, 50000)['HQNR_MAX50']['update'], 1010)

    def test_display_rounding_never_controls_selection(self):
        rows = candidates(); set_hqnr(rows[0], .970000001); set_hqnr(rows[1], .970000002)
        self.assertEqual(f'{rows[0]["fr"]["hqnr"]:.4f}', f'{rows[1]["fr"]["hqnr"]:.4f}')
        self.assertEqual(select_candidates(rows, 50000)['HQNR_MAX50']['update'], 2020)

    def test_all_grid_and_actual50k_are_mandatory(self):
        rows = candidates()
        for bad, actual in ((rows[:-1], 50000), (rows[:-1]+[rows[0]], 50000), (rows, 49490)):
            with self.subTest(actual=actual, count=len(bad)), self.assertRaises(ValueError):
                select_candidates(bad, actual)

    def test_raw_fr_population_numerical_and_sha_gates(self):
        for mutation in ('fr19', 'v64', 'masking', 'mean_product', 'sha', 'nan', 'scene_hash'):
            rows = candidates(); row = rows[0]
            if mutation == 'fr19': row['fr']['per_scene'].pop()
            if mutation == 'v64': row['fr']['support'] = 'V64'
            if mutation == 'masking': row['fr']['masking'] = True
            if mutation == 'mean_product': row['fr']['hqnr'] += .0000001
            if mutation == 'sha': row['checkpoint_identity']['model_sha256'] = 'x'*64
            if mutation == 'nan': row['fr']['hqnr'] = float('nan')
            if mutation == 'scene_hash': row['metadata']['population']['scene_manifest_sha256'] = 'x'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_candidates(rows, 50000)

    def test_exact_alias_and_same_checkpoint_rrfr(self):
        summary = complete_summary(best_exact=True)
        validate_summary(summary)
        summary['selections']['EXACT_50000']['alias_of'] = None
        with self.assertRaises(ValueError): validate_summary(summary)
        summary = complete_summary(); summary['selections']['HQNR_MAX50']['rr']['checkpoint_sha256'] = 'x'*64
        with self.assertRaises(ValueError): validate_summary(summary)

    def test_old_campaign_and_target_cannot_be_new_base(self):
        summary = complete_summary(); summary['campaign_id'] = 'PANDA_G23_SENS_WV3_S45_20260923_v1'
        with self.assertRaises(ValueError): validate_summary(summary)
        summary = complete_summary(); summary['selection_policy'] = dict(SELECTION_POLICY, primary_selection='TARGET')
        with self.assertRaises(ValueError): validate_summary(summary)

    def test_rng_restoration_success_and_interrupt(self):
        for interrupted in (False, True):
            random.seed(81); np.random.seed(82); torch.manual_seed(83)
            expected = (random.random(), np.random.random(), torch.rand(1))
            random.seed(81); np.random.seed(82); torch.manual_seed(83)
            try:
                with preserve_rng():
                    random.random(); np.random.random(10); torch.rand(100)
                    if interrupted: raise InterruptedError('safe')
            except InterruptedError: pass
            actual = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(expected[:2], actual[:2]); self.assertTrue(torch.equal(expected[2], actual[2]))


class BalancedReports(unittest.TestCase):
    def test_balanced_axis_excludes_anchor_and_missing_high(self):
        rows = [complete_summary('s4', 0, code) for code in ('BASE', 'AL05', 'AL15')]
        rows += [complete_summary('s4', 1, code) for code in ('BASE', 'AL05', 'AL15')]
        rows += [complete_summary('s4', 2, code) for code in ('BASE', 'AL05')]
        result = summarize(rows, 's4')
        alpha = [r for r in result['aggregate'] if r['axis'] == 'alpha' and r['selection'] == 'HQNR_MAX50']
        self.assertEqual(len(alpha), 1); self.assertEqual(alpha[0]['n_paired_student_seeds'], 1)
        self.assertEqual(alpha[0]['balanced_seed_set'], [2100000])
        self.assertEqual(len(result['anchors']), 6)
        self.assertTrue(any(r['cycle'] == 2 for r in result['incomplete_blocks']))

    def test_pairing_rejects_foreign_server_and_source(self):
        variant = complete_summary('s5', 0, 'AL05')
        with self.assertRaises(ValueError): paired_deltas(variant, complete_summary('s4', 1), 'HQNR_MAX50')
        baseline = complete_summary('s5', 0); variant['provenance']['stream_sha256'] = 'x'*64
        with self.assertRaises(ValueError): paired_deltas(variant, baseline, 'HQNR_MAX50')

    def test_failures_and_partial_attempts_remain_visible(self):
        partial = complete_summary('s5', 0, 'ED006')
        partial.update(complete=False, status='DIVERGED', actual_updates=202, selections={})
        result = summarize([complete_summary('s5', 0), partial], 's5')
        self.assertEqual(result['status_counts']['DIVERGED'], 1)
        self.assertEqual(result['failures'][0]['actual_updates'], 202)


class SupplementalNumericAcceptance(unittest.TestCase):
    def test_rmse_cc_bitwise_match_historical_hwc_reduction(self):
        from reporting_extra.evaluation import rr_metrics as original_extra
        from types import SimpleNamespace
        base = np.arange(8*256*256, dtype=np.float32).reshape(1, 8, 256, 256) % 1000
        gt = np.broadcast_to(base, (20, 8, 256, 256))
        sr = gt + np.arange(20, dtype=np.float32)[:, None, None, None] + .1234567
        expected = original_extra(sr, gt)
        fixture = complete_summary()['selections']['HQNR_MAX50']; digest = fixture['checkpoint_sha256']
        keys = ('ergas', 'sam', 'psnr', 'scc', 'ssim', 'q8')
        bare = dict(per_scene=[{key: row[key] for key in keys} for row in fixture['rr']['per_scene']],
                    standard_deviation={}, crop='20:-21', q_block=32)
        bare.update({key: fixture['rr'][key] for key in keys})
        pop = candidates()[0]['metadata']['population']
        with mock.patch('fh12.evaluation.infer', return_value=(sr, None)), \
             mock.patch('fh12.evaluation.native_gt', return_value=gt), \
             mock.patch('fh12.evaluation.rr_metrics', return_value=bare):
            measured = evaluate_rr(None, SimpleNamespace(has_gt=True), 'cpu', digest, pop=pop)['rr']
        for key in ('rmse', 'cc'):
            self.assertEqual(measured[key], expected[key])
            self.assertEqual(measured['standard_deviation'][key], expected['standard_deviation'][key])
            self.assertEqual([row[key] for row in measured['per_scene']], [row[key] for row in expected['per_scene']])


if __name__ == '__main__':
    unittest.main()
