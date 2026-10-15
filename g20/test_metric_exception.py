from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g20.common import CAMPAIGN_ID, atomic_json, camp, object_sha, read_json, sha256
from g20.metric_exception import (APPROVAL_TEXT, POLICY, accept_or_validate,
                                  authorization, authorization_fields, authorization_path)
from g20.parity import compare_metrics


class MetricExceptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = camp(self.root, 's1')
        self.source = dict(files={'g20/metric_exception.py': 'new'},
                           content_sha256='source', torch='2.4.0', cuda='11.8',
                           cudnn=90100, numeric_method_revision='QG40_SYNC_FREQ_C4_v1')
        self.source_patch = patch('g20.metric_exception.source_identity', return_value=self.source)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        registration = dict(identity=dict(campaign_id=CAMPAIGN_ID, server_id='s1', device='cuda',
                           source_identity=dict(self.source, files={'old': 'old'}, content_sha256='old')))
        atomic_json(self.folder / 'registration.json', registration)
        self.local = {'local': 'dataset'}
        atomic_json(self.folder / 'dataset_manifest.json', self.local)
        self.evidence_path = self.folder / 'r0_parity_evidence.json'
        self.raw_path = self.folder / 'diagnostics/r0_evaluator_parity.json'
        self.receipt_path = self.folder / 'diagnostics/r0_evaluator_metric_exception.json'
        self.seal_path = self.receipt_path.with_suffix('.seal.json')
        self.evidence = dict(schema='G20_R0_EVALUATOR_EVIDENCE_v1', assets={
            label: dict(checkpoint_sha256=label + '-checkpoint', prior_report_sha256=label + '-report')
            for label in ('TA', 'S92001')})
        atomic_json(self.evidence_path, self.evidence)
        self.prior = dict(hqnr=.9452, ergas=.5729)
        self.equality = {'all_four_splits': 'bound'}
        self.checked = ({}, {}, self.prior, self.equality, {})
        self.prior_patch = patch('g20.parity._prior', return_value=self.checked)
        self.prior_mock = self.prior_patch.start()
        self.addCleanup(self.prior_patch.stop)
        self.raw = self.make_raw()
        atomic_json(self.raw_path, self.raw)
        atomic_json(authorization_path(self.root), authorization_fields(self.root))

    def report(self, hqnr, ergas):
        common = dict(official_complete=True, sensor='GF2', num_bands=4, max_dn=1023, n_scenes=20)
        return dict(rr=dict(common, ergas=ergas, q4=.99, crop='20:-21', q_block=32),
                    fr=dict(common, hqnr=hqnr, reference='native_PAN', support='full512', masking=False,
                            aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original'))

    def make_raw(self, ta_hqnr=.94521303, student_hqnr=.94520424, ta_ergas=.57297289,
                 student_ergas=.5729491):
        rows = {}
        for label, hqnr, ergas in [('TA', ta_hqnr, ta_ergas),
                                   ('S92001', student_hqnr, student_ergas)]:
            row = compare_metrics(self.prior, dict(hqnr=hqnr, ergas=ergas))
            rows[label] = dict(row, **self.evidence['assets'][label], data_equality=self.equality,
                               measured=self.report(hqnr, ergas))
        return dict(schema='G20_R0_EVALUATOR_PARITY_v1', status='BLOCKED_PARITY', passed=False,
                    assets=rows, source_identity=self.source,
                    evidence_sha256=sha256(self.evidence_path), local_data_sha256=object_sha(self.local))

    def test_acceptance_preserves_failed_strict_receipt_and_thresholds(self):
        before = self.raw_path.read_bytes()
        result = accept_or_validate(self.root)
        self.assertEqual(result['status'], 'ACCEPTED_WITH_METRIC_EXCEPTION')
        self.assertTrue(result['accepted'])
        self.assertIs(result['passed'], False)
        self.assertIs(result['strict_passed'], False)
        self.assertFalse(result['assets']['TA']['passed'])
        self.assertTrue(result['assets']['S92001']['passed'])
        self.assertEqual(result['assets']['TA']['tolerance'], dict(hqnr=1e-5, ergas=1e-4))
        self.assertEqual(self.raw_path.read_bytes(), before)
        self.assertFalse(self.raw_path.with_suffix('.seal.json').exists())
        self.assertEqual(result['receipt_sha256'], sha256(self.receipt_path))
        self.assertEqual(result['seal_sha256'], sha256(self.seal_path))

    def test_cached_acceptance_revalidates_actual_original_chains(self):
        first = accept_or_validate(self.root)
        self.assertEqual(self.prior_mock.call_count, 2)
        self.assertEqual(first, accept_or_validate(self.root))
        self.assertEqual(self.prior_mock.call_count, 4)
        self.prior_mock.side_effect = ValueError('original artifact changed')
        with self.assertRaisesRegex(ValueError, 'original artifact changed'):
            accept_or_validate(self.root)

    def test_other_server_is_never_authorized(self):
        with self.assertRaisesRegex(ValueError, 'only for s1/R0'):
            accept_or_validate(self.root, 's2')

    def test_missing_authorization_never_accepts(self):
        authorization_path(self.root).unlink()
        with self.assertRaises(FileNotFoundError):
            accept_or_validate(self.root)
        self.assertFalse(self.receipt_path.exists())

    def test_exact_user_approval_and_policy_are_required(self):
        auth = read_json(authorization_path(self.root))
        for change in [dict(user_approved=1), dict(approval_text='승인한다'),
                       dict(policy=dict(POLICY, abs_difference=dict(hqnr=.001, ergas=1e-4))),
                       dict(reference_id='R1'), dict(original_registration_sha256='changed'),
                       dict(consumer_source_identity={'changed': True})]:
            with self.subTest(change=change):
                atomic_json(authorization_path(self.root), dict(auth, **change))
                with self.assertRaisesRegex(ValueError, 'authorization identity/policy'):
                    authorization(self.root)
        self.assertEqual(auth['approval_text'], APPROVAL_TEXT)

    def test_runtime_alignment_is_not_authorized(self):
        self.source['torch'] = '2.8.0+cu128'
        with self.assertRaisesRegex(ValueError, 'registered execution environment'):
            accept_or_validate(self.root)

    def test_missing_or_partial_or_stale_replay_is_rejected(self):
        variants = [dict(status='WAIT_PARITY'), dict(passed=True), dict(passed=0),
                    dict(source_identity={'old': True}), dict(evidence_sha256='changed'),
                    dict(local_data_sha256='changed'), dict(assets={'TA': self.raw['assets']['TA']})]
        for change in variants:
            with self.subTest(change=change):
                atomic_json(self.raw_path, dict(self.raw, **change))
                with self.assertRaisesRegex(ValueError, 'complete current-source strict FAILED'):
                    accept_or_validate(self.root)
        self.raw_path.unlink()
        with self.assertRaises(FileNotFoundError):
            accept_or_validate(self.root)

    def test_both_models_and_both_caps_are_enforced(self):
        for change in [dict(ta_hqnr=.9452201), dict(student_hqnr=.9452201),
                       dict(ta_ergas=.5730001), dict(student_ergas=.5730001)]:
            with self.subTest(change=change):
                atomic_json(self.raw_path, self.make_raw(**change))
                with self.assertRaisesRegex(ValueError, 'variance cap exceeded'):
                    accept_or_validate(self.root)
                self.assertFalse(self.receipt_path.exists())

    def test_cached_receipt_and_seal_are_immutable(self):
        accept_or_validate(self.root)
        receipt = read_json(self.receipt_path)
        atomic_json(self.receipt_path, dict(receipt, passed=True))
        with self.assertRaisesRegex(ValueError, 'Immutable metric exception receipt changed'):
            accept_or_validate(self.root)
        atomic_json(self.receipt_path, receipt)
        seal = read_json(self.seal_path)
        atomic_json(self.seal_path, dict(seal, source_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'Immutable artifact differs'):
            accept_or_validate(self.root)

    def test_raw_receipt_cannot_be_replaced_after_acceptance(self):
        accept_or_validate(self.root)
        atomic_json(self.raw_path, dict(self.raw, added='even non-numerical raw byte change'))
        with self.assertRaisesRegex(ValueError, 'Immutable artifact differs'):
            accept_or_validate(self.root)

    def test_strict_pass_seal_cannot_be_reused_or_overwritten(self):
        atomic_json(self.raw_path.with_suffix('.seal.json'), {'receipt_sha256': sha256(self.raw_path)})
        with self.assertRaisesRegex(ValueError, 'strict parity PASS seal'):
            accept_or_validate(self.root)

    def test_all_strict_pass_cannot_be_labeled_exception(self):
        atomic_json(self.raw_path, self.make_raw(ta_hqnr=.945201))
        with self.assertRaisesRegex(ValueError, 'actual strict parity failure'):
            accept_or_validate(self.root)

    def test_recomputed_measurements_bind_full_official_protocol(self):
        raw = deepcopy(self.raw)
        raw['assets']['TA']['measured']['fr']['masking'] = True
        atomic_json(self.raw_path, raw)
        with self.assertRaisesRegex(ValueError, 'official GF2'):
            accept_or_validate(self.root)

    def test_recomputed_comparisons_and_checkpoint_data_bindings_are_required(self):
        for change in [dict(abs_difference=dict(hqnr=0, ergas=0)), dict(checkpoint_sha256='changed'),
                       dict(prior_report_sha256='changed'), dict(data_equality={}),
                       dict(tolerance=dict(hqnr=2e-5, ergas=1e-4)), dict(passed=0)]:
            with self.subTest(change=change):
                raw = deepcopy(self.raw)
                raw['assets']['TA'].update(change)
                atomic_json(self.raw_path, raw)
                with self.assertRaisesRegex(ValueError, 'measurement binding differs'):
                    accept_or_validate(self.root)

    def test_artifact_change_during_validation_is_rejected(self):
        def mutate(*args, **kwargs):
            atomic_json(self.evidence_path, dict(self.evidence, new='changed during validation'))
            return self.checked
        self.prior_mock.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'changed during validation'):
            accept_or_validate(self.root)
        self.assertFalse(self.receipt_path.exists())


if __name__ == '__main__':
    unittest.main()
