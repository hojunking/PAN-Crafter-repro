"""G20-only exception cannot become a historical PASS or skip load checks."""
from unittest.mock import patch
import unittest
from pathlib import Path
import tempfile

from g20 import references as ref
from g20.common import read_json
from g20 import test_references as fixtures
from qg40.test_references import ProbeModel, ProbeDataset


class ExceptionIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.PortableReferenceTests.setUp(self)
        self.exception = dict(status='ACCEPTED_WITH_PARITY_EXCEPTION', passed=False,
            accepted=True, authorization_sha256='a' * 64,
            historical_comparison={key: dict(passed=False, max_abs_error=value)
                for key, value in (('q', 2e-5), ('output', 1e-4), ('delta', 3e-5))})
        self.probe = ref._probe(ProbeModel(), ProbeDataset(), 'cpu')
        self.evaluator = dict(path='verified-production-evaluator', sha256='b' * 64)
        for name, kwargs in (
            ('g20.references._validate_payload', dict(return_value=(self.q, {}, self.data))),
            ('g20.references.verify_data_equivalence', dict(return_value={'passed': True})),
            ('g20.references._exception_evaluator', dict(return_value=self.evaluator)),
            ('g20.gpu_exception.authorization', dict(return_value={'user_approved': True})),
            ('g20.gpu_exception.measure', dict(return_value=(self.exception, self.probe))),
            ('g20.gpu_exception.validate', dict(return_value=True)),
            ('g20.gpu_exception.verify_probe_binding', dict(return_value=True)),
        ):
            mocker = patch(name, **kwargs)
            mocker.start()
            self.addCleanup(mocker.stop)

    def publish(self):
        return ref._publish_bridge(self.origin, {}, 'R0', 's1', self.root, self.data,
                                   'cpu', None, probe=self.probe)

    def test_exception_keeps_historical_failure_and_binds_consumer_identity(self):
        bridge = read_json(self.publish())
        self.assertFalse(bridge['q_cache_online_parity']['passed'])
        self.assertFalse(bridge['output_parity']['passed'])
        self.assertEqual(bridge['acceptance_status'], 'ACCEPTED_WITH_PARITY_EXCEPTION')
        manifest, _, _, _ = ref.validate_reference('R0', 's1', self.root)
        self.assertFalse(manifest['consumer_parity_exception']['historical_parity_passed'])
        self.assertEqual(manifest['consumer_parity_exception']['authorization_sha256'], 'a' * 64)

    def test_exception_load_remeasures_local_reference(self):
        self.publish()
        with patch('g20.gpu_exception.verify_local', return_value={'local_replay': True}) as verify:
            with patch.object(ref, '_online', side_effect=AssertionError('Historical gate must not be forged')):
                model, _, q = ref.load_reference('R0', 's1', self.root, device='cpu')
        verify.assert_called_once()
        self.assertEqual(model.q_cache_parity_receipt, {'local_replay': True})
        self.assertIs(q, self.q)

    def test_exception_requires_original_portable_probe(self):
        with self.assertRaisesRegex(ValueError, 'original portable probe'):
            ref._publish_bridge(self.origin, {}, 'R0', 's1', self.root, self.data, 'cpu', None)

    def test_exception_cannot_claim_historical_pass(self):
        path = self.publish()
        from g20.common import atomic_json
        bridge = read_json(path)
        bridge['q_cache_online_parity'] = dict(bridge['q_cache_online_parity'], passed=True)
        atomic_json(path, bridge)
        with self.assertRaisesRegex(ValueError, 'raw comparisons'):
            ref.validate_reference('R0', 's1', self.root)

    def test_exception_requires_unchanged_evaluator_gate(self):
        with patch.object(ref, '_exception_evaluator', side_effect=ValueError('metric gate failed')):
            with self.assertRaisesRegex(ValueError, 'metric gate failed'):
                self.publish()


class MetricAcceptanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.auth = self.root / 'approval.json'
        self.auth.touch()
        self.receipt = dict(status='ACCEPTED_WITH_METRIC_EXCEPTION',
                            passed=False, strict_passed=False, accepted=True)

    def test_reference_requires_valid_separate_metric_authorization(self):
        with patch('g20.metric_exception.authorization_path', return_value=self.auth), \
             patch('g20.metric_exception.accept_or_validate', return_value=self.receipt) as accept, \
             patch('g20.parity.verify_r0_parity', side_effect=AssertionError('Do not rewrite failed replay')):
            self.assertEqual(ref._exception_evaluator(self.root, 's1'), self.receipt)
            accept.assert_called_once()

    def test_invalid_metric_approval_does_not_fall_back(self):
        with patch('g20.metric_exception.authorization_path', return_value=self.auth), \
             patch('g20.metric_exception.accept_or_validate', side_effect=ValueError('approval changed')):
            with self.assertRaisesRegex(ValueError, 'approval changed'):
                ref._exception_evaluator(self.root, 's1')

    def test_controller_accepts_exception_without_claiming_strict_pass(self):
        from g20 import controller
        from g20.policy import CampaignWindow
        state = {}
        with patch.object(ref, 'reference_path', return_value=self.auth), \
             patch.object(ref, 'validate_reference'), \
             patch('g20.metric_exception.authorization_path', return_value=self.auth), \
             patch('g20.metric_exception.accept_or_validate', return_value=self.receipt), \
             patch('g20.parity.verify_r0_parity', side_effect=AssertionError('Retain strict failure')):
            self.assertTrue(controller.ensure_reference(self.root, 's1', 'R0', state,
                                                        CampaignWindow('2050-01-01T00:00:00Z')))
        self.assertFalse(state['parity']['passed'])
