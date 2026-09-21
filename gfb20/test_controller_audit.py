"""Regression tests for controller interruption and train/postrun crash windows."""
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gfb20 import controller
from gfb20.plan import case_for
from gfb20.policy import CampaignWindow
from gfb20.test_policy import row


class ControllerAuditTest(unittest.TestCase):
    def _promotion_fixture(self, *, low_k1=False):
        rows = {key: row() for key in ('B01', 'B03', 'B04', 'B06')}
        rows.update({key: row(h=.953, ds=.027) for key in ('B02', 'B05')})
        for value in rows.values():
            value.update(config_sha256='cfg', source_identity={'content_sha256': 'src'})
        names = {case_for(key).run_id: key for key in rows}
        def read(path, default=None):
            path = Path(path)
            if path.name == 'identity.json':
                return {'state_hash': 'weights'}
            if path.name.startswith('probe_'):
                identifier = names[path.parent.parent.name]
                low = low_k1 and identifier in ('B02', 'B05')
                return dict(local_step=int(path.stem.split('_')[1]), model_state_hash='weights',
                    config_sha256='cfg', source_identity={'content_sha256': 'src'},
                    gradient={'soft_over_hard': 1e-5 if low else .1},
                    soft_active_fraction=1e-6 if low else .1)
            return {} if default is None else default
        return rows, read

    def test_s4_metric_success_without_gradient_evidence_cannot_promote(self):
        rows, _ = self._promotion_fixture()
        with patch.object(controller, 'summaries', return_value=rows), \
                patch.object(controller, 'read', return_value={}):
            result = controller.promotion_evidence(Path('/unused'), 's4')
        self.assertFalse(result['ready'])
        self.assertFalse(result['fresh_signal'])
        self.assertEqual(result['status'], 'REQUIRED_GRADIENT_DIAGNOSTICS_PENDING')

    def test_s4_valid_active_diagnostics_preserve_metric_winner(self):
        rows, read = self._promotion_fixture()
        with patch.object(controller, 'summaries', return_value=rows), \
                patch.object(controller, 'read', side_effect=read):
            result = controller.promotion_evidence(Path('/unused'), 's4')
        self.assertTrue(result['ready'])
        self.assertTrue(result['fresh_signal'])
        self.assertEqual(result['selected_profile'], 'K1')

    def test_s4_low_active_mass_does_not_borrow_k1_positive_signal(self):
        rows, read = self._promotion_fixture(low_k1=True)
        with patch.object(controller, 'summaries', return_value=rows), \
                patch.object(controller, 'read', side_effect=read):
            result = controller.promotion_evidence(Path('/unused'), 's4')
        self.assertTrue(result['ready'])
        self.assertEqual(result['selected_profile'], 'E100')
        self.assertFalse(result['fresh_signal'])
        self.assertTrue(result['fresh_allowed_after_negative'])
        self.assertEqual(result['selection_override'], 'LOW_ACTIVE_MASS_BOTH_PARENTS')

    def test_s4_stale_diagnostic_checkpoint_is_not_promotion_evidence(self):
        rows, read = self._promotion_fixture()
        def stale(path, default=None):
            result = read(path, default)
            if Path(path).name == 'probe_5000.json':
                result['model_state_hash'] = 'another_checkpoint'
            return result
        with patch.object(controller, 'summaries', return_value=rows), \
                patch.object(controller, 'read', side_effect=stale):
            result = controller.promotion_evidence(Path('/unused'), 's4')
        self.assertFalse(result['ready'])

    def _case_setup(self, folder):
        root = Path(folder)
        case = case_for('A01')
        window = CampaignWindow(dt.datetime.now(dt.timezone.utc))
        state = dict(runs={}, observations=[])
        return root, case, window, state

    def test_completed_training_metadata_goes_directly_to_postrun_after_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            root, case, window, state = self._case_setup(folder)
            calls = []
            def read(path, default=None):
                if Path(path).name == 'training_status.json':
                    return dict(training_complete=True, actual_updates=case.updates,
                                status='TRAIN_COMPLETE_EVAL_PENDING')
                if Path(path).name == 'summary.json':
                    return dict(complete=True)
                return {} if default is None else default
            def command(root, server, args, log, deadline):
                calls.append(args[0])
                return 0, .01
            with patch.object(controller, 'resolve_config', return_value=root/'config.yaml'), \
                    patch.object(controller, 'read', side_effect=read), \
                    patch.object(controller, '_command', side_effect=command), \
                    patch.object(controller, '_save'), patch.object(controller, 'append_event'), \
                    patch('gfb20.training.recover_exact_endpoint', return_value=False):
                controller._run_case(root, 's3', state, case, window, False)
            self.assertEqual(calls, ['postrun'], 'Verified completed training must never be relaunched')

    def test_safe_signal_pause_remains_resumable_not_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            root, case, window, state = self._case_setup(folder)
            def read(path, default=None):
                if Path(path).name == 'training_status.json':
                    return dict(training_complete=False, actual_updates=1234, status='PAUSED_SIGNAL')
                return {} if default is None else default
            with patch.object(controller, 'resolve_config', return_value=root/'config.yaml'), \
                    patch.object(controller, 'read', side_effect=read), \
                    patch.object(controller, '_command', return_value=(75, .01)), \
                    patch.object(controller, '_save'), patch.object(controller, 'append_event'), \
                    patch('gfb20.training.recover_exact_endpoint', return_value=False):
                proceed = controller._run_case(root, 's3', state, case, window, False)
            self.assertEqual(proceed, 'PAUSED_SAFE')
            self.assertFalse(state['runs'][case.run_id].get('terminal', False),
                             'Operator safe pause must retain exact-resume eligibility')

    def test_actual_time_limit_is_not_mislabeled_safe_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            root, case, window, state = self._case_setup(folder)
            def read(path, default=None):
                if Path(path).name == 'training_status.json':
                    return dict(training_complete=False, actual_updates=1234, status='PARTIAL_TIME_LIMIT')
                return {} if default is None else default
            with patch.object(controller, 'resolve_config', return_value=root/'config.yaml'), \
                    patch.object(controller, 'read', side_effect=read), \
                    patch.object(controller, '_command', return_value=(75, .01)), \
                    patch.object(controller, '_save'), patch.object(controller, 'append_event'), \
                    patch('gfb20.training.recover_exact_endpoint', return_value=False):
                self.assertFalse(controller._run_case(root, 's3', state, case, window, False))
            self.assertEqual(state['runs'][case.run_id]['status'], 'PARTIAL_TIME_LIMIT')


if __name__ == '__main__':
    unittest.main()
