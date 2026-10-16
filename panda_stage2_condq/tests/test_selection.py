"""User policy: HQNR threshold feasibility followed by minimum ERGAS."""
from copy import deepcopy
import unittest

from panda_method.calibration import canonical_sha256
from panda_stage2_condq.evaluation import checkpoint_rank, validate_result
from panda_stage2_condq.selection import SelectionPolicy, resolve_policy
from panda_stage2_condq.tests.test_checkpoint_guards import evaluation_fixture, change_identities
from panda_stage2_condq.tests.test_training_evaluation import stage_and_trainer


class SelectionTests(unittest.TestCase):
    def test_exact_threshold_and_no_scc_in_rank(self):
        policy = SelectionPolicy(.959)
        self.assertGreater(policy.rank(.959, 3., 5), policy.rank(.958999999, 1., 5))
        self.assertGreater(policy.rank(.959, 2., 5), policy.rank(.960, 2.1, 5))
        self.assertGreater(policy.rank(.958, 3., 5), policy.rank(.957, 1., 5))

    def test_ties_use_hqnr_or_ergas_then_earlier_update(self):
        policy = SelectionPolicy(.959)
        self.assertGreater(policy.rank(.961, 2., 5), policy.rank(.960, 2., 5))
        self.assertGreater(policy.rank(.958, 2., 5), policy.rank(.958, 3., 5))
        self.assertGreater(policy.rank(.960, 2., 5), policy.rank(.960, 2., 6))

    def test_invalid_metrics_and_targets_fail_closed(self):
        for target in (None, True, float('nan'), float('inf'), 0, -1, 1.1):
            with self.subTest(target=target), self.assertRaises(ValueError):
                SelectionPolicy(target)
        for h, e, step in ((float('nan'), 2, 1), (1.1, 2, 1), (.959, -1, 1),
                           (.959, 2, True), (.959, 2, -1)):
            with self.assertRaises(ValueError):
                SelectionPolicy(.959).rank(h, e, step)

    def test_only_wv3_has_an_established_default_target(self):
        self.assertEqual(resolve_policy('WV3'), SelectionPolicy(.959))
        for sensor in ('GF2', 'QB'):
            with self.assertRaises(ValueError):
                resolve_policy(sensor)
            self.assertEqual(resolve_policy(sensor, .97), SelectionPolicy(.97))

    def test_target_is_bound_to_resume_and_cannot_drift(self):
        trainer = stage_and_trainer()
        state = trainer.state_dict()
        changed = deepcopy(state)
        changed['contract']['checkpoint_selection'] = SelectionPolicy(.960).manifest()
        with self.assertRaises(ValueError):
            trainer.load_state_dict(changed)
        trainer.selection_policy = SelectionPolicy(.960)
        with self.assertRaises(ValueError):
            trainer.assert_contract()

    def test_eval_metadata_target_tamper_and_cross_target_comparison_rejected(self):
        trainer = stage_and_trainer()
        record = evaluation_fixture(trainer)
        bad = deepcopy(record)
        bad['checkpoint_selection']['hqnr_target'] = .960
        with self.assertRaises(ValueError):
            validate_result(bad)
        record['metric_function_injection'] = False  # Comparator fixture only, never published.
        with self.assertRaises(ValueError):
            checkpoint_rank(record, expected_policy=SelectionPolicy(.960))
        first = checkpoint_rank(record, expected_policy=SelectionPolicy(.959))
        record['rr']['metrics']['SCC'] = .5
        for row in record['rr']['per_scene']:
            row['metrics']['SCC'] = .5
        self.assertEqual(first, checkpoint_rank(record))
        # Internally self-consistent policy metadata is still bound to the
        # trainer when attaching the record to an actual checkpoint.
        bad = deepcopy(record)
        bad['checkpoint_selection'] = SelectionPolicy(.960).manifest()
        change_identities(bad, 'checkpoint_selection_sha256', canonical_sha256(bad['checkpoint_selection']))
        self.assertTrue(validate_result(bad))
        from panda_stage2_condq.checkpoint import _validate_attachment
        with self.assertRaises(ValueError):
            _validate_attachment(bad, trainer, trainer.stage.state_dict(), trainer.updates)


if __name__ == '__main__':
    unittest.main()
