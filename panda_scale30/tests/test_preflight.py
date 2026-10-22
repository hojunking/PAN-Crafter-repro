"""Executed preflight evidence must exclude new losses and a second Student view."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_scale30.preflight import validate_hard_only_step, attempt_directory


class PreflightTests(unittest.TestCase):
    def test_retry_preserves_previous_immutable_disposable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('panda_scale30.preflight.time.time_ns', side_effect=(123, 124)):
                first = attempt_directory(directory)
                checkpoint = first / 'S020BASE_resume.pt'
                checkpoint.write_bytes(b'previous complete validation checkpoint')
                second = attempt_directory(directory)
            self.assertNotEqual(first, second)
            self.assertTrue(second.is_dir())
            self.assertFalse((second / checkpoint.name).exists())
            self.assertEqual(checkpoint.read_bytes(), b'previous complete validation checkpoint')

    def test_executed_hard_only_step_accepts_original_single_view(self):
        validate_hard_only_step({'student_views': 1, 'consistency_lambda': 0.,
            'soft_weighted': 0., 'loss': .27, 'hard_weighted': .27})

    def test_extra_loss_or_second_view_is_rejected(self):
        original = {'student_views': 1, 'consistency_lambda': 0.,
                    'soft_weighted': 0., 'loss': .27, 'hard_weighted': .27}
        for changed in ({'student_views': 2}, {'consistency_lambda': .05},
                        {'soft_weighted': .01}, {'loss': .28},
                        {'consistency_raw': 0.}, {'consistency_weighted': 0.}):
            with self.assertRaisesRegex(ValueError, 'extra loss or Student view'):
                validate_hard_only_step({**original, **changed})


if __name__ == '__main__':
    unittest.main()
