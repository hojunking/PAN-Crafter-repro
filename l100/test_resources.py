from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from l100 import resources as r
from l100.plan import case_for, fullstate_steps, grid_steps


class ResourceTests(unittest.TestCase):
    def inventory(self, gpu='NVIDIA GeForce RTX 5090', free=10 ** 15):
        return dict(disk=dict(status='MEASURED', free_bytes=free),
                    gpu=dict(selected_device=dict(name=gpu) if gpu else None))

    def assess(self, measurement):
        case = case_for('L100I1-T03')
        with patch.object(r, 'inventory', return_value=measurement), \
                patch.object(r, '_model_footprint', return_value=dict(state_bytes=100, parameter_bytes=80)):
            return r.assess_block(Path('/tmp'), [case])

    def test_disk_includes_final_candidate_and_restart_fullstate_copies(self):
        case = case_for('L100I1-T03')
        state, weight = 260, 100
        total = len(grid_steps(case.updates)) * weight
        total += len(set(grid_steps(case.updates)) & set(fullstate_steps(case.role, case.updates))) * state
        total += len(fullstate_steps(case.role, case.updates)) * (weight + state)
        total += 2 * state + 2 * weight + 256 * 1024 ** 2
        value = self.assess(self.inventory())
        self.assertEqual(value['required_disk_bytes'], int(total * 1.25) + 1024 ** 3)
        self.assertTrue(value['allowed'])
        self.assertEqual(value['training_vram_requirement'], 'UNMEASURED_UNTIL_REAL_TRAIN')

    def test_missing_or_wrong_gpu_fails_closed(self):
        self.assertFalse(self.assess(self.inventory(gpu=None))['allowed'])
        self.assertFalse(self.assess(self.inventory(gpu='RTX 4090'))['allowed'])

    def test_unknown_disk_fails_closed(self):
        measurement = self.inventory(); measurement['disk'] = dict(status='UNKNOWN')
        self.assertIn('INSUFFICIENT_OR_UNKNOWN_DISK', self.assess(measurement)['reasons'])

    def test_gpu_idle_does_not_override_live_legacy_cpu_process(self):
        with patch.object(r, 'legacy_processes', return_value=[dict(pid=10, command='legacy')]), \
                patch.object(r.subprocess, 'check_output', return_value=''):
            self.assertFalse(r.idle_evidence()['idle'])

    def test_unknown_gpu_process_inventory_does_not_claim_idle(self):
        with patch.object(r, 'legacy_processes', return_value=[]), \
                patch.object(r.subprocess, 'check_output', side_effect=OSError('unavailable')):
            self.assertIsNone(r.idle_evidence()['gpu_pids'])
            self.assertFalse(r.idle_evidence()['idle'])

    def test_active_and_idle_gpu_queries(self):
        with patch.object(r, 'legacy_processes', return_value=[]):
            with patch.object(r.subprocess, 'check_output', return_value='12\n'):
                self.assertFalse(r.idle_evidence()['idle'])
            with patch.object(r.subprocess, 'check_output', return_value=''):
                self.assertTrue(r.idle_evidence()['idle'])


if __name__ == '__main__':
    unittest.main()
