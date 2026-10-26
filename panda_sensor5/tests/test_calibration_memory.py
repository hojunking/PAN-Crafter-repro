"""Exercise the exact collector's allocation guard without allocating arrays."""
import unittest
from unittest.mock import patch

from panda_sensor5.calibration import HOST_CALIBRATION_BUDGET_BYTES
from panda_sensor5.contract import LANES
from panda_stage2_condq.calibration import CalibrationBuilder, sample_ids_sha256
from panda_sensor5.tests.test_core import make_donor


class CalibrationMemoryTests(unittest.TestCase):
    def test_complete_gf2_population_fits_declared_host_budget_and_old_budget_fails(self):
        lane=LANES['s3']
        self.assertEqual(lane['dataset'],'GF2')
        ids=[f'GF2:train:{i}' for i in range(lane['train_count'])]
        binding=make_donor('GF2').binding
        binding['sample_ids_sha256']=sample_ids_sha256(ids)
        binding['sample_count']=len(ids)
        def builder(budget):
            return CalibrationBuilder(ids,binding,k=5,memory_budget_bytes=budget,include_condq=True)
        old=builder(3*1024**3)
        with patch('panda_stage2_condq.calibration.np.empty') as allocation:
            with self.assertRaisesRegex(MemoryError,'3590412288'):
                old._allocate((64,64))
            allocation.assert_not_called()
        current=builder(HOST_CALIBRATION_BUDGET_BYTES)
        with patch('panda_stage2_condq.calibration.np.empty',return_value=None) as allocation:
            current._allocate((64,64))
            self.assertEqual(allocation.call_count,3)
            self.assertEqual([c.args[0] for c in allocation.call_args_list],
                             [19809*64*64,19809*48*48,19809*48*48])
        self.assertEqual(current.estimated_peak_bytes,3590412288)
        self.assertEqual(HOST_CALIBRATION_BUDGET_BYTES,4*1024**3)
        self.assertLess(current.estimated_peak_bytes,HOST_CALIBRATION_BUDGET_BYTES)

if __name__=='__main__':unittest.main()
