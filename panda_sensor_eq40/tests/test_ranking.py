import unittest
from panda_sensor_eq40.ranking import choose

def row(step,e,p,h,corrected=.999):
    return dict(step=step,rr=dict(ERGAS=e,PSNR=p),fr=dict(HQNR=h),
                fr_corrected_pan=dict(HQNR=corrected))

class RankingTests(unittest.TestCase):
    def test_explicit_gates_before_hqnr(self):
        a,b,c=row(1,.523,45.2,.99),row(2,.52,45.07,.98),row(3,.522,45.08,.965)
        self.assertIs(choose([a,b,c],ergas_max=.522,psnr_min=45.08),c)
    def test_psnr_order_ignores_corrected_metric(self):
        a,b=row(1,3.49,38.5,.90,.99),row(2,3.50,38.6,.89,.85)
        self.assertIs(choose([a,b],ergas_max=3.5,order='PSNR_HQNR'),b)
    def test_no_fallback_or_nonfinite(self):
        with self.assertRaisesRegex(ValueError,'NO_QUALIFIED'):
            choose([row(1,3.6,39,.99)],ergas_max=3.5)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            choose([row(1,3.5,float('nan'),.99)],ergas_max=3.5)
    def test_ties_deterministic(self):
        a,b=row(2000,3.49,38.6,.94),row(1000,3.49,38.6,.94)
        self.assertIs(choose([a,b],ergas_max=3.5,order='PSNR_HQNR'),b)

if __name__=='__main__':unittest.main()
