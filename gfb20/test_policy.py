import copy
import unittest
from datetime import timedelta
from gfb20.policy import *


def row(h=.95,e=.55,ds=.03,dl=.02,vh=None):
    one=dict(update=20000,rr=dict(ergas=e),fr=dict(hqnr=h,d_s=ds,d_lambda=dl))
    val=copy.deepcopy(one);val['fr']['hqnr']=h if vh is None else vh
    return dict(complete=True,selections=dict(EXACT_FINAL=one,RR_VAL_SELECTED=val))


class PolicyTests(unittest.TestCase):
    def setUp(self):self.w=CampaignWindow('2026-09-22T00:00:00+00:00')
    def test_common_window_cannot_reset(self):
        self.assertEqual(CampaignWindow.from_dict(self.w.to_dict()),self.w)
        altered=self.w.to_dict();altered['deadline_utc']='2026-09-23T00:00:00+00:00'
        with self.assertRaises(ValueError):CampaignWindow.from_dict(altered)
        with self.assertRaises(ValueError):CampaignWindow('2026-09-22T00:00:00')

    def test_whole_block_admission_cutoffs_and_debt(self):
        r=admission(self.w,self.w.t0_utc,'B_PANEL1')
        self.assertTrue(r['allowed']);self.assertEqual(len(r['run_ids']),3)
        with self.assertRaises(ValueError):admission(self.w,self.w.t0_utc,'B_PANEL1',remaining_hours=.65)
        self.assertFalse(admission(self.w,self.w.admission_cutoff_utc,'A_HI')['allowed'])
        self.assertFalse(admission(self.w,self.w.train_finish_utc,'A_HI')['allowed'])
        self.assertFalse(admission(self.w,self.w.t0_utc,'A_HI',debt_hours=17)['allowed'])

    def test_already_admitted_suffix_after16h(self):
        b=block_for('A_HI');r=admission(self.w,self.w.t0_utc,'A_HI')
        cont=admission(self.w,self.w.admission_cutoff_utc,'A_HI',completed=[b.run_ids[0]],receipt=r)
        self.assertTrue(cont['allowed'])
        wrong=copy.deepcopy(r);wrong['run_ids']=wrong['run_ids'][:1]
        with self.assertRaises(ValueError):admission(self.w,self.w.admission_cutoff_utc,'A_HI',receipt=wrong)

    def test_cost_evidence_same_local_scope_total_including_eval(self):
        c=block_for('A_HI').cases[0]
        observation=dict(server='s3',updates=20000,input_views='NATIVE',wall_hours=2.,
                         includes_eval_io_diagnostics=True,complete=True)
        self.assertAlmostEqual(estimate_case(c,[observation]),2.3)
        self.assertEqual(estimate_case(c,[dict(observation,server='s4')]),.6)
        self.assertEqual(estimate_case(c,[dict(observation,includes_eval_io_diagnostics=False)]),.6)

    def test_b_negative_ft_selects_e100_not_claimed_success(self):
        rows={k:row() for k in ('B01','B02','B03','B04','B05','B06')}
        p=promotion('s4',rows)
        self.assertEqual(p['selected_profile'],'E100')
        self.assertEqual(p['status'],'FRESH_PATH_TEST_AFTER_NEGATIVE_FT')
        self.assertTrue(block_eligible('B_FRESH1',p)[0])

    def test_b_positive_priority_worst_parent_then_e_then_k1(self):
        rows={k:row() for k in ('B01','B06')}
        rows.update({k:row(h=.953,ds=.027) for k in ('B02','B05','B03','B04')})
        self.assertEqual(promotion('s4',rows)['selected_profile'],'K1')
        rows['B03']=row(h=.954,ds=.027);rows['B04']=row(h=.954,ds=.027)
        self.assertEqual(promotion('s4',rows)['selected_profile'],'E100')

    def test_sensitive_checkpoint_not_positive_promotion(self):
        rows={k:row() for k in ('A01','A05')}
        rows.update({k:row(h=.953,e=.54,ds=.027,vh=.948) for k in ('A02','A06')})
        p=promotion('s3',rows)
        self.assertFalse(p['fresh_signal'])
        self.assertFalse(block_eligible('A_REPLAY1',p)[0])
        self.assertTrue(block_eligible('A_HI_REPEAT',p)[0])

    def test_c_calibration_only_gain_is_not_augmentation_success(self):
        rows={k:row() for k in ('C01','C06')}
        rows.update({k:row(h=.953,ds=.027) for k in ('C02','C03','C04','C05')})
        p=promotion('s5',rows)
        self.assertFalse(p['fresh_signal']);self.assertTrue(p['fresh_allowed_after_negative'])
        self.assertTrue(block_eligible('C_FRESH1',p,mixed_calibration_hours=3)[0])
        self.assertFalse(block_eligible('C_FRESH2',p,mixed_calibration_hours=3)[0])

    def test_missing_or_failed_pairs_never_success(self):
        self.assertFalse(promotion('s4',{})['ready'])
        self.assertFalse(block_eligible('B_FRESH1',promotion('s4',{}))[0])
        with self.assertRaises(ValueError):metric_vector(row(e=float('nan'))['selections']['EXACT_FINAL'])


if __name__=='__main__':unittest.main()
