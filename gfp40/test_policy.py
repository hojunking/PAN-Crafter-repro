import copy
import unittest
from datetime import timedelta
from gfp40.plan import *
from gfp40.policy import *
from gfp40.reporting import confirmation_summary,confirmation_pairs

def row(h=.95,e=.55,ds=.03,dl=.02,vh=None,completed='2026-09-22T20:00:00+00:00'):
    record=dict(update=20000,rr=dict(ergas=e),fr=dict(hqnr=h,d_s=ds,d_lambda=dl))
    val=copy.deepcopy(record);val['fr']['hqnr']=h if vh is None else vh
    return dict(complete=True,integrity_verified=True,official_completed_at_utc=completed,
        completed_at_utc='2026-09-22T19:00:00+00:00',selections=dict(EXACT_FINAL=record,RR_VAL_SELECTED=val))

def screen():
    rows={f'B{i:02d}':row() for i in range(1,23)}
    for f,pairs in SCREEN_PAIRS.items():
        for c,m in pairs:rows[m]=row(h=.953,ds=.027)
    return rows

class PolicyTests(unittest.TestCase):
    def setUp(self):self.w=CampaignWindow('2026-09-22T00:00:00Z')
    def test_local_clock_exact_34_37_40_and_immutable(self):
        self.assertEqual(self.w.admission_cutoff_utc-self.w.t0_utc,timedelta(hours=34))
        self.assertNotIn('lock_utc',self.w.to_dict())
        self.assertEqual(self.w.train_finish_utc-self.w.t0_utc,timedelta(hours=37))
        wrong=self.w.to_dict();wrong['deadline_utc']=wrong['train_finish_utc']
        with self.assertRaises(ValueError):CampaignWindow.from_dict(wrong)
    def test_atomic_6segment_reservation(self):
        receipt=admission(self.w,self.w.t0_utc,'C_SCHEDULE_1',diagnostic_hours=0)
        self.assertAlmostEqual(receipt['remaining_hours'],2*2.6+4*.65)
        self.assertEqual(len(receipt['run_ids']),6)
        self.assertFalse(admission(self.w,self.w.t0_utc+timedelta(hours=31),'C_SCHEDULE_1')['allowed'])
    def test_admission_rejects_other_server_clock(self):
        w=CampaignWindow(self.w.t0_utc,'s3')
        with self.assertRaises(ValueError):admission(w,w.t0_utc,'C_SCHEDULE_1')
    def test_outstanding_tail_and_cache_debt_cannot_be_ignored(self):
        self.assertFalse(admission(self.w,self.w.t0_utc+timedelta(hours=30),'S3_CONFIRM_2',
            outstanding_hours=3,cache_hours=1.5,debt_hours=.1)['allowed'])
    def test_already_started_pair_not_shed_due_to_updated_estimate(self):
        b=block_for('A_LONG60_1');receipt=admission(self.w,self.w.t0_utc,b)
        result=admission(self.w,self.w.train_finish_utc-timedelta(minutes=5),b,
            receipt=receipt,completed=[b.run_ids[0]])
        self.assertTrue(result['allowed']);self.assertTrue(result['projected_overrun'])
        self.assertFalse(admission(self.w,self.w.train_finish_utc,b,receipt=receipt)['allowed'])
    def test_after34_only_original_admitted_suffix(self):
        b=block_for('S3_CONFIRM_1');r=admission(self.w,self.w.t0_utc,b)
        self.assertFalse(admission(self.w,self.w.admission_cutoff_utc,b)['allowed'])
        self.assertTrue(admission(self.w,self.w.admission_cutoff_utc,b,receipt=r,completed=[b.run_ids[0]],diagnostic_hours=0)['allowed'])
        r['run_ids']=r['run_ids'][:1]
        with self.assertRaises(ValueError):admission(self.w,self.w.admission_cutoff_utc,b,receipt=r)
    def test_empty_or_winning_screen_does_not_change_confirmation_recipe(self):
        self.assertEqual(select_family({})['family'],'G025')
        rows=screen()
        for c,m in SCREEN_PAIRS['G050']:rows[m]=row(h=.96,ds=.02)
        self.assertEqual(select_family(rows)['family'],'G050')
        for c in CASES:
            if c.phase=='FIXED_CONFIRM' and c.arm!='NATIVE0':
                self.assertEqual(build_config(c)['gfp40']['family'],'G025')
    def test_no_upgrade_without_both_parents_and_incumbent_guard(self):
        rows=screen();rows['B12']=row(h=.99,ds=.01);rows['B13']=row(h=.952,ds=.028)
        self.assertEqual(select_family(rows)['family'],'G025')
        rows['B13']=row(h=.956,ds=.025);rows['B12']=row(h=.956,ds=.025)
        self.assertEqual(select_family(rows)['family'],'G0125')
        rows['B14']['integrity_verified']=False
        self.assertEqual(select_family(rows)['family'],'G025')
    def test_val_sensitive_candidate_excluded_and_low_high_not_candidates(self):
        rows=screen();rows.update(B12=row(h=.96,ds=.02,vh=.948),B13=row(h=.96,ds=.02))
        rows.update(C06=row(h=.999,ds=.001),C07=row(h=.999,ds=.001))
        self.assertEqual(select_family(rows)['family'],'G025')
        self.assertTrue(select_family(rows)['input_effects']['G0125']['checkpoint_sensitive'])
    def test_worst_parent_tie_then_ergas_then_fixed_order(self):
        rows=screen()
        for f in UPGRADE_ORDER:
            for c,m in SCREEN_PAIRS[f]:rows[m]=row(h=.956,ds=.025)
        self.assertEqual(select_family(rows)['family'],'G0125')
        for c,m in SCREEN_PAIRS['G050']:rows[m]=row(h=.956,ds=.025,e=.548)
        self.assertEqual(select_family(rows)['family'],'G050')
    def test_shedding_protects_started_blocks_and_confirmation(self):
        allblocks=blocks_for('s4')
        kept,dropped=trim_unstarted('s4',allblocks,16,started=['B_G0125_TWO_PARENTS'])
        self.assertEqual(dropped[:3],['S4_CONFIRM_3','B_P075_TWO_PARENTS','B_P025_TWO_PARENTS'])
        self.assertNotIn('B_G0125_TWO_PARENTS',dropped)
    def test_s5_third_seed_initial_budget_not_admitted(self):
        _,dropped=trim_unstarted('s5',blocks_for('s5'),37-2-1.5)
        self.assertIn('S5_CONFIRM_3',dropped)
    def test_six_pair_success_cannot_be_claimed_from_four_or_one(self):
        rows={}
        for s in SERVERS:
            for c,m in confirmation_pairs(s)[:2]:rows.update({c:row(),m:row(h=.953,ds=.027)})
        self.assertTrue(confirmation_summary(rows)['operational_success'])
        del rows['C29']
        self.assertFalse(confirmation_summary(rows)['operational_success'])

if __name__=='__main__':unittest.main()
