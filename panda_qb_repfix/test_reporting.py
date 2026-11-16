from copy import deepcopy
import json
import unittest
from directshift48.reporting import INDEX
from panda_input3 import plan,publication as pub
from panda_input3_original.test_reporting import original_record
from panda_input3_original import reporting as old
from panda_input3_targets import reporting as prior
from panda_fr_columns import publication as fr
from . import reporting
from .policy import POLICY,choose,passes

CFG=deepcopy(plan.resolve('s2','C_CORRECTED_DIFF'))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):reporting.install()

    def test_failed_hqnr_still_has_original_reference_representative(self):
        a=original_record(CFG,1000,original_h=.92,h=.8,e=3.6)
        b=original_record(CFG,2000,original_h=.91,h=.99,e=3.4)
        selected=choose([b,a],'stage2',CFG)
        self.assertEqual([(r['step'],selector) for r,selector,_ in selected],[(1000,'BEST_HQNR_FALLBACK')])
        self.assertFalse(passes(a,CFG))
        self.assertEqual(choose([],'stage2',CFG),[])

    def test_qualified_min_ergas_remains_even_when_rr_misses(self):
        a=original_record(CFG,1000,original_h=.94,e=3.6)
        b=original_record(CFG,2000,original_h=.93,e=3.4)
        self.assertEqual(choose([a,b],'stage2',CFG)[0][0]['step'],1000)
        self.assertEqual(len(choose([a,b],'stage2',CFG)),1)

    def test_raw_threshold_nonfinite_and_conflicting_evidence(self):
        r=original_record(CFG,original_h=.93149999999,e=3.5)
        self.assertEqual(choose([r],'stage2',CFG)[0][1],'BEST_HQNR_FALLBACK')
        r['fr']['metrics']['HQNR']=float('nan')
        with self.assertRaises(ValueError):choose([r],'stage2',CFG)
        a=original_record(CFG);b=deepcopy(a);b['rr']['metrics']['ERGAS']+=.1
        with self.assertRaises(ValueError):choose([a,b],'stage2',CFG)

    def test_v4_blank_upgrades_same_row_id_and_verifies_prior_evidence(self):
        r=original_record(CFG,original_h=.92,h=.96,e=3.51)
        with reporting.old_context():
            blank,blank_note=old.make_row(CFG,'COMPLETED',None,'STATUS','ROW0_ORIGINAL')
        row,note=pub.make_row(CFG,'COMPLETED',r,'BEST_HQNR_FALLBACK','ROW0_ORIGINAL')
        value=dict(policy=POLICY,configs={CFG['run_id']:CFG},proof={(CFG['run_id'],1000):r},rows=[row],notes=[note])
        self.assertEqual(row[INDEX['Result_ID']],blank[INDEX['Result_ID']])
        pub.verify_owned(blank,value,'s2');pub.verify_owned(row,value,'s2')
        self.assertEqual(prior.retired_rows(dict(rows=[blank],notes=[blank_note]),value,'s2'),[])
        info=json.loads(note)['publication']
        self.assertTrue(info['is_case_representative']);self.assertTrue(info['fallback_used'])
        self.assertFalse(info['joint_pass']);self.assertIn('HQNR 미달',row[INDEX['Selection@Step']])
        extra=fr.extras(value)[row[INDEX['Result_ID']]]
        self.assertEqual(extra[0]['userEnteredValue']['numberValue'],.96)
        self.assertEqual(json.loads(extra[0]['note'])['official_selection_reference'],'ORIGINAL_PAN')
        self.assertEqual(json.loads(note)['configuration'],CFG)
        bad=deepcopy(row);bad[INDEX['HQNR↑']]=.99
        with self.assertRaises(ValueError):pub.verify_owned(bad,value,'s2')

    def test_passing_details_and_existing_qualified_row_stay_valid(self):
        r=original_record(CFG,CFG['primary_step'],original_h=.94,e=3.49)
        selected=choose([r],'stage2',CFG)
        self.assertEqual(len(selected),2);self.assertIn('FIXED',selected[1][1])
        with reporting.old_context():
            previous,note=old.make_row(CFG,'COMPLETED',r,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        value=dict(configs={CFG['run_id']:CFG},proof={(CFG['run_id'],r['step']):r})
        pub.verify_owned(previous,value,'s2')


if __name__=='__main__':unittest.main()
