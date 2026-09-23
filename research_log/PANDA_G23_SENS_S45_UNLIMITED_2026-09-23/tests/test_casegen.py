from __future__ import annotations
import copy
import itertools
import math
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import casegen as c

class CaseTests(unittest.TestCase):
    def test_fourteen_templates(self):
        self.assertEqual(sum(len(c.cycle_cases(s,0)) for s in c.GROUPS),14)
    def test_thirteen_unique_conditions(self):
        allc=[x for s in c.GROUPS for x in c.cycle_cases(s,0)]
        self.assertEqual(len({x['parameter_signature_sha256'] for x in allc}),13)
    def test_one_factor_only(self):
        for s in c.GROUPS:
            for x in c.cycle_cases(s,0):
                d=[k for k in c.BASE if x['parameters'][k]!=c.BASE[k]]
                self.assertEqual(len(d),0 if x['case_id']=='BASE' else 1)
    def test_s4_axes(self):
        self.assertEqual({x['sweep_axis'] for x in c.cycle_cases('s4',0)}, {'baseline','alpha','beta','lambda_E'})
    def test_s5_axes(self):
        self.assertEqual({x['sweep_axis'] for x in c.cycle_cases('s5',0)}, {'baseline','q_ref_scale','tau_R_scale','r_A'})
    def test_all_fresh50(self):
        for x in itertools.islice(c.iter_cases('s4'),30):
            self.assertEqual(x['fixed']['train_updates'],50000)
            self.assertTrue(x['runtime_requirements']['no_parent_student_checkpoint'])
    def test_all_share_teacher(self):
        shas={x['fixed']['teacher_checkpoint_sha256'] for s in c.GROUPS for x in c.cycle_cases(s,9)}
        self.assertEqual(shas,{c.TEACHER_SHA256})
    def test_seed_anchor(self):
        self.assertEqual(c.seed_for(0),1234)
    def test_new_seeds(self):
        self.assertEqual([c.seed_for(i) for i in range(1,4)],[2000000,2000001,2000002])
    def test_paired_server_seeds(self):
        for i in range(8):
            a,b=c.cycle_cases('s4',i),c.cycle_cases('s5',i)
            self.assertEqual({x['seed'] for x in a+b},{c.seed_for(i)})
    def test_unique_run_ids(self):
        cases=[x for i in range(20) for s in c.GROUPS for x in c.cycle_cases(s,i)]
        self.assertEqual(len({x['run_id'] for x in cases}),len(cases))
    def test_local_baseline(self):
        for s in c.GROUPS:
            for i in range(6):
                for x in c.cycle_cases(s,i):
                    self.assertEqual(x['local_baseline_run_id'],c.run_id(s,i,'BASE'))
    def test_baseline_first(self):
        for s in c.GROUPS:
            for i in range(20): self.assertEqual(c.order_for(s,i)[0],'BASE')
    def test_six_cycle_order_balance(self):
        for s in c.GROUPS:
            orders=[c.order_for(s,i) for i in range(6)]
            self.assertEqual(len({tuple(o) for o in orders}),6)
            for code in orders[0][1:]:
                self.assertEqual(sorted(o.index(code) for o in orders),list(range(1,7)))
    def test_q_low_high(self):
        self.assertAlmostEqual(c.make_case('s5',0,'QR05')['resolved_values']['q_ref'], c.Q0/2)
        self.assertAlmostEqual(c.make_case('s5',0,'QR20')['resolved_values']['q_ref'], c.Q0*2)
    def test_weights_at_old_median(self):
        for code,w in [('QR05',1/3),('BASE',.5),('QR20',2/3)]:
            self.assertAlmostEqual(c.make_case('s5',0,code)['resolved_values']['w_at_original_q_median'],w)
    def test_no_qrenormalization(self):
        for x in c.cycle_cases('s5',1):
            self.assertFalse(x['fixed']['q_weight_mean_renormalization'])
            self.assertEqual(x['fixed']['q_numerator_factor'],1.)
    def test_tau_scaled_once(self):
        for code,m in [('TR05',.5),('BASE',1),('TR20',2)]:
            x=c.make_case('s5',0,code)
            o=x['backend_overrides']
            self.assertEqual(o['kdv.rec.tau'],c.TAU0)
            self.assertEqual(o['kdv.rec.tau_scale'],m)
            self.assertEqual(o['kdv.rec.tau']*o['kdv.rec.tau_scale'],x['resolved_values']['tau_R_used'])
    def test_rA_actual_lr(self):
        for code,v in [('RA01',1e-6),('BASE',3e-6),('RA06',6e-6)]:
            x=c.make_case('s5',0,code)
            self.assertAlmostEqual(x['resolved_values']['A_peak_lr'],v)
            self.assertEqual(x['backend_overrides']['learning_rate'],1e-4)
    def test_q_registry_keys(self):
        for x in c.cycle_cases('s5',0):
            self.assertNotIn('kdv.qrecon.q_ref_scale',x['backend_overrides'])
    def test_selection_fixed(self):
        for s in c.GROUPS:
            for x in c.cycle_cases(s,5):
                self.assertEqual(x['fixed']['primary_selection'],'EXACT_50000')
                self.assertFalse(x['fixed']['raw_max_selection_used_for_sensitivity'])
    def test_no_server_dependency(self):
        for s in c.GROUPS:
            self.assertFalse(c.make_case(s,0,'BASE')['runtime_requirements']['interserver_dependency'])
    def test_deepcopy_no_leak(self):
        x=c.make_case('s4',0,'BASE'); x['fixed']['student_depth'][0]=99
        self.assertEqual(c.make_case('s4',0,'BASE')['fixed']['student_depth'],[1,2,1])
    def test_tampered_case_rejected(self):
        x=c.make_case('s4',0,'AL05'); x['parameters']['beta']=.2
        with self.assertRaises(ValueError): c.validate_case(x)
    def test_spec_determinism(self):
        self.assertEqual(c.make_case('s4',10,'BE005'),c.make_case('s4',10,'BE005'))
    def test_resume_cursor(self):
        self.assertEqual(c.next_cursor(2,6),{'cycle':3,'position':0})
        self.assertEqual(c.next_cursor(2,3),{'cycle':2,'position':4})
        first=next(c.iter_cases('s5',2,5))
        self.assertEqual(first,c.cycle_cases('s5',2)[5])
    def test_invalid_requests(self):
        for f in [lambda:c.seed_for(-1),lambda:c.seed_for(True),lambda:c.make_case('s1',0,'BASE'),
                  lambda:c.make_case('s4',0,'QR05'),lambda:c.next_cursor(0,7)]:
            with self.assertRaises(ValueError): f()
    def test_no_seed_wrap(self):
        with self.assertRaises(ValueError): c.seed_for(2**32)
    def test_idempotent_file_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'; c.write_json(p,{'a':1}); c.write_json(p,{'a':1})
            with self.assertRaises(FileExistsError): c.write_json(p,{'a':2})

class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.case=c.make_case('s5',0,'RA06')
        self.receipt={k:self.case[k] for k in ['campaign_id','run_id','case_spec_sha256']}
        # Synthetic fixture only. These hashes are not actual GPU checkpoints.
        h='a'*64
        self.receipt.update(training_status='COMPLETE_EXACT_50000',actual_updates=50000,
                            evaluation_status='COMPLETE',primary_selection='EXACT_50000',
                            protocol_verified=True,checkpoint_sha256=h,
                            rr_checkpoint_sha256=h,fr_checkpoint_sha256=h,
                            metrics={'ergas':2.1,'hqnr':.95,'d_lambda':.02,'d_s':.03})
    def test_receipt_structure(self):
        c.verify_receipt(self.case,self.receipt)
    def test_early_finish_rejected(self):
        self.receipt['actual_updates']=49000
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)
    def test_mixed_checkpoints_rejected(self):
        self.receipt['fr_checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)
    def test_wrong_selection_rejected(self):
        self.receipt['primary_selection']='RAW_MAX'
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)
    def test_nan_metric_rejected(self):
        self.receipt['metrics']['hqnr']=math.nan
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)
    def test_unverified_protocol_rejected(self):
        self.receipt['protocol_verified']=False
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)
    def test_different_run_rejected(self):
        self.receipt['run_id']='not_this_run'
        with self.assertRaises(ValueError): c.verify_receipt(self.case,self.receipt)

if __name__=='__main__': unittest.main()
