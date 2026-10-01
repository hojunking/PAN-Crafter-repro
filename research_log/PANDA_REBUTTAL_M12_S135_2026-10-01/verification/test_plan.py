"""Offline registry/algebra tests only; no model, GPU, data, or Sheets access."""
from __future__ import annotations
import csv, json, math, sys, unittest
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'planning/experiment_registry.json').read_text())
with (ROOT/'planning/server_schedule.csv').open(encoding='utf-8-sig',newline='') as f:
    SCHEDULE=list(csv.DictReader(f))
T={x['case_id']:x for x in REG['templates']}

def mean(xs): return sum(xs)/len(xs)
def full_hard(d,e): return mean([(1+x)*y for x,y in zip(d,e)])
def mean_hard(d,e): return mean([1+x for x in d])*mean(e)
def soft(t,a,res): return .1*mean([x*y*z for x,y,z in zip(t,a,res)])
def abar(t,a):
    denom=sum(t)
    return sum(x*y for x,y in zip(t,a))/denom if denom>0 else 0.

def objectives(h,k,edge,we,wa):
    return mean([x+y+.002*z*w for x,y,z,w in zip(h,k,edge,we)]),mean([x*w for x,w in zip(h,wa)])

class PlanTests(unittest.TestCase):
    def test_01_case_count(self): self.assertEqual(len(T),10)
    def test_02_run_count_and_unique_ids(self):
        self.assertEqual(len(REG['training_runs']),120)
        self.assertEqual(len({r['run_id'] for r in REG['training_runs']}),120)
    def test_03_four_repeats_per_server(self):
        for s in REG['servers']:
            rows=[r for r in REG['training_runs'] if r['server']==s]
            self.assertEqual(len(rows),40)
            self.assertEqual(set(r['repeat'] for r in rows),{1,2,3,4})
    def test_04_distinct_seed_cohort(self):
        seeds=[s for values in REG['seed_table'].values() for s in values]
        self.assertEqual(len(set(seeds)),12)
        self.assertTrue(set(seeds).isdisjoint({9281101,9281102,9281301,9281302,9281501,9281502}))
    def test_05_all_cases_in_each_block(self):
        for s in REG['servers']:
            for r in range(1,5):
                rows=[x for x in REG['training_runs'] if x['server']==s and x['repeat']==r]
                self.assertEqual({x['case_id'] for x in rows},set(T))
                self.assertEqual(len({x['seed'] for x in rows}),1)
    def test_06_every_case_has_twelve(self):
        self.assertEqual(set(Counter(r['case_id'] for r in REG['training_runs']).values()),{12})
    def test_07_two_phases(self):
        self.assertEqual(Counter(r['phase'] for r in REG['training_runs']),{'STEP1_QROUTE':72,'STEP2_FITTING':48})
    def test_08_validation_grid(self):
        self.assertEqual(REG['training']['validation_grid'],[1010*n for n in range(1,50)]+[50000])
    def test_09_native_tasks(self):
        self.assertEqual(len(REG['native_tasks']),240)
        self.assertEqual(set(Counter(r['source_run_id'] for r in REG['native_tasks']).values()),{2})
    def test_10_stress_tasks(self):
        c=REG['stress_tasks']
        self.assertEqual(len(c),288)
        self.assertEqual(set(Counter(r['source_run_id'] for r in c).values()),{4})
        self.assertEqual(sum(r['shifts']*r['rr_scenes'] for r in c),282240)
    def test_11_step1_balance(self):
        counts=Counter()
        for s in REG['servers']:
            for rep in range(1,5):
                rows=[r for r in SCHEDULE if r['server']==s and r['repeat']==str(rep) and r['phase']=='STEP1_QROUTE' and r['action']=='TRAIN_FRESH_50K_AND_NATIVE']
                for p,r in enumerate(rows): counts[(r['case_id'],p)]+=1
        self.assertEqual(len(counts),36);self.assertEqual(set(counts.values()),{2})
    def test_12_step2_balance(self):
        counts=Counter()
        for s in REG['servers']:
            for rep in range(1,5):
                rows=[r for r in SCHEDULE if r['server']==s and r['repeat']==str(rep) and r['phase']=='STEP2_FITTING' and r['action']=='TRAIN_FRESH_50K_AND_NATIVE']
                for p,r in enumerate(rows): counts[(r['case_id'],p)]+=1
        self.assertEqual(len(counts),16);self.assertEqual(set(counts.values()),{3})
    def test_13_finite_schedule(self):
        for s in REG['servers']:
            rows=[r for r in SCHEDULE if r['server']==s]
            self.assertEqual([int(r['position']) for r in rows],list(range(1,len(rows)+1)))
            self.assertEqual(rows[-1]['action'],'FINAL_REPORT_AND_STOP_FOR_REVIEW')
    def test_14_b01_not_reused(self):
        self.assertTrue(all(not r['old_B01_reuse'] and r['run_id'].startswith('RBM12_') for r in REG['training_runs']))
    def test_15_single_frozen_teacher(self):
        self.assertEqual(len({r['teacher_run_id'] for r in REG['training_runs']}),1)
        self.assertEqual(REG['expected']['independent_teacher_models'],1)
    def test_16_geometry_2x2(self):
        observed={(T[k]['edge_weight'],T[k]['aligner_weight']) for k in ('QFULL','QMEAN','QEDGE','QALIGN')}
        self.assertEqual(observed,{('QFULL','QFULL'),('QFULL','TRAIN_MEAN'),('TRAIN_MEAN','QFULL'),('TRAIN_MEAN','TRAIN_MEAN')})
    def test_17_hard_spatial_mass_preserved(self):
        for d in ([0,.1,.8,.2],[.5,.5,.5,.5]):
            h=[1+x for x in d]
            self.assertAlmostEqual(sum(h),len(h)*mean(h))
    def test_18_hard_constant_parity(self):
        self.assertAlmostEqual(full_hard([.5]*3,[.1,.4,.8]),mean_hard([.5]*3,[.1,.4,.8]))
    def test_19_advantage_weighted_mass_preserved(self):
        t=[.1,.7,.9,.3];a=[0.,.4,.9,.2]
        self.assertAlmostEqual(sum(x*y for x,y in zip(t,a)),sum(t)*abar(t,a))
    def test_20_advantage_constant_parity(self):
        t=[.1,.7,.9];a=[.6]*3;r=[.2,.6,.3]
        self.assertAlmostEqual(soft(t,a,r),soft(t,[abar(t,a)]*3,r))
    def test_21_zero_gate_and_zero_trust(self):
        self.assertEqual(soft([.2,.6],[0.,0.],[2.,4.]),0)
        self.assertEqual(abar([0.,0.],[.3,.5]),0)
    def test_22_routing_algebra(self):
        h=[1.,2.];k=[.1,.2];e=[.5,.9];q=[.2,.7];m=[.45,.45]
        full=objectives(h,k,e,q,q);edge=objectives(h,k,e,q,m);align=objectives(h,k,e,m,q)
        self.assertEqual(full[0],edge[0]);self.assertEqual(full[1],align[1])
    def test_23_h0_noadv_contract(self):
        self.assertEqual(T['H0']['alpha'],0);self.assertEqual(T['H0']['advantage_mode'],'ADAPTIVE')
        self.assertEqual(T['NOADV']['advantage_mode'],'ONE');self.assertEqual(T['NOADV']['beta'],.1)
    def test_24_grid_and_roi(self):
        grid=json.loads((ROOT/'planning/shift_grid.json').read_text())['shifts']
        self.assertEqual(len(grid),49);self.assertEqual(grid[0]['id'],'D000')
        self.assertTrue(all(math.isclose(math.hypot(r['dy'],r['dx']),r['radius_hr'],abs_tol=1e-12) for r in grid))
        self.assertTrue(all(r['primary_roi']=='48:-48_FIXED160' and r['auxiliary_roi']=='32:-32_FIXED192' for r in REG['stress_tasks']))

if __name__=='__main__':unittest.main(verbosity=2)
