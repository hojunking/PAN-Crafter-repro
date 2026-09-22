import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fh12.source_documents import pinned_document
from gfp40.plan import *

class PlanTests(unittest.TestCase):
    def test_counts_updates_servers_and_no_new_teachers(self):
        self.assertEqual(validate_registry(),dict(cases=83,primary=74,optional=9,new_teachers=0))
        self.assertEqual([len(cases_for(s)) for s in SERVERS],[19,31,33])
        self.assertEqual(sum(c.updates for c in PRIMARY_CASES),2440000)
        self.assertEqual(sum(c.updates for c in CASES),2860000)
    def test_all_configs_roundtrip_strict_invariants(self):
        for c in CASES:
            cfg=json.loads(json.dumps(build_config(c)))
            self.assertEqual(validate_config(cfg),c)
            cfg['batch_size']=24
            with self.assertRaises(ValueError):validate_config(cfg)
    def test_confirmation_concrete_fixed_without_external_dependency(self):
        for f in ('LOW','HIGH','OTHER','P075','MSTAR'):
            with self.assertRaises(ValueError):build_config('A12',f)
        cfg=build_config('A12')
        with self.assertRaises(ValueError):validate_config(cfg,require_bound=True)
        self.assertEqual(cfg['gfp40']['family'],'G025')
        self.assertEqual(case_for('A12').depends_on,('A11',))
        self.assertIn('_G025_FIXED_',case_for('A12').run_id)
        self.assertEqual(cfg['gfp40']['execution_policy_sha256'],EXECUTION_POLICY_SHA256)
        self.assertNotIn('selection_lock_path',cfg['gfp40'])
        self.assertEqual(case_for('A12').source_calibration_family,'MSTAR')
    def test_grids_exclude_curves_and_keep50500(self):
        self.assertEqual(len(grid_steps(60000)),20)
        self.assertNotIn(20000,grid_steps(60000));self.assertNotIn(40000,grid_steps(60000))
        self.assertIn(20000,fullstate_steps(60000));self.assertIn(40000,fullstate_steps(60000))
        self.assertIn(50500,grid_steps(100000));self.assertNotIn(50000,grid_steps(100000))
    def test_groups_atomic_dag_and_schedule_seed_separation(self):
        self.assertEqual(len(block_for('C_SCHEDULE_1').cases),6)
        self.assertEqual(len(block_for('B_G050_TWO_PARENTS').cases),4)
        self.assertEqual(case_for('C13').calibration_family,'G025')
        self.assertEqual(case_for('C25').calibration_family,'NATIVE')
        self.assertNotEqual(case_for('C13').model_seed,case_for('C25').model_seed)
        self.assertEqual(case_for('A07').parent_id,case_for('A02').parent_id)
        self.assertEqual(case_for('A07').stream_seed,case_for('A02').stream_seed)
    def test_archived_reader_never_bypasses_changed_original(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);name='research_log/0920/a.md';value=b'original';h=hashlib.sha256(value).hexdigest()
            archive=root/'research_log/past/0920/a.md';archive.parent.mkdir(parents=True);archive.write_bytes(value)
            self.assertEqual(pinned_document(root,name,h),archive)
            orig=root/name;orig.parent.mkdir(parents=True);orig.write_bytes(b'changed')
            with self.assertRaises(ValueError):pinned_document(root,name,h)
            with self.assertRaises(ValueError):pinned_document(root,'../a.md',h)

if __name__=='__main__':unittest.main()
