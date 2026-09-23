"""Deterministic GF2 seed collision pre-registration; no real campaign writes."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablr2.common import atomic_json,camp,read_json
from ablr2.seeds import (authored_cases,scan_used_seeds,resolve_gf2_boot,
    validate_resolution,validate_resolved_boot_case,validate_boot_admission,resolved_boot_seeds)


class SeedTests(unittest.TestCase):
    def started(self,root,name,seed):
        wd=Path(root)/'work_dir'/name
        atomic_json(wd/'meta/config.resolved.yaml',{'seed':seed,'teacher_seed':999})
        atomic_json(wd/'meta/training_start_manifest.json',{'run_id':name,'started_at_utc':'fixture'})
        return wd

    def test_queued_definitions_do_not_count_as_actual_used_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            atomic_json(Path(root)/'work_dir/planned/meta/config.resolved.yaml',{'seed':981001})
            atomic_json(Path(root)/'work_dir/queue.json',{'seed':991001})
            self.assertEqual(scan_used_seeds(root),[])
            cases,receipt=resolve_gf2_boot(root)
            self.assertEqual(cases,authored_cases())
            self.assertTrue(all(r['collision_counter']==0 for r in receipt['mapping']))

    def test_collisions_resolve_once_before_results_and_preserve_pairing(self):
        with tempfile.TemporaryDirectory() as root:
            self.started(root,'ACTUAL_OLD_TEACHER',981001);self.started(root,'ACTUAL_OLD_STUDENT',991003)
            cases,receipt=resolve_gf2_boot(root)
            self.assertEqual(len(cases),100);self.assertEqual(len({c.run_id for c in cases}),100)
            self.assertEqual(sum(r['collision_counter']>0 for r in receipt['mapping']),2)
            self.assertEqual(resolve_gf2_boot(root),(cases,receipt))
            self.assertNotEqual(resolved_boot_seeds(root,'P01')[0],981001)
            for sweep in (f'P{i:02}' for i in range(1,6)):
                rows=[c for c in cases if c.sweep==sweep]
                teachers={c.teacher_kind:c for c in rows if c.role=='T'}
                self.assertEqual(len({c.seed for c in teachers.values()}),1)
                self.assertEqual(len({c.seed for c in rows if c.role=='S'}),1)
                for c in rows:
                    self.assertEqual(validate_resolved_boot_case(c),c)
                    self.assertEqual(validate_resolved_boot_case(c,root),c)
                    if c.requires_teacher:
                        t=teachers[c.teacher_kind]
                        self.assertEqual((c.teacher_run_id,c.reference_id,c.teacher_seed),(t.run_id,t.reference_id,t.seed))
            self.assertEqual([r['seed'] for r in scan_used_seeds(root)],[981001,991003])

    def test_repeated_identical_histories_give_identical_seed_mapping(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            for root in (a,b):self.started(root,'OLD',981001)
            ca,ra=resolve_gf2_boot(a);cb,rb=resolve_gf2_boot(b)
            self.assertEqual(ca,cb);self.assertEqual(ra['mapping'],rb['mapping'])

    def test_missing_resolution_after_first_ablation_start_is_not_silent_remap(self):
        with tempfile.TemporaryDirectory() as root:
            atomic_json(camp(root,'s3')/'runs/ABLR2_STARTED/meta/training_start_manifest.json',{'seed':981001})
            with self.assertRaisesRegex(ValueError,'pre-result'):resolve_gf2_boot(root)

    def test_mapping_and_role_or_reference_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cases,receipt=resolve_gf2_boot(root)
            forged=replace(next(c for c in cases if c.case_id=='C17'),teacher_kind='TPLUS')
            with self.assertRaises(ValueError):validate_resolved_boot_case(forged)
            receipt['mapping'][0]['resolved_seed']+=100
            with self.assertRaises(ValueError):validate_resolution(receipt)

    def test_new_external_collision_blocks_without_new_seed_search(self):
        with tempfile.TemporaryDirectory() as root:
            cases,receipt=resolve_gf2_boot(root);case=cases[0]
            self.assertEqual(validate_boot_admission(root,case),case)
            self.started(root,'EXTERNAL_STARTED_LATER',case.seed)
            with self.assertRaisesRegex(ValueError,'New external seed collision'):validate_boot_admission(root,case)
            self.assertEqual(resolve_gf2_boot(root)[1],receipt)

    def test_unverifiable_started_seed_and_falsely_marked_used_registry_block(self):
        with tempfile.TemporaryDirectory() as root:
            atomic_json(Path(root)/'work_dir/unknown/meta/training_start_manifest.json',{'run_id':'unknown'})
            with self.assertRaisesRegex(ValueError,'cannot be authenticated'):scan_used_seeds(root)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'used.json';atomic_json(path,{'schema':'ACTUAL_USED_TRAINING_SEEDS_v1',
                'runs':[{'seed':981001,'run_id':'queued','training_started':False}]})
            with self.assertRaisesRegex(ValueError,'Queued/planned'):scan_used_seeds(root,[path])

    def test_controller_build_lookup_and_extension_sync_keep_remapped_teacher_links(self):
        from ablr2 import controller as ctl
        from ablr2.plan import case_for
        from ablr2.extension import sync
        for collision in (981001,991001):
            with self.subTest(collision=collision),tempfile.TemporaryDirectory() as root, \
                    patch.object(ctl,'verify_sources'),patch.object(ctl,'verify_extension_sources'), \
                    patch.object(ctl,'_command',side_effect=AssertionError('No actual training')):
                self.started(root,'OLD_STARTED',collision);ctl.build(root,'s3')
                state=ctl._state(root,'s3');cases,receipt=resolve_gf2_boot(root)
                self.assertEqual([case_for(c.run_id,root) for c in cases],list(cases))
                registry_before=read_json(camp(root,'s3')/'cases.json')
                sync(root,'s3',state);ctl.build(root,'s3')
                self.assertEqual(read_json(camp(root,'s3')/'cases.json'),registry_before)
                group=[c for c in cases if c.sweep=='P01'];by_id={c.case_id:c for c in group}
                self.assertEqual(case_for(by_id['C03'].run_id,root).teacher_run_id,by_id['TPLUS'].run_id)
                self.assertEqual(case_for(by_id['C17'].run_id,root).teacher_run_id,by_id['TZERO'].run_id)


if __name__=='__main__':unittest.main()
