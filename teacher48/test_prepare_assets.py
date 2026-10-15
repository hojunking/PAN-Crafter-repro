"""Native preparation smoke: synthetic source scenes, never production data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from align.resample import interp23tap
from ta2.evaluation import file_sha
from .prepare_assets import (Raster, NativeScene, SCHEMA, LMS_RECIPE, lms_window,
                             crop_native, exact_gt_candidates, support_regions, intersects)


class NativeFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.file = self.root/'native.h5'
        rng = np.random.default_rng(73)
        self.pan = rng.integers(0, 1024, size=(1, 128, 128), dtype=np.uint16)
        self.ms = rng.integers(0, 1024, size=(4, 32, 32), dtype=np.uint16)
        self.lms = rng.random((4, 128, 128)) * 1050 - 10
        with h5py.File(self.file, 'w') as f:
            f['pan'] = self.pan
            f['ms'] = self.ms
            f['lms'] = self.lms
            f['pan_hwc'] = self.pan.transpose(1, 2, 0)
            f['pan_nchw'] = self.pan[None]
            f['pan_hw'] = self.pan[0]
        self.proof = self.root/'synthetic-provenance.json'
        self.proof.write_text(json.dumps({'kind': 'Synthetic test fixture, no real geography claim'}))
        proof = {'path': self.proof.name, 'sha256': file_sha(self.proof)}
        self.scene = dict(source_scene_id='synthetic-a', acquisition_id='synthetic-acq-a',
            native_pan_grid='synthetic-native-pan', source_identity_basis='Synthetic fixture identity',
            geometry_basis='Synthetic common FOV; no PAN shift or resize',
            native_source_evidence=proof, geometry_evidence=proof,
            role='native_training_source', radiometry='native_DN', ratio=4,
            pan_origin_native_yx=[16,32], lms_mode='provided_global',
            lms_support={'radius_native_pan':0,'boundary':'bounded','basis':'Synthetic independent stored LMS'},
            **{key: self.ref(key) for key in ('pan', 'ms', 'lms')})
        self.spec = dict(schema=SCHEMA, dataset='GF2', scenes=[self.scene], heldout_crops=[],
                         split_basis='Synthetic independent acquisitions; no real-world claim')
        self.path = self.root/'scenes.json'
        self.write()

    def ref(self, key, layout='CHW', **kwargs):
        return dict(path=self.file.name, sha256=file_sha(self.file), key=key, layout=layout, **kwargs)

    def write(self):
        self.path.write_text(json.dumps(self.spec))


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = NativeFixture(self.tmp.name)

    def test_explicit_layouts_and_bounded_read_preserve_pixels(self):
        for key, layout, rest in [('pan','CHW',{}), ('pan_hw','HW',{}),
                                 ('pan_hwc','HWC',{}), ('pan_nchw','NCHW',{'index':0})]:
            with Raster(self.f.root, self.f.ref(key, layout, **rest)) as r:
                np.testing.assert_array_equal(r.read(4,8,16,20), self.f.pan[:,4:20,8:28])
        with self.assertRaisesRegex(ValueError, 'explicit index'):
            Raster(self.f.root, self.f.ref('pan_nchw','NCHW'))

    def test_global_interp23tap_halo_matches_whole_scene_at_all_edges(self):
        whole = interp23tap(torch.from_numpy(self.f.ms.astype(np.float64))[None],4)[0].numpy()
        with Raster(self.f.root, self.f.ref('ms')) as ms:
            for crop in ([0,0,64,64],[64,64,64,64],[28,44,48,32],[0,112,16,16]):
                y,x,h,w=crop
                np.testing.assert_allclose(lms_window(ms,crop),whole[:,y:y+h,x:x+w],rtol=0,atol=1e-10)
        # This specifically guards against independent periodic patch interpolation.
        local = interp23tap(torch.from_numpy(self.f.ms[:,:16,:16].astype(np.float64))[None],4)[0].numpy()
        self.assertGreater(float(np.abs(local-whole[:,:64,:64]).max()),1.)

    def test_native_crop_preserves_pan_ms_lms_no_shift_or_clip(self):
        with NativeScene(self.f.root,self.f.scene,'GF2') as scene:
            arrays=scene.crop([16,32,64,64])
            np.testing.assert_array_equal(arrays['pan'],self.f.pan[:,16:80,32:96])
            np.testing.assert_array_equal(arrays['ms'],self.f.ms[:,4:20,8:24])
            np.testing.assert_array_equal(arrays['lms'],self.f.lms[:,16:80,32:96])
        self.assertEqual(arrays['lms'].dtype,np.float64)

    def test_explicit_generated_lms_mode_does_not_touch_pan(self):
        self.f.scene['lms_mode']=LMS_RECIPE
        self.f.scene.pop('lms')
        with NativeScene(self.f.root,self.f.scene,'GF2') as scene:
            arrays=scene.crop([0,0,64,64])
        np.testing.assert_array_equal(arrays['pan'],self.f.pan[:,:64,:64])
        self.assertEqual(arrays['lms'].shape,(4,64,64))

    def test_crop_staging_records_real_coordinates_and_excludes_declared_holdout(self):
        self.f.spec['heldout_crops']=[{'acquisition_id':'synthetic-acq-a',
                                     'source_crop_native_pan_yxhw':[16,32,64,64]}]
        self.f.write()
        before=file_sha(self.f.file)
        out=crop_native(self.f.path,self.f.root/'out',patch_hw=[64,64],stride_hw=[64,64],margin=0)
        self.assertEqual(out['status'],'STAGED_NOT_ADMITTED')
        self.assertEqual(len(out['samples']['fr_train']),3)
        self.assertEqual(len(out['declared_heldout_exclusions']),1)
        first=out['samples']['fr_train'][0]
        self.assertEqual(first['source_crop_native_pan_yxhw'],[16,96,64,64])
        self.assertEqual(first['source_scene_id'],'synthetic-a')
        self.assertEqual(file_sha(self.f.file),before)
        self.assertIn('RR_SOURCE_MAPPING_AND_GRID_PHASE',out['pending'])
        with self.assertRaises(FileExistsError):
            crop_native(self.f.path,self.f.root/'out',patch_hw=[64,64],stride_hw=[64,64],margin=0)

    def test_nonzero_split_buffer_is_enforced(self):
        self.f.spec['heldout_crops']=[{'acquisition_id':'synthetic-acq-a',
                                     'source_crop_native_pan_yxhw':[16,32,64,64]}]
        self.f.write()
        with self.assertRaisesRegex(ValueError,'No non-heldout'):
            crop_native(self.f.path,self.f.root/'out',patch_hw=[64,64],stride_hw=[64,64],margin=1)
        self.assertTrue((self.f.root/'out/FAILED.json').is_file())
        self.assertFalse((self.f.root/'out/preparation.json').exists())

    def test_bad_role_geometry_evidence_and_grid_rejected(self):
        for edit in ({'role':'benchmark'}, {'ratio':1}, {'lms_mode':'bicubic_auto'},
                     {'source_identity_basis':''}):
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                NativeScene(self.f.root,{**self.f.scene,**edit},'GF2')
        bad=copy.deepcopy(self.f.scene)
        bad['geometry_evidence']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'evidence'):
            NativeScene(self.f.root,bad,'GF2')
        with NativeScene(self.f.root,self.f.scene,'GF2') as scene:
            with self.assertRaisesRegex(ValueError,'grid'):
                scene.crop([1,0,64,64])

    def test_sha_and_mutation_detection(self):
        bad=self.f.ref('pan');bad['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'):
            Raster(self.f.root,bad)
        with Raster(self.f.root,self.f.ref('pan')) as r:
            import os
            st=self.f.file.stat()
            os.utime(self.f.file,ns=(st.st_atime_ns,st.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'changed'):
                r.read(0,0,4,4)

    def test_external_and_virtual_h5_dependencies_rejected(self):
        alias=self.f.root/'alias.h5'
        with h5py.File(alias,'w') as f:
            f['pan']=h5py.ExternalLink(str(self.f.file),'pan')
        ref=dict(path=alias.name,sha256=file_sha(alias),key='pan',layout='CHW')
        with self.assertRaisesRegex(ValueError,'External/virtual'):
            Raster(self.f.root,ref)
        virtual=self.f.root/'virtual.h5'
        layout=h5py.VirtualLayout(shape=self.f.pan.shape,dtype=np.uint16)
        layout[:]=h5py.VirtualSource(str(self.f.file),'pan',shape=self.f.pan.shape)
        with h5py.File(virtual,'w',libver='latest') as f:
            f.create_virtual_dataset('pan',layout)
        ref.update(path=virtual.name,sha256=file_sha(virtual))
        with self.assertRaisesRegex(ValueError,'External/virtual'):
            Raster(self.f.root,ref)

    def test_lms_support_exclusion_includes_adjacent_and_wrapped_heldout(self):
        # Crop does not intersect holdout, but interpolation accesses its pixels.
        crop=[64,64,64,64]; held=[0,0,64,256]
        self.assertFalse(intersects(crop,held,0))
        supports=support_regions(crop,64,(256,256),'wrap')
        self.assertTrue(any(intersects(s,held,0) for s in supports))
        supports=support_regions([0,0,64,64],64,(256,256),'wrap')
        self.assertTrue(any(intersects(s,[240,240,16,16],0) for s in supports))
        self.f.scene['lms_mode']=LMS_RECIPE
        self.f.spec['heldout_crops']=[{'acquisition_id':'synthetic-acq-a',
                                     'source_crop_native_pan_yxhw':[16,32,4,4]}]
        self.f.write()
        with self.assertRaisesRegex(ValueError,'No non-heldout'):
            crop_native(self.f.path,self.f.root/'out',patch_hw=[64,64],stride_hw=[64,64],margin=0)

    def test_exact_gt_match_recovers_all_augmentation_transforms(self):
        source=self.f.ms
        crop=source[:,5:12,9:17]
        for flip in (False,True):
            for rot in range(4):
                gt=np.rot90(crop[...,::-1] if flip else crop,rot,axes=(-2,-1))
                out=exact_gt_candidates(gt,source)
                self.assertEqual(out['status'],'EXACT_GT_ONLY_CANDIDATE')
                self.assertFalse(out['accepted'])
                self.assertEqual(out['candidates'],[dict(native_ms_crop_yxhw=[5,9,7,8],
                    preparation_augmentation={'hflip':flip,'vflip':False,'rot90':rot})])

    def test_no_near_match_promotion_and_duplicate_or_flat_ambiguity(self):
        source=self.f.ms.astype(np.float64)
        gt=source[:,5:12,9:17].copy();gt[1,2,2]+=.00001
        self.assertEqual(exact_gt_candidates(gt,source)['status'],'UNMATCHED')
        source[:,20:27,20:28]=source[:,5:12,9:17]
        self.assertEqual(exact_gt_candidates(source[:,5:12,9:17],source)['status'],'AMBIGUOUS')
        flat=exact_gt_candidates(np.ones((4,4,4)),np.ones((4,8,8)),max_candidates=2)
        self.assertEqual(flat['status'],'AMBIGUOUS')
        self.assertTrue(flat['truncated'])


if __name__=='__main__':
    unittest.main()
