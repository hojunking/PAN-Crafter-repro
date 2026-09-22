"""Synthetic CPU checks of the production gain/LP/AXIS16/stream paths."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from fh12.calibration import CONSTANT_ALIGNER_Q, axis16_q
from fh12.data import RECIPE, canonical_sha, sha256_file
from g20.data import G20Dataset
from g20.plan import sensor_spec
from gfp40.augmentation import (gaussian_detail_view, prepare_augmentation,
                                verify_augmentation_cache, VIEW_RECIPE)
from gfp40.calibration import (compute_gamma_shard, weighted_median_arrays,
    weighted_median_tokens, verify_native_slice, _load_shard, _native_paths,
    _validate_b20_reuse, runtime_identity)
from gfp40.data import GFP40Dataset
from gfp40.plan import FAMILIES
from gfp40.stream import PairedBatchStream, gamma_uniforms, map_uniforms
from pa.warp import warp_pan
from tools.repair_lpan import make_lpan


def synthetic_dataset(folder, count=17):
    folder=Path(folder); rng=np.random.default_rng(487)
    pan=rng.uniform(0,1023,(count,1,64,64)).astype(np.float64)
    source,lp=folder/'train.h5',folder/'native_lp.h5'
    with h5py.File(source,'w') as dst:
        dst['pan']=pan
        dst['gt']=np.zeros((count,4,64,64),np.float32)
        dst['lms']=np.zeros((count,4,64,64),np.float32)
        dst['ms']=np.zeros((count,4,16,16),np.float32)
    with h5py.File(lp,'w') as dst:
        dst['lpan']=make_lpan(pan).astype(np.float32)
        dst.attrs['source_sha256']=sha256_file(source)
        dst.attrs['recipe_sha256']=canonical_sha(RECIPE)
    spec=replace(sensor_spec('GF2'),band_order=('blue','green','red','nir'))
    return G20Dataset(source,lp,spec=spec,split='train'),pan


class ConstantTeacher(torch.nn.Module):
    bands=4
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
    def predict_delta(self,pan,base):return pan.new_zeros((len(pan),2))+self.anchor
    def forward(self,pan,ms,lp):return {'y':pan.expand(-1,4,-1,-1)+self.anchor}


def b20_fixture(folder,native,source):
    """Synthetic old-format evidence; does not import the archived B20 package."""
    folder=Path(folder);root=Path(__file__).resolve().parents[1];n=len(native)
    arrays=[gaussian_detail_view(source,gamma) for gamma in (.75,1.,1.25)]
    cache=folder/'b20_views.h5'
    with h5py.File(cache,'w') as dst:
        dst['pan']=np.stack(arrays,axis=1).astype(np.float32)
        dst['lpan']=np.stack([make_lpan(x) for x in arrays],axis=1).astype(np.float32)
    aug=dict(schema='GFB20_AUGMENTATION_v1',complete=True,native_bitwise=True,population=n,
        source=dict(source_sha256=sha256_file(native.raw_h5_path),
                    native_lp_sha256=sha256_file(native.lp_path),
                    recipe=dict(VIEW_RECIPE,schema='GFB20_PAN_DETAIL_GAIN_v1')),
        cache_path=str(cache),cache_sha256=sha256_file(cache))
    aug_path=folder/'b20_augmentation.json';aug_path.write_text(json.dumps(aug))
    q=np.full((n,4,3),CONSTANT_ALIGNER_Q,np.float32)
    native_q=q[:,:,1].copy();qpath=folder/'b20_q.npz'
    np.savez(qpath,q=q,native_online_q=native_q,calibration_indices=np.arange(n),
        per_radius_q=np.full((n,4,3,4),CONSTANT_ALIGNER_Q,np.float32),
        native_delta=np.zeros((n,4,3,2),np.float32))
    cal=dict(numerical_source_sha256={name:sha256_file(root/'gfb20'/name)
                                    for name in ('augmentation.py','data.py','calibration.py')})
    calpath=folder/'b20_cal.json';calpath.write_text(json.dumps(cal))
    names=('g20/model.py','qg40/model.py','pa/warp.py','fh12/calibration.py','fh12/data.py',
           'g20/data.py','tools/repair_lpan.py')
    files={name:sha256_file(root/name) for name in names}
    reference=dict(train_sha256=sha256_file(native.raw_h5_path),
                   train_lpan_sha256=sha256_file(native.lp_path))
    old=dict(schema='GFB20_MIXED_REFERENCE_v1',server='s5',native_reference=reference,
        runtime=runtime_identity('cpu'),q_ref=CONSTANT_ALIGNER_Q,
        q_cache_path=str(qpath),q_cache_sha256=sha256_file(qpath),
        calibration_path=str(calpath),calibration_sha256=sha256_file(calpath),
        augmentation_manifest_path=str(aug_path),augmentation_manifest_sha256=sha256_file(aug_path),
        native_slice_parity=verify_native_slice(native_q,native_q),
        source_identity=dict(files=files,content_sha256=canonical_sha(files)))
    path=folder/'b20_reference.json';path.write_text(json.dumps(old))
    return path,root,reference,native_q


class WeightedMedianTests(unittest.TestCase):
    def test_every_family_matches_explicit_tokens_both_float_dtypes(self):
        rng=np.random.default_rng(731)
        for dtype in (np.float32,np.float64):
            for spec in FAMILIES.values():
                for shape in ((2,len(spec['tokens']),3),(3,len(spec['tokens']),7)):
                    x=rng.random(shape).astype(dtype)
                    expected=float(np.median(np.repeat(x,spec['tokens'],axis=1)))
                    self.assertEqual(weighted_median_tokens(x,1,spec['tokens']),expected)

    def test_even_average_ties_zeros_and_odd_population(self):
        self.assertEqual(weighted_median_arrays([np.array([0,100],np.float32),
                                               np.array([10,110],np.float32)],[1,1]),55)
        self.assertEqual(weighted_median_arrays([np.array([-0.,0.,0.],np.float32)],[3]),0)
        self.assertEqual(weighted_median_arrays([np.array([0.,2.,9.])],[1]),2)
        self.assertEqual(weighted_median_arrays([np.array([0.,2.]),np.array([9.])],[2,1]),2)

    def test_reject_bad_weights_nonfinite_and_negative(self):
        for arrays,tokens in (([[1.]],[0]),([[1.]],[1.5]),([[np.nan]],[1]),([[-1.]],[1])):
            with self.assertRaises(ValueError):weighted_median_arrays(arrays,tokens)


class StreamTests(unittest.TestCase):
    def test_all_families_have_identical_source_geometry_uniform_prefix(self):
        prefixes=[PairedBatchStream(103,48,98101,family).initial_prefix() for family in FAMILIES]
        self.assertEqual(len({p['sample_view_uniform_sha256'] for p in prefixes}),1)
        self.assertEqual(len({p['uniform_sha256'] for p in prefixes}),1)
        self.assertGreater(len({p['sample_view_gamma_sha256'] for p in prefixes}),1)
        self.assertEqual(prefixes[0]['seed_mapping'],dict(source=398101,geometry=498101,workers=598101,gamma=698101))

    def test_integer_cdf_boundaries_and_all_distributions(self):
        for name,spec in FAMILIES.items():
            size=sum(spec['tokens']);uniforms=np.arange(size,dtype=np.int64)*(2**53//size)
            expected=np.repeat(np.arange(len(spec['tokens'])),spec['tokens'])
            np.testing.assert_array_equal(map_uniforms(uniforms,name),expected)
        with self.assertRaises(ValueError):map_uniforms(np.array([2**53]))

    def test_exact_resume_and_independent_rng(self):
        before=torch.get_rng_state().clone();np_before=np.random.get_state()
        stream=PairedBatchStream(103,48,123,'P075');stream.advance();state=stream.state_dict()
        resumed=PairedBatchStream(103,48,123,'P075');resumed.load_state_dict(state)
        self.assertEqual(stream.remaining_batches(),resumed.remaining_batches())
        stream.advance();resumed.advance();stream.new_epoch();resumed.new_epoch()
        self.assertEqual(stream.remaining_batches(),resumed.remaining_batches())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        np.testing.assert_array_equal(np_before[1],np.random.get_state()[1])
        with self.assertRaises(ValueError):PairedBatchStream(103,48,123,'G025').load_state_dict(state)
        state['gamma_counter']+=1
        with self.assertRaises(ValueError):resumed.load_state_dict(state)

    def test_uniform_counter_chunks_concatenate(self):
        whole=gamma_uniforms(601234,0,100)
        self.assertTrue(torch.equal(whole,torch.cat([gamma_uniforms(601234,0,37),gamma_uniforms(601234,37,63)])))


class ViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_gain_original_bypass_no_clipping_and_float64(self):
        x=np.zeros((1,1,64,64),np.float64);x[0,0,32,32]=1023
        self.assertIs(gaussian_detail_view(x,1.),x)
        high=gaussian_detail_view(x,1.5)
        self.assertGreater(high.max(),1023);self.assertLess(high.min(),0)
        np.testing.assert_array_equal(high[0,0,29:36,29:36],high[0,0,29:36,29:36][::-1,::-1])
        with self.assertRaises(ValueError):gaussian_detail_view(x.astype(np.float32),.75)
        with self.assertRaises(ValueError):gaussian_detail_view(x,.6)

    def test_shards_append_without_mutating_old_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,2);bank=Path(folder)/'bank'
            first=prepare_augmentation(native,bank);sha=sha256_file(first)
            original=json.loads(first.read_text());slice_sha={s['gamma']:s['cache_sha256'] for s in original['shards']}
            second=prepare_augmentation(native,bank,(.5,1.,1.5))
            self.assertNotEqual(first,second);self.assertEqual(sha256_file(first),sha)
            second_native=[s for s in json.loads(second.read_text())['shards'] if s['gamma']==1.][0]
            self.assertEqual(second_native['cache_sha256'],slice_sha[1.])
            verify_augmentation_cache(first,native,full=True)
            self.assertEqual(first,prepare_augmentation(native,bank))

    def test_family_native_index_and_uniform_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,2)
            for family in ('LOW','HIGH','G025'):
                spec=FAMILIES[family];bank=prepare_augmentation(native,Path(folder)/'bank',spec['gammas'])
                ctrl=GFP40Dataset(native,bank,arm='CTRL',family=family)
                mix=GFP40Dataset(native,bank,arm='MIX',family=family)
                native_id=spec['gammas'].index(1.);drawn=1-native_id
                row=ctrl[(0,1,drawn,12345)];base=native.get_view(0,1,augment=True)
                self.assertEqual(row[-1].tolist(),[0,1,1,1,native_id,drawn,12345])
                for got,wanted in zip(row[:5],base[:5]):self.assertTrue(torch.equal(got,wanted))
                row=mix[(0,1,drawn,12345)]
                self.assertEqual(row[-1][-3:].tolist(),[drawn,drawn,12345])
                for i in (0,1,2):self.assertTrue(torch.equal(row[i],base[i]))

    def test_lp_created_before_fixed_hv_rotation_and_gain_before_q_probe(self):
        class Recorder(ConstantTeacher):
            def __init__(self):super().__init__();self.inputs=[]
            def predict_delta(self,pan,base):self.inputs.append(pan.clone());return super().predict_delta(pan,base)
        with tempfile.TemporaryDirectory() as folder:
            native,source=synthetic_dataset(folder,1)
            bank=prepare_augmentation(native,Path(folder)/'bank',FAMILIES['HIGH']['gammas'])
            ds=GFP40Dataset(native,bank,arm='MIX',family='HIGH')
            row=ds.get_view(0,1,augment=True,gamma_id=1)
            gain=gaussian_detail_view(source,1.25)
            wanted=np.rot90(make_lpan(gain)[0,:,::-1,::-1],1,axes=(1,2)).copy().astype(np.float32)
            self.assertTrue(torch.equal(row[3],torch.from_numpy(wanted).mul_(2/1023).sub_(1)))
            teacher=Recorder();axis16_q(teacher,row[4][None],row[2][None])
            self.assertTrue(torch.equal(teacher.inputs[0],row[4][None]))
            self.assertTrue(torch.equal(teacher.inputs[1],warp_pan(row[4][None],torch.tensor([[.25,0.]]))))

    def test_native_evaluation_cannot_receive_augmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,1);native.split='rr'
            ds=GFP40Dataset(native,family='HIGH')
            with self.assertRaisesRegex(ValueError,'remain native'):ds.get_view(0,gamma_id=1)
            with self.assertRaises(ValueError):prepare_augmentation(native,Path(folder)/'bank')

    def test_pan_deadline_and_corrupt_cache_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,1)
            with self.assertRaises(TimeoutError):prepare_augmentation(native,Path(folder)/'bank',deadline_utc='2000-01-01T00:00:00Z')
            path=prepare_augmentation(native,Path(folder)/'bank');data=json.loads(path.read_text())
            data['shards'][0]['cache_sha256']='0'*64
            with self.assertRaises(ValueError):verify_augmentation_cache(data,native)

    def test_full_q_shard_pixel_errors_and_exact_resume_teacher_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,17);bank=prepare_augmentation(native,Path(folder)/'bank')
            ds=GFP40Dataset(native,bank,arm='MIX');teacher=ConstantTeacher();teacher.train()
            indices=np.array([0,7,16]);q=np.full((17,4),CONSTANT_ALIGNER_Q,np.float32)
            path=compute_gamma_shard(teacher,ds,indices,q,1,Path(folder)/'cal',device='cpu',synthetic_test=True)
            manifest=json.loads(path.read_text());arrays=_load_shard(manifest,synthetic_test=True)
            np.testing.assert_array_equal(arrays['q'],q)
            wanted=np.stack([(ds.base(int(i))[4].expand(4,-1,-1)-ds.base(int(i))[0]).abs().mean(0).numpy() for i in indices])
            np.testing.assert_array_equal(arrays['errors'],wanted)
            self.assertTrue(teacher.training);self.assertTrue(teacher.anchor.requires_grad)
            with patch('gfp40.calibration.axis16_q',side_effect=AssertionError('must reuse')):
                self.assertEqual(path,compute_gamma_shard(teacher,ds,indices,q,1,Path(folder)/'cal',device='cpu',synthetic_test=True))
            with self.assertRaises(ValueError):_load_shard(manifest)

    def test_partial_q_deadline_retains_prefix_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,1);bank=prepare_augmentation(native,Path(folder)/'bank')
            ds=GFP40Dataset(native,bank,arm='MIX');q=np.full((1,4),CONSTANT_ALIGNER_Q,np.float32)
            kwargs=dict(device='cpu',synthetic_test=True)
            with patch('gfp40.calibration.check_deadline',side_effect=[None,None,TimeoutError('pause')]):
                with self.assertRaises(TimeoutError):compute_gamma_shard(ConstantTeacher(),ds,[0],q,0,Path(folder)/'cal',**kwargs)
            partial=Path(folder)/'cal/q_shards/g0750/progress.h5'
            with h5py.File(partial,'r+') as state:
                self.assertTrue(state['q_checksums'][0,0]);self.assertFalse(state['q_checksums'][0,1])
                state['online_q'][0,0]+=.01
            with self.assertRaisesRegex(ValueError,'chunk changed'):
                compute_gamma_shard(ConstantTeacher(),ds,[0],q,0,Path(folder)/'cal',**kwargs)

    def test_production_count_and_full_native_parity_are_mandatory(self):
        with tempfile.TemporaryDirectory() as folder:
            native,_=synthetic_dataset(folder,1);ds=GFP40Dataset(native)
            q=np.full((1,4),CONSTANT_ALIGNER_Q,np.float32)
            with self.assertRaisesRegex(ValueError,'train19809'):
                compute_gamma_shard(ConstantTeacher(),ds,[0],q,1,folder,device='cpu')
            changed=q.copy();changed[0,3]+=.01
            with self.assertRaisesRegex(ValueError,'parity failed'):verify_native_slice(changed,q)

    def test_reuse_requires_actual_b20_archive_and_bound_paths(self):
        with self.assertRaisesRegex(ValueError,'actual frozen source'):
            _validate_b20_reuse(None,None,None,None,None,None,'cpu')
        with self.assertRaisesRegex(ValueError,'binder-resolved'):
            _native_paths(dict(teacher_checkpoint='weights/file',q_cache_path='q/file'))

    def test_actual_b20_pan_import_verified_against_all_source_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            native,source=synthetic_dataset(folder,2)
            old,_,_,_=b20_fixture(folder,native,source)
            old=json.loads(old.read_text());aug=old['augmentation_manifest_path']
            path=prepare_augmentation(native,Path(folder)/'new',reuse_manifest=aug)
            bank=verify_augmentation_cache(path,native,full=True)
            self.assertTrue(all(s['reuse_provenance']['status']=='REUSED_EXTERNAL' for s in bank['shards']))
            metadata=json.loads(Path(aug).read_text())
            with h5py.File(metadata['cache_path'],'r+') as dst:dst['pan'][0,0,0,0,0]+=1
            metadata['cache_sha256']=sha256_file(metadata['cache_path'])
            Path(aug).write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'disagrees with original DN'):
                prepare_augmentation(native,Path(folder)/'bad',reuse_manifest=aug)

    def test_actual_b20_q_reuse_preserves_full_cache_after_current_online_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            native,source=synthetic_dataset(folder,17)
            path,root,reference,q=b20_fixture(folder,native,source)
            bank=prepare_augmentation(native,Path(folder)/'new')
            ds=GFP40Dataset(native,bank,arm='MIX')
            with patch('gfp40.calibration.select_calibration_indices',return_value=np.arange(17)):
                slices,receipt=_validate_b20_reuse(path,root,ConstantTeacher(),ds,reference,q,'cpu')
                self.assertEqual(set(slices),{.75,1.,1.25});self.assertEqual(len(receipt['current_parity']),12)
                np.testing.assert_array_equal(slices[1.]['online_q'],q)
                wrong=dict(reference,train_sha256='0'*64)
                with self.assertRaisesRegex(ValueError,'Teacher/reference/runtime'):
                    _validate_b20_reuse(path,root,ConstantTeacher(),ds,wrong,q,'cpu')


if __name__=='__main__':unittest.main()
