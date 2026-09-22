"""Synthetic CPU fixtures only: no real data rebinding, caches or GPU execution."""
from copy import deepcopy
import random
import inspect
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.io import savemat
import torch

from fh12.common import sha256
from pcrepro.data import (CANONICAL_BANDS,NativeDataset,PairedBatchStream,SENSORS,
                         bind_datasets,canonical_indices,validate_manifest)


def make_manifest(tmp_path,sensor='GF2',rr_count=3,fr_count=20,train_count=8,larger=False):
    bands,dn=SENSORS[sensor];rng=np.random.default_rng(934)
    manifest=dict(schema='PCREPRO_DATA_v1',dataset=sensor,bands=bands,max_dn=dn,
        band_order=list(CANONICAL_BANDS[sensor]),units='DN',splits={})
    for split,count in (('train',train_count),('val',2),('rr',rr_count),('fr',fr_count)):
        if sensor=='WV2' and split in ('train','val'):continue
        low=18 if larger and split=='train' else 16
        ms=rng.uniform(dn*.15,dn*.65,(count,bands,low,low)).astype(np.float32)
        lms=np.repeat(np.repeat(ms,4,-1),4,-2)
        pan=rng.uniform(dn*.1,dn*.8,(count,1,low*4,low*4)).astype(np.float64)
        gt=(lms+np.asarray(pan*.02,dtype=np.float32)).astype(np.float32)
        arrays=dict(ms=ms,lms=lms,pan=pan,**({} if split=='fr' else {'gt':gt}))
        ids=[f'{sensor}_{split}_{i:02d}' for i in range(count)]
        spec=dict(split=split,count=count,scene_ids=ids,keys={k:k for k in arrays},
                  shapes={k:list(v.shape[1:]) for k,v in arrays.items()})
        if split=='fr' and sensor!='WV2':
            spec.update(format='mat_list',layout='HWC',scenes=[],
                paper_identity=dict(origin='PANCOLLECTION_PAPER_MAT'))
            for i,scene_id in enumerate(ids):
                path=tmp_path/f'{sensor}_{split}_{i}.mat'
                savemat(path,{k:v[i].transpose(1,2,0) for k,v in arrays.items()})
                spec['scenes'].append(dict(scene_id=scene_id,path=str(path),sha256=sha256(path)))
        else:
            path=tmp_path/f'{sensor}_{split}.h5'
            with h5py.File(path,'w') as store:
                for key,value in arrays.items():store[key]=value
            spec.update(format='h5',layout='NCHW',path=str(path),sha256=sha256(path),
                        origin='ORIGINAL_PANCOLLECTION')
        manifest['splits'][split]=spec
    return manifest


def raises(error,match='.*'):
    return unittest.TestCase().assertRaisesRegex(error,match)


def test_explicit_sensor_units_and_band_binding(tmp_path,sensor):
    binding=make_manifest(tmp_path,sensor)
    validated=validate_manifest(binding,tmp_path)
    data=NativeDataset(validated,'rr',tmp_path);raw=data.raw(0);row=data[0]
    assert row['ms'].shape[0]==SENSORS[sensor][0]
    assert torch.equal(row['pan'],row['pan_dn'].float().mul(2/SENSORS[sensor][1]).sub(1))
    assert np.array_equal(row['pan_dn'].numpy(),raw['pan'])
    assert validated['binding_status']==('PAPERSET_IDENTITY_UNVERIFIED' if sensor=='WV2' else 'READY')
    changed=deepcopy(binding);changed['max_dn']=4095
    with raises(ValueError,match='Sensor bands'):validate_manifest(changed,tmp_path)
    changed=deepcopy(binding);changed['band_order'][0]='UNKNOWN'
    with raises(ValueError,match='band order'):validate_manifest(changed,tmp_path)
    data.close()


def test_original_qb_only_blocks_qb_independently(tmp_path):
    qb=make_manifest(tmp_path,'QB');gf2=make_manifest(tmp_path,'GF2')
    qb['splits']['train']['origin']='MSFIX_REPAIRED'
    result=bind_datasets({'datasets':{'QB':qb,'GF2':gf2}},tmp_path)
    assert result['QB']['status']=='BLOCKED_DATASET'
    assert result['GF2']['status']=='READY'
    qb['splits']['train']['origin']='ORIGINAL_PANCOLLECTION'
    qb['splits']['train']['path']='dishonest_msfix.h5'
    with raises(ValueError,match='msfix'):validate_manifest(qb,tmp_path,verify_files=False)


def test_population_sha_split_and_sidecar_fail_closed(tmp_path):
    binding=make_manifest(tmp_path)
    changes=[('count',19),('scene_ids',['duplicate']*20)]
    for key,value in changes:
        changed=deepcopy(binding);changed['splits']['fr'][key]=value
        with raises(ValueError):validate_manifest(changed,tmp_path)
    changed=deepcopy(binding);changed['splits']['val']['sha256']='0'*64
    with raises(ValueError,match='SHA'):validate_manifest(changed,tmp_path)
    changed=deepcopy(binding);changed['splits']['train']['keys']['lpan']='lp'
    with raises(ValueError,match='sidecar'):validate_manifest(changed,tmp_path)
    changed=deepcopy(binding);changed['splits']['val']['split']='train'
    with raises(ValueError,match='split identity'):validate_manifest(changed,tmp_path)
    changed=deepcopy(binding);changed['splits']['val']['shapes']['pan']=[1,63,64]
    with raises(ValueError,match='shapes'):validate_manifest(changed,tmp_path)


def test_fr_h5_cannot_silently_replace_paper_set(tmp_path):
    binding=make_manifest(tmp_path,'WV2')
    binding['dataset']='WV3'
    training=make_manifest(tmp_path,'WV3')
    binding['splits'].update({s:training['splits'][s] for s in ('train','val')})
    with raises(ValueError,match='FR requires'):validate_manifest(binding,tmp_path)


def test_wv3_rr20_with_nineteen_mat_matches_keeps_twentieth(tmp_path):
    binding=make_manifest(tmp_path,'WV3',rr_count=20)
    spec=binding['splits']['rr'];references=[]
    with h5py.File(spec['path'],'r') as store:
        for i in range(19):
            path=tmp_path/f'rr_mat_match_{i}.mat'
            savemat(path,{k:store[k][i].transpose(1,2,0) for k in spec['keys']})
            references.append(dict(source_index=i,scene_id=spec['scene_ids'][i],path=str(path),sha256=sha256(path)))
    spec['paper_identity']=dict(mat_references=references)
    checked=validate_manifest(binding,tmp_path);identity=checked['splits']['rr']['identity_validation']
    assert identity['status']=='WV3_RR20_WITH_19_MATCHED_MATS'
    assert identity['declared_scene_count']==20 and identity['matched_scene_count']==19
    assert identity['discarded_samples']==0 and len(NativeDataset(checked,'rr',tmp_path))==20
    references[0]['scene_id']=spec['scene_ids'][1]
    with raises(ValueError,match='scene ID'):validate_manifest(binding,tmp_path)


def test_pixel_correspondence_rejects_wrong_native_pan(tmp_path):
    binding=make_manifest(tmp_path,'WV2',rr_count=1,fr_count=1)
    spec=binding['splits']['fr'];path=tmp_path/'wrong.mat'
    with h5py.File(spec['path'],'r') as store:
        arrays={k:store[k][0].transpose(1,2,0) for k in spec['keys']}
    arrays['pan']=arrays['pan']+1
    savemat(path,arrays)
    spec['paper_identity']=dict(mat_references=[dict(source_index=0,scene_id=spec['scene_ids'][0],
        path=str(path),sha256=sha256(path))])
    with raises(ValueError,match='not identical'):validate_manifest(binding,tmp_path)


def test_paired_crop_flip_rotation_and_raw_dn_share_same_view(tmp_path):
    binding=validate_manifest(make_manifest(tmp_path,larger=True),tmp_path)
    data=NativeDataset(binding,'train',tmp_path);native=data.raw(2)
    row=data[(2,1,1,3,1,2)]
    for key in ('pan','ms','gt','lms'):
        factor=1 if key=='ms' else 4
        expected=native[key][:,factor:17*factor,2*factor:18*factor][:,::-1,::-1]
        expected=np.rot90(expected,3,axes=(1,2)).copy()
        assert torch.equal(row[key],torch.from_numpy(expected.astype(np.float32)).mul(2/data.max_pixel).sub(1))
        if key=='pan':assert np.array_equal(row['pan_dn'],expected)
    assert row['pan'].shape[-2:]==(64,64) and row['ms'].shape[-2:]==(16,16)
    with raises(ValueError,match='without resize'):data[(2,0,0,0,5,5)]
    with raises(ValueError,match='native'):NativeDataset(binding,'val',tmp_path)[(0,1,0,0,0,0)]


def test_run_owned_stream_resumes_exact_next_batch_without_global_rng(tmp_path):
    binding=validate_manifest(make_manifest(tmp_path,train_count=16),tmp_path)
    data=NativeDataset(binding,'train',tmp_path)
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    first=PairedBatchStream(data,4,2025);first.advance();state=first.state_dict()
    resumed=PairedBatchStream(data,4,2025);resumed.load_state_dict(state)
    assert first.remaining_batches()==resumed.remaining_batches()
    assert first.completed_updates==1 and all(y==x==0 for batch in first.remaining_batches() for _,_,_,_,y,x in batch)
    assert first.remaining_batches()!=PairedBatchStream(data,4,2026).remaining_batches()
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    all_meta=sum((first.batch_at(i) for i in range(first.count)),[])
    assert {r[1] for r in all_meta}=={0,1} and {r[2] for r in all_meta}=={0,1}
    with raises(ValueError,match='skip'):first.new_epoch()
    while first.cursor<first.count:first.advance()
    old=first.batch_at(0);first.new_epoch()
    assert first.completed_updates==4 and first.batch_at(0)!=old
    altered=dict(state,dataset_manifest_sha256='0'*64)
    with raises(ValueError,match='identity'):resumed.load_state_dict(altered)


def test_band_permutation_is_explicit_physical_order():
    assert canonical_indices('QB',['NIR','Red','Green','Blue'])==(3,2,1,0)
    with raises(ValueError):canonical_indices('WV2',['B','G','R','NIR'])


def test_manifest_revalidation_is_idempotent_and_interruptible(tmp_path):
    binding=make_manifest(tmp_path)
    first=validate_manifest(binding,tmp_path)
    assert validate_manifest(first,tmp_path)==first
    with raises(InterruptedError):validate_manifest(binding,tmp_path,stopcheck=lambda:True)
    calls=[0]
    def stop():calls[0]+=1;return calls[0]==5
    with raises(InterruptedError):validate_manifest(binding,tmp_path,stopcheck=stop)
    assert calls[0]==5


def test_split_and_paper_scene_reuse_are_rejected(tmp_path):
    binding=make_manifest(tmp_path)
    altered=deepcopy(binding);altered['splits']['val']=deepcopy(altered['splits']['train'])
    altered['splits']['val']['split']='val'
    with raises(ValueError,match='disjoint'):validate_manifest(altered,tmp_path)
    altered=deepcopy(binding);altered['splits']['fr']['scenes'][1]['path']=altered['splits']['fr']['scenes'][0]['path']
    with raises(ValueError,match='Duplicate'):validate_manifest(altered,tmp_path)


class DataTests(unittest.TestCase):
    pass


def _install_tests():
    for name,fn in list(globals().items()):
        if not name.startswith('test_') or not callable(fn):continue
        parameters=inspect.signature(fn).parameters
        for sensor in (list(SENSORS) if 'sensor' in parameters else [None]):
            def run(self,fn=fn,sensor=sensor,parameters=parameters):
                with tempfile.TemporaryDirectory() as directory:
                    kwargs={}
                    if 'tmp_path' in parameters:kwargs['tmp_path']=Path(directory)
                    if sensor is not None:kwargs['sensor']=sensor
                    fn(**kwargs)
            setattr(DataTests,name+('_'+sensor if sensor else ''),run)


_install_tests()
