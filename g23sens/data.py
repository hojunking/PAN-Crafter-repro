"""Actual legacy Feeder methods with lazy I/O and an immutable paired stream.

The data transforms are not reimplemented: PanFeeder.__getitem__, augment and
np2tensor are called directly. Both flips are unconditional. Only rotation is
random. The complete stream isolates the legacy sampler/worker RNG algorithm
from validation, diagnostics and optimizer graphs, and records that distinction.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import feeders.feeder as legacy
from g23sens.assets import ROOT,object_sha,sha256,BAND_ORDER,validate_bindings


class _ArrayView:
    def __init__(self,owner,key,lp=False):self.owner,self.key,self.lp=owner,key,lp
    def __getitem__(self,index):return self.owner._array(self.key,index,self.lp).transpose(1,2,0)
    @property
    def shape(self):
        value=self.owner.binding['data'][self.owner.split_key+('_lp' if self.lp else '')]['arrays'][self.key]['shape']
        return (value[0],value[2],value[3],value[1])


@contextmanager
def _rotation(value):
    """Replay one registered draw inside the unchanged legacy method.

    Each DataLoader process is single-threaded for fetching. Replacing only the
    feeder module's local random reference never advances global diagnostic RNG.
    """
    original=legacy.random
    class Draw:
        @staticmethod
        def randint(low,high):
            if (low,high)!=(0,3):raise ValueError('Legacy rotation draw contract changed')
            return int(value)
    legacy.random=Draw
    try:yield
    finally:legacy.random=original


class NativeDataset(legacy.PanFeeder):
    sensor='WV3';bands=8;max_pixel=2047.
    def __init__(self,bindings,split,root=ROOT):
        if split not in ('train','val','rr','fr'):raise ValueError('Unknown native split')
        if isinstance(bindings,(str,Path)):
            import json
            bindings=json.loads(Path(bindings).read_text())
        self.binding=bindings;self.split_key=split;self.root=Path(root)
        data=bindings['data'][split];self.dataroot=data['path'];self._handles={}
        self.spec=(bindings.get('evaluation',{}).get(split+'_manifest') or
            dict(scene_ids=[f'{split}:native_h5_index:{i:03d}' for i in range(data['count'])],
                 identity_validation={'status':'NOT_PAPER_EVALUATION_SPLIT'}))
        self.scene_ids=list(self.spec['scene_ids']);self.manifest=dict(data,band_order=BAND_ORDER)
        self.manifest_sha256=object_sha(dict(native=data,lp=bindings['data'][split+'_lp'],scene_manifest=self.spec))
        self.split={'train':'train','val':'val','rr':'test_reduced','fr':'test_full'}[split]
        self.has_gt=split!='fr';self.crop=False;self.hflip=self.vflip=self.rot=(split=='train')
        self.return_meta=True;self._last_rot=0;self.crop_ratio=.75;self.ms_size=16
        # Skip only eager I/O and constructor's unrelated global random.seed(2025).
        # Workers are explicitly seeded as in the original DataLoader below.
        for key in ('gt','lms','ms','pan'):
            if key!='gt' or self.has_gt:setattr(self,key,_ArrayView(self,key))
        self.lpan=_ArrayView(self,'lpan',True)
        if split!='train':self.augment=False  # evaluator contract; legacy never calls it outside train.

    def __len__(self):return int(self.binding['data'][self.split_key]['count'])

    def _array(self,key,index,lp=False):
        if not 0<=int(index)<len(self):raise IndexError(index)
        path=self.binding['data'][self.split_key+('_lp' if lp else '')]['path']
        pid=os.getpid();identity=(pid,path)
        if identity not in self._handles:self._handles[identity]=h5py.File(path,'r')
        return np.array(self._handles[identity][key][int(index)])

    def raw(self,index):
        keys=['lms','ms','pan']+(['gt'] if self.has_gt else [])
        return {k:self._array(k,index) for k in keys}

    def __getitem__(self,index):
        if isinstance(index,(tuple,list,np.ndarray)):
            sample,rot,hflip,vflip=map(int,index)
            if self.split_key!='train' or rot not in range(4) or (hflip,vflip)!=(1,1):
                raise ValueError('Registered metadata violates actual legacy training augmentation')
            with _rotation(rot):value=legacy.PanFeeder.__getitem__(self,sample)
        else:value=legacy.PanFeeder.__getitem__(self,int(index))
        keys=(['gt'] if self.has_gt else [])+['lms','ms','lpan','pan','meta']
        return dict(zip(keys,value))

    def close(self):
        for handle in self._handles.values():handle.close()
        self._handles={}

    def __getstate__(self):
        value=self.__dict__.copy();value['_handles']={};return value

    def __del__(self):
        if hasattr(self,'_handles'):self.close()


def stream_schedule(count,batch_size,initial_torch_rng_state,*,workers=4,total_updates=50000,seed=None):
    """Pure CPU metadata; exact legacy RandomSampler+worker Python draws.

    Original DataLoader(generator=None) draws base_seed then RandomSampler's seed
    from global torch. Each epoch gets fresh workers; batches round-robin across
    workers, which seed Python random with base_seed+worker_id. No probabilities
    or extra flip draws are introduced. Validation never consumes this stream.
    """
    if count<batch_size or workers<1 or total_updates<1:raise ValueError('Invalid stream dimensions')
    if initial_torch_rng_state is None:raise ValueError('Post-legacy-U-constructor Torch RNG state is required')
    master=torch.Generator(device='cpu');master.set_state(initial_torch_rng_state.cpu())
    batches_per_epoch=count//batch_size;plan=np.empty((total_updates,batch_size,4),dtype=np.int32)
    written=0;epochs=[]
    while written<total_updates:
        base_seed=int(torch.empty((),dtype=torch.int64).random_(generator=master).item())
        sampler_seed=int(torch.empty((),dtype=torch.int64).random_(generator=master).item())
        generator=torch.Generator(device='cpu').manual_seed(sampler_seed)
        indices=torch.randperm(count,generator=generator).numpy()
        worker_rng=[random.Random(base_seed+i) for i in range(workers)]
        take=min(batches_per_epoch,total_updates-written)
        for j in range(take):
            block=plan[written+j];block[:,0]=indices[j*batch_size:(j+1)*batch_size]
            block[:,1]=[worker_rng[j%workers].randint(0,3) for _ in range(batch_size)]
            block[:,2:]=1
        epochs.append(dict(epoch=len(epochs),first_update=written+1,n_batches=take,
            worker_base_seed=base_seed,sampler_seed=sampler_seed,discarded_per_full_epoch=count%batch_size))
        written+=take
    manifest=dict(schema='G23SENS_FULL_PAIRED_STREAM_v1',seed=seed,count=count,batch_size=batch_size,
        workers=workers,total_updates=total_updates,drop_last=True,shuffle=True,
        semantics='legacy RandomSampler and fresh4worker Python rotation RNG; isolated from val/diagnostic RNG',
        exact_historical_whole_training_stream_claim=False,
        initial_torch_rng_sha256=hashlib.sha256(initial_torch_rng_state.cpu().numpy().tobytes()).hexdigest(),
        metadata_dtype='little-endian int32',metadata_columns=['index','actual_rotation','hflip','vflip'],
        metadata_sha256=hashlib.sha256(plan.astype('<i4',copy=False).tobytes()).hexdigest(),epochs=epochs,
        feeder_source_sha256=sha256(ROOT/'feeders/feeder.py'),torch_version=torch.__version__)
    manifest['manifest_sha256']=object_sha(manifest)
    return plan,manifest


class _RemainingBatches:
    def __init__(self,plan,cursor):self.plan,self.cursor=plan,int(cursor)
    def __iter__(self):
        for row in self.plan[self.cursor:]:yield [tuple(map(int,meta)) for meta in row]
    def __len__(self):return len(self.plan)-self.cursor


class BatchStream:
    """Cursor advances only after successful optimizer step; prefetch is not progress."""
    def __init__(self,dataset,batch_size=48,seed=1234,initial_torch_rng_state=None,workers=4,total_updates=50000):
        if dataset.split_key!='train':raise ValueError('Training stream requires native train')
        self.dataset,self.seed,self.workers=dataset,int(seed),int(workers)
        self.plan,self.manifest=stream_schedule(len(dataset),batch_size,initial_torch_rng_state,
            workers=workers,total_updates=total_updates,seed=int(seed))
        self.manifest['dataset_manifest_sha256']=dataset.manifest_sha256
        self.manifest['manifest_sha256']=object_sha({k:v for k,v in self.manifest.items() if k!='manifest_sha256'})
        self.cursor=0;self._iterator=None;self._loader=None;self._pending=None

    @property
    def completed_updates(self):return self.cursor
    @property
    def count(self):return len(self.dataset)
    def remaining_batches(self):return len(self.plan)-self.cursor

    def next_batch(self):
        if self._pending is not None:return self._pending
        if self.cursor>=len(self.plan):raise StopIteration
        if self._iterator is None:
            self._loader=DataLoader(self.dataset,batch_sampler=_RemainingBatches(self.plan,self.cursor),
                num_workers=self.workers,persistent_workers=False,generator=torch.Generator().manual_seed(0))
            self._iterator=iter(self._loader)
        self._pending=next(self._iterator)
        if not np.array_equal(self._pending['meta'].numpy(),self.plan[self.cursor]):
            raise ValueError('Actual legacy Feeder rotation/index differs from immutable stream')
        return self._pending

    def advance(self):
        if self._pending is None:raise ValueError('Cannot advance without delivered optimizer batch')
        self.cursor+=1;self._pending=None

    def state_dict(self):
        return dict(schema='G23SENS_STREAM_CURSOR_v1',manifest_sha256=self.manifest['manifest_sha256'],
            metadata_sha256=self.manifest['metadata_sha256'],cursor=self.cursor,completed_updates=self.cursor,
            consumed_prefix_sha256=hashlib.sha256(self.plan[:self.cursor].astype('<i4',copy=False).tobytes()).hexdigest())

    def load_state_dict(self,state):
        if state.get('schema')!='G23SENS_STREAM_CURSOR_v1' or state.get('manifest_sha256')!=self.manifest['manifest_sha256']:
            raise ValueError('Resume stream identity differs')
        cursor=state.get('cursor')
        if not isinstance(cursor,int) or not 0<=cursor<=len(self.plan) or state.get('completed_updates')!=cursor:
            raise ValueError('Resume stream cursor differs')
        if (state.get('metadata_sha256')!=self.manifest['metadata_sha256'] or state.get('consumed_prefix_sha256')!=
                hashlib.sha256(self.plan[:cursor].astype('<i4',copy=False).tobytes()).hexdigest()):
            raise ValueError('Resume stream consumed prefix differs')
        self.close();self.cursor=cursor

    def fixed_probe(self,batch_size=None):
        count=min(int(batch_size or self.plan.shape[1]),self.plan.shape[1])
        return default_collate([self.dataset[tuple(map(int,row))] for row in self.plan[0,:count]])

    def close(self):
        if self._iterator is not None and hasattr(self._iterator,'_shutdown_workers'):
            self._iterator._shutdown_workers()
        self._iterator=None;self._loader=None;self._pending=None

    def __del__(self):
        if hasattr(self,'_iterator'):self.close()


def load_evaluation_datasets(bindings,root=ROOT):
    return {split:NativeDataset(bindings,split,root) for split in ('rr','fr')}
