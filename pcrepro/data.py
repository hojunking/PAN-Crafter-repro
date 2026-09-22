"""Explicit sensor/split bindings and run-owned paired geometry; no LP sidecar.

Original QB train/val are mandatory. Paper-set claims require declared .mat
provenance or measured H5-to-.mat correspondence, never a filename heuristic.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import os
import re

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from fh12.common import ROOT, object_sha, read_json, resolved_path, sha256

SCHEMA = 'PCREPRO_DATA_v1'
SENSORS = {'WV3': (8, 2047), 'WV2': (8, 2047), 'QB': (4, 2047), 'GF2': (4, 1023)}
CANONICAL_BANDS = {s: (('C','B','G','Y','R','RE','NIR1','NIR2') if c == 8 else ('B','G','R','NIR'))
                   for s, (c, _) in SENSORS.items()}
ALIASES = {'COASTAL':'C','COASTALBLUE':'C','BLUE':'B','GREEN':'G','YELLOW':'Y',
           'RED':'R','REDEDGE':'RE','NIR':'NIR','NIR1':'NIR1','NIR2':'NIR2'}


def canonical_indices(dataset, order):
    labels = [ALIASES.get(str(v).upper().replace('_','').replace(' ',''),
                          str(v).upper().replace('_','').replace(' ','')) for v in order]
    expected = CANONICAL_BANDS[dataset]
    if len(labels) != len(expected) or set(labels) != set(expected):
        raise ValueError('Explicit physical band order is required for sensor MTF')
    return tuple(labels.index(v) for v in expected)


def _path(value, root):
    path = resolved_path(value, root).resolve()
    if not path.is_file(): raise FileNotFoundError('Explicit data asset missing: ' + str(path))
    return path


def _check_stop(stopcheck):
    if stopcheck is not None and stopcheck():raise InterruptedError('Safe data verification pause requested')


def _digest_file(item, root, stopcheck=None):
    path = _path(item['path'], root)
    _check_stop(stopcheck)
    if stopcheck is None:digest=sha256(path)
    else:
        hasher=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                _check_stop(stopcheck);hasher.update(block)
        digest=hasher.hexdigest()
    if digest != item.get('sha256'): raise ValueError('Data file SHA differs: ' + str(path))
    return path


def _layout(array, layout, pan=False):
    a = np.asarray(array)
    if pan and a.ndim == 2: a = a[None]
    elif layout == 'HWC': a = a.transpose(2, 0, 1)
    elif layout != 'CHW': raise ValueError('Scene layout must be explicitly CHW or HWC')
    if a.ndim != 3: raise ValueError('Each scene must have exactly three CHW dimensions')
    return a


def read_mat_scene(item, keys, layout, root=ROOT):
    source = loadmat(_path(item['path'], root))
    result = {}
    for key, variable in keys.items():
        if key not in ('gt','lms','ms','pan'): raise ValueError('Unexpected input key; LP sidecar is forbidden')
        if variable not in source: raise ValueError('Missing declared .mat variable: ' + variable)
        result[key] = _layout(source[variable], layout, pan=key == 'pan')
    return result


def _scene_arrays(spec, index, root):
    if spec['format'] == 'mat_list':
        return read_mat_scene(spec['scenes'][index], spec['keys'], spec['layout'], root)
    with h5py.File(_path(spec['path'], root), 'r') as store:
        layout = {'NCHW':'CHW','NHWC':'HWC'}[spec['layout']]
        return {key: _layout(store[variable][index], layout, pan=key == 'pan')
                for key, variable in spec['keys'].items()}


def _shape_check(rows, sensor, split, declared):
    bands, _ = SENSORS[sensor]
    required = {'pan','ms'} | ({'gt'} if split != 'fr' else {'lms'})
    if not required <= set(rows): raise ValueError('Missing required original split arrays')
    pan, ms = rows['pan'], rows['ms']
    if pan.ndim != 3 or pan.shape[0] != 1 or ms.ndim != 3 or ms.shape[0] != bands:
        raise ValueError('PAN/band count differs from sensor contract')
    if pan.shape[-2:] != (ms.shape[-2]*4, ms.shape[-1]*4):
        raise ValueError('Native PAN/MS ratio must be exactly four, without resizing')
    for key in ('gt','lms'):
        if key in rows and rows[key].shape != (bands, *pan.shape[-2:]):
            raise ValueError('GT/LMS must match native PAN spatial support and bands')
    if split == 'train' and min(ms.shape[-2:]) < 16:
        raise ValueError('Training crop cannot upscale a smaller patch')
    if split == 'val' and pan.shape[-2:] != (64,64):
        raise ValueError('Native validation must be the declared unaugmented64/16 patches')
    actual = {key:list(value.shape) for key,value in rows.items()}
    if actual != declared: raise ValueError('Actual scene shapes differ from explicit manifest')
    if any(not np.isfinite(value).all() for value in rows.values()):
        raise ValueError('Nonfinite native source data')


def _paper_identity(spec, split, sensor, root, verify_files, stopcheck=None):
    identity = spec.get('paper_identity') or {}
    if spec['format'] == 'mat_list' and identity.get('origin') == 'PANCOLLECTION_PAPER_MAT':
        exact_population=split!='fr' or spec['count']==20
        return dict(status='DECLARED_PAPER_MAT_SHA_BOUND' if exact_population else 'PAPERSET_IDENTITY_UNVERIFIED',
            verified=bool(verify_files and exact_population),
            declared_scene_count=spec['count'], matched_scene_count=spec['count'],
            basis='Explicit original paper .mat paths/IDs/SHA; no H5 substitution')
    references = identity.get('mat_references', [])
    if references:
        used = set()
        for item in references:
            _check_stop(stopcheck)
            index = item['source_index']
            if type(index) is not int or not 0 <= index < spec['count'] or index in used:
                raise ValueError('Duplicate or invalid H5-to-paper scene index')
            used.add(index)
            if item['scene_id'] != spec['scene_ids'][index]:
                raise ValueError('Paper/H5 correspondence scene ID mismatch')
            if verify_files:
                _digest_file(item, root,stopcheck)
                expected = read_mat_scene(item, item.get('keys',spec['keys']), item.get('layout','HWC'), root)
                actual = _scene_arrays(spec,index,root)
                if set(expected) != set(actual) or any(not np.array_equal(expected[k],actual[k]) for k in actual):
                    raise ValueError('H5 and declared paper .mat tensors are not identical')
        full = len(used) == spec['count']
        special = sensor == 'WV3' and split == 'rr' and spec['count'] == 20 and len(used) == 19
        if not full and not special: raise ValueError('Partial paper correspondence is not a complete paper set')
        return dict(status='VERIFIED_H5_PAPER_MAT_EQUALITY' if full else 'WV3_RR20_WITH_19_MATCHED_MATS',
            verified=full and verify_files and (split!='fr' or spec['count']==20),
            declared_scene_count=spec['count'], matched_scene_count=len(used),
            all_declared_samples_used=True, discarded_samples=0,
            caveat='' if full else '20 declared H5 scenes evaluated; only19 available .mat matches are proven')
    if split == 'fr' and sensor != 'WV2':
        raise ValueError('FR requires explicit paper .mat set or measured full H5/.mat correspondence')
    return dict(status='PAPERSET_IDENTITY_UNVERIFIED',verified=False,
        declared_scene_count=spec['count'],matched_scene_count=0,discarded_samples=0)


def validate_manifest(binding, root=ROOT, verify_files=True, stopcheck=None):
    """Validate one dataset independently; never inherit a sensor-source catalog."""
    manifest = deepcopy(read_json(_path(binding,root)) if isinstance(binding,(str,Path)) else binding)
    sensor = manifest.get('dataset')
    if manifest.get('schema') != SCHEMA or sensor not in SENSORS:
        raise ValueError('Explicit PCREPRO sensor manifest required')
    bands, dn = SENSORS[sensor]
    if (manifest.get('bands'),manifest.get('max_dn'),manifest.get('units')) != (bands,dn,'DN'):
        raise ValueError('Sensor bands/DN units mismatch; no per-image rescale or guessed range')
    canonical_indices(sensor,manifest.get('band_order',[]))
    required = {'rr','fr'} if sensor == 'WV2' else {'train','val','rr','fr'}
    if set(manifest.get('splits',{})) != required: raise ValueError('Wrong dataset splits; WV2 is evaluation-only')
    if sensor!='WV2':
        training=manifest['splits']['train'];validation=manifest['splits']['val']
        if (training.get('path')==validation.get('path') or
                training.get('sha256')==validation.get('sha256') or
                set(training.get('scene_ids',[]))&set(validation.get('scene_ids',[]))):
            raise ValueError('Train/validation must be explicitly disjoint, not reused original samples')
    for split,spec in manifest['splits'].items():
        _check_stop(stopcheck)
        if (spec.get('split') != split or spec.get('format') not in ('h5','mat_list')
                or type(spec.get('count')) is not int or spec['count'] < 1
                or len(spec.get('scene_ids',[])) != spec['count']
                or len(set(spec['scene_ids'])) != spec['count']
                or any(not isinstance(v,str) or not v for v in spec['scene_ids'])):
            raise ValueError('Every split needs explicit unique scene IDs, count, format, and split identity')
        if not isinstance(spec.get('keys'),dict) or set(spec['keys'])-{'pan','ms','lms','gt'}:
            raise ValueError('Explicit raw tensor keys required; LP sidecars are forbidden')
        if not isinstance(spec.get('shapes'),dict): raise ValueError('Explicit native per-scene shapes required')
        required_keys = {'pan','ms'} | ({'lms'} if split=='fr' else {'gt'})
        if not required_keys <= set(spec['keys']) or set(spec['shapes']) != set(spec['keys']):
            raise ValueError('Declared keys/shapes do not cover required native inputs')
        for shape in spec['shapes'].values():
            if not isinstance(shape,list) or len(shape)!=3 or any(type(v)is not int or v<1 for v in shape):
                raise ValueError('Native shapes must contain three positive CHW integers')
        if split in ('train','val') and spec.get('format') != 'h5': raise ValueError('Train/val must bind original H5')
        paths = [spec['path']] if spec['format']=='h5' else [v['path'] for v in spec.get('scenes',[])]
        for item in ([spec] if spec['format']=='h5' else spec.get('scenes',[])):
            if not re.fullmatch('[0-9a-f]{64}',str(item.get('sha256',''))):
                raise ValueError('Every data asset needs explicit SHA256')
        if sensor == 'QB' and split in ('train','val') and (
                spec.get('origin') != 'ORIGINAL_PANCOLLECTION' or any('msfix' in str(p).lower() for p in paths)
                or spec.get('derived_from') or spec.get('repair_applied')):
            raise ValueError('QB original train/val required; msfix or repaired substitutes are forbidden')
        if spec['format']=='h5':
            if spec.get('layout') not in ('NCHW','NHWC'): raise ValueError('Explicit H5 tensor layout required')
            if verify_files:
                path = _digest_file(spec,root,stopcheck)
                with h5py.File(path,'r') as store:
                    if any(var not in store or len(store[var]) != spec['count'] for var in spec['keys'].values()):
                        raise ValueError('H5 sample population differs from declared IDs/count')
                    layout={'NCHW':'CHW','NHWC':'HWC'}[spec['layout']]
                    for index in range(spec['count']):
                        _check_stop(stopcheck)
                        rows={key:_layout(store[var][index],layout,pan=key=='pan')
                              for key,var in spec['keys'].items()}
                        _shape_check(rows,sensor,split,spec['shapes'])
        else:
            if (spec.get('layout') not in ('CHW','HWC') or len(spec.get('scenes',[])) != spec['count']
                    or [v.get('scene_id') for v in spec['scenes']] != spec['scene_ids']):
                raise ValueError('Explicit ordered .mat scene list required')
            if len({str(resolved_path(v['path'],root).resolve()) for v in spec['scenes']})!=spec['count']:
                raise ValueError('Duplicate .mat file cannot count as multiple independent scenes')
            if verify_files:
                for item in spec['scenes']: _digest_file(item,root,stopcheck)
        if split == 'fr' and spec['count'] != 20 and sensor != 'WV2':
            raise ValueError('Paper FR population must be20 scenes; no subset fallback')
        if verify_files and spec['format']=='mat_list':
            for index in range(spec['count']):
                _check_stop(stopcheck)
                _shape_check(_scene_arrays(spec,index,root),sensor,split,spec['shapes'])
        if split in ('rr','fr'):
            spec['identity_validation'] = _paper_identity(spec,split,sensor,root,verify_files,stopcheck)
    manifest['binding_status'] = ('PAPERSET_IDENTITY_UNVERIFIED' if sensor=='WV2' and any(
        not manifest['splits'][s]['identity_validation']['verified'] for s in ('rr','fr')) else 'READY')
    manifest['normalization'] = '2*DN/max_dn-1; no clipping/rounding/min-max'
    return manifest


def bind_datasets(bindings, root=ROOT):
    raw = read_json(bindings) if isinstance(bindings,(str,Path)) else bindings
    datasets = raw.get('datasets',raw); result = {}
    for sensor in SENSORS:
        try:
            if sensor not in datasets: raise FileNotFoundError('Dataset binding absent: '+sensor)
            doc = validate_manifest(datasets[sensor],root)
            result[sensor] = dict(status=doc['binding_status'],manifest=doc,manifest_sha256=object_sha(doc))
        except (ValueError,OSError,KeyError,TypeError) as exc:
            result[sensor] = dict(status='BLOCKED_DATASET',reason=f'{type(exc).__name__}: {exc}')
    return result


def inventory_candidates(root=ROOT):
    """Discovery only; filenames never activate or authenticate a dataset."""
    base = Path(root)/'data'
    return dict(status='CANDIDATES_NOT_BOUND',files=[str(p) for p in sorted(base.rglob('*'))
        if p.is_file() and p.suffix.lower() in ('.h5','.mat','.json')],automatic_binding=False)


class NativeDataset(Dataset):
    def __init__(self, manifest, split, root=ROOT):
        self.root=Path(root);self.manifest=deepcopy(read_json(_path(manifest,root)) if isinstance(manifest,(str,Path)) else manifest)
        self.sensor=self.manifest['dataset'];self.bands,self.max_pixel=SENSORS[self.sensor]
        self.split=split;self.spec=self.manifest['splits'][split];self.base_count=self.spec['count']
        self.scene_ids=list(self.spec['scene_ids']);self.has_gt='gt' in self.spec['keys']
        self.augment=split=='train';self.manifest_sha256=object_sha(self.manifest)
        self._handle=None;self._pid=None
    def __len__(self):return self.base_count
    def __getstate__(self):
        state=dict(self.__dict__);state['_handle']=None;state['_pid']=None;return state
    def close(self):
        if self._handle is not None:self._handle.close()
        self._handle=None;self._pid=None
    def raw(self,index):
        if not 0<=int(index)<len(self):raise IndexError(index)
        if self.spec['format']=='mat_list':return _scene_arrays(self.spec,int(index),self.root)
        if self._handle is None or self._pid!=os.getpid():
            self._handle=h5py.File(_path(self.spec['path'],self.root),'r');self._pid=os.getpid()
        layout={'NCHW':'CHW','NHWC':'HWC'}[self.spec['layout']]
        return {key:_layout(self._handle[var][int(index)],layout,pan=key=='pan') for key,var in self.spec['keys'].items()}
    def __getitem__(self,index):
        meta=(int(index),0,0,0,0,0) if not isinstance(index,(tuple,list)) else tuple(map(int,index))
        if len(meta)!=6:raise ValueError('Metadata must be source,hflip,vflip,rot,crop_yLR,crop_xLR')
        source,h,v,r,y,x=meta
        if h not in (0,1) or v not in (0,1) or r not in range(4) or min(y,x)<0:
            raise ValueError('Invalid paired augmentation metadata')
        rows=self.raw(source)
        if self.split!='train' and any(meta[1:]):raise ValueError('Evaluation geometry must remain native')
        result={}
        for key,array in rows.items():
            if self.split=='train':
                scale=1 if key=='ms' else 4
                if y*scale+16*scale>array.shape[-2] or x*scale+16*scale>array.shape[-1]:
                    raise ValueError('Crop must be native64/16 without resize')
                array=array[:,y*scale:(y+16)*scale,x*scale:(x+16)*scale]
                if h:array=array[:,:,::-1]
                if v:array=array[:,::-1,:]
                array=np.rot90(array,r,axes=(1,2))
            raw=torch.from_numpy(np.array(array,dtype=np.float32,copy=True))
            if key=='pan':result['pan_dn']=torch.from_numpy(np.array(array,dtype=np.float64,copy=True))
            result[key]=raw.mul(2./self.max_pixel).sub(1.)
        for key in ('gt','lms'):result.setdefault(key,torch.empty(0,dtype=torch.float32))
        result['meta']=torch.tensor(meta,dtype=torch.int64)
        return result
    def base(self,index):return self[index]


class PairedBatchStream:
    """Only successful optimizer updates advance the resumable data cursor."""
    def __init__(self,dataset,batch_size,seed):
        if dataset.split!='train' or batch_size<1 or len(dataset)<batch_size or not 0<=seed<2**32:
            raise ValueError('Valid train dataset/batch/32-bit run seed required')
        self.dataset=dataset;self.n=len(dataset);self.batch_size=int(batch_size);self.seed=int(seed)
        self.count=self.n//self.batch_size;self.epoch=0;self.cursor=0;self._refresh()
    @property
    def completed_updates(self):return self.epoch*self.count+self.cursor
    @property
    def worker_seed(self):return (self.seed+self.epoch)%2**32
    def _refresh(self):
        rng=lambda axis:np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,axis]))
        self.order=rng(1).permutation(self.n)
        self.geometry=rng(2).integers(0,16,size=self.n)
        h,w=self.dataset.spec['shapes']['ms'][-2:]
        crop=rng(3);self.ys=crop.integers(0,h-16+1,size=self.n);self.xs=crop.integers(0,w-16+1,size=self.n)
    def batch_at(self,i):
        if not 0<=i<self.count:raise ValueError('Batch index outside epoch')
        return [(int(self.order[j]),int(self.geometry[j]&1),int((self.geometry[j]>>1)&1),
                 int(self.geometry[j]>>2),int(self.ys[j]),int(self.xs[j]))
                for j in range(i*self.batch_size,(i+1)*self.batch_size)]
    def remaining_batches(self):return [self.batch_at(i) for i in range(self.cursor,self.count)]
    def advance(self):
        if self.cursor>=self.count:raise ValueError('Cursor beyond epoch')
        self.cursor+=1
    def new_epoch(self):
        if self.cursor!=self.count:raise ValueError('Cannot skip unconsumed batches')
        self.epoch+=1;self.cursor=0;self._refresh()
    def state_dict(self):
        return dict(schema='PCREPRO_PAIRED_STREAM_v1',seed=self.seed,n=self.n,batch_size=self.batch_size,
            epoch=self.epoch,cursor=self.cursor,completed_updates=self.completed_updates,
            dataset_manifest_sha256=self.dataset.manifest_sha256,drop_last=True)
    def load_state_dict(self,state):
        expected=self.state_dict()
        for key in ('schema','seed','n','batch_size','dataset_manifest_sha256','drop_last'):
            if state.get(key)!=expected[key]:raise ValueError('Resume sampler/data/seed identity differs')
        epoch,cursor=state.get('epoch'),state.get('cursor')
        if type(epoch)is not int or epoch<0 or type(cursor)is not int or not 0<=cursor<=self.count:
            raise ValueError('Invalid resume data cursor')
        if state.get('completed_updates')!=epoch*self.count+cursor:raise ValueError('Sampler update count mismatch')
        self.epoch,self.cursor=epoch,cursor;self._refresh()


BatchStream=PairedBatchStream
