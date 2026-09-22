"""Atomic exact-resume snapshots and two named model selections.

Only superseded resume snapshots and superseded best-validation weights owned
by this module are pruned. Exact50K/final results are never automatically removed.
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import torch
from safetensors.torch import load_file,save_file
from fh12.common import atomic_json,object_sha,read_json,sha256
from pcrepro.model import build_model,state_hash


def tensor_cpu_tree(value):
    """Detach a nested own-run optimizer/full-state payload for atomic storage."""
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:tensor_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [tensor_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(tensor_cpu_tree(v) for v in value)
    return value


def _torch_save(path,value):
    path=Path(path)
    with path.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())


def _owned_remove(path,parent,prefix):
    path,parent=Path(path),Path(parent)
    if path.parent.resolve()!=parent.resolve() or not path.name.startswith(prefix) or path.is_symlink():
        raise ValueError('Checkpoint prune target is not owned')
    if path.is_dir():shutil.rmtree(path)


def save_resume(work_dir,state,identity):
    folder=Path(work_dir)/'resume';folder.mkdir(parents=True,exist_ok=True)
    if not state.get('full_state') or state.get('update')!=identity.get('update'):
        raise ValueError('Resume requires exact update/fullstate identity')
    target=folder/f'state_{state["update"]:06d}_{uuid.uuid4().hex}'
    pending=Path(tempfile.mkdtemp(prefix='.pending_',dir=folder))
    try:
        _torch_save(pending/'training_state.pt',state)
        receipt=dict(identity,training_state_sha256=sha256(pending/'training_state.pt'),
                     model_state_hash=state_hash(state['model_state']))
        atomic_json(pending/'identity.json',receipt)
        os.replace(pending,target)
    finally:
        if pending.exists():_owned_remove(pending,folder,'.pending_')
    index_path=folder/'index.json'
    previous=read_json(index_path)['snapshots'] if index_path.exists() else []
    names=[target.name]+previous
    index=dict(schema='PCREPRO_RESUME_INDEX_v1',snapshots=names[:2])
    atomic_json(index_path,index)
    if read_json(index_path)!=index or read_json(target/'identity.json')!=receipt:
        raise ValueError('Resume pointer readback differs; old snapshots preserved')
    for old in names[2:]:_owned_remove(folder/old,folder,'state_')
    return receipt


def load_resume(work_dir,expected=None):
    folder=Path(work_dir)/'resume'
    index=read_json(folder/'index.json')
    if index.get('schema')!='PCREPRO_RESUME_INDEX_v1' or not 1<=len(index.get('snapshots',[]))<=2:
        raise ValueError('Invalid retained resume index')
    path=folder/index['snapshots'][0]
    if path.parent.resolve()!=folder.resolve() or not path.name.startswith('state_'):
        raise ValueError('Invalid resume path')
    identity=read_json(path/'identity.json')
    if sha256(path/'training_state.pt')!=identity['training_state_sha256']:
        raise ValueError('Resume checksum mismatch; never silently reset')
    if any(identity.get(k)!=v for k,v in (expected or {}).items()):
        raise ValueError('Resume source/data/config identity differs')
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
    if (not state.get('full_state') or state.get('update')!=identity['update']
            or state_hash(state['model_state'])!=identity['model_state_hash']):
        raise ValueError('Resume tensor/update differs')
    return state,identity


def save_model(work_dir,label,model,identity,full_state=None):
    if label not in ('best_val','exact_50000'):raise ValueError('Unregistered checkpoint selector')
    if label=='exact_50000' and (identity.get('update')!=50000 or full_state is None
                               or full_state.get('full_state') is not True):
        raise ValueError('Primary checkpoint requires actual50K fullstate')
    folder=Path(work_dir)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    pointer=folder/(label+'.json')
    previous=read_json(pointer) if pointer.exists() else None
    tensor_hash=state_hash(model.state_dict())
    if previous and label=='exact_50000':
        old=read_json(folder/previous['directory']/'identity.json')
        if old.get('model_state_hash')!=tensor_hash or any(old.get(k)!=v for k,v in identity.items()):
            raise ValueError('Exact50K checkpoint is immutable')
        return old
    target=folder/f'{label}_{identity["update"]:06d}_{uuid.uuid4().hex}'
    pending=Path(tempfile.mkdtemp(prefix='.pending_',dir=folder))
    try:
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(pending/'model.safetensors'))
        receipt=dict(identity,label=label,architecture=model.architecture(),model_state_hash=tensor_hash,
            model_sha256=sha256(pending/'model.safetensors'),full_state=full_state is not None)
        if full_state is not None:
            if full_state.get('update')!=identity['update'] or state_hash(full_state['model_state'])!=tensor_hash:
                raise ValueError('Final fullstate differs from model')
            _torch_save(pending/'training_state.pt',full_state)
            receipt['training_state_sha256']=sha256(pending/'training_state.pt')
        atomic_json(pending/'identity.json',receipt);os.replace(pending,target)
    finally:
        if pending.exists():_owned_remove(pending,folder,'.pending_')
    pointer_value=dict(directory=target.name,identity_sha256=object_sha(receipt))
    atomic_json(pointer,pointer_value)
    if read_json(pointer)!=pointer_value or read_json(target/'identity.json')!=receipt:
        raise ValueError('Model pointer readback differs; old best preserved')
    if previous and label=='best_val':_owned_remove(folder/previous['directory'],folder,'best_val_')
    return receipt


def selection_checkpoints(work_dir):
    folder=Path(work_dir)/'checkpoints';out={}
    for selection,label in [('EXACT_50000','exact_50000'),('RR_VAL_ERGAS_MIN','best_val')]:
        pointer=read_json(folder/(label+'.json'))
        path=folder/pointer['directory']
        if path.parent.resolve()!=folder.resolve() or not path.name.startswith(label+'_'):
            raise ValueError('Checkpoint pointer escaped run')
        identity=read_json(path/'identity.json')
        if object_sha(identity)!=pointer['identity_sha256'] or identity.get('label')!=label:
            raise ValueError('Selection checkpoint identity mismatch')
        out[selection]=path
    return out


def load_model_checkpoint(path,expected=None,device='cpu'):
    path=Path(path);identity=read_json(path/'identity.json')
    if any(identity.get(k)!=v for k,v in (expected or {}).items()):raise ValueError('Checkpoint provenance mismatch')
    if sha256(path/'model.safetensors')!=identity['model_sha256']:raise ValueError('Model bytes mismatch')
    tensors=load_file(str(path/'model.safetensors'),device='cpu')
    if state_hash(tensors)!=identity['model_state_hash'] or not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
        raise ValueError('Nonfinite/model tensor checksum mismatch')
    arch=identity['architecture']
    model=build_model(arch['num_bands'],seed=0,max_pixel=arch['max_pixel'],
        hidden_size=arch['hidden_size'],depth=tuple(arch['depth']),num_heads=arch['num_heads'])
    if arch!=model.architecture():raise ValueError('Checkpoint architecture/recipe mismatch')
    model.load_state_dict(tensors,strict=True)
    if identity.get('full_state'):
        if sha256(path/'training_state.pt')!=identity['training_state_sha256']:raise ValueError('Final fullstate checksum mismatch')
        state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
        if (state.get('full_state') is not True or state['update']!=identity['update']
                or state_hash(state['model_state'])!=identity['model_state_hash']):
            raise ValueError('Final fullstate/model mismatch')
    return model.to(device).eval(),identity
