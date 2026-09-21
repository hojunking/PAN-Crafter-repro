"""Local 50K/100K endpoint calibration; the existing numerical core is unchanged."""
from pathlib import Path

import numpy as np
import torch

from fh12.calibration import _write_npz_immutable, AXIS16, RADII, axis16_q
from fh12.data import RECIPE, AUGMENTATION
from g20.data import build_dataset
from l100.common import (ROOT, apply_runtime_policy, check_deadline, immutable_json,
                         object_sha, resolved_path, sha256, source_identity)
from l100.plan import CAMPAIGN_ID, REFERENCES
from l100.references import (teacher_endpoint, reference_path, validate_reference,
                            data_signature, FORWARD_CONTRACT)
from qg40.calibration import select_calibration_indices, compute_calibration as _compute
from qg40.reference_parity import identity_for, verify_q_cache


def compute_calibration(model, dataset, indices, device='cuda', batch_size=16,
                        deadline_utc=None, *, synthetic_test=False):
    if getattr(model, 'bands', None) != 4:
        raise ValueError('LOCAL-T calibration requires GF2 C4')
    if not synthetic_test and batch_size != 16:
        raise ValueError('Production calibration and online q verification use batch16')
    result, arrays = _compute(model, dataset, indices, device=device, batch_size=batch_size,
        deadline_utc=deadline_utc, synthetic_test=synthetic_test)
    result['schema'] = 'L100_CALIBRATION_v2'
    result['batch_size'] = batch_size
    return result, arrays


def calibrate(teacher_run, root=ROOT, server=None, device='cuda', deadline_utc=None,
              batch_size=16, *, deadline=None):
    if deadline is not None:
        if deadline_utc is not None and deadline_utc != deadline:
            raise ValueError('Conflicting calibration deadlines')
        deadline_utc = deadline
    root = Path(root).resolve()
    if batch_size != 16:
        raise ValueError('Production calibration requires batch16')
    apply_runtime_policy(root)
    check_deadline(deadline_utc)
    case, cfg, data, identity, model, paths = teacher_endpoint(teacher_run, root, server)
    from l100.training import runtime_context
    window, _ = runtime_context(cfg, root, deadline_utc, resume=True)
    deadline_utc = window['deadline_utc']
    check_deadline(deadline_utc)
    destination = reference_path(case.reference_id, server, root)
    if destination.exists():
        validate_reference(case.reference_id, server, root, dataset_manifest=data)
        return destination
    model.to(device).eval().requires_grad_(False)
    dataset = build_dataset(data, 'train', root=root)
    indices = select_calibration_indices(dataset.base_count)
    calibration, arrays = compute_calibration(model, dataset, indices, device=device,
        batch_size=batch_size, deadline_utc=deadline_utc)
    check_deadline(deadline_utc)
    release = identity['source_identity']
    candidate, cfg_path, data_path = paths['candidate'], paths['config'], paths['data']
    if (source_identity(root) != release or sha256(candidate / 'model.safetensors') != identity['model_sha256']
            or sha256(candidate / 'training_state.pt') != identity['training_state_sha256']):
        raise ValueError('Local Teacher/source changed during calibration')
    for item in data['splits'].values():
        for name, key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(resolved_path(item[name], root)) != item[key]:
                raise ValueError('Actual local data/LP changed during calibration')
    folder = destination.parent
    q_path, cal_path = folder / 'q_cache.npz', folder / 'calibration.json'
    _write_npz_immutable(q_path, arrays)
    ref = REFERENCES[case.reference_id]
    teacher_identity = dict(teacher_run_id=case.run_id, teacher_seed=case.seed, teacher_update=case.updates,
        teacher_checkpoint=str(candidate / 'model.safetensors'), teacher_checkpoint_sha256=identity['model_sha256'],
        teacher_training_state=str(candidate / 'training_state.pt'),
        teacher_training_state_sha256=identity['training_state_sha256'],
        teacher_config=str(cfg_path), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(cfg_path),
        teacher_checkpoint_identity=str(candidate / 'identity.json'),
        teacher_checkpoint_identity_sha256=sha256(candidate / 'identity.json'),
        source_identity=release, teacher_layout='P0', server=server, producer_server=server,
        owner_server=server, teacher_alias=ref.alias, consistency_weight=1e-4)
    calibration.update(teacher_identity)
    immutable_json(cal_path, calibration)
    shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    train = data['splits']['train']
    manifest = dict(schema='L100_LOCAL_REFERENCE_v2', campaign_id=CAMPAIGN_ID,
        reference_id=case.reference_id, calibration_id=ref.calibration_id, **teacher_identity,
        sensor='GF2', num_bands=4, max_pixel=1023, tau_R=calibration['tau_R'], q_ref=calibration['q_ref'],
        calibration_n=3072, calibration_batch_size=16, calibration_path=str(cal_path), calibration_sha256=sha256(cal_path),
        q_cache_path=str(q_path), q_cache_sha256=sha256(q_path), q_shape=calibration['q_shape'],
        dataset_manifest_path=str(data_path), dataset_manifest_sha256=sha256(data_path), data_sha256=object_sha(data),
        train_sha256=train['sha256'], train_lpan_sha256=train['lpan_sha256'],
        train_sample_order_sha256=train['sample_order_sha256'], LP_recipe=RECIPE,
        augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        calibration_indices_sha256=calibration['calibration_indices_sha256'],
        architecture_state_shapes=shapes, architecture_state_shapes_sha256=object_sha(shapes),
        architecture_forward=FORWARD_CONTRACT,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=str(device)))
    parity = verify_q_cache(model, dataset, arrays['q'], reference_identity=identity_for(manifest),
        data_identity=object_sha(data_signature(data)), device=device, deadline=deadline_utc)
    parity_path = folder / 'q_cache_parity.json'
    immutable_json(parity_path, parity)
    manifest.update(q_cache_parity_path=str(parity_path), q_cache_parity_sha256=sha256(parity_path))
    check_deadline(deadline_utc)
    immutable_json(destination, manifest)  # Publish consumable reference only after online integrity.
    validate_reference(case.reference_id, server, root, dataset_manifest=data)
    return destination
