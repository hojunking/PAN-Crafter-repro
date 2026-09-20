"""Fresh G20 exact50K reference calibration, with no inherited TA values."""
from pathlib import Path

import numpy as np
import torch
import yaml

from qg40.calibration import select_calibration_indices, compute_calibration as _compute
from fh12.calibration import _write_npz_immutable, AXIS16, RADII, axis16_q
from fh12.data import RECIPE, AUGMENTATION


def compute_calibration(model, dataset, indices, device='cuda', batch_size=64,
                        deadline_utc=None, *, synthetic_test=False):
    if getattr(model, 'bands', None) != 4:
        raise ValueError('G20 is GF2 C4 only')
    calibration, arrays = _compute(model, dataset, indices, device=device,
        batch_size=batch_size, deadline_utc=deadline_utc, synthetic_test=synthetic_test)
    calibration['schema'] = 'G20_CALIBRATION_v1'
    return calibration, arrays


def calibrate(teacher_run, root, server, device='cuda', deadline_utc=None, batch_size=64):
    from g20.common import (check_deadline, load_checkpoint_model, object_sha, read_json,
        resolved_path, sha256, source_identity, immutable_json, camp)
    from g20.plan import case_for
    from g20.data import build_dataset
    from g20.references import REFERENCE_SPECS, FORWARD_CONTRACT, data_signature, validate_reference
    from qg40.reference_parity import verify_q_cache, identity_for
    from qg40.model import state_hash
    root = Path(root).resolve()
    check_deadline(deadline_utc)
    run = Path(teacher_run)
    if not run.is_absolute():
        run = root / (run if len(run.parts) > 1 else Path('work_dir') / run)
    cfg_path = run / 'meta/config.resolved.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    from g20.training import validate_config, runtime_context
    case = case_for(run.name)
    if validate_config(cfg) != case or case.role != 'T' or case.server_id != server:
        raise ValueError('Calibration requires the registered local G20 Teacher')
    window, _train_finish = runtime_context(cfg, root, deadline_utc, resume=True)
    # Exact50K is already frozen. Calibration/closing evaluation may complete
    # between the 18h training stop and the absolute 20h campaign deadline.
    deadline_utc = window['deadline_utc']
    check_deadline(deadline_utc)
    alias, owner, seed, coefficient = REFERENCE_SPECS[case.reference_id]
    if owner != server or seed != cfg['seed'] or case.reference_id == 'R0':
        raise ValueError('New reference cannot fall back to the shared old TA')
    release = source_identity(root)
    candidate = run / 'candidates/50000'
    model, identity = load_checkpoint_model(cfg, candidate, device, expected_source=release)
    state_path = candidate / 'training_state.pt'
    if identity.get('update') != 50000 or identity.get('training_state_sha256') != sha256(state_path):
        raise ValueError('Checksum-verified exact50K full state is required')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if (state.get('full_state') is not True or state.get('update') != 50000
            or state.get('config_sha256') != object_sha(cfg) or state.get('source_identity') != release
            or state.get('model_sha256') != identity['model_sha256']
            or state_hash(state['model_state']) != identity['state_hash']):
        raise ValueError('Teacher full-state/config/model provenance differs')
    data_path = resolved_path(cfg['g20']['dataset_manifest'], root)
    data = read_json(data_path)
    if state.get('data_sha256') != object_sha(data) or identity.get('data_sha256') != object_sha(data):
        raise ValueError('Teacher train data identity differs')
    del state
    folder = camp(root, server) / 'references' / case.reference_id
    destination = folder / 'reference_manifest.json'
    if destination.exists():
        validate_reference(case.reference_id, server, root, dataset_manifest=data)
        return destination
    dataset = build_dataset(data, 'train', root=root)
    indices = select_calibration_indices(dataset.base_count)
    calibration, arrays = compute_calibration(model, dataset, indices, device=device,
                            batch_size=batch_size, deadline_utc=deadline_utc)
    check_deadline(deadline_utc)
    if (sha256(candidate / 'model.safetensors') != identity['model_sha256']
            or object_sha(read_json(data_path)) != object_sha(data) or source_identity(root) != release):
        raise ValueError('Reference source/data/checkpoint changed during calibration')
    train = data['splits']['train']
    for path_key, hash_key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
        if sha256(train[path_key]) != train[hash_key]:
            raise ValueError('Actual train/LP changed during calibration')
    q_path, cal_path = folder / 'q_cache.npz', folder / 'calibration.json'
    _write_npz_immutable(q_path, arrays)
    teacher_identity = dict(teacher_run_id=case.run_id, teacher_update=50000,
        teacher_checkpoint=str(candidate / 'model.safetensors'), teacher_checkpoint_sha256=identity['model_sha256'],
        teacher_training_state=str(state_path), teacher_training_state_sha256=sha256(state_path),
        teacher_config=str(cfg_path), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(cfg_path),
        teacher_checkpoint_identity=str(candidate / 'identity.json'),
        teacher_checkpoint_identity_sha256=sha256(candidate / 'identity.json'), source_identity=release,
        teacher_layout='P0', server=server, teacher_alias=alias, consistency_weight=coefficient)
    calibration.update(teacher_identity)
    immutable_json(cal_path, calibration)
    manifest = dict(schema='G20_REFERENCE_v1', sensor='GF2', reference_id=case.reference_id,
        num_bands=4, max_pixel=1023, **teacher_identity, tau_R=calibration['tau_R'], q_ref=calibration['q_ref'],
        calibration_n=3072, calibration_path=str(cal_path), calibration_sha256=sha256(cal_path),
        q_cache_path=str(q_path), q_cache_sha256=sha256(q_path), q_shape=calibration['q_shape'],
        dataset_manifest_path=str(data_path), dataset_manifest_sha256=sha256(data_path),
        data_sha256=object_sha(data), train_sha256=train['sha256'], train_lpan_sha256=train['lpan_sha256'],
        train_sample_order_sha256=train['sample_order_sha256'], LP_recipe=RECIPE, augmentation=AUGMENTATION,
        augmentation_sha256=object_sha(AUGMENTATION),
        calibration_indices_sha256=calibration['calibration_indices_sha256'],
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=str(device)))
    shapes = {key: list(value.shape) for key, value in model.state_dict().items()}
    manifest.update(architecture_state_shapes=shapes, architecture_state_shapes_sha256=object_sha(shapes),
                    architecture_forward=FORWARD_CONTRACT)
    parity = verify_q_cache(model, dataset, arrays['q'], reference_identity=identity_for(manifest),
        data_identity=object_sha(data_signature(data)), device=device, deadline=deadline_utc)
    parity_path = folder / 'q_cache_parity.json'
    immutable_json(parity_path, parity)
    manifest.update(q_cache_parity_path=str(parity_path), q_cache_parity_sha256=sha256(parity_path))
    check_deadline(deadline_utc)
    # The consumable manifest is published last, only after all actual measurements.
    immutable_json(destination, manifest)
    validate_reference(case.reference_id, server, root, dataset_manifest=data)
    return destination
