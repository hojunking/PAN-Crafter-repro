"""Pretraining response-only diagnostics on the fixed RR20 crop manifest.

Weak cancellation is a measurement, never a teacher replacement gate.  Clone
identity/sign/dimension/source failures are technical errors.  No crop's
reconstruction metrics are mixed into full256 native results.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from fh12.data import build_dataset
from fh12.model import build_model, state_hash
from pa.warp import warp_pan
from reporting_extra.evaluation import _precision
from panda_rb_m12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow, source_identity
from panda_rb_m12.evaluation import seal, validate_artifacts, save_csv
from panda_rb_m12.stress import array_digest, relative_response, validate_grid, warp_convention_test


def crop_manifest():
    crops = []
    for size in (64, 128):
        end = 256-size; center = end//2
        for name, top, left in (('center', center, center), ('top_left', 0, 0),
                                ('top_right', 0, end), ('bottom_left', end, 0), ('bottom_right', end, end)):
            crops.append(dict(crop_id=f'S{size}_{name}', size=size, top=top, left=left,
                              bottom=top+size, right=left+size))
    crops.append(dict(crop_id='S256_full', size=256, top=0, left=0, bottom=256, right=256))
    return dict(schema='PANDA_M12_RESPONSE_CROPS_v1', scene_indices=list(range(20)), crops=crops,
                source_shape=[256, 256], paired_inputs='native PAN and bicubic native MS reference at identical HR coordinates',
                injection='shift full256 PAN once before taking fixed crop; MS reference unchanged',
                aligner_input_margin=4, prediction_metrics_included=False)


def _crop(tensor, row):
    return tensor[..., row['top']:row['bottom'], row['left']:row['right']]


def verify_clone(teacher, clone, pan, ms_base):
    if (state_hash(teacher.aligner.state_dict()) != state_hash(clone.aligner.state_dict())
            or teacher.aligner_margin != clone.aligner_margin
            or any(m.training for m in teacher.aligner.modules())
            or any(m.training for m in clone.aligner.modules())):
        raise ValueError('Initial A clone parameter/buffer/eval/margin identity differs from F1')
    with torch.no_grad():
        a = teacher.predict_delta(pan, ms_base)
        b = clone.predict_delta(pan, ms_base)
    if not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError('Initial A clone response differs from frozen F1 on identical input')
    return dict(passed=True, aligner_state_sha256=state_hash(teacher.aligner.state_dict()),
                maximum_absolute_error=float((a-b).abs().max()))


def _old_aligners(server, teacher, root, device):
    """Only verified completed B01 exact50K assets, never opportunistic weights."""
    from safetensors.torch import load_file
    old_root = Path(root) / 'work_dir/_panda_rb/20260928/B01/RB01' / server
    recovered, missing = [], []
    for repeat in (1, 2):
        for case in ('QFULL', 'QMEAN', 'QSHUF', 'QESUR'):
            directory = old_root / f'R{repeat}' / case / 'checkpoints'
            selection_path = directory / 'selection_manifest.json'
            if not selection_path.is_file():
                missing.append(dict(repeat=repeat, case_id=case, status='OLD_SLOT_NOT_FOUND'))
                continue
            selection = read_json(selection_path)
            exact = selection.get('primary', {})
            model_path = directory / 'exact50000/model.safetensors'
            identity_path = directory / 'exact50000/identity.json'
            if not selection.get('complete') or selection.get('actual_updates') != 50000 or not model_path.is_file() or not identity_path.is_file():
                missing.append(dict(repeat=repeat, case_id=case, status='OLD_SLOT_INCOMPLETE'))
                continue
            identity = read_json(identity_path); digest = sha256(model_path)
            if (exact.get('checkpoint_sha256') != digest or identity.get('model_sha256') != digest
                    or exact.get('update') != 50000 or identity.get('update') != 50000):
                missing.append(dict(repeat=repeat, case_id=case, status='OLD_SLOT_IDENTITY_FAILURE'))
                continue
            state = load_file(str(model_path)); aligner = {key[8:]: value for key, value in state.items() if key.startswith('aligner.')}
            model = copy.deepcopy(teacher); model.aligner.load_state_dict(aligner, strict=True)
            model.to(device).eval().requires_grad_(False)
            recovered.append((f'B01_{server}_R{repeat}_{case}', model,
                              dict(checkpoint_sha256=digest, identity_sha256=sha256(identity_path),
                                   source_identity=identity.get('source_identity'), source_path=str(model_path),
                                   aligner_state_sha256=state_hash(aligner))))
    return recovered, missing


@torch.no_grad()
def response_probe(binding_path, server, root=ROOT, device='cuda', recover_old=True):
    from panda_rb_m12.binding import load_binding
    from panda_rb_m12.common import campaign_dir
    from panda_rb_m12.plan import training_runs, shift_grid
    teacher, _cfg, binding, data, _q = load_binding(binding_path, device=device)
    teacher.eval().requires_grad_(False)
    registered = [row for row in training_runs(server=server, root=root) if row['case_id'] == 'QFULL']
    if len(registered) != 4 or len({row['seed'] for row in registered}) != 4:
        raise ValueError('Response probe requires exactly four registered initial A clones on s1/s3/s5')
    manifest = crop_manifest(); grid = shift_grid(root); shifts = validate_grid(grid)
    identity = dict(schema='PANDA_M12_RESPONSE_v1', server=server, binding_sha256=object_sha(binding),
                    teacher_checkpoint_sha256=binding['teacher_checkpoint_sha256'],
                    teacher_aligner_state_sha256=state_hash(teacher.aligner.state_dict()),
                    rr_identity={k: data['splits']['rr'][k] for k in ('sha256', 'sample_order_sha256', 'count')},
                    crop_manifest_sha256=object_sha(manifest), grid_sha256=object_sha(grid), recover_existing_b01=bool(recover_old),
                    source_identity=source_identity(root), blocks=[{k: row[k] for k in ('run_id', 'seed', 'repeat')} for row in registered])
    directory = campaign_dir(root) / 'common/response' / server; report_path = directory / 'report.json'
    if report_path.exists():
        report = validate_artifacts(read_json(report_path), directory)
        if report['identity'] != identity:
            raise ValueError('M12 response probe identity changed')
        return report
    directory.mkdir(parents=True, exist_ok=True)
    crop_path = directory / 'crop_manifest.json'
    if crop_path.exists() and read_json(crop_path) != manifest:
        raise ValueError('Predetermined response crops changed')
    atomic_json(crop_path, manifest)
    started = time.monotonic(); dataset = build_dataset(data, 'rr', root=root)
    if len(dataset) != 20:
        raise ValueError('Response diagnostic requires the fixed RR20 scene manifest')
    clones = []
    for row in registered:
        clone, init = build_model('PLH', 104, [1, 2, 2], row['seed'], role='S', teacher_aligner_state=teacher.aligner.state_dict())
        clones.append((row, clone.to(device).eval().requires_grad_(False), init))
    recovered, missing = _old_aligners(server, teacher, root, device) if recover_old else ([], [])
    models = [('F1', teacher, dict(checkpoint_sha256=binding['teacher_checkpoint_sha256']))] + recovered
    rows = []; clone_receipts = []; files = {}; conventions = warp_convention_test(device)
    try:
        with _precision(identity['source_identity'], device):
            for label, model, evidence in models:
                for scene in range(20):
                    _gt, _lms, ms, _lp, pan, indices = dataset.base(scene)
                    if int(indices[0]) != scene:
                        raise ValueError('Response RR scene order changed')
                    pan = pan[None].to(device); ms = ms[None].to(device)
                    base = F.interpolate(ms.float(), scale_factor=4, mode='bicubic', align_corners=False)
                    corrections = {crop['crop_id']: model.predict_delta(_crop(pan, crop), _crop(base, crop)) for crop in manifest['crops']}
                    if label == 'F1' and scene == 0:
                        for row, clone, init in clones:
                            for crop in manifest['crops']:
                                proof = verify_clone(teacher, clone, _crop(pan, crop), _crop(base, crop))
                                clone_receipts.append(dict(run_id=row['run_id'], seed=row['seed'], repeat=row['repeat'],
                                                           crop_id=crop['crop_id'], proof=proof, initial_hashes=init['hashes']))
                    for shift in shifts:
                        epsilon = pan.new_tensor([[shift['dy'], shift['dx']]])
                        shifted = pan if shift['id'] == 'D000' else warp_pan(pan, epsilon)
                        for crop in manifest['crops']:
                            c0 = corrections[crop['crop_id']]
                            estimated = c0 if shift['id'] == 'D000' else model.predict_delta(_crop(shifted, crop), _crop(base, crop))
                            if label == 'F1' and scene == 0 and shift['id'] in ('D017', 'D042'):
                                for row, clone, _init in clones:
                                    proof = verify_clone(teacher, clone, _crop(shifted, crop), _crop(base, crop))
                                    clone_receipts.append(dict(run_id=row['run_id'], seed=row['seed'], repeat=row['repeat'],
                                                               crop_id=crop['crop_id'], shift_id=shift['id'], proof=proof))
                            response = relative_response(estimated, epsilon, c0)
                            if not torch.isfinite(estimated).all() or not torch.isfinite(c0).all() or not torch.isfinite(response).all():
                                raise FloatingPointError('Nonfinite response diagnostic is a technical failure, not a weak cancellation result')
                            rows.append(dict(reference_id=label, scene_index=scene, crop_id=crop['crop_id'], crop_size=crop['size'],
                                             shift_id=shift['id'], radius_hr=shift['radius_hr'], epsilon_dy=shift['dy'], epsilon_dx=shift['dx'],
                                             c0_dy=float(c0[0, 0]), c0_dx=float(c0[0, 1]),
                                             estimated_dy=float(estimated[0, 0]), estimated_dx=float(estimated[0, 1]),
                                             relative_response_l1_sum=float(response[0]),
                                             native_pan_crop_digest=array_digest(_crop(pan, crop)[0].cpu().numpy()),
                                             shifted_pan_crop_digest=array_digest(_crop(shifted, crop)[0].cpu().numpy()),
                                             ms_base_crop_digest=array_digest(_crop(base, crop)[0].cpu().numpy()), status='MEASURED'))
        save_csv(directory / 'per_crop_shift.csv', rows)
        files = {'crop_manifest.json': sha256(crop_path), 'per_crop_shift.csv': sha256(directory / 'per_crop_shift.csv')}
        report = seal(dict(complete=True, schema='PANDA_M12_RESPONSE_v1', identity=identity,
                           conventions=conventions, n_independent_students=0, n_independent_teachers=1,
                           n_initial_clones=4, initial_clone_response_alias_of='F1; verified identical A parameters/eval/buffers/margin and all 11 crop positions at native/D017/D042',
                           initial_clone_parity=clone_receipts, recovered_b01=[dict(reference_id=label, **evidence) for label, _, evidence in recovered],
                           old_missing=missing, n_observations=len(rows), file_hashes=files,
                           elapsed_seconds=time.monotonic()-started, completed_at_utc=utcnow(),
                           outcome='MEASURED; weak response does not fail setup or select a new Teacher',
                           original_q_coordinate_reduction='coordinate mean; distinct from response abs(dy)+abs(dx) sum',
                           initial_clones_are_new_teacher_seeds=False, native_rr_reconstruction_metrics_included=False))
        atomic_json(report_path, report)
        return validate_artifacts(report, directory)
    finally:
        del clones, recovered, models, teacher
