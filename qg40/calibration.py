"""Train-only calibration for each fresh QG40 exact50K Teacher.

The numerical AXIS16 and pooled pixel median routines are reused unchanged;
publication binds their results to the explicit C4 sensor/data/reference.
"""
from __future__ import annotations

from pathlib import Path
import datetime as dt
import numpy as np
import torch
import yaml

from fh12.calibration import (AXIS16, RADII, CONSTANT_ALIGNER_Q, axis16_q,
                              _write_npz_immutable)
from fh12.calibration import compute_calibration as _compute_calibration
from fh12.data import write_immutable_json


def select_calibration_indices(base_count):
    if base_count < 3072:
        raise ValueError("QG40 requires 3072 distinct train base IDs; no smaller fallback")
    return np.random.default_rng(1234).choice(base_count, 3072, replace=False)


def compute_calibration(model, dataset, indices, device="cuda", batch_size=64,
                        deadline_utc=None, *, synthetic_test=False):
    count = getattr(dataset, "base_count", len(dataset))
    if count != len(dataset):
        raise ValueError("Calibration requires a base-ID dataset, not repeated view indexing")
    indices = np.asarray(indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Calibration IDs must be integers")
    if not synthetic_test and not np.array_equal(indices, select_calibration_indices(count)):
        raise ValueError("Calibration must use the shared seed1234 train3072 base IDs")
    if getattr(model, "bands", None) not in (4, 8):
        raise ValueError("Calibration model needs explicit band identity")
    from qg40.reference_parity import isolated_teacher
    with isolated_teacher(model):
        calibration, arrays = _compute_calibration(model, dataset, indices, device=device,
                                                  batch_size=batch_size, deadline_utc=deadline_utc)
    raw_median = calibration['diagnostics']['calibration_output_error_quantiles'][2]
    calibration.update(schema="QG40_CALIBRATION_v1", calibration_seed=1234,
                       num_bands=model.bands, synthetic_test=bool(synthetic_test),
                       tau_raw_pooled_median=raw_median, tau_floor_used=bool(raw_median < 1e-6))
    return calibration, arrays


def calibrate(teacher_run, root, server, device="cuda", deadline_utc=None, batch_size=64):
    from qg40.common import (check_deadline, load_checkpoint_model, object_sha,
                            read_json, resolved_path, sha256, source_identity)
    from qg40.data import AUGMENTATION, RECIPE, build_dataset
    from qg40.plan import case_for
    root = Path(root).resolve()
    check_deadline(deadline_utc)
    run_path = Path(teacher_run)
    if not run_path.is_absolute():
        run_path = root / (run_path if len(run_path.parts) > 1 else Path("work_dir") / run_path)
    cfg_path = run_path / "meta/config.resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    f = cfg["qg40"]
    from qg40.training import runtime_context
    _, deadline_utc = runtime_context(cfg, root, deadline_utc, resume=True)
    check_deadline(deadline_utc)
    case = case_for(run_path.name)
    if case.role != "T" or case.server_id != server or f["role"] != "T":
        raise ValueError("Calibration needs the registered local fresh Teacher")
    release = source_identity(root)
    candidate = run_path / "candidates/50000"
    model, identity = load_checkpoint_model(cfg, candidate, device, expected_source=release)
    state_path = candidate / "training_state.pt"
    if identity.get("update") != 50000 or identity.get("training_state_sha256") != sha256(state_path):
        raise ValueError("Calibration requires a checksum-verified exact50K full state")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not state.get("full_state") or state.get("update") != 50000:
        raise ValueError("Teacher reference cannot use a weights-only or incomplete checkpoint")
    if state.get("config_sha256") != object_sha(cfg) or state.get("source_identity") != release:
        raise ValueError("Teacher full-state provenance differs from its config/release")
    if state.get("model_sha256") != identity["model_sha256"]:
        raise ValueError("Teacher full-state/model identities disagree")
    from qg40.model import state_hash
    if state_hash(state["model_state"]) != identity["state_hash"]:
        raise ValueError("Teacher full-state tensors differ from the candidate")
    data_path = resolved_path(f["dataset_manifest"], root)
    data = read_json(data_path)
    data_hash = object_sha(data)
    if identity.get("data_sha256") != data_hash or state.get("data_sha256") != data_hash:
        raise ValueError("Teacher was trained against another data identity")
    ref_dir = root / "work_dir/_qg40" / server / "references" / case.reference_id
    destination = ref_dir / "reference_manifest.json"
    if destination.exists():
        from qg40.references import validate_reference
        validate_reference(case.reference_id, server, root, manifest_path=destination, dataset_manifest=data)
        return destination
    dataset = build_dataset(data, "train", root=root)
    indices = select_calibration_indices(dataset.base_count)
    calibration, arrays = compute_calibration(model, dataset, indices, device=device,
                                              batch_size=batch_size, deadline_utc=deadline_utc)
    check_deadline(deadline_utc)
    if (sha256(candidate / "model.safetensors") != identity["model_sha256"]
            or object_sha(read_json(data_path)) != data_hash or source_identity(root) != release):
        raise ValueError("Teacher, dataset, or numerical source changed during calibration")
    train = data["splits"]["train"]
    original = resolved_path(train.get("path", train.get("dataroot")), root)
    if sha256(original) != train["sha256"]:
        raise ValueError("Training source changed during calibration")
    q_path, cal_path = ref_dir / "q_cache.npz", ref_dir / "calibration.json"
    _write_npz_immutable(q_path, arrays)
    teacher_identity = dict(teacher_run_id=case.run_id, teacher_update=50000,
        teacher_checkpoint=str(candidate / "model.safetensors"), teacher_checkpoint_sha256=identity["model_sha256"],
        teacher_training_state=str(state_path), teacher_training_state_sha256=sha256(state_path),
        teacher_config=str(cfg_path), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(cfg_path),
        teacher_checkpoint_identity=str(candidate / "identity.json"),
        teacher_checkpoint_identity_sha256=sha256(candidate / "identity.json"),
        source_identity=release, teacher_layout="P0", server=server)
    calibration.update(teacher_identity)
    write_immutable_json(cal_path, calibration)
    manifest = dict(schema="QG40_REFERENCE_v1", sensor=case.sensor, reference_id=case.reference_id,
        num_bands=4, max_pixel=cfg["max_pixel"], **teacher_identity,
        tau_R=calibration["tau_R"], q_ref=calibration["q_ref"], calibration_n=3072,
        calibration_path=str(cal_path), calibration_sha256=sha256(cal_path),
        q_cache_path=str(q_path), q_cache_sha256=sha256(q_path), q_shape=calibration["q_shape"],
        dataset_manifest_path=str(data_path), dataset_manifest_sha256=sha256(data_path),
        data_sha256=data_hash, train_sha256=train["sha256"],
        train_lpan_sha256=train.get("lpan_sha256"),
        train_sample_order_sha256=train.get("sample_order_sha256"),
        LP_recipe=RECIPE, augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        calibration_indices_sha256=calibration["calibration_indices_sha256"],
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=str(device)))
    from qg40.reference_parity import identity_for, verify_q_cache
    from qg40.references import data_signature
    parity = verify_q_cache(model, dataset, arrays['q'], reference_identity=identity_for(manifest),
        data_identity=object_sha(data_signature(data)), device=device, deadline=deadline_utc)
    parity_path = ref_dir / 'q_cache_parity.json'
    write_immutable_json(parity_path, parity)
    manifest.update(q_cache_parity_path=str(parity_path), q_cache_parity_sha256=sha256(parity_path))
    # P0 publication is complete before optional reconstruction distributions.
    write_immutable_json(destination, manifest)
    from qg40.common import atomic_json
    report_path = ref_dir / 'reconstruction_distribution.json'
    try:
        from qg40.error_distribution import write_error_distribution
        common_end = dt.datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
        diagnostic_end = min(common_end, dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60))
        check_deadline(diagnostic_end.isoformat())
        validation = build_dataset(data, 'val', root=root)
        write_error_distribution(report_path, model, dataset, validation, indices, manifest['tau_R'],
            raw_calibration_median=calibration['tau_raw_pooled_median'], device=device,
            batch_size=batch_size, deadline=diagnostic_end.isoformat(),
            reference_identity=identity_for(manifest))
    except Exception as error:
        # Report generation/data loading is P1, never grounds to suppress the
        # verified reference or substitute another tau. Keep failure explicit.
        report = dict(schema='QG40_RECONSTRUCTION_DISTRIBUTION_v1', status='DIAGNOSTIC_FAILED',
                      p0_gate=False, tau_R=manifest['tau_R'], tau_refitted=False,
                      reason=f'{type(error).__name__}: {error}')
        print('QG40 P1 reconstruction diagnostics failed (P0 reference preserved): ' + report['reason'], flush=True)
        try:
            atomic_json(report_path, report)
        except OSError:
            pass  # Read-only/full diagnostic destination cannot invalidate P0.
    return destination
