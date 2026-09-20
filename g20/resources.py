"""Read-only local capacity evidence for an admitted G20 atomic block.

Disk uses actual CPU C4 state sizes, uncompressed artifact counts, a 25% margin,
and 1 GiB filesystem headroom. VRAM is never inferred from parameter counts:
only an exact matching completed profile's measured peak supplies a requirement.
An unknown VRAM requirement is explicitly uncertified, not a reason to invent a
nominal GPU size or to wait on another server's measurements.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess

from g20.common import camp, read_json, utcnow
from g20.plan import FULLSTATE_STEPS, GRID_STEPS, case_for

MIB = 1024 ** 2
GIB = 1024 ** 3
CAPACITY_MARGIN = 1.25
DISK_HEADROOM_BYTES = GIB
RUN_METADATA_BYTES = 64 * MIB


def _memory_bytes(value):
    try:
        number = Decimal(value.strip())
        if not number.is_finite() or number < 0:
            return None
        return int(number * MIB)
    except (InvalidOperation, ValueError):
        return None


def _visible_devices(gpus, visible):
    if visible is None:
        return list(gpus)
    if visible.strip() in ("", "-1"):
        return []
    result = []
    for token in visible.split(","):
        token = token.strip()
        matches = [gpu for gpu in gpus if str(gpu["index"]) == token or gpu["uuid"].startswith(token)]
        if len(matches) != 1 or matches[0] in result:
            return None
        result.append(matches[0])
    return result


def inventory(root):
    """Measure write-volume free space and nvidia-smi GPU memory, without CUDA."""
    root = Path(root)
    volume = root / "work_dir" if (root / "work_dir").exists() else root
    measurement = dict(measured_at_utc=utcnow(), root=str(root.resolve()),
                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        total, used, free = shutil.disk_usage(volume)
        measurement["disk"] = dict(status="MEASURED", path=str(volume.resolve()),
                                   total_bytes=total, used_bytes=used, free_bytes=free)
    except OSError as error:
        measurement["disk"] = dict(status="UNKNOWN", path=str(volume), error=str(error))
    try:
        command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
                   "--format=csv,noheader,nounits"]
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        gpus = []
        for row in csv.reader(process.stdout.splitlines()):
            if len(row) != 5:
                raise ValueError("Unexpected nvidia-smi GPU inventory columns")
            index, uuid, name, total, free = (value.strip() for value in row)
            total_bytes, free_bytes = _memory_bytes(total), _memory_bytes(free)
            if not index.isdigit() or not uuid or not name:
                raise ValueError("Invalid nvidia-smi GPU identity")
            if total_bytes is not None and free_bytes is not None and free_bytes > total_bytes:
                raise ValueError("nvidia-smi free memory exceeds total memory")
            gpus.append(dict(index=int(index), uuid=uuid, name=name,
                             total_bytes=total_bytes, free_bytes=free_bytes))
        gpus.sort(key=lambda gpu: gpu["index"])
        visible = _visible_devices(gpus, measurement["cuda_visible_devices"])
        measurement["gpu"] = dict(status="MEASURED", devices=gpus,
            visible_devices=visible, selected_device=visible[0] if visible else None,
            selection="first CUDA_VISIBLE_DEVICES entry, otherwise default CUDA device 0",
            requirement_certified=False)
        if visible is None:
            measurement["gpu"]["status"] = "UNKNOWN_VISIBILITY"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        measurement["gpu"] = dict(status="UNKNOWN", devices=[], selected_device=None,
                                   requirement_certified=False, error=str(error)[:512])
    return measurement


@lru_cache(maxsize=8)
def _model_footprint(role, width, depth):
    """Construct the real model on CPU; no CUDA call or checkpoint load occurs."""
    import torch
    from pa.aligner import PANGlobalAligner
    from g20.model import build_model
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        donor = PANGlobalAligner(ms_bands=4).state_dict() if role == "S" else None
        model, _ = build_model("P0" if role == "T" else "PLH", width, list(depth), 0,
                              role=role, teacher_aligner_state=donor, num_bands=4)
        model.float()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    return dict(parameters=parameters, parameter_bytes=parameter_bytes, state_bytes=state_bytes,
                device="cpu", precision="fp32", num_bands=4)


def _train_count(root, case):
    path = camp(root, case.server_id) / "dataset_manifest.json"
    try:
        data = read_json(path)
        count = data["splits"]["train"]["count"]
        if data["sensor"] != case.sensor or data["num_bands"] != 4 or isinstance(count, bool) or int(count) != count or count < 3072:
            raise ValueError("Dataset must bind this sensor's actual train count >=3072")
        return int(count), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, str(error)


def _disk_case(root, case):
    model = _model_footprint(case.role, case.width, tuple(case.depth))
    state_bytes, parameter_bytes = model["state_bytes"], model["parameter_bytes"]
    # AdamW stores two FP32 moments, in addition to the complete model state.
    fullstate_bytes = state_bytes + 2 * parameter_bytes
    candidate_states = len(set(GRID_STEPS) & set(FULLSTATE_STEPS))
    checkpoints = len(GRID_STEPS) * state_bytes + candidate_states * fullstate_bytes
    restart = len(FULLSTATE_STEPS) * (state_bytes + fullstate_bytes)
    # Atomic last publication temporarily preserves both the previous and next bundle.
    last = 2 * fullstate_bytes
    errors_per_sample = 1 if case.role == "T" else 2
    diagnostic_step = 128 * 64 * 64 * 4 * errors_per_sample + 128 * 128 * 4 + 3072
    diagnostics = len(FULLSTATE_STEPS) * diagnostic_step
    calibration, export, count, error = 0, 0, None, None
    if case.role == "T":
        count, error = _train_count(root, case)
        if count is not None:
            # q[N,4], per_radius[N,4,4], c[N,4,2], and exact int64 base IDs.
            cache = count * (4 + 16 + 8) * 4 + 3072 * 8
            calibration = cache
            # Portable reference export contains one additional full q cache,
            # exact model and complete training state. Compression gets no credit.
            export = cache + state_bytes + fullstate_bytes
    total = checkpoints + restart + last + diagnostics + calibration + export + RUN_METADATA_BYTES
    return dict(run_id=case.run_id, sensor=case.sensor, role=case.role, profile=case.profile,
                model=model, checkpoint_weights=len(GRID_STEPS), candidate_fullstates=candidate_states,
                restart_fullstates=len(FULLSTATE_STEPS), simultaneous_last_fullstates=2,
                candidate_bytes=checkpoints, restart_bytes=restart, last_bytes=last,
                diagnostic_bytes=diagnostics, calibration_bytes=calibration, export_bytes=export,
                metadata_bytes=RUN_METADATA_BYTES, train_count=count, estimate_bytes=total,
                complete=error is None, missing_measurement=error)


def _matching_peaks(case, observations, gpu):
    values = []
    for observation in observations:
        row = asdict(observation) if is_dataclass(observation) else dict(observation)
        if (row.get("completed") is not True or str(row.get("component", "")).upper() != "TRAIN"
                or (row.get("server_id"), row.get("sensor"), row.get("role"), row.get("width"),
                    tuple(row.get("depth", ())), row.get("profile")) !=
                   (case.server_id, case.sensor, case.role, case.width, tuple(case.depth), case.profile)):
            continue
        if row.get("gpu_uuid") and (gpu is None or row["gpu_uuid"] != gpu["uuid"]):
            continue
        peak = row.get("peak_training_memory_bytes")
        if isinstance(peak, (int, float)) and not isinstance(peak, bool) and math.isfinite(peak) and peak > 0:
            values.append(int(peak))
    return values


def assess_block(root, cases, observations=()):
    """Return capacity evidence for local pending cases, without reserving resources.

    Observations use the timing identity fields plus ``profile`` and
    ``peak_training_memory_bytes``; component must be TRAIN and completed=True.
    Batch-1 postrun memory never supplies a training requirement. Pass only
    pending cases when a matching completed BASE is reused in a paired block.
    """
    cases = tuple(case_for(case) if isinstance(case, str) else case for case in cases)
    if not cases or len({case.server_id for case in cases}) != 1 or len({case.run_id for case in cases}) != len(cases):
        raise ValueError("Capacity assessment needs distinct cases on one local server")
    observations = tuple(observations)
    measured = inventory(root)
    budgets = [_disk_case(root, case) for case in cases]
    reasons, warnings = [], []
    disk_bytes = math.ceil(sum(row["estimate_bytes"] for row in budgets) * CAPACITY_MARGIN) + DISK_HEADROOM_BYTES
    if not all(row["complete"] for row in budgets):
        reasons.append("UNKNOWN_TRAIN_COUNT_FOR_CALIBRATION_DISK")
    disk = measured["disk"]
    if disk.get("status") != "MEASURED":
        reasons.append("DISK_AVAILABILITY_UNKNOWN")
    elif disk["free_bytes"] < disk_bytes:
        reasons.append("INSUFFICIENT_DISK")
    gpu_record = measured["gpu"]
    gpu = gpu_record.get("selected_device")
    if gpu_record["status"] != "MEASURED":
        reasons.append("GPU_INVENTORY_UNKNOWN")
    elif gpu is None:
        reasons.append("NO_VISIBLE_GPU")
    elif gpu["free_bytes"] is None:
        reasons.append("GPU_FREE_MEMORY_UNKNOWN")
    elif gpu["free_bytes"] <= 0:
        reasons.append("NO_FREE_GPU_MEMORY")
    gpu_budgets = []
    for case in cases:
        peaks = _matching_peaks(case, observations, gpu)
        required = math.ceil(max(peaks) * CAPACITY_MARGIN) if peaks else None
        gpu_budgets.append(dict(run_id=case.run_id, profile=case.profile,
            status="MEASURED_PROFILE_PEAK" if peaks else "UNKNOWN_UNMEASURED",
            measured_peak_bytes=max(peaks) if peaks else None, measurements=len(peaks),
            required_bytes=required))
    known_requirements = [row["required_bytes"] for row in gpu_budgets if row["required_bytes"] is not None]
    gpu_bytes = max(known_requirements) if known_requirements else None
    if gpu is not None and gpu["free_bytes"] is not None and gpu_bytes is not None and gpu["free_bytes"] < gpu_bytes:
        reasons.append("INSUFFICIENT_MEASURED_PROFILE_VRAM")
    gpu_complete = all(row["required_bytes"] is not None for row in gpu_budgets)
    if not gpu_complete:
        warnings.append("VRAM_REQUIREMENT_UNMEASURED: capacity is not certified for every profile")
    allowed = not reasons
    status = "WAIT_LOCAL_RESOURCE" if reasons else ("LOCAL_RESOURCE_AVAILABLE" if gpu_complete else "LOCAL_RESOURCE_UNCERTIFIED")
    return dict(allowed=allowed, status=status, reasons=reasons, warnings=warnings, measurement=measured,
        budget=dict(margin=CAPACITY_MARGIN, disk_headroom_bytes=DISK_HEADROOM_BYTES,
            disk_required_bytes=disk_bytes, disk_estimate_complete=all(row["complete"] for row in budgets),
            disk_cases=budgets, gpu_required_bytes=gpu_bytes, gpu_requirement_complete=gpu_complete,
            gpu_cases=gpu_budgets, scope="pending local cases; no credit for compression or unmeasured VRAM"))
