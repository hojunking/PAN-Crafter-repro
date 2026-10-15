"""Isolated fresh-architecture deployment profiling, never active-model timing.

The real trainer remains CPU-resident. The profiled deployment graph is the
existing native A+warp+reconstructor graph, with Student replacing Teacher in
Stage 2; the training-only frozen Teacher is not allocated on the GPU.
"""
from __future__ import annotations

import gc
import os
import subprocess
import time

from .reporting import cost_config, phase


def _gpu_pids():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    result = set()
    for line in text.splitlines():
        value = line.strip()
        if value:
            if not value.isdecimal():
                raise RuntimeError("Cannot authenticate NVIDIA compute PID: " + value)
            result.add(int(value))
    return result


def _assert_retained_cpu(trainer):
    import torch
    if trainer is None or not hasattr(trainer, "stage") or not hasattr(trainer, "optimizer"):
        raise ValueError("Actual retained trainer and optimizer required for profiling")
    for tensor in list(trainer.stage.parameters()) + list(trainer.stage.buffers()):
        if tensor.device.type != "cpu" or tensor.grad is not None and tensor.grad.device.type != "cpu":
            raise ValueError("Move actual trainer parameters, buffers and gradients to CPU before cost profiling")

    def inspect(value):
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError("Retained optimizer state must remain on CPU")
        if isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect(item)
    inspect(trainer.optimizer.state)


def build_deployment(cfg, sources):
    """Fresh native graph only; no checkpoint and no Stage 2 training Teacher."""
    from directshift48.training import build_direct_stage1
    from panda_method.reconstruction import ReconstructionSpec
    keyed = cost_config(cfg)
    b, seeds = keyed["baseline_resolved"], keyed["stream_seeds"]
    return build_direct_stage1(sources,
        teacher_spec=ReconstructionSpec(**b["teacher_architecture"], input_layout="P0", base_reference="BICUBIC_MS"),
        teacher_seed=seeds["S_init"] if phase(cfg) == "stage2" else seeds["T_init"],
        aligner_seed=seeds["A_init"], epsilon_seed=seeds["rr_epsilon"],
        aligner_view_margin=b["aligner_architecture"]["view_margin"],
        roi_margin=b["absolute_roi_margin"], boundary_policy=b["boundary_policy"],
        lambda_abs=cfg.get("lambda_abs", 1.), lambda_eps=cfg.get("lambda_epsilon", .001))


def _stage2_labels(measurement):
    measurement["parameter_components"]["student"] = measurement["parameter_components"].pop("teacher")
    measurement["memory_definition"] = measurement["memory_definition"].replace("A+T+", "A+S+")
    for split in ("rr", "fr"):
        fields = measurement["observations"][split]["sheet_fields"]
        fields["Cost_scope"] = fields["Cost_scope"].replace("Teacher", "Student") + "; frozen training Teacher and training integrity bookkeeping excluded"
    measurement["sheet_cost"] = measurement["observations"]["rr"]["sheet_fields"]


def ensure_cost(cfg, sources, *, retained_trainer, device, deadline_unix,
                own_gpu_pid=None, cost_root=None):
    """Load validated local cost or profile before any real trainer GPU allocation.

    A fresh measurement needs at least 120 seconds remaining. On cache miss,
    CPU-only execution and competing CUDA jobs fail explicitly, never inventing
    a completed profile. Use --pid=host in Docker or a verified host PID.
    """
    from experiment_cost.store import find_profile, save_profile
    from experiment_cost.measurement import profile_stage
    keyed = cost_config(cfg)
    found = find_profile(keyed, server=cfg["server"], cost_root=cost_root)
    if found is not None:
        return found
    if deadline_unix is None or deadline_unix - time.time() < 120:
        raise TimeoutError("Fewer than 120 seconds remain for isolated cost profiling; no fake cost profile saved")
    import torch
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Production cost measurement needs CUDA; CPU profiles cannot fill sheet cost columns")
    _assert_retained_cpu(retained_trainer)
    gc.collect(); torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(target) != 0:
        raise RuntimeError("Nonzero CUDA allocation before fresh cost graph; actual trainer must remain CPU-resident")
    pid = os.getpid() if own_gpu_pid is None else own_gpu_pid
    if type(pid) is not int or pid < 1:
        raise ValueError("Verified positive NVIDIA host PID required")

    def isolation():
        other = _gpu_pids() - {pid}
        if other:
            raise RuntimeError("Other CUDA compute processes detected: " + str(sorted(other)) + "; profiling skipped")
    isolation()
    assertion = {"server": cfg["server"], "mode": "FRESH_DEPLOYMENT_RETAINED_TRAINER_CPU",
        "training_state_retained_in_memory": True, "other_compute_pids": [],
        "profiler_host_pid": pid, "retained_optimizer_updates": int(retained_trainer.updates),
        "retained_train_device": "cpu", "cuda_allocated_before_graph": 0}
    graph = None
    try:
        graph = build_deployment(cfg, sources).to(target)
        measured = profile_stage(graph, sources, device=str(target),
            deadline_unix=deadline_unix, gpu_isolation_assertion=assertion)
        isolation()
        _assert_retained_cpu(retained_trainer)
        measured["deployment_binding"] = {
            "phase": phase(cfg), "inference_graph": "A+warp+Student" if phase(cfg) == "stage2" else "A+warp+Teacher",
            "training_teacher_included": phase(cfg) == "donor",
            "scope": "Native deployment operators; excludes trainer/Stage2 integrity bookkeeping, losses and calibration",
            "fresh_weights_not_selected_checkpoint": True}
        if phase(cfg) == "stage2":
            _stage2_labels(measured)
        return save_profile(keyed, measured, server=cfg["server"], cost_root=cost_root)
    finally:
        del graph
        gc.collect(); torch.cuda.empty_cache()
