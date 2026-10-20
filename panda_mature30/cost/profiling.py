"""Explicit standalone CUDA profiling. No signals or training changes here."""
from copy import deepcopy
import gc
import os
import time

from . import signature, find_profile, save_profile, R


def build_deployment(cfg, sources, step):
    import torch
    from torch import nn
    if cfg["phase"] == "donor":
        from panda_mature30.stage1 import build_stage1
        return build_stage1(cfg, sources, device="cpu").stage
    from panda_band30.aligner import BandPANAligner
    from panda_band30.reconstruction import build_reconstructor
    from panda_mature30.method import MatureInference
    from panda_mature30.spec import MatureSpec
    spec = MatureSpec(**cfg["stage2_spec"])
    architecture = {"bands": 8, "width": 112, "depth": [1, 2, 3]}
    if cfg["student_architecture"] != architecture:
        raise ValueError("Student architecture differs from deployed MatureSpec")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["stream_seeds"]["A_init"])
        aligner = BandPANAligner(cfg["aligner_outputs"])
    student = build_reconstructor(spec.reconstruction_spec, seed=spec.seed, role="S")

    class Deployment(nn.Module):
        def __init__(self):
            super().__init__()
            self.inference = MatureInference(aligner, student, spec, completed_update=step)

        @property
        def aligner(self):
            return self.inference.aligner

        @property
        def teacher(self):
            # Historical profiler component name; this is S, never the KD T.
            return self.inference.student

        def native_prediction(self, batch):
            return self.inference(batch)

    return Deployment()


def measure(cfg, step, sources, root, *, device="cuda", deadline_unix=None):
    existing = find_profile(cfg, step, root)
    if existing is not None:
        return existing
    import torch
    from donor30.cost import _gpu_pids, _require_empty_cuda, _stage2_labels
    from experiment_cost.measurement import profile_stage
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise ValueError("Production latency/memory require an isolated CUDA GPU")
    pid = os.getpid()

    def isolation():
        other = _gpu_pids() - {pid}
        if other:
            raise RuntimeError("Cost profiling refused concurrent CUDA jobs: " + str(sorted(other)))

    isolation()
    _require_empty_cuda(device)
    source_sig = signature(cfg, step)
    graph = None
    try:
        graph = build_deployment(cfg, sources, step).to(device)
        value = profile_stage(graph, sources, device=device,
            deadline_unix=deadline_unix or time.time() + 600,
            gpu_isolation_assertion={"server": cfg["server"], "other_compute_pids": [],
                "mode": "STANDALONE_ISOLATED_NO_TRAINER_GPU", "profiler_host_pid": pid,
                "training_state_retained_in_memory": False})
        isolation()
        if signature(cfg, step) != source_sig:
            raise ValueError("Cost implementation changed during measurement")
        value["deployment_signature_sha256"] = R.digest(source_sig)
        value["deployment"] = {"phase": cfg["phase"], "geometry_key": source_sig["geometry_key"],
            "training_teacher_included": False if cfg["phase"] == "stage2" else None,
            "graph": "A+S" if cfg["phase"] == "stage2" else "A+T"}
        if cfg["phase"] == "stage2":
            _stage2_labels(value)
        for split in ("rr", "fr"):
            value["observations"][split]["sheet_fields"]["Cost_scope"] += (
                "; Mature30 interp23; geometry=" + source_sig["geometry_key"]
                + "; exact native deployment; independent isolated architecture measurement")
        return save_profile(cfg, step, root, value)
    finally:
        del graph
        gc.collect()
        torch.cuda.empty_cache()


def registered_profiles(server):
    from panda_mature30 import plan
    # Every distinct deployment, including all three DOWN schedule phases.
    cases = [(plan.DONORS[0], 50000), ("FULL", 1000), ("HALF", 1000),
             ("DOWN", 10001), ("LOW", 1000), ("CAP085", 1000)]
    return [(plan.resolve_case(server, case), step) for case, step in cases]
