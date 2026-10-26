"""Explicit standalone CUDA profiling. No signals or training changes here."""
from copy import deepcopy
import gc
import os
import time

from . import signature, find_profile, save_profile, R


def build_deployment(cfg, sources, step):
    R._config(cfg)
    import torch
    from torch import nn
    if cfg["phase"] == "donor":
        from panda_sensor5.stage1 import build_stage1
        return build_stage1(cfg, sources, device="cpu").stage
    from panda_sensor5.aligner import BandPANAligner
    from panda_sensor5.reconstruction import build_reconstructor, assert_sampling_contract
    from panda_sensor5.method import MatureInference
    from panda_sensor5.spec import MatureSpec
    spec = MatureSpec(**cfg["stage2_spec"])
    architecture = {"bands": 4, "width": spec.student_width, "depth": [1, 2, 3],
                    "input_layout": "P0", "base_reference": "BICUBIC_MS"}
    if cfg["student_architecture"] != architecture:
        raise ValueError("Student architecture differs from deployed MatureSpec")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["stream_seeds"]["A_init"])
        aligner = BandPANAligner(cfg["aligner_outputs"])
    student = build_reconstructor(spec.reconstruction_spec, seed=spec.seed, role="S",
                                  sampling_variant=spec.sampling_variant, input_variant=spec.input_variant)
    actual = assert_sampling_contract(student, cfg["sampling_variant"], input_variant=cfg["input_variant"])
    if actual["parameter_count"] != cfg["student_sampling"]["student_parameters"]:
        raise ValueError("Actual Student parameter count differs from registered sampling graph")

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


def account_student_lanczos4(measurement):
    """Add the two separable eight-tap sums omitted by the shared dispatcher.

    This remains a partial MAC estimate; gather/indexing, weight evaluation,
    normalization and PAN difference subtraction are explicitly omitted.
    Native measured latency and memory already include every operation.
    """
    for obs in measurement["observations"].values():
        accounting = obs["accounting"]
        component = "lanczos4_two_axis_8tap_MAC_equivalent_ESTIMATE"
        if component in accounting["components_MAC_equivalent"]:
            raise ValueError("Lanczos arithmetic already accounted")
        b, channels, height, width = obs["pan_shape"]
        estimate = 16 * b * channels * height * width
        accounting["components_MAC_equivalent"][component] = estimate
        accounting["mac_equivalent"] += estimate
        accounting["arithmetic_flops_covered_estimate"] += 2 * estimate
        accounting["omissions"].append(
            "Lanczos weight sinc/normalization, reflection indexing/gather, and optional PAN difference subtraction")
        obs["sheet_fields"]["FLOPs(G)"] = accounting["mac_equivalent"] / 1e9
        obs["sheet_fields"]["Cost_scope"] += "; Lanczos4 two-axis 8-tap products included as 16 MAC/output estimate"


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
            "pan_shift_scale": source_sig["pan_shift_scale"],
            "pan_execution": source_sig["pan_execution"],
            "student_sampling": source_sig["student_sampling"],
            "reconstructor_sampling": source_sig["reconstructor_sampling"],
            "student_input": source_sig["student_input"], "student_warp": source_sig["student_warp"],
            "reconstructor_input": source_sig["reconstructor_input"],
            "training_teacher_included": False if cfg["phase"] == "stage2" else None,
            "graph": "A+S" if cfg["phase"] == "stage2" else "A+T"}
        if cfg["phase"] == "stage2":
            _stage2_labels(value)
        account_student_lanczos4(value)
        for split in ("rr", "fr"):
            value["observations"][split]["sheet_fields"]["Cost_scope"] += (
                "; Sensor5 interp23; geometry=" + source_sig["geometry_key"]
                + "; sampling=" + source_sig["reconstructor_sampling"]["variant"]
                + "; input=" + source_sig["reconstructor_input"]["variant"]
                + "; warp=" + "Lanczos4"
                + "; exact native deployment; independent isolated architecture measurement")
        return save_profile(cfg, step, root, value)
    finally:
        del graph
        gc.collect()
        torch.cuda.empty_cache()


def registered_profiles(server):
    from panda_sensor5 import plan
    # R1/R2 share one graph; each actual sampler requires a separate measurement.
    result = [(plan.resolve_case(server, plan.DONORS[0]), 50000)]
    seen = set()
    for group in plan.groups(server):
        for case, replica in group["runs"]:
            cfg = plan.resolve_case(server, case, replica)
            if cfg["phase"] != "stage2":
                continue
            key = (cfg["policy_id"], cfg["sampling_variant"], cfg["input_variant"],
                   cfg["student_architecture"]["width"])
            if key not in seen:
                seen.add(key)
                result.append((cfg, max(1000, cfg.get("inherited_updates", 0)+1000)))
    return result
