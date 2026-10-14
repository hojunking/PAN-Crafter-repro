"""Read-only, separately executed architecture cost measurements.

The sheet's historical ``FLOPs(G)`` column uses MAC scale (one multiply-add
counts as one). We preserve that scale, but explicitly describe its limited
coverage; latency and memory always execute the *entire* native pipeline.
Importing this module is stdlib-only, so a CPU-only sheet watcher can validate
profile keys without importing torch or allocating GPU memory.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "EXPERIMENT_COST_V1"
PROFILE_PROTOCOL = "NATIVE_BATCH1_FP32_MAC_PARTIAL_V1"
FAMILIES = {"DirectPANAligner": "directshift48", "ResidualPANAligner": "edgeres48"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def architecture_signature(config, *, root=ROOT):
    """Exact inference structure/source key; never key costs on LR or seed.

    The server stays in this key: another GPU's latency/memory must not be
    silently propagated merely because its convolution shapes match.
    """
    b = config.get("baseline_resolved", config)
    family = FAMILIES.get(b.get("aligner_architecture", {}).get("name"))
    if family is None:
        raise ValueError("Only explicit DirectShift48/EdgeRes48 aligners are supported")
    required = ("dataset", "server", "teacher_architecture", "aligner_architecture",
                "teacher_input_layout", "teacher_base", "method_revision")
    if any(key not in b for key in required):
        raise ValueError("Incomplete architecture identity")
    if b["teacher_input_layout"] != "P0" or b["teacher_base"] != "BICUBIC_MS":
        raise ValueError("Cost protocol only supports actual P0/BICUBIC_MS inference")
    paths = ["experiment_cost/measurement.py", "panda_method/reconstruction.py",
             "panda_method/edge_data.py", "panda_method/stages.py",
             "model/pancrafter_paper.py", "model/pancrafter.py", "fh12/model.py",
             "pa/warp.py", "pa/aligner.py", "pa/offset.py"]
    paths += (["directshift48/method.py", "directshift48/training.py"]
              if family == "directshift48" else
              ["panda_method/edge_guided.py", "panda_method/edge_match.py", "teacher48/edge_training.py"])
    root = Path(root)
    signature = {key: b[key] for key in required}
    signature.update(profile_protocol=PROFILE_PROTOCOL, method_family=family,
        alignment_protocol=b.get("alignment_protocol"), edge_match=b.get("edge_match"),
        source_registry_sha256=b.get("source_registry_sha256"),
        lms_override_sha256=b.get("lms_override_sha256"),
        precision="FP32", amp=False, tf32=False, batch=1,
        input_shapes={"rr": [1, 1, 256, 256], "fr": [1, 1, 512, 512]},
        code_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    return json.loads(json.dumps(signature, allow_nan=False))


def architecture_key(config, *, root=ROOT):
    return _digest(architecture_signature(config, root=root))


def build_architecture(config, sources, device):
    """Fresh *same-architecture* stage, never a checkpoint or optimizer state."""
    from panda_method.reconstruction import ReconstructionSpec
    b, seeds = config["baseline_resolved"], config["stream_seeds"]
    family = FAMILIES.get(b["aligner_architecture"]["name"])
    if family == "directshift48":
        from directshift48.plan import validate_method_config
        from directshift48.training import build_direct_stage1 as build
        extra = {}
    elif family == "edgeres48":
        from edgeres48.plan import validate_method_config
        from teacher48.edge_training import build_edge_stage1 as build
        from panda_method.edge_match import EdgeMatchSpec
        extra = {"match_spec": EdgeMatchSpec(**b["edge_match"])}
    else:
        raise ValueError("Unrecognized inference architecture")
    validate_method_config(config)
    return build(sources,
        teacher_spec=ReconstructionSpec(**b["teacher_architecture"], input_layout="P0", base_reference="BICUBIC_MS"),
        teacher_seed=seeds["T_init"], aligner_seed=seeds["A_init"], epsilon_seed=seeds["rr_epsilon"],
        aligner_view_margin=b["aligner_architecture"]["view_margin"], roi_margin=b["absolute_roi_margin"],
        boundary_policy=b["boundary_policy"], lambda_abs=config["lambda_abs"],
        lambda_eps=config["lambda_epsilon"], **extra).to(device)


def _counter_mode():
    """Count dispatched Conv/MatMul plus stated resampling estimates only.

    Dispatch sees functional Scharr convolution, unlike nn.Module hooks.
    Uncounted operations are inventoried rather than assigned hidden zeros.
    It is used for one separate accounting forward, never latency timing.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = Counter()
            self.covered = Counter()
            self.macs = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            name = str(func)
            self.calls[name] += 1
            kind, value = None, 0
            if name in ("aten.conv1d.default", "aten.conv2d.default", "aten.conv3d.default"):
                value = result.numel() * math.prod(args[1].shape[1:])
                kind = "convolution_MAC"
            elif name in ("aten.conv_transpose1d.default", "aten.conv_transpose2d.input", "aten.conv_transpose3d.input"):
                value = args[0].numel() * args[1].shape[1] * math.prod(args[1].shape[2:])
                kind = "convolution_MAC"
            elif name in ("aten.convolution.default", "aten._convolution.default"):
                x, weight = args[:2]
                transposed = args[6]
                if transposed:
                    value = x.numel() * weight.shape[1] * math.prod(weight.shape[2:])
                else:
                    value = result.numel() * math.prod(weight.shape[1:])
                kind = "convolution_MAC"
            elif name == "aten.linear.default":
                value = result.numel() * args[0].shape[-1]
                kind = "matrix_MAC"
            elif name in ("aten.matmul.default", "aten.dot.default", "aten.mv.default"):
                value = result.numel() * args[0].shape[-1]
                kind = "matrix_MAC"
            elif name == "aten.addmm.default":
                value = args[1].shape[-2] * args[1].shape[-1] * args[2].shape[-1]
                kind = "matrix_MAC"
            elif name == "aten.mm.default":
                value = args[0].shape[-2] * args[0].shape[-1] * args[1].shape[-1]
                kind = "matrix_MAC"
            elif name == "aten.bmm.default":
                value = args[0].shape[0] * args[0].shape[-2] * args[0].shape[-1] * args[1].shape[-1]
                kind = "matrix_MAC"
            elif name.startswith("aten.upsample_bicubic2d."):
                # 16 sample products + accumulation ~32 arithmetic operations.
                value, kind = 16 * result.numel(), "bicubic_32op_per_output_MAC_equivalent_ESTIMATE"
            elif name in ("aten.grid_sampler_2d.default", "aten.grid_sampler.default"):
                mode = args[2]
                if mode in (0, 2):
                    value = (16 if mode == 2 else 4) * result.numel()
                    kind = "grid_sample_kernel_MAC_equivalent_ESTIMATE"
            if kind is not None:
                self.covered[name] += 1
                self.macs[kind] += int(value)
            return result

        def report(self):
            return {"mac_equivalent": sum(self.macs.values()),
                "arithmetic_flops_covered_estimate": 2 * sum(self.macs.values()),
                "components_MAC_equivalent": dict(self.macs),
                "covered_operator_calls": dict(self.covered),
                "unaccounted_operator_calls": dict(self.calls - self.covered),
                "complete_arithmetic_coverage": False,
                "convention": "legacy sheet MAC scale: Conv/MatMul MAC=1; bicubic/grid kernel 32-op/output ->16 MAC-equivalent estimate",
                "omissions": ["normalization and activations", "bias and residual arithmetic",
                    "grid-coordinate creation and interpolation-weight calculation", "validation and tensor movement",
                    "EdgeMatch candidate subtraction/abs/multiply/reduction, ordering and tie handling (when present)"],
                "warning": "PARTIAL estimate, not complete pipeline FLOPs; whole-pipeline Infer(ms)/Mem(MB) include every operation"}
    return Count()


def _model_hash(stage):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(stage.state_dict().items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            h.update(str((tuple(value.shape), str(value.dtype))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered)-1)*q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)


def profile_stage(stage, sources, *, device, warmup=3, repeats=10, rows=None,
                  deadline_unix=None, gpu_isolation_assertion=None,
                  expected_shapes=None):
    """Measure resident-input batch1 RR+FR; never modifies live trainer state.

    GPU calls require a caller-provided isolation assertion. The launcher owns
    acquiring/releasing its lease; this routine never sends process signals.
    ``expected_shapes`` is overridable only for explicit CPU unit tests.
    Memory is this profiling process's peak allocated bytes, including model,
    resident inputs and output/workspace; other processes and reserved-cache
    memory are excluded and reported separately.
    """
    import torch
    dev = torch.device(device)
    rows = {"rr": 0, "fr": 0} if rows is None else dict(rows)
    if (type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1
            or set(rows) != {"rr", "fr"} or any(type(v) is not int or not 0 <= v < 20 for v in rows.values())):
        raise ValueError("Pin warmup, positive repeats and one official scene per RR/FR")
    expected_shapes = expected_shapes or {"rr": (1, 1, 256, 256), "fr": (1, 1, 512, 512)}
    if dev.type == "cuda" and (not gpu_isolation_assertion or expected_shapes != {"rr": (1, 1, 256, 256), "fr": (1, 1, 512, 512)}):
        raise ValueError("Production GPU profile requires caller isolation and official RR256/FR512")
    if any(p.dtype != torch.float32 or p.device != dev for p in stage.parameters()):
        # CUDA's unindexed spelling denotes the current device.
        actual = torch.device("cuda", torch.cuda.current_device()) if dev.type == "cuda" and dev.index is None else dev
        if any(p.dtype != torch.float32 or p.device != actual for p in stage.parameters()):
            raise ValueError("Stage must already reside in requested FP32 device")
    def check():
        if deadline_unix is not None and time.time() >= deadline_unix:
            raise TimeoutError("Cost profiling deadline exceeded; no COMPLETE profile")
    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    def call(batch):
        check()
        output = stage.native_prediction(batch)
        if not isinstance(output, dict) or "y" not in output:
            raise ValueError("Full native stage output required")
        return output
    started = time.monotonic()
    before_hash = _model_hash(stage)
    modes = [(m, m.training) for m in stage.modules()]
    params = sum(p.numel() for p in stage.parameters())
    result = {"schema_version": SCHEMA_VERSION, "profile_protocol": PROFILE_PROTOCOL,
        "status": "RUNNING", "weight_basis": "FRESH_ARCHITECTURE_NOT_CHECKPOINT",
        "default_sheet_split": "rr", "batch": 1, "precision": "FP32",
        "warmup": warmup, "repeats": repeats, "optimizer_updates": 0,
        "model_state_sha256_before": before_hash, "gpu_isolation_assertion": gpu_isolation_assertion,
        "observations": {}, "params": params, "trainable_params": sum(p.numel() for p in stage.parameters() if p.requires_grad),
        "parameter_components": {k: sum(p.numel() for p in getattr(stage, k).parameters())
            for k in ("aligner", "teacher", "matcher") if hasattr(stage, k)},
        "environment": {"torch": str(torch.__version__), "device": str(dev),
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cpu_threads": torch.get_num_threads(), "python": platform.python_version()},
        "weight_dependence": "Architecture-level fresh-weight timing; not checkpoint-specific latency. EdgeMatch uses actual selected native scene, with no result cache.",
        "memory_definition": "Per-process torch peak allocated MiB, including A+T+resident inputs+output+workspace; not GPU process RSS or training high-water mark",
        "no_artifacts_saved": True}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        stage.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=False):
            for split, row in rows.items():
                check()
                sync()
                tick = time.perf_counter()
                batch = sources.batch(split, [row], device=device)
                sync()
                load_ms = (time.perf_counter()-tick)*1000
                if tuple(batch.pan.shape) != tuple(expected_shapes[split]) or batch.pan.dtype != torch.float32:
                    raise ValueError("Official batch1 FP32 observation shape differs from cost contract")
                for _ in range(warmup):
                    call(batch)
                sync()
                baseline_bytes = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else None
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                timings = []
                for _ in range(repeats):
                    check()
                    sync()
                    tick = time.perf_counter()
                    output = call(batch)
                    sync()
                    timings.append((time.perf_counter()-tick)*1000)
                    del output
                peak_bytes = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
                reserved_bytes = torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None
                counter = _counter_mode()
                module_macs = {"convolution_MAC": 0, "matrix_MAC": 0}
                def verify_hook(layer, inputs, output):
                    if isinstance(layer, torch.nn.Linear):
                        module_macs["matrix_MAC"] += output.numel()*layer.in_features
                    elif isinstance(layer, torch.nn.ConvTranspose2d):
                        module_macs["convolution_MAC"] += inputs[0].numel()*layer.weight[0].numel()
                    else:
                        module_macs["convolution_MAC"] += output.numel()*layer.weight[0].numel()
                handles = [m.register_forward_hook(verify_hook) for m in stage.modules()
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear))]
                try:
                    with counter:
                        counted = call(batch)
                finally:
                    for handle in handles:
                        handle.remove()
                del counted
                accounting = counter.report()
                if module_macs["convolution_MAC"] <= 0 or any(
                        accounting["components_MAC_equivalent"].get(key, 0) < count
                        for key, count in module_macs.items()):
                    raise RuntimeError("Operator accounting missed executed CNN/Linear modules")
                accounting["module_MAC_crosscheck"] = module_macs
                h, w = batch.pan.shape[-2:]
                scope = (f"{split.upper()}{h}x{w} batch1 FP32 resident-input full native A+warp+Teacher"
                    + (" +uncached EdgeMatch+prewarp" if hasattr(stage, "matcher") else " direct-CNN")
                    + "; fresh architecture, not checkpoint; MAC-scale PARTIAL Conv/MatMul+resampling estimate"
                    + "; peak allocated MiB incl model/input/output; excludes H2D/data read; no normalization/activation/search arithmetic in MAC estimate")
                sheet = {"Params(M)": params/1e6, "FLOPs(G)": accounting["mac_equivalent"]/1e9,
                    "Infer(ms)": statistics.median(timings),
                    "Mem(MB)": None if peak_bytes is None else peak_bytes/(2**20), "Cost_scope": scope}
                result["observations"][split] = {"sample_id": batch.lineage.sample_ids[0],
                    "scene_row": row, "pan_shape": list(batch.pan.shape),
                    "ms_shape": list(batch.ms.shape), "native_lms_shape": list(batch.native_lms.shape),
                    "observation_read_normalize_H2D_ms": load_ms,
                    "full_pipeline": {"median_ms": statistics.median(timings), "mean_ms": statistics.mean(timings),
                        "p90_ms": _quantile(timings, .9), "samples_ms": timings},
                    "memory": {"peak_allocated_bytes": peak_bytes, "baseline_allocated_bytes": baseline_bytes,
                        "incremental_peak_bytes": None if peak_bytes is None else peak_bytes-baseline_bytes,
                        "peak_reserved_bytes": reserved_bytes, "CPU_memory_not_measured": dev.type != "cuda"},
                    "accounting": accounting, "sheet_fields": sheet}
                del batch
    finally:
        for module, training in modes:
            module.training = training
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        torch.backends.cudnn.benchmark = old_benchmark
    result["model_state_sha256_after"] = _model_hash(stage)
    if result["model_state_sha256_after"] != before_hash:
        raise RuntimeError("Read-only cost profiling changed model state")
    result["elapsed_seconds"] = time.monotonic()-started
    result["sheet_cost"] = result["observations"]["rr"]["sheet_fields"]
    result["status"] = "COMPLETE"
    json.dumps(result, allow_nan=False)
    return result
