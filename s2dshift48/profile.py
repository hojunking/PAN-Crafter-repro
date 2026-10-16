"""Measured parameter count, executed Conv/Linear MACs and CPU-only latency.

Not an end-to-end training FLOP/throughput estimator. Normalization, nonlinear
functions, memory movement, validation checks and PAN warp are NOT in MACs.
Forward latency includes the real Aligner's validation checks, but not warp.
"""
import platform
import statistics
import time

import torch
from torch import nn

from directshift48.method import DirectPANAligner
from .aligner import S2DGlobalPANAligner


def _executed_macs(model, pan, lms):
    rows, handles = [], []
    names = {id(module): name for name, module in model.named_modules()}

    def record(module, args, output):
        if isinstance(module, nn.Conv2d):
            per_element = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            per_element = module.in_features
        rows.append({'name': names[id(module)], 'type': type(module).__name__,
                     'output_shape': list(output.shape), 'macs': output.numel() * per_element})

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(record))
        with torch.no_grad():
            model(pan, lms)
    finally:
        for handle in handles:
            handle.remove()
    return sum(row['macs'] for row in rows), rows


def compare(*, bands=8, sides=(64, 256, 512), repeats=10, warmup=3, threads=2):
    if (type(bands) is not int or bands not in (4, 8)
            or type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0
            or type(threads) is not int or threads < 1 or not sides
            or any(type(n) is not int or n < 24 or n % 4 for n in sides)):
        raise ValueError('4/8 bands, valid S2D input sizes and explicit positive CPU profiling limits required')
    previous_threads = torch.get_num_threads()
    rows = []
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(20261016)
            models = [('DirectShift', DirectPANAligner(bands).eval()),
                      ('S2D4', S2DGlobalPANAligner(bands).eval())]
            for side in sides:
                pan, lms = torch.randn(1, 1, side, side), torch.randn(1, bands, side, side)
                for name, model in models:
                    macs, modules = _executed_macs(model, pan, lms)
                    for _ in range(warmup):
                        model(pan, lms)
                    times = []
                    for _ in range(repeats):
                        tick = time.perf_counter()
                        model(pan, lms)
                        times.append((time.perf_counter() - tick) * 1000)
                    rows.append({'architecture': name, 'bands': bands, 'batch': 1, 'height': side,
                        'width': side, 'parameters': sum(p.numel() for p in model.parameters()),
                        'conv_linear_macs': macs, 'conv_linear_flops_2mac': 2 * macs,
                        'cpu_latency_ms_median': statistics.median(times),
                        'cpu_latency_ms_min': min(times), 'latency_ms_samples': times,
                        'module_costs': modules, 'manifest': model.manifest()})
    finally:
        torch.set_num_threads(previous_threads)
    return {'scope': 'UNTRAINED_ALIGNER_CPU_ONLY_NOT_QUALITY_OR_GPU_THROUGHPUT',
            'mac_convention': 'Conv2d and Linear only; one multiply-accumulate = one MAC = two FLOPs',
            'excluded_from_macs': ['normalization', 'activation', 'bias_add', 'pooling', 'znorm',
                                   'pixel_unshuffle/memory', 'finite_guards', 'PAN_warp', 'Teacher/Student'],
            'latency_scope': 'actual forward including guards; no PAN warp, no Teacher/Student',
            'torch_version': torch.__version__, 'platform': platform.platform(),
            'cpu_threads': threads, 'warmup': warmup, 'repeats': repeats, 'rows': rows}
