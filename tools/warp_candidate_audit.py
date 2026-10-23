"""CPU-only interpolation audit; no training, GPU, or official RR/FR evaluation.

Example:
  python tools/warp_candidate_audit.py --out results_log/assets/2026-10-23_warp_candidate_audit.json

Analytic kernel responses and synthetic one-way targets supplement the existing
PAN round-trip probe.  An interpolation operator can be nearly reversible yet
give the wrong one-way answer, so round-trip error alone cannot rank methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa.warp import warp_pan  # noqa: E402
from panda_warp_candidates.ops import METHODS, candidate_manifest, warp_global  # noqa: E402


FREQUENCIES = (.05, .1, .15, .25, .375, .45)
PRIMARY_SOURCES = [
    {"title": "Quantitative comparison of sinc-approximating kernels for medical image interpolation",
     "url": "https://research.tue.nl/en/publications/quantitative-comparison-of-sinc-approximating-kernels-for-medical/"},
    {"title": "Mastering Windows: Improving Reconstruction",
     "url": "https://www.cg.tuwien.ac.at/research/vis/vismed/Windows/"},
    {"title": "Boundary effect free and adaptive discrete signal sinc-interpolation algorithms for signal and image resampling",
     "url": "https://opg.optica.org/ao/abstract.cfm?uri=ao-42-20-4166"},
    {"title": "SciPy ndimage.shift: spline interpolation and boundary conventions",
     "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html"},
]


def kernel_weights(kind, fraction=.5):
    if kind == "bicubic":
        j = np.arange(-1, 3)
        z = np.abs(fraction - j)
        a = -.75
        w = np.where(z < 1, (a + 2) * z**3 - (a + 3) * z**2 + 1,
                     np.where(z < 2, a * z**3 - 5 * a * z**2 + 8 * a * z - 4 * a, 0.))
    elif kind.startswith("lanczos"):
        radius = int(kind[7:])
        j = np.arange(-radius + 1, radius + 1)
        z = fraction - j
        w = np.sinc(z) * np.sinc(z / radius) * (np.abs(z) < radius)
    elif kind.startswith("kaiser"):
        taps, beta = kind[6:].split("_beta")
        radius, beta = int(taps) // 2, float(beta)
        j = np.arange(-radius + 1, radius + 1)
        z = fraction - j
        w = (np.sinc(z) * np.i0(beta * np.sqrt(np.maximum(1 - (z / radius)**2, 0)))
             / np.i0(beta) * (np.abs(z) < radius))
    else:
        raise ValueError(kind)
    return j, w / w.sum()


def transfer_responses():
    kinds = ("bicubic", "lanczos4", "lanczos6", "kaiser6_beta4",
             "kaiser8_beta4", "kaiser8_beta6", "kaiser12_beta4", "kaiser12_beta6")
    result = {}
    for kind in kinds:
        j, w = kernel_weights(kind)
        result[kind] = {str(f): float(abs((w * np.exp(2j * np.pi * f * j)).sum())) for f in FREQUENCIES}
    # Cardinal B-splines require coefficient prefiltering; they are not the
    # same four-tap cubic convolution used by torch grid_sample.
    x = np.arange(4096)
    for order in (3, 5):
        response = {}
        for f in FREQUENCIES:
            q = ndimage.shift(np.cos(2 * np.pi * f * x), -.5, order=order,
                               mode="mirror", prefilter=True)[64:-64]
            ideal = np.cos(2 * np.pi * f * (x[64:-64] + .5))
            response[str(f)] = float(np.dot(q, ideal) / np.dot(ideal, ideal))
        result[f"spline{order}_prefiltered"] = response
    return result


def apply(image, delta, kind, *, padding=32):
    if kind == "bicubic":
        return warp_pan(image, delta)
    if kind == "spline5_prefiltered":
        # Offline CPU diagnostic only.  This SciPy path is not differentiable
        # through torch and is deliberately not an offered training operator.
        array = image[0, 0].numpy()
        dy, dx = delta[0].tolist()
        shifted = ndimage.shift(array, (-dy, -dx), order=5, mode="mirror", prefilter=True)
        return torch.from_numpy(shifted)[None, None]
    return warp_global(image, delta, method=kind, fft_padding=padding)


def synthetic_checks():
    x = torch.arange(256, dtype=torch.float64)
    kinds = ("bicubic", *METHODS, "spline5_prefiltered")
    result = {"oneway_sines": {}, "step_overshoot": {}}
    # A continuous sinusoid supplies a one-way target independent of the warp.
    for frequency in (.125, .45):
        source = torch.cos(2 * math.pi * frequency * x + .3)[None, None, None].expand(1, 1, 128, 256)
        ideal = torch.cos(2 * math.pi * frequency * (x + .5) + .3)
        delta = torch.tensor([[0., .5]], dtype=torch.float64)
        for kind in kinds:
            for padding in ((16, 32, 64) if kind == "fft" else (32,)):
                shifted = apply(source, delta, kind, padding=padding)
                roundtrip = apply(shifted, -delta, kind, padding=padding)
                row = shifted[0, 0, 64]
                rt = roundtrip[0, 0, 64]
                result["oneway_sines"][f"f{frequency}_{kind}_pad{padding}"] = {
                    "oneway_full_rmse": float((row - ideal).square().mean().sqrt()),
                    "oneway_interior16_rmse": float((row[16:-16] - ideal[16:-16]).square().mean().sqrt()),
                    "roundtrip_interior16_rmse": float((rt[16:-16] - source[0, 0, 64, 16:-16]).square().mean().sqrt()),
                }
    source = (x >= 128).double()[None, None, None].expand(1, 1, 128, 256)
    delta = torch.tensor([[0., .5]], dtype=torch.float64)
    for kind in kinds:
        shifted = apply(source, delta, kind)
        result["step_overshoot"][kind] = {"min": float(shifted.min()), "max": float(shifted.max())}
    source = ((-1.) ** x)[None, None, None].expand(1, 1, 128, 256)
    shifted = apply(source, delta, "fft")
    rt = apply(shifted, -delta, "fft")
    result["real_fft_nyquist_counterexample"] = {
        "source_rms": float(source.square().mean().sqrt()),
        "half_shift_output_rms": float(shifted.square().mean().sqrt()),
        "roundtrip_rmse": float((rt - source).square().mean().sqrt()),
        "interpretation": "Discarding the quadrature component at a half-pixel Nyquist shift loses information.",
    }
    coordinate = torch.arange(32, dtype=torch.float64)
    corner = ((-1.) ** (coordinate[:, None] + coordinate[None, :]))[None, None]
    delta = torch.tensor([[.25, .25]], dtype=corner.dtype)
    rotated_delta = torch.stack((-delta[:, 1], delta[:, 0]), dim=1)
    expected = torch.rot90(apply(corner, delta, "fft", padding=8), 1, (-2, -1))
    actual = apply(torch.rot90(corner, 1, (-2, -1)), rotated_delta, "fft", padding=8)
    result["joint_2d_nyquist_rotation_counterexample"] = {
        "input": "unit-amplitude Nyquist checkerboard", "shift": [.25, .25],
        "rotation_commutation_rmse": float((actual - expected).square().mean().sqrt()),
        "interpretation": "The joint .real projection gives cos(pi*(dy+dx)) at the Nyquist corner, not the separable cos(pi*dy)*cos(pi*dx).",
        "future_policy_candidate": "Axis-wise real-compatible Nyquist coefficients, explicitly tested against this probe policy.",
    }
    return result


def supplied_probe_audit():
    path = ROOT / "results_log/assets/2026-10-23_s1_warp_error_probe.json"
    if not path.exists():
        return {"available": False}
    raw = path.read_bytes()
    data = json.loads(raw)
    baseline = data["M"]["bicubic"]["0.0"]["ERGAS"]
    return {
        "available": True, "source": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(), "baseline_ERGAS": baseline,
        "percent_ERGAS_increase": {
            kind: {magnitude: 100 * (row["ERGAS"] / baseline - 1) for magnitude, row in values.items()}
            for kind, values in data["M"].items()},
        "diagonal_displacements": {
            str(magnitude): {"axis_magnitude": magnitude / math.sqrt(2),
                             "distance_to_nearest_axis_integer": abs(magnitude / math.sqrt(2) - round(magnitude / math.sqrt(2)))}
            for magnitude in (.5, 1., 1.5, 2.)},
        "fft_FR_roundtrip_relative_RMSE_at_axis_half_pixel": data["S"]["roundtrip"]["fft"]["0.5"]["fr"]["rel_rmse"],
        "notes": [
            "Reported 2px is diagonal vector magnitude; both axes are 1.4142px, not integers.",
            "Reported 1.5px diagonal has .06066 axis fractional distance; bicubic error is small, not zero.",
            "The source hf_energy_retained computes a norm ratio, not squared energy ratio.",
            "Inference-only FFT swaps cannot separate geometric effects from train/inference distribution mismatch.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (ROOT / "work_dir").resolve() in (args.out.resolve(), *args.out.resolve().parents):
        raise SystemExit("audit output must be outside work_dir")
    torch.set_num_threads(1)
    files = (Path(__file__).resolve(), ROOT / "panda_warp_candidates/ops.py", ROOT / "pa/warp.py")
    result = {
        "schema": "PANDA_WARP_CANDIDATE_CPU_AUDIT_V1", "device": "cpu", "dtype": "float64",
        "scope": "Synthetic operator diagnostics; no new PAN RR/FR model result and no training change.",
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "candidate_manifests": {kind: candidate_manifest(kind) for kind in METHODS},
        "single_half_pixel_amplitude_response": transfer_responses(),
        "synthetic_checks": synthetic_checks(), "supplied_probe_audit": supplied_probe_audit(),
        "primary_sources": PRIMARY_SOURCES,
        "limitations": [
            "FFT exact phase magnitude applies to the complex periodic transform, not cropped real-valued reflect-padded images.",
            "A round trip is not a ground-truth test of one-way interpolation or physical sensor registration.",
            "Negative lobes can cause ringing/overshoot; sharper spectra alone do not establish better model metrics.",
            "Spline5 is an offline SciPy reference, not a differentiable production operator.",
            "Candidate methods are isolated and not imported by active training/evaluation campaigns.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "device": "cpu", "methods": list(METHODS),
                      "new_model_evaluations": 0, "training_changed": False}))


if __name__ == "__main__":
    main()
