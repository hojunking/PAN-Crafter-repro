"""Internal isolated-process numerical reader; invoked by historical.run_parity.

The controlling importer validates original bytes/source/config before calling
this reader. Loading model tensors strictly is not an exact-resume operation.
"""
import json
from pathlib import Path
import sys

source, request_path, output_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fh12.model import build_model, state_hash
from fh12.losses import teacher_loss, student_losses
from fh12.calibration import axis16_q
from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
from tools.metrics.eval_rr import ergas, sam
from tools.metrics.q2n import q2n
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s


def read_split(spec, count, device):
    with h5py.File(spec["dataroot"], "r") as h:
        native = {key: h[key][:count] for key in ("pan", "ms", "lms") + (("gt",) if "gt" in h else ())}
    with h5py.File(spec["lpan_path"], "r") as h:
        native["lpan"] = h["lpan"][:count]
    tensors = {key: torch.from_numpy(x.astype(np.float32)).to(device).mul(2 / 2047).sub(1)
               for key, x in native.items()}
    return native, tensors


def main():
    req = json.loads(request_path.read_text())
    device = torch.device(req["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Parity requested CUDA but CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = bool(req["probe_runtime_flags"]["tf32_matmul"])
    torch.backends.cudnn.allow_tf32 = bool(req["probe_runtime_flags"]["tf32_cudnn"])
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load(Path(req["config"]).read_text())
    f, m = cfg["fh12"], cfg["model_args"]
    model, init = build_model(f["input_layout"], m["hidden_size"], m["depth"], cfg["seed"], role="T")
    initial_hash = state_hash(model.state_dict())
    model.load_state_dict(load_file(req["checkpoint"]), strict=True)
    model = model.to(device).eval().requires_grad_(False)
    result = {}
    train = None
    for split, count, shape in (("train", 8, 64), ("rr", 2, 256), ("fr", 2, 512)):
        raw, batch = read_split(req["splits"][split], count, device)
        if batch["pan"].shape != (count, 1, shape, shape):
            raise ValueError(f"Invalid parity subset geometry for {split}")
        outputs = []
        with torch.no_grad():
            for index in range(count):
                outputs.append(model(batch["pan"][index:index+1], batch["ms"][index:index+1], batch["lpan"][index:index+1]))
        out = {key: torch.cat([row[key] for row in outputs]) for key in ("y", "delta", "ms_base")}
        result[f"{split}_y"] = out["y"].cpu().numpy()
        result[f"{split}_c"] = out["delta"].cpu().numpy()
        if split == "train":
            train = (batch, out)
        else:
            pred = ((out["y"].clamp(-1, 1) + 1) * 1023.5).cpu().numpy().transpose(0, 2, 3, 1).astype(np.float64)
            if split == "rr":
                rows = []
                for image, target in zip(pred, raw["gt"].transpose(0, 2, 3, 1).astype(np.float64)):
                    a, b = image[20:-21, 20:-21], target[20:-21, 20:-21]
                    rows.append([ergas(a, b), sam(a, b), scc_dlpan(a, b), psnr_global(a, b, 2047),
                                 ssim_skimage(a, b, 2047), q2n(b, a, 32, 32)[0]])
            else:
                wald = load_dlpan(req["dlpan"])
                rows = []
                for image, lms, pan in zip(pred, raw["lms"].transpose(0, 2, 3, 1), raw["pan"][:, 0]):
                    dl = d_lambda_k(image, lms.astype(np.float64), "wv3", 4, 32, wald)
                    ds = d_s(image, lms.astype(np.float64), pan.astype(np.float64), 4, 32, wald)
                    rows.append([dl, ds, (1-dl)*(1-ds)])
            result[f"{split}_official_metrics"] = np.asarray(rows)
    batch, out = train
    for rotation in range(4):
        pan, ms = (torch.rot90(batch[key].flip((-2, -1)), rotation, (-2, -1)) for key in ("pan", "ms"))
        with torch.no_grad():
            q, per_radius, native = axis16_q(model, pan, ms)
        result[f"q_rot{rotation}"] = q.cpu().numpy()
        result[f"q_radius_rot{rotation}"] = per_radius.cpu().numpy()
    for update in (0, 1):
        terms = teacher_loss(model, out, batch["gt"], batch["pan"], batch["ms"], update,
                             torch.Generator().manual_seed(1234))
        result[f"teacher_loss_{update}"] = np.asarray([float(terms[k]) for k in ("total", "rec", "off")])
    y = (out["y"] + .01).detach().requires_grad_(True)
    with np.load(req["q_cache"], allow_pickle=False) as cache:
        weights = req["q_ref"] / (req["q_ref"] + cache["q"][:8, 0])
    losses = student_losses({"y": y}, out, batch["gt"], req["tau_R"], torch.tensor(weights, device=device))
    result["student_loss"] = np.asarray([float(losses[k].detach()) for k in ("L_U", "L_A", "hard", "soft", "edge")])
    result["student_grad_u_y"] = torch.autograd.grad(losses["L_U"], y, retain_graph=True)[0].cpu().numpy()
    result["student_grad_a_y"] = torch.autograd.grad(losses["L_A"], y)[0].cpu().numpy()
    np.savez_compressed(output_path, **result)
    output_path.with_suffix(".json").write_text(json.dumps({
        "source_root": str(source), "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "device": str(device), "step0_state_hash": initial_hash,
        "init_policy": init.get("policy", init.get("init_policy")), "precision": "fp32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32}, indent=2))


if __name__ == "__main__":
    main()
