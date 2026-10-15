"""T0 Teacher 를 WV3 reduced-resolution 20 장면에 돌려 framework 그림용 산출물을 만든다."""
import os, sys, json
import numpy as np, h5py, yaml, torch

ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from safetensors.torch import load_file
from main import import_class
from kdv.teacher_assets import skeleton_from_cfg

T0 = os.path.join(ROOT, "assets", "pakd50", "T0_run")
cfg = yaml.safe_load(open(os.path.join(T0, "meta", "config.yaml")))
m, info = skeleton_from_cfg(cfg, import_class(cfg["model"]))
sd = load_file(os.path.join(T0, "best_hqnr", "model.safetensors"))
m.load_state_dict(sd, strict=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev).eval().requires_grad_(False)
print("margin", info["margin"], "dev", dev, "bands", cfg["num_bands"])

h5 = os.path.join(ROOT, "data/PanCollection/WV3/reduced_examples_h5_regen/test_wv3_multiExm1.h5")
with h5py.File(h5) as f:
    gt = np.asarray(f["gt"], dtype=np.float32); ms = np.asarray(f["ms"], dtype=np.float32)
    pan = np.asarray(f["pan"], dtype=np.float32)
R = 2047.0
out = {}
deltas = []
with torch.no_grad():
    for i in range(len(gt)):
        p = torch.from_numpy(pan[i])[None].to(dev) / R * 2 - 1
        s = torch.from_numpy(ms[i])[None].to(dev) / R * 2 - 1
        o = m(p, s)
        deltas.append(o["delta"][0].cpu().numpy())
        out[i] = dict(y=o["y"][0].cpu().numpy(), ms_base=o["ms_base"][0].cpu().numpy(),
                      pan_aligned=o["pan_aligned"][0, 0].cpu().numpy())
np.savez_compressed(os.path.join(os.path.dirname(__file__), "t0_rr.npz"),
                    gt=gt, ms=ms, pan=pan, delta=np.stack(deltas),
                    y=np.stack([out[i]["y"] for i in out]),
                    ms_base=np.stack([out[i]["ms_base"] for i in out]),
                    pan_aligned=np.stack([out[i]["pan_aligned"] for i in out]))
d = np.stack(deltas)
print("delta median (dy,dx)", np.median(d, 0), " |d| median", np.median(np.linalg.norm(d, axis=1)))
