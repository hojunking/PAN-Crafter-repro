#!/usr/bin/env python
"""Teacher 신호 cache (계획 §5.5): train 전 표본에 대해 R_T=Y_T−LMS(fp16), U_T(fp16, [0,1]), δ_T, c_T 를 한 번 저장.

  python tools/uvs_build_cache.py --teacher c0_hqnr [--out outputs/uvs_cache/c0_hqnr_wv3_train.npz]

MS mode 는 aligned PAN(ĉ_T δ_T, gate 0.35)으로 teacher(native 경로) forward. 증강 없이 원본 격자에서 계산하고,
feeder(feeders/feeder_uvs.py)가 로드 시 같은 flip/rot 로 변환한다.
"""
import argparse, hashlib, json, os, sys
import h5py, numpy as np, torch, torch.nn.functional as F, yaml
from safetensors.torch import load_file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                                   # noqa: E402
from uvs.shift import ShiftModule, edge_rep, warp, warp_pan_channels, gated_delta   # noqa: E402
from uvs.losses import percentile_normalize                                     # noqa: E402
from kd.features import WithUncertainty                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="c0_hqnr"); ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--out", default=None); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd = os.path.join(ROOT, "work_dir", a.teacher); tdir = os.path.join(wd, "uvs_teacher")
    gate = json.load(open(os.path.join(tdir, "gate.json"))); norm = json.load(open(os.path.join(tdir, "norm.json")))
    cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    bb = import_class(cfg["model"])(**cfg["model_args"]); bb.load_state_dict(load_file(os.path.join(wd, a.ckpt, "model.safetensors")))
    unc = WithUncertainty(bb, cfg["model_args"].get("hidden_size", 128), head_out="logvar").to(dev).eval()
    unc.head.load_state_dict(load_file(os.path.join(tdir, "unc.safetensors")))
    shift = ShiftModule((16, 32, 32), 3, 0.07).to(dev).eval(); shift.load_state_dict(load_file(os.path.join(tdir, "shift.safetensors")))
    thr = float(norm.get("conf_threshold", 0.35))
    with h5py.File(os.path.join(ROOT, "data/PanCollection/WV3/train_wv3.h5")) as f:
        N = f["ms"].shape[0]; ms_all = f["ms"][:] / 1023.5 - 1; pan_all = f["pan"][:] / 1023.5 - 1; lms_all = f["lms"][:] / 1023.5 - 1
    with h5py.File(os.path.join(ROOT, "data/PanCollection/WV3/train_wv3_pan.h5")) as f:
        lp_all = f["lpan"][:] / 1023.5 - 1
    r_t = np.zeros((N, 8, 64, 64), np.float16); u_t = np.zeros((N, 1, 64, 64), np.float16)
    d_t = np.zeros((N, 2), np.float32); c_t = np.zeros((N,), np.float32); dr_t = np.zeros((N, 2), np.float32)
    with torch.no_grad():
        for i in range(0, N, a.batch):
            ms, pan, lms, lp = (torch.tensor(x[i:i + a.batch], dtype=torch.float32, device=dev) for x in (ms_all, pan_all, lms_all, lp_all))
            o = shift(edge_rep(lp), edge_rep(ms)); d = gated_delta(o["delta"], o["conf"], thr)
            lpu = F.interpolate(lp, scale_factor=4, mode="bicubic")
            pan_a, _, _ = warp_pan_channels(pan, lpu, pan - lpu, d, "bicubic"); lp_a = warp(lp, d, 1.0, "bicubic")
            sw = torch.ones(ms.shape[0], device=dev)
            y = unc.base(pan_a, lp_a, ms, sw) + F.interpolate(ms, scale_factor=4, mode="bicubic")
            theta = torch.exp(unc.theta())
            r_t[i:i + a.batch] = (y - lms).cpu().numpy().astype(np.float16)
            u_t[i:i + a.batch] = percentile_normalize(theta, norm["theta_q10"], norm["theta_q90"]).cpu().numpy().astype(np.float16)
            d_t[i:i + a.batch] = o["delta"].cpu().numpy(); c_t[i:i + a.batch] = o["conf"].cpu().numpy(); dr_t[i:i + a.batch] = d.cpu().numpy()
            if (i // a.batch) % 50 == 0: print(f"  {i}/{N}", end="\r")
    out = a.out or os.path.join(ROOT, "outputs/uvs_cache", f"{a.teacher}_wv3_train.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, r_t=r_t, u_t=u_t, delta_t=d_t, c_t=c_t, delta_applied=dr_t)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    meta = dict(teacher=a.teacher, ckpt=a.ckpt, n=int(N), sha256=h, norm=norm, gate_pass=gate.get("pass"),
                stats=dict(c_t_mean=float(c_t.mean()), gated_zero_ratio=float((np.abs(dr_t).sum(1) == 0).mean()),
                           delta_mag_p50=float(np.median(np.linalg.norm(d_t, axis=1))), u_t_mean=float(u_t.astype(np.float32).mean()),
                           r_t_abs_mean=float(np.abs(r_t.astype(np.float32)).mean())))
    json.dump(meta, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(f"\n[cache] {out}  {os.path.getsize(out) / 2**20:.0f} MB  sha {h[:8]}  c_T {meta['stats']['c_t_mean']:.2f}  gated0 {meta['stats']['gated_zero_ratio']:.2f}  |δ| p50 {meta['stats']['delta_mag_p50']:.3f}")


if __name__ == "__main__":
    main()
