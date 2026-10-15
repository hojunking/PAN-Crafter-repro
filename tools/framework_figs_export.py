"""framework 그림(Stage 1)의 회색 박스용 샘플 영상 — 실제 T0 Teacher/Aligner 산출물."""
import os, sys, csv, json
import numpy as np, yaml, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from safetensors.torch import load_file
from main import import_class
from kdv.teacher_assets import skeleton_from_cfg
from pa.warp import warp_pan

OUT = os.path.join(ROOT, "results_log", "paper_figs", "framework_figs")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t0_rr.npz")
R, RGB, N, UP = 2047.0, (4, 2, 1), 128, 4
EPS = (1.6, -1.2)                       # |eps| = 2.0 HR px — PO10 R200 의 corruption 반경과 같은 크기
SAMPLES = [("1_buildings", 4, 64, 112), ("2_cars", 19, 112, 96), ("3_city", 17, 24, 32)]


def st(a, lo, hi):
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def save(path, img):
    """[H,W] 또는 [H,W,3] in [0,1] -> x4 nearest PNG (원 픽셀 그대로, 보간 없음)."""
    a = np.repeat(np.repeat(img, UP, 0), UP, 1)
    plt.imsave(path, a, cmap=None if a.ndim == 3 else "gray", vmin=None if a.ndim == 3 else 0,
               vmax=None if a.ndim == 3 else 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(SRC)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "assets/pakd50/T0_run/meta/config.yaml")))
    m, info = skeleton_from_cfg(cfg, import_class(cfg["model"]))
    m.load_state_dict(load_file(os.path.join(ROOT, "assets/pakd50/T0_run/best_hqnr/model.safetensors")), strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval().requires_grad_(False)

    rows = []
    for name, s, y0, x0 in SAMPLES:
        d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
        sl = (slice(y0, y0 + N), slice(x0, x0 + N))
        pan = z["pan"][s, 0]                                   # [256,256] DN
        gt = z["gt"][s].transpose(1, 2, 0)                     # [256,256,8] DN
        msb = (z["ms_base"][s].transpose(1, 2, 0) + 1) / 2 * R
        yT = (z["y"][s].transpose(1, 2, 0) + 1) / 2 * R
        pa = (z["pan_aligned"][s] + 1) / 2 * R

        # Step 1-b: 같은 aligner 에 eps 만큼 옮긴 PAN 을 넣어 c_eps 를 잰다
        p = torch.from_numpy(pan)[None, None].to(dev) / R * 2 - 1
        sm = torch.from_numpy(z["ms"][s])[None].to(dev) / R * 2 - 1
        with torch.no_grad():
            eps = torch.tensor([EPS], dtype=torch.float32, device=dev)
            p_eps = warp_pan(p.float(), eps)
            msu = torch.nn.functional.interpolate(sm, scale_factor=4, mode="bicubic")
            c0 = m.predict_delta(p, msu)[0].cpu().numpy()
            ce = m.predict_delta(p_eps.to(p.dtype), msu)[0].cpu().numpy()
        pe = (p_eps[0, 0].cpu().numpy() + 1) / 2 * R

        plo, phi = np.percentile(pan[sl], [1, 99])             # PAN 계열 공통 스트레치
        g = gt[sl][..., RGB]                                   # 분광 계열: GT 기준 **밴드별** 스트레치
        clo = np.percentile(g, 1, axis=(0, 1)); chi = np.percentile(g, 99, axis=(0, 1))
        rgb = lambda a: st(a[sl][..., RGB], clo, chi)

        save(os.path.join(d, "P_pan.png"), st(pan[sl], plo, phi))
        save(os.path.join(d, "P_aligned.png"), st(pa[sl], plo, phi))
        save(os.path.join(d, "P_shifted.png"), st(pe[sl], plo, phi))
        save(os.path.join(d, "M_ms_up.png"), rgb(msb))
        save(os.path.join(d, "Y_gt.png"), rgb(gt))
        save(os.path.join(d, "YT_teacher.png"), rgb(yT))
        ms_lr = z["ms"][s].transpose(1, 2, 0)[y0 // 4:y0 // 4 + N // 4, x0 // 4:x0 // 4 + N // 4]
        save(os.path.join(d, "S_ms_native.png"), st(ms_lr[..., RGB], clo, chi))     # 표시 크기는 M 의 1/4

        rows.append(dict(sample=name, scene=s, crop_row=y0, crop_col=x0, crop_px=N, upscale=UP,
                         c0_dy=round(float(c0[0]), 4), c0_dx=round(float(c0[1]), 4),
                         c0_norm=round(float(np.linalg.norm(c0)), 4),
                         eps_dy=EPS[0], eps_dx=EPS[1],
                         ce_dy=round(float(ce[0]), 4), ce_dx=round(float(ce[1]), 4),
                         resid_dy=round(float(ce[0] + EPS[0] - c0[0]), 4),
                         resid_dx=round(float(ce[1] + EPS[1] - c0[1]), 4)))
        print(f"{name}: scene {s} ({y0},{x0})  c0=({c0[0]:+.3f},{c0[1]:+.3f}) |c0|={np.linalg.norm(c0):.3f}  "
              f"c_eps=({ce[0]:+.3f},{ce[1]:+.3f})  c_eps+eps-c0=({ce[0]+EPS[0]-c0[0]:+.3f},{ce[1]+EPS[1]-c0[1]:+.3f})")
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("->", OUT)


main()
