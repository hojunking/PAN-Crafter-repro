"""PAN aligner 가 실제로 무엇을 바꾸는가 — 가장 크게 이동한 scene 의 육안 확인.

sub-pixel 이동은 나란히 놓으면 보이지 않는다. 그래서
  (1) 확대 crop 전/후, (2) 차분 지도(정합된 PAN − 원본 PAN), (3) 경계 단면 프로파일,
  (4) 모델 출력의 차분 — 네 가지로 보인다.

대상은 FR 논문 세트에서 |c| 가 최대인 scene:
  T0 (lam*=1e-4)  scene 19  |c| 0.564 px   <- 실제 배포 Teacher
  donor N2        scene 18  |c| 1.839 px   <- 가장 강한 aligner (육안 확인용)

산출 → results_log/assets/0917_aligner_shift_example.png
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from tools.eqrec4 import common as CM                      # noqa: E402
from pa.warp import warp_pan                                # noqa: E402
from main import import_class                               # noqa: E402

OUT = os.path.join(ROOT, "results_log", "assets")
CASES = [("L1E4", 2025, "best_raw", "T0   lambda* = 1e-4  (deployed teacher)"),
         ("N2", 2025, "last", "donor N2  (strongest aligner)")]
Z = 72                                                      # 확대 crop 한 변


def best_window(g, z=Z, margin=40):
    """경사 에너지가 가장 큰 z x z 창 (가장자리 margin 제외)."""
    H, W = g.shape
    best, bij = -1, (H // 2, W // 2)
    for i in range(margin, H - margin - z, 16):
        for j in range(margin, W - margin - z, 16):
            v = g[i:i + z, j:j + z].sum()
            if v > best:
                best, bij = v, (i, j)
    return bij


def stretch(a, lo=1, hi=99):
    p, q = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p) / (q - p + 1e-12), 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = CM.localize_cfg(yaml.safe_load(open(os.path.join(
        ROOT, "work_dir/PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/meta/config.yaml"))))
    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["test_full_feeder_args"])

    fig = plt.figure(figsize=(18.2, 8.4))
    gs = GridSpec(2, 5, hspace=.50, wspace=.40, width_ratios=[1, 1, 1, 1, 1.55])

    for r, (fam, seed, tag, lab) in enumerate(CASES):
        L = CM.load_model(fam, seed, tag)
        C = []
        with torch.no_grad():
            for i in range(len(ds)):
                lms, ms, lpan, pan = [x.unsqueeze(0).to(CM.DEV) for x in ds[i]]
                C.append(L.m(pan, ms, lpan)["delta"][0].float().cpu().numpy())
        C = np.array(C)
        k = int(np.linalg.norm(C, axis=1).argmax())
        c = C[k]

        lms, ms, lpan, pan = [x.unsqueeze(0).to(CM.DEV) for x in ds[k]]
        with torch.no_grad():
            o_on = L.m(pan, ms, lpan)
            o_off = L.m(pan, ms, lpan, aligner_enabled=False)
        p0 = pan[0, 0].float().cpu().numpy()
        p1 = o_on["pan_aligned"][0, 0].float().cpu().numpy()
        y_on = o_on["y"][0].float().cpu().numpy().mean(0)
        y_off = o_off["y"][0].float().cpu().numpy().mean(0)

        gx = np.gradient(p0, axis=1); gy = np.gradient(p0, axis=0)
        i0, j0 = best_window(np.sqrt(gx ** 2 + gy ** 2))
        sl = (slice(i0, i0 + Z), slice(j0, j0 + Z))

        a = fig.add_subplot(gs[r, 0])
        a.imshow(stretch(p0[sl]), cmap="gray"); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(f"{lab}\nscene #{k}   |c| = {np.linalg.norm(c):.3f} px\n"
                     f"(dy {c[0]:+.3f}, dx {c[1]:+.3f})", fontsize=8.5)
        if r == 0:
            a.set_title("PAN  original  (zoom)", fontsize=10)

        a = fig.add_subplot(gs[r, 1])
        a.imshow(stretch(p1[sl]), cmap="gray"); a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title("PAN  after aligner  (zoom)\nlooks identical to the eye", fontsize=10)

        a = fig.add_subplot(gs[r, 2])
        d = (p1 - p0)[sl]
        v = np.percentile(np.abs(d), 99.5)
        im = a.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=.046, pad=.02)
        if r == 0:
            a.set_title("difference  (aligned - original)\nthe shift lives on edges", fontsize=10)

        a = fig.add_subplot(gs[r, 3])
        dy_ = (y_on - y_off)[sl]
        v2 = np.percentile(np.abs(dy_), 99.5)
        im = a.imshow(dy_, cmap="PuOr_r", vmin=-v2, vmax=v2)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=.046, pad=.02)
        if r == 0:
            a.set_title("model OUTPUT difference\n(aligner on - off), band mean", fontsize=10)

        # 가장 가파른 행을 골라 단면 프로파일
        a = fig.add_subplot(gs[r, 4])
        seg0 = p0[sl]; seg1 = p1[sl]
        row = int(np.abs(np.diff(seg0, axis=1)).sum(1).argmax())
        xs = np.arange(Z)
        a.plot(xs, seg0[row], "-", lw=1.6, color="#4C78A8", label="original, same row")
        a.plot(xs, seg1[row], "-", lw=2.6, color="#E45756", alpha=.55, label="after aligner")
        # 원본을 (y+dy, x+dx) 에서 재샘플 -> 이동이 순수 평행이동이면 위 곡선과 겹친다
        rs = map_coordinates(p0, [np.full(Z, i0 + row + c[0]), j0 + xs + c[1]],
                             order=3, mode="nearest")
        a.plot(xs, rs, ":", lw=1.5, color="#111",
               label=r"original resampled at $(y+dy,\,x+dx)$")
        rho = float(np.corrcoef(rs, seg1[row])[0, 1])
        rho_same = float(np.corrcoef(seg0[row], seg1[row])[0, 1])
        j = int(np.abs(np.diff(seg0[row])).argmax())
        lo_, hi_ = max(0, j - 11), min(Z - 1, j + 11)
        if hi_ - lo_ < 12:
            lo_, hi_ = max(0, min(lo_, Z - 23)), min(Z - 1, max(hi_, 22))
        a.set_xlim(lo_, hi_)
        a.legend(fontsize=7.2, loc="best")
        a.set_xlabel("column (px)", fontsize=8.5)
        a.set_ylabel("PAN value", fontsize=8.5, labelpad=1)
        a.tick_params(labelsize=7.5)
        a.set_title(f"cut along row {row}:  it IS a pure translation\n"
                    f"resampled vs aligned r={rho:.4f}   (same row only r={rho_same:.2f})",
                    fontsize=9.2, pad=6)

        print(f"{lab}: scene {k}  |c|={np.linalg.norm(c):.3f}  "
              f"PAN diff RMS={np.sqrt((d**2).mean()):.4f}  "
              f"output diff RMS={np.sqrt((dy_**2).mean()):.4f}")

    fig.suptitle("What the PAN aligner actually does — the most-shifted scene of the FR paper set "
                 "(72x72 zoom on the strongest-gradient window)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .91])
    p = os.path.join(OUT, "0917_aligner_shift_example.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
