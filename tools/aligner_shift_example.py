"""PAN aligner 가 PAN 을 어디로 옮기는가 — 전/후 이미지와 이동 방향만.

가장 크게 이동한 scene 을 골라 확대 crop 의 before/after 를 크게 보이고,
화살표로 **PAN 내용이 움직인 방향**을 표시한다. (차분·단면 패널은 뺐다 — 2026-09-17 요청)

부호: warp 규약은 aligned[y,x] = original[y+dy, x+dx] 이므로 원본 (y+dy, x+dx) 의 내용이
      (y,x) 로 온다 = 내용은 (-dy, -dx) 만큼 움직인다. 화면 좌표(행이 아래로 증가)에서
      화살표 벡터는 (-dx, -dy).

화살표 길이는 보이도록 과장했다 — 실제 크기는 라벨에 적는다.

산출 → results_log/assets/0917_aligner_shift_example.png
"""
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from tools.eqrec4 import common as CM                      # noqa: E402
from main import import_class                               # noqa: E402

OUT = os.path.join(ROOT, "results_log", "assets")
CASES = [("L1E4", 2025, "best_raw", "T0   lambda* = 1e-4\n(deployed teacher)"),
         ("N2", 2025, "last", "donor N2\n(strongest aligner)")]
Z = 128                     # 확대 crop 한 변
EXAG = 14.0                 # 화살표 과장 배율 (실제 이동량은 라벨에)


def best_window(g, z=Z, margin=40):
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
    ds = import_class(cfg["feeder"])(**cfg["test_full_feeder_args"])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 12.4))

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
            o = L.m(pan, ms, lpan)
        p0 = pan[0, 0].float().cpu().numpy()
        p1 = o["pan_aligned"][0, 0].float().cpu().numpy()

        gx = np.gradient(p0, axis=1); gy = np.gradient(p0, axis=0)
        i0, j0 = best_window(np.sqrt(gx ** 2 + gy ** 2))
        sl = (slice(i0, i0 + Z), slice(j0, j0 + Z))

        ax_, ay_ = -float(c[1]), -float(c[0])          # 내용이 움직이는 방향 (화면 좌표)
        mag = float(np.hypot(c[0], c[1]))
        vert = "up" if ay_ < 0 else "down"
        horz = "right" if ax_ > 0 else "left"

        for col, (img, ttl) in enumerate([(p0, "BEFORE   original PAN"),
                                          (p1, "AFTER   aligner applied")]):
            a = axes[r, col]
            a.imshow(stretch(img[sl]), cmap="gray", interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(ttl, fontsize=13.5, pad=9)
            if col == 0:
                a.set_ylabel(f"{lab}\nscene #{k}   |c| = {mag:.2f} px",
                             fontsize=11.5, labelpad=10)
            else:
                cx, cy = Z * .5, Z * .5
                a.annotate("", xy=(cx + ax_ * EXAG, cy + ay_ * EXAG), xytext=(cx, cy),
                           arrowprops=dict(arrowstyle="-|>,head_width=.5,head_length=.9",
                                           color="#FF3B30", lw=3.6, shrinkA=0, shrinkB=0))
                a.plot([cx], [cy], "o", ms=6.5, color="#FF3B30")
                a.text(.5, .045,
                       f"PAN content moves {mag:.2f} px\n"
                       f"{vert} {abs(c[0]):.2f}  ·  {horz} {abs(c[1]):.2f}",
                       transform=a.transAxes, ha="center", fontsize=11.5, color="#C1121F",
                       bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#FF3B30", alpha=.92))
        print(f"{lab.splitlines()[0]}: scene {k}  |c|={mag:.3f}px  "
              f"(dy {c[0]:+.3f}, dx {c[1]:+.3f})  content -> {vert}/{horz}")

    fig.suptitle("PAN before / after the aligner — most-shifted scene of the FR paper set\n"
                 f"{Z}x{Z} zoom;  arrow = DIRECTION only, length exaggerated {EXAG:.0f}x "
                 "(the real shift is sub-pixel to ~2 px and is invisible by eye)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .935])
    p = os.path.join(OUT, "0917_aligner_shift_example.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
