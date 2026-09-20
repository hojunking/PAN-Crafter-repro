"""framework 도식 박스용 — "PAN 이 이동한다" 를 작은 크기에서도 읽히게.

문제: 실제 이동은 0.2–2 px 이라 도식 박스(1 인치 남짓)로 줄이면 edge 오버레이조차 뭉개진다.
해법: **실제 PAN 에서 뽑은 윤곽선 몇 개만** 굵게 그리고, 원본/이동본을 두 색으로 겹친 뒤
화살표로 이동 방향을 찍는다. 배경에는 원 PAN 을 흐리게 깔아 위성영상임이 보이게 한다.

  - 윤곽선은 `skimage.measure.find_contours` 로 실제 PAN 에서 추출한다(그려 넣은 것이 아니다)
  - 길이 상위 몇 개만 남겨 작은 크기에서도 선이 구분되게 한다
  - **이동량은 도식용으로 과장할 수 있다** — 파일명과 캡션 텍스트에 배율을 박는다

내는 것(묶음마다):
  SHIFT_schem_wide_x<k>.png   128² 전체 크롭, 윤곽선 상위 N 개
  SHIFT_schem_zoom_x<k>.png   32² 확대, 윤곽선 1–2 개 + 화살표 (박스가 아주 작을 때)

사용:
  python tools/framework_figs_shift_schematic.py --only 3_city
  python tools/framework_figs_shift_schematic.py --only 3_city --exaggerate 6 --vec eps
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from pa.warp import warp_pan                                # noqa: E402

OUT = os.path.join(ROOT, "results_log", "paper_figs", "framework_figs")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t0_rr.npz")
IDX = os.path.join(OUT, "index.csv")
N = 128
EPS = (1.6, -1.2)
CY, RD = "#00E5FF", "#FF2D2D"       # 원본 / 이동본


def contours_of(p, level_q=62, smooth=1.0, keep=14, min_len=14):
    """PAN 에서 실제 윤곽선 추출. 길이 상위 keep 개만."""
    g = gaussian_filter(p.astype(np.float64), smooth)
    lv = np.percentile(g, level_q)
    cs = [c for c in find_contours(g, lv) if len(c) >= min_len]
    cs.sort(key=len, reverse=True)
    return cs[:keep]


def draw(ax, p_bg, cs, dy, dx, lw, arrows=0, bg_alpha=.35):
    lo, hi = np.percentile(p_bg, [2, 98])
    ax.imshow(np.clip((p_bg - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", alpha=bg_alpha,
              interpolation="bilinear")
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], color=CY, lw=lw, solid_capstyle="round")
    # 내용은 (-dy,-dx) 로 움직인다 (aligned[y,x] = orig[y+dy, x+dx])
    for c in cs:
        ax.plot(c[:, 1] - dx, c[:, 0] - dy, color=RD, lw=lw, solid_capstyle="round")
    if arrows:
        for c in cs[:arrows]:
            i = len(c) // 2
            y, x = c[i, 0], c[i, 1]
            ax.annotate("", xy=(x - dx, y - dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>,head_width=.34,head_length=.62",
                                        color="w", lw=lw * .8, shrinkA=0, shrinkB=0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("black")


def panel(path, p_bg, cs, dy, dx, real_mag, k, lw, arrows, size, note):
    fig = plt.figure(figsize=(size, size), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, p_bg, cs, dy, dx, lw, arrows)
    ax.set_xlim(-.5, p_bg.shape[1] - .5); ax.set_ylim(p_bg.shape[0] - .5, -.5)
    txt = (f"shift {real_mag:.2f} px" if k == 1 else
           f"shift {real_mag:.2f} px  (drawn {k}x)")
    ax.text(.5, .022, txt + ("" if not note else f"  ·  {note}"), transform=ax.transAxes,
            ha="center", fontsize=8.6, color="w",
            bbox=dict(boxstyle="round,pad=.28", fc="black", ec="w", lw=.7, alpha=.8))
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--vec", choices=("eps", "c0"), default="eps",
                    help="어느 이동을 그릴 것인가 (기본 eps = 학습에 주입하는 2.0 px)")
    ap.add_argument("--exaggerate", type=float, nargs="+", default=[1, 3, 6],
                    help="그릴 배율들 (1 = 실제값)")
    a = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 없음 — python tools/framework_figs_run_t0.py 먼저")
    z = np.load(SRC)
    idx = {r["sample"]: r for r in csv.DictReader(open(IDX))}

    for name, r in idx.items():
        if a.only and name != a.only:
            continue
        s, y0, x0 = int(r["scene"]), int(r["crop_row"]), int(r["crop_col"])
        v = (np.array(EPS, np.float32) if a.vec == "eps"
             else np.array([float(r["c0_dy"]), float(r["c0_dx"])], np.float32))
        mag = float(np.linalg.norm(v))
        d = os.path.join(OUT, name)
        pan = z["pan"][s, 0][y0:y0 + N, x0:x0 + N]

        cs_w = contours_of(pan, keep=14)
        # 확대용: 윤곽선이 가장 굵게 지나는 32² 창
        best, bij = -1, (48, 48)
        for i in range(8, N - 40, 4):
            for j in range(8, N - 40, 4):
                n = sum(((c[:, 0] >= i) & (c[:, 0] < i + 32) & (c[:, 1] >= j) & (c[:, 1] < j + 32)).sum()
                        for c in cs_w)
                if n > best:
                    best, bij = n, (i, j)
        zi, zj = bij
        pz = pan[zi:zi + 32, zj:zj + 32]
        cs_z = contours_of(pz, keep=3, min_len=8)

        for k in a.exaggerate:
            ks = f"{k:g}".replace(".", "p")
            dy, dx = float(v[0] * k), float(v[1] * k)
            panel(os.path.join(d, f"SHIFT_schem_wide_{a.vec}_x{ks}.png"), pan, cs_w, dy, dx,
                  mag, k, lw=1.7, arrows=0, size=3.4, note=f"{a.vec}")
            panel(os.path.join(d, f"SHIFT_schem_zoom_{a.vec}_x{ks}.png"), pz, cs_z, dy, dx,
                  mag, k, lw=3.0, arrows=2, size=3.4, note=f"{a.vec}")
        print(f"{name:12} {a.vec} |v|={mag:.3f} px  contours wide {len(cs_w)} / zoom {len(cs_z)} "
              f"@({zi},{zj})  배율 {a.exaggerate}")

    print("\ncyan = 원본 PAN 윤곽 · red = 이동한 PAN 윤곽 · 화살표 = 이동 방향")
    print("윤곽선은 실제 PAN 에서 추출한 것이다. **배율 != 1 이면 캡션에 반드시 적을 것.**")
    print("->", OUT)


if __name__ == "__main__":
    main()
