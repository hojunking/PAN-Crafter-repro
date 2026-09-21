"""논문용 그림 (초안, PNG 미리보기) → results_log/paper_figs/

    python tools/paper_figs.py

  fig4_plane_corners.png       (q_A, e) 평면 네 모서리 칸에서 4장씩 — 2 x 2 묶음 넷을 가로로.
  fig4_plane_corners_grad.png  같은 그림 + 각 patch 오른쪽 아래에 PAN Scharr gradient
  fig4_patches/                위 16 장의 원본 (표시용 PNG · 원 DN .npy · index.csv)

격자·대표 선택·표시(patch 별 percentile stretch)는 S5a
(`results_log/assets/0915_eqrec4_S5_plane.png`) 와 같다.
자료: work_dir/_eqrec4_s1_campaign/ (L1E4_S2025_best_raw, native64, n=4,096)

라벨은 ASCII (matplotlib 에 한글 글리프 없음).
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

OUT = os.path.join(ROOT, "results_log", "paper_figs")
PATCH_DIR = os.path.join(OUT, "fig4_patches")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
DPI = 260
CB_TICK, CB_LABEL = 15.0, 14.5      # gradient colorbar 눈금·라벨 글자 크기 (2026-09-21 키움: 9.5/11.5)

# (q_A, e) 5분위 평면의 네 모서리 칸. 배치는 q 낮은 쌍 -> q 높은 쌍, 각 쌍 안에서 error 낮은 쪽 -> 높은 쪽.
# q_A 는 offset consistency **잔차** 라 낮을수록 일관적이다 -> 이름은 consistent / inconsistent.
NB = 5
CORNERS = [                       # (e 분위, q 분위, 그림에 쓸 이름, 축 표기, 옛 4분면 코드)
    (0,      0,      "Consistent - low recon error",   r"$q$ low  /  $e$ low",   "EdCd"),
    (NB - 1, 0,      "Consistent - high recon error",  r"$q$ low  /  $e$ high",  "EuCd"),
    (0,      NB - 1, "Inconsistent - low recon error", r"$q$ high  /  $e$ low",  "EdCu"),
    (NB - 1, NB - 1, "Inconsistent - high recon error", r"$q$ high  /  $e$ high", "EuCu"),
]
PER_CELL = 4
CROP43 = slice(8, 56)             # 64 x 64 -> 64 x 48 중앙 크롭 (가로 4 : 세로 3). ROI [16:48] 은 안에 남는다


def _pick(g, n):
    """칸 중앙에 가까운 순 (S5a 규칙), source block 이 겹치지 않게 n 장."""
    g = g.copy()
    g["dist"] = ((g.q_A.rank(pct=True) - .5) ** 2 + (g.e_native_roi.rank(pct=True) - .5) ** 2) ** .5
    g = g.sort_values("dist")
    seen, out = set(), []
    for _, r in g.iterrows():
        if r.source_group_id in seen:
            continue
        seen.add(r.source_group_id); out.append(r)
        if len(out) == n:
            break
    return out + [r for _, r in g.iterrows() if len(out) < n][:n - len(out)]


def _grad_maps(pmap):
    """pan_scharr_energy 와 같은 연산자(pa.losses.scharr) 의 크기 맵. **16 장 공통 스케일** —
    patch 별로 정규화하면 집단 간 texture 차이가 그림에서 사라진다."""
    import torch
    from pa.losses import scharr
    ids = list(pmap)
    a = torch.from_numpy(np.stack([pmap[i] for i in ids]))[:, None].float()
    gx, gy = scharr(a)
    m = (gx ** 2 + gy ** 2).sqrt()[:, 0].numpy()
    hi = float(np.percentile(m[:, CROP43], 99.5))            # 보이는 영역의 상위 꼬리 하나로 자른다
    return {i: np.clip(m[k] / (hi + 1e-12), 0, 1) for k, i in enumerate(ids)}, hi


def _render(cells, pmap, stretch, path, gmap=None, gmax=None):
    cb = gmap is not None and gmax is not None
    W = 18.45 if cb else 16.5
    fig = plt.figure(figsize=(W, 4.05))
    outer = GridSpec(1, 4, figure=fig, wspace=0.055, left=0.008,
                     right=(0.008 + 16.32 / W) if cb else 0.992, top=0.972, bottom=0.256)
    for i, c in enumerate(cells):
        inner = outer[0, i].subgridspec(2, 2, hspace=0.0, wspace=0.0)
        for k, r in enumerate(c["rows"]):
            ax = fig.add_subplot(inner[k // 2, k % 2])
            a = pmap[int(r.sample_id)]
            ax.imshow(stretch(a)[CROP43], cmap="gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("black"); sp.set_linewidth(1.0)
            if gmap is not None:
                ins = ax.inset_axes([0.525, 0.012, 0.463, 0.463])
                ins.imshow(gmap[int(r.sample_id)][CROP43], cmap="magma", interpolation="nearest",
                           vmin=0, vmax=1)
                ins.set_xticks([]); ins.set_yticks([])
                for sp in ins.spines.values():
                    sp.set_color("white"); sp.set_linewidth(1.4)
        pos = outer[0, i].get_position(fig)
        xc = (pos.x0 + pos.x1) / 2
        fig.text(xc, pos.y0 - 0.032, c["axes"], ha="center", va="top", fontsize=11.5, color=MUTED)
        fig.text(xc, pos.y0 - 0.120, c["name"], ha="center", va="top", fontsize=14,
                 color=INK, fontweight="bold")
    if cb:
        p0 = outer[0, 0].get_position(fig)
        cax = fig.add_axes([0.008 + 16.55 / W, p0.y0, 0.24 / W, p0.y1 - p0.y0])
        sm = ScalarMappable(norm=Normalize(0, gmax), cmap="magma")
        bar = fig.colorbar(sm, cax=cax)
        bar.set_label(r"PAN gradient magnitude  $|\nabla P|$", fontsize=CB_LABEL, color=INK2, labelpad=10)
        bar.ax.tick_params(labelsize=CB_TICK, colors=INK2, length=4, width=1.0)
        bar.outline.set_edgecolor(MUTED); bar.outline.set_linewidth(0.8)
    fig.savefig(path, dpi=DPI, facecolor="white"); plt.close(fig)
    print("->", path)


def _save_patches(cells, pmap, stretch):
    """16 장의 원본을 따로 남긴다 — 표시용 PNG · 원 DN .npy · index.csv."""
    os.makedirs(PATCH_DIR, exist_ok=True)
    for f in os.listdir(PATCH_DIR):
        os.remove(os.path.join(PATCH_DIR, f))
    rows, k = [], 0
    for c in cells:
        for j, r in enumerate(c["rows"]):
            k += 1
            sid = int(r.sample_id)
            base = f"{k:02d}_{c['code']}_{j + 1}_id{sid}"
            a = pmap[sid]
            plt.imsave(os.path.join(PATCH_DIR, base + ".png"), stretch(a), cmap="gray")
            np.save(os.path.join(PATCH_DIR, base + ".npy"), a)
            rows.append(dict(order=k, name=c["name"], axes=c["axes"].replace("$", ""),
                             group=c["code"], slot=j + 1,
                             sample_id=sid, source_group_id=str(r.source_group_id),
                             e_native_roi=round(float(r.e_native_roi), 6),
                             q_A=round(float(r.q_A), 6),
                             pan_scharr_energy=round(float(r.pan_scharr_energy), 6),
                             file=base + ".png"))
    with open(os.path.join(PATCH_DIR, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {PATCH_DIR}  ({len(rows)} patches: png + npy + index.csv)")


def fig4():
    import pandas as pd
    from tools.eqrec4_visuals import load_table, stretch
    from tools.eqrec4 import common as CM

    _, d = load_table()
    z = d.copy()
    z["qb"] = pd.qcut(z.q_A, NB, labels=False)
    z["eb"] = pd.qcut(z.e_native_roi, NB, labels=False)

    cells = []
    for ib, jb, name, axes, code in CORNERS:
        g = z[(z.eb == ib) & (z.qb == jb)]
        cells.append(dict(code=code, name=name, axes=axes, g=g, rows=_pick(g, PER_CELL)))

    ids = [int(r.sample_id) for c in cells for r in c["rows"]]
    pan = CM.load_patches(ids)[3].numpy()[:, 0]
    pmap = {sid: pan[i] for i, sid in enumerate(ids)}

    for c in cells:
        g = c["g"]
        print(f"   {c['name']:22s} {c['axes']:22s} e {g.e_native_roi.median():.4f}  "
              f"q {g.q_A.median():.4f}  texture {g.pan_scharr_energy.mean():.4f}  n {len(g)}")
    gmap, gmax = _grad_maps(pmap)
    print(f"   gradient colour scale: 0 .. {gmax:.4f}  (p99.5 over the 16 shown crops)")
    _render(cells, pmap, stretch, os.path.join(OUT, "fig4_plane_corners.png"))
    _render(cells, pmap, stretch, os.path.join(OUT, "fig4_plane_corners_grad.png"),
            gmap=gmap, gmax=gmax)
    _save_patches(cells, pmap, stretch)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig4()


if __name__ == "__main__":
    main()
