"""fig4_plane_corners_grad.png 의 **낱개 그림 전부**를 한 폴더에 저장한다.

`tools/paper_figs.py` 의 선택·크롭·스트레치·gradient 스케일을 그대로 재사용한다
(같은 모듈에서 import 하므로 본 그림과 픽셀 단위로 같은 자료다).

산출 → results_log/paper_figs/fig4_singles/

  NN_<code>_<slot>_id<sid>_pan.png    PAN patch 만 (회색, 64x48 크롭)
  NN_<code>_<slot>_id<sid>_grad.png   PAN Scharr gradient 크기만 (magma, **16 장 공통 스케일**)
  NN_<code>_<slot>_id<sid>_combo.png  본 그림에 나오는 그대로 (PAN + 우하단 gradient inset)
  group<N>_<code>.png                 한 묶음 2x2 (본 그림의 한 칸)
  colorbar_grad.png                   gradient colorbar 단독 (숫자 크게)
  index.csv                           각 파일의 sample_id·e·q·texture

전부 축·여백 없는 이미지다(colorbar 제외). gradient 는 patch 별로 정규화하지 않는다 —
그러면 집단 간 texture 차이가 사라진다.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

import tools.paper_figs as PF                                # noqa: E402

OUTDIR = os.path.join(PF.OUT, "fig4_singles")
PX = 640                                                     # 낱개 저장 긴 변(px)


def save_img(path, arr, cmap=None, vmin=None, vmax=None, border=None):
    """축·여백 없이. border 색이 주어지면 테두리만 그린다."""
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(PX / 100, PX / 100 * h / w), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if border:
        for sp in ax.spines.values():
            sp.set_color(border); sp.set_linewidth(2.0)
    else:
        ax.set_axis_off()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def save_combo(path, pan_s, grad, border="black"):
    """본 그림과 같은 배치 — PAN 위 우하단에 gradient inset(흰 테두리)."""
    h, w = pan_s.shape
    fig = plt.figure(figsize=(PX / 100, PX / 100 * h / w), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pan_s, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(border); sp.set_linewidth(2.0)
    ins = ax.inset_axes([0.525, 0.012, 0.463, 0.463])
    ins.imshow(grad, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color("white"); sp.set_linewidth(1.8)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def save_group(path, items):
    """한 묶음 2x2 (본 그림의 한 칸) — combo 4 장."""
    fig = plt.figure(figsize=(PX / 100 * 2, PX / 100 * 2 * 48 / 64), dpi=100)
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)
    for k, (pan_s, grad) in enumerate(items):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(pan_s, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("black"); sp.set_linewidth(1.0)
        ins = ax.inset_axes([0.525, 0.012, 0.463, 0.463])
        ins.imshow(grad, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color("white"); sp.set_linewidth(1.4)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def save_colorbar(path, gmax, vertical=True, tick=None, label=None):
    tick = tick if tick is not None else PF.CB_TICK
    label = label if label is not None else PF.CB_LABEL
    if vertical:
        fig = plt.figure(figsize=(2.30, 5.2), dpi=160)
        cax = fig.add_axes([0.06, 0.05, 0.20, 0.90])
    else:
        fig = plt.figure(figsize=(6.4, 1.55), dpi=160)
        cax = fig.add_axes([0.06, 0.52, 0.88, 0.22])
    sm = ScalarMappable(norm=Normalize(0, gmax), cmap="magma")
    bar = fig.colorbar(sm, cax=cax, orientation="vertical" if vertical else "horizontal")
    bar.set_label(r"PAN gradient magnitude  $|\nabla P|$", fontsize=label,
                  color=PF.INK2, labelpad=10)
    bar.ax.tick_params(labelsize=tick, colors=PF.INK2, length=4, width=1.0)
    bar.outline.set_edgecolor(PF.MUTED); bar.outline.set_linewidth(0.8)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tick", type=float, default=None, help="colorbar 눈금 글자 크기")
    a = ap.parse_args()

    import pandas as pd
    from tools.eqrec4_visuals import load_table, stretch
    from tools.eqrec4 import common as CM

    _, d = load_table()
    z = d.copy()
    z["qb"] = pd.qcut(z.q_A, PF.NB, labels=False)
    z["eb"] = pd.qcut(z.e_native_roi, PF.NB, labels=False)
    cells = []
    for ib, jb, name, axes, code in PF.CORNERS:
        g = z[(z.eb == ib) & (z.qb == jb)]
        cells.append(dict(code=code, name=name, axes=axes, g=g, rows=PF._pick(g, PF.PER_CELL)))
    ids = [int(r.sample_id) for c in cells for r in c["rows"]]
    pan = CM.load_patches(ids)[3].numpy()[:, 0]
    pmap = {sid: pan[i] for i, sid in enumerate(ids)}
    gmap, gmax = PF._grad_maps(pmap)

    os.makedirs(OUTDIR, exist_ok=True)
    for f in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, f))

    rows, k = [], 0
    for gi, c in enumerate(cells, start=1):
        items = []
        for j, r in enumerate(c["rows"], start=1):
            k += 1
            sid = int(r.sample_id)
            base = f"{k:02d}_{c['code']}_{j}_id{sid}"
            ps = stretch(pmap[sid])[PF.CROP43]
            gr = gmap[sid][PF.CROP43]
            items.append((ps, gr))
            save_img(os.path.join(OUTDIR, base + "_pan.png"), ps, cmap="gray")
            save_img(os.path.join(OUTDIR, base + "_grad.png"), gr, cmap="magma", vmin=0, vmax=1)
            save_combo(os.path.join(OUTDIR, base + "_combo.png"), ps, gr)
            rows.append(dict(order=k, group=c["code"], name=c["name"],
                             axes=c["axes"].replace("$", ""), slot=j, sample_id=sid,
                             source_group_id=str(r.source_group_id),
                             e_native_roi=round(float(r.e_native_roi), 6),
                             q_A=round(float(r.q_A), 6),
                             pan_scharr_energy=round(float(r.pan_scharr_energy), 6),
                             grad_scale_max=round(gmax, 6),
                             pan=base + "_pan.png", grad=base + "_grad.png",
                             combo=base + "_combo.png"))
        save_group(os.path.join(OUTDIR, f"group{gi}_{c['code']}.png"), items)

    save_colorbar(os.path.join(OUTDIR, "colorbar_grad.png"), gmax, True, a.tick)
    save_colorbar(os.path.join(OUTDIR, "colorbar_grad_horizontal.png"), gmax, False, a.tick)
    with open(os.path.join(OUTDIR, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(os.listdir(OUTDIR))
    print(f"-> {os.path.relpath(OUTDIR, ROOT)}   파일 {n} 개")
    print(f"   patch 16 x 3(pan/grad/combo) = 48 · group 4 · colorbar 2 · index.csv")
    print(f"   gradient 공통 스케일 0 .. {gmax:.4f}  (16 장 crop 의 p99.5)")
    print(f"   colorbar 눈금 글자 {a.tick if a.tick else PF.CB_TICK}")


if __name__ == "__main__":
    main()
