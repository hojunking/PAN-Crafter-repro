"""PAN aligner 가 PAN 을 어디로 옮기는가 — 전/후와 이동 방향.

이동이 sub-pixel ~2 px 이라 넓게 보면 절대 안 보인다. 그래서
  - **가로 경계**(세로 경사가 가장 큰 곳)를 아주 좁게(기본 22 px) 확대하고
  - 픽셀을 블록으로(nearest) 그려 밝기 재분배가 보이게 하고
  - **양쪽 패널에 똑같은 좌표로 기준선**을 그어 경계가 그 선에 대해 이동한 것을 보이게 한다
  - 화살표는 내용이 움직인 방향 (실측 확인: delta (dy,dx) -> 내용은 (-dy,-dx))

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
CASES = [("L1E4", 2025, "best_raw", "T0  lambda* = 1e-4\n(deployed teacher)"),
         ("N2", 2025, "last", "donor N2\n(strongest aligner)")]
Z = 22                      # 확대 crop 한 변 (px) — 아주 좁게
CTX = 128                   # 왼쪽 맥락 이미지


def horiz_edge_window(p, z=Z, margin=60):
    """가로 경계가 **crop 중앙**에 오도록 창을 고른다. 경계 세기 x crop 대비 로 점수."""
    gy = np.abs(np.gradient(p, axis=0))
    H, W = p.shape
    h = z // 2
    best, bij = -1, (margin, margin)
    for i in range(margin + h, H - margin - h, 3):
        for j in range(margin, W - margin - z, 6):
            crop = p[i - h:i - h + z, j:j + z]
            if crop.shape != (z, z):
                continue
            edge = gy[i, j:j + z].mean()              # 중앙 행의 가로 경계 세기
            spread = (gy[i, j:j + z] > gy[i, j:j + z].max() * .35).mean()   # 가로로 이어지는가
            v = edge * spread * crop.std()            # 대비까지 반영
            if v > best:
                best, bij = v, (i - h, j)
    return bij


def stretch(a, lo=1, hi=99):
    p, q = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p) / (q - p + 1e-12), 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = CM.localize_cfg(yaml.safe_load(open(os.path.join(
        ROOT, "work_dir/PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/meta/config.yaml"))))
    ds = import_class(cfg["feeder"])(**cfg["test_full_feeder_args"])

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.6),
                             gridspec_kw=dict(width_ratios=[1.05, 1, 1], wspace=.13, hspace=.20))

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

        i0, j0 = horiz_edge_window(p0)
        sl = (slice(i0, i0 + Z), slice(j0, j0 + Z))
        a0, a1 = p0[sl], p1[sl]
        vmin, vmax = np.percentile(np.r_[a0.ravel(), a1.ravel()], [1, 99])
        # 경계가 지나는 행 (원본 기준)
        er = float(np.abs(np.gradient(a0, axis=0)).sum(1).argmax())
        mag = float(np.hypot(c[0], c[1]))
        vert, horz = ("up" if -c[0] < 0 else "down"), ("right" if -c[1] > 0 else "left")

        # --- 왼쪽: 맥락 + 확대 위치
        ci = max(0, min(512 - CTX, i0 + Z // 2 - CTX // 2))
        cj = max(0, min(512 - CTX, j0 + Z // 2 - CTX // 2))
        a = axes[r, 0]
        a.imshow(stretch(p0[ci:ci + CTX, cj:cj + CTX]), cmap="gray")
        a.add_patch(plt.Rectangle((j0 - cj - .5, i0 - ci - .5), Z, Z, fill=False,
                                  ec="#FFD60A", lw=2.2))
        a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(f"{lab}\nscene #{k}   |c| = {mag:.2f} px", fontsize=11)
        if r == 0:
            a.set_title(f"context ({CTX}x{CTX})\nyellow box = zoom below", fontsize=11.5, pad=7)

        # --- 가운데/오른쪽: 전·후 아주 좁은 확대 + 동일 기준선
        for col, (img, ttl) in enumerate([(a0, "BEFORE  original PAN"),
                                          (a1, "AFTER  aligner applied")], start=1):
            a = axes[r, col]
            a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            a.axhline(er, color="#00E5FF", lw=1.9, ls="-", alpha=.95)      # 동일 좌표 기준선
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(ttl + f"\n{Z}x{Z} px, pixels drawn as blocks", fontsize=11.5, pad=7)
            if col == 2:
                cx, cy = Z * .5, Z * .62
                a.annotate("", xy=(cx - c[1] * 3.0, cy - c[0] * 3.0), xytext=(cx, cy),
                           arrowprops=dict(arrowstyle="-|>,head_width=.45,head_length=.8",
                                           color="#FF3B30", lw=3.0, shrinkA=0, shrinkB=0))
                a.text(.5, .03, f"content moves {mag:.2f} px  ({vert} {abs(c[0]):.2f}, "
                                f"{horz} {abs(c[1]):.2f})",
                       transform=a.transAxes, ha="center", fontsize=10.5, color="#C1121F",
                       bbox=dict(boxstyle="round,pad=.28", fc="white", ec="#FF3B30", alpha=.93))
        # 기준선 위/아래 평균으로 이동을 수치로도 확인
        up0, dn0 = a0[:int(er) + 1].mean(), a0[int(er) + 1:].mean()
        up1, dn1 = a1[:int(er) + 1].mean(), a1[int(er) + 1:].mean()
        print(f"{lab.splitlines()[0]}: scene {k} |c|={mag:.3f} (dy {c[0]:+.3f} dx {c[1]:+.3f}) "
              f"-> {vert}/{horz};  기준선 위 평균 {up0:.3f}->{up1:.3f}, 아래 {dn0:.3f}->{dn1:.3f}")

    fig.suptitle("PAN before / after the aligner — cyan line is at the SAME pixel row in both panels\n"
                 "arrow = measured direction the PAN content moves (impulse-verified); "
                 "drawn at 3x for visibility",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    p = os.path.join(OUT, "0917_aligner_shift_example.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
