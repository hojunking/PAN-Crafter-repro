"""framework 그림용 추가 맵 — Teacher 오차 e_T 와 GT edge 타깃 ∇Y.

`tools/framework_figs_export.py` 와 **같은 규약**으로 같은 폴더에 낸다 — 같은 npz(`t0_rr.npz`),
같은 장면·크롭, 같은 ×4 nearest, 같은 512×512. 기존 PNG 는 건드리지 않고 두 장만 추가한다.

정의는 학습 코드 그대로다(그림과 구현이 어긋나지 않게):

  e_T   = |Y_T − Y| 의 **밴드 평균**                  (kdv/losses_rec.py 의 e_t 와 같은 식)
  ∇Y    = Y(GT) 의 signed Scharr (gx,gy) 크기, 밴드 평균 (pa/losses.py scharr — L_edge 가 쓰는 커널)

Scharr 는 **256² 전체 장면에서 계산한 뒤 크롭**한다 — 크롭 경계에서 reflect pad 인공물이 생기지 않게.
표시 스케일은 0 이 검정(오차 0 = 아무것도 아님), 상한은 크롭의 99 퍼센타일.

사용: python tools/framework_figs_maps.py        (t0_rr.npz 가 없으면 framework_figs_run_t0.py 먼저)
"""
import csv
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from pa.losses import scharr                                # noqa: E402

OUT = os.path.join(ROOT, "results_log", "paper_figs", "framework_figs")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t0_rr.npz")
R, N, UP = 2047.0, 128, 4                                   # export 와 동일
RGB = (4, 2, 1)
UP_S = 16                                                   # 네이티브 MS(32²) 를 512 로 (×16 nearest)
SAMPLES = [("1_buildings", 4, 64, 112), ("2_cars", 19, 112, 96), ("3_city", 17, 24, 32)]


def save(path, img, cmap="gray", up=UP):
    """[H,W] 또는 [H,W,3] in [0,1] -> x{up} nearest PNG (원 픽셀 그대로). export.save 와 같은 규약."""
    a = np.repeat(np.repeat(img, up, 0), up, 1)
    plt.imsave(path, a, cmap=(None if a.ndim == 3 else cmap),
               vmin=(None if a.ndim == 3 else 0), vmax=(None if a.ndim == 3 else 1))


def st(a, lo, hi):
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 없음 — python tools/framework_figs_run_t0.py 를 먼저 돌린다")
    z = np.load(SRC)
    rows = []
    for name, s, y0, x0 in SAMPLES:
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        sl = (slice(y0, y0 + N), slice(x0, x0 + N))

        gt = z["gt"][s].transpose(1, 2, 0)                          # [256,256,8] DN
        yT = (z["y"][s].transpose(1, 2, 0) + 1) / 2 * R             # [256,256,8] DN

        # e_T : 밴드 평균 절대오차 (DN). 학습 코드는 [-1,1] 이라 값은 R/2 배 차이 — 지도는 같다
        e_full = np.abs(yT - gt).mean(2)
        e = e_full[sl]

        # GT edge : 전체에서 Scharr -> 크롭 (경계 인공물 회피)
        g = torch.from_numpy(gt.transpose(2, 0, 1))[None].float()
        gx, gy = scharr(g)
        edge_full = torch.sqrt(gx[0] ** 2 + gy[0] ** 2).mean(0).numpy()
        ed = edge_full[sl]

        ev = float(np.percentile(e, 99))
        gv = float(np.percentile(ed, 99))
        en, gn = np.clip(e / (ev + 1e-9), 0, 1), np.clip(ed / (gv + 1e-9), 0, 1)

        save(os.path.join(d, "ET_teacher_err.png"), en, cmap="inferno")
        save(os.path.join(d, "ET_teacher_err_magma.png"), en, cmap="magma")
        save(os.path.join(d, "ET_teacher_err_gray.png"), en, cmap="gray")
        save(os.path.join(d, "G_gt_edge.png"), gn, cmap="gray")
        save(os.path.join(d, "G_gt_edge_inv.png"), 1 - gn, cmap="gray")
        save(os.path.join(d, "G_gt_edge_cividis.png"), gn, cmap="cividis")

        # 업샘플 전 네이티브 MS 를 **다른 박스와 같은 512 크기**로 (×16 nearest, 보간 없음).
        # 기존 S_ms_native.png(128 = 32×4) 는 M 의 1/4 로 두는 판이라 그대로 남긴다.
        g_rgb = gt[sl][..., RGB]                                    # export 와 같은 GT 기준 밴드별 1-99%
        clo = np.percentile(g_rgb, 1, axis=(0, 1)); chi = np.percentile(g_rgb, 99, axis=(0, 1))
        ms_lr = z["ms"][s].transpose(1, 2, 0)[y0 // 4:y0 // 4 + N // 4, x0 // 4:x0 // 4 + N // 4]
        save(os.path.join(d, "S_ms_native_512.png"), st(ms_lr[..., RGB], clo, chi), up=UP_S)

        rows.append(dict(sample=name, scene=s, crop_row=y0, crop_col=x0, crop_px=N, upscale=UP,
                         eT_mean_dn=round(float(e.mean()), 3), eT_med_dn=round(float(np.median(e)), 3),
                         eT_p99_dn=round(ev, 3), eT_max_dn=round(float(e.max()), 3),
                         eT_p99_model_unit=round(ev / (R / 2), 5),
                         edge_med_dn=round(float(np.median(ed)), 3), edge_p99_dn=round(gv, 3)))
        print(f"{name}: scene {s} ({y0},{x0})  e_T 중앙 {np.median(e):7.2f} DN  p99 {ev:7.2f} DN "
              f"(모델 단위 {ev/(R/2):.4f})   edge 중앙 {np.median(ed):7.2f}  p99 {gv:7.2f} DN")

    p = os.path.join(OUT, "index_maps.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {OUT}  (+ {os.path.basename(p)})")


if __name__ == "__main__":
    main()
