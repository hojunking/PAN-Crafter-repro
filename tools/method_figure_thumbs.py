"""method 개념도용 썸네일 — Teacher 출력·오차맵 e_T·GT edge 타깃.

개념도에 그대로 끼워 넣을 수 있게 **축·여백·테두리 없는 정사각 PNG** 로 낸다.
정의는 코드 그대로 쓴다(그림과 구현이 어긋나지 않게):

  I^hr_ms        GT (RR 테스트셋의 gt)
  Î^hr,T         Teacher T0 출력 (assets/pakd50/T0_run = PALS24 L1E4 S2025 best_raw)
  e_T            |Î^hr,T − I^hr_ms| 의 **밴드 평균**            (kdv/losses_rec.py:68)
  ∇I^hr_ms       GT 의 signed Scharr (gx,gy) 크기, 밴드 평균     (pa/losses.py scharr)
  s_n            cue 자산의 raw q 로 만든 신뢰도 — 여기서는 만들지 않는다(scene 단위가 아님)

RR 테스트셋을 쓰는 이유: e_T 와 edge 타깃 둘 다 **GT 가 있어야** 한다. FR 에는 GT 가 없다.

사용:
    python tools/method_figure_thumbs.py                 # 기본 scene 자동 선택
    python tools/method_figure_thumbs.py --scene 7       # scene 지정
    python tools/method_figure_thumbs.py --list          # scene 목록·특성만 출력
산출 → outputs/method_figure/<scene>/*.png  + contact_sheet.png (미리보기)
"""
import argparse
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
from pa.losses import scharr                                # noqa: E402
from main import import_class                               # noqa: E402

OUTDIR = os.path.join(ROOT, "outputs", "method_figure")
RGB = (4, 2, 1)                     # WV3 8밴드 -> red/green/blue
PX = 512                            # 저장 해상도 (256 원본을 2배로)


def to_rgb(cube, lo=2, hi=98, ref=None):
    """[C,H,W] -> [H,W,3] float [0,1]. ref 가 있으면 그 통계로 맞춘다(전/후 비교용)."""
    src = cube if ref is None else ref
    out = np.zeros(cube.shape[1:] + (3,))
    for j, b in enumerate(RGB):
        p, q = np.percentile(src[b], lo), np.percentile(src[b], hi)
        out[..., j] = np.clip((cube[b] - p) / (q - p + 1e-12), 0, 1)
    return out


def stretch1(a, lo=2, hi=98):
    """단일 채널 퍼센타일 스트레치 -> [0,1] (PAN 처럼 히스토그램이 한쪽에 몰린 경우 필수)."""
    p, q = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p) / (q - p + 1e-12), 0, 1)


def save_clean(arr, path, cmap=None, vmin=None, vmax=None):
    """축·여백·테두리 없이 정사각 PNG 로."""
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_axis_off()
    fig.savefig(path, dpi=100, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def load_scene(ds, idx, dev):
    gt, lms, ms, lpan, pan = [x.unsqueeze(0).to(dev) for x in ds[idx]]
    return dict(gt=gt, lms=lms, ms=ms, lpan=lpan, pan=pan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=None, help="RR 테스트셋 scene 번호")
    ap.add_argument("--list", action="store_true", help="scene 특성만 출력하고 종료")
    a = ap.parse_args()

    cfg = CM.localize_cfg(yaml.safe_load(open(os.path.join(
        ROOT, "work_dir/PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/meta/config.yaml"))))
    ds = import_class(cfg["feeder"])(**cfg["test_reduced_feeder_args"])
    dev = CM.DEV
    L = CM.load_model("L1E4", 2025, "best_raw")          # = assets/pakd50/T0_run 과 같은 가중치

    # scene 특성 (구조가 많은 장면이 개념도에 어울린다)
    stats = []
    for i in range(len(ds)):
        g = ds[i][0].numpy()
        gx, gy = scharr(torch.from_numpy(g)[None].float())
        stats.append((i, float((gx ** 2 + gy ** 2).mean()), float(g.std())))
    if a.list:
        print(f"{'scene':>6}{'edge energy':>14}{'contrast':>11}")
        for i, e, s in sorted(stats, key=lambda x: -x[1]):
            print(f"{i:6d}{e:14.5f}{s:11.4f}")
        return
    k = a.scene if a.scene is not None else max(stats, key=lambda x: x[1])[0]
    print(f"scene #{k} 선택 (edge energy 상위)")

    S = load_scene(ds, k, dev)
    with torch.no_grad():
        out = L.m(S["pan"], S["ms"], S["lpan"])
    gt = S["gt"][0].float().cpu()
    y_t = out["y"][0].float().cpu()

    # --- e_T : |Teacher − GT| 밴드 평균 (kdv/losses_rec.py 와 동일)
    e_T = (y_t - gt).abs().mean(0).numpy()

    # --- GT edge : signed Scharr 크기, 밴드 평균 (pa/losses.py scharr)
    gx, gy = scharr(gt[None])
    edge = torch.sqrt(gx[0] ** 2 + gy[0] ** 2).mean(0).numpy()

    d = os.path.join(OUTDIR, f"scene{k:02d}")
    os.makedirs(d, exist_ok=True)
    gt_n, yt_n = gt.numpy(), y_t.numpy()

    # 입력·출력 (같은 스트레치로 GT 기준 통일 — 나란히 놓아도 색이 안 튄다)
    save_clean(to_rgb(gt_n), f"{d}/I_hr_ms__GT.png")
    save_clean(to_rgb(yt_n, ref=gt_n), f"{d}/I_hat_hr_T__teacher_output.png")
    save_clean(to_rgb(S["lms"][0].float().cpu().numpy(), ref=gt_n), f"{d}/I_lr_ms__upsampled.png")
    pan_np = S["pan"][0, 0].float().cpu().numpy()
    save_clean(stretch1(pan_np), f"{d}/I_pan.png", cmap="gray", vmin=0, vmax=1)

    # e_T : 상위 0.5% 클립 (소수 픽셀이 스케일을 먹지 않게)
    v = float(np.percentile(e_T, 99.5))
    save_clean(e_T, f"{d}/e_T__teacher_error.png", cmap="inferno", vmin=0, vmax=v)
    save_clean(e_T, f"{d}/e_T__teacher_error_magma.png", cmap="magma", vmin=0, vmax=v)
    save_clean(e_T, f"{d}/e_T__teacher_error_gray.png", cmap="gray", vmin=0, vmax=v)

    # edge : 밝은 선이 검은 배경에 (∇ 기호 대체용으로 가장 읽기 쉽다)
    ve = float(np.percentile(edge, 99.5))
    save_clean(edge, f"{d}/grad_I_hr_ms__edge_target.png", cmap="gray", vmin=0, vmax=ve)
    save_clean(1 - np.clip(edge / ve, 0, 1), f"{d}/grad_I_hr_ms__edge_target_inv.png", cmap="gray")
    save_clean(edge, f"{d}/grad_I_hr_ms__edge_target_cividis.png", cmap="cividis", vmin=0, vmax=ve)

    # --- 미리보기 시트 (그림에 넣을 것은 위의 개별 PNG)
    items = [(to_rgb(S["lms"][0].float().cpu().numpy(), ref=gt_n), r"$I^{lr}_{ms}$ (upsampled)", None),
             (stretch1(pan_np), r"$I_{pan}$", "gray"),
             (to_rgb(gt_n), r"$I^{hr}_{ms}$  (GT)", None),
             (to_rgb(yt_n, ref=gt_n), r"$\hat{I}^{hr,T}_{ms}$  Teacher output", None),
             (e_T, r"$e_T = |\hat{I}^{hr,T}-I^{hr}_{ms}|$  band mean", "inferno"),
             (edge, r"$\nabla I^{hr}_{ms}$  Scharr magnitude", "gray")]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6))
    for ax, (img, ttl, cm) in zip(axes.ravel(), items):
        is_eT = ttl.startswith("$e_T"); is_edge = ttl.startswith(r"$\nabla")
        im = ax.imshow(img, cmap=cm, vmin=(0 if (is_eT or is_edge) else None),
                       vmax=(v if is_eT else (ve if is_edge else None)))
        ax.set_title(ttl, fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([])
        if is_eT or is_edge:
            plt.colorbar(im, ax=ax, fraction=.046, pad=.02)
    fig.suptitle(f"method figure thumbnails — RR scene #{k}  (Teacher T0 = PALS24 L1E4 S2025 best_raw)\n"
                 f"e_T max(99.5%) = {v:.4f}   edge max(99.5%) = {ve:.4f}   "
                 "(values in the model's [-1,1] scale)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(os.path.join(d, "contact_sheet.png"), dpi=110)
    plt.close(fig)

    print(f"\n산출 {os.path.relpath(d, ROOT)}/")
    for f in sorted(os.listdir(d)):
        print(f"   {f}")
    print(f"\ne_T  범위 [0, {e_T.max():.4f}]  중앙 {np.median(e_T):.4f}  (표시 상한 99.5% = {v:.4f})")
    print(f"edge 범위 [0, {edge.max():.4f}]  중앙 {np.median(edge):.4f}  (표시 상한 99.5% = {ve:.4f})")


if __name__ == "__main__":
    main()
