#!/usr/bin/env python
"""QRECON24 정성 결과 패널 (선택 checkpoint 기준) — 로컬 PNG 만 만든다. 위성영상은 외부 서비스에 올리지 않는다 (CLAUDE.md §6).

    python tools/qrc24_qualitative.py                                  # 기본: s1 G23 S1234(최고) vs 대조 G22 S1234
    python tools/qrc24_qualitative.py --run <run> --control <run> --tag 0917_qrc24 --out-dir results_log/assets
    python tools/qrc24_qualitative.py --fr-scenes 8,7,2,4,18 --rr-scenes 15,11,12,9,0 --zoom 8,18

만드는 것 (파일명 <tag>_<이름>.png):
  fr    full-resolution 논문 세트 20 장 중 선택 장면: PAN | EXP(lms) | 대조 | 이 run | |이 run − 대조| (GT 가 없다 — HQNR 이 판정)
  rr    reduced-resolution 20 장: PAN | EXP | 이 run | GT | |이 run − GT| | |이 run − 대조|
  zoom  FR 장면의 고에지 128 px 크롭을 nearest ×3 으로: PAN | EXP | 대조 | 이 run

읽는 것: work_dir/<run>/results/{reduced_best_hqnr.mat, full_best_hqnr_mat20.mat, fr_mat20.json, best_raw_meta.json} 와
FR 입력 data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5. **RR 20 장과 FR 20 장은 서로 다른 장면**이다
(KNOWN_ISSUES F-2) — 한 그림에 같은 번호로 나란히 두지 않는다.

표시 규약 (저장소 관행 그대로): WV3 8 밴드 RGB = (4, 2, 1) · max_pixel 2047 · mat 의 sr 은 이미 DN 이라 재변환하지 않는다 ·
스트레치는 **행마다 기준 배열 하나(RR 은 GT, FR 은 EXP)의 밴드별 1–99 %** 를 그 행의 모든 칸에 같이 쓴다(CONVENTION §4) ·
차분·오차맵은 전 행 공통 색 스케일 · 확대는 nearest(보간 금지) · 라벨은 ASCII 만 (matplotlib 에 한글 글리프가 없다).
"""
import argparse
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.metrics.eval_rr import ergas, sam  # noqa: E402

RGB = (4, 2, 1)                     # WV3 8 밴드 -> R,G,B (utils.py:50 · eqrec4_visuals.py:38 과 같다)
MAXP = 2047.0                       # WV3 11-bit
DIM_CUT = 21                        # 평가 crop (tools/eval_dlpan.py); RR 지표 표기를 평가와 맞춘다
FR_H5 = os.path.join("data", "PanCollection", "WV3", "full_examples_mat20", "test_wv3_OrigScale_mat20.h5")
DEF_RUN = "PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1"
DEF_CTRL = "PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1"
DEF_FR = (8, 7, 2, 4, 18)           # 절대 최고 / 전형 / 대조 대비 최대 개선(평탄) / 같은 개선(질감) / 가장 어려운 장면
DEF_RR = (15, 11, 12, 9, 0)         # 절대 최고 / 전형 / 대조 대비 최대 개선 / 대조에 지는 장면 / 가장 어려운 장면
DEF_ZOOM = (8, 18)


def stretch1(a, lo=1, hi=99):
    """단일 채널(PAN) 퍼센타일 스트레치 -> [0,1]."""
    a = np.asarray(a, dtype=np.float64); p, q = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p) / (q - p + 1e-12), 0, 1)


def to_rgb(cube, ref, lo=1, hi=99):
    """(8,H,W) DN -> (H,W,3) [0,1]. ref 의 밴드별 퍼센타일을 그대로 써서 같은 행의 칸끼리 배율을 맞춘다."""
    cube = np.asarray(cube, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    out = np.zeros(cube.shape[1:] + (3,))
    for j, b in enumerate(RGB):
        p, q = np.percentile(ref[b], lo), np.percentile(ref[b], hi)
        out[..., j] = np.clip((cube[b] - p) / (q - p + 1e-12), 0, 1)
    return out


def band_mean_abs(a, b):
    """|a − b| 의 밴드 평균 (DN)."""
    return np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean(axis=0)


def grad_energy(pan2d):
    """Sobel 기울기 크기 평균 — 크롭 위치 선택용."""
    k = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    from scipy.ndimage import convolve
    gx = convolve(pan2d, k, mode="nearest"); gy = convolve(pan2d, k.T, mode="nearest")
    return np.hypot(gx, gy)


def busiest_crop(pan2d, size=128):
    """에지 에너지가 가장 큰 size×size 크롭의 (y0, x0) — 평탄한 밭이 아니라 볼거리를 고른다."""
    g = grad_energy(pan2d); H, W = g.shape; step = max(size // 4, 1); best, pos = -1.0, (0, 0)
    for y in range(0, H - size + 1, step):
        for x in range(0, W - size + 1, step):
            v = float(g[y:y + size, x:x + size].mean())
            if v > best:
                best, pos = v, (y, x)
    return pos


def hbar(fig, col_axes, im, label):
    """열 아래에 가로 colorbar — ax= 로 붙이면 행 사이에 끼어 겹친다(실측)."""
    pos = col_axes[-1].get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.020, pos.width, 0.0075])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=7); cb.set_label(label, fontsize=7.5)
    return cb


def load_run(run):
    wd = os.path.join(ROOT, "work_dir", run); res = os.path.join(wd, "results")
    rr = loadmat(os.path.join(res, "reduced_best_hqnr.mat"))
    fr_sr = loadmat(os.path.join(res, "full_best_hqnr_mat20.mat"))["sr"].astype(np.float64)
    fr = json.load(open(os.path.join(res, "fr_mat20.json")))
    meta = json.load(open(os.path.join(wd, "best_raw_meta.json")))
    return dict(run=run, rr=rr, fr_sr=fr_sr, fr=fr, meta=meta)


def rr_scene_metrics(sr, gt):
    """장면 하나의 RR ERGAS·SAM (평가와 같은 crop21, (C,H,W) 입력)."""
    sl = slice(DIM_CUT - 1, -DIM_CUT)
    a = np.asarray(sr, dtype=np.float64).transpose(1, 2, 0)[sl, sl, :]
    b = np.asarray(gt, dtype=np.float64).transpose(1, 2, 0)[sl, sl, :]
    return ergas(a, b), sam(a, b)


def fig_fr(best, ctrl, scenes, out, dpi):
    """FR 트랙 (GT 없음; 판정이 여기 있다)."""
    with h5py.File(os.path.join(ROOT, FR_H5), "r") as h:                                   # h5 fancy index 는 오름차순만 받는다 — 정렬해서 읽고 원 순서로 되돌린다
        order = sorted(range(len(scenes)), key=lambda k: scenes[k]); srt = [scenes[k] for k in order]; back = np.argsort(order)
        pan = np.asarray(h["pan"][srt, 0], dtype=np.float64)[back]; lms = np.asarray(h["lms"][srt], dtype=np.float64)[back]
    sb, sc = best["fr_sr"][list(scenes)], ctrl["fr_sr"][list(scenes)]
    diffs = [band_mean_abs(sb[i], sc[i]) for i in range(len(scenes))]
    vmax = float(np.percentile(np.concatenate([d.ravel() for d in diffs]), 99))
    cols = ["PAN (512 px)", "EXP  (lms, interp23tap)", "control %s" % short(ctrl), "this run %s" % short(best), "|this - control|  DN"]
    fig, ax = plt.subplots(len(scenes), 5, figsize=(16.2, 3.32 * len(scenes)))
    ax = np.atleast_2d(ax)
    for r, s in enumerate(scenes):
        ref = lms[r]                                   # GT 가 없으므로 공통 입력(EXP) 을 스트레치 기준으로 (행 안에서 동일 배율)
        hb, hc = best["fr"]["per_scene_hqnr"][s], ctrl["fr"]["per_scene_hqnr"][s]
        ds_b, ds_c = best["fr"]["per_scene_d_s"][s], ctrl["fr"]["per_scene_d_s"][s]
        ax[r, 0].imshow(stretch1(pan[r]), cmap="gray", interpolation="nearest")
        ax[r, 1].imshow(to_rgb(lms[r], ref), interpolation="nearest")
        ax[r, 2].imshow(to_rgb(sc[r], ref), interpolation="nearest")
        ax[r, 3].imshow(to_rgb(sb[r], ref), interpolation="nearest")
        im = ax[r, 4].imshow(diffs[r], cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax[r, 0].set_ylabel("FR scene %d\nHQNR %.4f (ctrl %.4f)\nD_s %.4f (ctrl %.4f)" % (s, hb, hc, ds_b, ds_c), fontsize=8.5)
        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=9.5)
    fig.suptitle("QRECON24 full-resolution (paper .mat 20, no GT) - %s vs control %s | selected step %s | stretch: per-band 1-99%% of EXP, shared per row"
                 % (short(best), short(ctrl), best["meta"].get("step")), fontsize=11)
    fig.subplots_adjust(top=0.945, left=0.075, right=0.995, wspace=0.02, hspace=0.06)
    hbar(fig, ax[:, 4], im, "mean |this - control|, DN (shared)")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig); return out


def fig_rr(best, ctrl, scenes, out, dpi):
    """RR 트랙 (GT 와 오차맵이 가능한 유일한 view; FR 과 다른 장면이다)."""
    rb, rc = best["rr"], ctrl["rr"]
    gt, lms, pan = rb["gt"].astype(np.float64), rb["lms"].astype(np.float64), rb["pan"].astype(np.float64)
    sb, sc = rb["sr"].astype(np.float64), rc["sr"].astype(np.float64)
    errs = [band_mean_abs(sb[s], gt[s]) for s in scenes]; diffs = [band_mean_abs(sb[s], sc[s]) for s in scenes]
    vmax_e = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99))
    vmax_d = float(np.percentile(np.concatenate([d.ravel() for d in diffs]), 99))
    cols = ["PAN (256 px)", "EXP  (lms)", "this run %s" % short(best), "GT", "|this - GT|  DN", "|this - control|  DN"]
    fig, ax = plt.subplots(len(scenes), 6, figsize=(18.6, 3.25 * len(scenes)))
    ax = np.atleast_2d(ax)
    for r, s in enumerate(scenes):
        ref = gt[s]
        eb, ab = rr_scene_metrics(sb[s], gt[s]); ec, _ = rr_scene_metrics(sc[s], gt[s])
        ax[r, 0].imshow(stretch1(pan[s, 0]), cmap="gray", interpolation="nearest")
        ax[r, 1].imshow(to_rgb(lms[s], ref), interpolation="nearest")
        ax[r, 2].imshow(to_rgb(sb[s], ref), interpolation="nearest")
        ax[r, 3].imshow(to_rgb(gt[s], ref), interpolation="nearest")
        ime = ax[r, 4].imshow(errs[r], cmap="inferno", vmin=0.0, vmax=vmax_e, interpolation="nearest")
        imd = ax[r, 5].imshow(diffs[r], cmap="inferno", vmin=0.0, vmax=vmax_d, interpolation="nearest")
        ax[r, 0].set_ylabel("RR scene %d\nERGAS %.4f (ctrl %.4f)\nSAM %.4f" % (s, eb, ec, ab), fontsize=8.5)
        for c in range(6):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=9.5)
    fig.suptitle("QRECON24 reduced-resolution (test h5 20, GT available; DIFFERENT scenes from the FR figure) - %s vs control %s | step %s | stretch: per-band 1-99%% of GT"
                 % (short(best), short(ctrl), best["meta"].get("step")), fontsize=11)
    fig.subplots_adjust(top=0.945, left=0.068, right=0.995, wspace=0.02, hspace=0.06)
    hbar(fig, ax[:, 4], ime, "mean |this - GT|, DN (shared)")
    hbar(fig, ax[:, 5], imd, "mean |this - control|, DN (shared)")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig); return out


def fig_zoom(best, ctrl, scenes, out, dpi, size=128, up=3):
    """FR 장면의 에지가 가장 많은 크롭을 nearest 확대 — 질감·번짐을 눈으로 보는 칸."""
    with h5py.File(os.path.join(ROOT, FR_H5), "r") as h:                                   # h5 fancy index 는 오름차순만 받는다 — 정렬해서 읽고 원 순서로 되돌린다
        order = sorted(range(len(scenes)), key=lambda k: scenes[k]); srt = [scenes[k] for k in order]; back = np.argsort(order)
        pan = np.asarray(h["pan"][srt, 0], dtype=np.float64)[back]; lms = np.asarray(h["lms"][srt], dtype=np.float64)[back]
    sb, sc = best["fr_sr"][list(scenes)], ctrl["fr_sr"][list(scenes)]
    cols = ["PAN", "EXP (lms)", "control %s" % short(ctrl), "this run %s" % short(best)]
    fig, ax = plt.subplots(len(scenes), 4, figsize=(13.4, 3.55 * len(scenes)))
    ax = np.atleast_2d(ax)
    rep = lambda a: np.repeat(np.repeat(a, up, axis=0), up, axis=1)
    for r, s in enumerate(scenes):
        y0, x0 = busiest_crop(pan[r], size); sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        ref = lms[r][(slice(None),) + sl]
        ax[r, 0].imshow(rep(stretch1(pan[r][sl])), cmap="gray", interpolation="nearest")
        ax[r, 1].imshow(rep(to_rgb(lms[r][(slice(None),) + sl], ref)), interpolation="nearest")
        ax[r, 2].imshow(rep(to_rgb(sc[r][(slice(None),) + sl], ref)), interpolation="nearest")
        ax[r, 3].imshow(rep(to_rgb(sb[r][(slice(None),) + sl], ref)), interpolation="nearest")
        ax[r, 0].set_ylabel("FR scene %d\ncrop %dx%d at (y=%d, x=%d), nearest x%d\nHQNR %.4f (ctrl %.4f)"
                            % (s, size, size, y0, x0, up, best["fr"]["per_scene_hqnr"][s], ctrl["fr"]["per_scene_hqnr"][s]), fontsize=8.5)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=10)
    fig.suptitle("QRECON24 full-resolution zoom (highest edge-energy crop; no interpolation) - %s vs control %s" % (short(best), short(ctrl)), fontsize=11)
    fig.subplots_adjust(top=0.93, left=0.11, right=0.99, wspace=0.02, hspace=0.06)
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig); return out


def short(d):
    """run 이름에서 profile+seed 만 (라벨용, ASCII)."""
    r = d["run"] if isinstance(d, dict) else d; p = r.replace("PAKD50_QRC24_", "").split("_W104")[0]
    sd = r.split("_T0_S")[1].split("_")[0] if "_T0_S" in r else "?"
    return "%s S%s" % (p.split("_", 1)[1] if p.startswith(("S1_", "S2_", "S3_", "S4_", "S5_")) else p, sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEF_RUN, help="정성 결과를 볼 run (기본: s1 최고 G23 S1234)")
    ap.add_argument("--control", default=DEF_CTRL, help="같은 서버·같은 seed 의 대조 run (기본: G22 S1234)")
    ap.add_argument("--out-dir", default=os.path.join("results_log", "assets")); ap.add_argument("--tag", default="0917_qrc24")
    ap.add_argument("--fr-scenes", default=",".join(map(str, DEF_FR))); ap.add_argument("--rr-scenes", default=",".join(map(str, DEF_RR)))
    ap.add_argument("--zoom", default=",".join(map(str, DEF_ZOOM))); ap.add_argument("--dpi", type=int, default=125)
    ap.add_argument("--only", default=None, choices=("fr", "rr", "zoom"), help="하나만 그린다")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in str(s).split(",") if str(x).strip() != ""]
    best, ctrl = load_run(a.run), load_run(a.control)
    od = os.path.join(ROOT, a.out_dir); os.makedirs(od, exist_ok=True); made = []
    print(f"[qrc24-qual] run {a.run}\n           control {a.control}\n           selected step {best['meta'].get('step')} · raw HQNR {best['fr'].get('hqnr'):.6f} (control {ctrl['fr'].get('hqnr'):.6f})")
    if a.only in (None, "fr"):
        made.append(fig_fr(best, ctrl, ints(a.fr_scenes), os.path.join(od, f"{a.tag}_fr_panels.png"), a.dpi))
    if a.only in (None, "rr"):
        made.append(fig_rr(best, ctrl, ints(a.rr_scenes), os.path.join(od, f"{a.tag}_rr_panels.png"), a.dpi))
    if a.only in (None, "zoom"):
        made.append(fig_zoom(best, ctrl, ints(a.zoom), os.path.join(od, f"{a.tag}_fr_zoom.png"), a.dpi))
    for p in made:
        print(f"   {os.path.relpath(p, ROOT)}  ({os.path.getsize(p) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
