"""EQREC4 보조 시각화 — sample 수준·feature 수준 자료.

`results_log/2026-09-15_s1_eqrec4-results.md` 의 통계 판정에 붙일 그림을 만든다.
초점은 **q(offset consistency) 와 e(native 복원 오차) 가 어긋나는 집단** —
EuCd(q↓·e↑) 와 EdCu(q↑·e↓) 가 실제로 어떤 patch 인가.

산출 → results_log/assets/0915_eqrec4_*.png

  S1 atlas       4분면 대표 patch 의 PAN/GT/출력/오차/edge (EuCd vs EdCu 대비)
  S2 fingerprint 4분면 feature 표준화 평균 + EuCd−EdCu Cohen's d
  S3 confound    텍스처를 통제하면 rho(q,e) 의 부호가 뒤집힌다 (기전)
  S4 profile     밴드별 오차 · stress · PAN 민감도 · learned-zero gain 의 4분면 대비

라벨은 ASCII (matplotlib 에 한글 글리프 없음).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

CAMP = os.path.join(ROOT, "work_dir", "_eqrec4_s1_campaign")
OUT = os.path.join(ROOT, "results_log", "assets")
PK = "L1E4_S2025_best_raw"                    # 보고서의 primary
QORDER = ["EdCd", "EdCu", "EuCd", "EuCu"]
QLAB = {"EdCd": "EdCd  e-low  q-low", "EdCu": "EdCu  e-low  q-HIGH",
        "EuCd": "EuCd  e-HIGH q-low", "EuCu": "EuCu  e-HIGH q-HIGH"}
QCOL = {"EdCd": "#4C78A8", "EdCu": "#54A24B", "EuCd": "#E45756", "EuCu": "#B279A2"}
RGB = (4, 2, 1)                               # WV3 8밴드 -> red/green/blue
MARGIN = 16                                   # native64 ROI = [16:48]^2


def load_table():
    nm = pd.read_csv(os.path.join(CAMP, "native_sample_metrics.csv"), low_memory=False)
    qa = pd.read_csv(os.path.join(CAMP, "quadrant_assignments.csv"))
    n64 = nm[nm.scale == "native64"]
    d = n64[n64.model_key == PK].merge(
        qa[qa.model_key == PK][["sample_id", "quadrant_id"]], on="sample_id")
    return nm, d                                  # nm 은 전 scale (S6 가 fr512 를 쓴다)


def stretch(a, lo=2, hi=98):
    """표시용 퍼센타일 스트레치 -> [0,1]."""
    a = np.asarray(a, dtype=np.float64)
    p, q = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p) / (q - p + 1e-12), 0, 1)


def to_rgb(cube, ref=None):
    """[8,H,W] -> [H,W,3]. ref 가 있으면 그 밴드 통계로 맞춘다(같은 장면 비교용)."""
    src = cube if ref is None else ref
    out = np.zeros(cube.shape[1:] + (3,))
    for j, b in enumerate(RGB):
        p, q = np.percentile(src[b], 2), np.percentile(src[b], 98)
        out[..., j] = np.clip((cube[b] - p) / (q - p + 1e-12), 0, 1)
    return out


def roi_box(ax, n=64, m=MARGIN):
    ax.add_patch(plt.Rectangle((m - .5, m - .5), n - 2 * m, n - 2 * m,
                               fill=False, ec="w", lw=1.0, ls=":"))


def pick_representative(d, q, k=3, seed=0):
    """4분면 안에서 e·q 순위가 중앙에 가까운 sample k개 (극단 cherry-pick 회피).
    동률은 source block 을 흩어 고른다."""
    s = d[d.quadrant_id == q].copy()
    for c in ("e_native_roi", "q_A"):
        s[c + "_r"] = s[c].rank(pct=True)
    s["dist"] = ((s.e_native_roi_r - .5) ** 2 + (s.q_A_r - .5) ** 2) ** .5
    s = s.sort_values("dist")
    out, used = [], set()
    for _, r in s.iterrows():                 # source block 중복 회피
        if r.source_group_id in used:
            continue
        out.append(r); used.add(r.source_group_id)
        if len(out) == k:
            break
    return pd.DataFrame(out)


# --------------------------------------------------------------- S1 atlas
def fig_atlas(d):
    from tools.eqrec4 import common as CM
    import torch

    pairs = ["EuCd", "EdCu"]                  # 사용자 요청의 두 집단
    picks = {q: pick_representative(d, q, k=3) for q in pairs}
    ids = [int(i) for q in pairs for i in picks[q].sample_id]
    gt, ms, lpan, pan = CM.load_patches(ids)
    L = CM.load_model(*PK.split("_")[0:1] + [int(PK.split("_")[1][1:])] + [PK.split("_", 2)[2]])
    with torch.no_grad():
        y, delta = CM.native_forward(L, pan, ms, lpan)
    gt_n, y_n, pan_n = gt.numpy(), y.numpy(), pan.numpy()[:, 0]
    err = np.abs(y_n - gt_n).mean(1)

    gx = np.gradient(pan_n, axis=2); gy = np.gradient(pan_n, axis=1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    emax = np.percentile(err[:, MARGIN:-MARGIN, MARGIN:-MARGIN], 99.5)
    mmax = np.percentile(mag, 99.5)

    cols = ["PAN (input)", "GT (RGB)", "model output (RGB)",
            "|output - GT|  mean over bands", "PAN gradient magnitude"]
    n = len(ids)
    fig, axes = plt.subplots(n, 5, figsize=(14.5, 2.95 * n))
    k = 0
    for q in pairs:
        for _, r in picks[q].iterrows():
            sid = int(r.sample_id)
            a = axes[k]
            a[0].imshow(stretch(pan_n[k]), cmap="gray")
            a[1].imshow(to_rgb(gt_n[k]))
            a[2].imshow(to_rgb(y_n[k], ref=gt_n[k]))
            im3 = a[3].imshow(err[k], cmap="inferno", vmin=0, vmax=emax)
            im4 = a[4].imshow(mag[k], cmap="viridis", vmin=0, vmax=mmax)
            for j, ax in enumerate(a):
                ax.set_xticks([]); ax.set_yticks([])
                roi_box(ax)
                if k == 0:
                    ax.set_title(cols[j], fontsize=9.5)
            plt.colorbar(im3, ax=a[3], fraction=.046, pad=.02)
            plt.colorbar(im4, ax=a[4], fraction=.046, pad=.02)
            a[0].set_ylabel(f"{q}  #{sid}\n" + r"$e$=" + f"{r.e_native_roi:.4f}  "
                            + r"$q_A$=" + f"{r.q_A:.3f}\ntex={r.pan_scharr_energy:.4f}  "
                            + r"$|\hat c_0|$=" + f"{r.c0_norm:.2f}px",
                            fontsize=8.5, color=QCOL[q])
            k += 1
    fig.suptitle("S1  The two disagreeing groups, as images  (L1E4 seed 2025 best_raw; "
                 "median-rank representatives, distinct source blocks)\n"
                 "EuCd = consistency GOOD (q low) but reconstruction BAD (e high)   vs   "
                 "EdCu = consistency BAD (q high) but reconstruction GOOD (e low)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUT, "0915_eqrec4_S1_atlas.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)
    return picks


# --------------------------------------------------------------- S2 fingerprint
FEAT = ["pan_scharr_energy", "pan_std", "pan_range", "ms_band_var_mean", "pan_msmean_corr",
        "pan_orient_entropy", "pan_mean", "edge_l1_roi", "A_fit_rmse", "A_sv_max",
        "A_sv_min", "A_B_yy", "A_B_xx", "epe_A", "c0_norm"]
NICE = {"pan_scharr_energy": "PAN Scharr energy (texture)", "pan_std": "PAN std (contrast)",
        "pan_range": "PAN range", "ms_band_var_mean": "MS band variance",
        "pan_msmean_corr": "PAN-MS structure corr", "pan_orient_entropy": "PAN orient. entropy",
        "pan_mean": "PAN mean (brightness)", "edge_l1_roi": "edge L1 (output)",
        "A_fit_rmse": "response fit RMSE", "A_sv_max": "response sv_max",
        "A_sv_min": "response sv_min", "A_B_yy": "response B_yy", "A_B_xx": "response B_xx",
        "epe_A": "offset EPE (bank A)", "c0_norm": "|c0| predicted shift"}


def fig_fingerprint(d):
    z = d.copy()
    for f in FEAT:
        z[f] = (z[f] - z[f].mean()) / (z[f].std(ddof=0) + 1e-12)
    M = z.groupby("quadrant_id")[FEAT].mean().reindex(QORDER)

    a_, b_ = d[d.quadrant_id == "EuCd"], d[d.quadrant_id == "EdCu"]
    rows = []
    for f in FEAT:
        x, y = a_[f].dropna(), b_[f].dropna()
        sp = np.sqrt((x.var(ddof=1) * (len(x) - 1) + y.var(ddof=1) * (len(y) - 1))
                     / (len(x) + len(y) - 2))
        rows.append((f, (x.mean() - y.mean()) / sp if sp > 0 else np.nan))
    cd = pd.DataFrame(rows, columns=["f", "d"]).sort_values("d")

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=.62)

    ax = fig.add_subplot(gs[0])
    v = np.abs(M.values).max()
    im = ax.imshow(M.values.T, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([QLAB[q].replace("  ", "\n", 1) for q in QORDER], fontsize=8.5)
    ax.set_yticks(range(len(FEAT))); ax.set_yticklabels([NICE[f] for f in FEAT], fontsize=8.5)
    for i in range(len(FEAT)):
        for j in range(4):
            ax.text(j, i, f"{M.values[j, i]:+.2f}", ha="center", va="center", fontsize=7.4,
                    color="w" if abs(M.values[j, i]) > .55 * v else "k")
    cb = plt.colorbar(im, ax=ax, fraction=.038, pad=.03)
    cb.set_label("standardized mean (z)", fontsize=8.5)
    ax.set_title("S2a  Feature fingerprint of each quadrant\n(z-scored over all 4,096 patches)",
                 fontsize=10.5)

    ax = fig.add_subplot(gs[1])
    c = ["#E45756" if x > 0 else "#54A24B" for x in cd.d]
    ax.barh(range(len(cd)), cd.d, color=c)
    ax.set_yticks(range(len(cd))); ax.set_yticklabels([NICE[f] for f in cd.f], fontsize=8.5)
    ax.axvline(0, c="k", lw=.8)
    for x in (-0.8, 0.8):
        ax.axvline(x, c="gray", ls=":", lw=.8)
    for i, x in enumerate(cd.d):
        ax.text(x + (.06 if x > 0 else -.06), i, f"{x:+.2f}", va="center",
                ha="left" if x > 0 else "right", fontsize=7.6)
    ax.set_xlabel("Cohen's d     <- higher in EdCu      higher in EuCd ->")
    ax.set_title("S2b  What separates EuCd (q low, e HIGH)\nfrom EdCu (q HIGH, e low)",
                 fontsize=10.5)
    ax.set_xlim(cd.d.min() - .5, cd.d.max() + .5)

    fig.suptitle("S2  The disagreement is a texture axis, not an alignment axis "
                 f"({PK})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "0915_eqrec4_S2_fingerprint.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)
    return cd


# --------------------------------------------------------------- S3 confound
def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey).correlation


def fig_confound(nm, d):
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = GridSpec(2, 3, hspace=.36, wspace=.42)

    # (a) q vs e, colour = texture
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(d.q_A, d.e_native_roi, c=np.log10(d.pan_scharr_energy + 1e-6),
                    s=5, cmap="magma", alpha=.75)
    ax.axvline(d.q_A.median(), c="k", lw=.8, ls="--")
    ax.axhline(d.e_native_roi.median(), c="k", lw=.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel(r"$q_A$  (offset consistency residual)")
    ax.set_ylabel(r"$e$  native ROI L1")
    cb = plt.colorbar(sc, ax=ax, fraction=.046, pad=.03)
    cb.set_label("log10 PAN Scharr energy", fontsize=8)
    r = spearmanr(d.q_A, d.e_native_roi).correlation
    ax.set_title(f"(a) raw: rho(q,e) = {r:+.3f}\nthe red/bright (textured) cloud sits "
                 "low-q, high-e", fontsize=10)

    # (b) sign flip across models
    ax = fig.add_subplot(gs[0, 1])
    keys = [f"{f}_S{s}_best_raw" for f in ("L1E4", "L000", "L1E2") for s in (1234, 7777, 2025)]
    lab, raw, par = [], [], []
    for k in keys:
        s = nm[(nm.model_key == k) & (nm.scale == "native64")].dropna(
            subset=["q_A", "e_native_roi", "pan_scharr_energy"])
        if len(s) < 100:
            continue
        lab.append(k.replace("_best_raw", "").replace("_S", " s"))
        raw.append(spearmanr(s.q_A, s.e_native_roi).correlation)
        par.append(partial_spearman(s.q_A.values, s.e_native_roi.values,
                                    s.pan_scharr_energy.values))
    x = np.arange(len(lab))
    ax.bar(x - .2, raw, .4, label="raw rho(q,e)", color="#B279A2")
    ax.bar(x + .2, par, .4, label="partial | texture", color="#4C78A8")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho"); ax.legend(fontsize=8.5)
    ax.set_title("(b) controlling for texture flips the sign\n"
                 "H1's predicted direction (rho>0) holds within texture", fontsize=10)

    # (c) within texture deciles
    ax = fig.add_subplot(gs[0, 2])
    dd = d.copy()
    dd["dec"] = pd.qcut(dd.pan_scharr_energy, 10, labels=False)
    rr = [spearmanr(g.q_A, g.e_native_roi).correlation for _, g in dd.groupby("dec")]
    ax.bar(range(10), rr, color=["#E45756" if v < 0 else "#4C78A8" for v in rr])
    ax.axhline(0, c="k", lw=.8)
    ax.axhline(spearmanr(dd.q_A, dd.e_native_roi).correlation, c="#B279A2", ls="--",
               label="pooled (raw)")
    ax.set_xlabel("PAN texture decile (low -> high)"); ax.set_ylabel("rho(q,e) within decile")
    ax.legend(fontsize=8.5)
    ax.set_title(f"(c) {sum(1 for v in rr if v > 0)}/10 deciles are positive\n"
                 "Simpson-style reversal", fontsize=10)

    # (d,e) the two marginals
    for i, (col, name, cmapc) in enumerate(
            [("e_native_roi", "e  native ROI L1", "#E45756"),
             ("q_A", r"$q_A$  consistency residual", "#4C78A8")]):
        ax = fig.add_subplot(gs[1, i])
        dd = d.copy(); dd["dec"] = pd.qcut(dd.pan_scharr_energy, 10, labels=False)
        g = dd.groupby("dec")[col]
        m, lo, hi = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(range(10), lo, hi, color=cmapc, alpha=.22)
        ax.plot(range(10), m, "o-", color=cmapc)
        ax.set_xlabel("PAN texture decile"); ax.set_ylabel(name)
        rho = spearmanr(d.pan_scharr_energy, d[col]).correlation
        ax.set_title(f"({'de'[i]}) texture -> {col.split('_')[0]}   rho = {rho:+.3f}",
                     fontsize=10)

    # (f) quadrant texture distribution
    ax = fig.add_subplot(gs[1, 2])
    for q in QORDER:
        v = np.log10(d[d.quadrant_id == q].pan_scharr_energy + 1e-6)
        ax.hist(v, bins=45, histtype="step", lw=1.9, color=QCOL[q], label=QLAB[q], density=True)
    ax.set_xlabel("log10 PAN Scharr energy"); ax.set_ylabel("density")
    ax.legend(fontsize=7.6)
    ax.set_title("(f) EuCd is the textured tail;\nEdCu is the flat tail", fontsize=10)

    fig.suptitle("S3  Why q and e disagree: PAN texture drives e UP and q DOWN at the same "
                 f"time  ({PK}; texture = Scharr energy of the PAN patch)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = os.path.join(OUT, "0915_eqrec4_S3_confound.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)
    return list(zip(lab, raw, par))


# --------------------------------------------------------------- S4 profiles
def fig_profiles(d):
    fig = plt.figure(figsize=(14.5, 8.2))
    gs = GridSpec(2, 3, hspace=.36, wspace=.28)

    # (a) per-band error
    ax = fig.add_subplot(gs[0, 0])
    bands = [f"e_band{i}_roi" for i in range(8)]
    for q in QORDER:
        s = d[d.quadrant_id == q][bands].mean()
        ax.plot(range(8), s.values, "o-", color=QCOL[q], label=QLAB[q])
    ax.set_xticks(range(8)); ax.set_xlabel("WV3 band index")
    ax.set_ylabel("mean ROI L1"); ax.legend(fontsize=7.4)
    ax.set_title("(a) spectral error profile\nsame shape, different scale", fontsize=10)

    # (b) shape-normalised
    ax = fig.add_subplot(gs[0, 1])
    for q in QORDER:
        s = d[d.quadrant_id == q][bands].mean()
        ax.plot(range(8), s.values / s.values.mean(), "o-", color=QCOL[q])
    ax.set_xticks(range(8)); ax.set_xlabel("WV3 band index")
    ax.set_ylabel("band L1 / mean"); ax.axhline(1, c="k", lw=.8, ls=":")
    ax.set_title("(b) normalised: the quadrants are NOT\na spectral phenomenon", fontsize=10)

    # (c) stress by quadrant and path
    st = pd.read_csv(os.path.join(CAMP, "output_stress.csv"), low_memory=False)
    s = st[(st.model_key == PK) & (st.scale == "native64") & (st.probe_bank == "B")
           & st.valid_support]
    ax = fig.add_subplot(gs[0, 2])
    g = s.groupby(["quadrant_primary", "path"]).d_e.mean().unstack().reindex(QORDER)
    g.plot.bar(ax=ax, color=["#4C78A8", "#E45756", "#9D9D9D"], width=.78)
    ax.set_ylabel("d_e  (stress ROI L1 - native)"); ax.set_xlabel("")
    ax.set_xticklabels(QORDER, rotation=0)
    ax.legend(fontsize=7.6, title="path", title_fontsize=7.6)
    ax.set_title("(c) response < no_response everywhere\ncorrection does work", fontsize=10)

    # (d) PAN sensitivity
    ps = pd.read_csv(os.path.join(CAMP, "pan_sensitivity.csv"))
    ps = ps[ps.model_key == PK]
    ax = fig.add_subplot(gs[1, 0])
    g = ps.groupby(["quadrant_primary", "variant"]).d_e_c0_fixed.mean().unstack().reindex(QORDER)
    g.plot.bar(ax=ax, width=.78)
    ax.set_ylabel("d_e  (PAN perturbed - native)"); ax.set_xlabel("")
    ax.set_xticklabels(QORDER, rotation=0); ax.legend(fontsize=7.4, title="PAN variant",
                                                      title_fontsize=7.4)
    ax.set_title("(d) EuCd is the most PAN-content\nsensitive group", fontsize=10)

    # (e) learned - zero gain vs q
    ci = pd.read_csv(os.path.join(CAMP, "correction_interventions_core.csv"))
    ci = ci[ci.model_key == PK].merge(d[["sample_id", "q_A", "pan_scharr_energy"]],
                                      on="sample_id")
    ax = fig.add_subplot(gs[1, 1])
    for q in QORDER:
        s = ci[ci.quadrant_own == q]
        ax.scatter(s.q_A, s.gain_learned_vs_zero_roi, s=5, alpha=.5, color=QCOL[q],
                   label=f"{q}  n={len(s)}")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xlabel(r"$q_A$"); ax.set_ylabel("gain  (zero-corr L1 - learned-corr L1)")
    ax.set_yscale("symlog", linthresh=1e-4); ax.legend(fontsize=7.4)
    rho = spearmanr(ci.q_A, ci.gain_learned_vs_zero_roi).correlation
    ax.set_title(f"(e) H3b  rho(q, gain) = {rho:+.3f}\nlow q -> more to correct", fontsize=10)

    # (f) gain vs texture - the same axis
    ax = fig.add_subplot(gs[1, 2])
    ci["dec"] = pd.qcut(ci.pan_scharr_energy, 10, labels=False)
    g = ci.groupby("dec").gain_learned_vs_zero_roi
    ax.fill_between(range(10), g.quantile(.25), g.quantile(.75), color="#E45756", alpha=.22)
    ax.plot(range(10), g.median(), "o-", color="#E45756")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xlabel("PAN texture decile"); ax.set_ylabel("gain (learned - zero)")
    rho = spearmanr(ci.pan_scharr_energy, ci.gain_learned_vs_zero_roi).correlation
    ax.set_title(f"(f) gain tracks texture too   rho = {rho:+.3f}\n"
                 "H3b and H1 share one axis", fontsize=10)

    fig.suptitle(f"S4  Behaviour of the four groups under stress, PAN perturbation and "
                 f"correction ablation  ({PK})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = os.path.join(OUT, "0915_eqrec4_S4_profiles.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


# --------------------------------------------------------------- S5 plane grid
def fig_plane(d, nb=5):
    """(q_A, e) 평면을 분위수 nb x nb 로 잘라 각 칸의 대표 PAN patch 를 그 자리에 놓는다.
    4분면은 이 평면을 중앙값으로 두 번 자른 것이라, 칸 사이 '중간' 이 어떻게 변하는지 본다."""
    from tools.eqrec4 import common as CM

    z = d.copy()
    z["qb"] = pd.qcut(z.q_A, nb, labels=False)
    z["eb"] = pd.qcut(z.e_native_roi, nb, labels=False)
    reps, tex, cnt = {}, np.full((nb, nb), np.nan), np.zeros((nb, nb), int)
    for (ib, jb), g in z.groupby(["eb", "qb"]):
        cnt[ib, jb] = len(g)
        tex[ib, jb] = np.log10(g.pan_scharr_energy.mean() + 1e-9)
        gg = g.copy()
        gg["dist"] = ((gg.q_A.rank(pct=True) - .5) ** 2
                      + (gg.e_native_roi.rank(pct=True) - .5) ** 2) ** .5
        reps[(ib, jb)] = gg.sort_values("dist").iloc[0]

    ids = [int(r.sample_id) for r in reps.values()]
    _, _, _, pan = CM.load_patches(ids)
    pnp = pan.numpy()[:, 0]
    pmap = {sid: pnp[i] for i, sid in enumerate(ids)}

    fig = plt.figure(figsize=(15.0, 7.4))
    outer = GridSpec(1, 2, width_ratios=[1.32, 1], wspace=.22)
    inner = outer[0].subgridspec(nb, nb, hspace=.06, wspace=.06)

    for ib in range(nb):
        for jb in range(nb):
            ax = fig.add_subplot(inner[nb - 1 - ib, jb])       # e 는 위로 갈수록 크게
            r = reps.get((ib, jb))
            if r is None:
                ax.axis("off"); continue
            ax.imshow(stretch(pmap[int(r.sample_id)]), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            qd = "EdC" if ib < nb / 2 else "EuC"
            qd += "d" if jb < nb / 2 else "u"
            for s in ax.spines.values():
                s.set_color(QCOL[qd]); s.set_linewidth(2.2)
            ax.text(.03, .96, f"{r.pan_scharr_energy:.3f}", transform=ax.transAxes,
                    fontsize=6.6, va="top", color="yellow")
            if jb == 0:
                ax.set_ylabel(f"{z[z.eb == ib].e_native_roi.median():.3f}", fontsize=7.6)
            if ib == 0:
                ax.set_xlabel(f"{z[z.qb == jb].q_A.median():.3f}", fontsize=7.6)
    fig.text(.012, .5, r"$e$  native ROI L1  (quintile median) $\rightarrow$", rotation=90,
             va="center", fontsize=9.5)
    fig.text(.30, .028, r"$q_A$  consistency residual  (quintile median) $\rightarrow$",
             ha="center", fontsize=9.5)
    fig.text(.30, .905, "S5a  PAN patch at each cell of the (q, e) plane\n"
             "yellow = Scharr energy;  frame colour = which quadrant the cell falls in",
             ha="center", fontsize=10)

    ax = fig.add_subplot(outer[1])
    im = ax.imshow(tex, origin="lower", cmap="magma", aspect="auto")
    for ib in range(nb):
        for jb in range(nb):
            if not np.isnan(tex[ib, jb]):
                ax.text(jb, ib, f"{10 ** tex[ib, jb]:.4f}\nn={cnt[ib, jb]}", ha="center",
                        va="center", fontsize=7.4,
                        color="w" if tex[ib, jb] < np.nanmean(tex) else "k")
    ax.axvline(nb / 2 - .5, c="w", lw=1.6); ax.axhline(nb / 2 - .5, c="w", lw=1.6)
    ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
    ax.set_xticklabels([f"Q{j+1}" for j in range(nb)])
    ax.set_yticklabels([f"E{i+1}" for i in range(nb)])
    ax.set_xlabel(r"$q_A$ quintile  (low $\rightarrow$ high)")
    ax.set_ylabel(r"$e$ quintile  (low $\rightarrow$ high)")
    plt.colorbar(im, ax=ax, fraction=.046, pad=.03, label="log10 mean Scharr energy")
    ax.set_title("S5b  Mean PAN texture per cell\n"
                 "texture rises going UP (e) and LEFT (q) — one diagonal axis,\n"
                 "which is exactly why the two labels disagree", fontsize=10)

    fig.suptitle("S5  The plane between the quadrants - how samples change as q and e move "
                 f"({PK}, quintile bins)", fontsize=12, y=.985)
    fig.tight_layout(rect=[.02, .05, 1, .875])
    p = os.path.join(OUT, "0915_eqrec4_S5_plane.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


# --------------------------------------------------------------- S6 patch vs scene
def fig_scene(nm):
    """patch 수준의 텍스처 교란이 scene 수준(H1 FR)에도 적용되는지 — 적용되지 않는다."""
    import h5py
    import torch
    from tools.eqrec4 import common as CM

    dp = CM.data_paths()
    with h5py.File(dp["fr"]) as f:
        pan = np.asarray(f["pan"], dtype=np.float32)
        ms = np.asarray(f["ms"], dtype=np.float32)
    st = CM.input_stats(torch.from_numpy(pan) * 2 / CM.MAX_PIXEL - 1,
                        torch.from_numpy(ms) * 2 / CM.MAX_PIXEL - 1)
    tex = st["pan_scharr_energy"].numpy()
    fr = nm[nm.scale == "fr512"]
    seeds = (1234, 7777, 2025)

    fig = plt.figure(figsize=(14.5, 4.6))
    gs = GridSpec(1, 3, wspace=.31)

    ax = fig.add_subplot(gs[0])
    for s, c in zip(seeds, ("#4C78A8", "#E45756", "#54A24B")):
        d = fr[fr.model_key == f"L1E4_S{s}_best_raw"].dropna(
            subset=["q_A", "raw_original_hqnr"]).sort_values("sample_id")
        r = spearmanr(d.q_A, d.raw_original_hqnr).correlation
        ax.scatter(d.q_A, d.raw_original_hqnr, s=34, color=c, label=f"seed {s}  rho={r:+.2f}")
    ax.set_xlabel(r"scene $q_A$"); ax.set_ylabel("scene raw HQNR")
    ax.legend(fontsize=8)
    ax.set_title("(a) FR20 scene level: low q -> high HQNR\n(H1 FR, Supported 3/3)",
                 fontsize=10)

    ax = fig.add_subplot(gs[1])
    lab, raw, par = [], [], []
    for s in seeds:
        d = fr[fr.model_key == f"L1E4_S{s}_best_raw"].dropna(
            subset=["q_A", "raw_original_hqnr"]).sort_values("sample_id")
        t = tex[d.sample_id.astype(int).values]
        lab.append(f"s{s}")
        raw.append(spearmanr(d.q_A, d.raw_original_hqnr).correlation)
        par.append(partial_spearman(d.q_A.values, d.raw_original_hqnr.values, t))
    # patch 수준 비교값
    plab, praw, ppar = [], [], []
    for s in seeds:
        g = nm[(nm.model_key == f"L1E4_S{s}_best_raw") & (nm.scale == "native64")].dropna(
            subset=["q_A", "e_native_roi", "pan_scharr_energy"])
        plab.append(f"s{s}")
        praw.append(spearmanr(g.q_A, g.e_native_roi).correlation)
        ppar.append(partial_spearman(g.q_A.values, g.e_native_roi.values,
                                     g.pan_scharr_energy.values))
    x = np.arange(3)
    ax.bar(x - .2, praw, .4, color="#B279A2", label="raw")
    ax.bar(x + .2, ppar, .4, color="#4C78A8", label="partial | texture")
    ax.bar(x + 4 - .2, raw, .4, color="#B279A2")
    ax.bar(x + 4 + .2, par, .4, color="#4C78A8")
    ax.axhline(0, c="k", lw=.8); ax.axvline(3.5, c="gray", ls=":")
    ax.set_xticks(list(x) + list(x + 4)); ax.set_xticklabels(plab + lab, fontsize=8.5)
    ax.text(1, .62, "patch:  rho(q, e)\nsign FLIPS", ha="center", fontsize=9, color="#4C78A8")
    ax.text(5, .62, "scene:  rho(q, HQNR)\nunchanged", ha="center", fontsize=9, color="#333")
    ax.set_ylim(-1.05, 1.05); ax.set_ylabel("Spearman rho"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(b) the texture confound is a PATCH-level\nartefact; it does not explain the "
                 "scene effect", fontsize=10)

    ax = fig.add_subplot(gs[2])
    d = fr[fr.model_key == "L1E4_S2025_best_raw"].dropna(
        subset=["q_A", "raw_original_hqnr"]).sort_values("sample_id")
    t = tex[d.sample_id.astype(int).values]
    ax.scatter(t, d.raw_original_hqnr, s=34, color="#E45756", label="HQNR")
    ax.set_xlabel("scene PAN Scharr energy"); ax.set_ylabel("scene raw HQNR", color="#E45756")
    r1 = spearmanr(t, d.raw_original_hqnr).correlation
    ax2 = ax.twinx()
    ax2.scatter(t, d.q_A, s=34, marker="^", color="#4C78A8")
    ax2.set_ylabel(r"scene $q_A$", color="#4C78A8")
    r2 = spearmanr(t, d.q_A).correlation
    ax.set_title(f"(c) at scene level texture predicts neither\n"
                 f"rho(tex,HQNR)={r1:+.2f}   rho(tex,q)={r2:+.2f}  (n=20)", fontsize=10)

    fig.suptitle("S6  Patch level and scene level are different phenomena "
                 "(L1E4, best_raw, 3 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .9])
    p = os.path.join(OUT, "0915_eqrec4_S6_scene.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    nm, d = load_table()
    print(f"{PK}: n={len(d)}  quadrants={dict(d.quadrant_id.value_counts())}")
    fig_atlas(d)
    fig_fingerprint(d)
    fig_confound(nm, d)
    fig_profiles(d)
    fig_plane(d)
    fig_scene(nm)


if __name__ == "__main__":
    main()
