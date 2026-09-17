"""PAN aligner 검증 보고서용 그림 (2026-09-17).

기 실험 산출물만 읽는다 (PALS24 / PALSV18 / NF16 / PO10 + tools/aligner_constant_control.py).
산출 → results_log/assets/0917_aligner_*.png   라벨은 ASCII.

  A1 epsilon   lambda_off 별 응답 B_diag 과 학습 중 궤적 (이상 -1, donor -0.91)
  A2 predict   scene 별 예측 대 proxy 실제 오정합 (상관·수축)
  A3 inverted  정합 능력이 좋을수록 HQNR 이 나쁘다 (P1 frozen donor 가 최악)
  A4 selection 선택 있는 best 대 선택 없는 last/plateau
  A5 constant  전역 상수 하나가 scene 별 예측과 같은 HQNR 을 낸다
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_log", "assets")
CAMP = os.path.join(ROOT, "work_dir", "_palsv18_campaign")
BAND = 0.0031

LAM = {"CTRLP0": None, "P0": None, "L000": 0.0, "P2": 0.0, "L3E5": 3e-5, "L1E4": 1e-4,
       "L3E4": 3e-4, "L1E3": 1e-3, "L3E3": 3e-3, "P3": 1e-2, "P1": None, "N2": None}
CCOL = {"L000": "#9D9D9D", "L3E5": "#72B7B2", "L1E4": "#4C78A8", "L3E4": "#F58518",
        "L1E3": "#E45756", "L3E3": "#B279A2", "P3": "#54A24B"}


def diag_rows():
    """po10_diag 의 fr512 응답 (B_diag, |c|, EPE) 을 case/seed/ckpt 로."""
    rows, seen = [], set()
    for p in sorted(glob.glob(f"{ROOT}/work_dir/*/results/po10_diag_*_palsv18.json")) + \
             sorted(glob.glob(f"{ROOT}/work_dir/*/results/po10_diag_*_pals24.json")):
        run = p.split("/")[-3]
        ck = re.sub(r"^po10_diag_|_(palsv18|pals24)\.json$", "", os.path.basename(p))
        if (run, ck) in seen:
            continue
        seen.add((run, ck))
        m = re.match(r"(PALSV18|PALS24|NF16|PO10)_([A-Z0-9]+)_", run)
        case = m.group(2) if m else run
        s = re.search(r"_S(\d+)_", run)
        try:
            fr = json.load(open(p))["response"]["fr512"]
        except Exception:
            continue
        bd = fr.get("B_diag") or [None, None]
        if bd[0] is None:
            continue
        rows.append(dict(case=case, seed=(s.group(1) if s else "-"), ckpt=ck,
                         B=(bd[0] + bd[1]) / 2, c=fr.get("native_c0_norm_median"),
                         epe=fr.get("offset_epe"),
                         iqr=(fr.get("native_c0_iqr") or [np.nan])[0]))
    return pd.DataFrame(rows)


def fig_epsilon(D):
    fig = plt.figure(figsize=(14.5, 5.2))
    gs = GridSpec(1, 3, wspace=.29)

    ax = fig.add_subplot(gs[0])
    d = D[(D.ckpt == "best_hqnr") & D.case.isin(LAM) & D.case.map(lambda c: LAM.get(c) is not None)]
    cs = sorted({c for c in d.case}, key=lambda c: LAM[c])
    pos = {c: i for i, c in enumerate(cs)}
    for case, g in d.groupby("case"):
        ax.scatter([pos[case]] * len(g), g.B, s=52, color=CCOL.get(case, "#333"), label=case,
                   zorder=3)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([("0" if LAM[c] == 0 else f"{LAM[c]:g}") for c in cs], fontsize=8.5)
    ax.set_xlim(-.6, len(cs) - .4)
    ax.axhline(-1, c="k", ls=":", lw=1.2)
    ax.text(.02, .03, "ideal  B = -1 (correction cancels the injected eps)", fontsize=7.6,
            transform=ax.transAxes)
    don = D[(D.case == "N2") & (D.ckpt == "last")].B.mean()
    ax.axhline(don, c="#E45756", ls="--", lw=1.4)
    ax.text(.02, .10, f"donor N2 (PO10 displacement supervision)  {don:.2f}",
            fontsize=7.6, color="#E45756", transform=ax.transAxes)
    ax.set_xlabel(r"$\lambda_{off}$  (offset consistency weight)")
    ax.set_ylabel(r"$B_{diag}$   response to injected $\epsilon$")
    ax.legend(fontsize=7.4, ncol=2, loc="upper left")
    ax.set_title("A1a  the chosen lambda* = 1e-4 sits at\n~1/3 of the donor's ability",
                 fontsize=10)
    ax.axvline(pos.get("L1E4", 0), c="#4C78A8", ls=":", lw=1)

    ax = fig.add_subplot(gs[1])
    order = {"checkpoint-10000": 10000, "epoch-125": 25000, "best_hqnr": None, "last": 50000}
    for case in ("L000", "L3E5", "L1E4", "L3E4"):
        g = D[(D.case == case) & D.ckpt.isin(("checkpoint-10000", "epoch-125", "last"))]
        if g.empty:
            continue
        xs, ys = [], []
        for ck, x in (("checkpoint-10000", 10), ("epoch-125", 25), ("last", 50)):
            v = g[g.ckpt == ck].B
            if len(v):
                xs.append(x); ys.append(v.mean())
        ax.plot(xs, ys, "o-", color=CCOL.get(case, "#333"), label=case)
    ax.axhline(don, c="#E45756", ls="--", lw=1.2)
    ax.text(12, don + .02, "donor level at init", fontsize=8, color="#E45756")
    ax.axhline(-1, c="k", ls=":", lw=1)
    ax.set_xlabel("training step (k)"); ax.set_ylabel(r"$B_{diag}$")
    ax.legend(fontsize=8)
    ax.set_title("A1b  joint fine-tuning destroys the donor within 10k;\n"
                 "lambda_off only partly rebuilds it", fontsize=10)

    ax = fig.add_subplot(gs[2])
    d2 = D[(D.ckpt == "best_hqnr") & D.c.notna() & D.B.notna()]
    ax.scatter(d2.c, -d2.B, s=44, c=[CCOL.get(c, "#333") for c in d2.case], zorder=3)
    for _, r in d2.iterrows():
        ax.annotate(r.case, (r.c, -r.B), fontsize=6.6, xytext=(3, 3),
                    textcoords="offset points")
    ax.axvline(1.749, c="#E45756", ls="--", lw=1.3)
    ax.text(1.76, .3, "proxy-estimated true\nmisalignment 1.75 px", fontsize=8, color="#E45756")
    ax.set_xlabel(r"predicted shift magnitude  $|c|$  (HR px)")
    ax.set_ylabel(r"$-B_{diag}$  (1 = ideal)")
    ax.set_title("A1c  responsiveness and applied shift move together;\n"
                 "both fall far short of the real misalignment", fontsize=10)

    fig.suptitle("A1  Does the epsilon mechanism actually train the aligner?  "
                 "(FR paper mat20, W112-D123 family)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .9])
    p = os.path.join(OUT, "0917_aligner_A1_epsilon.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def proxy_table():
    rows = []
    for p in sorted(glob.glob(f"{ROOT}/work_dir/*/palsv18/*_native_proxy.csv")):
        run = p.split("/")[-3]
        ck = os.path.basename(p).replace("_native_proxy.csv", "")
        d = pd.read_csv(p)
        if "canon_before_dy" not in d:
            continue
        m = re.match(r"(PALSV18|PALS24|NF16)_([A-Z0-9]+)_", run)
        s = re.search(r"_S(\d+)_", run)
        rows.append(dict(case=(m.group(2) if m else run), seed=(s.group(1) if s else "-"),
                         ckpt=ck, df=d))
    return rows


def fig_predict(P):
    use = [r for r in P if r["ckpt"] == "best_hqnr"]
    fig = plt.figure(figsize=(14.5, 5.0))
    gs = GridSpec(1, 3, wspace=.3)

    ax = fig.add_subplot(gs[0])
    for r in use:
        d = r["df"]
        ax.scatter(d.canon_before_dy, d.delta_dy, s=22, alpha=.8,
                   color=CCOL.get(r["case"], "#333"))
    allc = pd.concat([r["df"] for r in use])
    rr = pearsonr(allc.canon_before_dy, allc.delta_dy)[0]
    lo, hi = allc.canon_before_dy.min(), allc.canon_before_dy.max()
    ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="perfect (slope 1)")
    b = np.polyfit(allc.canon_before_dy, allc.delta_dy, 1)
    ax.plot([lo, hi], np.polyval(b, [lo, hi]), "r-", lw=1.4,
            label=f"fit slope {b[0]:.2f}")
    ax.set_xlabel("proxy TRUE misalignment  dy (HR px)")
    ax.set_ylabel("model predicted correction  dy")
    ax.legend(fontsize=8)
    ax.set_title(f"A2a  the prediction tracks the truth\nr = {rr:+.2f}, but slope {b[0]:.2f} "
                 "= heavy shrinkage", fontsize=10)

    ax = fig.add_subplot(gs[1])
    lab, mag, tru, aft = [], [], [], []
    for r in use:
        d = r["df"]
        lab.append(f"{r['case']}\ns{r['seed']}")
        mag.append(d.delta_norm.mean()); tru.append(d.before_mag.mean())
        aft.append(d.after_mag.mean())
    x = np.arange(len(lab))
    ax.bar(x - .27, tru, .27, label="misalignment before", color="#E45756")
    ax.bar(x, mag, .27, label="correction applied", color="#4C78A8")
    ax.bar(x + .27, aft, .27, label="misalignment after", color="#54A24B")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("HR px"); ax.legend(fontsize=8)
    ax.set_title("A2b  it removes ~40% of the misalignment\nand never closes the gap",
                 fontsize=10)

    ax = fig.add_subplot(gs[2])
    for r in use:
        d = r["df"]
        ax.scatter([d.canon_before_dy.std(ddof=1)], [d.delta_dy.std(ddof=1)], s=60,
                   color=CCOL.get(r["case"], "#333"))
        ax.annotate(f"{r['case']} s{r['seed']}",
                    (d.canon_before_dy.std(ddof=1), d.delta_dy.std(ddof=1)),
                    fontsize=6.6, xytext=(4, 2), textcoords="offset points")
    m = max(allc.canon_before_dy.std(ddof=1), .25)
    ax.plot([0, m], [0, m], "k:", lw=1, label="tracks all variation")
    ax.set_xlabel("scene-to-scene sd of TRUE dy")
    ax.set_ylabel("scene-to-scene sd of PREDICTED dy")
    ax.legend(fontsize=8)
    ax.set_title("A2c  the per-scene part is small in absolute\nterms (sd <= 0.13 px)",
                 fontsize=10)

    fig.suptitle("A2  Are the predicted values meaningful?  "
                 "(FR20 scenes; 'true' = Scharr-ZNCC proxy, not sensor GT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .9])
    p = os.path.join(OUT, "0917_aligner_A2_predict.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def fig_inverted(D):
    fr = {}
    for d in sorted(glob.glob(f"{ROOT}/work_dir/NF16_P[0-4]_*S1234*")) + \
             sorted(glob.glob(f"{ROOT}/work_dir/PALS24_*S2025*N2LAST_R200_v1")) + \
             sorted(glob.glob(f"{ROOT}/work_dir/PO10_N2_*")):
        f = os.path.join(d, "results", "fr_mat20.json")
        if os.path.exists(f):
            fr[os.path.basename(d)] = json.load(open(f))
    pts = []
    NAME = {"NF16_P0": ("P0 no aligner", 0.0, None), "NF16_P1": ("P1 frozen donor", 1.511, -0.907),
            "NF16_P2": ("P2 lambda 0", 0.347, -0.197), "NF16_P3": ("P3 lambda 1e-2", 1.138, -0.759),
            "NF16_P4": ("P4 +geometry", None, None),
            "PALS24_L1E4": ("lambda* 1e-4", 0.452, -0.343),
            "PALS24_CTRLP0": ("CTRLP0 no aligner", 0.0, None),
            "PALS24_L000": ("lambda 0", 0.351, -0.196)}
    for run, j in fr.items():
        key = next((k for k in NAME if run.startswith(k)), None)
        if not key:
            continue
        nm, c, B = NAME[key]
        pts.append(dict(name=nm, c=c, B=B, hqnr=j.get("hqnr"), d_s=j.get("d_s"),
                        d_lam=j.get("d_lambda"), fscc=j.get("fscc")))
    t = pd.DataFrame(pts).dropna(subset=["hqnr"]).drop_duplicates("name")

    fig = plt.figure(figsize=(14.5, 5.0))
    gs = GridSpec(1, 3, wspace=.31)
    for i, (xc, xl, tt) in enumerate([
            ("c", r"applied shift $|c|$ (HR px)",
             "A3a  more alignment -> WORSE HQNR"),
            ("d_s", r"$D_s$ (spatial distortion vs ORIGINAL PAN)",
             "A3b  the penalty lands on $D_s$"),
            ("fscc", "fSCC vs ORIGINAL PAN",
             "A3c  structural fidelity is what is traded away")]):
        ax = fig.add_subplot(gs[i])
        s = t.dropna(subset=[xc])
        ax.scatter(s[xc], s.hqnr, s=70, c=["#E45756" if "frozen" in n else "#4C78A8"
                                           for n in s.name], zorder=3)
        for _, r in s.iterrows():
            ax.annotate(r["name"], (r[xc], r.hqnr), fontsize=7, xytext=(5, 3),
                        textcoords="offset points")
        if xc == "c":
            ax.axvline(1.749, c="#E45756", ls="--", lw=1.2)
            ax.text(1.4, s.hqnr.min(), "true 1.75 px", fontsize=7.6, color="#E45756",
                    rotation=90, va="bottom")
        ax.set_xlabel(xl); ax.set_ylabel("raw HQNR (FR paper mat20)")
        ax.set_title(tt, fontsize=10)

    fig.suptitle("A3  The model that aligns BEST scores WORST -- HQNR does not reward "
                 "alignment (W112-D123, seed 1234/2025 best_hqnr)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .9])
    p = os.path.join(OUT, "0917_aligner_A3_inverted.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def fig_selection():
    D = {}
    for d in sorted(glob.glob(f"{ROOT}/work_dir/PALS24_*_N2LAST_R200_v1")
                    + glob.glob(f"{ROOT}/work_dir/PALSV18_*_N2LAST_R200_v1")
                    + glob.glob(f"{ROOT}/work_dir/NF16_P[023]_*")):
        f = os.path.join(d, "checkpoint_metrics.csv")
        if not os.path.exists(f):
            continue
        run = os.path.basename(d)
        m = re.search(r"(PALS24|PALSV18|NF16)_([A-Z0-9]+)_", run)
        case = m.group(2) if m else run
        case = {"P0": "CTRLP0", "P2": "L000"}.get(case, case)
        s = re.search(r"_S(\d+)_", run)
        t = pd.read_csv(f).sort_values("step")
        h = t["raw_original.hqnr"]
        D[(case, s.group(1) if s else "-")] = dict(
            best=h.max(), last=h.iloc[-1],
            plateau=h[t.step >= t.step.max() * .8].mean())

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    cases = ["L000", "L3E5", "L1E4", "L3E4"]
    w = .26
    for i, stat in enumerate(("best", "last", "plateau")):
        vals, errs = [], []
        for c in cases:
            v = [D[(c, s)][stat] - D[("CTRLP0", s)][stat] for s in ("1234", "2025", "7777")
                 if (c, s) in D and ("CTRLP0", s) in D]
            vals.append(np.mean(v)); errs.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        x = np.arange(len(cases)) + (i - 1) * w
        ax.bar(x, vals, w, yerr=errs, capsize=3,
               label={"best": "best ckpt (SELECTED on this same FR20)",
                      "last": "last (no selection)",
                      "plateau": "plateau, top-20% steps (no selection)"}[stat],
               color=["#E45756", "#4C78A8", "#54A24B"][i])
    ax.axhline(0, c="k", lw=.9)
    ax.axhspan(-BAND, BAND, color="gray", alpha=.20, zorder=0)
    ax.text(len(cases) - .5, BAND, f"  judgment band +/-{BAND}", fontsize=8, va="bottom",
            ha="right")
    ax.set_xticks(range(len(cases))); ax.set_xticklabels(cases)
    ax.set_ylabel("HQNR difference vs CTRLP0 (no aligner)")
    ax.legend(fontsize=8.5)
    ax.set_title("A4  Remove the test-set checkpoint selection and the gain falls inside "
                 "the band\n(3 seeds; lambda*=L1E4: +0.0044 selected -> +0.0023 at last)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "0917_aligner_A4_selection.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def fig_constant():
    rows = []
    for p in sorted(glob.glob(f"{ROOT}/work_dir/*/palsv18/best_hqnr_constant_control.json")):
        j = json.load(open(p))
        s = re.search(r"_S(\d+)_", j["run"])
        rows.append(dict(seed=s.group(1) if s else "-", **j["modes"]["learned"],
                         hq_learned=j["modes"]["learned"]["hqnr"],
                         hq_const=j["modes"]["const_mean"]["hqnr"],
                         hq_zero=j["modes"]["zero"]["hqnr"],
                         hq_shuf=j["modes"]["shuffle"]["hqnr"],
                         d_lz=j["learned_minus_zero"], d_lc=j["learned_minus_const_mean"],
                         d_ls=j["learned_minus_shuffle"],
                         sd_dy=j["learned_c_sd_dy"], cnorm=j["learned_c_norm_mean"]))
    if not rows:
        print("constant_control 결과 없음 — 건너뜀"); return
    t = pd.DataFrame(rows).sort_values("seed")

    fig = plt.figure(figsize=(13.2, 5.0))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=.3)

    ax = fig.add_subplot(gs[0])
    x = np.arange(len(t)); w = .2
    for i, (k, lab, c) in enumerate([
            ("hq_zero", "no correction", "#9D9D9D"),
            ("hq_const", "ONE global constant (mean of the predictions)", "#54A24B"),
            ("hq_learned", "learned per-scene prediction", "#4C78A8"),
            ("hq_shuf", "another scene's prediction", "#B279A2")]):
        ax.bar(x + (i - 1.5) * w, t[k], w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in t.seed])
    ax.set_ylim(min(t.hq_zero) - .002, max(t.hq_learned) + .002)
    ax.set_ylabel("raw HQNR (FR paper mat20)")
    ax.legend(fontsize=7.6, loc="upper left", framealpha=.92)
    ax.set_title("A5a  lambda* = 1e-4 teacher, same checkpoint,\n"
                 "only the applied correction changes", fontsize=10.5)

    ax = fig.add_subplot(gs[1])
    lab = ["learned - zero", "learned - const", "learned - shuffle"]
    for i, k in enumerate(("d_lz", "d_lc", "d_ls")):
        ax.scatter([i] * len(t), t[k], s=70, color="#4C78A8", zorder=3)
    ax.axhline(0, c="k", lw=.9)
    ax.axhspan(-BAND, BAND, color="gray", alpha=.20, zorder=0)
    ax.text(2.4, BAND, f"band +/-{BAND}", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("HQNR difference")
    ax.set_title("A5b  the shift matters; WHICH shift does not.\n"
                 "One constant reproduces the whole gain", fontsize=10.5)

    fig.suptitle("A5  The missing control: does the per-scene prediction beat a single "
                 "global constant?", fontsize=12, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .87])
    p = os.path.join(OUT, "0917_aligner_A5_constant.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = diag_rows()
    fig_epsilon(D)
    fig_predict(proxy_table())
    fig_inverted(D)
    fig_selection()
    fig_constant()


if __name__ == "__main__":
    main()
