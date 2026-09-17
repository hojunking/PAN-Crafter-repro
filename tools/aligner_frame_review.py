"""평가 프레임 가설 검토 — "HQNR 이 나쁜 건 원본(misaligned) PAN 으로 재기 때문 아닌가".

A6a  같은 가중치를 두 기준 프레임에서 평가하면 순위가 뒤집힌다 (가설 지지)
A6b  그런데 GT 가 있는 RR 에서는 정합이 강할수록 나빠진다 (가설 반증)
A6c  RR 에서도 aligner 는 제대로 정합한다 — 실제 0.59 px 중 0.40 px 적용

산출 → results_log/assets/0917_aligner_A6_frame.png
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_log", "assets")

# 정합 강도 사다리 (seed 1234), |c| 는 fr512 native 예측
LADDER = [("P0", "no aligner", 0.00), ("P2", "lambda 0", 0.34), ("L3E5", "lambda 3e-5", 0.38),
          ("L1E4", "lambda* 1e-4", 0.54), ("L3E4", "lambda 3e-4", 0.51),
          ("L3E3", "lambda 3e-3", 0.82), ("L1E3", "lambda 1e-3", 1.00),
          ("P3", "lambda 1e-2", 1.13), ("P1", "donor frozen", 1.51)]
TRUE_FR, TRUE_RR = 1.788, 0.590
DONOR_FR, DONOR_RR = 1.51, 0.40


def load_views():
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/work_dir/*")):
        f = os.path.join(d, "checkpoint_metrics.csv")
        if not os.path.exists(f):
            continue
        run = os.path.basename(d)
        m = re.search(r"(PALS24|PALSV18|NF16)_([A-Z0-9]+)_", run)
        if not m or "_S1234_" not in run:
            continue
        t = pd.read_csv(f)
        if "aligned_fixed_v64.hqnr" not in t:
            continue
        i = t["raw_original.hqnr"].idxmax()
        out[m.group(2)] = dict(raw=t["raw_original.hqnr"][i],
                               fix=t["aligned_fixed_v64.hqnr"][i])
    return out


def load_rr():
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/work_dir/*")):
        f = os.path.join(d, "metrics.csv")
        if not os.path.exists(f):
            continue
        run = os.path.basename(d)
        m = re.search(r"(PALS24|PALSV18|NF16)_([A-Z0-9]+)_", run)
        if not m or "_S1234_" not in run:
            continue
        t = pd.read_csv(f)
        t = t[t.global_step >= t.global_step.max() * .8]
        if t.empty or "ergas" not in t:
            continue
        out[m.group(2)] = dict(ergas=t.ergas.mean(), scc=t.scc.mean())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    V, R = load_views(), load_rr()

    fig = plt.figure(figsize=(15.0, 5.2))
    gs = GridSpec(1, 3, wspace=.33)

    # (a) frame flip
    ax = fig.add_subplot(gs[0])
    lab = [l for k, l, c in LADDER if k in V]
    cs = [c for k, l, c in LADDER if k in V]
    raw = [V[k]["raw"] for k, l, c in LADDER if k in V]
    fix = [V[k]["fix"] for k, l, c in LADDER if k in V]
    ax.plot(cs, raw, "o-", color="#E45756", label="reference = ORIGINAL PAN (raw_original)")
    ax.plot(cs, fix, "s-", color="#4C78A8",
            label="reference = COMMON aligned frame (aligned_fixed_v64)")
    for x, y, l in zip(cs, fix, lab):
        ax.annotate(l, (x, y), fontsize=6.4, xytext=(3, 4), textcoords="offset points")
    ax.axvline(TRUE_FR, c="gray", ls="--", lw=1.1)
    ax.text(TRUE_FR - .05, min(raw), "true 1.79 px", fontsize=7.4, rotation=90,
            ha="right", va="bottom", color="gray")
    ax.set_xlabel(r"alignment applied  $|c|$ at FR (HR px)")
    ax.set_ylabel("HQNR")
    ax.legend(fontsize=7.4, loc="lower center")
    ax.set_title("A6a  FR: the ordering REVERSES with the frame\n"
                 "-> the user's diagnosis is correct", fontsize=10)

    # (b) RR falsification
    ax = fig.add_subplot(gs[1])
    lab2 = [l for k, l, c in LADDER if k in R]
    cs2 = [c for k, l, c in LADDER if k in R]
    er = [R[k]["ergas"] for k, l, c in LADDER if k in R]
    sc = [R[k]["scc"] for k, l, c in LADDER if k in R]
    ax.plot(cs2, er, "o-", color="#E45756", label="RR ERGAS (lower better)")
    for x, y, l in zip(cs2, er, lab2):
        ax.annotate(l, (x, y), fontsize=6.4, xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel(r"alignment applied  $|c|$ at FR (HR px)")
    ax.set_ylabel("RR ERGAS (test, plateau mean)", color="#E45756")
    ax2 = ax.twinx()
    ax2.plot(cs2, sc, "^--", color="#4C78A8")
    ax2.set_ylabel("RR SCC (higher better)", color="#4C78A8")
    ax.set_title("A6b  RR: GT is in the MS frame, no frame excuse --\n"
                 "yet MORE alignment is WORSE", fontsize=10)

    # (c) is the aligner right at RR too?
    ax = fig.add_subplot(gs[2])
    x = np.arange(2); w = .34
    ax.bar(x - w / 2, [TRUE_FR, TRUE_RR], w, label="true misalignment (same estimator)",
           color="#9D9D9D")
    ax.bar(x + w / 2, [DONOR_FR, DONOR_RR], w, label="donor N2 applies", color="#4C78A8")
    for i, (a, b) in enumerate([(TRUE_FR, DONOR_FR), (TRUE_RR, DONOR_RR)]):
        ax.text(i + w / 2, b + .04, f"{b/a*100:.0f}%", ha="center", fontsize=9,
                color="#4C78A8")
        ax.text(i - w / 2, a + .04, f"{a:.2f}", ha="center", fontsize=8.5, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(["FR (pan 512, ms 128)", "RR (pan 256, ms 64)"])
    ax.set_ylabel("HR px"); ax.legend(fontsize=8)
    ax.set_title("A6c  the aligner scales correctly -- at RR it\n"
                 "removes 68% of a real 0.59 px offset", fontsize=10)

    fig.suptitle("A6  Is the poor HQNR just an evaluation-frame artefact?  "
                 "Partly yes at FR, but RR says alignment itself does not help",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .9])
    p = os.path.join(OUT, "0917_aligner_A6_frame.png")
    fig.savefig(p, dpi=115); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
