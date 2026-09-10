#!/usr/bin/env python
"""results_log/2026-09-10_pa-a1-a3-s1-results.md §10 그림: PO10 R100/R200 학습 곡선 + 반응·stress. python outputs/po10/report_figs.py [--diag po10_diag|po10_diag_last]"""
import argparse, csv, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--diag", default="po10_diag"); ap.add_argument("--suffix", default=""); a = ap.parse_args()
runs = [("A1 W96 (no corruption)", "PA_A1_REC_W96_D124_9CH_S2025", "gray"), ("N1 R100 W96 D124", "PO10_N1_REC_W96_D124_WV3_S2025", "tab:blue"),
        ("N1 R200 W112 D123", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT", "tab:orange"), ("N2 SG R200 W112 D123", "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT", "tab:green"), ("N3 noSG R200 W112 D123", "PO10_N3_OFFNOSG_W112_D123_WV3_S2025_R200_FRSTAT", "tab:red")]
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for name, T, col in runs:
    rows = list(csv.DictReader(open(f"{ROOT}/work_dir/{T}/checkpoint_metrics.csv"))); st = np.array([int(r["step"]) for r in rows]) / 1000
    g = lambda k: np.array([float(r[k]) for r in rows])
    ax[0, 0].plot(st, g("raw_original.hqnr"), color=col, label=name); ax[0, 1].plot(st, g("raw_original.d_s"), color=col); ax[0, 2].plot(st, g("raw_original.d_lambda"), color=col)
    ax[1, 0].plot(st, g("aligned_valid.hqnr"), color=col); ax[1, 1].plot(st, g("dy_median"), color=col); ax[1, 1].plot(st, g("dx_median"), color=col, ls="--"); ax[1, 2].plot(st, g("rr_ergas"), color=col)
for a_, t in zip(ax.flat, ["HQNR raw_original (paper protocol, full frame)", "D_s raw_original", "D_lambda raw_original", "HQNR aligned_valid (warped-PAN reference, V64)", "FR delta median: dy (solid), dx (dashed) [HR px]", "RR ERGAS (train log def.)"]):
    a_.set_title(t, fontsize=10); a_.set_xlabel("update (K)"); a_.grid(alpha=.3)
ax[0, 0].legend(fontsize=8); ax[1, 2].set_ylim(2.0, 3.0); plt.tight_layout(); plt.savefig(f"{ROOT}/outputs/po10/0910_po10_r200_curves{a.suffix}.png", dpi=110)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); names = ["N1 R200", "N2 SG", "N3 noSG"]; Ts = [r[1] for r in runs[2:]]
for scale, mk in (("native64", "o"), ("rr256", "s"), ("fr512", "^")):
    dys, dxs = [], []
    for T in Ts:
        r = json.load(open(f"{ROOT}/work_dir/{T}/results/{a.diag}.json"))["response"][scale]; dys.append(r["B_diag"][0]); dxs.append(r["B_diag"][1])
    ax[0].plot(names, dys, marker=mk, ls="-", label=f"dy slope ({scale})"); ax[0].plot(names, dxs, marker=mk, ls="--", label=f"dx slope ({scale})")
ax[0].axhline(-1, color="k", lw=.8, ls=":"); ax[0].axhline(-0.05, color="gray", lw=.8, ls=":"); ax[0].set_title(f"Response slope B_diag ({a.diag}; ideal -1; A1 donor ~ -0.05)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
for T, name, col in zip(Ts, names, ["tab:orange", "tab:green", "tab:red"]):
    j = json.load(open(f"{ROOT}/work_dir/{T}/results/{a.diag}.json")); be = j["stress_hqnr_fr512"]["by_eps"]; ks = list(be.keys())
    ax[1].plot(ks, [be[k]["raw_valid_hqnr"] for k in ks], marker="o", color=col, label=f"{name} raw_valid"); ax[1].plot(ks, [be[k]["aligned_valid_hqnr"] for k in ks], marker="s", ls="--", color=col, label=f"{name} aligned_valid")
    cb = j["response"]["native64"]["closure_by_eps_bin"]; bins = list(cb.keys()); ax[2].plot(bins, [cb[b]["closure_mean"] for b in bins], marker="o", color=col, label=name)
ax[1].set_title("FR512 stress: known extra shift eps (dy,dx), ROI margin 96"); ax[1].tick_params(axis="x", labelsize=7); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
ax[2].plot(bins, [np.mean([float(x) for x in b.split("-")]) for b in bins], color="k", ls=":", label="no response (closure = |eps|)"); ax[2].set_title("native64 closure |c_eps + eps - c0| by |eps| bin"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{ROOT}/outputs/po10/0910_po10_r200_response{a.suffix}.png", dpi=110); print("saved")
