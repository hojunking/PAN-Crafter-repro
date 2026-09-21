"""GF2 aligner 분석 그림 (2026-09-21) — 왜 GF2 에서만 aligner 가 크게 듣는가.

기 실험 산출물 + 이 세션의 실측만 쓴다. 라벨은 ASCII.
산출 → results_log/assets/0921_gf2_aligner.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_log", "assets")

# --- 기 실험 산출물 (results/fr_mat20.json · metrics.csv 에서 읽은 값)
GF2 = [("P0\nno aligner", 0.0, 0.9314, 0.0264, 0.0433, 0.8770, 0.5994),
       ("L000\nlam 0", 0.828, 0.9468, 0.0247, 0.0292, 0.8556, 0.5799),
       ("L1E4\nlam* 1e-4", 0.880, 0.9526, 0.0254, 0.0225, 0.8423, 0.5780),
       ("DONN2\ndonor R200", 1.714, 0.9670, 0.0214, 0.0119, 0.8280, 0.6550)]
WV3 = [("CTRLP0\nno aligner", 0.0, 0.95068, 0.02384, 0.02613, 0.89661, 2.0336),
       ("L000\nlam 0", 0.351, 0.95547, 0.02104, 0.02399, 0.88047, 2.0276),
       ("L1E4\nlam* 1e-4", 0.452, 0.95698, 0.01990, 0.02359, 0.87856, 2.0562),
       ("P1\ndonor frozen", 1.511, 0.93298, 0.02157, 0.04659, 0.81320, 2.0834)]

# --- 이 세션 실측 (같은 추정기 align/estimator.py, RR 세트)
MEAS = {"WV3": dict(proxy=0.590, true=0.628, art=0.677, c=0.138,
                    cos_proxy=+1.000, cos_true=-0.398, gt_bef=0.628, gt_aft=0.633, gt_imp=15),
        "GF2": dict(proxy=0.946, true=0.515, art=0.702, c=0.510,
                    cos_proxy=+0.996, cos_true=+0.812, gt_bef=0.515, gt_aft=0.440, gt_imp=11)}
CG, CW = "#E45756", "#4C78A8"


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(15.5, 8.8))
    gs = GridSpec(2, 3, hspace=.42, wspace=.30)

    # (a) HQNR ladder
    ax = fig.add_subplot(gs[0, 0])
    for D, c, lab in ((GF2, CG, "GF2"), (WV3, CW, "WV3")):
        ax.plot([d[1] for d in D], [d[2] for d in D], "o-", color=c, label=lab)
        for d in D:
            ax.annotate(d[0].split("\n")[0], (d[1], d[2]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points", color=c)
    ax.set_xlabel(r"aligner applied $|c|$ at FR (px)")
    ax.set_ylabel("raw HQNR (FR paper mat20)")
    ax.legend(fontsize=9)
    ax.set_title("(a) the aligner buys 5x more in GF2\n"
                 "GF2 +0.021 (P0->L1E4)  ·  WV3 +0.006", fontsize=10.5)

    # (b) D_s
    ax = fig.add_subplot(gs[0, 1])
    for D, c, lab in ((GF2, CG, "GF2"), (WV3, CW, "WV3")):
        ax.plot([d[1] for d in D], [d[4] for d in D], "o-", color=c, label=lab)
    ax.set_xlabel(r"$|c|$ at FR (px)"); ax.set_ylabel(r"$D_s$  (lower better)")
    ax.legend(fontsize=9)
    ax.set_title("(b) in GF2 the aligner removes the\n"
                 "recipe's PAN over-correlation", fontsize=10.5)

    # (c) RR ERGAS (GT 가 있는 지표) — 센서마다 스케일이 달라 상대값
    ax = fig.add_subplot(gs[0, 2])
    for D, c, lab in ((GF2, CG, "GF2"), (WV3, CW, "WV3")):
        base = D[0][6]
        ax.plot([d[1] for d in D], [d[6] / base * 100 for d in D], "o-", color=c,
                label=f"{lab} (base {base:.3f})")
    ax.axhline(100, c="k", lw=.8, ls=":")
    ax.set_xlabel(r"$|c|$ at FR (px)"); ax.set_ylabel("RR ERGAS, % of no-aligner")
    ax.legend(fontsize=8.5)
    ax.set_title("(c) GT-referenced metric: GF2 improves,\nWV3 does not", fontsize=10.5)

    # (d) 무엇을 보고 무엇이 참인가
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(2); w = .26
    for i, (k, lab, c) in enumerate([("proxy", "PAN vs bicubic-up MS\n(what the aligner sees)", "#9D9D9D"),
                                     ("true", "PAN vs GT\n(what reconstruction wants)", "#54A24B"),
                                     ("c", "aligner applies", "#B279A2")]):
        ax.bar(x + (i - 1) * w, [MEAS["WV3"][k], MEAS["GF2"][k]], w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels(["WV3", "GF2"])
    ax.set_ylabel("HR px (RR set)"); ax.legend(fontsize=7.6)
    ax.set_title("(d) GF2: applied 0.51 ~ true 0.52\nWV3: applied 0.14 << true 0.63", fontsize=10.5)

    # (e) 방향 — 결정적
    ax = fig.add_subplot(gs[1, 1])
    for i, s in enumerate(("WV3", "GF2")):
        ax.bar(i - .18, MEAS[s]["cos_proxy"], .34, color="#9D9D9D",
               label="vs what it sees (proxy)" if i == 0 else None)
        ax.bar(i + .18, MEAS[s]["cos_true"], .34, color="#54A24B",
               label="vs the truth (PAN vs GT)" if i == 0 else None)
        ax.text(i + .18, MEAS[s]["cos_true"] + (.05 if MEAS[s]["cos_true"] > 0 else -.12),
                f"{MEAS[s]['cos_true']:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, c="k", lw=.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WV3", "GF2"])
    ax.set_ylim(-.7, 1.2); ax.set_ylabel("cosine of direction")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(e) THE mechanism: in GF2 the proxy and the\n"
                 "truth agree; in WV3 they conflict", fontsize=10.5)

    # (f) 실제 오정합이 줄어드는가
    ax = fig.add_subplot(gs[1, 2])
    for i, s in enumerate(("WV3", "GF2")):
        ax.bar(i - .18, MEAS[s]["gt_bef"], .34, color="#9D9D9D",
               label="before" if i == 0 else None)
        ax.bar(i + .18, MEAS[s]["gt_aft"], .34, color="#54A24B",
               label="after aligner" if i == 0 else None)
        d = MEAS[s]["gt_aft"] - MEAS[s]["gt_bef"]
        ax.text(i, max(MEAS[s]["gt_bef"], MEAS[s]["gt_aft"]) + .02,
                f"{d:+.3f} px\n{MEAS[s]['gt_imp']}/20 scenes", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WV3", "GF2"])
    ax.set_ylabel("PAN <-> GT misalignment (px)")
    ax.set_ylim(0, .78); ax.legend(fontsize=8.5)
    ax.set_title("(f) only in GF2 does the correction actually\n"
                 "reduce the real misalignment", fontsize=10.5)

    fig.suptitle("Why the PAN aligner works in GF2 but not in WV3 — "
                 "same estimator, RR set (GT available)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .945])
    p = os.path.join(OUT, "0921_gf2_aligner.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
