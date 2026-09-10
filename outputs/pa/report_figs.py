"""A1–A3 (s1, seed 2025) 결과 그림 — results_log/assets/0910_pa_{dynamics,diag}.png"""
import os, sys, json, csv, numpy as np, torch, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"; sys.path.insert(0, ROOT)
RUNS = {"A1 (L_rec)": "PA_A1_REC_W96_D124_9CH_S2025", "A2 (+edge)": "PA_A2_OUTEDGE_W96_D124_9CH_S2025", "A3 (+geo)": "PA_A3_GEO_W96_D124_9CH_S2025"}
B0 = "BASE_W96_D124_MSPAN_WV3_S2025"
COL = {"A1 (L_rec)": "C0", "A2 (+edge)": "C1", "A3 (+geo)": "C2", "B0": "k"}
# ---------- per-e shift response (재계산: 모델 forward)
from main import import_class
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.warp import warp_pan
from safetensors.torch import load_file
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resp = {}
grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
for lab, r in RUNS.items():
    wd = f"{ROOT}/work_dir/{r}"; cfg = yaml.safe_load(open(f"{wd}/meta/config.yaml"))
    m = PAModel(import_class(cfg["model"])(**cfg["model_args"]), PANGlobalAligner(8)); m.load_state_dict(load_file(f"{wd}/best_hqnr/model.safetensors")); m = m.to(dev).eval()
    ds = import_class(cfg["feeder"])(**cfg["test_full_feeder_args"])
    ry, rx = {e: [] for e in grid}, {e: [] for e in grid}
    with torch.no_grad():
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); d0 = m(pan, ms, lpan)["delta"][0].cpu().numpy()
            for e in grid:
                dy = m(warp_pan(pan, torch.tensor([[e, 0.0]], device=dev)).to(pan.dtype), ms, lpan)["delta"][0].cpu().numpy(); ry[e].append(dy[0] - d0[0])
                dx = m(warp_pan(pan, torch.tensor([[0.0, e]], device=dev)).to(pan.dtype), ms, lpan)["delta"][0].cpu().numpy(); rx[e].append(dx[1] - d0[1])
    resp[lab] = dict(e=grid, dy=[float(np.mean(ry[e])) for e in grid], dx=[float(np.mean(rx[e])) for e in grid])
json.dump(resp, open(f"{ROOT}/outputs/pa/shift_response_curves.json", "w"), indent=1)
# ---------- data
ck = {lab: pd.read_csv(f"{ROOT}/work_dir/{r}/checkpoint_metrics.csv") for lab, r in RUNS.items()}
b0 = pd.read_csv(f"{ROOT}/work_dir/{B0}/metrics.csv")
diag = {lab: json.load(open(f"{ROOT}/work_dir/{r}/results/pa_diag.json")) for lab, r in RUNS.items()}
dp = {lab: pd.read_csv(f"{ROOT}/work_dir/{r}/delta_predictions.csv") for lab, r in RUNS.items()}
best_step = {lab: json.load(open(f"{ROOT}/work_dir/{r}/best_hqnr_meta.json"))["step"] for lab, r in RUNS.items()}
# ---------- Fig 1: dynamics
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for lab, d in ck.items():
    ax[0].plot(d.epoch, d.dy_median, color=COL[lab], label=f"{lab} dy"); ax[0].plot(d.epoch, d.dx_median, color=COL[lab], ls="--", label=f"{lab} dx")
ax[0].axhline(0.34, color="gray", lw=0.8, ls=":"); ax[0].text(5, 0.345, "train-scale GT<-PAN dy (+0.34)", fontsize=7, color="gray")
ax[0].axhline(-0.07, color="gray", lw=0.8, ls=":"); ax[0].text(5, -0.09, "dx (-0.07)", fontsize=7, color="gray")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("median delta_hat on FR scenes (HR px)"); ax[0].set_title("(a) predicted PAN shift during training"); ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
ax[1].plot(b0.epoch, b0.hqnr_official, color="k", label="B0 (no aligner)")
for lab, d in ck.items():
    ax[1].plot(d.epoch, d["raw_original.hqnr"], color=COL[lab], label=lab)
ax[1].set_ylim(0.93, 0.956); ax[1].set_xlabel("epoch"); ax[1].set_title("(b) HQNR raw_original (paper set 20, full frame)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ax[2].plot(b0.epoch, b0.fscc_official, color="k", label="B0")
for lab, d in ck.items():
    ax[2].plot(d.epoch, d["raw_original.fscc"], color=COL[lab], label=lab)
ax[2].set_ylim(0.88, 0.94); ax[2].set_xlabel("epoch"); ax[2].set_title("(c) fSCC vs original PAN"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.suptitle("A1-A3 vs B0, s1 seed 2025 (B0 3-seed HQNR 0.9487-0.9516)"); fig.tight_layout(); fig.savefig(f"{ROOT}/results_log/assets/0910_pa_dynamics.png", dpi=130)
# ---------- Fig 2: diagnostics at best_hqnr
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for lab, d in dp.items():
    s = d[d.step == best_step[lab]]; ax[0].scatter(s.dx_hr, s.dy_hr, s=18, color=COL[lab], label=f"{lab} (20 scenes)")
ax[0].scatter([-0.07], [0.34], marker="*", s=140, color="gray", label="train-scale target (+0.34,-0.07)")
ax[0].annotate("", xy=(-0.62, 1.64), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=1)); ax[0].text(-0.6, 1.5, "FR-scale target\n(+1.64,-0.62)", fontsize=7, color="red")
ax[0].scatter([0], [0], marker="+", s=80, color="k"); ax[0].set_xlim(-0.8, 0.3); ax[0].set_ylim(-0.2, 1.8); ax[0].set_xlabel("dx (HR px)"); ax[0].set_ylabel("dy (HR px)")
ax[0].set_title("(a) per-scene delta_hat at best_hqnr"); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3); ax[0].set_aspect("equal")
for lab, r in resp.items():
    ax[1].plot(r["e"], r["dy"], "o-", color=COL[lab], label=f"{lab} dy"); ax[1].plot(r["e"], r["dx"], "s--", color=COL[lab], label=f"{lab} dx")
ax[1].plot(grid, [-e for e in grid], color="red", lw=1, label="ideal (-e)"); ax[1].set_xlabel("added PAN shift e (HR px)"); ax[1].set_ylabel("delta_hat(W(P,e)) - delta_hat(P)")
ax[1].set_title("(b) response to a known input shift"); ax[1].legend(fontsize=6, ncol=2); ax[1].grid(alpha=.3)
modes = ["zero", "learned", "wrong_sign"]; w = 0.25
for k, (lab, j) in enumerate(diag.items()):
    h = [j["controls"][m]["views"]["raw_original"]["hqnr"] for m in modes]; f = [j["controls"][m]["views"]["raw_original"]["fscc"] for m in modes]
    ax[2].bar(np.arange(3) + (k - 1) * w, h, w, color=COL[lab], label=lab)
    for i, (hv, fv) in enumerate(zip(h, f)):
        ax[2].text(i + (k - 1) * w, hv + 0.0003, f"{fv:.3f}", fontsize=6, ha="center", rotation=90)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["delta=0", "learned delta", "-delta (wrong sign)"]); ax[2].set_ylim(0.944, 0.956); ax[2].set_ylabel("HQNR raw_original (labels: fSCC)")
ax[2].set_title("(c) same weights, three PAN inputs"); ax[2].legend(fontsize=7); ax[2].grid(axis="y", alpha=.3)
data, labels = [], []
for lab, j in diag.items():
    st = j["independent_structure"]["per_scene"]; data += [[r["orig_mag"] for r in st], [r["aligned_mag"] for r in st]]; labels += [f"{lab.split()[0]}\noriginal P", f"{lab.split()[0]}\nwarped P~"]
ax[3].boxplot(data, labels=labels); ax[3].set_ylabel("|shift| PAN_blur <- up(MS), audit estimator (HR px)"); ax[3].set_title("(d) is the warped PAN closer to the MS grid?"); ax[3].grid(axis="y", alpha=.3); ax[3].tick_params(axis="x", labelsize=7)
fig.suptitle("Diagnostics at best_hqnr (s1 seed 2025). Sampling convention: P~[y,x] = P[y+dy, x+dx]"); fig.tight_layout(); fig.savefig(f"{ROOT}/results_log/assets/0910_pa_diag.png", dpi=130)
print(json.dumps(resp, indent=0)[:600])
