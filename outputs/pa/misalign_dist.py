"""데이터셋 안의 MS–PAN 어긋남 분포 — 센서(WV3·QB·GF2·WV2) × split(train patch · RR test · FR 논문 세트).
audit 추정기(align/estimator.py, Scharr-ZNCC-quadratic)로 잰다. 단위는 전부 PAN(HR) 격자 px, 부호는 aligned[y,x]=moving[y+dy,x+dx].
  FR : PAN_b(PAN MTF blur) ← bicubic↑MS            (모델 입력의 어긋남, 512²)
  RR : GT(band mean) ← PAN                          (학습 규모, 256²; GT 는 원 MS 이므로 LRMS 와 정합)
  train : GT(band mean) ← PAN, 64² patch 무작위 N 장  + 노이즈 하한(같은 GT 의 짝수/홀수 band 평균 쌍, 참값 0)
산출 outputs/pa/misalign_dist.json · results_log/assets/0910_misalign_dist.png
"""
import os, sys, json, numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"; sys.path.insert(0, ROOT)
from align.estimator import estimate_shift, GATES
from tools.align_after_training_diag import blur_hr, up_bicubic
from tools.metrics.jqm import _pan_kernel
rng = np.random.default_rng(0)
G_HR = dict(GATES, search_int=4, max_magnitude=4.0); G_P = dict(GATES, search_int=3, max_magnitude=3.0)
D = "data/PanCollection"
SENS = {"WV3": dict(train="WV3/train_wv3.h5", rr="WV3/reduced_examples_h5/test_wv3_multiExm1.h5", fr="WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5"),
        "QB": dict(train="QB/train_qb_msfix.h5", rr="QB/reduced_examples_h5/test_qb_multiExm1.h5", fr="QB/full_examples_mat20/test_qb_OrigScale_mat20.h5"),
        "GF2": dict(train="GF2/train_gf2.h5", rr="GF2/reduced_examples_h5/test_gf2_multiExm1.h5", fr="GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5"),
        "WV2": dict(train=None, rr="WV2/reduced_examples_h5/test_wv2_multiExm1.h5", fr="WV2/full_examples_mat20/test_wv2_OrigScale_mat20.h5")}
N_TRAIN = 400

def est(ref, mov, g):
    r = estimate_shift(ref.astype(np.float32), mov.astype(np.float32), g); return [r["dy_lr_raw"], r["dx_lr_raw"], r["magnitude_raw"]]

def stats(a):
    a = np.asarray(a); dy, dx, m = a[:, 0], a[:, 1], a[:, 2]
    def q(x, p): return float(np.percentile(x, p))
    return dict(n=int(len(a)), dy_median=float(np.median(dy)), dx_median=float(np.median(dx)), mag_median=float(np.median(m)), mag_mean=float(m.mean()), mag_sd=float(m.std(ddof=1)),
                mag_iqr=[q(m, 25), q(m, 75)], mag_p5_p95=[q(m, 5), q(m, 95)], mag_min_max=[float(m.min()), float(m.max())],
                dy_iqr=[q(dy, 25), q(dy, 75)], dx_iqr=[q(dx, 25), q(dx, 75)],
                sign_agree_dy=float(np.mean(np.sign(dy) == np.sign(np.median(dy)))) if abs(np.median(dy)) > 1e-6 else None,
                sign_agree_dx=float(np.mean(np.sign(dx) == np.sign(np.median(dx)))) if abs(np.median(dx)) > 1e-6 else None)

out, raw = {}, {}
for s, p in SENS.items():
    out[s] = {}; raw[s] = {}
    with h5py.File(f"{D}/{p['fr']}") as f:
        ms = np.asarray(f["ms"], dtype=np.float64).transpose(0, 2, 3, 1); pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    kp = _pan_kernel(s, 4)
    raw[s]["fr"] = [est(blur_hr(pan[i], kp), up_bicubic(ms[i]).mean(2), G_HR) for i in range(len(pan))]
    with h5py.File(f"{D}/{p['rr']}") as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1); prr = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    raw[s]["rr"] = [est(gt[i].mean(2), prr[i], G_HR) for i in range(len(gt))]
    if p["train"]:
        with h5py.File(f"{D}/{p['train']}") as f:
            n = f["gt"].shape[0]; idx = np.sort(rng.choice(n, N_TRAIN, replace=False))
            gtt = np.asarray(f["gt"][idx], dtype=np.float64); ptt = np.asarray(f["pan"][idx], dtype=np.float64)[:, 0]
        raw[s]["train"] = [est(gtt[i].mean(0), ptt[i], G_P) for i in range(N_TRAIN)]
        raw[s]["train_floor"] = [est(gtt[i][0::2].mean(0), gtt[i][1::2].mean(0), G_P) for i in range(N_TRAIN)]     # 같은 기하, 스펙트럼만 다름 → 참값 0
    for k, v in raw[s].items():
        out[s][k] = stats(v)
    print(s, {k: f"|d| med {v['mag_median']:.2f} IQR [{v['mag_iqr'][0]:.2f},{v['mag_iqr'][1]:.2f}] (dy,dx) med ({v['dy_median']:+.2f},{v['dx_median']:+.2f}) n={v['n']}" for k, v in out[s].items()})
json.dump(dict(stats=out, raw={s: {k: np.asarray(v).tolist() for k, v in r.items()} for s, r in raw.items()}), open(f"{ROOT}/outputs/pa/misalign_dist.json", "w"), indent=1)

# ---------- figure: rows = sensors, cols = FR scatter | RR scatter | train patch scatter (+floor)
fig, axs = plt.subplots(4, 3, figsize=(15, 16))
for r, s in enumerate(SENS):
    for c, (k, title) in enumerate([("fr", "FR paper set (20 scenes): PAN_blur <- up(MS)"), ("rr", "RR test (20 scenes): GT <- PAN"), ("train", f"train patches 64x64 (n={N_TRAIN}): GT <- PAN")]):
        ax = axs[r, c]
        if k not in raw[s]:
            ax.text(0.5, 0.5, "no training set (zero-shot sensor)", ha="center", va="center", transform=ax.transAxes); ax.set_title(f"{s}: {title}", fontsize=9); ax.set_axis_off(); continue
        a = np.asarray(raw[s][k]); st = out[s][k]
        if k == "train":
            fl = np.asarray(raw[s]["train_floor"])
            ax.scatter(fl[:, 1], fl[:, 0], s=6, color="lightgray", label=f"noise floor (true 0): |d| med {out[s]['train_floor']['mag_median']:.2f}")
        ax.scatter(a[:, 1], a[:, 0], s=(18 if k != "train" else 6), color=f"C{r}", label=f"|d| median {st['mag_median']:.2f} px, IQR [{st['mag_iqr'][0]:.2f},{st['mag_iqr'][1]:.2f}]")
        ax.scatter([st["dx_median"]], [st["dy_median"]], marker="x", s=80, color="k", label=f"median ({st['dy_median']:+.2f},{st['dx_median']:+.2f})")
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        lim = max(0.6, float(np.abs(a[:, :2]).max()) * 1.15); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("dx (HR px)"); ax.set_ylabel("dy (HR px)"); ax.set_title(f"{s}: {title}", fontsize=9); ax.legend(fontsize=6, loc="lower left"); ax.grid(alpha=.3)
fig.suptitle("Within-dataset MS-PAN misalignment (audit estimator; sign: aligned[y,x] = moving[y+dy, x+dx]; HR px)"); fig.tight_layout()
fig.savefig(f"{ROOT}/results_log/assets/0910_misalign_dist.png", dpi=110)
# summary bars
fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(4); w = 0.27
for j, (k, lab) in enumerate([("train", "train patch (GT<-PAN)"), ("rr", "RR test (GT<-PAN)"), ("fr", "FR paper set (PAN_b<-up(MS))")]):
    med = [out[s][k]["mag_median"] if k in out[s] else np.nan for s in SENS]; lo = [out[s][k]["mag_iqr"][0] if k in out[s] else np.nan for s in SENS]; hi = [out[s][k]["mag_iqr"][1] if k in out[s] else np.nan for s in SENS]
    ax.bar(x + (j - 1) * w, med, w, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], capsize=3, label=lab)
ax.set_xticks(x); ax.set_xticklabels(list(SENS)); ax.set_ylabel("|shift| median, IQR (HR px)"); ax.set_title("Misalignment magnitude by sensor and split"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{ROOT}/results_log/assets/0910_misalign_summary.png", dpi=130)
