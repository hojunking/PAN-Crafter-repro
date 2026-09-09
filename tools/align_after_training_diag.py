#!/usr/bin/env python
"""학습만으로(재구성 loss, jitter 없음) 정렬 문제가 얼마나 해결되는가 — 학습된 모델의 FR 출력을 audit 추정기로 재본다.

  python tools/align_after_training_diag.py --part A --run S1_T05_W168_D123_DUAL            # 저장된 출력(mat)만, 추론 없음
  python tools/align_after_training_diag.py --part B --run S1_T05_W168_D123_DUAL [--eps ...] # 입력 MS 를 통제 shift 한 추론

Part A (장면별, 논문 세트 20장): PAN 격자(HR)에서 audit 추정기(align/estimator.py: Scharr→median/MAD→상위30% edge→ZNCC→quadratic)
  δ_in  = PAN(MTF_PAN blur) ← up(MS)            모델 입력의 어긋남 (audit pair C, PAN 격자)
  δ_out = PAN ← 출력(밴드평균)                    출력이 PAN geometry 를 따르는가 (0 이면 정렬됨)
  δ_oM  = MTF blur(출력) ← up(MS)                 출력의 저주파(분광)가 MS 격자에서 얼마나 옮겨갔는가
  같은 것을 EXP(lms) 에도. 블록(64px) 단위 국소 추정으로 |δ| 중앙값도 낸다. 장면별 D_λ/D_s/HQNR 은 fr_mat20.json, fSCC 는 utils.SCC_full_numpy.
Part B: sr_infer 경로로 조건 MS 를 ε(HR px) 만큼 옮겨 넣고 HQNR·D_s·fSCC·δ_out 을 잰다 — 출력이 PAN 을 따르는지(δ_out≈0) MS 를 따르는지(δ_out≈ε).
결과: outputs/align_after_training/<run>_A.csv / _B.csv
"""
import os, sys, json, argparse, csv
import numpy as np, h5py, yaml, torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from align.estimator import estimate_shift, GATES                      # noqa: E402
from tools.metrics.jqm import _pan_kernel                              # noqa: E402
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE, load_dlpan, d_lambda_k, d_s   # noqa: E402
from utils import SCC_full_numpy                                        # noqa: E402
import tools.eval_fr_paperset as efp                                   # noqa: E402

OUT = os.path.join(ROOT, "outputs", "align_after_training"); os.makedirs(OUT, exist_ok=True)
HR_GATES = dict(GATES, search_int=4, max_magnitude=4.0)                # HR 격자: ±4px 탐색


def blur_hr(img2d, k):
    p = k.shape[0] // 2
    return fftconvolve(np.pad(img2d, p, mode="edge"), k[::-1, ::-1], mode="valid")


def up_bicubic(ms_hwc):
    t = torch.from_numpy(ms_hwc.transpose(2, 0, 1)[None]).float()
    return F.interpolate(t, scale_factor=4, mode="bicubic")[0].numpy().transpose(1, 2, 0)


def shift(ref, mov, gates=HR_GATES):
    r = estimate_shift(ref.astype(np.float32), mov.astype(np.float32), gates)
    return r["dy_lr_raw"], r["dx_lr_raw"], r["magnitude_raw"], r["peak_zncc"], r["accepted"]


def block_shift_median(ref, mov, B=64, gates=dict(GATES, search_int=3, max_magnitude=3.0)):
    H, W = ref.shape; mags = []
    for i in range(0, H - B + 1, B):
        for j in range(0, W - B + 1, B):
            r = estimate_shift(ref[i:i + B, j:j + B].astype(np.float32), mov[i:i + B, j:j + B].astype(np.float32), gates)
            if not r["boundary_hit"] and r["peak_margin"] >= 0.02:
                mags.append(r["magnitude_raw"])
    return float(np.median(mags)) if mags else float("nan"), len(mags)


def fscc(pan_hw, out_hwc, R):
    return float(SCC_full_numpy(pan_hw[..., None] / R, out_hwc / R))


def load_scene_data(sensor):
    h5 = efp.h5_for(sensor)
    with h5py.File(h5) as f:
        ms = np.asarray(f["ms"], dtype=np.float64).transpose(0, 2, 3, 1); lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    return h5, ms, lms, pan


def part_a(run):
    wd = os.path.join(ROOT, "work_dir", run)
    j = json.load(open(os.path.join(wd, "results", "fr_mat20.json")))
    cfgp = os.path.join(wd, "meta", "config.yaml")
    sensor = efp.sensor_of(yaml.safe_load(open(cfgp))) if os.path.exists(cfgp) else (j.get("sensor") or "wv3")
    S = sensor.upper(); R = 1023.0 if sensor == "gf2" else 2047.0
    if j.get("mat"):
        # _sel1219 처럼 옮겨진 run 은 json 의 mat 경로가 옛 위치(지금은 새 run 이 쓰는 폴더)를 가리킨다 — 자기 폴더의 것을 쓴다
        own = os.path.join(wd, "results", os.path.basename(j["mat"]))
        sr = loadmat(own if os.path.exists(own) else os.path.join(ROOT, j["mat"]))["sr"].astype(np.float64).transpose(0, 2, 3, 1)
    else:   # CANConv 배포 가중치 출력 (tools/make_cannet_reference.py 가 평가한 컨테이너 산출물)
        import glob
        with h5py.File(sorted(glob.glob(os.path.join(ROOT, "..", "CANConv", "data", "datasets", sensor, "sr_cannet_mat20*.h5")))[0]) as f:
            sr = np.asarray(f["sr"], dtype=np.float64).transpose(0, 2, 3, 1)
    _, ms, lms, pan = load_scene_data(sensor)
    kp = _pan_kernel(S, 4); kms = genmtf_matlab(GNYQ_TABLE.get(S) or [0.3] * ms.shape[3], 4, 41)
    rows = []
    for i in range(len(sr)):
        up = up_bicubic(ms[i]); pan_b = blur_hr(pan[i], kp)
        out_mean = sr[i].mean(2); lms_mean = lms[i].mean(2); up_mean = up.mean(2)
        out_b = np.mean([blur_hr(sr[i][:, :, b], kms[:, :, b]) for b in range(sr.shape[3])], axis=0)
        d_in = shift(pan_b, up_mean); d_out = shift(pan[i], out_mean); d_exp = shift(pan_b, lms_mean); d_oM = shift(up_mean, out_b)
        d_oP = shift(pan_b, out_b)      # 출력의 저주파(분광 성분)가 PAN geometry 에 있는가 — 0 이면 색이 구조를 따라갔다
        bl_in, _ = block_shift_median(pan_b, up_mean); bl_out, _ = block_shift_median(pan[i], out_mean)
        rows.append(dict(scene=i, in_dy=d_in[0], in_dx=d_in[1], in_mag=d_in[2], in_zncc=d_in[3],
                         out_dy=d_out[0], out_dx=d_out[1], out_mag=d_out[2], out_zncc=d_out[3],
                         exp_dy=d_exp[0], exp_dx=d_exp[1], exp_mag=d_exp[2],
                         oM_dy=d_oM[0], oM_dx=d_oM[1], oM_mag=d_oM[2], oPlow_dy=d_oP[0], oPlow_dx=d_oP[1], oPlow_mag=d_oP[2],
                         blk_in=bl_in, blk_out=bl_out,
                         d_lambda=j.get("per_scene_d_lambda", [float("nan")] * 20)[i], d_s=j.get("per_scene_d_s", [float("nan")] * 20)[i], hqnr=j["per_scene_hqnr"][i],
                         fscc_out=fscc(pan[i], sr[i], R), fscc_exp=fscc(pan[i], lms[i], R)))
        print(f"  [{sensor} {i:2d}] in |δ| {d_in[2]:.2f} ({d_in[0]:+.2f},{d_in[1]:+.2f})  out |δ| {d_out[2]:.2f} ({d_out[0]:+.2f},{d_out[1]:+.2f})  "
              f"exp |δ| {d_exp[2]:.2f}  out←MS |δ| {d_oM[2]:.2f}  outlow←PAN |δ| {d_oP[2]:.2f}  blk in/out {bl_in:.2f}/{bl_out:.2f}  D_s {rows[-1]['d_s']:.4f} HQNR {rows[-1]['hqnr']:.4f} fSCC {rows[-1]['fscc_out']:.3f}/{rows[-1]['fscc_exp']:.3f}", flush=True)
    p = os.path.join(OUT, f"{run}_A.csv")
    with open(p + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.replace(p + ".tmp", p)
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    from scipy.stats import spearmanr
    print(f"[{run} / {sensor}] median |δ_in| {np.median(a['in_mag']):.2f}  |δ_exp| {np.median(a['exp_mag']):.2f}  |δ_out| {np.median(a['out_mag']):.2f}  "
          f"|δ_out←MS| {np.median(a['oM_mag']):.2f}  |δ_outlow←PAN| {np.median(a['oPlow_mag']):.2f}  block in/out {np.nanmedian(a['blk_in']):.2f}/{np.nanmedian(a['blk_out']):.2f} px | "
          f"scenes |δ_out|<|δ_in| {int((a['out_mag'] < a['in_mag']).sum())}/20 | "
          f"Spearman(|δ_in|, D_s) {spearmanr(a['in_mag'], a['d_s']).correlation:+.2f}  (|δ_in|, HQNR) {spearmanr(a['in_mag'], a['hqnr']).correlation:+.2f}  "
          f"(|δ_out|, D_s) {spearmanr(a['out_mag'], a['d_s']).correlation:+.2f} | fSCC out {a['fscc_out'].mean():.4f} vs EXP {a['fscc_exp'].mean():.4f}")
    return p


def part_b(run, eps_list, device, alphas=()):
    from sr.forward import sr_infer
    from train_sr import SRModel
    from main import import_class
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    sensor = efp.sensor_of(cfg); S = sensor.upper(); R = 1023.0 if sensor == "gf2" else 2047.0
    m, fwd, how = efp.build(cfg, wd, "best_hqnr")
    model = m if hasattr(m, "backbone") else SRModel(m, None)                 # plain backbone 도 sr_infer 경로로
    model = model.to(device).eval(); variant = "j1"
    h5, ms_all, lms_all, pan_all = load_scene_data(sensor)
    Feeder = import_class(cfg["feeder"]); fa = dict(cfg["test_full_feeder_args"]); fa["dataroot"] = h5; ds = Feeder(**fa); mp = float(ds.max_pixel)
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    rows = []
    din = None
    if alphas:
        with open(os.path.join(OUT, f"{run}_A.csv")) as f:
            din = [(float(r["in_dy"]), float(r["in_dx"])) for r in csv.DictReader(f)]     # Part A: PAN ← up(MS) 장면별 어긋남 (HR px)
    cases = [("uniform", eps, None) for eps in eps_list] + [("alpha", (float("nan"), float("nan")), a) for a in alphas]
    for kind, eps, alpha in cases:
        dl, dsv, fs, dout = [], [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                if kind == "alpha":
                    ev = (alpha * din[i][0], alpha * din[i][1])
                else:
                    ev = eps
                e = torch.tensor([[ev[0], ev[1]]], dtype=torch.float32, device=device) if ev != (0, 0) else None
                lms, ms, lpan, pan = (t.unsqueeze(0).to(device) for t in ds[i])
                y = sr_infer(model, variant, pan, lpan, ms, eps=e)["y"]
                s = ((y.clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp)[0].transpose(1, 2, 0).astype(np.float64)
                dl.append(d_lambda_k(s, lms_all[i], sensor, 4, 32, wald)); dsv.append(d_s(s, lms_all[i], pan_all[i], 4, 32, wald))
                fs.append(fscc(pan_all[i], s, R)); dout.append(shift(pan_all[i], s.mean(2))[:3])
        dl, dsv = np.array(dl), np.array(dsv); h = (1 - dl) * (1 - dsv); do = np.array(dout)
        rows.append(dict(kind=kind, alpha=alpha, eps_dy=eps[0], eps_dx=eps[1], eps_mag=float(np.hypot(*eps)) if kind == "uniform" else float(np.median([abs(alpha) * np.hypot(*d) for d in din])), hqnr=h.mean(), hqnr_sd=h.std(ddof=1), d_lambda=dl.mean(), d_s=dsv.mean(), fscc=float(np.mean(fs)),
                         out_dy=float(np.median(do[:, 0])), out_dx=float(np.median(do[:, 1])), out_mag=float(np.median(do[:, 2]))))
        print(f"  [{run}] {kind} eps=({eps[0]:+.2f},{eps[1]:+.2f}) alpha={alpha}  HQNR {h.mean():.4f}  D_l {dl.mean():.4f}  D_s {dsv.mean():.4f}  fSCC {np.mean(fs):.4f}  "
              f"δ_out(PAN←out) median ({np.median(do[:,0]):+.2f},{np.median(do[:,1]):+.2f}) |δ| {np.median(do[:,2]):.2f}", flush=True)
    p = os.path.join(OUT, f"{run}_B.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B"], required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--eps", default="0,0;0.5,0.5;-0.5,-0.5;1,1;-1,-1;2,2", help="Part B: 'dy,dx;dy,dx;...' HR px")
    ap.add_argument("--alpha", default="", help="Part B: 'a1;a2' — Part A 의 장면별 δ_in 을 a 배 넣는다 (+1: 조건 MS 를 PAN 에 정렬, -1: 어긋남 2배)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    eps_list = [tuple(float(v) for v in s.split(",")) for s in a.eps.split(";")]
    for run in a.run:
        print(f"=== {run} part {a.part}")
        (part_a(run) if a.part == "A" else part_b(run, eps_list, a.device, [float(v) for v in a.alpha.split(";") if v]))


if __name__ == "__main__":
    main()
