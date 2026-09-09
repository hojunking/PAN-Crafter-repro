#!/usr/bin/env python
"""A1–A3 run 의 사후 진단·교차 평가 (명세 §10.10 · §11). 학습이 끝난 run 에 대해 한 번 돈다 (tools/_upload.sh 가 PA_ run 에 대해 부른다).

    PANCRAFTER_DLPAN=... python tools/pa_diag.py --run PA_A1_REC_W96_D124_9CH_S2025 [--ckpt best_hqnr]

산출 work_dir/<run>/results/pa_diag.json + stdout 표:
  1. §10.10 교차 평가: best_raw(=best_hqnr) · best_aligned · last 각각의 raw_original / raw_valid / aligned_valid HQNR·fSCC (학습 중 기록 checkpoint_metrics.csv 에서)
  2. §11.1 aligner 사용 대조 (best_hqnr, 같은 평가 조건 raw_original·raw_valid): learned Δ̂ / Δ=0 / −Δ̂ 세 추론 (서로 다른 SR)
  3. §11.2 추가 shift 반응: Aφ(W(P,e),M) − Aφ(P,M) ≈ −e, e ∈ {−1,−0.5,0,0.5,1}² HR px (20장 평균 오차·축별 기울기)
  4. §11.4 full-scene Δ̂ vs 128px tile Δ̂ 분포
이 값들은 진단이며 best 선택·시트 값을 바꾸지 않는다.
"""
import argparse, csv, json, os, sys
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                           # noqa: E402
from pa.aligner import PANGlobalAligner                                 # noqa: E402
from pa.model import PAModel                                            # noqa: E402
from pa.warp import warp_pan                                            # noqa: E402
from pa.evalviews import scene_views                                    # noqa: E402
from tools.metrics.eval_fr import load_dlpan                            # noqa: E402
import tools.eval_fr_paperset as efp                                    # noqa: E402

VIEWS = ("raw_original", "raw_valid", "aligned_valid")


def load_run(run, ckpt, dev):
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    from safetensors.torch import load_file
    bb = import_class(cfg["model"])(**cfg["model_args"]); m = PAModel(bb, PANGlobalAligner(int(cfg["num_bands"])))
    m.load_state_dict(load_file(os.path.join(wd, ckpt, "model.safetensors")), strict=True)
    Feeder = import_class(cfg["feeder"]); fa = dict(cfg["test_full_feeder_args"]); ds = Feeder(**fa)
    return wd, cfg, m.to(dev).eval(), ds


def dn(t, mp):
    return ((t.clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64)


@torch.no_grad()
def views_for(m, ds, sensor, wald, mp, dev, mode):
    """mode: learned | zero | wrong_sign. 반환 view 별 평균 dict + Δ 배열."""
    acc = {v: dict(hqnr=[], fscc=[], d_s=[], d_lambda=[]) for v in VIEWS}; D = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        o = m(pan, ms, lpan)
        if mode != "learned":
            d = torch.zeros_like(o["delta"]) if mode == "zero" else -o["delta"]
            o = m(pan, ms, lpan, delta_override=d)
        sr = dn(o["y"][0], mp).transpose(1, 2, 0); lm = dn(lms[0], mp).transpose(1, 2, 0); p = dn(pan[0, 0], mp); pa_ = dn(o["pan_aligned"][0, 0], mp)
        d = o["delta"][0].float().cpu().numpy(); D.append(d)
        v, _ = scene_views(sr, lm, p, pa_, sensor, wald, d, 4, mp)
        for k in VIEWS:
            for q in acc[k]:
                acc[k][q].append(v[k][q])
    return {k: {q: float(np.mean(acc[k][q])) for q in acc[k]} for k in VIEWS}, np.array(D)


@torch.no_grad()
def shift_response(m, ds, dev, grid=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    """§11.2: 기대 Aφ(W(P,e),M) − Aφ(P,M) = −e."""
    rows = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        d0 = m(pan, ms, lpan)["delta"][0].float().cpu().numpy()
        for ey in grid:
            for ex in grid:
                e = torch.tensor([[ey, ex]], device=dev)
                pe = warp_pan(pan, e)
                de = m(pe.to(pan.dtype), ms, lpan)["delta"][0].float().cpu().numpy()
                rows.append(dict(scene=i, ey=ey, ex=ex, resp_dy=float(de[0] - d0[0]), resp_dx=float(de[1] - d0[1])))
    R = np.array([[r["ey"], r["ex"], r["resp_dy"], r["resp_dx"]] for r in rows])
    err = np.hypot(R[:, 2] + R[:, 0], R[:, 3] + R[:, 1])
    def slope(x, y):
        x = x - x.mean(); return float((x * (y - y.mean())).sum() / ((x * x).sum() + 1e-12))
    return dict(mean_abs_err_hr=float(err.mean()), median_abs_err_hr=float(np.median(err)),
                slope_dy=slope(R[:, 0], R[:, 2]), slope_dx=slope(R[:, 1], R[:, 3]), ideal_slope=-1.0,
                sign_ok_frac_dy=float(np.mean(np.sign(R[R[:, 0] != 0, 2]) == -np.sign(R[R[:, 0] != 0, 0]))),
                sign_ok_frac_dx=float(np.mean(np.sign(R[R[:, 1] != 0, 3]) == -np.sign(R[R[:, 1] != 0, 1]))), n=len(rows))


@torch.no_grad()
def tile_vs_full(m, ds, dev, tile=128):
    out = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        ms_up = torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic")
        full = m.aligner(pan.float(), ms_up.float())[0].float().cpu().numpy()
        H, W = pan.shape[-2:]; ts = []
        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                ts.append(m.aligner(pan[:, :, y:y + tile, x:x + tile].float(), ms_up[:, :, y:y + tile, x:x + tile].float())[0].float().cpu().numpy())
        ts = np.array(ts)
        out.append(dict(scene=i, full_dy=float(full[0]), full_dx=float(full[1]), tile_dy_median=float(np.median(ts[:, 0])), tile_dx_median=float(np.median(ts[:, 1])),
                        tile_dy_iqr=float(np.subtract(*np.percentile(ts[:, 0], [75, 25]))), tile_dx_iqr=float(np.subtract(*np.percentile(ts[:, 1], [75, 25]))),
                        tile_minus_full_norm_median=float(np.median(np.hypot(ts[:, 0] - full[0], ts[:, 1] - full[1])))))
    return out


def cross_table(wd):
    """§10.10: 세 checkpoint 의 세 view (학습 중 기록에서)."""
    p = os.path.join(wd, "checkpoint_metrics.csv")
    rows = {int(r["step"]): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
    out = {}
    for tag in ("best_hqnr", "best_aligned", "last"):
        mp = os.path.join(wd, f"{tag}_meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp)); step = meta.get("step")
        r = rows.get(int(step)) if step is not None else None
        out[tag] = dict(step=step, status=meta.get("status"), **({k: float(v) for k, v in r.items() if "." in k or k in ("aligned_eligible",) and v not in ("", None)} if r else {}))
        if r:
            out[tag]["aligned_eligible"] = r.get("aligned_eligible")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args(); dev = torch.device(a.device)
    wd, cfg, m, ds = load_run(a.run, a.ckpt, dev)
    sensor = efp.sensor_of(cfg); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = float(ds.max_pixel)
    out = dict(run=a.run, ckpt=a.ckpt, case=(cfg.get("pa") or {}).get("case"), cross_evaluation=cross_table(wd))
    print(f"[{a.run}] §10.10 교차 평가 (학습 중 기록):")
    for tag, r in out["cross_evaluation"].items():
        print(f"  {tag:13s} step {r.get('step')}  " + "  ".join(f"{v}: HQNR {r.get(f'{v}.hqnr', float('nan')):.4f} fSCC {r.get(f'{v}.fscc', float('nan')):.4f}" for v in VIEWS) + f"  eligible {r.get('aligned_eligible')}")
    if not a.skip_controls:
        ctrl = {}
        for mode in ("learned", "zero", "wrong_sign"):
            v, D = views_for(m, ds, sensor, wald, mp, dev, mode)
            ctrl[mode] = dict(views=v, delta_median=[float(np.median(D[:, 0])), float(np.median(D[:, 1]))], delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))))
            print(f"  §11.1 {mode:10s} raw_original HQNR {v['raw_original']['hqnr']:.4f} fSCC {v['raw_original']['fscc']:.4f} | raw_valid HQNR {v['raw_valid']['hqnr']:.4f} | aligned_valid HQNR {v['aligned_valid']['hqnr']:.4f}  Δ median ({np.median(D[:,0]):+.3f},{np.median(D[:,1]):+.3f})")
        out["controls"] = ctrl
        out["shift_response"] = shift_response(m, ds, dev)
        s = out["shift_response"]
        print(f"  §11.2 known-shift response: |err| mean {s['mean_abs_err_hr']:.3f} px, slope dy {s['slope_dy']:+.2f} dx {s['slope_dx']:+.2f} (ideal −1), sign ok dy {s['sign_ok_frac_dy']:.2f} dx {s['sign_ok_frac_dx']:.2f}")
        out["tile_vs_full"] = tile_vs_full(m, ds, dev)
        t = out["tile_vs_full"]
        print(f"  §11.4 tile(128) vs full: |tile−full| median {np.median([r['tile_minus_full_norm_median'] for r in t]):.3f} px, tile IQR dy {np.median([r['tile_dy_iqr'] for r in t]):.3f} dx {np.median([r['tile_dx_iqr'] for r in t]):.3f}")
    os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(wd, "results", "pa_diag.json"), "w"), indent=1)
    print(f"  -> {os.path.relpath(os.path.join(wd, 'results', 'pa_diag.json'), ROOT)}")


if __name__ == "__main__":
    main()
