#!/usr/bin/env python
"""A1–A3 run 의 사후 진단·교차 평가 (명세 §10.10 · §11). 학습이 끝난 run 에 대해 한 번 돈다 (tools/_upload.sh 가 PA_ run 에 대해 부른다).

    PANCRAFTER_DLPAN=... python tools/pa_diag.py --run PA_A1_REC_W96_D124_9CH_S2025            # PA run
    PANCRAFTER_DLPAN=... python tools/pa_diag.py --run BASE_W96_D124_MSPAN_WV3_S2025            # B0: Δ=0 로 같은 view·공통 V 재평가 (§10.10)

산출 work_dir/<run>/results/pa_diag.json + stdout 표:
  1. §10.10 교차 평가 — 저장된 checkpoint(best_hqnr=best_raw · best_aligned · last)를 **다시 추론**해 raw_original / raw_valid / aligned_valid
     HQNR·D_λ·D_s·fSCC 와 RR-original / RR-valid 를 같은 evaluator 로 계산 (학습 중 CSV 에 의존하지 않는다). B0 는 best_hqnr 만, aligned_valid = raw_valid.
  2. §11.1 aligner 사용 대조 (best_hqnr): learned Δ̂ / Δ=0 / −Δ̂ 세 추론 — 서로 다른 SR. SR 은 results/controls_<mode>_best_hqnr.mat 에 저장.
  3. §11.2 추가 shift 반응: Aφ(W(P,e),M) − Aφ(P,M) ≈ −e, e ∈ {−1,−0.5,0,0.5,1}² HR px.
  4. §11.4 full-scene Δ̂ vs 128px tile Δ̂ 분포.
  5. §11.3 독립 구조 진단: audit 추정기(align/estimator.py, Scharr-ZNCC)로 PAN_b←up(MS) 와 P̃_b←up(MS) 의 전역 shift — 보정 PAN 이 MS 에 가까워졌는가.
     RR band 별 Scharr edge error (SR vs GT).
이 값들은 진단이며 best 선택·시트 값을 바꾸지 않는다.
"""
import argparse, csv, json, os, sys
import numpy as np, torch, yaml, h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                           # noqa: E402
from pa.aligner import PANGlobalAligner                                 # noqa: E402
from pa.model import PAModel                                            # noqa: E402
from pa.warp import warp_pan                                            # noqa: E402
from pa.evalviews import scene_views, fixed_roi, VIEWS, MAX_ELIGIBLE_SHIFT   # noqa: E402
from pa.losses import scharr                                            # noqa: E402
from tools.metrics.eval_fr import load_dlpan                            # noqa: E402
from utils import reduced_metrics                                       # noqa: E402
import tools.eval_fr_paperset as efp                                    # noqa: E402


def load_run(run, ckpt, dev):
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    from safetensors.torch import load_file
    sd = load_file(os.path.join(wd, ckpt, "model.safetensors"))
    bb = import_class(cfg["model"])(**cfg["model_args"]); is_pa = cfg.get("trainer") in ("pa", "po")
    from pa.offset import aligner_margin
    mg = aligner_margin(float((cfg.get("po") or {}).get("radius_hr", 1.0))) if cfg.get("trainer") == "po" else 0
    m = PAModel(bb, PANGlobalAligner(int(cfg["num_bands"])), aligner_margin=mg)
    if is_pa:
        m.load_state_dict(sd, strict=True)
    else:
        bb.load_state_dict(sd, strict=True)                               # B0: aligner 는 쓰지 않는다 (Δ=0)
    return wd, cfg, m.to(dev).eval(), is_pa


def refs(cfg):
    with h5py.File(cfg["test_full_feeder_args"]["dataroot"]) as f:
        lms = np.asarray(f["lms"], dtype=np.float64); pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    return lms, pan


def dn(t, mp):
    return ((t.clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64)


@torch.no_grad()
def fr_views(m, ds, lms_raw, pan_raw, sensor, wald, mp, dev, mode="learned", is_pa=True):
    """mode: learned | zero | wrong_sign. 반환 (view 별 평균, Δ 배열, 적격 수, SR 배열(DN, NCHW))."""
    acc = {v: dict(hqnr=[], fscc=[], d_s=[], d_lambda=[]) for v in VIEWS}; D, ok_n, srs = [], 0, []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        if not is_pa or mode == "zero":
            o = m(pan, ms, lpan, aligner_enabled=False)
        else:
            o = m(pan, ms, lpan)
            if mode == "wrong_sign":
                o = m(pan, ms, lpan, delta_override=-o["delta"])
        sr = dn(o["y"][0], mp).transpose(1, 2, 0); d = o["delta"][0].float().cpu().numpy(); D.append(d); srs.append(sr.transpose(2, 0, 1))
        p = pan_raw[i]; pa_eval = warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d.astype(np.float64))[None])[0, 0].numpy()
        v, ok, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp); ok_n += int(ok)
        for k in VIEWS:
            for q in acc[k]:
                acc[k][q].append(v[k][q])
    return {k: {q: float(np.mean(acc[k][q])) for q in acc[k]} for k in VIEWS}, np.array(D), ok_n, np.array(srs)


@torch.no_grad()
def rr_metrics(m, cfg, mp, dev, is_pa):
    """RR-original / RR-valid (같은 64px margin 규칙) + band 별 Scharr edge error (§11.3)."""
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_reduced_feeder_args"])
    agg, aggv, edge = {}, {}, []
    for i in range(len(ds)):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        o = m(pan, ms, lpan) if is_pa else m(pan, ms, lpan, aligner_enabled=False)
        r = reduced_metrics(x_true=gt, x_pred=o["y"], max_pixel=mp)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
        H, W = gt.shape[-2:]
        if H > 128:
            y0, y1, x0, x1 = fixed_roi(H, W)
            rv = reduced_metrics(x_true=gt[..., y0:y1, x0:x1], x_pred=o["y"][..., y0:y1, x0:x1], max_pixel=mp)
            for k, v in rv.items():
                aggv.setdefault(k, []).append(v)
        gx, gy = scharr(o["y"].float()); hx, hy = scharr(gt.float())
        edge.append(((gx - hx).abs() + (gy - hy).abs()).mean(dim=(0, 2, 3)).cpu().numpy() * 0.5)
    return ({k: float(np.mean(v)) for k, v in agg.items()}, {k: float(np.mean(v)) for k, v in aggv.items()},
            dict(per_band=np.mean(edge, axis=0).tolist(), mean=float(np.mean(edge))))


@torch.no_grad()
def shift_response(m, ds, dev, grid=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    rows = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        d0 = m(pan, ms, lpan)["delta"][0].float().cpu().numpy()
        for ey in grid:
            for ex in grid:
                e = torch.tensor([[ey, ex]], device=dev)
                de = m(warp_pan(pan, e).to(pan.dtype), ms, lpan)["delta"][0].float().cpu().numpy()
                rows.append([ey, ex, float(de[0] - d0[0]), float(de[1] - d0[1])])
    R = np.array(rows); err = np.hypot(R[:, 2] + R[:, 0], R[:, 3] + R[:, 1])
    def slope(x, y):
        x = x - x.mean(); return float((x * (y - y.mean())).sum() / ((x * x).sum() + 1e-12))
    return dict(mean_abs_err_hr=float(err.mean()), median_abs_err_hr=float(np.median(err)), slope_dy=slope(R[:, 0], R[:, 2]), slope_dx=slope(R[:, 1], R[:, 3]), ideal_slope=-1.0,
                sign_ok_frac_dy=float(np.mean(np.sign(R[R[:, 0] != 0, 2]) == -np.sign(R[R[:, 0] != 0, 0]))),
                sign_ok_frac_dx=float(np.mean(np.sign(R[R[:, 1] != 0, 3]) == -np.sign(R[R[:, 1] != 0, 1]))), n=len(rows))


@torch.no_grad()
def tile_vs_full(m, ds, dev, tile=128):
    out = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        ms_up = torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic")
        full = m.aligner(m._view(pan.float()), m._view(ms_up.float()))[0].float().cpu().numpy()      # po: 고정 내부 view 적용
        H, W = pan.shape[-2:]; ts = []
        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                ts.append(m.aligner(m._view(pan[:, :, y:y + tile, x:x + tile].float()), m._view(ms_up[:, :, y:y + tile, x:x + tile].float()))[0].float().cpu().numpy())
        ts = np.array(ts)
        out.append(dict(scene=i, full_dy=float(full[0]), full_dx=float(full[1]), tile_dy_median=float(np.median(ts[:, 0])), tile_dx_median=float(np.median(ts[:, 1])),
                        tile_dy_iqr=float(np.subtract(*np.percentile(ts[:, 0], [75, 25]))), tile_dx_iqr=float(np.subtract(*np.percentile(ts[:, 1], [75, 25]))),
                        tile_minus_full_norm_median=float(np.median(np.hypot(ts[:, 0] - full[0], ts[:, 1] - full[1])))))
    return out


@torch.no_grad()
def independent_structure(m, ds, pan_raw, dev, sensor):
    """§11.3: audit 추정기로 PAN_b←up(MS) 와 P̃_b←up(MS) (HR 격자, PAN 은 PAN MTF 로 흐림). 보정 PAN 이 MS 격자에 가까워졌는가."""
    from align.estimator import estimate_shift, GATES
    from tools.align_after_training_diag import blur_hr, up_bicubic
    from tools.metrics.jqm import _pan_kernel
    G = dict(GATES, search_int=4, max_magnitude=4.0); kp = _pan_kernel(sensor.upper(), 4); rows = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        d = m(pan, ms, lpan)["delta"][0].double().cpu()
        p = pan_raw[i]; pt = warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy()
        up = up_bicubic(((ms[0].float().cpu().numpy() + 1) / 2 * 2047.0).transpose(1, 2, 0)).mean(2)
        a = estimate_shift(blur_hr(p, kp).astype(np.float32), up.astype(np.float32), G); b = estimate_shift(blur_hr(pt, kp).astype(np.float32), up.astype(np.float32), G)
        rows.append(dict(scene=i, orig_dy=a["dy_lr_raw"], orig_dx=a["dx_lr_raw"], orig_mag=a["magnitude_raw"], aligned_dy=b["dy_lr_raw"], aligned_dx=b["dx_lr_raw"], aligned_mag=b["magnitude_raw"],
                         delta_dy=float(d[0]), delta_dx=float(d[1])))
    om = np.median([r["orig_mag"] for r in rows]); am = np.median([r["aligned_mag"] for r in rows])
    return dict(per_scene=rows, orig_mag_median=float(om), aligned_mag_median=float(am), note="PAN_b←up(MS) 전역 shift (HR px). 값이 줄면 P̃ 가 MS 격자에 가까워진 것")


def cross_table(wd):
    """학습 중 기록(checkpoint_metrics.csv)에서 세 checkpoint 행을 뽑는다 (재추론 값의 대조용). bool/문자열 열은 float 변환하지 않는다."""
    p = os.path.join(wd, "checkpoint_metrics.csv")
    rows = {int(r["step"]): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
    out = {}
    for tag in ("best_hqnr", "best_aligned", "last"):
        mp = os.path.join(wd, f"{tag}_meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp)); step = meta.get("step"); r = rows.get(int(step)) if step is not None else None
        rec = dict(step=step, status=meta.get("status"))
        if r:
            for k, v in r.items():
                if k in ("step", "epoch", "invalid_reasons"):
                    continue
                if k == "aligned_eligible":
                    rec[k] = v in ("True", "1", "true", 1, True)
                else:
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        rec[k] = v
        out[tag] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args(); dev = torch.device(a.device)
    wd0 = os.path.join(ROOT, "work_dir", a.run); cfg0 = yaml.safe_load(open(os.path.join(wd0, "meta", "config.yaml")))
    is_pa = cfg0.get("trainer") in ("pa", "po")
    sensor = efp.sensor_of(cfg0); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    Feeder = import_class(cfg0["feeder"]); ds = Feeder(**cfg0["test_full_feeder_args"]); mp = float(ds.max_pixel)
    lms_raw, pan_raw = refs(cfg0)
    out = dict(run=a.run, trainer=(cfg0.get("trainer") if is_pa else "b0"), case=((cfg0.get("pa") or cfg0.get("po") or {}).get("case")), cross_evaluation={}, training_records=cross_table(wd0) if is_pa else {})
    tags = ["best_hqnr", "best_aligned", "last"] if is_pa else ["best_hqnr"]
    print(f"[{a.run}] §10.10 교차 평가 (재추론, 같은 evaluator):")
    for tag in tags:
        if not os.path.exists(os.path.join(wd0, tag, "model.safetensors")):
            continue
        wd, cfg, m, _ = load_run(a.run, tag, dev)
        v, D, ok_n, _ = fr_views(m, ds, lms_raw, pan_raw, sensor, wald, mp, dev, "learned", is_pa)
        rr, rrv, edge = rr_metrics(m, cfg, mp, dev, is_pa)
        meta = json.load(open(os.path.join(wd, f"{tag}_meta.json"))) if os.path.exists(os.path.join(wd, f"{tag}_meta.json")) else {}
        out["cross_evaluation"][tag] = dict(step=meta.get("step"), views=v, rr_original=rr, rr_valid=rrv, rr_band_edge_error=edge, delta_median=[float(np.median(D[:, 0])), float(np.median(D[:, 1]))],
                                            delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))), n_eligible=ok_n, n_scenes=len(D), max_eligible_abs_shift=MAX_ELIGIBLE_SHIFT)
        print(f"  {tag:13s} step {meta.get('step')}  " + "  ".join(f"{k}: HQNR {v[k]['hqnr']:.4f} fSCC {v[k]['fscc']:.4f}" for k in VIEWS)
              + f"  | RR ERGAS {rr.get('ergas', float('nan')):.4f} SCC {rr.get('scc', float('nan')):.4f} (valid ERGAS {rrv.get('ergas', float('nan')):.4f}) | Δ median ({np.median(D[:,0]):+.3f},{np.median(D[:,1]):+.3f}) eligible {ok_n}/{len(D)}")
    if is_pa and not a.skip_controls:
        wd, cfg, m, _ = load_run(a.run, "best_hqnr", dev)
        from scipy.io import savemat
        ctrl = {}
        for mode in ("learned", "zero", "wrong_sign"):
            v, D, ok_n, srs = fr_views(m, ds, lms_raw, pan_raw, sensor, wald, mp, dev, mode, True)
            ctrl[mode] = dict(views=v, delta_median=[float(np.median(D[:, 0])), float(np.median(D[:, 1]))], delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))), n_eligible=ok_n)
            if mode != "learned":
                savemat(os.path.join(wd, "results", f"controls_{mode}_best_hqnr.mat"), dict(sr=srs.astype(np.float32), delta=D))
            print(f"  §11.1 {mode:10s} raw_original HQNR {v['raw_original']['hqnr']:.4f} fSCC {v['raw_original']['fscc']:.4f} | raw_valid HQNR {v['raw_valid']['hqnr']:.4f} | aligned_valid HQNR {v['aligned_valid']['hqnr']:.4f}  Δ median ({np.median(D[:,0]):+.3f},{np.median(D[:,1]):+.3f})")
        out["controls"] = ctrl
        out["shift_response"] = s = shift_response(m, ds, dev)
        print(f"  §11.2 known-shift response: |err| mean {s['mean_abs_err_hr']:.3f} px, slope dy {s['slope_dy']:+.2f} dx {s['slope_dx']:+.2f} (ideal −1), sign ok dy {s['sign_ok_frac_dy']:.2f} dx {s['sign_ok_frac_dx']:.2f}")
        out["tile_vs_full"] = t = tile_vs_full(m, ds, dev)
        print(f"  §11.4 tile(128) vs full: |tile−full| median {np.median([r['tile_minus_full_norm_median'] for r in t]):.3f} px, tile IQR dy {np.median([r['tile_dy_iqr'] for r in t]):.3f} dx {np.median([r['tile_dx_iqr'] for r in t]):.3f}")
        out["independent_structure"] = st = independent_structure(m, ds, pan_raw, dev, sensor)
        print(f"  §11.3 audit 추정기 PAN_b←up(MS) |δ| median: 원본 {st['orig_mag_median']:.3f} → 보정 P̃ {st['aligned_mag_median']:.3f} HR px")
    os.makedirs(os.path.join(wd0, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(wd0, "results", "pa_diag.json"), "w"), indent=1)
    print(f"  -> {os.path.relpath(os.path.join(wd0, 'results', 'pa_diag.json'), ROOT)}")


if __name__ == "__main__":
    main()
