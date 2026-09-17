"""PAN aligner 검증 — 빠진 대조군: **학습된 예측의 평균을 상수로 적용**.

PALSV18 V4 의 `constant_calibration` 은 train patch 의 median c0 (|c| ~ 0.14 px) 을 썼다.
FR scene 예측(|c| ~ 0.4-1.1 px) 보다 3-8 배 작아 사실상 zero 와 같았고, 그래서
"scene 별 예측이 단일 전역 상수보다 나은가" 는 검증된 적이 없다.

이 도구는 같은 평가 경로(pa.evalviews.scene_views, raw_original)로 다음을 비교한다.

  learned      scene i 에 그 scene 의 예측 c_i
  zero         보정 없음
  const_mean   모든 scene 에 mean(c_i) — **빠져 있던 대조군**
  const_median 모든 scene 에 median(c_i)
  shuffle      c_{(i+1) mod n} (PALSV18 과 같은 고정 derangement, 재현 확인용)

learned 과 const_mean 의 차이가 판정선(2sigma = 0.0031) 안이면, 이 지표 해상도에서
**scene 적응은 단일 상수 대비 이득이 증명되지 않은 것**이다.

사용:
    python tools/aligner_constant_control.py --run <run> [--ckpt best_hqnr]
    python tools/aligner_constant_control.py --runs-file <목록>       # 한 줄에 run[,ckpt]
산출: work_dir/<run>/palsv18/<ckpt>_constant_control.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from tools.po10_diag import load_run                                     # noqa: E402
from pa.warp import warp_pan                                             # noqa: E402
from pa.evalviews import scene_views                                     # noqa: E402
from tools.pa_diag import refs, dn                                       # noqa: E402
from tools.eval_fr_paperset import sensor_of, load_dlpan                 # noqa: E402
from main import import_class                                            # noqa: E402

BAND = 0.0031          # HQNR 판정선 (실측 2 sigma)


@torch.no_grad()
def run_one(run, ckpt, dev):
    wd, cfg, m, R, mg = load_run(run, ckpt, dev)
    if m.aligner is None:
        return dict(run=run, ckpt=ckpt, skipped="aligner 없음")
    sensor = sensor_of(cfg)
    wald = load_dlpan(os.environ["PANCRAFTER_DLPAN"])
    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["test_full_feeder_args"])
    mp = float(ds.max_pixel)
    lms_raw, pan_raw = refs(cfg)
    n = len(ds)

    learned = []
    for i in range(n):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
        learned.append(m(pan, ms, lpan)["delta"][0].float().cpu().numpy())
    L = np.stack(learned)
    c_mean = L.mean(0)
    c_med = np.median(L, 0)

    modes = {
        "learned": lambda i: L[i],
        "zero": lambda i: np.zeros(2, np.float32),
        "const_mean": lambda i: c_mean.astype(np.float32),
        "const_median": lambda i: c_med.astype(np.float32),
        "shuffle": lambda i: L[(i + 1) % n],
    }

    out = {}
    for mode, fn in modes.items():
        per = []
        acc = dict(hqnr=[], fscc=[], d_s=[], d_lambda=[])
        for i in range(n):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            d = fn(i)
            o = (m(pan, ms, lpan, aligner_enabled=False) if mode == "zero"
                 else m(pan, ms, lpan, delta_override=torch.from_numpy(d)[None].to(dev)))
            sr = dn(o["y"][0], mp).transpose(1, 2, 0)
            p = pan_raw[i]
            pa_eval = warp_pan(torch.from_numpy(p)[None, None],
                               torch.from_numpy(d.astype(np.float64))[None])[0, 0].numpy()
            v, ok, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_eval,
                                   sensor, wald, d, 4, mp)
            per.append(v["raw_original"]["hqnr"])
            for k in acc:
                acc[k].append(v["raw_original"][k])
        out[mode] = {k: float(np.mean(x)) for k, x in acc.items()}
        out[mode]["per_scene_raw_hqnr"] = per

    p_ = lambda k: np.asarray(out[k]["per_scene_raw_hqnr"])               # noqa: E731
    stat = dict(
        run=run, ckpt=ckpt, n_scenes=n,
        learned_c_mean=list(map(float, c_mean)), learned_c_median=list(map(float, c_med)),
        learned_c_norm_mean=float(np.linalg.norm(L, axis=1).mean()),
        learned_c_sd_dy=float(L[:, 0].std(ddof=1)), learned_c_sd_dx=float(L[:, 1].std(ddof=1)),
        band=BAND,
    )
    for k in ("zero", "const_mean", "const_median", "shuffle"):
        dlt = p_("learned") - p_(k)
        stat[f"learned_minus_{k}"] = float(dlt.mean())
        stat[f"learned_minus_{k}_sd"] = float(dlt.std(ddof=1))
        stat[f"learned_minus_{k}_n_pos"] = int((dlt > 0).sum())
        stat[f"learned_minus_{k}_inside_band"] = bool(abs(dlt.mean()) <= BAND)
    stat["modes"] = {k: {q: v for q, v in out[k].items() if q != "per_scene_raw_hqnr"}
                     for k in out}
    stat["per_scene"] = {k: out[k]["per_scene_raw_hqnr"] for k in out}
    stat["note"] = ("raw_original view, FR paper mat20 20 scenes. const_mean = 학습된 예측의 "
                    "scene 평균을 모든 scene 에 동일 적용 — PALSV18 의 constant_calibration"
                    "(train patch median) 과 다르다")

    od = os.path.join(wd, "palsv18")
    os.makedirs(od, exist_ok=True)
    pth = os.path.join(od, f"{ckpt}_constant_control.json")
    json.dump(stat, open(pth, "w"), indent=1)
    print(f"  -> {os.path.relpath(pth, ROOT)}")
    return stat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--runs-file", help="한 줄에 run[,ckpt]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    jobs = []
    if a.runs_file:
        for ln in open(a.runs_file):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [x.strip() for x in ln.split(",")]
            jobs.append((parts[0], parts[1] if len(parts) > 1 else "best_hqnr"))
    if a.run:
        jobs.append((a.run, a.ckpt))
    if not jobs:
        sys.exit("--run 또는 --runs-file 이 필요하다")

    res = []
    for run, ckpt in jobs:
        print(f"[{run} / {ckpt}]")
        try:
            res.append(run_one(run, ckpt, dev))
        except Exception as e:
            print(f"  ! {e!r}")
            res.append(dict(run=run, ckpt=ckpt, error=repr(e)[:200]))

    print(f"\n{'run':52}{'ckpt':>10}{'|c|':>7}{'L-zero':>9}{'L-cmean':>9}{'L-cmed':>9}{'L-shuf':>9}")
    for r in res:
        if "learned_minus_zero" not in r:
            continue
        mark = "" if r["learned_minus_const_mean_inside_band"] else "  *밴드밖"
        print(f"{r['run'][:52]:52}{r['ckpt']:>10}{r['learned_c_norm_mean']:7.3f}"
              f"{r['learned_minus_zero']:+9.5f}{r['learned_minus_const_mean']:+9.5f}"
              f"{r['learned_minus_const_median']:+9.5f}{r['learned_minus_shuffle']:+9.5f}{mark}")
    print(f"\n판정선 2sigma = {BAND}")


if __name__ == "__main__":
    main()
