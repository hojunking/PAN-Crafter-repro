#!/usr/bin/env python
"""Global shift cache 생성 (계획 §5). GT 는 읽지 않는다 — 입력은 lpan(MTF↓PAN, phase 2)과 ms 뿐.

  python tools/build_shift_cache.py [--out outputs/global_shift_cache] [--splits train rr fr]

train 은 parent scene ID 가 h5 에 없으므로 16x16 LR patch 단위(§5.2 우선순위 2)다. RR/FR 은
scene 전체(64²/128² LR). 결과 통계(accept 율·|δ| 분위)를 stdout 에 찍고 cache_meta.json 에 남긴다.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from align.estimator import estimate_shift, GATES          # noqa: E402
from align.cache import sha256_file, fingerprint_h5, ESTIMATOR_VERSION, SPLIT_FILE   # noqa: E402

R = os.path.join(ROOT, "data/PanCollection/WV3")
SRC = {"train": (os.path.join(R, "train_wv3.h5"), 0),
       "rr": (os.path.join(R, "reduced_examples_h5/test_wv3_multiExm1.h5"), 2),
       "fr": (os.path.join(R, "full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5"), 3)}


def build(split, out_dir):
    h5, code = SRC[split]
    pan_h5 = h5.replace(".h5", "_pan.h5")
    with h5py.File(h5) as f:
        ms = f["ms"][:]
    with h5py.File(pan_h5) as f:
        lpan = f["lpan"][:]
    assert ms.shape[0] == lpan.shape[0] and ms.shape[-2:] == lpan.shape[-2:]
    t0, rows = time.time(), []
    for i in range(ms.shape[0]):
        r = estimate_shift(lpan[i, 0], ms[i].mean(0))
        r.update(split=split, sample_id=i, scene_id=(i if split != "train" else -1),
                 estimator_version=ESTIMATOR_VERSION, source_file_hash=fingerprint_h5(h5) if i == 0 else "")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["source_file_hash"] = df["source_file_hash"].iloc[0]
    cols = ["split", "sample_id", "scene_id", "dy_lr_raw", "dx_lr_raw", "magnitude_raw", "peak_zncc", "peak_margin",
            "secondary_dy", "secondary_dx", "primary_secondary_diff", "boundary_hit", "accepted",
            "dy_lr_applied", "dx_lr_applied", "estimator_version", "source_file_hash"]
    df = df[cols]
    fn = SPLIT_FILE[code]
    df.to_csv(os.path.join(out_dir, fn), index=False)
    mag = df["magnitude_raw"].values
    stats = dict(n=int(len(df)), accepted=float(df["accepted"].mean()), boundary_hit=float(df["boundary_hit"].mean()),
                 mag_p50=float(np.median(mag)), mag_p90=float(np.quantile(mag, .9)), mag_max=float(mag.max()),
                 dy_mean=float(df["dy_lr_raw"].mean()), dy_sd=float(df["dy_lr_raw"].std()),
                 dx_mean=float(df["dx_lr_raw"].mean()), dx_sd=float(df["dx_lr_raw"].std()),
                 sign_dy_pos=float((df["dy_lr_raw"] > 0).mean()), peak_p50=float(df["peak_zncc"].median()),
                 seconds=round(time.time() - t0, 1))
    print(f"[cache] {split:5s} n={stats['n']:5d} accepted {stats['accepted']*100:5.1f}%  |δ| p50 {stats['mag_p50']:.3f} "
          f"p90 {stats['mag_p90']:.3f}  dy {stats['dy_mean']:+.3f}±{stats['dy_sd']:.3f}  dx {stats['dx_mean']:+.3f}±{stats['dx_sd']:.3f}"
          f"  sign(dy)>0 {stats['sign_dy_pos']*100:.0f}%  ({stats['seconds']}s)")
    return fn, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/global_shift_cache"))
    ap.add_argument("--splits", nargs="+", default=["train", "rr", "fr"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta_p = os.path.join(a.out, "cache_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.update(sign_convention="aligned[y,x]=moving[y+dy,x+dx]", order="(dy,dx)", scale=4,
                search_radius_lr_int=GATES["search_int"], max_magnitude_lr=GATES["max_magnitude"],
                mtf_sigma=1.98, mtf_ksize=41, decimation_phase=2, primary="scharr_zncc_q3x3",
                secondary="census5_hamming", gates=GATES, estimator_version=ESTIMATOR_VERSION,
                note="train 은 parent scene ID 부재로 16x16 patch 단위(§5.2 우선순위 2). lpan 은 데이터셋 제공(MTF↓PAN phase2)")
    meta.setdefault("sha256", {}); meta.setdefault("stats", {})
    for s in a.splits:
        fn, st = build(s, a.out)
        meta["sha256"][fn] = sha256_file(os.path.join(a.out, fn))
        meta["stats"][s] = st
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"[cache] meta -> {meta_p}")


if __name__ == "__main__":
    main()
