#!/usr/bin/env python
"""Teacher uncertainty calibration 검사 — 명세 §8.3.

T1/T2 의 best_hqnr checkpoint 에서 validation set 으로 θ-오차 정합을 잰다:
  - Spearman(θ, |err|)  (픽셀 서브샘플)
  - θ 5분위별 MAE 단조성
pass = Spearman > 0 이고 quintile MAE 가 비내림(허용오차 1e-4).
결과를 <workdir>/calibration.json 에 기록한다. K2+ 게이트가 이 파일을 읽는다.

  python tools/check_calibration.py work_dir/T1_c6_unc
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_kd import load_teacher  # noqa: E402
from main import import_class  # noqa: E402


def main():
    wd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work_dir", "T1_c6_unc")
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", os.path.basename(wd.rstrip("/")) + ".yaml")
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, has_unc = load_teacher(cfg_path, os.path.join(wd, "best_hqnr"), dev, torch.float32)
    assert has_unc, "uncertainty head 가 없는 checkpoint — T1/T2 만 검사 대상"

    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["val_feeder_args"])
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    thetas, errs = [], []
    with torch.no_grad():
        for i, (gt, lms, ms, lpan, pan) in enumerate(loader):
            if i >= 32:                                   # ≈256 샘플이면 충분
                break
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            ones = torch.ones(pan.shape[0], device=dev)
            out = model(pan, lpan, ms, ones)
            recon = out + F.interpolate(ms, scale_factor=4, mode="bicubic")
            err = (recon - gt).abs().mean(dim=1, keepdim=True)
            th = model.theta()
            thetas.append(th[..., ::2, ::2].flatten().cpu().numpy())
            errs.append(err[..., ::2, ::2].flatten().cpu().numpy())
    th = np.concatenate(thetas)
    er = np.concatenate(errs)
    if len(th) > 200_000:
        sel = np.random.default_rng(0).choice(len(th), 200_000, replace=False)
        th, er = th[sel], er[sel]

    rho, _ = spearmanr(th, er)
    q = np.quantile(th, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(th, q)
    mae = [float(er[bins == b].mean()) for b in range(5)]
    monotonic = all(mae[i + 1] >= mae[i] - 1e-4 for i in range(4))
    ok = bool(rho > 0 and monotonic)
    sd_path = os.path.join(wd, "best_hqnr", "model.safetensors")
    st = os.stat(sd_path)
    result = {"spearman": float(rho), "quintile_mae": mae,
              "monotonic": monotonic, "pass": ok, "n_pixels": int(len(th)),
              "ckpt_signature": f"{st.st_size}-{st.st_mtime_ns}"}
    json.dump(result, open(os.path.join(wd, "calibration.json"), "w"), indent=1)
    print(f"[calib] {os.path.basename(wd)}: Spearman {rho:.4f}  "
          f"quintile MAE {['%.4f' % m for m in mae]}  단조 {monotonic}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
