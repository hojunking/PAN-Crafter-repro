#!/usr/bin/env python
"""추론 전용 global-alignment sweep — 학습 없이 기존 checkpoint 에 Δ 를 적용해 HQNR/fSCC 를 본다.

  python tools/align_infer_sweep.py --config config/S1_T05_W152_D123_DUAL.yaml \
      --ckpt work_dir/S1_T05_W152_D123_DUAL/best_hqnr --base-upsampler bicubic_phase2 --base-phase 1.5

왜 필요한가: train patch 의 audit Δ 는 추정기 노이즈(16² 에서 오차 0.27 px > shift 0.06)라
frozen-cache case 의 학습 시 정렬은 사실상 sub-pixel jitter 이고, 의미 있는 Δ(FR 0.2~0.45 px)는
추론 시에만 들어간다. 그러면 "추론 시 Δ 적용" 만으로 50K 학습 없이 C1/C2/C3 의 방향을 미리
볼 수 있다. 결과는 <run>/results/infer_sweep.csv.

case 표기  delta_source:alpha:output_frame:inverse_location  (예 cache:1:P:loss_branch)
--base-upsampler/--base-phase 는 checkpoint 가 학습된 입력 보간(기존 코드 = bicubic phase 1.5)이다.
GA_* run 은 interp23tap 로 학습됐으니 --base-upsampler interp23tap 를 준다.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from main import import_class                                        # noqa: E402
from align.model import AlignCfg, AlignedModel                       # noqa: E402
from align.cache import ShiftCache                                   # noqa: E402
from feeders.feeder_align import PanFeederAlign                      # noqa: E402
from utils import tensor2img, SCC_full_numpy, reduced_metrics        # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s        # noqa: E402

DEFAULT_CASES = ["zero:0:M:none", "cache:0.25:M:none", "cache:0.5:M:none", "cache:0.75:M:none",
                 "cache:1:M:none", "cache:1:M:final_output", "cache:1:P:loss_branch"]


def load_backbone(cfg, ckpt):
    Model = import_class(cfg["model"])
    m = Model(**cfg["model_args"])
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    m.load_state_dict(sd)
    return m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="accelerate 디렉터리 (model.safetensors)")
    ap.add_argument("--base-upsampler", default="bicubic_phase2", choices=["bicubic_phase2", "interp23tap"])
    ap.add_argument("--base-phase", type=float, default=1.5, help="bicubic_phase2 일 때 phase (기존 코드 = 1.5)")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs/global_shift_cache"))
    ap.add_argument("--indices", default="12-19")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(cfg, a.ckpt).to(dev)
    cache = ShiftCache(a.cache_dir)
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    lo, hi = (int(x) for x in a.indices.split("-"))
    maxp = float(cfg.get("max_pixel", 2047.0))
    fr = PanFeederAlign(**cfg["test_full_feeder_args"])
    rr = PanFeederAlign(**cfg["test_reduced_feeder_args"])

    def dn(t):
        return np.round((((t + 1) / 2).clamp(0, 1).float().cpu().numpy().astype(np.float64)) * maxp)

    rows = []
    for case in a.cases:
        src, alpha, frame, inv = case.split(":")
        acfg = AlignCfg.from_dict(dict(upsampler=a.base_upsampler, phase=a.base_phase, delta_source=src,
                                       alpha=float(alpha), output_frame=frame, inverse_location=inv,
                                       cache_dir=a.cache_dir))
        M = AlignedModel(backbone, acfg).to(dev).eval()
        dl, ds, fs, dl2, ds2, fs2, mags = [], [], [], [], [], [], []
        with torch.no_grad():
            for i in range(len(fr)):
                lms, ms, lpan, pan, meta = fr[i]
                lms, ms, lpan, pan = (t[None].to(dev) for t in (lms, ms, lpan, pan))
                d = torch.zeros(1, 2, device=dev) if src == "zero" else cache.lookup(3, meta[None, 0], device=dev)[0]
                out = M.infer_ms(pan, lpan, ms, d)
                if not (lo <= i <= hi):
                    continue
                y = out["y_final"]
                sr, lm, pn = dn(y[0]).transpose(1, 2, 0), dn(lms[0]).transpose(1, 2, 0), dn(pan[0, 0])
                dl.append(d_lambda_k(sr, lm, "wv3", 4, 32, wald)); ds.append(d_s(sr, lm, pn, 4, 32, wald))
                fs.append(SCC_full_numpy(tensor2img(pan, maxp), tensor2img(y, maxp)))
                mags.append(d.norm().item())
                alt = out["y_ms"] if frame == "P" else out["y_pan"]
                if alt is not None:
                    sa = dn(alt[0]).transpose(1, 2, 0)
                    dl2.append(d_lambda_k(sa, lm, "wv3", 4, 32, wald)); ds2.append(d_s(sa, lm, pn, 4, 32, wald))
                    fs2.append(SCC_full_numpy(tensor2img(pan, maxp), tensor2img(alt, maxp)))
            er, sm, sc = [], [], []
            for i in range(len(rr)):
                gt, lms, ms, lpan, pan, meta = rr[i]
                gt, ms, lpan, pan = (t[None].to(dev) for t in (gt, ms, lpan, pan))
                d = torch.zeros(1, 2, device=dev) if src == "zero" else cache.lookup(2, meta[None, 0], device=dev)[0]
                out = M.infer_ms(pan, lpan, ms, d)
                r = reduced_metrics(x_true=gt, x_pred=out["y_loss"], max_pixel=maxp)
                er.append(r["ergas"]); sm.append(r["sam"]); sc.append(r["scc"])
        row = dict(case=case, hqnr=(1 - np.mean(dl)) * (1 - np.mean(ds)), d_lambda=np.mean(dl), d_s=np.mean(ds),
                   fscc=np.mean(fs), delta_mag=np.mean(mags), rr_ergas=np.mean(er), rr_sam=np.mean(sm), rr_scc=np.mean(sc),
                   hqnr_alt=((1 - np.mean(dl2)) * (1 - np.mean(ds2)) if dl2 else ""), fscc_alt=(np.mean(fs2) if fs2 else ""))
        rows.append(row)
        print(f"{case:26s} HQNR {row['hqnr']:.4f}  D_l {row['d_lambda']:.4f} D_s {row['d_s']:.4f}  fSCC {row['fscc']:.5f}"
              f"  |Δ| {row['delta_mag']:.3f}  RR ERGAS {row['rr_ergas']:.4f} SCC {row['rr_scc']:.5f}"
              + (f"  [alt HQNR {row['hqnr_alt']:.4f} fSCC {row['fscc_alt']:.5f}]" if dl2 else ""))
    out = a.out or os.path.join(os.path.dirname(a.ckpt.rstrip("/")), "results", "infer_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
