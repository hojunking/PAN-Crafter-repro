#!/usr/bin/env python
"""ShiftNet pseudo-label pretraining (계획 §15.3). C4 두 run 이 같은 checkpoint 를 쓴다.

  python tools/pretrain_shiftnet.py --cache-dir outputs/global_shift_cache --out outputs/global_shift_cache/shiftnet_pretrained.pt

target = train cache 의 accepted Δ. 검증 10% 는 sample_id 해시. 통과 조건(§15.3):
  accepted val median |err| <= 0.10 · P90 <= 0.25 · sign accuracy >= 95%
결과 json(<out>.json) 에 pass 여부와 수치를 남긴다 — AlignTrainer 가 읽어 C4 시작 여부를 정한다.

--synthetic : (계획 외 옵션) cache 대신 LR-MS 를 알려진 Δ 로 옮겨 만든 합성 라벨로 학습한다.
              train patch 의 audit Δ 가 노이즈(|δ| p50≈0.06, sign 50%)일 때의 대안. 기본 꺼짐.
"""
import argparse
import hashlib
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from align.shiftnet import GlobalShiftNet, structural_input     # noqa: E402
from align.resample import _sample, hr_grid                     # noqa: E402

TRAIN_H5 = os.path.join(ROOT, "data/PanCollection/WV3/train_wv3.h5")


def _val_mask(ids, frac=0.10):
    return np.array([int(hashlib.md5(f"wv3_train:{i}".encode()).hexdigest(), 16) % 1000 < frac * 1000 for i in ids])


def _norm(x, max_pixel=2047.0):
    return 2.0 * x / max_pixel - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs/global_shift_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/global_shift_cache/shiftnet_pretrained.pt"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-shift", type=float, default=1.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(a.cache_dir, "wv3_train.csv")).sort_values("sample_id")
    with h5py.File(TRAIN_H5) as f:
        ms = torch.tensor(_norm(f["ms"][:]), dtype=torch.float32)
    with h5py.File(TRAIN_H5.replace(".h5", "_pan.h5")) as f:
        lpan = torch.tensor(_norm(f["lpan"][:]), dtype=torch.float32)
    tgt = torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)
    acc = torch.tensor(df["accepted"].values.astype(bool))
    ids = df["sample_id"].values
    val = torch.tensor(_val_mask(ids))
    tr_idx = torch.where(acc & ~val)[0]; va_idx = torch.where(acc & val)[0]
    print(f"[pretrain] accepted train {len(tr_idx)} / val {len(va_idx)}  synthetic={a.synthetic}")

    net = GlobalShiftNet(a.max_shift).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def batch(idx, synthetic):
        m, p = ms[idx].to(dev), lpan[idx].to(dev)
        if synthetic:
            # moving = ms 를 -Δ 만큼 옮긴 것 (aligned[y,x]=moving[y+dy,x+dx] 가 원래 ms 로 돌아오도록)
            d = (torch.rand(len(idx), 2, device=dev) * 2 - 1) * 0.75
            B, _, H, W = m.shape
            yy, xx = hr_grid(B, H, W, m.dtype, dev)
            m = _sample(m, yy - d[:, 0].view(B, 1, 1), xx - d[:, 1].view(B, 1, 1))
            return structural_input(p, m), d
        return structural_input(p, m), tgt[idx].to(dev)

    for step in range(1, a.steps + 1):
        idx = tr_idx[torch.randint(len(tr_idx), (a.batch,))]
        x, y = batch(idx, a.synthetic)
        loss = F.smooth_l1_loss(net(x), y, beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"[pretrain] step {step} loss {loss.item():.5f}")

    net.eval()
    with torch.no_grad():
        x, y = batch(va_idx, False)
        pred = torch.cat([net(x[i:i + 256]) for i in range(0, len(x), 256)])
        err = (pred - y).norm(dim=1).cpu().numpy()
        sign_ok = ((torch.sign(pred) == torch.sign(y)) | (y.abs() < 1e-6)).float().mean(dim=1).cpu().numpy()
        big = (y.abs().max(dim=1).values >= 0.15).cpu().numpy()
        rep = {"pass": bool(np.median(err) <= 0.10 and np.quantile(err, .9) <= 0.25 and sign_ok.mean() >= 0.95)}
        rep.update(dict(
                   val_n=int(len(err)), val_median_err=float(np.median(err)), val_p90_err=float(np.quantile(err, .9)),
                   val_sign_acc=float(sign_ok.mean()),
                   val_sign_acc_big=float(sign_ok[big].mean()) if big.any() else None, val_n_big=int(big.sum()),
                   pred_mag_mean=float(pred.norm(dim=1).mean()), pred_zero_ratio=float((pred.norm(dim=1) < 0.01).float().mean()),
                   target_mag_median=float(y.norm(dim=1).median()), synthetic=bool(a.synthetic), steps=a.steps,
                   corr_dy=float(np.corrcoef(pred[:, 0].cpu(), y[:, 0].cpu())[0, 1]),
                   corr_dx=float(np.corrcoef(pred[:, 1].cpu(), y[:, 1].cpu())[0, 1])))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(net.state_dict(), a.out)
    json.dump(rep, open(a.out.replace(".pt", ".json"), "w"), indent=1)
    print(f"[pretrain] {'PASS' if rep['pass'] else 'FAIL'}  median {rep['val_median_err']:.4f}  P90 {rep['val_p90_err']:.4f}  "
          f"sign {rep['val_sign_acc']:.3f} (|t|>=0.15: {rep['val_sign_acc_big']})  corr dy {rep['corr_dy']:+.3f} dx {rep['corr_dx']:+.3f}"
          f"  pred|.| {rep['pred_mag_mean']:.3f} zero {rep['pred_zero_ratio']:.2f}")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
