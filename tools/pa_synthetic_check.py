#!/usr/bin/env python
"""§9.4 aligner 학습 가능성 사전 검사 — modality mismatch 없는 통제 쌍에서 input-dependent shift 를 배우는가.

    python tools/pa_synthetic_check.py [--steps 400] [--n 256]

RR 테스트 GT(WV3, 256²) 에서 64² patch 를 뽑아: PAN_syn = GT band 평균(부호 반전), MS_up_syn = 같은 GT 를 알려진 e~U(−2,2)² HR px 로 옮긴 것.
aligner 만(zero head 초기화) 학습: L = |W(PAN_syn, Aφ) − target|₁, target = PAN_syn 을 e 로 옮긴 것 (정답 Δ = e, sampling 부호).
합격: hold-out 에서 Δ̂ 와 e 의 축별 상관 ≥ 0.9, |Δ̂ − e| 중앙값 ≤ 0.25 px. 이 checkpoint 는 버린다 (학습 augmentation 아님).
"""
import argparse, os, sys, json
import numpy as np, torch, h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--n", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(os.path.join(ROOT, "data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5")) as f:
        gt = torch.from_numpy(np.asarray(f["gt"], dtype=np.float32)) / 2047.0 * 2 - 1          # [20,8,256,256]
    def batch(n):
        idx = torch.randint(0, gt.shape[0], (n,)); ys = torch.randint(0, 256 - 64, (n,)); xs = torch.randint(0, 256 - 64, (n,))
        g = torch.stack([gt[i, :, y:y + 64, x:x + 64] for i, y, x in zip(idx, ys, xs)]).to(dev)
        e = (torch.rand(n, 2, device=dev) * 4 - 2)                                             # 알려진 shift
        pan = -g.mean(1, keepdim=True)                                                          # 대비 반전 PAN
        ms_up = warp_pan(g, -e) if False else torch.cat([warp_pan(g[:, c:c + 1], -e) for c in range(g.shape[1])], 1)   # MS 를 e 만큼 어긋나게
        target = warp_pan(pan, e)                                                                # 정답: PAN 을 e 로 sampling 하면 MS 와 정합
        return pan, ms_up, target, e
    al = PANGlobalAligner(8).to(dev); opt = torch.optim.AdamW(al.parameters(), lr=1e-3)
    for it in range(a.steps):
        pan, ms_up, target, e = batch(a.n)
        d = al(pan, ms_up); loss = (warp_pan(pan, d) - target).abs()[:, :, 8:-8, 8:-8].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == a.steps - 1:
            with torch.no_grad():
                err = (d - e).norm(dim=1)
            print(f"  step {it:4d} loss {loss.item():.5f}  |Δ̂−e| median {err.median().item():.3f} px")
    al.eval()
    with torch.no_grad():
        pan, ms_up, target, e = batch(512); d = al(pan, ms_up)
        err = (d - e).norm(dim=1); cy = np.corrcoef(d[:, 0].cpu(), e[:, 0].cpu())[0, 1]; cx = np.corrcoef(d[:, 1].cpu(), e[:, 1].cpu())[0, 1]
        med = err.median().item()
    ok = cy >= 0.9 and cx >= 0.9 and med <= 0.25
    res = dict(steps=a.steps, corr_dy=float(cy), corr_dx=float(cx), median_abs_err_hr=med, p95_abs_err_hr=float(err.quantile(0.95)), passed=bool(ok))
    print(f"[synthetic §9.4] corr dy {cy:.3f} dx {cx:.3f}, |Δ̂−e| median {med:.3f} px p95 {res['p95_abs_err_hr']:.3f} -> {'PASS' if ok else 'FAIL'}")
    os.makedirs(os.path.join(ROOT, "outputs", "pa"), exist_ok=True); json.dump(res, open(os.path.join(ROOT, "outputs", "pa", "synthetic_check.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
