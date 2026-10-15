#!/usr/bin/env python
"""Teacher uncertainty head-only calibration — s2 계획(2026-09-01) §2·§6.

완료된 clean MS-only teacher 의 **mean backbone 을 고정**하고 log-variance head 만
따로 학습한다. mean 이 바뀌지 않으므로 이 단계는 모델 선택(HQNR)의 대상이 아니다 —
그래서 학습 체인의 case 가 아니라 사후 도구로 둔다.

산출물 (<mean_workdir>/uq_head/):
  model.safetensors   base+head 결합 state_dict — train_kd.load_teacher 가 그대로 읽는다
  uq_norm.json        고정 상수: q05/q95(s 분포), kappa(GT-variance squash), 진단 수치

  python tools/calibrate_head.py work_dir/S2_T00_W160_D122_MS2 --iters 8000

절차 (계획 §2):
  1. mean = best_hqnr checkpoint, freeze
  2. e_loc = A_3( (1/C)Σ_c (Y−μ)² )
  3. head-only loss = 0.5·exp(−s)·e_loc + 0.5·s
  4. 학습셋에서 s 의 q05/q95, V(R_GT) 의 중앙값(κ) 고정
  5. calibration 검사: Spearman(σ², e_loc) > 0 · 오차 5분위 단조 ·
     전역 분산 기준선 대비 NLL 개선  (계획 §2 게이트)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import import_class  # noqa: E402
from kd.features import WithUncertainty  # noqa: E402
from kd.losses import local_error_map, logvar_nll  # noqa: E402
from kd.ops import multiscale_variance  # noqa: E402


def build(wd):
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", os.path.basename(wd.rstrip("/")) + ".yaml")
    cfg = yaml.safe_load(open(cfg_path))
    base = import_class(cfg["model"])(**cfg["model_args"])
    sd = load_file(os.path.join(wd, "best_hqnr", "model.safetensors"))
    assert not any(k.startswith("head.") for k in sd), \
        "이미 uncertainty head 가 있는 checkpoint — mean-only teacher 를 지정할 것"
    base.load_state_dict(sd)
    return cfg, base


def loaders(cfg, batch, workers=4):
    Feeder = import_class(cfg["feeder"])
    tr = torch.utils.data.DataLoader(Feeder(**cfg["train_feeder_args"]), batch_size=batch,
                                     shuffle=True, num_workers=workers, drop_last=True)
    va = torch.utils.data.DataLoader(Feeder(**cfg["val_feeder_args"]), batch_size=8,
                                     shuffle=False, num_workers=2)
    return tr, va


def forward_mean(model, ms, lpan, pan, res):
    sw = torch.ones(pan.shape[0], device=pan.device, dtype=pan.dtype)
    out = model(pan, lpan, ms, sw)
    return out + F.interpolate(ms, scale_factor=4, mode="bicubic") if res else out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mean_workdir")
    ap.add_argument("--iters", type=int, default=8000, help="head-only 학습 iteration (계획 5-10K)")
    ap.add_argument("--lr", type=float, default=1e-3, help="head 만 학습하므로 backbone 보다 크게")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", default=None, help="기본 <mean_workdir>/uq_head")
    a = ap.parse_args()

    wd = a.mean_workdir.rstrip("/")
    out_dir = a.out or os.path.join(wd, "uq_head")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, base = build(wd)
    res = bool(cfg.get("res", True))
    model = WithUncertainty(base, cfg["model_args"].get("hidden_size", 128),
                            head_out="logvar").to(dev)
    model.base.requires_grad_(False)
    model.base.eval()                      # mean 고정 — dropout/통계 변동 없음
    model.head.requires_grad_(True)
    tr, va = loaders(cfg, a.batch, cfg.get("num_worker", 4))

    # warm start: 마지막 conv 를 (weight 0, bias log E[e_loc]) 로 두면 head 는
    # **전역 상수 분산 해**에서 출발한다. NLL 게이트가 "전역 상수보다 나은가" 이므로
    # 이 초기화 뒤의 학습은 원리적으로 개선만 한다 (미학습 head 가 게이트를 떨어뜨리는
    # 실패 모드 제거). 공간 변조는 전부 학습으로 얻는다.
    with torch.no_grad():
        acc, nb = 0.0, 0
        for gt, lms, ms, lpan, pan in tr:
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            acc += local_error_map(forward_mean(model, ms, lpan, pan, res), gt).mean().item()
            nb += 1
            if nb >= 8:
                break
        e0 = acc / max(1, nb)
        last = model.head.net[-1]
        last.weight.zero_()
        last.bias.fill_(float(np.log(e0 + 1e-12)))
    opt = torch.optim.AdamW(model.head.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

    print(f"[calib] mean={os.path.basename(wd)}  head-only {a.iters} iters  "
          f"(head params {sum(p.numel() for p in model.head.parameters()):,}, "
          f"warm start s0={np.log(e0 + 1e-12):+.3f})")

    step = 0
    while step < a.iters:
        for gt, lms, ms, lpan, pan in tr:
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            with torch.no_grad():
                mu = forward_mean(model, ms, lpan, pan, res)
                e_loc = local_error_map(mu, gt)          # mean 고정 -> 상수 타깃
            s = model.theta()                            # log sigma^2
            loss, d = logvar_nll(e_loc, s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            step += 1
            if step % 500 == 0 or step == 1:
                print(f"[calib] {step}/{a.iters}  NLL {loss.item():.5f}  "
                      f"s {d['s_mean']:+.3f}  e_loc {d['e_loc_mean']:.5f}")
            if step >= a.iters:
                break

    # ---------------- 고정 상수(train) + PASS 판정(held-out val) --------------
    # 계획 §4·§6 은 q05/q95·κ 를 "training set 에서 고정" 하라고 명시한다.
    # 반면 calibration PASS(§2)는 **일반화** 주장이므로 학습에 쓰지 않은 validation
    # 에서 재야 한다 — 같은 train 배치에서 재면 head 가 외운 것을 통과로 읽는다.
    model.eval()

    def collect(loader, limit, want_v):
        S_, E_, V_ = [], [], []
        with torch.no_grad():
            for i, (gt, lms, ms, lpan, pan) in enumerate(loader):
                if i >= limit:
                    break
                gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32)
                                     for x in (gt, ms, lpan, pan))
                mu = forward_mean(model, ms, lpan, pan, res)
                sv = model.theta()
                S_.append(sv[..., ::2, ::2].flatten().cpu().numpy())
                E_.append(local_error_map(mu, gt)[..., ::2, ::2].flatten().cpu().numpy())
                if want_v:
                    up = F.interpolate(ms, scale_factor=4, mode="bicubic")
                    V_.append(multiscale_variance(gt - up).flatten().cpu().numpy())
        return (np.concatenate(S_), np.concatenate(E_),
                np.concatenate(V_) if want_v else None)

    S, _Etr, V = collect(tr, 40, True)            # 고정 상수용 (train)
    q05, q95 = float(np.quantile(S, 0.05)), float(np.quantile(S, 0.95))
    kappa = float(np.median(V))                   # Ṽ 중앙값이 0.5 가 되는 스케일

    Sv, Ev, _ = collect(va, 10**9, False)         # 판정용 (held-out validation 전체)
    sel = np.random.default_rng(0).choice(len(Sv), min(200_000, len(Sv)), replace=False)
    rho = float(spearmanr(Sv[sel], Ev[sel]).statistic)
    qs = np.quantile(Sv, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(Sv, qs)
    quint = [float(Ev[bins == b].mean()) for b in range(5)]
    monotonic = bool(all(quint[i + 1] >= quint[i] - 1e-9 for i in range(4)))
    # NLL 비교: 예측 분산 vs 전역 상수 분산(최적 상수 = val 오차 평균)
    nll_pred = float(np.mean(0.5 * np.exp(-Sv) * Ev + 0.5 * Sv))
    s_glob = float(np.log(Ev.mean() + 1e-12))
    nll_glob = float(np.mean(0.5 * np.exp(-s_glob) * Ev + 0.5 * s_glob))
    ok = bool(rho > 0 and monotonic and nll_pred < nll_glob)

    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    info = {"q05": q05, "q95": q95, "kappa": kappa, "iters": a.iters,
            "diag_split": "val", "const_split": "train", "n_val_px": int(len(Sv)),
            "spearman": rho, "quintile_e_loc": quint, "monotonic": monotonic,
            "nll_pred": nll_pred, "nll_global": nll_glob,
            "nll_gain": nll_glob - nll_pred, "pass": ok,
            "mean_workdir": wd, "head_out": "logvar"}
    json.dump(info, open(os.path.join(out_dir, "uq_norm.json"), "w"), indent=1)

    print(f"\n[calib] q05 {q05:+.3f}  q95 {q95:+.3f}  kappa {kappa:.6f}")
    print(f"[calib] (held-out val) Spearman(s, e_loc) {rho:.4f} · "
          f"5분위 {['%.5f' % q for q in quint]} 단조 {monotonic}")
    print(f"[calib] NLL 예측 {nll_pred:.5f} vs 전역상수 {nll_glob:.5f} "
          f"(개선 {nll_glob - nll_pred:+.5f})")
    print(f"[calib] -> {'PASS' if ok else 'FAIL'}   저장: {out_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
