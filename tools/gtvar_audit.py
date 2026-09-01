#!/usr/bin/env python
"""GT-variance gradient audit — s2 계획 §6 의 필수 절차.

  ||grad(λ_V · L_GTVar)|| / ||grad(L_hard)||  를 1K step 부근에서 재고,
  계획이 정한 gradient 예산 대역(0.05-0.10)에 들어오도록 λ_V 를 정한다.
  계획의 이분 규칙(0.10 -> 0.05)을 먼저 적용하되, 그것으로도 대역에 못 들어가면
  대역 중앙을 맞추는 λ_V 를 풀어 쓴다 — 규칙의 문언보다 규칙이 명시한 목표가 우선이다.
  어느 경로를 썼는지는 산출 JSON 의 rule 에 남는다.

계획이 못 박은 조건: **"이 조정은 두 seed 전에 한 번만 하고 양 seed 에 동일하게
고정한다."** 그래서 학습 루프 안에서 seed 별로 조정하지 않고, 이 도구가 한 번 재서
teacher 디렉터리(uq_head/)에 결정을 남긴다. 두 seed 의 KDTrainer 가 같은 파일을
읽으므로 λ_V 는 자동으로 동일해진다.

  python tools/gtvar_audit.py config/S2_GTVAR_L010_S2025.yaml --steps 1000

산출: <teacher_checkpoint>/gtvar_audit.json
  {"ratio_at_0.10": …, "lambda_gtvar": …, "expected_ratio": …, "rule": …}
"""
import argparse
import json
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import import_class  # noqa: E402
from train_kd import load_teacher  # noqa: E402
from kd.losses import gtvar_loss, weighted_l1, uknow_weights_fixed  # noqa: E402

TARGET_LO, TARGET_HI = 0.05, 0.10   # 계획 §6 이 명시한 gradient 예산 대역
TARGET_MID = 0.075
LAM_HI, LAM_LO = 0.10, 0.05         # 계획 §6 의 이분 규칙 값


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((x * x).sum() for x in g if x is not None)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=1000, help="워밍업 step 수 (계획: 1K)")
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = bool(cfg.get("res", True))
    bs = a.batch or int(cfg.get("batch_size", 48))

    student = import_class(cfg["model"])(**cfg["model_args"]).to(dev)
    teacher, has_unc = load_teacher(os.path.join(ROOT, cfg["teacher_config"]),
                                    os.path.join(ROOT, cfg["teacher_checkpoint"]),
                                    dev, torch.float32)
    assert has_unc, "uncertainty teacher 가 필요하다 (calibrate_head.py 먼저)"
    nrm = json.load(open(os.path.join(ROOT, cfg["teacher_checkpoint"], "uq_norm.json")))
    ka = cfg.get("kd_args", {})
    kappa = ka.get("gtvar_kappa", nrm["kappa"])
    q05, q95 = ka.get("uq_q05", nrm["q05"]), ka.get("uq_q95", nrm["q95"])

    Feeder = import_class(cfg["feeder"])
    tr = torch.utils.data.DataLoader(Feeder(**cfg["train_feeder_args"]), batch_size=bs,
                                     shuffle=True, num_workers=cfg.get("num_worker", 4),
                                     drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg.get("learning_rate", 1e-4)),
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    params = [p for p in student.parameters() if p.requires_grad]
    up = lambda x: torch.nn.functional.interpolate(x, scale_factor=4, mode="bicubic")

    print(f"[gtvar-audit] {os.path.basename(a.config)}  {a.steps} step 워밍업 후 측정")
    step, ratios = 0, []
    while step < a.steps + 20:
        for gt, lms, ms, lpan, pan in tr:
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            sw = torch.ones(pan.shape[0], device=dev)
            out = student(pan, lpan, ms, sw)
            res_ms = up(ms)
            pred = out + res_ms if res else out
            with torch.no_grad():
                t_out = teacher(pan, lpan, ms, sw)
                t_pred = t_out + res_ms if res else t_out
                theta = teacher.theta()
            w_hard, _w_soft, _ = uknow_weights_fixed(theta, q05, q95)
            hard = weighted_l1(pred, gt, w_hard)
            if step >= a.steps:                       # 워밍업 뒤에만 잰다
                l_gv, _ = gtvar_loss(pred - res_ms, gt - res_ms, kappa)
                ratios.append(grad_norm(LAM_HI * l_gv, params) / (grad_norm(hard, params) + 1e-12))
            opt.zero_grad(set_to_none=True)
            hard.backward()
            opt.step()
            step += 1
            if step >= a.steps + 20:
                break

    ratio = float(sum(ratios) / len(ratios))      # λ_V = LAM_HI 기준 측정값
    per_unit = ratio / LAM_HI                     # 비율은 λ_V 에 선형이다
    # 계획 §6 의 이분 규칙을 먼저 적용하되, 그것으로도 목표 대역(0.05-0.10)에
    # 들어가지 못하면 대역 중앙을 맞추는 λ_V 를 풀어서 쓴다 — 규칙의 문언보다
    # 규칙이 명시한 **목표**를 지키는 쪽이 맞다. 어느 쪽을 썼는지 기록에 남긴다.
    if ratio <= TARGET_HI:
        lam, rule = LAM_HI, "계획 이분규칙: 목표 대역 내 -> 0.10 유지"
    elif LAM_LO * per_unit <= TARGET_HI:
        lam, rule = LAM_LO, "계획 이분규칙: 0.10 초과 -> 0.05 하향"
    else:
        lam = round(TARGET_MID / per_unit, 4)
        rule = (f"이분규칙 부족(0.05 에서도 {LAM_LO * per_unit:.3f} > {TARGET_HI}) "
                f"-> 목표 중앙 {TARGET_MID} 를 맞추는 λ_V 로 해석")
    out_p = os.path.join(ROOT, cfg["teacher_checkpoint"], "gtvar_audit.json")
    json.dump({"ratio_at_0.10": ratio, "ratio_per_unit_lambda": per_unit,
               "lambda_gtvar": lam, "expected_ratio": lam * per_unit,
               "steps": a.steps, "n_measured": len(ratios),
               "config": os.path.basename(a.config), "rule": rule,
               "target_band": [TARGET_LO, TARGET_HI]},
              open(out_p, "w"), indent=1)
    print(f"[gtvar-audit] ||grad(0.10·L_GTVar)|| / ||grad(L_hard)|| = {ratio:.4f} "
          f"({len(ratios)} step 평균)")
    print(f"[gtvar-audit] {rule}")
    print(f"[gtvar-audit] λ_V = {lam}  -> 예상 비율 {lam * per_unit:.4f} "
          f"(목표 {TARGET_LO}-{TARGET_HI})")
    print(f"[gtvar-audit] -> {out_p}")


if __name__ == "__main__":
    main()
