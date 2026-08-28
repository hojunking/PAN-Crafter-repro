#!/usr/bin/env python
"""아키텍처 사전 smoke 검증 — 체인이 2시간짜리 슬롯을 태우기 전에 분 단위로 잡는다.

계획서 §8.1 (research_log/2026-08-28_architecture-search-24h-plan.md) 의 점검 항목:
  1. config 로드 -> 모델 build -> params 실측
  2. 학습 형상 forward + backward (dual 이면 switch 0/1 혼합, mars: ms 면 1만)
     -> 출력 shape (B,8,64,64) · finite · grad finite
  3. FR 형상(512², WV3 full-res) no-grad forward -> shape · finite
     (PixelUnshuffle 배수 문제·해상도 의존 버그 검출)
  4. peak VRAM · 대략적인 step 시간

zero_module 함정: 출력 conv 가 0 으로 초기화돼 있어 그대로면 상류 grad 가 전부 0
이라 backward 검사가 공허하다. 검사 전에 0 파라미터를 난수화한다 (CLAUDE.md 함정 목록).

사용:
  python tools/smoke_cases.py <config이름> [<config이름> ...]
종료코드: 하나라도 실패면 1 (체인은 이걸 보고 해당 case 를 학습 없이 FAILED 처리)
"""
import os
import sys
import time
import importlib

import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_cfg(name):
    for p in (os.path.join(ROOT, "config", f"{name}.yaml"),
              os.path.join(ROOT, "config", f"pancrafter_{name}.yaml")):
        if os.path.exists(p):
            return yaml.safe_load(open(p))
    raise FileNotFoundError(f"config 없음: {name}")


def build(cfg):
    mod, cls = cfg["model"].rsplit(".", 1)
    M = getattr(importlib.import_module(mod), cls)
    return M(**cfg["model_args"])


def randomize_zero_params(m):
    """zero_module 로 0 초기화된 파라미터를 난수화 — backward 가 실제로 흐르게."""
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dtype.is_floating_point and p.abs().sum() == 0:
                p.normal_(0, 1e-3)


def smoke_one(name, dev):
    cfg = load_cfg(name)
    m = build(cfg).to(dev)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    randomize_zero_params(m)
    dual = cfg.get("mars", "dual") != "ms"

    # 학습 형상: PAN 64², LPAN/MS 16² (feeder ms_size=16 기준)
    B = 4
    pan = torch.randn(B, 1, 64, 64, device=dev)
    lpan = torch.randn(B, 1, 16, 16, device=dev)
    ms = torch.randn(B, cfg.get("num_bands", 8), 16, 16, device=dev)
    switch = (torch.tensor([0., 0., 1., 1.], device=dev) if dual
              else torch.ones(B, device=dev))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    out = m(pan, lpan, ms, switch)
    assert out.shape == (B, cfg.get("num_bands", 8), 64, 64), f"학습 출력 shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "학습 출력에 NaN/Inf"
    out.abs().mean().backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"grad NaN/Inf: {bad[:3]}"
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "0 이 아닌 grad 가 없다 — backward 가 흐르지 않음"
    step_ms = (time.time() - t0) * 1000

    # FR 형상: WV3 full-res 는 PAN 512², LPAN/MS 128²
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_fr = m(torch.randn(1, 1, 512, 512, device=dev),
                   torch.randn(1, 1, 128, 128, device=dev),
                   torch.randn(1, cfg.get("num_bands", 8), 128, 128, device=dev),
                   torch.ones(1, device=dev))
    assert out_fr.shape[-2:] == (512, 512), f"FR 출력 shape {tuple(out_fr.shape)}"
    assert torch.isfinite(out_fr).all(), "FR 출력에 NaN/Inf"

    peak = (torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else 0
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return n_params, step_ms, peak, "dual" if dual else "ms"


def main():
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        sys.exit(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failed = []
    print(f"[smoke] device={dev}  cases={len(names)}")
    for name in names:
        try:
            p, ms_t, peak, mars = smoke_one(name, dev)
            print(f"[smoke] OK   {name:26s} {p:7.4f}M  step≈{ms_t:6.0f}ms  "
                  f"peakVRAM {peak:6.0f}MB  mars={mars}")
        except Exception as e:
            print(f"[smoke] FAIL {name:26s} {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print(f"[smoke] 실패 {len(failed)}: {' '.join(failed)}")
        sys.exit(1)
    print("[smoke] 전부 통과")


if __name__ == "__main__":
    main()
