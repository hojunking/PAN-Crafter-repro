#!/usr/bin/env python
"""SE gate 사후 진단 — 계획 §7 (research_log/se_ablation_two_experiments.md).

checkpoint 를 로드해 validation 배치에서 SE gate 를 MS/PAN 양 mode 로 뽑아:
  위치별 gate mean/std/min/max · 포화 비율(s<0.05 또는 s>0.95)
  MS-mode vs PAN-mode gate cosine 유사도 (1.0 에 가깝고 분산도 0 이면
  SE 가 실질적으로 쓰이지 않는 것 — 계획 §7 판정)

  python tools/analyze_se_gates.py work_dir/SE1_R4_btl_se
"""
import json
import os
import sys

import torch
import yaml
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import import_class  # noqa: E402
from model.se import SEGate  # noqa: E402


def main():
    wd = sys.argv[1]
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", os.path.basename(wd.rstrip("/")) + ".yaml")
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = import_class(cfg["model"])(**cfg["model_args"])
    sd = load_file(os.path.join(wd, "best_hqnr", "model.safetensors"))
    m.load_state_dict(sd)
    m.to(dev).eval()

    # gate 캡처 hook
    gates = {}
    def cap(name):
        def hook(mod, inp, out):
            gates.setdefault(name, []).append(mod.gate(inp[0]).detach())
        return hook
    n_se = 0
    for name, mod in m.named_modules():
        if isinstance(mod, SEGate):
            mod.register_forward_hook(cap(name))
            n_se += 1
    assert n_se > 0, "SE 모듈이 없다 — config 확인"

    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["val_feeder_args"])
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    result = {}
    with torch.no_grad():
        for i, (gt, lms, ms, lpan, pan) in enumerate(loader):
            if i >= 8:
                break
            ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (ms, lpan, pan))
            for mode, sw in (("MS", torch.ones(pan.shape[0], device=dev)),
                             ("PAN", torch.zeros(pan.shape[0], device=dev))):
                gates.clear()
                m(pan, lpan, ms, sw)
                for name, gs in gates.items():
                    g = torch.cat(gs).squeeze(-1).squeeze(-1)     # (B, C)
                    result.setdefault(name, {}).setdefault(mode, []).append(g.cpu())

    report = {}
    for name, modes in result.items():
        gm = torch.cat(modes["MS"]); gp = torch.cat(modes["PAN"])
        cos = torch.nn.functional.cosine_similarity(gm.mean(0), gp.mean(0), dim=0).item()
        sat = ((gm < 0.05) | (gm > 0.95)).float().mean().item()
        report[name] = {"ms_mean": gm.mean().item(), "ms_std": gm.std().item(),
                        "ms_min": gm.min().item(), "ms_max": gm.max().item(),
                        "pan_mean": gp.mean().item(), "mode_cosine": cos,
                        "saturated_ratio": sat}
        print(f"[SE] {name:28s} MS {gm.mean():.3f}±{gm.std():.3f} "
              f"[{gm.min():.3f},{gm.max():.3f}]  PAN {gp.mean():.3f}  "
              f"mode-cos {cos:.4f}  포화 {sat:.1%}")
    dead = all(r["ms_std"] < 0.01 and r["mode_cosine"] > 0.999 for r in report.values())
    print(f"\n[SE] 판정: {'게이트 미사용 의심 (상수 수렴)' if dead else '게이트 활성'}")
    json.dump(report, open(os.path.join(wd, "se_gates.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
