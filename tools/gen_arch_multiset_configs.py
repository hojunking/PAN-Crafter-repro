#!/usr/bin/env python
"""아키텍처 고정(W168·d123·dual·11ch·nocrop·attn 없음·50K) 다중 데이터셋 3-seed config 생성.

    python tools/gen_arch_multiset_configs.py            # config/ARCH_W168_D123_DUAL_<DS>_S<seed>.yaml 8벌

템플릿은 config/S1_T05_W168_D123_DUAL.yaml (WV3·seed 2025 — 이 run 자체가 WV3 seed 2025 멤버다).
데이터셋별로 바뀌는 것: 경로 · num_bands/out_channels(QB·GF2 4) · max_pixel(GF2 1023) · expect_params_m.
WV3 FR 은 복구 lpan(full_examples_h5_repaired), QB FR 도 복구본(F-1: 배포 lpan 손상), GF2 는 배포본 정상.
QB 학습·검증은 ms 복구본 train_qb_msfix.h5 / valid_qb_msfix.h5 (F-3: 배포본 ms 의 2/3 가 gt 대비 LR 1px 어긋남).
best 선택은 종전대로 FR H5 12-19 HQNR(fr_select_indices 기본값) — 보고는 논문 세트(.mat 20)로 별도.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "config", "S1_T05_W168_D123_DUAL.yaml")
DS = {  # sensor -> (bands, max_pixel, params_M(8band 5.9731 / 4band 5.9610), FR dir)
    "wv3": (8, 2047.0, 5.9731, "full_examples_h5_repaired"),
    "qb":  (4, 2047.0, 5.9610, "full_examples_h5_repaired"),   # train/valid 는 _msfix (F-3)
    "gf2": (4, 1023.0, 5.9610, "full_examples_h5"),
}
SEEDS = {"wv3": (1234, 7777), "qb": (2025, 1234, 7777), "gf2": (2025, 1234, 7777)}   # WV3 2025 = S1_T05_W168_D123_DUAL

def main():
    tpl = open(TPL).read()
    made = []
    for s, (nb, mp, pm, frdir) in DS.items():
        for seed in SEEDS[s]:
            tag = f"ARCH_W168_D123_DUAL_{s.upper()}_S{seed}"
            t = tpl
            t = re.sub(r"^# S1 T05 .*\n(#.*\n)*", "", t)      # 머리 주석 교체
            head = (f"# 아키텍처 고정 다중 데이터셋 3-seed — {s.upper()} seed {seed}\n"
                    f"# 생성: tools/gen_arch_multiset_configs.py (템플릿 S1_T05_W168_D123_DUAL). 손으로 고치지 말 것.\n"
                    f"# 공통: W168 · depth [1,2,3] · dual MARs · 11ch · crop=False · attention 없음 · 50K · AdamW 1e-4/wd0.01\n"
                    f"#       best 선택 = 공식 HQNR(FR H5 12-19) · 보고 = 논문 세트(.mat 20, results/fr_mat20.json)\n")
            t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {pm}", t)
            t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
            t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
            fix = "_msfix" if s == "qb" else ""      # QB 학습·검증셋은 ms 복구본 (KNOWN_ISSUES F-3, tools/repair_qb_ms.py)
            t = t.replace("data/PanCollection/WV3/train_wv3.h5", f"data/PanCollection/{s.upper()}/train_{s}{fix}.h5")
            t = t.replace("data/PanCollection/WV3/valid_wv3.h5", f"data/PanCollection/{s.upper()}/valid_{s}{fix}.h5")
            t = t.replace("data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5",
                          f"data/PanCollection/{s.upper()}/reduced_examples_h5/test_{s}_multiExm1.h5")
            t = t.replace("data/PanCollection/WV3/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5",
                          f"data/PanCollection/{s.upper()}/{frdir}/test_{s}_OrigScale_multiExm1.h5")
            t = re.sub(r"^num_bands: \d+", f"num_bands: {nb}", t, flags=re.M)
            t = re.sub(r"^max_pixel: [\d.]+", f"max_pixel: {mp}", t, flags=re.M)
            t = re.sub(r"^  out_channels: \d+", f"  out_channels: {nb}", t, flags=re.M)
            assert f"/{s.upper()}/" in t and f"out_channels: {nb}" in t and f"seed: {seed}" in t
            open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t)
            made.append(tag)
    q = os.path.join(ROOT, "config", "queues", "arch_w168_multiset_3seed.txt")
    with open(q, "w") as f:
        f.write("# 아키텍처 고정(W168 d123 dual 11ch nocrop) 다중 데이터셋 3-seed. 각 서버가 같은 큐를 돌린다.\n"
                "# WV3 seed 2025 는 기존 S1_T05_W168_D123_DUAL. WV3 run 이 끝나면 _upload.sh 가 WV2 zero-shot(_zs_wv2) 도 만든다.\n")
        for tag in made:
            f.write(tag + "\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
