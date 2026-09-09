#!/usr/bin/env python
"""아키텍처 고정(W168·d123·dual·11ch·nocrop·attn 없음·50K) 다중 데이터셋 3-seed config 생성.

    python tools/gen_arch_multiset_configs.py            # config/ARCH_W168_D123_DUAL_<DS>_S<seed>.yaml 8벌

템플릿은 config/S1_T05_W168_D123_DUAL.yaml (WV3·seed 2025 — 이 run 자체가 WV3 seed 2025 멤버다).
데이터셋별로 바뀌는 것: 경로 · num_bands/out_channels(QB·GF2 4) · max_pixel(GF2 1023) · expect_params_m.
WV3 FR 은 복구 lpan(full_examples_h5_repaired), QB FR 도 복구본(F-1: 배포 lpan 손상), GF2 는 배포본 정상.
QB 학습·검증은 ms 복구본 train_qb_msfix.h5 / valid_qb_msfix.h5 (F-3: 배포본 ms 의 2/3 가 gt 대비 LR 1px 어긋남).
best 선택은 논문 세트(.mat 20장 전체) HQNR — 보고 세트와 같다 (2026-09-09: 12-19 부분집합 폐기).
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "config", "S1_T05_W168_D123_DUAL.yaml")
DS = {  # sensor -> (bands, max_pixel, params_M(8band 5.9731 / 4band 5.9610))
    "wv3": (8, 2047.0, 5.9731),
    "qb":  (4, 2047.0, 5.9610),      # train/valid 는 _msfix (F-3)
    "gf2": (4, 1023.0, 5.9610),
}
# 학습 중 FR 평가·best 선택은 **논문 세트(.mat 20장) 전체**로 한다 (2026-09-09 결정: 12-19 부분집합 폐기, 논문 프로토콜).
# 입력 h5 는 tools/build_paperset_all.sh 가 만든 full_examples_mat20/ (lpan 은 F-1 레시피).
SEEDS = {"wv3": (2025, 1234, 7777), "qb": (2025, 1234, 7777), "gf2": (2025, 1234, 7777)}

def main():
    tpl = open(TPL).read()
    made = []
    for s, (nb, mp, pm) in DS.items():
        for seed in SEEDS[s]:
            tag = f"ARCH_W168_D123_DUAL_{s.upper()}_S{seed}"
            t = tpl
            t = re.sub(r"^# S1 T05 .*\n(#.*\n)*", "", t)      # 머리 주석 교체
            head = (f"# 아키텍처 고정 다중 데이터셋 3-seed — {s.upper()} seed {seed}\n"
                    f"# 생성: tools/gen_arch_multiset_configs.py (템플릿 S1_T05_W168_D123_DUAL). 손으로 고치지 말 것.\n"
                    f"# 공통: W168 · depth [1,2,3] · dual MARs · 11ch · crop=False · attention 없음 · 50K · AdamW 1e-4/wd0.01\n"
                    f"#       best 선택 = 공식 HQNR, 논문 세트(.mat 20장 전체, full_examples_mat20) · 보고도 같은 세트(results/fr_mat20.json)\n")
            t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {pm}", t)
            t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
            t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
            fix = "_msfix" if s == "qb" else ""      # QB 학습·검증셋은 ms 복구본 (KNOWN_ISSUES F-3, tools/repair_qb_ms.py)
            t = t.replace("data/PanCollection/WV3/train_wv3.h5", f"data/PanCollection/{s.upper()}/train_{s}{fix}.h5")
            t = t.replace("data/PanCollection/WV3/valid_wv3.h5", f"data/PanCollection/{s.upper()}/valid_{s}{fix}.h5")
            t = t.replace("data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5",
                          f"data/PanCollection/{s.upper()}/reduced_examples_h5/test_{s}_multiExm1.h5")
            t = t.replace("data/PanCollection/WV3/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5",
                          f"data/PanCollection/{s.upper()}/full_examples_mat20/test_{s}_OrigScale_mat20.h5")
            t = t.replace("select_on: hqnr", "select_on: hqnr\nfr_select_indices: \"0-19\"   # 논문 세트 20장 전체로 선택 (부분집합 없음)")
            t = re.sub(r"^num_bands: \d+", f"num_bands: {nb}", t, flags=re.M)
            t = re.sub(r"^max_pixel: [\d.]+", f"max_pixel: {mp}", t, flags=re.M)
            t = re.sub(r"^  out_channels: \d+", f"  out_channels: {nb}", t, flags=re.M)
            assert f"/{s.upper()}/" in t and f"out_channels: {nb}" in t and f"seed: {seed}" in t
            open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t)
            made.append(tag)
    q = os.path.join(ROOT, "config", "queues", "arch_w168_multiset_3seed.txt")
    with open(q, "w") as f:
        f.write("# 아키텍처 고정(W168 d123 dual 11ch nocrop) 다중 데이터셋 3-seed. 각 서버가 같은 큐를 돌린다.\n"
                "# 선택·보고 모두 논문 세트(.mat 20장 전체). 순서 GF2 → QB(ms 복구본) → WV3. WV3 run 이 끝나면 _upload.sh 가 WV2 zero-shot 도 만든다.\n")
        order = [t for t in made if "_GF2_" in t] + [t for t in made if "_QB_" in t] + [t for t in made if "_WV3_" in t]
        for tag in order:
            f.write(tag + "\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
