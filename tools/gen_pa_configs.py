#!/usr/bin/env python
"""A1–A3 config 생성 (research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md §14).

    python tools/gen_pa_configs.py     # config/PA_A{1,2,3}_*_W96_D124_9CH_S{2025,1234,7777}.yaml + config/queues/pa_s{1,2,3}.txt

템플릿 = B0 config (config/BASE_W96_D124_MSPAN_WV3_S<seed>.yaml, tools/gen_w96_d124_mspan_configs.py). 학습 조건은 B0 와 동일하고
trainer: pa · pa.case 만 다르다 (§7.2). run 이름은 서버 접미사 없이(시트가 server.txt 로 구분) PA_<case>_<loss>_W96_D124_9CH_S<seed>.
서버-seed block (§8.1, B0 seed 에 맞춤): s1 2025 A1→A2→A3 · s2 1234 A2→A3→A1 · s3 7777 A3→A1→A2. s2/s3 큐는 자기 block 의 B0 를 먼저 돈다 (§8.3).
"""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = {"A1": ("REC", 0.0, 0.0), "A2": ("OUTEDGE", 0.1, 0.0), "A3": ("GEO", 0.0, 0.01)}
BLOCKS = {"s1": (2025, ["A1", "A2", "A3"]), "s2": (1234, ["A2", "A3", "A1"]), "s3": (7777, ["A3", "A1", "A2"])}

def main():
    made = {}
    for server, (seed, order) in BLOCKS.items():
        tpl_p = os.path.join(ROOT, "config", f"BASE_W96_D124_MSPAN_WV3_S{seed}.yaml"); tpl = open(tpl_p).read()
        for case in order:
            name, le, lg = CASES[case]
            tag = f"PA_{case}_{name}_W96_D124_9CH_S{seed}"
            t = re.sub(r"^(#.*\n)+", "", tpl)
            head = (f"# PA {case} — PAN 앞단 전역 정합, seed {seed} (block {server}). 생성: tools/gen_pa_configs.py (템플릿 BASE_W96_D124_MSPAN_WV3_S{seed}). 손으로 고치지 말 것.\n"
                    f"# 명세: research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md · 구현 train_pa.py / pa/\n"
                    f"# B0 와 같은 골격·입력 9ch·mars ms·γβ 제거·50K AdamW 1e-4/wd0.01 cosine · aligner(0.105M) 는 같은 LR 의 param group\n"
                    f"# {case}: L_rec" + (" + λE·L_edge (0.1, 5K ramp)" if le else "") + (" + λG·L_geo (0.01, 5K ramp)" if lg else "") + "\n"
                    f"# best_hqnr = best_raw(raw_original HQNR → fSCC → later step) · best_aligned · last 보존. 시트 = raw_original\n")
            t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
            t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                          "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\n"
                          f"trainer: pa\npa:\n  case: {case}\n  lambda_edge: {le}\n  lambda_geo: {lg}\n  ramp_steps: 5000\n  geometry_sigma_hr: 2.0     # r/2\n"
                          "  geometry_margin_hr: 11     # §6.2 (64px patch 에서 [11:53])\n  init_dir: work_dir/_pa_init   # seed 별 init_unet/aligner .pt (§7.3)\n  diag_iter: 500\n")
            t = re.sub(r"expect_params_m: [\d.]+", "expect_params_m: 2.123     # backbone 만 (aligner 0.1053M 은 smoke 가 따로 확인)", t)
            assert "trainer: pa" in t and f"case: {case}" in t and f"seed: {seed}" in t
            open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t)
            made.setdefault(server, []).append(tag)
        q = os.path.join(ROOT, "config", "queues", f"pa_{server}.txt")
        with open(q, "w") as f:
            f.write(f"# PA A1–A3 block {server} (seed {seed}, 순서 {'→'.join(order)}). 명세 §8.1. s2/s3 는 자기 block 의 B0 를 먼저 돈다.\n")
            if server != "s1":
                f.write(f"BASE_W96_D124_MSPAN_WV3_S{seed}\n")
            f.write("\n".join(made[server]) + "\n")
        print(server, seed, made[server], "->", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
