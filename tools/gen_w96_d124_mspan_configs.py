#!/usr/bin/env python
"""새 baseline(research_log/PAN_research_baseline_W96_D124_2026-09-09.md) WV3 3-seed config 생성 — s3 용.

    python tools/gen_w96_d124_mspan_configs.py     # config/BASE_W96_D124_MSPAN_WV3_S{2025,1234,7777}.yaml + queue

고정 기준(문서 §1·§3.1): U-Net W96 · depth [1,2,4] · 입력 MS+PAN 만(9ch = concat(PAN, ↑MS), in_mode: paper) ·
LPAN/HPAN 채널 없음 · PAN reconstruction task·loss 없음(mars: ms — batch 복제·PAN forward 자체가 없다, 계수 0 이 아님) ·
attention 없음 · crop=False · bicubic 잔차 base · M-frame 출력 · 50K AdamW 1e-4/wd0.01 cosine(기존 실행 조건 유지).
MARs 의 mode 조건 γ/β(mode_modulation)는 **끈다** — 단일 mode 에서는 상수 affine 이라 LN 의 affine 과 중복이고,
문서가 'dual MARs 제거' 라 했으므로 파라미터째 뺀다. 되돌리려면 아래 MODE_MOD 만 true 로.
best 선택·보고 = 논문 세트(.mat 20장 전체) 공식 HQNR (fr_select_indices 0-19).
템플릿은 config/MS1_w96_9ch_msonly.yaml (같은 골격·9ch·mars ms 의 기존 run).
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
TPL = os.path.join(ROOT, "config", "MS1_w96_9ch_msonly.yaml")
SEEDS = (2025, 1234, 7777)
MODE_MOD = "false"

def params_m():
    import yaml, torch
    from main import import_class
    cfg = yaml.safe_load(open(TPL)); ma = dict(cfg["model_args"], mode_modulation=(MODE_MOD == "true"))
    m = import_class(cfg["model"])(**ma)
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 4)

def main():
    pm = params_m(); tpl = open(TPL).read(); made = []
    for seed in SEEDS:
        tag = f"BASE_W96_D124_MSPAN_WV3_S{seed}"
        t = re.sub(r"^(#.*\n)+", "", tpl)
        head = (f"# 새 baseline W96·D124 · MS+PAN 9ch · 단일 HRMS task — WV3 seed {seed} (s3 3-seed)\n"
                f"# 생성: tools/gen_w96_d124_mspan_configs.py (템플릿 MS1_w96_9ch_msonly). 손으로 고치지 말 것.\n"
                f"# 기준 문서: research_log/PAN_research_baseline_W96_D124_2026-09-09.md §1·§3\n"
                f"# 공통: W96 · depth [1,2,4] · in_mode paper(9ch: PAN, ↑MS) · LPAN/HPAN 없음 · mars ms(PAN task·loss·batch 복제 없음)\n"
                f"#       mode_modulation {MODE_MOD}(MARs γ/β 제거) · attention 없음 · crop=False · 50K AdamW 1e-4/wd0.01 cosine\n"
                f"#       best 선택 = 공식 HQNR, 논문 세트(.mat 20장 전체, full_examples_mat20) · 보고도 같은 세트\n")
        t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {pm}", t)
        t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = t.replace("data/PanCollection/WV3/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5",
                      "data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5")
        t = t.replace("select_on: hqnr", "select_on: hqnr\nfr_select_indices: \"0-19\"   # 논문 세트 20장 전체로 선택 (부분집합 없음)")
        t = t.replace("  attn_locations: []", f"  attn_locations: []\n  mode_modulation: {MODE_MOD}   # MARs mode 조건 γ/β 제거 (단일 task)")
        assert "mars: ms" in t and "in_mode: paper" in t and "hidden_size: 96" in t and "depth: [1, 2, 4]" in t and "mat20" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t); made.append(tag)
    q = os.path.join(ROOT, "config", "queues", "base_w96_d124_mspan_wv3_3seed.txt")
    with open(q, "w") as f:
        f.write("# 새 baseline W96·D124·MS+PAN 9ch·단일 task — WV3 3-seed (s3). 선택·보고 = 논문 세트 20장 전체.\n" + "\n".join(made) + "\n")
    print(f"params {pm} M"); print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
