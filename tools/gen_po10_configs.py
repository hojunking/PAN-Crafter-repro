#!/usr/bin/env python
"""PO10 config 생성 (research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md §11.3·§14).

    python tools/gen_po10_configs.py [--updates 50000]   # config/PO10_{N1_REC,N2_OFFSG,N3_OFFNOSG}_W96_D124_WV3_S2025.yaml + config/queues/po10_s1.txt

템플릿 = A1 config (config/PA_A1_REC_W96_D124_9CH_S2025.yaml) — 골격·입력·optimizer·50K·seed 2025·init 파일(work_dir/_pa_init) 그대로, trainer 만 po.
--updates 25000 이면 SCREEN25K 접미사 (§11.2, N1/N2 pair 가 예산에 안 들어갈 때만).
"""
import argparse, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("N1", "N1_REC"), ("N2_SG", "N2_OFFSG"), ("N3_NOSG", "N3_OFFNOSG")]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--seed", type=int, default=2025); a = ap.parse_args()
    tpl = open(os.path.join(ROOT, "config", f"PA_A1_REC_W96_D124_9CH_S{a.seed}.yaml")).read()
    sfx = "" if a.updates == 50000 else f"_SCREEN{a.updates // 1000}K"
    made = []
    for case, name in CASES:
        tag = f"PO10_{name}_W96_D124_WV3_S{a.seed}{sfx}"
        t = re.sub(r"^(#.*\n)+", "", tpl)
        head = (f"# PO10 {case} — PAN 추가 변위 + offset consistency, seed {a.seed}, {a.updates} updates. 생성: tools/gen_po10_configs.py (템플릿 PA_A1_REC_W96_D124_9CH_S{a.seed}). 손으로 고치지 말 것.\n"
                f"# 명세: research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md · 구현 train_po.py / pa/offset.py\n"
                f"# A1 과 같은 골격·9ch·mars ms·γβ 제거·AdamW 1e-4/wd0.01 cosine·batch 48·init(work_dir/_pa_init seed {a.seed}) · aligner 고정 내부 view margin 4\n"
                f"# native/corrupt 1:1 (update 짝/홀). corrupt: ε ~ disk R=1.0 HR px (audit 부록 E WV3 train P90 0.250 LR px × 4). "
                + {"N1": "L_rec 만 (offset loss 계수 0)", "N2_SG": "L_rec + 0.01·|ĉε+ε−sg(ĉ0)| (5K ramp)", "N3_NOSG": "L_rec + 0.01·|ĉε+ε−ĉ0| (양쪽 gradient, 5K ramp)"}[case] + "\n"
                f"# best_hqnr = best_raw(raw_original HQNR → fSCC → later step) · best_aligned · last 보존. 시트 = raw_original\n")
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = re.sub(r"^trainer: pa\npa:\n(  .*\n)+", "", t, flags=re.M)
        t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                      "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\n"
                      f"trainer: po\npo:\n  case: {case}\n  radius_hr: 1.0                 # audit 부록 E: WV3 train 전역 |δ| P90 0.250 LR px × 4 (train_aggregate_proxy)\n"
                      f"  lambda_off_max: {0.0 if case == 'N1' else 0.01}\n  ramp_updates: 5000\n  diag_every: 1000\n  corruption_seed_offset: 2000\n"
                      "  budget_total_gpu_hours: 10.0\n  budget_diag_reserve_hours: 1.0\n  init_dir: work_dir/_pa_init\n")
        t = re.sub(r"^num_iter: \d+", f"num_iter: {a.updates}", t, flags=re.M)
        assert "trainer: po" in t and f"case: {case}" in t and "trainer: pa" not in t and f"num_iter: {a.updates}" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t); made.append(tag)
    q = os.path.join(ROOT, "config", "queues", f"po10_s1{sfx.lower()}.txt")
    with open(q, "w") as f:
        f.write(f"# PO10 offset consistency, s1 seed {a.seed}, {a.updates} updates. N1·N2 필수, N3 는 예산 gate(train_po._budget_gate) 통과 시. 명세 §11\n" + "\n".join(made) + "\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
