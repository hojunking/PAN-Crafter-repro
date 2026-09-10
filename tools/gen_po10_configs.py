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
    ap = argparse.ArgumentParser(); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--radius", type=float, default=1.0, help="b (HR px). 1.0 = R100 (train P90 proxy) · 2.0 = R200_FRSTAT (변경 명세 2026-09-10)")
    ap.add_argument("--profile", default=None, help="R100_TRAINP90 | R200_FRSTAT (기본: radius 로 결정)")
    ap.add_argument("--width", type=int, default=96); ap.add_argument("--depth", default="1,2,4", help="예: 1,2,3")
    ap.add_argument("--params", type=float, default=None, help="expect_params_m (backbone). 없으면 모델을 만들어 잰다")
    a = ap.parse_args()
    depth = [int(x) for x in a.depth.split(",")]; arch = f"W{a.width}_D{''.join(map(str, depth))}"
    if a.params is None:
        import sys, yaml; sys.path.insert(0, ROOT)
        from main import import_class
        _c = yaml.safe_load(open(os.path.join(ROOT, "config", f"PA_A1_REC_W96_D124_9CH_S{a.seed}.yaml")))
        _m = import_class(_c["model"])(**dict(_c["model_args"], hidden_size=a.width, depth=depth)); a.params = round(sum(p.numel() for p in _m.parameters()) / 1e6, 4)
    init_dir = "work_dir/_pa_init" if arch == "W96_D124" else f"work_dir/_pa_init_{arch.lower()}"      # 골격이 다르면 init snapshot 도 별도 (shape 가 다르다)
    tpl = open(os.path.join(ROOT, "config", f"PA_A1_REC_W96_D124_9CH_S{a.seed}.yaml")).read()
    profile = a.profile or ("R100_TRAINP90" if a.radius == 1.0 else f"R{int(round(a.radius * 100)):03d}_FRSTAT")
    sfx = "" if a.updates == 50000 else f"_SCREEN{a.updates // 1000}K"
    if profile != "R100_TRAINP90":
        sfx = f"_{profile}" + sfx
    if profile == "R100_TRAINP90":
        prov = ("  protocol_id: pan_offset_rel_10h_v1\n  radius_profile: R100_TRAINP90\n  radius_source: 'audit appendix E WV3 train global |δ| P90 0.250 LR px × 4'\n")
    else:
        prov = ("  protocol_id: pan_offset_rel_10h_v2_wv3_r200_frstat\n  radius_profile: R200_FRSTAT\n"
                "  radius_source: 'results_log/2026-09-10_pa-a1-a3-s1-results.md §7 WV3 FR |δ| P95 1.98 HR px (FR-input-statistics-informed development)'\n"
                "  radius_provenance:\n    dataset: WV3\n    profile: FRSTAT_R200\n    selected_radius_hr: " + str(a.radius) + "\n    observed_statistic: report_abs_delta_P95\n    observed_value_hr: 1.98\n"
                "    source_split: FR_paper_mat20\n    source_scene_count: 20\n    report_file: 2026-09-10_pa-a1-a3-s1-results.md\n    report_section: 7\n    reference_pair: PAN_b_from_bicubic_upsampled_MS\n"
                "    calibration_quality: fr_input_statistics_informed_development\n    uses_evaluation_input_statistics: true\n    uses_FR_HRMS_ground_truth: false\n"
                "    selection_note: nearby_execution_scale_chosen_before_new_run\n    radius_is_trainable: false\n    radius_changes_within_pure_run: false\n    runtime_recalibration: false\n"
                "  training_regime: restart_pure\n")
    made = []
    for case, name in CASES:
        tag = f"PO10_{name}_{arch}_WV3_S{a.seed}{sfx}"
        t = re.sub(r"^(#.*\n)+", "", tpl)
        loss_desc = {"N1": "L_rec 만 (offset loss 계수 0)", "N2_SG": "L_rec + 0.01·|ĉε+ε−sg(ĉ0)| (5K ramp)", "N3_NOSG": "L_rec + 0.01·|ĉε+ε−ĉ0| (양쪽 gradient, 5K ramp)"}[case]
        head = (f"# PO10 {case} [{profile}, b={a.radius} HR px] — PAN 추가 변위 + offset consistency, seed {a.seed}, {a.updates} updates. 생성: tools/gen_po10_configs.py (템플릿 PA_A1_REC_W96_D124_9CH_S{a.seed}). 손으로 고치지 말 것.\n"
                + ("# 변경 명세: research_log/PAN_OffsetConsistency_ChangeNote_R100_to_R200_2026-09-10.md — 반경 1.0→2.0, 근거는 FR 논문 세트 입력 통계(P95 1.98). 옛 R100 run 과 별개의 새 독립 학습\n" if profile != "R100_TRAINP90" else "")
                + "# 명세: research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md · 구현 train_po.py / pa/offset.py\n"
                + f"# 골격 {arch}(hidden {a.width}, depth {depth}) · 9ch·mars ms·γβ 제거·AdamW 1e-4/wd0.01 cosine·batch 48·init({init_dir} seed {a.seed}) · aligner 고정 내부 view margin 4\n"
                + ("# 2026-09-10 사용자 결정: R200 부터 골격을 W112·D123 으로 — W96·D124 B0/A1/R100 과 골격이 다르므로 그 run 들과의 직접 대응 비교는 하지 않는다 (block 안 N2−N1, N3−N2 만)\n" if arch != "W96_D124" else "")
                + f"# native/corrupt 1:1 (update 짝/홀). corrupt: ε ~ disk b={a.radius} HR px. {loss_desc}\n"
                + "# best_hqnr = best_raw(raw_original HQNR → fSCC → later step) · best_aligned · last 보존. 시트 = raw_original(HQNR↑) + V64(HQNR(V64)↑)\n")
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = re.sub(r"^trainer: pa\npa:\n(  .*\n)+", "", t, flags=re.M)
        t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                      "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\n"
                      f"trainer: po\npo:\n  case: {case}\n  radius_hr: {a.radius}                 # b (HR px), run 안에서 고정; ε 만 sample 마다 뽑는다\n" + prov
                      + f"  lambda_off_max: {0.0 if case == 'N1' else 0.01}\n  ramp_updates: 5000\n  diag_every: 1000\n  corruption_seed_offset: 2000\n"
                      + f"  budget_total_gpu_hours: 10.0\n  budget_diag_reserve_hours: 1.0\n  init_dir: {init_dir}\n")
        t = re.sub(r"^num_iter: \d+", f"num_iter: {a.updates}", t, flags=re.M)
        t = re.sub(r"^  hidden_size: \d+", f"  hidden_size: {a.width}", t, flags=re.M)
        t = re.sub(r"^  depth: \[.*?\]", f"  depth: {depth}", t, flags=re.M)
        t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {a.params}", t)
        assert f"hidden_size: {a.width}" in t and f"depth: {depth}" in t and f"expect_params_m: {a.params}" in t
        assert "trainer: po" in t and f"case: {case}" in t and "trainer: pa" not in t and f"num_iter: {a.updates}" in t and f"radius_hr: {a.radius}" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t); made.append(tag)
    q = os.path.join(ROOT, "config", "queues", f"po10_s1{sfx.lower()}{'' if arch == 'W96_D124' else '_' + arch.lower()}.txt")
    with open(q, "w") as f:
        f.write(f"# PO10 offset consistency [{profile}, b={a.radius}, {arch}], s1 seed {a.seed}, {a.updates} updates. N1·N2 필수, N3 는 예산 gate(train_po._budget_gate) 통과 시. 명세 §11\n" + "\n".join(made) + "\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))

if __name__ == "__main__":
    main()
