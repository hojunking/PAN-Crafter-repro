#!/usr/bin/env python
"""NF16 config 생성 (research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md §3·§13·§14).

    python tools/gen_nf16_configs.py [--updates 50000] [--version dry]      # config/NF16_P{0..4}_W112_D123_WV3_S1234_N2LAST_v1.yaml + P3/P4 S7777 + config/queues/nf16_s1.txt

trainer 는 kdv (train_kdv.py). donor = PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT 의 정확한 50K last 의 aligner.* (내부 view margin 4, strict) — U-Net 은 seed 별 저장 초기값에서 새로.
  P0 NOALIGN  : A-ID (aligner·sampler 없음), I-A, L_rec
  P1 FROZEN   : A-FR donor, I-A, L_rec
  P2 FT-REC   : A-FT donor (aligner LR 1e-5), I-A, L_rec
  P3 FT-EQ    : A-FT, I-AEQ (홀수 update 에 P_ε 를 aligner 에만; |ĉε+ε−sg ĉ0|, λ_off 0.01 즉시, b=2 원판), L_rec
  P4 FT-EQ-GEO: P3 + A3 geometry (λ 0.01, 5K ramp, native P̃0 vs GT, aligner 에만 gradient)
평가: 세 view + aligned_fixedN2_v64 (고정 donor 참조) + best_rr_val(valid_wv3.h5). 예산 ledger work_dir/_nf16_budget/ledger.json (16 GPU-h)."""
import argparse, os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe

DONOR_RUN = "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT"
# P1/P2: N2 donor 를 corruption 없이 쓰므로 계획(S0 §5.4) 용어로 I-NATIVE-TRANSFER (native 입력, offset 계수 0). P3/P4: I-AEQ (aligner 전용 offset 연습)
CASES = {0: ("NOALIGN", "A-ID", "I-A", False, False), 1: ("FROZEN", "A-FR", "I-NATIVE-TRANSFER", False, False), 2: ("FT-REC", "A-FT", "I-NATIVE-TRANSFER", False, False), 3: ("FT-EQ", "A-FT", "I-AEQ", True, False), 4: ("FT-EQ-GEO", "A-FT", "I-AEQ", True, True)}
RESERVED_HOURS = {0: 1.6, 1: 1.6, 2: 1.8, 3: 2.0, 4: 2.0}      # 명세 §10 예약 표 (P0 1.6 · P1 1.6 · P2 1.8 · P3 2.0 · P4 2.0)
PURPOSE = {0: "P0 NOALIGN — 같은 block 의 W112·D123 기준 (aligner·sampler 없음)", 1: "P1 FROZEN — N2 last aligner 고정 재사용 (P1−P0: 재사용 효과)", 2: "P2 FT-REC — aligner 를 L_rec 로만 공동 미세조정 (P2−P1: 망각)",
           3: "P3 FT-EQ — native 복원 + aligner 전용 offset 연습 (P3−P2: 반응 보존)", 4: "P4 FT-EQ-GEO — P3 + native PAN–GT 구조 감독 (P4−P3: 위치 기준)"}


def build(q, seed, donor_sha, donor_step, required, projected, projected_map=None, ledger="work_dir/_nf16_budget/ledger.json", total_hours=16.0):
    name, pol, proto, off, geo = CASES[q]
    k = dict(campaign_id="NF16_W112D123_N2LAST_v1", run_kind="CONTROLLED", version="v1", check_run_name=False, input_protocol=proto, aligner_policy=pol, diag_every=1000,
             calibration=dict(n_patches=3072, seed=1234), aligner_lr=1.0e-5, rec=dict(case="N0"), stat=dict(enabled=False), geom_kd=dict(mode="G0"),
             eval=(dict(fixed_reference_from_donor=True) if pol != "A-ID" else dict(reference_donor=dict(source=f"work_dir/{DONOR_RUN}/last", view_margin_hr=4, expected_sha256=donor_sha, expected_step=donor_step))),
             budget=dict(ledger=ledger, total_gpu_hours=total_hours, reserve_hours=1.0, required=required, projected_hours=projected))
    k["budget"]["projected_map"] = dict(projected_map or {})                       # 남은 필수·pair run 의 예상 (case 별 예약, 검토 지적 2)
    if pol == "A-ID":
        k["recipe"] = "NOALIGN"
    else:
        k["recipe"] = "N2_SG"                                                # donor 의 recipe (N2 SG); 계승할 항은 aux 에서 명시
        k["donor"] = dict(source=f"work_dir/{DONOR_RUN}/last", view_margin_hr=4, expected_sha256=donor_sha, expected_step=donor_step)
    if proto == "I-AEQ":
        k["corruption"] = dict(radius_hr=2.0, corruption_seed_offset=2000)
        k["aux"] = dict(offset_weight=0.01, offset_ramp_updates=0, offset_stop_reference=True, geometry_weight=(0.01 if geo else 0.0), ramp_updates=5000, geometry_sigma_hr=2.0, geometry_margin_hr=11)
    elif pol != "A-ID":
        k["aux"] = dict(offset_weight=0.0, geometry_weight=0.0)             # P1/P2: L_rec 만
    return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--version", default="v1"); ap.add_argument("--only", default=None)
    ap.add_argument("--projected-hours", type=float, default=None, help="예산 gate 용 run 당 예상 GPU 시간 (smoke 로 잰 값; 없으면 ledger 의 완료 run 평균)")
    ap.add_argument("--ledger", default="work_dir/_nf16_budget/ledger.json"); ap.add_argument("--total-hours", type=float, default=16.0)
    a = ap.parse_args()
    import yaml
    from kdv.teacher_assets import sha256_file
    dp = os.path.join(ROOT, "work_dir", DONOR_RUN, "last", "model.safetensors"); donor_sha = sha256_file(dp) if os.path.exists(dp) else None
    dm = os.path.join(ROOT, "work_dir", DONOR_RUN, "last_meta.json"); donor_step = json.load(open(dm))["step"] if os.path.exists(dm) else 50000
    tpl = open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml")).read()      # W112·D123 골격·optimizer·평가 그대로
    cells = [(q, 1234) for q in range(5)] + [(3, 7777), (4, 7777)]
    only = [int(x) for x in a.only.split(",")] if a.only else None
    made = []
    names = {(q, seed): f"NF16_P{q}_W112_D123_WV3_S{seed}_N2LAST_{a.version}" for q, seed in cells}
    pmap = {names[(q, seed)]: RESERVED_HOURS[q] for q, seed in cells}
    core = [(q, 1234) for q in range(5)]                                     # 필수 사슬 P0→P4 (s1234); 반복 pair 는 조건부
    for q, seed in cells:
        if only is not None and q not in only:
            continue
        tag = names[(q, seed)]
        required = (seed == 1234); projected = a.projected_hours              # 명세 §10: 필수 case 의 학습량은 유지(경고만), 반복 pair 는 gate
        k = build(q, seed, donor_sha, donor_step, required, projected, pmap, a.ledger, a.total_hours)
        if seed == 1234:
            k["budget"]["remaining_mandatory"] = [names[c] for c in core[core.index((q, 1234)) + 1:]]     # 남은 필수 run 전체를 예상에 포함 (검토 지적 2)
        else:
            k["budget"]["pair_with"] = names[(4 if q == 3 else 3, 7777)]                            # 반복 pair 는 한 묶음 — 양쪽 모두 상대를 예상에 넣는다 (끝난 상대는 _projection 이 0 으로 본다, 검토 지적 2)
            k["budget"]["remaining_mandatory"] = []
        sp = resolve(k)
        t = re.sub(r"^(#.*\n)+", "", tpl)
        head = (f"# {tag} — {PURPOSE[q]}. 생성: tools/gen_nf16_configs.py. 손으로 고치지 말 것.\n"
                f"# 명세: research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md · 구현 train_kdv.py(trainer kdv, protocol I-AEQ) · 노트 research_log/2026-09-11_nf16-implementation.md\n"
                f"# 세팅: {describe(sp)}\n"
                f"# 골격 W112·D123(2.6589 M) · 9ch · 단일 HRMS · AdamW 1e-4(backbone)/1e-5(aligner)/wd 0.01 cosine warmup100 · batch 48 · {a.updates} updates · seed {seed} · init work_dir/_kdv_init_w112_d123\n"
                f"# donor aligner = {DONOR_RUN}/last (step {donor_step}, file sha256 {(donor_sha or '?')[:16]}…), 내부 view margin 4, strict load, head 재초기화 없음, donor optimizer state 미사용\n"
                f"# 평가: raw_original · raw_valid(V64) · aligned_valid(self, V64) · aligned_fixed_v64(고정 donor 참조) · best_raw=best_hqnr · best_aligned · best_rr_val · last(정확한 {a.updates})\n")
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = re.sub(r"^trainer: po\npo:\n(  .*\n)+", "", t, flags=re.M)
        t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                      "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n" + "\n".join("  " + l for l in yaml.safe_dump(k, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n")
        t = re.sub(r"^num_iter: \d+", f"num_iter: {a.updates}", t, flags=re.M); t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
        assert "trainer: kdv" in t and "trainer: po" not in t and f"seed: {seed}" in t and "hidden_size: 112" in t and "depth: [1, 2, 3]" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t); made.append(tag)
    q = os.path.join(ROOT, "config", "queues", f"nf16_s1{'' if a.version == 'v1' else '_' + a.version}.txt")
    with open(q, "w") as f:
        f.write(f"# NF16 (명세 §10 순서): gate → P0 → P1 → P2 → P3 → P4 → (P3, P4 seed 7777 반복; 예산 gate) · s1 · {a.updates} updates · 예산 16 GPU-h (work_dir/_nf16_budget/ledger.json)\n"
                "# s1234 사슬 P0–P4 는 전부 required(초과해도 경고만 — 명세 §10 '필수 case 학습량 유지'), 반복 pair(s7777) 만 예산으로 막힌다(DEFERRED_BUDGET, exit 4 → 체인은 다음으로 넘어간다).\n"
                "# gate 식: used + 1.2·(이 run + 남은 필수 사슬 + pair) + reserve 1h ≤ 16h. 예상치는 config projected_hours → budget.projected_map(§10 예약 1.6/1.6/1.8/2.0/2.0) → 완료 run 평균 → smoke throughput.\n"
                "# P4 geometry gate 실패 시 반복 pair 는 P1/P3 로 바꾼다(수동)\n" + "\n".join(made) + "\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT), "| donor step", donor_step, "sha", (donor_sha or "?")[:16])


if __name__ == "__main__":
    main()
