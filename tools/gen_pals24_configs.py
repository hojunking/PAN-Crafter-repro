#!/usr/bin/env python
"""PALS24 config 생성 — P2 기반 offset-consistency 가중치(λ_off) 비교, 24 GPU-h (research_log/PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md).

    python tools/gen_pals24_configs.py                      # stage 1: seed 1234 신규 λ 3벌 (A1 L1E3 → A2 L1E4 → A3 L3E3) + 큐 config/queues/pals24_s1.txt
    python tools/gen_pals24_configs.py --stage2 --lambda-star L1E3   # stage 2 (λ* 확정 뒤): seed 7777 (P0 → L000 → λ*) + seed 2025 (L000 → λ* → P0)

trainer 는 kdv (train_kdv.py) 위, NF16 과 같은 골격·donor·평가 (tools/gen_nf16_configs.py 의 P0/P2/P3 recipe 를 λ 만 바꿔 그대로).
약명 → 세팅 (계획 §2.2; 시트·문서에는 이 표로만 읽는다)
  CTRLP0 : W112·D123, aligner·sampler 없음 (A-ID, I-A)                      = NF16 P0 정의
  L000   : λ_off 0     — A-FT donor aligner + native L_rec 만 (I-NATIVE-TRANSFER)  = NF16 P2 정의
  L1E4   : λ_off 0.0001 — A-FT, I-AEQ (홀수 update 에 P_ε 를 aligner 에만; |ĉε+ε−sg ĉ0|, ramp 0, b=2 원판)
  L1E3   : λ_off 0.001  — 같음
  L3E3   : λ_off 0.003  — 같음
  L1E2   : λ_off 0.01   — 같음                                              = NF16 P3 정의 (seed 1234 는 NF16 P3 재사용)
run 이름: PALS24_<case>_W112_D123_WV3_S<seed>_N2LAST_R200_v1. seed 1234 의 CTRLP0/L000/L1E2 는 NF16 P0/P2/P3 를 재사용 gate(tools/pals24_metric_gate.py) 를 거쳐 쓴다 — 다시 학습하지 않는다.
예산: work_dir/_pals24_budget/ledger.json (24 GPU-h) — gate 식 used + 1.1·(이 run + 같은 block 의 나머지) + reserve 4.0(최종 평가 3.0 + buffer 1.0) ≤ 24, required 아님(초과면 DEFERRED_BUDGET, exit 4)."""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe

CAMPAIGN_ID = "PALS24_W112D123_N2LAST_R200_v2"; PROTOCOL_ID = "PALS_W112D123_N2LAST_R200_v1"; DOC_REV = "24GPUh_v2"
PLAN = "research_log/PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md"; NOTE = "research_log/2026-09-12_pals24-implementation.md"
DONOR_RUN = "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT"
LEDGER = "work_dir/_pals24_budget/ledger.json"; TOTAL_HOURS = 24.0; RESERVE_HOURS = 4.0; MARGIN = 1.1; RUN_RESERVED_HOURS = 2.0
LAMBDA = {"L000": 0.0, "L1E4": 1e-4, "L1E3": 1e-3, "L3E3": 3e-3, "L1E2": 1e-2, "L3E5": 3e-5, "L3E4": 3e-4}      # L3E5/L3E4: PALSV18 (2026-09-13) 의 L1E4 근방 미세조정
CAMPAIGN_DEFAULT = None                                                                                  # (아래에서 채운다) 캠페인 상수 묶음 — PALSV18 은 gen_palsv18_configs.py 가 덮어쓴다
NEW_LAMBDAS = ["L1E3", "L1E4", "L3E3"]                                    # §11.5 A1 → A2 → A3
SCREEN_SEED = 1234; CONFIRM_SEEDS = [7777, 2025]
NF16 = {q: f"NF16_P{q}_W112_D123_WV3_S1234_N2LAST_v1" for q in range(5)}
REUSED = {("CTRLP0", 1234): NF16[0], ("L000", 1234): NF16[2], ("L1E2", 1234): NF16[3]}        # §11.2 재사용 (gate 통과 조건)
BACKGROUND = {("L1E2", 7777): "NF16_P3_W112_D123_WV3_S7777_N2LAST_v1"}                          # 배경 대조로만 (§11.2)
DIAG_REFERENCE = {("P1", 1234): NF16[1]}                                                        # G-M4 (frozen donor: self == fixedN2)
PURPOSE = {"CTRLP0": "CTRL-P0 — aligner·sampler 없는 no-align 대조 (NF16 P0 정의)", "L000": "L000 — λ_off 0: P2 와 같은 recon-only 공동 학습 (NF16 P2 정의)",
           "L1E4": "L1E4 — λ_off 0.0001 (기존 P3 의 1/100)", "L1E3": "L1E3 — λ_off 0.001 (기존 P3 의 1/10)", "L3E3": "L3E3 — λ_off 0.003 (기존 P3 의 0.3배)", "L1E2": "L1E2 — λ_off 0.01 (NF16 P3 와 같은 감독 강도)",
           "L3E5": "L3E5 — λ_off 0.00003 (L1E4 의 0.3배; PALSV18 근방 미세조정)", "L3E4": "L3E4 — λ_off 0.0003 (L1E4 의 3배; PALSV18 근방 미세조정)"}
CAMPAIGN_DEFAULT = dict(prefix="PALS24", campaign_id=CAMPAIGN_ID, plan_protocol_id=PROTOCOL_ID, document_revision=DOC_REV, ledger=LEDGER, total_hours=TOTAL_HOURS, reserve_hours=RESERVE_HOURS,
                        run_reserved=RUN_RESERVED_HOURS, margin=MARGIN, plan=PLAN, note=NOTE)


def campaign(camp=None):
    return dict(CAMPAIGN_DEFAULT, **(camp or {}))


def run_name(case, seed, version="v1", prefix="PALS24"):
    return f"{prefix}_{case}_W112_D123_WV3_S{seed}_N2LAST_R200_{version}"


def stage1():
    return [(c, SCREEN_SEED) for c in NEW_LAMBDAS]


def stage2(lam_star):
    assert lam_star in NEW_LAMBDAS, lam_star
    return [("CTRLP0", 7777), ("L000", 7777), (lam_star, 7777), ("L000", 2025), (lam_star, 2025), ("CTRLP0", 2025)]     # §11.5 B1–B3, C1–C3


def blocks(lam_star=None):
    """block 별 구성원 — 예산 gate 는 block 전체(P0·L000·λ*) 가 완결 가능할 때만 연다 (§11.6)."""
    b = {"A": stage1()}
    if lam_star:
        s2 = stage2(lam_star); b["B"] = [c for c in s2 if c[1] == 7777]; b["C"] = [c for c in s2 if c[1] == 2025]
    return b


def kdv_block(case, seed, donor_sha, donor_step, members, projected=None, version="v1", diag_every=1000, ledger=None, camp=None):
    """NF16 build() 와 같은 recipe (P0 / P2 / P3) 에 λ 만 바꾼다. members: 같은 block 의 (case, seed) 목록 (예산 gate 용). camp: 캠페인 상수 덮어쓰기 (PALSV18)."""
    C = campaign(camp); ledger = ledger or C["ledger"]; lam = LAMBDA.get(case)
    if case == "CTRLP0":
        pol, proto = "A-ID", "I-A"
    elif lam == 0.0:
        pol, proto = "A-FT", "I-NATIVE-TRANSFER"
    else:
        pol, proto = "A-FT", "I-AEQ"
    donor = dict(source=f"work_dir/{DONOR_RUN}/last", view_margin_hr=4, expected_sha256=donor_sha, expected_step=donor_step)
    k = dict(campaign_id=C["campaign_id"], plan_protocol_id=C["plan_protocol_id"], document_revision=C["document_revision"], case_id=case, run_kind="CONTROLLED", version=version, check_run_name=False,
             input_protocol=proto, aligner_policy=pol, diag_every=diag_every, calibration=dict(n_patches=3072, seed=1234), aligner_lr=1.0e-5,
             rec=dict(case="N0"), stat=dict(enabled=False), geom_kd=dict(mode="G0"),
             eval=(dict(fixed_reference_from_donor=True) if pol != "A-ID" else dict(reference_donor=dict(donor))))
    me = run_name(case, seed, version, C["prefix"])
    k["budget"] = dict(ledger=ledger, total_gpu_hours=C["total_hours"], reserve_hours=C["reserve_hours"], margin=C["margin"], required=False, projected_hours=projected,
                       projected_map={run_name(c, s, version, C["prefix"]): C["run_reserved"] for c, s in members},
                       remaining_mandatory=[run_name(c, s, version, C["prefix"]) for c, s in members if run_name(c, s, version, C["prefix"]) != me])
    if pol == "A-ID":
        k["recipe"] = "NOALIGN"
    else:
        k["recipe"] = "N2_SG"; k["donor"] = dict(donor)
    if proto == "I-AEQ":
        k["corruption"] = dict(radius_hr=2.0, corruption_seed_offset=2000)
        k["aux"] = dict(offset_weight=float(lam), offset_ramp_updates=0, offset_stop_reference=True, geometry_weight=0.0, ramp_updates=5000, geometry_sigma_hr=2.0, geometry_margin_hr=11)
    elif pol != "A-ID":
        k["aux"] = dict(offset_weight=0.0, geometry_weight=0.0)
    return k


def render(tag, case, seed, k, updates, eval_epoch, tpl, donor_sha, donor_step, camp=None):
    import yaml
    C = campaign(camp); sp = resolve(k)
    t = re.sub(r"^(#.*\n)+", "", tpl)
    head = (f"# {tag} — {PURPOSE[case]}. 생성: tools/gen_{C['prefix'].lower()}_configs.py. 손으로 고치지 말 것.\n"
            f"# 캠페인 {C['campaign_id']} · protocol {C['plan_protocol_id']} · 계획 {C['plan']} · 노트 {C['note']}\n"
            f"# 세팅: {describe(sp)} · λ_off {LAMBDA.get(case, 'n/a')} (config kdv.aux.offset_weight; ramp 없음, 홀수 update 만, L_rec 계수 1)\n"
            f"# 약명→세팅: CTRLP0 = aligner 없음(NF16 P0 정의) · L000 = λ 0(NF16 P2 정의) · L3E5/L1E4/L3E4/L1E3/L3E3 = λ 3e-5/1e-4/3e-4/1e-3/3e-3 · L1E2 = λ 0.01(NF16 P3 정의)\n"
            f"# 골격 W112·D123(2.6589 M) · 9ch · 단일 HRMS · AdamW 1e-4(backbone)/1e-5(aligner)/wd 0.01 cosine warmup100 · batch 48 · {updates} updates · seed {seed} · init work_dir/_kdv_init_w112_d123/init_unet_seed{seed}.pt\n"
            f"# donor aligner = {DONOR_RUN}/last (step {donor_step}, file sha256 {(donor_sha or '?')[:16]}…), 내부 view margin 4, strict load, head 재초기화 없음, donor U-Net·optimizer 미사용\n"
            f"# 평가: raw_original(주 판정, best_raw=best_hqnr) · raw_valid(V64) · aligned_valid(self, V64; 진단) · aligned_fixed_v64(고정 donor 참조; 진단) · best_rr_val · last(정확한 {updates})\n"
            f"# 예산: {k['budget']['ledger']} {C['total_hours']} GPU-h, gate used + {C['margin']}·(이 run + 같은 block 나머지) + reserve {C['reserve_hours']} ≤ {C['total_hours']}, 초과 시 DEFERRED_BUDGET(exit 4)\n")
    t = re.sub(r"^eval_epoch: \d+$", f"eval_epoch: {eval_epoch}", t, flags=re.M)
    t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
    t = re.sub(r"^trainer: po\npo:\n(  .*\n)+", "", t, flags=re.M)
    t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                  "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n" + "\n".join("  " + l for l in yaml.safe_dump(k, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n")
    t = re.sub(r"^num_iter: \d+", f"num_iter: {updates}", t, flags=re.M); t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
    assert "trainer: kdv" in t and "trainer: po" not in t and f"seed: {seed}" in t and "hidden_size: 112" in t and "depth: [1, 2, 3]" in t and f"eval_epoch: {eval_epoch}" in t
    return head + t


def donor_identity():
    from kdv.teacher_assets import sha256_file
    dp = os.path.join(ROOT, "work_dir", DONOR_RUN, "last", "model.safetensors"); donor_sha = sha256_file(dp) if os.path.exists(dp) else None
    dm = os.path.join(ROOT, "work_dir", DONOR_RUN, "last_meta.json"); donor_step = json.load(open(dm))["step"] if os.path.exists(dm) else 50000
    return donor_sha, donor_step


def generate(cells, members_of, out_dir, updates=50000, eval_epoch=10, projected=None, version="v1", diag_every=1000, ledger=None, camp=None):
    C = campaign(camp); donor_sha, donor_step = donor_identity()
    tpl = open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml")).read()       # W112·D123 골격·optimizer·평가 (NF16 과 같은 템플릿)
    made = []
    for case, seed in cells:
        tag = run_name(case, seed, version, C["prefix"])
        k = kdv_block(case, seed, donor_sha, donor_step, members_of[(case, seed)], projected, version, diag_every, ledger, camp)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, tag + ".yaml"), "w").write(render(tag, case, seed, k, updates, eval_epoch, tpl, donor_sha, donor_step, camp)); made.append(tag)
    return made, donor_sha, donor_step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", action="store_true"); ap.add_argument("--lambda-star", default=None, help="stage 2 의 λ* case id (L1E4/L1E3/L3E3) — tools/pals24_select_lambda.py 가 고정한 값")
    ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--eval-epoch", type=int, default=10, help="NF16 P1–P4 와 같은 평가 격자 (P0 는 5 로 돌았다 — 재사용 gate 가 10 격자 재선택도 기록)")
    ap.add_argument("--projected-hours", type=float, default=None, help="이 run 의 예상 GPU 시간 (smoke throughput); 없으면 ledger 완료 평균 → block 예약 2.0")
    ap.add_argument("--version", default="v1"); ap.add_argument("--diag-every", type=int, default=1000); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config"))
    ap.add_argument("--ledger", default=LEDGER, help="dry run 은 별도 ledger 로 (본 예산 원장을 오염시키지 않는다)")
    a = ap.parse_args()
    if a.stage2:
        assert a.lambda_star in NEW_LAMBDAS, "--stage2 는 --lambda-star {L1E4,L1E3,L3E3} 가 필요하다 (tools/pals24_select_lambda.py 가 고정한 값)"
        cells = stage2(a.lambda_star); bl = blocks(a.lambda_star); members = {c: (bl["B"] if c[1] == 7777 else bl["C"]) for c in cells}
    else:
        cells = stage1(); members = {c: stage1() for c in cells}
    made, donor_sha, donor_step = generate(cells, members, a.out_dir, a.updates, a.eval_epoch, a.projected_hours, a.version, a.diag_every, a.ledger)
    if not a.stage2 and a.out_dir == os.path.join(ROOT, "config") and a.version == "v1":
        q = os.path.join(ROOT, "config", "queues", "pals24_s1.txt")
        with open(q, "w") as f:
            f.write(f"# PALS24 (계획 §11.5): stage 1 = seed {SCREEN_SEED} 신규 λ 3벌 (A1 L1E3 → A2 L1E4 → A3 L3E3) · s1 · {a.updates} updates · 예산 {TOTAL_HOURS} GPU-h ({LEDGER})\n"
                    "# seed 1234 의 CTRL-P0 / L000 / L1E2 는 NF16 P0 / P2 / P3 재사용 (tools/pals24_metric_gate.py 의 reuse_registry.json 승인) — 다시 학습하지 않는다.\n"
                    "# stage 2 (B1–B3 seed 7777: P0 → L000 → λ* · C1–C3 seed 2025: L000 → λ* → P0) 는 큐가 끝난 뒤 tools/campaign_gate.py 의 pals24 gate 가 연다:\n"
                    "#   A1–A3 완료 → tools/pals24_select_lambda.py 가 λ* 를 한 번 고정(raw best HQNR → fSCC(1e-4) → 더 작은 λ, work_dir/_pals24_campaign/selected_lambda.json)\n"
                    "#   → gen_pals24_configs.py --stage2 --lambda-star <id> → 6 run 을 순서대로. 세 신규 λ 가 모두 P2 보다 0.0031 초과 낮으면(§10.3/§11.5) stage 2 를 열지 않는다.\n"
                    f"# gate 식(run 마다): used + {MARGIN}·(이 run + 같은 block 의 나머지) + reserve {RESERVE_HOURS}h ≤ {TOTAL_HOURS}h — block(P0·L000·λ*) 전체가 완결 가능할 때만 (§11.6). 초과면 DEFERRED_BUDGET(exit 4), 체인은 다음으로.\n"
                    f"# eval_epoch {a.eval_epoch} (NF16 P1–P4 와 같은 격자) · 약명→세팅은 config 머리 주석 / research_log/2026-09-12_pals24-implementation.md §2\n" + "\n".join(made) + "\n")
        print("queue:", os.path.relpath(q, ROOT))
    print("\n".join(made)); print("donor step", donor_step, "sha", (donor_sha or "?")[:16])


if __name__ == "__main__":
    main()
