#!/usr/bin/env python
"""PALSV18 config 생성 — L1E4(λ_off 1e-4) 근방 미세조정·정합 능력 검증, s1 18 GPU-h (research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md §3–§4).

    python tools/gen_palsv18_configs.py            # 6 config + 큐 config/queues/palsv18_s1.txt (R1 seed1234: L3E5 → L3E4 · R2 seed7777: L3E4 → L3E5 · R3 seed2025: L3E5 → L3E4)

PALS24 생성기(tools/gen_pals24_configs.py) 의 recipe·템플릿을 그대로 쓰고 캠페인 상수만 바꾼다 — λ 외 변경 없음(계획 §11.1).
약명: L3E5 = λ_off 0.00003 · L3E4 = λ_off 0.0003 (그 외 CTRLP0/L000/L1E4/L1E2 는 PALS24 와 같다). run 이름 PALSV18_<case>_W112_D123_WV3_S<seed>_N2LAST_R200_v1.
예산 work_dir/_palsv18_budget/ledger.json 18 GPU-h: gate 0.8 + V-pre 2.0 + 학습 6×1.7 + V-post 3.5 + report 0.5 + buffer 1.0. run gate: used + 1.1·(이 run + 같은 seed 의 나머지 λ) + reserve 5.0(V-post 3.5 + report 0.5 + buffer 1.0) ≤ 18.
seed 묶음(두 λ) 단위로 완결 가능할 때만 시작한다 — 부족하면 seed 2025 → 7777 순으로 통째로 보류(큐 순서가 그 순서다)."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools import gen_pals24_configs as G

CAMP = dict(prefix="PALSV18", campaign_id="PALSV18_W112D123_N2LAST_R200_v1", plan_protocol_id="PALS_W112D123_N2LAST_R200_v1", document_revision="18GPUh_v1",
            ledger="work_dir/_palsv18_budget/ledger.json", total_hours=18.0, reserve_hours=5.0, run_reserved=1.7, margin=1.1,
            plan="research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md", note="research_log/2026-09-13_palsv18-implementation.md",
            extra=dict(select=dict(retain_all_candidates=True),          # 계획 §6.3: 평가한 native checkpoint 전부 보존 (리뷰 P1-1)
                       diag=dict(fixed_batch=True)))                     # 계획 §11.2: 고정 diagnostic batch 의 gradient 분해 (리뷰 P2-7)
NEW_LAMBDAS = ["L3E5", "L3E4"]; SEEDS = [1234, 7777, 2025]
CELLS = [("L3E5", 1234), ("L3E4", 1234), ("L3E4", 7777), ("L3E5", 7777), ("L3E5", 2025), ("L3E4", 2025)]        # §3.1 순번 1–6 (seed 7777 은 λ 순서를 뒤집는다)
PLAN_RESERVE = dict(gate=0.8, pre_validation=2.0, new_training=6 * 1.7, post_validation=3.5, reporting=0.5, buffer=1.0, total=18.0, drop_order=[2025, 7777])
# §3.2 재사용 대조군 (9) + 진단 참조
REUSED = {("CTRLP0", 1234): G.NF16[0], ("L000", 1234): G.NF16[2], ("L1E4", 1234): G.run_name("L1E4", 1234),
          ("CTRLP0", 7777): G.run_name("CTRLP0", 7777), ("L000", 7777): G.run_name("L000", 7777), ("L1E4", 7777): G.run_name("L1E4", 7777),
          ("CTRLP0", 2025): G.run_name("CTRLP0", 2025), ("L000", 2025): G.run_name("L000", 2025), ("L1E4", 2025): G.run_name("L1E4", 2025)}
DIAG_REFERENCE = {("L1E2", 1234): G.NF16[3], ("N2", 2025): G.DONOR_RUN}
WORKING_REFERENCE = "L1E4"


def run_name(case, seed, version="v1"):
    return G.run_name(case, seed, version, CAMP["prefix"])


def blocks():
    return {s: [c for c in CELLS if c[1] == s] for s in SEEDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--eval-epoch", type=int, default=10, help="PALS24 와 같은 평가 격자 (selection_grid.json 과 일치해야 한다)")
    ap.add_argument("--projected-hours", type=float, default=None, help="이 run 의 예상 GPU 시간 = max(1.7, 실측×1.1) (계획 §4.3)")
    ap.add_argument("--version", default="v1"); ap.add_argument("--diag-every", type=int, default=1000); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config")); ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    bl = blocks(); members = {c: bl[c[1]] for c in CELLS}
    made, donor_sha, donor_step = G.generate(CELLS, members, a.out_dir, a.updates, a.eval_epoch, a.projected_hours, a.version, a.diag_every, a.ledger, CAMP)
    if a.out_dir == os.path.join(ROOT, "config") and a.version == "v1":
        q = os.path.join(ROOT, "config", "queues", "palsv18_s1.txt")
        with open(q, "w") as f:
            f.write(f"# PALSV18 (계획 §3.1·§4.2): R1 seed1234 L3E5 → L3E4 · R2 seed7777 L3E4 → L3E5 · R3 seed2025 L3E5 → L3E4 · s1 · {a.updates} updates · 예산 18 GPU-h ({CAMP['ledger']})\n"
                    "# 대조군 9벌(P0/L000/L1E4 × 3 seed) 은 NF16/PALS24 재사용(work_dir/_palsv18_campaign/reuse_registry.json) — 다시 학습하지 않는다. 조건부 gate 없음(전부 무조건 큐).\n"
                    "# run gate: used + 1.1·(이 run + 같은 seed 의 나머지 λ) + reserve 5.0(V-post 3.5 + report 0.5 + buffer 1.0) ≤ 18 — seed 묶음 단위 완결. 초과면 DEFERRED_BUDGET(exit 4), 체인은 다음으로.\n"
                    "# 약명→세팅: L3E5 = λ_off 3e-5 · L3E4 = λ_off 3e-4 (NF16 P3 recipe: A-FT donor aligner, I-AEQ 홀수 update 에 P_ε 를 aligner 에만, b=2, ramp 없음). 자세히는 research_log/2026-09-13_palsv18-implementation.md\n" + "\n".join(made) + "\n")
        print("queue:", os.path.relpath(q, ROOT))
    print("\n".join(made)); print("donor step", donor_step, "sha", (donor_sha or "?")[:16])


if __name__ == "__main__":
    main()
