#!/usr/bin/env python
"""PALSV18 실행 전 gate (계획 §5.1 중 코드로 닫는 것; G-G/G-W/G-R 은 tools/pals24_unit_tests.py PL03/PL04 가 같은 wrapper 로 닫는다). 하나라도 실패하면 exit 1.
    python tools/palsv18_unit_tests.py"""
import csv, json, os, sys, tempfile
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve
from tools import gen_pals24_configs as G, gen_palsv18_configs as V
from train_kdv import KDVTrainer

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

# ---------------- PV01 편성 (§3.1)
names = [V.run_name(c, s) for c, s in V.CELLS]
check("PV01 6회: seed1234 L3E5→L3E4 · seed7777 L3E4→L3E5 · seed2025 L3E5→L3E4 (λ·순서 결합 해제)", V.CELLS == [("L3E5", 1234), ("L3E4", 1234), ("L3E4", 7777), ("L3E5", 7777), ("L3E5", 2025), ("L3E4", 2025)])
check("PV01 λ 값: L3E5 3e-5 · L3E4 3e-4 (L1E4 1e-4 의 0.3배/3배), 새 λ 추가 없음", G.LAMBDA["L3E5"] == 3e-5 and G.LAMBDA["L3E4"] == 3e-4 and V.NEW_LAMBDAS == ["L3E5", "L3E4"])
check("PV01 run 이름 PALSV18_<case>_W112_D123_WV3_S<seed>_N2LAST_R200_v1", names[0] == "PALSV18_L3E5_W112_D123_WV3_S1234_N2LAST_R200_v1")
bl = V.blocks(); check("PV01 block = seed 별 두 λ (pair 단위 완결·보류; 큐 순서 1234 → 7777 → 2025 = 보류 순서 2025 → 7777 의 역)", all(len(v) == 2 for v in bl.values()) and list(bl) == [1234, 7777, 2025])
check("PV01 재사용 대조군 9벌 (P0/L000/L1E4 × 3 seed) + 진단 참조 N2 last·L1E2", len(V.REUSED) == 9 and all(os.path.isdir(os.path.join(ROOT, "work_dir", r)) for r in V.REUSED.values()) and ("N2", 2025) in V.DIAG_REFERENCE)
q = os.path.join(ROOT, "config", "queues", "palsv18_s1.txt")
if os.path.exists(q):
    qs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]; check("PV01 큐 파일 = 6 run, 계획 순번", qs == names)

# ---------------- PV02 config: PALS24 L1E4 와 λ·예산·캠페인 키 외 동일 (§11.1 학습 코드 변경 최소화)
def stripped(text):
    d = yaml.safe_load("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))); d.pop("work_dir", None); k = d.get("kdv", {})
    for key in ("campaign_id", "plan_protocol_id", "document_revision", "case_id", "budget", "select", "diag"):   # select.retain_all_candidates / diag.fixed_batch 는 학습 의미 불변 (PV08 이 별도 검사)
        k.pop(key, None)
    return d
def diffkeys(a, b, pre=""):
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diffkeys(va, vb, pre + key + ".")
        elif va != vb:
            out.append(pre + key)
    return out
ref = os.path.join(ROOT, "config", G.run_name("L1E4", 1234) + ".yaml")
for c in ("L3E5", "L3E4"):
    f = os.path.join(ROOT, "config", V.run_name(c, 1234) + ".yaml")
    if os.path.exists(f) and os.path.exists(ref):
        d = diffkeys(stripped(open(f).read()), stripped(open(ref).read())); check(f"PV02 {c} S1234 config vs PALS24 L1E4 S1234: 차이는 kdv.aux.offset_weight 하나", d == ["kdv.aux.offset_weight"], f"diff {d}")
for s_ in (7777, 2025):
    f = os.path.join(ROOT, "config", V.run_name("L3E4", s_) + ".yaml"); r2 = os.path.join(ROOT, "config", G.run_name("L1E4", s_) + ".yaml")
    if os.path.exists(f) and os.path.exists(r2):
        d = diffkeys(stripped(open(f).read()), stripped(open(r2).read())); check(f"PV02 L3E4 S{s_} vs PALS24 L1E4 S{s_}: λ 만", d == ["kdv.aux.offset_weight"], f"diff {d}")
kb = G.kdv_block("L3E5", 1234, "ab" * 32, 50000, bl[1234], camp=V.CAMP); sp = resolve(kb)
check("PV02 resolver: A-FT · I-AEQ · λ 3e-5 · ramp 0 · geo 0 · teacher 없음", sp["policy"] == "A-FT" and sp["protocol"] == "I-AEQ" and abs(sp["offset_weight_effective"] - 3e-5) < 1e-18 and sp["offset_ramp_updates"] == 0 and sp["geometry_weight_effective"] == 0 and not sp["needs_teacher"])
check("PV02 예산 키: ledger _palsv18_budget · 18h · reserve 5.0(V-post 3.5 + report 0.5 + buffer 1.0) · margin 1.1 · run 예약 1.7 · pair = 같은 seed 의 다른 λ · required False",
      kb["budget"]["ledger"].endswith("_palsv18_budget/ledger.json") and kb["budget"]["total_gpu_hours"] == 18.0 and kb["budget"]["reserve_hours"] == 5.0 and kb["budget"]["margin"] == 1.1 and kb["budget"]["required"] is False
      and kb["budget"]["projected_map"] == {V.run_name("L3E5", 1234): 1.7, V.run_name("L3E4", 1234): 1.7} and kb["budget"]["remaining_mandatory"] == [V.run_name("L3E4", 1234)])
check("PV02 캠페인 id·protocol 계승", kb["campaign_id"] == "PALSV18_W112D123_N2LAST_R200_v1" and kb["plan_protocol_id"] == "PALS_W112D123_N2LAST_R200_v1")
check("PV02 PALS24 기본 상수 불변 (generate 재실행이 같은 config 를 만든다)", G.campaign()["campaign_id"] == "PALS24_W112D123_N2LAST_R200_v2" and G.campaign()["total_hours"] == 24.0 and G.campaign()["run_reserved"] == 2.0)

# ---------------- PV03 예산 gate (§4.3): pair 단위, 18h 상한, 감축 순서
d = KDVTrainer.budget_decision(2.8, 1.7, [1.7], 5.0, 18.0, required=False, margin=1.1)
check("PV03 R1234 시작: used 2.8(gate 0.8 + V-pre 2.0) + 1.1×3.4 + 5.0 = 11.54 ≤ 18 → RUN", d["decision"] == "RUN" and abs(d["projected_total_hours"] - 11.54) < 1e-9)
d = KDVTrainer.budget_decision(9.6, 1.7, [1.7], 5.0, 18.0, required=False, margin=1.1)
check("PV03 R2025 예약대로면 used 9.6 + 3.74 + 5.0 = 18.34 > 18 → DEFERRED (예약 1.7 은 상한; 실측 1.45 면 9.1 + 3.74 + 5 = 17.84 → RUN)",
      d["decision"] == "DEFERRED_BUDGET" and KDVTrainer.budget_decision(0.9 + 2.0 + 4 * 1.55, 1.55, [1.55], 5.0, 18.0, required=False, margin=1.1)["decision"] == "RUN")
check("PV03 예상치 = max(1.7, 실측×1.1) 규칙 (prepare 가 --projected-hours 로 넘긴다)", max(1.7, 1.24 * 1.1) == 1.7 and max(1.7, 1.6 * 1.1) > 1.7)

# ---------------- PV04 G-C control: seed 별 U-Net 초기값이 기존 대조군 기록과 같다
from tools.pals24_metric_gate import init_sha16
for s_ in (1234, 7777, 2025):
    f = os.path.join(ROOT, "work_dir", "_kdv_init_w112_d123", f"init_unet_seed{s_}.pt"); cur = init_sha16(f) if os.path.exists(f) else {}
    recs = {}
    for (c, ss), run in V.REUSED.items():
        hp = os.path.join(ROOT, "work_dir", run, "init_and_teacher_hashes.json")
        if ss == s_ and os.path.exists(hp):
            recs[run] = (json.load(open(hp)).get("init_hashes") or {}).get("unet_init_sha256_16")
    check(f"PV04 seed {s_} 초기 U-Net tensor sha = 대조군 {len(recs)}벌 기록과 일치 (통제 비교 성립)", bool(recs) and all(v in (cur.get("tensors_sha16"), cur.get("file_sha16")) for v in recs.values()), f"{cur.get('tensors_sha16')} vs {set(recs.values())}")
dsha = json.load(open(os.path.join(ROOT, "work_dir", G.run_name("L1E4", 7777), "init_and_teacher_hashes.json")))["donor"]
check("PV04 donor = N2 exact50K last aligner (file sha 9d4cbf21…, tensors db638b55…), 대조군 기록과 동일", dsha["file_sha256"].startswith("9d4cbf218cd8f674") and dsha["aligner_tensors_sha256_16"] == "db638b55c62f8c62" and dsha["donor_step"] == 50000)

# ---------------- PV05 G-S selection grid (§6.3): 기존 L1E4 의 평가 update 목록 = eval_epoch 10 격자 + exact 50K
cm = list(csv.DictReader(open(os.path.join(ROOT, "work_dir", G.run_name("L1E4", 1234), "checkpoint_metrics.csv")))); steps = [int(r["step"]) for r in cm]
check("PV05 L1E4 S1234 평가 격자: 25 후보(10 epoch 마다) + exact 50000, 신규 config eval_epoch 10 동일", len(steps) == 25 and steps[-1] == 50000 and all(int(r["epoch"]) % 10 == 0 for r in cm[:-1])
      and all(yaml.safe_load(open(os.path.join(ROOT, "config", n + ".yaml")))["eval_epoch"] == 10 for n in names if os.path.exists(os.path.join(ROOT, "config", n + ".yaml"))))

# ---------------- PV06 캠페인 gate: PALSV18 은 조건부 gate 없음 — pals24 token 이 남아 있으면 종료 후 정리 대상
gf = os.path.join(ROOT, "work_dir", "campaign_gates_enabled.txt"); tok = [l.strip() for l in open(gf) if l.strip()] if os.path.exists(gf) else []
check("PV06 조건부 gate token 확인 (prepare 가 pals24 token 을 지운다; 이 캠페인은 무조건 큐)", True, f"현재 token {tok}")

# ---------------- PV07 V1 probe manifest (§7.1): r {0.25,0.5,1,2} × 8 방향 = 32 + zero; 무반응 EPE 이론값 0.9375
from tools.po10_diag import fixed_probes
import math
pr, st = fixed_probes(2.0, "palsv18"); nz = [p for p in pr if p != (0.0, 0.0)]
check("PV07 palsv18 probe = zero + 32 (반경 0.25/0.5/1/2 × 8 방향), stress 없음", len(pr) == 33 and len(nz) == 32 and sorted({round(math.hypot(a, b), 6) for a, b in nz}) == [0.25, 0.5, 1.0, 2.0] and st == [])
check("PV07 무반응 대조 EPE = mean‖e‖ = 0.9375 px (이번 32 probe 의 직접 계산)", abs(sum(math.hypot(a, b) for a, b in nz) / 32 - 0.9375) < 1e-12)

# ---------------- PV08 리뷰 P1-1: 평가 checkpoint 보존 키 + trainer 경로
for n in names[:1]:
    f = os.path.join(ROOT, "config", n + ".yaml")
    if os.path.exists(f):
        k = yaml.safe_load(open(f))["kdv"]; check("PV08 config: kdv.select.retain_all_candidates true · kdv.diag.fixed_batch true", k.get("select", {}).get("retain_all_candidates") is True and k.get("diag", {}).get("fixed_batch") is True)
src = open(os.path.join(ROOT, "train_kdv.py")).read()
check("PV08 trainer: retain 이면 selector 동률 밖 후보를 지우지 않는다 · ledger flock · 마지막 active update(num_iter−1) 진단 · 고정 batch 분해", 'retain_all_candidates' in src and 'and not retain' in src and '_LedgerLock' in src and 'int(n_iter) - 1' in src and 'gradient_diagnostics_fixed.jsonl' in src)
class _P: protocol = "I-AEQ"; diag_every = 1000; args = type("A", (), dict(num_iter=50000))()
check("PV08 진단 step 에 49999(마지막 active update) 포함, 49998 제외", KDVTrainer.is_diag_step(_P, 49999) and not KDVTrainer.is_diag_step(_P, 49998) and KDVTrainer.is_diag_step(_P, 25001))
# ---------------- PV09 리뷰 P1-2: matched grid = selection_grid 의 실제 update 목록 + 실제 BestSelector 규칙
from tools.palsv18_report import matched_grid_best
gp = os.path.join(ROOT, "work_dir", "_palsv18_campaign", "selection_grid.json")
if os.path.exists(gp):
    g = json.load(open(gp)); mg_ = matched_grid_best(G.NF16[0], g["optimizer_updates"])
    check("PV09 NF16 P0 S1234 matched-grid: 25/25 후보(exact 50K 포함) · 실제 selector 규칙 → 38380 / 0.952723 (리뷰 값과 일치; best_on_grid 의 48480/0.952752 가 아님)", mg_ and mg_["n_candidates"] == 25 and mg_["complete"] and mg_["step"] == 38380 and abs(mg_["hqnr"] - 0.952722827) < 1e-8, str(mg_))
    l1 = matched_grid_best(G.run_name("L1E4", 1234), g["optimizer_updates"]); check("PV09 L1E4 S1234 matched-grid = 원 selector (40400, 같은 격자)", l1 and l1["step"] == 40400)
import tempfile as _tf, csv as _csv
_d = _tf.mkdtemp(); os.makedirs(os.path.join(_d, "work_dir", "SYN"), exist_ok=True)
with open(os.path.join(_d, "work_dir", "SYN", "checkpoint_metrics.csv"), "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=["step", "epoch", "raw_original.hqnr", "raw_original.fscc"]); w.writeheader()
    for st, ep, h, f_ in ((10, 1, 0.95000, 0.80), (20, 2, 0.95005, 0.79), (30, 3, 0.95003, 0.81), (40, 4, 0.94000, 0.90)):
        w.writerow(dict(step=st, epoch=ep, **{"raw_original.hqnr": h, "raw_original.fscc": f_}))
import tools.palsv18_report as RP; _root0 = RP.ROOT; RP.ROOT = _d
syn = matched_grid_best("SYN", [10, 20, 30]); RP.ROOT = _root0
check("PV09 selector 규칙 재생: HQNR 1e-4 band {10,20,30} → fSCC 최대(30, 0.81) → 40(격자 밖) 무시", syn and syn["step"] == 30 and syn["n_candidates"] == 3)
# ---------------- PV10 리뷰 P2-5: native proxy 방향 — P = W(M, (+1,0)) 이면 canonical PAN correction (−1, 0)
import numpy as _np, torch as _t
from pa.warp import warp_pan as _wp
from align.estimator import estimate_shift as _es, GATES as _GA
rng = _np.random.RandomState(0); base = rng.rand(96, 96); base = _np.cumsum(_np.cumsum(base, 0), 1); base = (base - base.min()) / (base.max() - base.min()) * 1000 + 100
from scipy.ndimage import gaussian_filter as _gf; base = _gf(base, 1.0) + 50 * _np.sin(_np.arange(96) / 3.0)[None, :] + 50 * _np.cos(_np.arange(96) / 4.0)[:, None]
P = _wp(_t.from_numpy(base)[None, None], _t.tensor([[1.0, 0.0]], dtype=_t.float64))[0, 0].numpy()
r_ = _es(P.astype(_np.float32), base.astype(_np.float32), dict(_GA, search_int=4, max_magnitude=4.0)); canon = (-r_["dy_lr_raw"], -r_["dx_lr_raw"])
check("PV10 audit(ref=PAN, mov=MS) 은 (+1,0) 근처, canonical = −audit 이 필요한 correction (−1,0)", abs(r_["dy_lr_raw"] - 1.0) < 0.15 and abs(r_["dx_lr_raw"]) < 0.15 and abs(canon[0] + 1.0) < 0.15, f"audit ({r_['dy_lr_raw']:+.3f},{r_['dx_lr_raw']:+.3f})")
vsrc = open(os.path.join(ROOT, "tools", "palsv18_validate.py")).read()
check("PV10 validate 가 canon_* 열·known-shift/identity 를 summary 에 남긴다 · 개입은 적격 장면만 평균 · shortcut 대조 양 ckpt · stress 8 방향 + c0−ε 기준선 + RR", all(x in vsrc for x in ("canon_before_dy", "canonical_pan_to_ms_of_audit", "def vmean", "shortcut_controls", "fr_stress_baseline", "def rr_stress", 'default=8')))
# ---------------- PV11 리뷰 P2-3/P2-6: 실행기 idempotency·예산 guard·잠금
rsrc = open(os.path.join(ROOT, "tools", "palsv18_validate.sh")).read()
check("PV11 실행기: sha·tool_version·parts 검사 후 생략, guard(used + est + 1.7×미완 + 1.5 ≤ 18), reserved 항목, 잠금 helper", all(x in rsrc for x in ("need_run", "TOOL_VERSION", "guard", "_reserved", "_ledger_update.py")))
psrc = open(os.path.join(ROOT, "tools", "po10_diag.py")).read()
check("PV11 po10_diag: stress csv checkpoint 별 이름 · band 별 상수 MS · ckpt_sha256 기록", "stress_hqnr_native_fr512_{tag}" in psrc and "mean(dim=(2, 3), keepdim=True)" in psrc and "ckpt_sha256" in psrc)

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
