#!/usr/bin/env python
"""조건부 게이트 — 본 큐 결과로 추가 실행을 판정한다. 캠페인마다 이 파일을 교체한다.

현재 담당 캠페인 (2026-09-01/02, 두 서버가 같은 파일을 쓴다):
  s1  research_log/2026-09-01_s1-teacher-architecture-4-6m-plan.md  (4-6M teacher 탐색)
  s2  research_log/2026-09-01_s2-uncertainty-distillation-gtvar-plan.md (uncertainty KD·GT-var)

서버 구분은 하지 않는다 — 전제 실행이 그 서버 work_dir 에 없으면 해당 게이트는
자연히 닫히므로, 같은 코드가 양쪽에서 자기 몫만 연다.

측정 원칙
  - HQNR: best_state.json (공식 12-19). 판정 1순위.
  - SCC: 보조 판정. **ERGAS 는 판정에 쓰지 않는다** (참고 지표).
동작: stdout 에 실행할 tag(한 줄 하나), 사유는 stderr(체인 로그).
러너가 다중 패스로 호출하므로 전제가 없으면 조용히 닫고 다음 패스를 기다린다.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

HQNR_BAND = 0.011      # **미검증 상한**. 공식 HQNR 은 시드 반복이 없다 —
                       # 이 값의 출처는 25K·crop·proxy QNR 체제다(2026-09-04 문서 §2).
                       # 실측 상한은 ~0.0027. 측정 전까지는 넓은 쪽을 유지한다.
SCC_TIE = 0.0005       # 이 이하 SCC 차이로는 승자 선언 금지

_mat_cache = {}
EMITTED = set()          # 이번 호출에서 연 tag — 같은 패스의 후속 게이트가 참조한다


def log(msg):
    print(f"[gate] {msg}", file=sys.stderr)


def complete(tag):
    r = os.path.join(ROOT, "work_dir", tag, "results")
    return (os.path.exists(os.path.join(r, "reduced_best_hqnr.mat"))
            and os.path.exists(os.path.join(r, "full_best_hqnr.mat")))


def ledgered(tag):
    p = os.path.join(ROOT, "work_dir", "cases_failed.txt")
    return os.path.exists(p) and any(l.startswith(tag + " ") for l in open(p))


def terminal(tag):
    return complete(tag) or ledgered(tag)


def hqnr_of(tag):
    p = os.path.join(ROOT, "work_dir", tag, "best_state.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("best_hqnr")


def rr_of(tag):
    """DLPan 프로토콜 RR 지표 {'ergas','scc'}. best_state.json 에 있으면 그걸 쓴다."""
    if tag in _mat_cache:
        return _mat_cache[tag]
    bs = os.path.join(ROOT, "work_dir", tag, "best_state.json")
    if os.path.exists(bs):
        d = json.load(open(bs))
        if d.get("scc_at_best") is not None and d.get("ergas_at_best") is not None:
            _mat_cache[tag] = {"scc": d["scc_at_best"], "ergas": d["ergas_at_best"]}
            return _mat_cache[tag]
    mat = os.path.join(ROOT, "work_dir", tag, "results", "reduced_best_hqnr.mat")
    if not os.path.exists(mat):
        return None
    out = subprocess.run([PY, os.path.join(ROOT, "tools", "eval_dlpan.py"),
                          mat, "--preset", "wv3"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    for line in out.splitlines():
        vals = re.findall(r"(\d+\.\d+)±", line)
        if len(vals) >= 6:      # PSNR SSIM SAM ERGAS SCC Q2n
            _mat_cache[tag] = {"ergas": float(vals[3]), "scc": float(vals[4])}
            return _mat_cache[tag]
    return None


def better(a, b):
    """확정 규칙: **HQNR(best checkpoint) -> SCC 에서 끝낸다.**

    둘 다 동급이면 "구분되지 않는다" 이므로 False 를 돌려준다(교체하지 않는다).
    ERGAS 로 내려가 tie-break 하지 않는다 — 참고 지표이지 판정 근거가 아니다.
    """
    ha, hb = hqnr_of(a), hqnr_of(b)
    if ha is not None and hb is not None and abs(ha - hb) > HQNR_BAND:
        return ha > hb
    ma, mb = rr_of(a), rr_of(b)
    if ma is None or mb is None:
        return False
    if abs(ma["scc"] - mb["scc"]) > SCC_TIE:
        return ma["scc"] > mb["scc"]
    log(f"{a} vs {b}: HQNR·SCC 모두 동급 — 구분되지 않음(교체 없음)")
    return False


def emit(tag, why):
    if not os.path.exists(os.path.join(ROOT, "config", f"{tag}.yaml")):
        log(f"{tag}: config 없음 — 열지 않는다")
        return
    if complete(tag):
        log(f"{tag}: 이미 완료 — 생략")
        return
    if ledgered(tag):
        log(f"{tag}: 실패 원장 기록 — 생략")
        return
    log(f"{tag}: 열림 — {why}")
    EMITTED.add(tag)
    print(tag)


# s1 게이트는 없다 — 2026-09-03 캠페인부터 큐가 전부 무조건 실행이고,
# 이전 캠페인(중간점·winner dual)의 조건부 게이트는 종료돼 삭제했다.
# 그 게이트의 "엇갈림" 정의는 SCC 와 ERGAS 의 방향 대립이었는데, 확정 규칙상
# ERGAS 는 판정에 쓰지 않으므로 정의 자체가 성립하지 않는다.

# ================================================================= s2 캠페인
S2_TEACHER = "S2_T00_W160_D122_MS2"
S2_UQ = os.path.join(ROOT, "work_dir", S2_TEACHER, "uq_head")
S2_STUDENTS = ([f"S2_PKD_L010_S{s}" for s in (2025, 1234)]
               + [f"S2_UKD_{t}_S{s}" for t in ("L003", "L010", "L030") for s in (2025, 1234)])
S2_LAMBDAS = ["L003", "L010", "L030"]


def gate_s2_calibrate():
    """teacher mean 이 끝나면 head-only calibration 을 돌리고 student 큐를 연다.

    calibration 은 mean 을 고정한 사후 보정이라 모델 선택(HQNR) 대상이 아니다 —
    체인 case 가 아니라 도구로 실행한다 (tools/calibrate_head.py).
    """
    if not complete(S2_TEACHER):
        return
    if not os.path.exists(os.path.join(S2_UQ, "model.safetensors")):
        log(f"{S2_TEACHER}: uncertainty head calibration 실행")
        r = subprocess.run([PY, os.path.join(ROOT, "tools", "calibrate_head.py"),
                            os.path.join(ROOT, "work_dir", S2_TEACHER)], cwd=ROOT)
        if r.returncode != 0:
            log("calibration FAIL — student 큐를 열지 않는다 "
                "(Spearman>0 · 5분위 단조 · 전역분산 대비 NLL 개선 중 하나가 불충족)")
            return
    info = json.load(open(os.path.join(S2_UQ, "uq_norm.json")))
    if not info.get("pass"):
        log(f"calibration FAIL 기록 (Spearman {info.get('spearman'):.4f}) — student 큐 닫힘")
        return
    log(f"calibration PASS (Spearman {info['spearman']:.4f}, "
        f"NLL 개선 {info['nll_gain']:+.5f}) — student 큐 개방")
    for t in S2_STUDENTS:
        emit(t, "teacher calibration PASS")


def gate_s2_gtvar():
    """λ_U 스윕 승자를 seed 쌍 평균으로 고르고 GT-variance 2벌만 연다 (계획 §5.3)."""
    ukd = {t: [f"S2_UKD_{t}_S2025", f"S2_UKD_{t}_S1234"] for t in S2_LAMBDAS}
    if not all(complete(r) for rs in ukd.values() for r in rs):
        return
    # 이미 어느 λ 로 시작했다면 그 λ 를 고정한다 (한 seed 만 끝난 중단 상황에서
    # 나머지 seed 를 자동 복구해야 한다 — 전체를 닫으면 복구가 막힌다).
    started = [t for t in S2_LAMBDAS
               if any(terminal(f"S2_GTVAR_{t}_S{s}") for s in (2025, 1234))]
    if started:
        t = started[0]
        log(f"GTVar: λ_U {t} 로 이미 시작됨 — 남은 seed 만 복구")
        for s_ in (2025, 1234):
            emit(f"S2_GTVAR_{t}_S{s_}", f"λ_U {t} (진행 중이던 쌍의 잔여 seed)")
        return
    rows = []
    for t, rs in ukd.items():
        h = sum(hqnr_of(r) for r in rs) / 2
        m = [rr_of(r) for r in rs]
        rows.append((t, h, sum(x["scc"] for x in m) / 2))
    for t, h, sc in rows:
        log(f"λ_U {t}: seed 평균 HQNR {h:.4f} · SCC {sc:.5f}")
    best = max(rows, key=lambda r: r[1])
    top = [r for r in rows if abs(r[1] - best[1]) <= HQNR_BAND]
    if len(top) > 1:                      # HQNR 동급이면 SCC 까지만 (ERGAS 안 씀)
        top.sort(key=lambda r: -r[2])
        best = top[0]
        why = f"HQNR 동급 {len(top)}개 중 SCC 우위"
    else:
        why = "HQNR 우위"
    log(f"λ_U* = {best[0]} ({why})")
    # 계획 §6: gradient audit 을 **두 seed 전에 한 번** 돌려 λ_V 를 고정한다
    aud = os.path.join(S2_UQ, "gtvar_audit.json")
    if not os.path.exists(aud):
        cfg = os.path.join(ROOT, "config", f"S2_GTVAR_{best[0]}_S2025.yaml")
        log("GTVar: gradient audit 실행 (계획 §6)")
        r = subprocess.run([PY, os.path.join(ROOT, "tools", "gtvar_audit.py"), cfg], cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(aud):
            log("GTVar: audit 실패 — 여는 것을 보류한다")
            return
    info = json.load(open(aud))
    log(f"GTVar: audit ratio {info['ratio']:.4f} -> λ_V {info['lambda_gtvar']}")
    for s in (2025, 1234):
        emit(f"S2_GTVAR_{best[0]}_S{s}",
             f"λ_U* = {best[0]} · {why} · λ_V {info['lambda_gtvar']} (audit)")


# ================================================================= shift-robust 캠페인 (2026-09-06)
SR_ANCHOR = "S1_T05_W168_D123_DUAL"            # best HQNR 0.9571 · fSCC 0.8785 · D_λ 0.0231 · D_s 0.0202 (추론 sweep 실측)
SR_ANCHOR_FSCC, SR_ANCHOR_DL, SR_ANCHOR_DS = 0.87849, 0.0231, 0.0202
SR_CANDS = {"SR_J1_C2RAND_BOTH_R050_W168_D123_DUAL": "SR_J1_C2RAND_BOTH_R050_S1234_W168_D123_DUAL",
            "SR_J2_C2RAND_MSONLY_R050_W168_D123_DUAL": "SR_J2_C2RAND_MSONLY_R050_S1234_W168_D123_DUAL",
            "SR_J4_CJCONS_R050_L010_W168_D123_DUAL": "SR_J4_CJCONS_R050_L010_S1234_W168_D123_DUAL",
            "AF_G1_PAN2M_GLOBALCORR_W168_D123_DUAL": "AF_G1_PAN2M_GLOBALCORR_S1234_W168_D123_DUAL"}


def _sr_stats(tag):
    import csv
    p = os.path.join(ROOT, "work_dir", tag, "metrics.csv")
    rows = list(csv.DictReader(open(p)))
    h = {int(r["epoch"]): float(r["hqnr_official"]) for r in rows}
    bs = json.load(open(os.path.join(ROOT, "work_dir", tag, "best_state.json")))
    bm = json.load(open(os.path.join(ROOT, "work_dir", tag, "best_hqnr_meta.json")))
    return dict(best=bs["best_hqnr"], fscc=bs.get("fscc_at_best") or 0.0,
                plateau=sum(v for e, v in h.items() if e >= 100) / max(1, sum(1 for e in h if e >= 100)),
                final=h[max(h)], dl=bm.get("d_lambda") or 0.0, ds=bm.get("d_s") or 0.0)


def gate_sr():
    """§13.2: 마지막 슬롯 — winner(HQNR→fSCC 1e-4→plateau→final) seed 1234 반복. 단 전 case 가 anchor 보다
    HQNR 0.002 이상 낮고 fSCC 도 낮으면 seed repeat 대신 J1 radius refinement (D_s 높음→R075, D_λ 악화→R025)."""
    done = {t: _sr_stats(t) for t in SR_CANDS if complete(t)}
    if len(done) < len(SR_CANDS):
        return
    anchor = hqnr_of(SR_ANCHOR) or 0.9571
    for t, st in done.items():
        log(f"SR {t}: best {st['best']:.4f} fSCC {st['fscc']:.4f} plateau {st['plateau']:.4f} final {st['final']:.4f}")
    all_fail = all(st["best"] < anchor - 0.002 and st["fscc"] < SR_ANCHOR_FSCC for st in done.values())
    if all_fail:
        j1 = done["SR_J1_C2RAND_BOTH_R050_W168_D123_DUAL"]
        tag = ("SR_J1_C2RAND_BOTH_R075_W168_D123_DUAL" if j1["ds"] > SR_ANCHOR_DS
               else "SR_J1_C2RAND_BOTH_R025_W168_D123_DUAL" if j1["dl"] > SR_ANCHOR_DL
               else "SR_J1_C2RAND_BOTH_R075_W168_D123_DUAL")
        emit(tag, f"전 case 가 anchor 대비 HQNR −0.002 이하·fSCC 열위 → radius refinement (J1 D_s {j1['ds']:.4f} vs anchor {SR_ANCHOR_DS})")
        return
    def key(t):
        st = done[t]; return (round(st["best"], 4), round(st["fscc"], 4), st["plateau"], st["final"])
    ranked = sorted(done, key=key, reverse=True)
    top = ranked[0]
    ties = [t for t in ranked if abs(done[t]["best"] - done[top]["best"]) <= 1e-4]
    if len(ties) > 1:
        ties.sort(key=lambda t: (done[t]["fscc"], done[t]["plateau"], done[t]["final"]), reverse=True)
        top = ties[0]
    emit(SR_CANDS[top], f"winner {top} (best {done[top]['best']:.4f} fSCC {done[top]['fscc']:.4f}) seed 1234 반복")


# ================================================================= UVS-KD 캠페인 (s2, 2026-09-06)
UVS_TEACHER = "c0_hqnr"
UVS_BASE = ["UVS_B0_lms_d122", "UVS_K0_outkd_d122", "UVS_K1_ukd_d122", "UVS_K2_uvkd_d122"]
UVS_SHIFT = ["UVS_S0_shift_d122", "UVS_M1_uvs_d122", "UVS_M2_uvs_tf_d122"]


def _uvs_teacher_gate():
    p = os.path.join(ROOT, "work_dir", UVS_TEACHER, "uvs_teacher", "gate.json")
    return json.load(open(p)) if os.path.exists(p) else None


def gate_uvs():
    """§10.1: teacher shift gate PASS 면 S0/M1/M2 를 연다. M2 가 끝나면 (a) shift MAE 만 좋고 품질 미개선 → M3,
    (b) M2 가 K2·B0 대비 우세 → R1 seed 반복, 그 뒤 C1 w96. teacher gate FAIL 이면 shift 계열은 열지 않는다."""
    g = _uvs_teacher_gate()
    if g is None or not all(complete(t) for t in UVS_BASE):
        return
    if not g.get("pass_shift"):
        log(f"UVS: teacher shift gate FAIL {g.get('checks')} — S0/M1/M2/M3 닫음 (K1/K2 반복은 사람이 결정)")
        return
    for t in UVS_SHIFT:
        if not terminal(t):
            emit(t, "teacher shift gate PASS")
    if not all(complete(t) for t in UVS_SHIFT):
        return
    b0, k2, m1, m2 = (hqnr_of(t) for t in ("UVS_B0_lms_d122", "UVS_K2_uvkd_d122", "UVS_M1_uvs_d122", "UVS_M2_uvs_tf_d122"))
    st = {t: json.load(open(os.path.join(ROOT, "work_dir", t, "best_state.json"))) for t in ("UVS_M1_uvs_d122", "UVS_M2_uvs_tf_d122")}
    log(f"UVS: B0 {b0:.4f} K2 {k2:.4f} M1 {m1:.4f} M2 {m2:.4f}")
    if m2 >= max(k2, b0) - 1e-4 and m2 >= m1 - 1e-4:
        emit("UVS_R1_m2_seed1234_d122", "M2 가 K2/B0/M1 대비 비열위 → seed 1234 반복 (§10.1)")
        if complete("UVS_R1_m2_seed1234_d122"):
            r1 = hqnr_of("UVS_R1_m2_seed1234_d122")
            if r1 is not None and r1 >= max(k2, b0) - 1e-4:
                emit("UVS_C1_m2_w96", "R1 재현 → w96 압축 확인 (§10.1)")
    else:
        emit("UVS_M3_uvs_tf_warp_d122", "M2 가 품질 미개선 → shift-effect loss 추가 (§10.1)")


# ================================================================= PALS24 λ_off sweep (s1, 2026-09-12)
def gate_pals24():
    """계획 §11.5: A1–A3(seed 1234 신규 λ) 가 끝나면 λ* 를 한 번 고정(tools/pals24_select_lambda.py) 하고 stage 2 (B1–B3 seed 7777, C1–C3 seed 2025) 를 연다.
    세 λ 가 모두 P2 보다 0.0031 초과 낮으면(TERMINATE) 열지 않는다. run 마다의 예산 gate 는 trainer(kdv.budget) 가 본다."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)                                        # python tools/campaign_gate.py 로 불리면 tools/ 만 path 에 있다
    from tools.gen_pals24_configs import run_name, NEW_LAMBDAS
    a_runs = [run_name(c, 1234) for c in NEW_LAMBDAS]
    if not all(terminal(t) for t in a_runs):
        log(f"PALS24: 탐색 run 미완 ({sum(1 for t in a_runs if terminal(t))}/{len(a_runs)}) — stage 2 닫힘"); return
    if not any(complete(t) for t in a_runs):
        log("PALS24: 탐색 run 이 전부 실패 — λ* 선택 불가"); return
    sel = os.path.join(ROOT, "work_dir", "_pals24_campaign", "selected_lambda.json")
    if not os.path.exists(sel):
        r = subprocess.run([PY, os.path.join(ROOT, "tools", "pals24_select_lambda.py")], cwd=ROOT, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            log(line)
        if r.returncode != 0 or not os.path.exists(sel):
            log("PALS24: λ* 선택 실패 — stage 2 닫힘"); return
    info = json.load(open(sel))
    if info.get("confirmation") != "PROCEED":
        log(f"PALS24: λ* {info.get('selected_case')} 고정됐으나 confirmation={info.get('confirmation')} — {info.get('confirmation_reason')}"); return
    for tag in info["stage2_runs"]:
        emit(tag, f"PALS24 stage 2 · λ* = {info['selected_case']} (λ {info['selected_lambda']}) · {info['why'].get('rule')}")


# ================================================================= NA104 20H 우선순위 (s2·s3, 2026-09-13)
def gate_na104_20h():
    """research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md §4–§6: P1(큐) 뒤 조건부 — Q36 3-seed 통과 → CF01 (s3: pilot S777·S1234 → 둘 다 Q36 대비 양성이면 S2026; s2: s3 token + 자기 Q36 통과 → 3벌),
    Q12 3-seed 통과 → X02 S777·S2026. 판정은 tools/na104_20h.decide (common-grid BestSelector 재생, 20h 예산 guard). 서버는 gspread/server.txt."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from tools.na104_20h import decide, server
    srv = server()
    if srv not in ("s2", "s3"):
        log(f"NA104-20H: s2·s3 전용 (현재 {srv}) — 닫힘"); return
    d = decide(srv)
    for r in d["reasons"]:
        log(f"NA104-20H[{srv}]: {r}")
    for tag in d["open"]:
        emit(tag, f"NA104-20H[{srv}] 조건부 (Q36 pass={d['q36']['passed']}, Q12 pass={d['q12']['passed']}, used {d['budget']['used_hours']:.2f}h/20)")


# ================================================================= PAKD50 통합 캠페인 (s1·s2·s3, 2026-09-14)
def gate_pakd50():
    """PAKD50 편성 gate (계획 §5.3·§9.4·§9.6·§11.2; 감사 F02/F03/F05). 큐에는 J0 만 있고, 그 뒤는 매 pass 이 gate 가 정한다:
    s1 은 pilot(J0 S1234 exact50K) 완료 즉시 λE 고정(tools/pakd50_calibrate.py --lambda-e → assets 사본 mirror) + 세 서버 stage 2 config 생성(빠진 것이 있으면 λE 와 별개로 다시).
    s2/s3 는 pull 로 받은 사본에서 λE 를 받는다. 편성 = gen_pakd50_configs.schedule: λE 없으면 τR-only 다음 한 벌(F0→JR→FR), 있으면 남은 전부 — 서버별 명시 순서(priority_for; 2026-09-15 재배정 s2/s4/s5) + extra_priority.txt;
    admission = 공통 training_deadline 까지 남은 시간 안에 run 별 예약(1.10×reference_train_h + 10/60; 재배정 §3·§8) 을 누적. 예약은 서버 로컬 work_dir/_pakd50/reservations.json 에도 써 trainer 예산 gate 가 같은 값을 본다."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from tools import gen_pakd50_configs as G
    srv = G.server_id(open(os.path.join(ROOT, "gspread", "server.txt")).read()); seed = G.SERVER_SEED.get(srv)      # 's3(5090)' 표시 문자열도 s3 로 (QEGX §12)
    if seed is None:
        log(f"PAKD50: 서버 {srv} 의 seed block 없음 — 닫힘"); return
    cal, cal_src = G.sync_calibration_from_assets(write=(srv != "s1"))      # s2/s3: s1 이 mirror 한 사본(assets) 의 λE 를 로컬 calibration 에 받는다 (같은 campaign·τR 일 때만)
    if cal_src == "assets" and cal.get("lambda_E"):
        log(f"PAKD50: λE {cal['lambda_E']:.4g} 를 {G.ASSET_CAL_PATH} 에서 받아 로컬 calibration 에 기록")
    if not cal.get("lambda_E") and srv == "s1":                          # s1: pilot(J0 S1234 exact50K) 이 끝나는 즉시 λE 고정 (감사 F02) — 실패해도 τR-only run 으로 slot 은 쓴다
        pilot = G.pilot_run()
        if not complete(pilot):
            log(f"PAKD50: pilot {pilot} 미완 — λE 대기, τR 만 필요한 run 으로 slot 사용")
        else:
            r = subprocess.run([PY, os.path.join(ROOT, "tools", "pakd50_calibrate.py"), "--lambda-e", "--server", "s1"], cwd=ROOT, capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines()[-4:]:
                log(line)
            cal = G.calibration()
            if r.returncode != 0 or not cal.get("lambda_E"):
                log("PAKD50: λE 고정 실패 — λE 가 필요한 case 는 닫힘")
    if cal.get("lambda_E") and srv == "s1":                              # config 생성은 λE 와 별개로 검사 (감사 F03): 빠진 stage 2 config 가 있으면 세 서버분을 다시 만든다
        missing = [G.run_name(c, sd) for sd in G.SERVER_SEED.values() for c in G.STAGE2 if not os.path.exists(os.path.join(ROOT, "config", G.run_name(c, sd) + ".yaml"))]
        if missing:
            r2 = subprocess.run([PY, os.path.join(ROOT, "tools", "gen_pakd50_configs.py"), "--all", "--stage", "2"], cwd=ROOT, capture_output=True, text=True)
            log("PAKD50: stage 2 config 생성 " + (("OK " + " ".join(missing)) if r2.returncode == 0 else "FAIL " + r2.stderr[-200:]))
    elif not cal.get("lambda_E") and srv != "s1":
        log("PAKD50: λE 미고정 (s1 의 λE 사본을 pull; 로컬 τR 과 같은 campaign 이어야 받는다) — λE 가 필요한 case 는 닫힘, τR-only run 으로 slot 사용")
    def _running(tag):                                                     # 지금 학습 중인 run 은 편성하지 않는다 (runner 는 본 큐 뒤에만 gate 를 부르지만, 손으로 불러도 중복 기동이 없게)
        r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        return any(("main.py" in ln and "--config" in ln and tag in ln) for ln in r.splitlines())
    # 편성 (계획 §9.4 우선순위) + admission (§11.2: 공통 training_deadline 까지 남은 시간 안에서만; 감사 F05)
    lp = os.path.join(ROOT, "work_dir", "_pakd50", "ledger.json"); led = json.load(open(lp)) if os.path.exists(lp) else {}
    import time as _t
    rem = G.hours_to_deadline()
    if rem is None and led.get("training_deadline"):
        rem = (_t.mktime(_t.strptime(led["training_deadline"][:19], "%Y-%m-%dT%H:%M:%S")) - _t.time()) / 3600.0
    bl = os.path.join(ROOT, G.LEDGER); bd = json.load(open(bl)) if os.path.exists(bl) else {}
    done = [float(e.get("hours_total") or e.get("hours")) for e in (bd.get("entries") or {}).values() if e.get("kind") == "run" and str(e.get("status", "")).startswith("FINISHED") and (e.get("hours_total") or e.get("hours"))]
    est = (sum(done) / len(done)) if done else float(led.get("measured_run_hours") or 4.0)          # 완료 run 실측 평균 → 없으면 smoke 예상 (reference 가 전혀 없을 때의 마지막 fallback)
    measured = G.measured_hours_all(srv)                                   # 같은 서버·같은 case 실측 {case@arch: h} — PAKD50 + QEDGE9 ledger (재배정 §3 '첫 실측이 나오면 곧바로 교체'; 감사 F07)
    extra = G.extra_priority()                                             # s4 등: 진단 뒤 사람이 고른 scalar/결합/확인 run (case id 또는 전체 run 이름)
    if extra:
        log(f"PAKD50: 추가 편성 목록({G.EXTRA_PRIORITY_FILE}): {' '.join(extra)}")
    tag_of = lambda it: G.to_tag(it, seed)
    def _reservation(it):                                                  # 재배정 §3: reservation_h = 1.10 × reference_train_h + 10/60 (여유 포함 — schedule 은 margin 을 다시 곱하지 않는다)
        r = G.reservation_for(srv, it, measured, seed)
        return r["reservation_h"] if r else G.reservation_hours(est)
    exempt = lambda it: G.branch_for(srv, it) in ("QEDGE9", "QEGX", "EDGEBAL")      # QEDGE9 §0.7·§9.3 / QEGX §9.3·§12 / EDGEBAL §7·§10.2: 절대 마감 admission 제외 (soft target · 상한 없음; NaN/오류/중복 보호는 그대로)
    blocked = lambda it: not G.cue_ready(it)                                # QEDGE9 §11.3 / QEGX: θq/c_E(c_E3) 자산이 없는 gate·route run 은 이번 pass 에 편성하지 않는다 (placeholder 금지)
    todo, dropped = G.schedule(bool(cal.get("lambda_E")), lambda it: terminal(tag_of(it)) or _running(tag_of(it)), rem, _reservation, priority=G.priority_for(srv), extra=extra, exempt=exempt, blocked=blocked)
    for it in list(G.priority_for(srv)) + list(extra):                      # 감사 F06: 완료 marker 만 있고 50K state·후보 격자·Teacher/데이터/init 동치가 확인되지 않는 run 은 그대로 재사용하지 않도록 기록 (판단은 사람)
        tg = tag_of(it)
        br_ = G.branch_for(srv, it)
        if complete(tg) and br_ in ("QEDGE9", "QEGX", "EDGEBAL"):
            v = G.verified_complete(tg)
            if not v["ok"]:
                log(f"{br_}: {tg} 는 완료 marker 는 있으나 검증 불통과 {[k for k, x in v['checks'].items() if x is False]} — COMPLETE_UNVERIFIED (work_dir/_{br_.lower()}/control_verification.json; 재실행 여부는 사람이)")
    waiting = [it for it in list(G.priority_for(srv)) + list(extra) if not (terminal(tag_of(it)) or _running(tag_of(it))) and blocked(it)]
    if waiting:
        log(f"cue 자산 대기 — θq({G.QEDGE9_CUE_ASSET}) / c_E({G.QEDGE9_CE_FILE}) / c_E3({G.QEGX_CE_FILE}) 미산출: {' '.join(waiting)} (tools/qedge9_cue.py build|pilot[ --branch qegx] 뒤 다음 pass)")
    if any(G.branch_for(srv, it) == "QEDGE9" for it in todo):
        log(f"QEDGE9: 시간 정책 soft target {G.QEDGE9_SOFT_HOURS}h — 절대 마감·50h 상속 없음 (해당 run 은 admission 제외, ledger {G.QEDGE9_LEDGER})")
    if any(G.branch_for(srv, it) == "QEGX" for it in todo):
        log(f"QEGX: 시간 정책 no_hard_limit — 절대 마감·50h·9h 상속 없음 (해당 run 은 admission 제외, ledger {G.QEGX_LEDGER}; 실패/NaN 은 자동 반복하지 않는다)")
    if any(G.branch_for(srv, it) == "EDGEBAL" for it in todo):
        log(f"EDGEBAL: 시간 정책 no_hard_limit — 절대 마감·50h·9h 상속 없음 (해당 run 은 admission 제외, ledger {G.EDGEBAL_LEDGER}; 실패/NaN 은 자동 반복하지 않는다)")
    try:                                                                   # 서버 로컬 예약 파일 (trainer budget.projection_file) — 편성·밀린 run 전부 (완료 run 은 trainer 가 0 으로 센다)
        G.write_reservation_file(srv, list(todo) + list(dropped), measured, seed)
    except Exception as e:                                                 # noqa — 예약 파일 실패가 편성을 막지 않는다 (trainer 는 projected_map 보수값으로)
        log(f"PAKD50: 예약 파일 기록 실패 — {e!r}")
    if dropped:
        log(f"PAKD50: admission — 남은 {rem:.1f}h 에 {len(todo)} run 만 들어간다 (예약 = 1.10×ref + 10min); 밀림: {' '.join(dropped)}")
    lam = ("%.4g" % cal["lambda_E"]) if cal.get("lambda_E") else "미고정"; rem_s = "∞" if rem is None else f"{rem:.1f}"; cum = 0.0
    for it in todo:
        r = G.reservation_for(srv, it, measured, seed); cum += _reservation(it)
        src = f"ref {r['reference_train_h']:.2f}h {r['reference_kind']}" if r else f"ref {est:.2f}h ledger_mean"
        emit(tag_of(it), f"{G.branch_for(srv, it) or 'PAKD50'} {G.case_of(it)} ({srv} 명시 순서 편성; λE {lam}, τR {cal.get('tau_R'):.4g}; 남은 {rem_s}h{' (' + G.branch_for(srv, it) + ': 마감 제외)' if exempt(it) else ''}, 예약 {_reservation(it):.2f}h [{src}], 누적 {cum:.2f}h)")


GATES = {"uvs": ("gate_uvs", "UVS-KD (2026-09-01 s2)"), "sr": ("gate_sr", "shift-robust (SR/AF)"),
         "s2cal": ("gate_s2_calibrate", "s2 uncertainty calibration"), "s2gtvar": ("gate_s2_gtvar", "s2 GT-variance KD"),
         "pals24": ("gate_pals24", "PALS24 λ_off sweep stage 2 (s1, 2026-09-12)"), "na104_20h": ("gate_na104_20h", "NA104 20H 우선순위 조건부 CF01/X02 (s2·s3, 2026-09-13)"),
         "pakd50": ("gate_pakd50", "PAKD50 통합 캠페인 편성 (λE 고정 뒤 서버별 명시 순서; s1–s5, 2026-09-14 · 재배정 2026-09-15 · QEDGE9 s5/s1 · QEGX s3/s4 · EDGEBAL s2/s5)")}


def enabled_gates():
    """**캠페인 격리** (2026-09-11 사용자 검토 4): 러너는 큐가 끝나면 캠페인 구분 없이 이 파일을 부른다.
    아무 것도 켜지 않으면 과거 캠페인(UVS·shift-robust·s2 KD) 의 조건부 실행이 지금 캠페인 뒤에 열릴 수 있다.
    기본값은 **전부 닫힘**이고, 켜려면 명시해야 한다:
        PANCRAFTER_CAMPAIGN_GATES=uvs,sr  (환경변수)  또는  work_dir/campaign_gates_enabled.txt (한 줄에 하나)
    'all' 이면 전부 연다."""
    raw = os.environ.get("PANCRAFTER_CAMPAIGN_GATES")            # 환경변수가 **있으면**(빈 문자열 포함) 그것이 전부 — 테스트가 '' 로 격리하면 token 파일을 읽지 않는다 (PAKD50 감사 F04)
    fp = os.path.join(ROOT, "work_dir", "campaign_gates_enabled.txt")
    if raw is None:
        raw = ",".join(l.strip() for l in open(fp) if l.strip() and not l.startswith("#")) if os.path.exists(fp) else ""
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if "all" in names:
        return list(GATES)
    return [n for n in names if n in GATES]


def main():
    on = enabled_gates()
    if not on:
        log("캠페인 게이트 비활성 — 이 큐의 캠페인에 속하지 않는 과거 조건부 실행(UVS·shift-robust·s2 KD)을 열지 않는다. "
            "필요하면 PANCRAFTER_CAMPAIGN_GATES=uvs,sr,pals24 또는 work_dir/campaign_gates_enabled.txt 로 명시한다.")
        return
    log(f"캠페인 게이트 활성: {', '.join(GATES[n][1] for n in on)}")
    for n in on:
        globals()[GATES[n][0]]()


if __name__ == "__main__":
    main()
