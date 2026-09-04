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


def main():
    gate_s2_calibrate()
    gate_s2_gtvar()


if __name__ == "__main__":
    main()
