#!/usr/bin/env python
"""Swin 캠페인 조건부 게이트 — 본 큐 결과로 추가 실행을 판정한다.

계획: research_log/2026-08-29_swin-final-24h-plan.md (판정 규칙은 신계획 §3·§4.3·§5.3).
게이트는 캠페인마다 이 파일의 GATE 함수를 교체한다 (직전 캠페인 R5/A3/L2 는 종료·삭제).

측정 원칙
  - HQNR: 학습이 남긴 best_state.json (공식 12-19, 재평가 불필요)
  - SCC·ERGAS: reduced_best_hqnr.mat 을 DLPan 프로토콜(tools/eval_dlpan.py)로 평가
    — 학습 중 metrics.csv(py) 수치는 프로토콜이 달라 임계값과 비교하면 안 된다
  - params 는 config 의 expect_params_m (build 실측 기입값)

동작: stdout 에 실행할 tag(한 줄 하나), 판정 사유는 stderr(체인 로그).
러너가 다중 패스로 호출하므로, 전제가 이번 패스에 없으면 조용히 닫고 다음 패스를 기다린다.

s1/s2 공용: 전제 실행이 그 서버 work_dir 에 없으면 해당 게이트는 자연히 닫힌다
(s1 에는 SW2_d024 가 없으므로 W112/W96/N3 게이트가 열리지 않는다).
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

HQNR_BAND = 0.011      # 확정 2σ ≈ 1.18% (절대값)
SCC_TIE = 0.0005       # 이 이하 SCC 차이로는 승자 선언 금지 (신계획 §3.2)
ERGAS_SIG = 0.0023     # 두 단일 실행 차이의 3σ (2σ 0.11% × √2)

_mat_cache = {}


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
    """DLPan 프로토콜 RR 지표. {'ergas':…, 'scc':…} 또는 None."""
    if tag in _mat_cache:
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


def quality_better(a, b):
    """신계획 §3.2: HQNR band -> SCC -> ERGAS. a 가 b 보다 좋으면 True."""
    ha, hb = hqnr_of(a), hqnr_of(b)
    if ha is not None and hb is not None and abs(ha - hb) > HQNR_BAND:
        return ha > hb
    ma, mb = rr_of(a), rr_of(b)
    if ma is None or mb is None:
        return False
    if abs(ma["scc"] - mb["scc"]) > SCC_TIE:
        return ma["scc"] > mb["scc"]
    return ma["ergas"] < mb["ergas"]


def emit(tag, why):
    if complete(tag):
        log(f"{tag}: 이미 완료 — 생략")
        return
    if ledgered(tag):
        log(f"{tag}: 실패 원장 기록 — 생략")
        return
    log(f"{tag}: 열림 — {why}")
    print(tag)


def gate_sw6():
    """SW6: SW4 가 SW2 대비 HQNR band 초과 개선, 또는 동급이며 SCC·ERGAS 동방향 개선."""
    if not (complete("SW2_add") and complete("SW4_add")):
        log("SW6_add: 전제(SW2_add·SW4_add) 미완 — 닫힘")
        return
    h2, h4 = hqnr_of("SW2_add"), hqnr_of("SW4_add")
    d = (h4 or 0) - (h2 or 0)
    if d > HQNR_BAND:
        emit("SW6_add", f"SW4 HQNR +{d:.4f} > band {HQNR_BAND}")
        return
    if abs(d) <= HQNR_BAND:
        m2, m4 = rr_of("SW2_add"), rr_of("SW4_add")
        if m2 and m4:
            de = m4["ergas"] / m2["ergas"] - 1
            ds = m4["scc"] - m2["scc"]
            if ds >= 0 and de <= -ERGAS_SIG:
                emit("SW6_add", f"HQNR 동급(Δ{d:+.4f}) · SCC {ds:+.4f} · ERGAS {de*100:+.2f}% 동방향 개선")
                return
            log(f"SW6_add: 닫힘 — HQNR 동급(Δ{d:+.4f}), SCC {ds:+.4f}, "
                f"ERGAS {de*100:+.2f}% (유의선 −{ERGAS_SIG*100:.2f}%) — depth 포화 판정")
            return
    log(f"SW6_add: 닫힘 — SW4 HQNR Δ{d:+.4f} (악화)")


def w112_winner():
    """d024 vs d122 quality parent. 둘 다 terminal 이어야 판정. (parent, 사유) 또는 None."""
    a, b = "SW2_d024", "SW2_d122"
    if not (terminal(a) and terminal(b)):
        return None
    ok_a, ok_b = complete(a), complete(b)
    if not (ok_a or ok_b):
        return None
    if ok_a and not ok_b:
        return a, f"{b} 실패 — 단독 parent"
    if ok_b and not ok_a:
        return b, f"{a} 실패 — 단독 parent"
    w = a if quality_better(a, b) else b
    ma, mb = rr_of(a), rr_of(b)
    return w, (f"quality parent (HQNR {hqnr_of(a):.4f}/{hqnr_of(b):.4f}, "
               f"SCC {ma['scc']:.4f}/{mb['scc']:.4f}, ERGAS {ma['ergas']:.4f}/{mb['ergas']:.4f})")


def gate_w112():
    r = w112_winner()
    if r is None:
        log("W112: 전제(SW2_d024·SW2_d122) 미완 — 닫힘")
        return None
    parent, why = r
    cand = f"{parent}_w112"
    emit(cand, why)
    return parent


def gate_w96(parent):
    """W96: w112 가 §5.3 게이트(HQNR ≤0.011 drop · ERGAS ≤3% · 비용 ≥15% 감소) 통과 시.

    params 는 w112 가 항상 w128 parent 대비 −23% 안팎이라 비용 조건은 자동 충족 —
    latency 조회는 생략하고 params 근거를 사유에 남긴다.
    """
    if parent is None:
        return
    w112 = f"{parent}_w112"
    if not complete(w112):
        log(f"W96: 전제({w112}) 미완 — 닫힘 (다음 패스 대기)")
        return
    hp, hw = hqnr_of(parent), hqnr_of(w112)
    mp, mw = rr_of(parent), rr_of(w112)
    if None in (hp, hw) or mp is None or mw is None:
        log("W96: 지표 조회 실패 — 닫힘")
        return
    drop = hp - hw
    de = mw["ergas"] / mp["ergas"] - 1
    if drop <= HQNR_BAND and de <= 0.03:
        emit(f"{parent}_w96",
             f"w112 통과 (HQNR drop {drop:+.4f} ≤ {HQNR_BAND}, ERGAS {de*100:+.2f}% ≤ 3%, params −23%)")
    else:
        log(f"W96: 닫힘 — w112 HQNR drop {drop:+.4f}, ERGAS {de*100:+.2f}% (폭 축소 종료)")


def gate_n3_reserve(parent):
    """N3 재현(9ch 서버 독립 확정) — post-gate 예비. w112 이 terminal 이 된 뒤에만 연다.

    s1 에서는 W112 전제가 없어 열리지 않고, N3 는 어차피 완료라 무해하다.
    """
    if parent is None:
        return
    w112 = f"{parent}_w112"
    if not terminal(w112):
        log(f"N3 예비: {w112} 진행 전 — 대기")
        return
    emit("N3_9_d124_noattn", "post-gate 예비 — 9ch 성립의 서버 독립 확정 (잔여 시간)")


def main():
    gate_sw6()
    parent = gate_w112()
    gate_w96(parent)
    gate_n3_reserve(parent)


if __name__ == "__main__":
    main()
