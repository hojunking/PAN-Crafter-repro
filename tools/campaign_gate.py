#!/usr/bin/env python
"""압축 귀속 캠페인 조건부 게이트 — 본 큐 결과로 추가 실행을 판정한다.

계획: research_log/2026-08-30_compression-attribution-20h-plan.md §4.
게이트는 캠페인마다 이 파일의 GATE 함수를 교체한다 (직전 캠페인 R5/A3/L2 는 종료·삭제).

측정 원칙
  - HQNR: 학습이 남긴 best_state.json (공식 12-19, 재평가 불필요)
  - SCC·ERGAS: reduced_best_hqnr.mat 을 DLPan 프로토콜(tools/eval_dlpan.py)로 평가
    — 학습 중 metrics.csv(py) 수치는 프로토콜이 달라 임계값과 비교하면 안 된다
  - params 는 config 의 expect_params_m (build 실측 기입값)

동작: stdout 에 실행할 tag(한 줄 하나), 판정 사유는 stderr(체인 로그).
러너가 다중 패스로 호출하므로, 전제가 이번 패스에 없으면 조용히 닫고 다음 패스를 기다린다.

s1/s2 공용: 전제 실행이 그 서버 work_dir 에 없으면 해당 게이트는 자연히 닫힌다
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


def gate_w80_9ch():
    """SW2_d122_w80_9ch: 9ch hybrid 의 폭 바닥 탐침.

    전제: SW2_d122_9ch(parent)·SW2_d122_w96_9ch 둘 다 그 서버에서 완료.
    조건: w96_9ch 가 parent 대비 HQNR drop <= 0.011 그리고 ERGAS <= +3% (운영 게이트).
    params -23% 는 자동 충족이라 비용 조건은 생략하고 사유에 남긴다.
    """
    parent, w96 = "SW2_d122_9ch", "SW2_d122_w96_9ch"
    if not (complete(parent) and complete(w96)):
        log(f"w80_9ch: 전제({parent}·{w96}) 미완 — 닫힘")
        return
    hp, hw = hqnr_of(parent), hqnr_of(w96)
    mp, mw = rr_of(parent), rr_of(w96)
    if None in (hp, hw) or mp is None or mw is None:
        log("w80_9ch: 지표 조회 실패 — 닫힘")
        return
    drop = hp - hw
    de = mw["ergas"] / mp["ergas"] - 1
    if drop <= HQNR_BAND and de <= 0.03:
        emit("SW2_d122_w80_9ch",
             f"w96_9ch 통과 (HQNR drop {drop:+.4f} <= {HQNR_BAND}, ERGAS {de*100:+.2f}% <= 3%, params -30%)")
    else:
        log(f"w80_9ch: 닫힘 — HQNR drop {drop:+.4f}, ERGAS {de*100:+.2f}% (폭 축소 종료)")


def main():
    gate_w80_9ch()


if __name__ == "__main__":
    main()
