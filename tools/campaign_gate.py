#!/usr/bin/env python
"""압축 귀속 캠페인 조건부 게이트 — 본 큐 결과로 추가 실행을 판정한다.

계획: research_log/2026-08-31_kd-mutual-implementation-report.md (KD 캠페인).
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


CAL_TAG = "T1_c6_unc"


def _ckpt_signature(tag):
    sd = os.path.join(ROOT, "work_dir", tag, "best_hqnr", "model.safetensors")
    if not os.path.exists(sd):
        return None
    st = os.stat(sd)
    return f"{st.st_size}-{st.st_mtime_ns}"


def calibration_ok():
    """T1 의 θ-오차 calibration (명세 §8.3). 없거나 checkpoint 가 갱신됐으면 재검사한다."""
    if not complete(CAL_TAG):
        return None                                    # 아직 판정 불가 (다음 패스 대기)
    p = os.path.join(ROOT, "work_dir", CAL_TAG, "calibration.json")
    sig = _ckpt_signature(CAL_TAG)
    if os.path.exists(p):
        try:
            if json.load(open(p)).get("ckpt_signature") != sig:
                log("calibration: checkpoint 갱신 감지 — 재검사")
                os.remove(p)
        except Exception:
            os.remove(p)
    if not os.path.exists(p):
        log(f"calibration: {CAL_TAG} 검사 실행 (tools/check_calibration.py)")
        subprocess.run([PY, os.path.join(ROOT, "tools", "check_calibration.py"),
                        os.path.join(ROOT, "work_dir", CAL_TAG)], cwd=ROOT)
    if not os.path.exists(p):
        log("calibration: 결과 파일 생성 실패 — K2+ 닫힘")
        return False
    r = json.load(open(p))
    log(f"calibration: Spearman {r.get('spearman', 0):.4f} 단조 {r.get('monotonic')} "
        f"-> {'PASS' if r.get('pass') else 'FAIL'}")
    return bool(r.get("pass"))


def gate_kd_ladder():
    """K2 -> K3 -> K4 순차 게이트 (다중 패스로 체인).

    K2: T1 calibration PASS 이고 대조군(K1B·K1B_T1)이 종결됐을 때만.
    K3/K4: 직전 단계가 완료됐을 때만. calibration FAIL 이면 사다리 전체 닫힘
    — 명세 §8.3 "uncertainty 가 오차와 연결되지 않으면 uncertainty KD 를 진행하지
    않는다" 의 자동화다.
    """
    cal = calibration_ok()
    if cal is None:
        log("KD 사다리: T1 미완 — 대기")
        return
    if not cal:
        log("KD 사다리: calibration FAIL — K2~K4 닫힘 (teacher head/loss 재설계 필요)")
        return
    for ctrl in ("K1B_R4_specKD", "K1B_T1_specKD"):
        if ledgered(ctrl) and not complete(ctrl):
            log(f"KD 사다리: 대조군 {ctrl} 실패 — K2~K4 영구 닫힘 (대조 불가한 비교는 돌리지 않는다)")
            return
        if not complete(ctrl):
            log(f"KD 사다리: 대조군 {ctrl} 미완 — 대기")
            return
    if not complete("K2_R4_uknow"):
        emit("K2_R4_uknow", "calibration PASS · 대조군 종결")
        return
    if not complete("K3_R4_uknow_gtvar"):
        emit("K3_R4_uknow_gtvar", "K2 완료")
        return
    if not complete("K4_R4_uknow_gtvar_sis"):
        emit("K4_R4_uknow_gtvar_sis", "K3 완료")
        return
    log("KD 사다리: 전부 완료")


def main():
    gate_kd_ladder()


if __name__ == "__main__":
    main()
