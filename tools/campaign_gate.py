#!/usr/bin/env python
"""조건부 게이트 — 본 큐 결과로 추가 실행을 판정한다. 캠페인마다 이 파일을 교체한다.

현재 담당 캠페인 (2026-09-01/02, 두 서버가 같은 파일을 쓴다):
  s1  research_log/2026-09-01_s1-teacher-architecture-4-6m-plan.md  (4-6M teacher 탐색)
  s2  research_log/2026-09-01_s2-uncertainty-distillation-gtvar-plan.md (uncertainty KD·GT-var)

서버 구분은 하지 않는다 — 전제 실행이 그 서버 work_dir 에 없으면 해당 게이트는
자연히 닫히므로, 같은 코드가 양쪽에서 자기 몫만 연다.

측정 원칙
  - HQNR: best_state.json (공식 12-19). 판정 1순위.
  - SCC·ERGAS: reduced_best_hqnr.mat 을 DLPan 프로토콜로 평가 (보조·참고).
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

HQNR_BAND = 0.011      # 확정 2σ. 이 안이면 HQNR 로는 "구분되지 않는다"
SCC_TIE = 0.0005       # 이 이하 SCC 차이로는 승자 선언 금지
ERGAS_SIG = 0.0023     # 두 단일 실행 차이의 3σ (참고 지표)

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
    """HQNR band -> SCC -> ERGAS. a 가 b 보다 좋으면 True."""
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


# ================================================================= s1 캠페인
S1_CORE = ["S1_T00_W160_D122_MS2", "S1_T01_W144_D124_MS2",
           "S1_T02_W160_D124_MS2", "S1_T03_W176_D122_MS2"]
S1_ANCHOR = "S1_A0_W128_D124_MS2"
S1_DUAL = {"S1_T00_W160_D122_MS2": "S1_T05_W160_D122_DUAL",
           "S1_T01_W144_D124_MS2": "S1_T05_W144_D124_DUAL",
           "S1_T02_W160_D124_MS2": "S1_T05_W160_D124_DUAL",
           "S1_T03_W176_D122_MS2": "S1_T05_W176_D122_DUAL",
           "S1_T04A_W152_D123_MS2": "S1_T05_W152_D123_DUAL",
           "S1_T04B_W168_D123_MS2": "S1_T05_W168_D123_DUAL"}
S1_MID = [("S1_T04A_W152_D123_MS2", "S1_T00_W160_D122_MS2", "S1_T01_W144_D124_MS2"),
          ("S1_T04B_W168_D123_MS2", "S1_T02_W160_D124_MS2", "S1_T03_W176_D122_MS2")]


def _ambiguity(x, y):
    """계획 §4.2 "서로 다른 지표 방향" 의 조작적 정의.

    HQNR 이 동급(|Δ| ≤ band)이면서 SCC 와 ERGAS 가 **서로 반대 모델을 가리킬 때**
    그 구간의 depth/width 결론이 불명확한 것으로 본다. 반환: (엇갈림?, |ΔHQNR|, 사유)
    """
    hx, hy = hqnr_of(x), hqnr_of(y)
    mx, my = rr_of(x), rr_of(y)
    if None in (hx, hy) or mx is None or my is None:
        return None
    dh = abs(hx - hy)
    if dh > HQNR_BAND:
        return (False, dh, f"HQNR 로 결정됨(Δ{dh:.4f} > {HQNR_BAND})")
    d_scc = mx["scc"] - my["scc"]            # + 면 x 우세
    d_erg = my["ergas"] - mx["ergas"]        # + 면 x 우세
    if abs(d_scc) <= SCC_TIE:
        return (False, dh, f"SCC 동률(Δ{d_scc:+.5f}) — 엇갈림 아님")
    split = (d_scc > 0) != (d_erg > 0)
    return (split, dh,
            f"HQNR 동급(Δ{dh:.4f}) · SCC {d_scc:+.5f} · ERGAS {d_erg:+.4f} → "
            + ("엇갈림" if split else "같은 방향"))


def gate_s1_midpoint():
    """조건부 중간점 — 두 구간 중 **하나만** 연다 (계획 §4.2)."""
    if not complete(S1_CORE[0]):
        return
    cands = []
    for mid, x, y in S1_MID:
        if not (complete(x) and complete(y)):
            log(f"{mid}: 전제({x}·{y}) 미완 — 대기")
            continue
        r = _ambiguity(x, y)
        if r is None:
            continue
        split, dh, why = r
        log(f"{mid}: {why}")
        if split:
            cands.append((dh, mid, why))
    if not cands:
        return
    for m in ("S1_T04A_W152_D123_MS2", "S1_T04B_W168_D123_MS2"):
        if complete(m) or ledgered(m):
            log("중간점: 이미 하나 실행됨 — 둘 다 돌리지 않는다")
            return
    dh, mid, why = min(cands)                # 더 모호한(ΔHQNR 작은) 구간 하나만
    emit(mid, f"{why} — 중간점 1벌만 실행")


def gate_s1_dual():
    """architecture winner 의 dual MARs 대조 (계획 §6). 탐색이 전부 끝난 뒤 한 벌만."""
    done = [t for t in S1_CORE if complete(t)]
    if len(done) < len(S1_CORE):
        log(f"T05 dual: 탐색 미완({len(done)}/{len(S1_CORE)}) — 대기")
        return
    # 중간점이 열렸다면 그것도 끝나야 한다.
    # 주의: 같은 게이트 호출에서 방금 emit 된 중간점은 work_dir 이 아직 없다.
    # EMITTED 를 함께 보지 않으면 T04 결과를 보지 않은 채 T05 winner 가 정해진다.
    for m in ("S1_T04A_W152_D123_MS2", "S1_T04B_W168_D123_MS2"):
        if m in EMITTED:
            log(f"T05 dual: 중간점 {m} 이 이번 패스에 열렸다 — 다음 패스로 대기")
            return
        if os.path.isdir(os.path.join(ROOT, "work_dir", m)) and not terminal(m):
            log(f"T05 dual: 중간점 {m} 진행 중 — 대기")
            return
        if complete(m):
            done.append(m)
    win = done[0]
    for t in done[1:]:
        if better(t, win):
            win = t
    if any(complete(v) for v in S1_DUAL.values()):
        log("T05 dual: 이미 완료 — 생략")
        return
    dual = S1_DUAL.get(win)
    if dual is None:
        log(f"T05 dual: winner={win} 의 dual config 가 없다 — 수동 생성 필요")
        return
    emit(dual, f"architecture winner = {win} (HQNR {hqnr_of(win):.4f}) 의 PAN reconstruction 대조")


def gate_s1_naf_note():
    """NAF-U fallback 조건 기록 (계획 §4.3). 구현본이 있으면 연다."""
    if not (complete(S1_CORE[0]) and complete(S1_CORE[1]) and complete(S1_ANCHOR)):
        return
    a = hqnr_of(S1_ANCHOR)
    worse = [t for t in S1_CORE[:2] if hqnr_of(t) is not None and hqnr_of(t) <= a]
    if len(worse) < 2:
        return
    log(f"NAF fallback 조건 성립 — T00·T01 이 anchor({S1_ANCHOR} HQNR {a:.4f})를 "
        "모두 개선하지 못했다")
    emit("S1_TF_NAFU_MS2", "capacity scaling 실패 — backbone family 대조")


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
        rows.append((t, h, sum(x["scc"] for x in m) / 2, sum(x["ergas"] for x in m) / 2))
    for t, h, sc, er in rows:
        log(f"λ_U {t}: seed 평균 HQNR {h:.4f} · SCC {sc:.5f} · ERGAS {er:.4f}")
    best = max(rows, key=lambda r: r[1])
    top = [r for r in rows if abs(r[1] - best[1]) <= HQNR_BAND]
    if len(top) > 1:                      # HQNR 동급이면 SCC, 그다음 ERGAS
        top.sort(key=lambda r: (-r[2], r[3]))
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
    gate_s1_midpoint()
    gate_s1_naf_note()
    gate_s1_dual()
    gate_s2_calibrate()
    gate_s2_gtvar()


if __name__ == "__main__":
    main()
