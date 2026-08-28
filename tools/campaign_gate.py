#!/usr/bin/env python
"""24h 탐색 조건부 case 게이트 — 본 큐 결과의 ERGAS 로 추가 실행을 판정한다.

계획서 §8.2 (research_log/2026-08-28_architecture-search-24h-plan.md, §10 추기 우선):

  R5_w80_d124_noattn  <- R4_w96_d124_noattn ERGAS <= 2.1034
                         (c6_c4d124 의 2.0826 x 1.01 — "같은 서버 1% 미만 = 동급")
  A3_asym_014_00      <- A2_asym_014_10 ERGAS <= 2.12   (주력 게이트)
  L2_11_lr_fuse_w96   <- L1 두 벌 중 최저 ERGAS <= 2.25 (초경량 게이트)

측정은 시트와 같은 DLPan 프로토콜(tools/eval_dlpan.py)을 재사용한다 — 학습 중
metrics.csv(py) 수치는 프로토콜이 달라 임계값과 비교하면 안 된다.

stdout: 실행할 tag 목록(한 줄 하나). 판정 사유는 stderr — 체인 로그에 남는다.
전제 실행이 미완료면 그 게이트는 조용히 닫힌다(추가 실행 없음).
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def ergas_of(tag):
    """완료된 실행의 best_hqnr mat 을 DLPan 프로토콜로 평가해 ERGAS 를 돌려준다."""
    mat = os.path.join(ROOT, "work_dir", tag, "results", "reduced_best_hqnr.mat")
    if not os.path.exists(mat):
        return None
    out = subprocess.run([PY, os.path.join(ROOT, "tools", "eval_dlpan.py"),
                          mat, "--preset", "wv3"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    # 데이터 행: "<이름>  PSNR±s SSIM±s SAM±s ERGAS±s SCC±s Q2n±s" — 4번째 값
    for line in out.splitlines():
        vals = re.findall(r"(\d+\.\d+)±", line)
        if len(vals) >= 6:
            return float(vals[3])
    return None


GATES = [
    # (실행할 case, 전제 목록(하나라도 통과), 임계 ERGAS, 설명)
    ("R5_w80_d124_noattn", ["R4_w96_d124_noattn"], 2.1034,
     "w96 이 c6 와 동급(1% 이내)일 때만 w80 극단 지점을 본다"),
    ("A3_asym_014_00", ["A2_asym_014_10"], 2.12,
     "A2 가 주력 게이트(<=2.12)를 넘을 때만 decoder ResBlock 전무 극단"),
    ("L2_11_lr_fuse_w96", ["L1_11_lr_fuse_w64", "L1_9_lr_fuse_w64"], 2.25,
     "L1 이 초경량 게이트(<=2.25)를 넘을 때만 폭 상한 확인"),
]


def main():
    for cand, prereqs, thr, why in GATES:
        done = os.path.join(ROOT, "work_dir", cand, "results", "reduced_best_hqnr.mat")
        if os.path.exists(done):
            print(f"[gate] {cand}: 이미 완료 — 생략", file=sys.stderr)
            continue
        vals = [(p, ergas_of(p)) for p in prereqs]
        known = [(p, v) for p, v in vals if v is not None]
        if not known:
            print(f"[gate] {cand}: 전제 미완료({','.join(prereqs)}) — 닫힘", file=sys.stderr)
            continue
        best_tag, best = min(known, key=lambda x: x[1])
        if best <= thr:
            print(f"[gate] {cand}: 열림 — {best_tag} ERGAS {best:.4f} <= {thr} ({why})",
                  file=sys.stderr)
            print(cand)
        else:
            print(f"[gate] {cand}: 닫힘 — {best_tag} ERGAS {best:.4f} > {thr}", file=sys.stderr)


if __name__ == "__main__":
    main()
