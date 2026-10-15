#!/usr/bin/env bash
# NOA/A_ON 전수 Student 평가 — **서버 담당자용 단일 진입점** (s1–s5 공통)
# protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918 · 설명은 research_log/2026-09-18_noa-eval-implementation.md (이 저장소에 있다)
# 계획 원문(research_log/PAN_AllServers_*.md)은 저장소에 두지 않는 규약이라 서버에는 없다 — 이 스크립트와 노트가 서버용 지시서다.
#
#   ./tools/noa_eval_switch.sh --dry-run   # ① gate ② 상태·평가 대상 미리보기. **아무것도 쓰지 않는다**
#   ./tools/noa_eval_switch.sh --hold      # ③ 새 학습만 정지(진행 중 run 은 원 정의로 끝난다) + 복귀 manifest 저장
#   ./tools/noa_eval_switch.sh             # ④ 사전 검사 + 자기 서버 완료 Student 전수 A_ON/NOA RR·FR 평가 (학습 중이면 거부 — --share-gpu 로 강행)
#   ./tools/noa_eval_switch.sh --upload    # ⑤ 자기 탭 NOA 열만 기록 (먼저 dry-run 을 보여준다)
#   ./tools/noa_eval_switch.sh --release   # ⑥ hold 해제 → 원래 본 실험 큐로 복귀
#
# 규칙(요약): 학습 method·Teacher·q cache·기존 checkpoint 를 바꾸지 않는다 · 같은 checkpoint 로 A_ON 과 NOA 를 쌍으로 잰다 ·
#   NOA = aligner 호출 0회·warp 호출 0회(영점 warp 와 다르다) · 기존 시트 열(원 RR/FR·Date·Train(h)·통합실험) 을 보존한다 ·
#   NOA 가 좋아도 학습·추론 정책을 자동으로 바꾸지 않는다 · **다른 서버 평가나 s1 분석을 기다리지 않는다**.
#   s1 만 전수 평가·업로드를 끝낸 뒤 tools/s1_aligner_analysis.py 로 크기·probe·네 모드 분석을 하고 복귀한다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"
DRY=0; HOLD=0; UPLOAD=0; RELEASE=0; SHARE=0; LIMIT=""
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1;; --hold) HOLD=1;; --upload) UPLOAD=1;; --release) RELEASE=1;; --share-gpu) SHARE=1;;
  --limit) LIMIT="--limit $2"; shift;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
CAMP=work_dir/_eval_phase; mkdir -p "$CAMP"
echo "[noa-switch] 서버 $SERVER · protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918 · 노트 research_log/2026-09-18_noa-eval-implementation.md"

if [ "$RELEASE" = 1 ]; then
  echo "[noa-switch] ⑥ 복귀 — hold 해제 후 원래 큐로"
  "$PY" tools/noa_eval.py --status 2>&1 | grep -v Warning | head -6
  "$PY" tools/eval_phase.py release || fail "release 실패"
  echo "   chain 이 죽어 있으면 ./tools/qrecon24_switch.sh 로 복원한다 (tools/eval_phase.py resume 도 같은 일을 한다)"
  exit 0
fi

echo "[noa-switch] ① unit gate (K01–K51)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep FAIL | head -5; fail "gate 실패 — $CAMP/gate_$SERVER.log"; }
tail -1 "$CAMP/gate_$SERVER.log"

echo "[noa-switch] ② 현재 상태 · 평가 대상"
"$PY" tools/eval_phase.py status 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then
  "$PY" tools/noa_eval.py --capture-cohort 2>&1 | grep -v Warning | head -20 || true
  echo "[noa-switch] --dry-run — 여기까지는 cohort 파일만 만든다(학습·시트·hold 변경 없음)."
  echo "   다음: ./tools/noa_eval_switch.sh --hold   (진행 중 학습은 그대로 끝나고 새 학습만 멈춘다)"
  exit 0
fi

if [ "$HOLD" = 1 ]; then
  echo "[noa-switch] ③ 새 학습 정지(hold) — 진행 중 run 은 원 정의로 끝난다"
  "$PY" tools/eval_phase.py hold || fail "hold 실패"
  "$PY" tools/noa_eval.py --capture-cohort 2>&1 | grep -v Warning | head -20
  echo "   진행 중 학습이 끝나면: ./tools/noa_eval_switch.sh   (평가 실행)"
  exit 0
fi

if [ "$UPLOAD" = 1 ]; then
  echo "[noa-switch] ⑤ 시트 — 자기 탭(WV3-$SERVER) NOA 열만. 먼저 dry-run"
  "$PY" gspread/noa_upload.py --dry-run 2>&1 | grep -v Warning | tail -25
  read -r -p "   위 내용을 그대로 기록한다. 진행? [y/N] " ans
  case "$ans" in y|Y) ;; *) echo "   중단 — 시트를 바꾸지 않았다"; exit 0;; esac
  "$PY" gspread/noa_upload.py 2>&1 | grep -v Warning | tail -8 || fail "업로드 실패"
  "$PY" tools/noa_eval.py --status 2>&1 | grep -v Warning | head -8
  echo "   다음: ($SERVER = s1 이면) 분석 → 복귀 · 그 밖 서버는 바로 ./tools/noa_eval_switch.sh --release"
  exit 0
fi

echo "[noa-switch] ④ 평가 — 같은 checkpoint 의 A_ON / NOA 쌍 (RR + FR)"
if ps -eo args | grep -q '[m]ain\.py --config'; then
  if [ "$SHARE" = 1 ]; then
    echo "   !! 학습이 돌고 있다 — --share-gpu 로 강행한다(둘 다 느려진다)"
  else
    echo "   학습이 돌고 있다: $(ps -eo args | grep '[m]ain\.py --config' | head -1 | sed 's/.*config //' | xargs basename)"
    echo "   계획대로 **현재 case 를 끝낸 뒤** 평가한다. 먼저 ./tools/noa_eval_switch.sh --hold 로 새 학습을 막고, 이 run 이 끝나면 다시 이 명령을 실행할 것."
    echo "   (GPU 를 나눠 써도 된다면 --share-gpu)"; exit 3
  fi
fi
"$PY" tools/noa_eval.py --capture-cohort 2>&1 | grep -v Warning | head -20
echo "[noa-switch] ④-1 사전 검사 (§4.4: NOA 호출 0회·state 불변·ZERO 대조)"
"$PY" tools/noa_eval.py --sanity 2>&1 | grep -v Warning | tail -8 || fail "사전 검사 실패"
echo "[noa-switch] ④-2 전수 평가"
"$PY" tools/noa_eval.py --all $LIMIT 2>&1 | grep -v Warning | tail -60 || fail "평가 실패"
"$PY" tools/noa_eval.py --status 2>&1 | grep -v Warning
echo "[noa-switch] 다음: ./tools/noa_eval_switch.sh --upload   (자기 탭 NOA 열 기록) → --release (복귀)"
[ "$SERVER" = "s1" ] && echo "   s1 추가: python tools/s1_aligner_analysis.py --assets --split rr --scenes 0-19 · --modes (전수 평가·업로드 뒤)"
exit 0
